from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, Iterable, TypeVar

from .types import CacheLookupResult
from .validator import normalize_template_whitespace, validate_template_against_log


K = TypeVar("K")
V = TypeVar("V")

WORD_RE = re.compile(r"[A-Za-z0-9]+")
PATTERN_CACHE_VERSION_CHOICES = ("v1", "v2")
MIN_PATTERN_CONSTANT_CHARS_V2 = 12
MIN_PATTERN_CONSTANT_TOKENS_V2 = 2
MIN_PATTERN_COVERAGE_V2 = 0.20
MIN_PATTERN_COVERAGE_UNANCHORED_V2 = 0.35
MAX_SIGNATURE_CANDIDATES = 32
UUID_RE = re.compile(r"\b[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}\b")
IP_PORT_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b")
MAC_RE = re.compile(r"\b[0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5}\b")
HEX_PREFIX_RE = re.compile(r"\b0x[0-9A-Fa-f]+\b")
NUMBER_RE = re.compile(r"\b[-+]?(?:\d+\.\d+|\d+)\b")
LONG_TOKEN_WITH_DIGIT_RE = re.compile(r"\b(?=\S*\d)(?=\S*[A-Za-z])[A-Za-z0-9_./:=+-]{5,}\b")


@dataclass(frozen=True, slots=True)
class TemplatePattern:
    template: str
    constant_parts: tuple[str, ...]
    constant_char_count: int
    constant_token_count: int
    wildcard_count: int
    starts_with_wildcard: bool
    ends_with_wildcard: bool

    @property
    def has_wildcard(self) -> bool:
        return self.wildcard_count > 0

    @property
    def anchored_start(self) -> bool:
        return not self.starts_with_wildcard

    @property
    def anchored_end(self) -> bool:
        return not self.ends_with_wildcard


def build_template_pattern(template: str) -> TemplatePattern:
    normalized = normalize_template_whitespace(template)
    raw_parts = normalized.split("<*>")
    constant_parts = tuple(part.strip() for part in raw_parts if part.strip())
    return TemplatePattern(
        template=normalized,
        constant_parts=constant_parts,
        constant_char_count=sum(len(part) for part in constant_parts),
        constant_token_count=sum(len(WORD_RE.findall(part)) for part in constant_parts),
        wildcard_count=normalized.count("<*>"),
        starts_with_wildcard=normalized.startswith("<*>"),
        ends_with_wildcard=normalized.endswith("<*>"),
    )


def normalize_log_signature(log_text: str) -> str:
    normalized = normalize_template_whitespace(log_text)
    for pattern in (
        UUID_RE,
        IP_PORT_RE,
        MAC_RE,
        HEX_PREFIX_RE,
        NUMBER_RE,
        LONG_TOKEN_WITH_DIGIT_RE,
    ):
        normalized = pattern.sub("<*>", normalized)
    return normalize_template_whitespace(normalized)


class LRUCache(Generic[K, V]):
    def __init__(self, maxsize: int) -> None:
        self.maxsize = maxsize
        self._cache: OrderedDict[K, V] = OrderedDict()

    def __contains__(self, key: K) -> bool:
        return key in self._cache

    def get(self, key: K) -> V | None:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: K, value: V) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)

    def values_mru_first(self) -> Iterable[V]:
        return reversed(self._cache.values())


class MultiLevelTemplateCache:
    def __init__(
        self,
        exact_cache_size: int,
        pattern_cache_size: int,
        signature_cache_size: int,
        pattern_cache_version: str = "v2",
    ) -> None:
        if pattern_cache_version not in PATTERN_CACHE_VERSION_CHOICES:
            raise ValueError(
                f"Unsupported pattern cache version: {pattern_cache_version}. "
                f"Expected one of {PATTERN_CACHE_VERSION_CHOICES}."
            )
        self.exact_cache = LRUCache[str, str](exact_cache_size)
        self.signature_cache = LRUCache[str, list[TemplatePattern]](signature_cache_size)
        self.pattern_cache = LRUCache[str, TemplatePattern](pattern_cache_size)
        self.pattern_cache_version = pattern_cache_version

    def _basic_pattern_score(self, pattern: TemplatePattern) -> tuple[int, int, int, int, int, int]:
        return (
            int(pattern.anchored_start),
            int(pattern.anchored_end),
            pattern.constant_char_count,
            pattern.constant_token_count,
            -pattern.wildcard_count,
            int(pattern.has_wildcard),
        )

    def _select_best_pattern(self, log_text: str, patterns: Iterable[TemplatePattern]) -> TemplatePattern | None:
        best_pattern: TemplatePattern | None = None
        best_score: tuple[int, ...] | None = None
        for pattern in patterns:
            is_valid, _errors, _replacements = validate_template_against_log(log_text, pattern.template)
            if not is_valid:
                continue

            score = self._match_pattern_v2(log_text, pattern)
            if score is not None:
                candidate_score: tuple[int, ...] = (2, *score)
            else:
                candidate_score = (1, *self._basic_pattern_score(pattern))

            if best_score is None or candidate_score > best_score:
                best_pattern = pattern
                best_score = candidate_score
        return best_pattern

    def _lookup_signature(self, log_text: str) -> CacheLookupResult | None:
        signature = normalize_log_signature(log_text)
        if signature == normalize_template_whitespace(log_text):
            return None

        candidates = self.signature_cache.get(signature)
        if not candidates:
            return None

        best_pattern = self._select_best_pattern(log_text, candidates)
        if best_pattern is None:
            return None

        self.exact_cache.put(log_text, best_pattern.template)
        self.signature_cache.put(signature, self._touch_signature_candidates(candidates, best_pattern))
        if self.pattern_cache_version == "v1" or self._is_v2_pattern_admissible(best_pattern):
            self.pattern_cache.put(best_pattern.template, best_pattern)
        return CacheLookupResult(template=best_pattern.template, source="signature_cache")

    def _touch_signature_candidates(
        self,
        patterns: list[TemplatePattern],
        touched: TemplatePattern,
    ) -> list[TemplatePattern]:
        reordered = [touched]
        reordered.extend(pattern for pattern in patterns if pattern.template != touched.template)
        return reordered[:MAX_SIGNATURE_CANDIDATES]

    def _update_signature_cache(self, log_text: str, pattern: TemplatePattern) -> None:
        normalized_log = normalize_template_whitespace(log_text)
        signature = normalize_log_signature(log_text)
        if signature == normalized_log:
            return

        is_valid, _errors, _replacements = validate_template_against_log(log_text, pattern.template)
        if not is_valid:
            return

        existing = self.signature_cache.get(signature) or []
        updated = [pattern]
        updated.extend(item for item in existing if item.template != pattern.template)
        self.signature_cache.put(signature, updated[:MAX_SIGNATURE_CANDIDATES])

    def _lookup_pattern_v1(self, log_text: str) -> CacheLookupResult | None:
        for pattern in self.pattern_cache.values_mru_first():
            is_valid, _errors, _replacements = validate_template_against_log(log_text, pattern.template)
            if is_valid:
                self.exact_cache.put(log_text, pattern.template)
                self.pattern_cache.put(pattern.template, pattern)
                return CacheLookupResult(template=pattern.template, source="pattern_cache")
        return None

    def _is_v2_pattern_admissible(self, pattern: TemplatePattern) -> bool:
        return (
            pattern.has_wildcard
            and pattern.constant_char_count >= MIN_PATTERN_CONSTANT_CHARS_V2
            and pattern.constant_token_count >= MIN_PATTERN_CONSTANT_TOKENS_V2
        )

    def _match_pattern_v2(self, log_text: str, pattern: TemplatePattern) -> tuple[int, int, int, int, int, int] | None:
        if not self._is_v2_pattern_admissible(pattern):
            return None

        first_idx: int | None = None
        last_end = 0
        start_idx = 0
        for part in pattern.constant_parts:
            idx = log_text.find(part, start_idx)
            if idx == -1:
                return None
            if first_idx is None:
                first_idx = idx
            last_end = idx + len(part)
            start_idx = last_end

        if first_idx is None:
            return None
        if pattern.anchored_start and first_idx != 0:
            return None
        if pattern.anchored_end and last_end != len(log_text):
            return None

        coverage = pattern.constant_char_count / max(len(log_text), 1)
        min_coverage = MIN_PATTERN_COVERAGE_UNANCHORED_V2
        if pattern.anchored_start or pattern.anchored_end:
            min_coverage = MIN_PATTERN_COVERAGE_V2
        if coverage < min_coverage:
            return None

        return (
            pattern.constant_char_count,
            pattern.constant_token_count,
            int(coverage * 1000),
            int(pattern.anchored_start),
            int(pattern.anchored_end),
            -pattern.wildcard_count,
        )

    def _lookup_pattern_v2(self, log_text: str) -> CacheLookupResult | None:
        best_pattern: TemplatePattern | None = None
        best_score: tuple[int, int, int, int, int, int] | None = None
        for pattern in self.pattern_cache.values_mru_first():
            score = self._match_pattern_v2(log_text, pattern)
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_pattern = pattern
                best_score = score
        if best_pattern is None:
            return None
        self.pattern_cache.put(best_pattern.template, best_pattern)
        return CacheLookupResult(template=best_pattern.template, source="pattern_cache")

    def lookup(self, log_text: str) -> CacheLookupResult | None:
        exact = self.exact_cache.get(log_text)
        if exact is not None:
            return CacheLookupResult(template=exact, source="exact_cache")

        signature = self._lookup_signature(log_text)
        if signature is not None:
            return signature

        if self.pattern_cache_version == "v1":
            return self._lookup_pattern_v1(log_text)
        return self._lookup_pattern_v2(log_text)

    def update(self, log_text: str, template: str) -> None:
        normalized = normalize_template_whitespace(template)
        self.exact_cache.put(log_text, normalized)
        pattern = build_template_pattern(normalized)
        self._update_signature_cache(log_text, pattern)
        if self.pattern_cache_version == "v1" or self._is_v2_pattern_admissible(pattern):
            self.pattern_cache.put(pattern.template, pattern)
