from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class LogRecord:
    dataset: str
    line_id: str
    content: str
    event_template: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedDemo:
    dataset: str
    line_id: str
    content: str
    template: str
    score: float
    rank: int


@dataclass(slots=True)
class CacheLookupResult:
    template: str
    source: str


@dataclass(slots=True)
class ParsePrediction:
    dataset: str
    line_id: str
    content: str
    predicted_template: str
    decision_source: str
    cache_hit: bool
    exact_cache_hit: bool
    signature_cache_hit: bool
    pattern_cache_hit: bool
    num_demos: int
    latency_ms: float
    cache_lookup_ms: float
    retrieval_ms: float
    prompt_build_ms: float
    llm_query_ms: float
    validation_ms: float
    model_attempts: int
    raw_response: str = ""
    ground_truth_template: str | None = None
    demo_line_ids: str = ""
    demo_scores: str = ""

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ParseStats:
    dataset: str
    shots: int
    total_logs: int = 0
    exact_cache_hits: int = 0
    signature_cache_hits: int = 0
    pattern_cache_hits: int = 0
    llm_calls: int = 0
    llm_valid_count: int = 0
    llm_invalid_count: int = 0
    fallback_count: int = 0
    failed_count: int = 0
    total_latency_ms: float = 0.0
    total_cache_lookup_ms: float = 0.0
    total_retrieval_ms: float = 0.0
    total_prompt_build_ms: float = 0.0
    total_llm_query_ms: float = 0.0
    total_validation_ms: float = 0.0

    @property
    def cache_hits(self) -> int:
        return self.exact_cache_hits + self.signature_cache_hits + self.pattern_cache_hits

    def _avg(self, value: float) -> float:
        if self.total_logs == 0:
            return 0.0
        return value / self.total_logs

    @property
    def avg_latency_ms(self) -> float:
        return self._avg(self.total_latency_ms)

    @property
    def avg_cache_lookup_ms(self) -> float:
        return self._avg(self.total_cache_lookup_ms)

    @property
    def avg_retrieval_ms(self) -> float:
        return self._avg(self.total_retrieval_ms)

    @property
    def avg_prompt_build_ms(self) -> float:
        return self._avg(self.total_prompt_build_ms)

    @property
    def avg_llm_query_ms(self) -> float:
        return self._avg(self.total_llm_query_ms)

    @property
    def avg_validation_ms(self) -> float:
        return self._avg(self.total_validation_ms)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["cache_hits"] = self.cache_hits
        data["avg_latency_ms"] = self.avg_latency_ms
        data["avg_cache_lookup_ms"] = self.avg_cache_lookup_ms
        data["avg_retrieval_ms"] = self.avg_retrieval_ms
        data["avg_prompt_build_ms"] = self.avg_prompt_build_ms
        data["avg_llm_query_ms"] = self.avg_llm_query_ms
        data["avg_validation_ms"] = self.avg_validation_ms
        return data
