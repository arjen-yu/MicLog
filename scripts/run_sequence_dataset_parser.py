#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from miclog2.online.cache import MultiLevelTemplateCache, PATTERN_CACHE_VERSION_CHOICES
from miclog2.online.config import DEFAULT_INSTRUCTION, DEFAULT_SYSTEM_PROMPT
from miclog2.online.io import load_support_bank
from miclog2.online.model_runner import HFTemplateGenerator
from miclog2.online.prompting import build_messages
from miclog2.online.retriever import BM25Retriever, demos_for_prompt
from miclog2.online.types import LogRecord, ParseStats, RetrievedDemo
from miclog2.online.validator import extract_template_from_response, validate_template_against_log


DEFAULT_SEQUENCE_DATASETS = ["BGL", "HDFS_v1", "Liberty", "Thunderbird"]
DEFAULT_SPLITS = ["train", "test"]
DEFAULT_MODEL_PATH = "/tempdisk2/yjb/Models/Qwen3.5-4B"
DEFAULT_ADAPTER_DIR = "outputs/qwen35_4b_lora_0-5-shot"
HDFS_SUPPORT_NAME = "HDFS"
GLOBAL_SUPPORT_SCOPE = "__ALL__"
SEQUENCE_DELIMITER = " ;-; "
SEQUENCE_SPLIT_RE = re.compile(r"\s*;-;\s*")


def parse_names(raw_value: str) -> list[str]:
    names = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not names:
        raise SystemExit("No items specified.")
    return names


def count_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return max(sum(1 for _ in reader) - 1, 0)


class SequenceDatasetParser:
    def __init__(
        self,
        *,
        datasets_root: Path,
        support_root: Path,
        model_path: str,
        adapter_dir: str | None,
        shots: int,
        exact_cache_size: int,
        signature_cache_size: int,
        pattern_cache_size: int,
        pattern_cache_version: str,
        exclude_same_content: bool,
        max_generation_attempts: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        attn_implementation: str,
        trust_remote_code: bool,
        force_cpu: bool,
        seed: int,
        show_progress: bool,
        flush_every_rows: int,
    ) -> None:
        self.datasets_root = datasets_root.resolve()
        self.support_root = support_root.resolve()
        self.shots = shots
        self.exact_cache_size = exact_cache_size
        self.signature_cache_size = signature_cache_size
        self.pattern_cache_size = pattern_cache_size
        self.pattern_cache_version = pattern_cache_version
        self.exclude_same_content = exclude_same_content
        self.max_generation_attempts = max_generation_attempts
        self.show_progress = show_progress
        self.flush_every_rows = max(flush_every_rows, 1)

        self.available_support_datasets = sorted(path.name for path in self.support_root.iterdir() if path.is_dir())
        self._support_records_cache: dict[str, list[LogRecord]] = {}
        self._retriever_cache: dict[str, BM25Retriever] = {}
        self.generator = HFTemplateGenerator(
            model_path=model_path,
            adapter_dir=adapter_dir,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            force_cpu=force_cpu,
            seed=seed,
        )

    def _scope_for_dataset(self, dataset_name: str) -> str:
        if dataset_name == "Liberty":
            return GLOBAL_SUPPORT_SCOPE
        if dataset_name == "HDFS_v1":
            return HDFS_SUPPORT_NAME
        return dataset_name

    def _support_names_for_scope(self, scope: str) -> list[str]:
        if scope == GLOBAL_SUPPORT_SCOPE:
            return self.available_support_datasets
        if scope not in self.available_support_datasets:
            raise FileNotFoundError(
                f"Support dataset '{scope}' not found under {self.support_root}. "
                f"Available: {', '.join(self.available_support_datasets)}"
            )
        return [scope]

    def _support_records_for_scope(self, scope: str) -> list[LogRecord]:
        if scope not in self._support_records_cache:
            records: list[LogRecord] = []
            for support_name in self._support_names_for_scope(scope):
                records.extend(load_support_bank(self.support_root, support_name))
            self._support_records_cache[scope] = records
        return self._support_records_cache[scope]

    def _retriever_for_scope(self, scope: str) -> BM25Retriever:
        if scope not in self._retriever_cache:
            self._retriever_cache[scope] = BM25Retriever(
                self._support_records_for_scope(scope),
                retrieval_field="Content",
                exclude_same_content=self.exclude_same_content,
            )
        return self._retriever_cache[scope]

    def _new_cache(self) -> MultiLevelTemplateCache:
        return MultiLevelTemplateCache(
            exact_cache_size=self.exact_cache_size,
            pattern_cache_size=self.pattern_cache_size,
            signature_cache_size=self.signature_cache_size,
            pattern_cache_version=self.pattern_cache_version,
        )

    def _add_timings(
        self,
        stats: ParseStats,
        *,
        cache_lookup_ms: float,
        retrieval_ms: float,
        prompt_build_ms: float,
        llm_query_ms: float,
        validation_ms: float,
        latency_ms: float,
    ) -> None:
        stats.total_cache_lookup_ms += cache_lookup_ms
        stats.total_retrieval_ms += retrieval_ms
        stats.total_prompt_build_ms += prompt_build_ms
        stats.total_llm_query_ms += llm_query_ms
        stats.total_validation_ms += validation_ms
        stats.total_latency_ms += latency_ms

    def _parse_single_log(
        self,
        *,
        dataset_name: str,
        query_id: str,
        log_text: str,
        retriever: BM25Retriever,
        cache: MultiLevelTemplateCache,
        stats: ParseStats,
    ) -> str:
        started = time.perf_counter()
        stats.total_logs += 1
        cache_lookup_ms = 0.0
        retrieval_ms = 0.0
        prompt_build_ms = 0.0
        llm_query_ms = 0.0
        validation_ms = 0.0

        cache_started = time.perf_counter()
        cache_result = cache.lookup(log_text)
        cache_lookup_ms = (time.perf_counter() - cache_started) * 1000.0
        if cache_result is not None:
            if cache_result.source == "exact_cache":
                stats.exact_cache_hits += 1
            elif cache_result.source == "signature_cache":
                stats.signature_cache_hits += 1
            else:
                stats.pattern_cache_hits += 1
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._add_timings(
                stats,
                cache_lookup_ms=cache_lookup_ms,
                retrieval_ms=retrieval_ms,
                prompt_build_ms=prompt_build_ms,
                llm_query_ms=llm_query_ms,
                validation_ms=validation_ms,
                latency_ms=latency_ms,
            )
            return cache_result.template

        query_record = LogRecord(dataset=dataset_name, line_id=query_id, content=log_text, event_template=None, extra={})
        retrieval_started = time.perf_counter()
        demos = retriever.retrieve(query_record, self.shots) if self.shots > 0 else []
        prompt_demos = demos_for_prompt(demos)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0

        prompt_started = time.perf_counter()
        messages = build_messages(
            query_content=log_text,
            demos=prompt_demos,
            instruction=DEFAULT_INSTRUCTION,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )
        prompt_build_ms = (time.perf_counter() - prompt_started) * 1000.0

        stats.llm_calls += 1
        predicted_template = ""
        raw_response = ""
        valid = False
        for _ in range(self.max_generation_attempts):
            llm_started = time.perf_counter()
            raw_response = self.generator.generate(messages)
            llm_query_ms += (time.perf_counter() - llm_started) * 1000.0

            validation_started = time.perf_counter()
            predicted_template = extract_template_from_response(raw_response)
            if predicted_template:
                valid, _errors, _replacements = validate_template_against_log(log_text, predicted_template)
            validation_ms += (time.perf_counter() - validation_started) * 1000.0
            if valid:
                break

        if valid:
            stats.llm_valid_count += 1
            cache.update(log_text, predicted_template)
        else:
            stats.llm_invalid_count += 1
            predicted_template = ""
            stats.failed_count += 1

        latency_ms = (time.perf_counter() - started) * 1000.0
        self._add_timings(
            stats,
            cache_lookup_ms=cache_lookup_ms,
            retrieval_ms=retrieval_ms,
            prompt_build_ms=prompt_build_ms,
            llm_query_ms=llm_query_ms,
            validation_ms=validation_ms,
            latency_ms=latency_ms,
        )
        return predicted_template

    def _split_sequence(self, content: str) -> list[str]:
        stripped = str(content).strip()
        if not stripped:
            return []
        return [part.strip() for part in SEQUENCE_SPLIT_RE.split(stripped) if part.strip()]

    def _parse_sequence_content(
        self,
        *,
        dataset_name: str,
        content: str,
        row_index: int,
        retriever: BM25Retriever,
        cache: MultiLevelTemplateCache,
        stats: ParseStats,
    ) -> str:
        logs = self._split_sequence(content)
        if not logs:
            return ""
        templates: list[str] = []
        for log_index, log_text in enumerate(logs, start=1):
            template = self._parse_single_log(
                dataset_name=dataset_name,
                query_id=f"{row_index}:{log_index}",
                log_text=log_text,
                retriever=retriever,
                cache=cache,
                stats=stats,
            )
            templates.append(template)
        return SEQUENCE_DELIMITER.join(templates)

    def parse_csv_file(self, dataset_name: str, input_csv: Path, output_csv: Path) -> ParseStats:
        scope = self._scope_for_dataset(dataset_name)
        retriever = self._retriever_for_scope(scope)
        cache = self._new_cache()
        stats = ParseStats(dataset=f"{dataset_name}/{input_csv.stem}", shots=self.shots)
        total_rows = count_rows(input_csv) if self.show_progress else None

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with input_csv.open("r", encoding="utf-8", newline="") as src, output_csv.open(
            "w", encoding="utf-8", newline=""
        ) as dst:
            reader = csv.DictReader(src)
            if not reader.fieldnames:
                raise RuntimeError(f"Empty CSV header: {input_csv}")
            if "Content" not in reader.fieldnames:
                raise RuntimeError(f"Missing 'Content' column in {input_csv}")

            fieldnames = list(reader.fieldnames)
            if "Template" not in fieldnames:
                fieldnames.append("Template")
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()

            iterator = reader
            if self.show_progress:
                iterator = tqdm(
                    reader,
                    total=total_rows,
                    desc=f"Parsing {dataset_name}/{input_csv.name}",
                    unit="row",
                    dynamic_ncols=True,
                )

            for row_index, row in enumerate(iterator, start=1):
                row["Template"] = self._parse_sequence_content(
                    dataset_name=dataset_name,
                    content=row.get("Content", ""),
                    row_index=row_index,
                    retriever=retriever,
                    cache=cache,
                    stats=stats,
                )
                writer.writerow(row)
                if row_index % self.flush_every_rows == 0:
                    dst.flush()
                if self.show_progress and hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(
                        logs=stats.total_logs,
                        cache_hits=stats.cache_hits,
                        llm_calls=stats.llm_calls,
                        failed=stats.failed_count,
                    )

        return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse sequence-style datasets under Datasets/*/{train,test}.csv.")
    parser.add_argument("--datasets-root", default="Datasets")
    parser.add_argument("--support-root", default="selected_balanced")
    parser.add_argument("--datasets", default=",".join(DEFAULT_SEQUENCE_DATASETS))
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--exact-cache-size", type=int, default=50000)
    parser.add_argument("--signature-cache-size", type=int, default=10000)
    parser.add_argument("--pattern-cache-size", type=int, default=10000)
    parser.add_argument("--pattern-cache-version", choices=PATTERN_CACHE_VERSION_CHOICES, default="v2")
    parser.add_argument("--no-exclude-same-content", dest="exclude_same_content", action="store_false")
    parser.set_defaults(exclude_same_content=True)
    parser.add_argument("--max-generation-attempts", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--attn-implementation", choices=["auto", "sdpa", "flash_attention_2", "eager"], default="sdpa")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flush-every-rows", type=int, default=1000)
    parser.add_argument("--no-progress", dest="show_progress", action="store_false")
    parser.set_defaults(show_progress=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = parse_names(args.datasets)
    splits = parse_names(args.splits)
    runner = SequenceDatasetParser(
        datasets_root=Path(args.datasets_root),
        support_root=Path(args.support_root),
        model_path=args.model_path,
        adapter_dir=args.adapter_dir,
        shots=args.shots,
        exact_cache_size=args.exact_cache_size,
        signature_cache_size=args.signature_cache_size,
        pattern_cache_size=args.pattern_cache_size,
        pattern_cache_version=args.pattern_cache_version,
        exclude_same_content=args.exclude_same_content,
        max_generation_attempts=args.max_generation_attempts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        force_cpu=args.force_cpu,
        seed=args.seed,
        show_progress=args.show_progress,
        flush_every_rows=args.flush_every_rows,
    )

    for dataset_name in datasets:
        dataset_dir = runner.datasets_root / dataset_name
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        for split in splits:
            input_csv = dataset_dir / f"{split}.csv"
            output_csv = dataset_dir / f"{split}_parsed.csv"
            if not input_csv.exists():
                raise FileNotFoundError(f"Missing input CSV: {input_csv}")
            print(f"parsing {input_csv} -> {output_csv}", flush=True)
            stats = runner.parse_csv_file(dataset_name, input_csv, output_csv)
            print(
                f"[{dataset_name}/{split}] rows_written={count_rows(output_csv)} "
                f"logs={stats.total_logs} cache_hits={stats.cache_hits} "
                f"signature_hits={stats.signature_cache_hits} llm_calls={stats.llm_calls} "
                f"failed={stats.failed_count} "
                f"avg_latency_ms={stats.avg_latency_ms:.3f}",
                flush=True,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
