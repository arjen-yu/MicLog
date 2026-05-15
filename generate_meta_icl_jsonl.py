#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SELECTED_ROOT = ROOT / "selected_balanced"
OUTPUT_ROOT = ROOT / "meta_incontext_data_variants"
DEFAULT_MAX_SHOTS = 5
TOKEN_RE = re.compile(r"<[a-z_]+>|[A-Za-z_]+|\d+|[^\sA-Za-z_0-9]")
PLACEHOLDER_RE = re.compile(r"<([a-z_]+)>")

INSTRUCTION = (
    "For each query log after the final <content> tag, try your best to extract one log template "
    "(substitute variable tokens in the log as <*> and remain constant tokens to construct the template) "
    "and put the template after the final <template> tag and between <START> and <END> tags. "
    "Use previous <content>/<template> pairs as in-context examples when they are provided."
)
SYSTEM_PROMPT = "You are a log parsing assistant that converts raw log messages into templates."


@dataclass(frozen=True)
class LogSample:
    dataset: str
    index: int
    line_id: int
    content: str
    event_template: str
    event_id: str
    normalized_content: str
    cluster_id: str
    selected_rank: int


@dataclass(frozen=True)
class Variant:
    name: str
    variant_type: str
    shot_numbers: tuple[int, ...]

    @property
    def shot_spec(self) -> str:
        return ",".join(str(shot) for shot in self.shot_numbers)


@dataclass(frozen=True)
class LiteDecision:
    cluster_kind: str
    keep_count: int


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_count = len(documents)
        self.doc_lengths = [len(document) for document in documents]
        self.avg_doc_length = sum(self.doc_lengths) / self.doc_count if self.doc_count else 0.0
        self.term_freqs = [Counter(document) for document in documents]
        doc_freq: Counter[str] = Counter()
        for document in documents:
            doc_freq.update(set(document))
        self.idf = {
            term: math.log(1.0 + (self.doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def score(self, query_terms: list[str], doc_index: int) -> float:
        if not query_terms or self.doc_count == 0:
            return 0.0
        doc_length = self.doc_lengths[doc_index]
        if doc_length == 0 or self.avg_doc_length == 0.0:
            return 0.0
        term_freq = self.term_freqs[doc_index]
        score = 0.0
        for term in set(query_terms):
            freq = term_freq.get(term, 0)
            if freq == 0:
                continue
            idf = self.idf.get(term, 0.0)
            denominator = freq + self.k1 * (1.0 - self.b + self.b * doc_length / self.avg_doc_length)
            score += idf * (freq * (self.k1 + 1.0)) / denominator
        return score

    def most_similar(self, query_index: int, top_k: int, samples: list[LogSample]) -> list[tuple[int, float]]:
        query_terms = self.documents[query_index]
        scored = []
        for doc_index in range(self.doc_count):
            if doc_index == query_index:
                continue
            scored.append((doc_index, self.score(query_terms, doc_index)))
        scored.sort(
            key=lambda item: (
                -item[1],
                samples[item[0]].line_id,
                samples[item[0]].index,
            )
        )
        return scored[:top_k]


def parse_line_id(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 10**18


def parse_selected_rank(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 10**9


def load_dataset_samples(dataset_dir: Path) -> list[LogSample]:
    csv_paths = sorted(dataset_dir.glob("*_structured.csv"))
    if not csv_paths:
        raise RuntimeError(f"No *_structured.csv found in {dataset_dir}")
    if len(csv_paths) > 1:
        raise RuntimeError(f"Multiple *_structured.csv files found in {dataset_dir}")

    samples = []
    with csv_paths[0].open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"LineId", "Content", "EventTemplate"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Missing columns in {csv_paths[0]}: {', '.join(sorted(missing))}")
        for index, row in enumerate(reader):
            samples.append(
                LogSample(
                    dataset=dataset_dir.name,
                    index=index,
                    line_id=parse_line_id(row.get("LineId", "")),
                    content=row["Content"],
                    event_template=row["EventTemplate"],
                    event_id=row.get("EventId", ""),
                    normalized_content=row.get("normalized_content", ""),
                    cluster_id=row.get("cluster_id", ""),
                    selected_rank=parse_selected_rank(row.get("selected_rank", "")),
                )
            )
    return samples


def selected_dataset_dirs(input_root: Path, dataset_names: list[str] | None) -> list[Path]:
    if not input_root.exists():
        raise RuntimeError(f"Input root does not exist: {input_root}")
    dataset_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
    if dataset_names:
        requested = set(dataset_names)
        dataset_dirs = [path for path in dataset_dirs if path.name in requested]
        missing = sorted(requested - {path.name for path in dataset_dirs})
        if missing:
            raise RuntimeError(f"Requested dataset(s) not found: {', '.join(missing)}")
    return dataset_dirs


def build_variants(max_shots: int) -> list[Variant]:
    variants: list[Variant] = []
    for shot in range(max_shots + 1):
        variants.append(Variant(f"{shot}-shot-only", "shot-only", (shot,)))
    for max_progressive_shot in range(1, max_shots + 1):
        variants.append(
            Variant(
                f"0-{max_progressive_shot}-shot",
                "progressive",
                tuple(range(max_progressive_shot + 1)),
            )
        )
    return variants


def build_documents(samples: list[LogSample], retrieval_field: str) -> list[list[str]]:
    if retrieval_field == "content":
        return [tokenize(sample.content) for sample in samples]
    if retrieval_field == "normalized_content":
        return [tokenize(sample.normalized_content or sample.content) for sample in samples]
    raise RuntimeError(f"Unsupported retrieval field: {retrieval_field}")


def precompute_top_examples(
    samples: list[LogSample],
    max_shots: int,
    retrieval_field: str,
) -> dict[int, list[tuple[int, float]]]:
    documents = build_documents(samples, retrieval_field)
    bm25 = BM25Index(documents)
    max_examples = min(max_shots, max(0, len(samples) - 1))
    return {
        sample.index: bm25.most_similar(sample.index, max_examples, samples)
        for sample in samples
    }


def template_answer(event_template: str) -> str:
    return f"<START>{event_template}<END>"


def demonstrations_for_shot(
    retrieval_samples: list[LogSample],
    top_examples: list[tuple[int, float]],
    shot: int,
) -> list[LogSample]:
    used = top_examples[:shot]
    return [retrieval_samples[doc_index] for doc_index, _score in reversed(used)]


def build_input(query: LogSample, demonstrations: list[LogSample]) -> str:
    lines: list[str] = []
    if demonstrations:
        lines.append("Examples:")
        for example in demonstrations:
            lines.append(f"<content>{example.content}")
            lines.append(f"<template>{template_answer(example.event_template)}")
            lines.append("")
        lines.append("Query:")
    lines.append(f"<content>{query.content}")
    lines.append("<template>")
    return "\n".join(lines).rstrip()


def build_instruction_record(query: LogSample, demonstrations: list[LogSample]) -> dict[str, str]:
    return {
        "instruction": INSTRUCTION,
        "input": build_input(query, demonstrations),
        "output": template_answer(query.event_template),
    }


def build_chat_record(query: LogSample, demonstrations: list[LogSample]) -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{INSTRUCTION}\n\n{build_input(query, demonstrations)}"},
            {"role": "assistant", "content": template_answer(query.event_template)},
        ]
    }


def build_record(query: LogSample, demonstrations: list[LogSample], output_format: str) -> dict[str, object]:
    if output_format == "instruction":
        return build_instruction_record(query, demonstrations)
    if output_format == "chat":
        return build_chat_record(query, demonstrations)
    raise RuntimeError(f"Unsupported output format: {output_format}")


def build_metadata(
    variant: Variant,
    query: LogSample,
    shot: int,
    demonstrations: list[LogSample],
    similarity_scores: dict[int, float],
) -> dict[str, object]:
    return {
        "variant_name": variant.name,
        "variant_type": variant.variant_type,
        "shot": shot,
        "dataset": query.dataset,
        "line_id": query.line_id,
        "event_id": query.event_id,
        "cluster_id": query.cluster_id,
        "example_line_ids": [example.line_id for example in demonstrations],
        "example_event_ids": [example.event_id for example in demonstrations],
        "example_cluster_ids": [example.cluster_id for example in demonstrations],
        "example_bm25_scores": [similarity_scores[example.index] for example in demonstrations],
    }


def write_jsonl_row(handle, row: dict[str, object]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    handle.write("\n")


def decide_lite_cluster(rows: list[LogSample]) -> LiteDecision:
    rows_sorted = sorted(rows, key=lambda sample: (sample.selected_rank, sample.line_id, sample.index))
    normalized_text = max((sample.normalized_content for sample in rows_sorted), key=len)
    content_text = max((sample.content for sample in rows_sorted), key=len)
    placeholders = PLACEHOLDER_RE.findall(normalized_text)
    ph_count = len(placeholders)
    ph_types = len(set(placeholders))
    token_count = len(TOKEN_RE.findall(normalized_text))
    symbol_count = sum(1 for ch in normalized_text if not ch.isalnum() and not ch.isspace() and ch not in "<>")
    long_text = len(content_text) >= 120
    very_long_text = len(content_text) >= 180
    complex_symbols = symbol_count >= 8
    very_complex_symbols = symbol_count >= 12
    complex_tokens = token_count >= 16
    very_complex_tokens = token_count >= 24
    multi_var = ph_count >= 3 or ph_types >= 2
    simple_var = ph_count == 1 and token_count <= 12 and symbol_count <= 5 and not long_text
    trivial = ph_count == 0
    very_complex = ph_count >= 5 or very_complex_tokens or very_complex_symbols or very_long_text

    if trivial:
        return LiteDecision("trivial", 0)
    if simple_var:
        return LiteDecision("single_var", min(1, len(rows_sorted)))
    if very_complex:
        return LiteDecision("very_complex", min(3, len(rows_sorted)))
    if multi_var or complex_symbols or complex_tokens or long_text:
        return LiteDecision("complex", min(2, len(rows_sorted)))
    return LiteDecision("medium", min(1, len(rows_sorted)))


def choose_query_samples(retrieval_samples: list[LogSample], query_mode: str) -> tuple[list[LogSample], dict[str, int]]:
    if query_mode == "full":
        return retrieval_samples, {
            "query_sample_count": len(retrieval_samples),
            "trivial_cluster_count": 0,
            "single_var_cluster_count": 0,
            "medium_cluster_count": 0,
            "complex_cluster_count": 0,
            "very_complex_cluster_count": 0,
        }

    if query_mode != "lite":
        raise RuntimeError(f"Unsupported query mode: {query_mode}")

    by_cluster: dict[str, list[LogSample]] = defaultdict(list)
    for sample in retrieval_samples:
        by_cluster[sample.cluster_id].append(sample)

    chosen: list[LogSample] = []
    stats = Counter()
    for rows in by_cluster.values():
        rows_sorted = sorted(rows, key=lambda sample: (sample.selected_rank, sample.line_id, sample.index))
        decision = decide_lite_cluster(rows_sorted)
        stats[f"{decision.cluster_kind}_cluster_count"] += 1
        if decision.keep_count > 0:
            chosen.extend(rows_sorted[: decision.keep_count])
    chosen.sort(key=lambda sample: sample.index)
    stats["query_sample_count"] = len(chosen)
    return chosen, dict(stats)


def generate_all(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    variants = build_variants(args.max_shots)
    dataset_dirs = selected_dataset_dirs(input_root, args.datasets)
    summary_counts: Counter[tuple[str, str], int] = Counter()
    retrieval_counts: dict[str, int] = {}
    query_counts: dict[str, int] = {}
    lite_stats_by_dataset: dict[str, dict[str, int]] = {}

    train_handles = {}
    metadata_handles = {}
    try:
        for variant in variants:
            variant_dir = output_root / variant.name
            variant_dir.mkdir(parents=True, exist_ok=True)
            train_handles[variant.name] = (variant_dir / "train.jsonl").open("w", encoding="utf-8")
            if args.write_metadata:
                metadata_handles[variant.name] = (variant_dir / "metadata.jsonl").open("w", encoding="utf-8")

        for dataset_dir in dataset_dirs:
            retrieval_samples = load_dataset_samples(dataset_dir)
            query_samples, lite_stats = choose_query_samples(retrieval_samples, args.query_mode)
            retrieval_counts[dataset_dir.name] = len(retrieval_samples)
            query_counts[dataset_dir.name] = len(query_samples)
            lite_stats_by_dataset[dataset_dir.name] = lite_stats
            top_examples_by_query = precompute_top_examples(retrieval_samples, args.max_shots, args.retrieval_field)

            for query in query_samples:
                top_examples = top_examples_by_query[query.index]
                similarity_scores = {doc_index: score for doc_index, score in top_examples}
                for variant in variants:
                    for shot in variant.shot_numbers:
                        demonstrations = demonstrations_for_shot(retrieval_samples, top_examples, shot)
                        record = build_record(query, demonstrations, args.output_format)
                        write_jsonl_row(train_handles[variant.name], record)
                        if args.write_metadata:
                            metadata = build_metadata(variant, query, shot, demonstrations, similarity_scores)
                            write_jsonl_row(metadata_handles[variant.name], metadata)
                        summary_counts[(variant.name, dataset_dir.name)] += 1

            print(
                f"[{dataset_dir.name}] retrieval={len(retrieval_samples)} queries={len(query_samples)} mode={args.query_mode}",
                flush=True,
            )
    finally:
        for handle in train_handles.values():
            handle.close()
        for handle in metadata_handles.values():
            handle.close()

    summary_path = output_root / "experiment_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "variant_name",
            "variant_type",
            "shot_numbers",
            "dataset_name",
            "retrieval_log_count",
            "query_log_count",
            "jsonl_record_count",
            "retrieval_field",
            "output_format",
            "query_mode",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for variant in variants:
            for dataset_dir in dataset_dirs:
                dataset_name = dataset_dir.name
                writer.writerow(
                    {
                        "variant_name": variant.name,
                        "variant_type": variant.variant_type,
                        "shot_numbers": variant.shot_spec,
                        "dataset_name": dataset_name,
                        "retrieval_log_count": retrieval_counts[dataset_name],
                        "query_log_count": query_counts[dataset_name],
                        "jsonl_record_count": summary_counts[(variant.name, dataset_name)],
                        "retrieval_field": args.retrieval_field,
                        "output_format": args.output_format,
                        "query_mode": args.query_mode,
                    }
                )

    if args.query_mode == "lite":
        lite_summary_path = output_root / "lite_query_summary.csv"
        with lite_summary_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "dataset_name",
                "retrieval_log_count",
                "query_log_count",
                "query_ratio",
                "trivial_cluster_count",
                "single_var_cluster_count",
                "medium_cluster_count",
                "complex_cluster_count",
                "very_complex_cluster_count",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            total_retrieval = 0
            total_queries = 0
            total_counter = Counter()
            for dataset_dir in dataset_dirs:
                dataset_name = dataset_dir.name
                retrieval_count = retrieval_counts[dataset_name]
                query_count = query_counts[dataset_name]
                stats = lite_stats_by_dataset[dataset_name]
                total_retrieval += retrieval_count
                total_queries += query_count
                total_counter.update(stats)
                writer.writerow(
                    {
                        "dataset_name": dataset_name,
                        "retrieval_log_count": retrieval_count,
                        "query_log_count": query_count,
                        "query_ratio": f"{query_count / retrieval_count:.6f}",
                        "trivial_cluster_count": stats.get("trivial_cluster_count", 0),
                        "single_var_cluster_count": stats.get("single_var_cluster_count", 0),
                        "medium_cluster_count": stats.get("medium_cluster_count", 0),
                        "complex_cluster_count": stats.get("complex_cluster_count", 0),
                        "very_complex_cluster_count": stats.get("very_complex_cluster_count", 0),
                    }
                )
            writer.writerow(
                {
                    "dataset_name": "TOTAL",
                    "retrieval_log_count": total_retrieval,
                    "query_log_count": total_queries,
                    "query_ratio": f"{total_queries / total_retrieval:.6f}",
                    "trivial_cluster_count": total_counter.get("trivial_cluster_count", 0),
                    "single_var_cluster_count": total_counter.get("single_var_cluster_count", 0),
                    "medium_cluster_count": total_counter.get("medium_cluster_count", 0),
                    "complex_cluster_count": total_counter.get("complex_cluster_count", 0),
                    "very_complex_cluster_count": total_counter.get("very_complex_cluster_count", 0),
                }
            )
        print(f"Lite query summary written to {lite_summary_path}", flush=True)

    print(f"Wrote {len(variants)} experiment variants under {output_root}", flush=True)
    print(f"Experiment summary written to {summary_path}", flush=True)
    if args.write_metadata:
        print("Metadata JSONL files were written because --write-metadata was enabled", flush=True)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate shot-only and progressive meta in-context learning JSONL variants."
    )
    parser.add_argument("--input-root", default=str(SELECTED_ROOT), help="selected_balanced root directory")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT), help="output root directory")
    parser.add_argument("--max-shots", type=int, default=DEFAULT_MAX_SHOTS, help="maximum number of in-context examples")
    parser.add_argument(
        "--retrieval-field",
        choices=["content", "normalized_content"],
        default="content",
        help="field used by BM25 to find same-dataset examples",
    )
    parser.add_argument(
        "--output-format",
        choices=["instruction", "chat"],
        default="instruction",
        help="JSONL schema: instruction/input/output or chat messages",
    )
    parser.add_argument(
        "--query-mode",
        choices=["full", "lite"],
        default="full",
        help="full uses all selected_balanced rows as queries; lite keeps full retrieval bank but downsamples query rows",
    )
    parser.add_argument(
        "--write-metadata",
        action="store_true",
        help="also write per-variant metadata.jsonl files for debugging/reproducibility",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="only process the named dataset; can be repeated",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    if args.max_shots < 0:
        raise RuntimeError("--max-shots must be non-negative")
    generate_all(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
