#!/usr/bin/env python3

import argparse
import csv
import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from miclog2.online import OnlineParserConfig, OnlineParserPipeline
from miclog2.online.cache import PATTERN_CACHE_VERSION_CHOICES


DEFAULT_DATASETS = [
    "Apache",
    "BGL",
    "Hadoop",
    "HDFS",
    "HealthApp",
    "HPC",
    "Linux",
    "Mac",
    "OpenSSH",
    "OpenStack",
    "Proxifier",
    "Spark",
    "Thunderbird",
    "Zookeeper",
]


def parse_dataset_names(raw_value: str) -> list[str]:
    names = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not names:
        raise SystemExit("No datasets specified.")
    return names


def model_label(model_path: str) -> str:
    return Path(model_path).name


def default_run_name(model_path: str, adapter_dir: str | None, shots: int) -> str:
    if adapter_dir:
        base = Path(adapter_dir).name
    else:
        base = f"{model_label(model_path)}_base"
    return f"{base}_{shots}shot"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MicLog2.0 online parsing over multiple datasets.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS), help="comma-separated dataset names")
    parser.add_argument("--model-path", required=True, help="absolute or relative path to the base/merged model directory")
    parser.add_argument("--adapter-dir", default=None, help="optional LoRA adapter directory")
    parser.add_argument("--target-root", default="loghub-2.0/full_dataset")
    parser.add_argument("--support-root", default="selected_balanced")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--run-name", default=None, help="subdirectory under output-root; default is derived from model/adapter/shots")
    parser.add_argument(
        "--output-layout",
        choices=["run-first", "dataset-first"],
        default="run-first",
        help="results layout: run-first -> <timestamp>/<run>/<dataset>, dataset-first -> <timestamp>/<dataset>/<run>",
    )
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--retrieval-field", default="Content")
    parser.add_argument("--exact-cache-size", type=int, default=50000)
    parser.add_argument("--signature-cache-size", type=int, default=10000)
    parser.add_argument("--pattern-cache-size", type=int, default=10000)
    parser.add_argument("--pattern-cache-version", choices=PATTERN_CACHE_VERSION_CHOICES, default="v2")
    parser.add_argument("--disable-retrieval-fallback", action="store_true")
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
    parser.add_argument("--max-target-logs", type=int, default=None, help="optional per-dataset limit for smoke tests")
    parser.add_argument("--no-progress", dest="show_progress", action="store_false")
    parser.set_defaults(show_progress=True)
    return parser.parse_args()


def dataset_output_dir(args: argparse.Namespace, dataset: str, run_dir: Path) -> Path:
    if args.output_layout == "dataset-first":
        return run_dir.parent / dataset / run_dir.name
    return run_dir / dataset


def config_for_dataset(args: argparse.Namespace, dataset: str, run_dir: Path) -> OnlineParserConfig:
    return OnlineParserConfig(
        dataset=dataset,
        model_path=args.model_path,
        adapter_dir=args.adapter_dir,
        target_root=args.target_root,
        support_root=args.support_root,
        output_dir=str(dataset_output_dir(args, dataset, run_dir)),
        shots=args.shots,
        retrieval_field=args.retrieval_field,
        exact_cache_size=args.exact_cache_size,
        signature_cache_size=args.signature_cache_size,
        pattern_cache_size=args.pattern_cache_size,
        pattern_cache_version=args.pattern_cache_version,
        exclude_same_content=args.exclude_same_content,
        enable_retrieval_fallback=not args.disable_retrieval_fallback,
        max_generation_attempts=args.max_generation_attempts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        force_cpu=args.force_cpu,
        seed=args.seed,
        max_target_logs=args.max_target_logs,
        show_progress=args.show_progress,
    )


def average(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)


def weighted_average(rows: list[dict[str, Any]], value_key: str, weight_key: str) -> float:
    total_weight = sum(float(row.get(weight_key, 0.0)) for row in rows)
    if total_weight == 0.0:
        return 0.0
    return sum(float(row.get(value_key, 0.0)) * float(row.get(weight_key, 0.0)) for row in rows) / total_weight


def write_dataset_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "dataset",
        "total_logs",
        "shots",
        "cache_hits",
        "exact_cache_hits",
        "signature_cache_hits",
        "pattern_cache_hits",
        "llm_calls",
        "llm_valid_count",
        "llm_invalid_count",
        "fallback_count",
        "failed_count",
        "total_latency_ms",
        "avg_latency_ms",
        "total_cache_lookup_ms",
        "avg_cache_lookup_ms",
        "total_llm_query_ms",
        "avg_llm_query_ms",
        "total_retrieval_ms",
        "avg_retrieval_ms",
        "total_prompt_build_ms",
        "avg_prompt_build_ms",
        "total_validation_ms",
        "avg_validation_ms",
    ]
    ordered = [name for name in preferred if name in fieldnames]
    ordered.extend(name for name in fieldnames if name not in ordered)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def build_overall_summary(rows: list[dict[str, Any]], args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    total_logs = sum(int(row.get("total_logs", 0)) for row in rows)
    total_latency_ms = sum(float(row.get("total_latency_ms", 0.0)) for row in rows)
    total_cache_lookup_ms = sum(float(row.get("total_cache_lookup_ms", 0.0)) for row in rows)
    total_llm_query_ms = sum(float(row.get("total_llm_query_ms", 0.0)) for row in rows)
    total_retrieval_ms = sum(float(row.get("total_retrieval_ms", 0.0)) for row in rows)
    total_prompt_build_ms = sum(float(row.get("total_prompt_build_ms", 0.0)) for row in rows)
    total_validation_ms = sum(float(row.get("total_validation_ms", 0.0)) for row in rows)
    return {
        "dataset_count": len(rows),
        "datasets": [row.get("dataset") for row in rows],
        "model_path": args.model_path,
        "adapter_dir": args.adapter_dir,
        "shots": args.shots,
        "exact_cache_size": args.exact_cache_size,
        "signature_cache_size": args.signature_cache_size,
        "pattern_cache_size": args.pattern_cache_size,
        "pattern_cache_version": args.pattern_cache_version,
        "run_dir": str(run_dir),
        "total_logs": total_logs,
        "total_latency_ms": total_latency_ms,
        "total_cache_lookup_ms": total_cache_lookup_ms,
        "total_llm_query_ms": total_llm_query_ms,
        "total_retrieval_ms": total_retrieval_ms,
        "total_prompt_build_ms": total_prompt_build_ms,
        "total_validation_ms": total_validation_ms,
        "avg_dataset_latency_ms": average(rows, "avg_latency_ms"),
        "avg_dataset_cache_lookup_ms": average(rows, "avg_cache_lookup_ms"),
        "avg_dataset_llm_query_ms": average(rows, "avg_llm_query_ms"),
        "avg_dataset_retrieval_ms": average(rows, "avg_retrieval_ms"),
        "avg_dataset_prompt_build_ms": average(rows, "avg_prompt_build_ms"),
        "avg_dataset_validation_ms": average(rows, "avg_validation_ms"),
        "weighted_avg_latency_ms": 0.0 if total_logs == 0 else total_latency_ms / total_logs,
        "weighted_avg_cache_lookup_ms": 0.0 if total_logs == 0 else total_cache_lookup_ms / total_logs,
        "weighted_avg_llm_query_ms": 0.0 if total_logs == 0 else total_llm_query_ms / total_logs,
        "weighted_avg_retrieval_ms": 0.0 if total_logs == 0 else total_retrieval_ms / total_logs,
        "weighted_avg_prompt_build_ms": 0.0 if total_logs == 0 else total_prompt_build_ms / total_logs,
        "weighted_avg_validation_ms": 0.0 if total_logs == 0 else total_validation_ms / total_logs,
        "total_cache_hits": sum(int(row.get("cache_hits", 0)) for row in rows),
        "total_exact_cache_hits": sum(int(row.get("exact_cache_hits", 0)) for row in rows),
        "total_signature_cache_hits": sum(int(row.get("signature_cache_hits", 0)) for row in rows),
        "total_pattern_cache_hits": sum(int(row.get("pattern_cache_hits", 0)) for row in rows),
        "total_llm_calls": sum(int(row.get("llm_calls", 0)) for row in rows),
        "total_fallback_count": sum(int(row.get("fallback_count", 0)) for row in rows),
        "total_failed_count": sum(int(row.get("failed_count", 0)) for row in rows),
    }


def main() -> int:
    args = parse_args()
    datasets = parse_dataset_names(args.datasets)
    run_name = args.run_name or default_run_name(args.model_path, args.adapter_dir, args.shots)
    run_dir = (Path(args.output_root) / timestamp() / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, dataset in enumerate(datasets, start=1):
        print(f"[{index}/{len(datasets)}] parsing {dataset}", flush=True)
        config = config_for_dataset(args, dataset, run_dir)
        pipeline = OnlineParserPipeline(config)
        stats = pipeline.run()
        row = stats.to_dict()
        rows.append(row)
        print(
            f"[{dataset}] total_logs={row['total_logs']} "
            f"avg_latency_ms={row['avg_latency_ms']:.3f} "
            f"avg_cache_lookup_ms={row['avg_cache_lookup_ms']:.3f} "
            f"signature_cache_hits={row.get('signature_cache_hits', 0)} "
            f"avg_llm_query_ms={row['avg_llm_query_ms']:.3f}",
            flush=True,
        )
        del stats
        del pipeline
        del config
        gc.collect()

    dataset_summary_path = run_dir / "dataset_summary.csv"
    overall_summary_path = run_dir / "overall_summary.json"
    write_dataset_summary(dataset_summary_path, rows)
    overall = build_overall_summary(rows, args, run_dir)
    overall_summary_path.write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"dataset_summary={dataset_summary_path}")
    print(f"overall_summary={overall_summary_path}")
    print(f"weighted_avg_latency_ms={overall['weighted_avg_latency_ms']:.3f}")
    print(f"weighted_avg_cache_lookup_ms={overall['weighted_avg_cache_lookup_ms']:.3f}")
    print(f"weighted_avg_llm_query_ms={overall['weighted_avg_llm_query_ms']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
