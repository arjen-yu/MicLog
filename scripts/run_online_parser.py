#!/usr/bin/env python3

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from miclog2.online import OnlineParserConfig, OnlineParserPipeline
from miclog2.online.cache import PATTERN_CACHE_VERSION_CHOICES


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def derive_output_dir(dataset: str, model_path: str, adapter_dir: str | None, shots: int) -> str:
    model_name = Path(model_path).name
    if adapter_dir:
        run_name = Path(adapter_dir).name
    else:
        run_name = f"{model_name}_base"
    return str((Path("results") / timestamp() / f"{dataset}_{run_name}_{shots}shot").resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MicLog online log parsing.")
    parser.add_argument("--dataset", required=True, help="dataset name, e.g. Apache")
    parser.add_argument("--model-path", required=True, help="absolute or relative path to the base/merged model directory")
    parser.add_argument("--adapter-dir", default=None, help="optional LoRA adapter directory")
    parser.add_argument("--target-root", default="loghub-2.0/full_dataset", help="root directory containing target structured CSVs")
    parser.add_argument("--support-root", default="selected_balanced", help="root directory containing retrieval support structured CSVs")
    parser.add_argument("--output-dir", default=None, help="output directory for predictions and summary")
    parser.add_argument("--shots", type=int, default=5, help="number of retrieved demonstrations to use")
    parser.add_argument("--retrieval-field", default="Content", help="support-bank field used for BM25 retrieval")
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
    parser.add_argument("--max-target-logs", type=int, default=None, help="optional limit for debugging/smoke tests")
    parser.add_argument("--no-progress", dest="show_progress", action="store_false")
    parser.set_defaults(show_progress=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or derive_output_dir(
        dataset=args.dataset,
        model_path=args.model_path,
        adapter_dir=args.adapter_dir,
        shots=args.shots,
    )
    config = OnlineParserConfig(
        dataset=args.dataset,
        model_path=args.model_path,
        adapter_dir=args.adapter_dir,
        target_root=args.target_root,
        support_root=args.support_root,
        output_dir=output_dir,
        shots=args.shots,
        retrieval_field=args.retrieval_field,
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
        max_target_logs=args.max_target_logs,
        show_progress=args.show_progress,
    )
    pipeline = OnlineParserPipeline(config)
    stats = pipeline.run()
    print(f"dataset={config.dataset}")
    print(f"output_dir={config.output_dir_path}")
    print(f"total_logs={stats.total_logs}")
    print(f"cache_hits={stats.cache_hits}")
    print(f"signature_cache_hits={stats.signature_cache_hits}")
    print(f"llm_calls={stats.llm_calls}")
    print(f"signature_cache_size={config.signature_cache_size}")
    print(f"pattern_cache_version={config.pattern_cache_version}")
    print(f"failed_count={stats.failed_count}")
    print(f"avg_latency_ms={stats.avg_latency_ms:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
