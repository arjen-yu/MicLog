#!/usr/bin/env python3

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.setrecursionlimit(3000)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.utils.evaluator_main import evaluator, prepare_results
from evaluation.utils.postprocess import post_average
from miclog2.eval_parser import LogParser, export_predictions_to_structured, load_parse_time_seconds
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
    datasets = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not datasets:
        raise SystemExit("No datasets specified.")
    return datasets


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MicLog2.0 with TA-Eval-Rep metrics.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS), help="comma-separated dataset names")
    parser.add_argument("--model-path", default=None, help="absolute or relative path to the base/merged model directory")
    parser.add_argument("--adapter-dir", default=None, help="optional LoRA adapter directory")
    parser.add_argument(
        "--parsed-root",
        default=None,
        help="existing online parsing result root; if set, evaluate existing predictions.csv files without rerunning parsing",
    )
    parser.add_argument(
        "--structured-root",
        default=None,
        help=(
            "existing structured CSV root for external parsers. Files may be either "
            "<root>/<dataset>_full.log_structured.csv or "
            "<root>/<dataset>/<dataset>_full.log_structured.csv"
        ),
    )
    parser.add_argument("--input-root", default="loghub-2.0/full_dataset", help="ground-truth dataset root")
    parser.add_argument("--support-root", default="selected_balanced", help="retrieval support bank root")
    parser.add_argument("--output-root", default="results/evaluation", help="root directory for evaluation outputs")
    parser.add_argument("--run-name", default=None, help="optional run subdirectory name")
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


def default_run_name(
    model_path: str | None,
    adapter_dir: str | None,
    shots: int,
    parsed_root: str | None = None,
    structured_root: str | None = None,
) -> str:
    if adapter_dir:
        base = Path(adapter_dir).name
    elif parsed_root:
        base = f"{Path(parsed_root).name}_eval"
    elif structured_root:
        base = f"{Path(structured_root).name}_eval"
    else:
        base = f"{Path(model_path).name}_base"
    return f"MicLog2_eval_{base}_{shots}shot"


def find_structured_result(structured_root: Path, dataset: str) -> Path:
    candidates = [
        structured_root / f"{dataset}_full.log_structured.csv",
        structured_root / dataset / f"{dataset}_full.log_structured.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing structured result for dataset {dataset}. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def main() -> int:
    args = parse_args()
    datasets = parse_dataset_names(args.datasets)
    if sum(value is not None for value in [args.parsed_root, args.structured_root]) > 1:
        raise SystemExit("Use only one of --parsed-root or --structured-root.")
    if args.parsed_root is None and args.structured_root is None and args.model_path is None:
        raise SystemExit("Either --model-path, --parsed-root, or --structured-root must be provided.")
    input_root = (REPO_ROOT / args.input_root).resolve()
    output_root = (REPO_ROOT / args.output_root).resolve()
    run_name = args.run_name or default_run_name(
        args.model_path,
        args.adapter_dir,
        args.shots,
        args.parsed_root,
        args.structured_root,
    )
    output_dir = output_root / timestamp() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed_root = None if args.parsed_root is None else (REPO_ROOT / args.parsed_root).resolve()
    structured_root = None if args.structured_root is None else (REPO_ROOT / args.structured_root).resolve()

    result_file = prepare_results(str(output_dir))
    for dataset in datasets:
        log_file = f"{dataset}/{dataset}_full.log"
        indir = input_root / dataset
        parsedresult = output_dir / f"{Path(log_file).name}_structured.csv"
        parse_time_override = None
        if parsed_root is not None:
            dataset_predictions = parsed_root / dataset / "predictions.csv"
            if not dataset_predictions.exists():
                raise FileNotFoundError(f"Missing predictions.csv for dataset {dataset}: {dataset_predictions}")
            export_predictions_to_structured(dataset_predictions, parsedresult)
            parse_time_override = load_parse_time_seconds(parsed_root / dataset / "summary.json")
            parser_cls = None
        elif structured_root is not None:
            structured_result = find_structured_result(structured_root, dataset)
            shutil.copyfile(structured_result, parsedresult)
            parser_cls = None
        else:
            parser_cls = None if parsedresult.exists() else LogParser
        evaluator(
            dataset=dataset,
            input_dir=str(input_root),
            output_dir=str(output_dir),
            log_file=log_file,
            LogParser=parser_cls,
            param_dict={
                "indir": str(indir),
                "outdir": str(output_dir),
                "model_path": args.model_path,
                "support_root": str((REPO_ROOT / args.support_root).resolve()),
                "adapter_dir": args.adapter_dir,
                "shots": args.shots,
                "retrieval_field": args.retrieval_field,
                "exact_cache_size": args.exact_cache_size,
                "signature_cache_size": args.signature_cache_size,
                "pattern_cache_size": args.pattern_cache_size,
                "pattern_cache_version": args.pattern_cache_version,
                "enable_retrieval_fallback": not args.disable_retrieval_fallback,
                "exclude_same_content": args.exclude_same_content,
                "max_generation_attempts": args.max_generation_attempts,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "attn_implementation": args.attn_implementation,
                "trust_remote_code": args.trust_remote_code,
                "force_cpu": args.force_cpu,
                "seed": args.seed,
                "show_progress": args.show_progress,
                "max_target_logs": args.max_target_logs,
            },
            result_file=result_file,
            parse_time_override=parse_time_override,
        )

    metric_file = output_dir / result_file
    post_average(metric_file, output_dir / "summary_average.csv")
    print(f"evaluation_output={output_dir}")
    print(f"summary_file={metric_file}")
    print(f"average_file={output_dir / 'summary_average.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
