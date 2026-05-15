#!/usr/bin/env python3

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_VARIANTS = [
    "0-1-shot",
    "0-2-shot",
    "0-3-shot",
    "0-4-shot",
    "0-5-shot",
]
DEFAULT_OUTPUT_TEMPLATE = "outputs/qwen35_{model_label}_{method}_{variant}"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Sequentially train progressive-shot Qwen3.5 runs with train_qwen35_meta.py."
    )
    parser.add_argument("--model", default="2b", help="model alias or absolute model path")
    parser.add_argument("--data-root", default="meta_incontext_data_variants", help="root directory containing variant subfolders")
    parser.add_argument(
        "--output-template",
        default=DEFAULT_OUTPUT_TEMPLATE,
        help="output path template, e.g. outputs/qwen35_{model_label}_{method}_{variant}",
    )
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="comma-separated variant names, e.g. 0-1-shot,0-2-shot,0-3-shot,0-4-shot,0-5-shot",
    )
    parser.add_argument("--method", choices=["lora", "qlora"], default="lora")
    parser.add_argument("--dataset-format", choices=["auto", "instruction", "messages"], default="instruction")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--eval-ratio", type=float, default=0.02)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-strategy", choices=["steps", "epoch"], default="epoch")
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--eval-strategy", choices=["auto", "no", "steps", "epoch"], default="auto")
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--attn-implementation", choices=["auto", "sdpa", "flash_attention_2", "eager"], default="sdpa")
    parser.add_argument("--python", default=sys.executable or "python3", help="python executable used to launch train_qwen35_meta.py")
    parser.add_argument("--train-script", default="train_qwen35_meta.py", help="path to the underlying training script")
    parser.add_argument("--cuda-visible-devices", default=None, help="if set, export CUDA_VISIBLE_DEVICES for all runs")
    parser.add_argument("--allow-user-site", action="store_true", help="do not force PYTHONNOUSERSITE=1")
    parser.add_argument("--skip-existing", action="store_true", help="skip a variant if adapter_model.safetensors already exists")
    parser.add_argument("--continue-on-error", action="store_true", help="keep running later variants even if one run fails")
    parser.add_argument("--dry-run", action="store_true", help="print commands without executing them")
    args, extra_args = parser.parse_known_args()
    return args, extra_args


def split_variants(raw_value: str) -> list[str]:
    variants = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not variants:
        raise SystemExit("No variants specified.")
    return variants


def project_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def sanitize_label(value: str) -> str:
    name = Path(value).name if any(sep in value for sep in ("/", "\\")) else value
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return cleaned or "model"


def build_output_dir(args: argparse.Namespace, project_root: Path, variant: str) -> Path:
    rendered = args.output_template.format(
        model=args.model,
        model_label=sanitize_label(args.model),
        method=args.method,
        variant=variant,
    )
    return project_path(project_root, rendered)


def build_command(args: argparse.Namespace, extra_args: list[str], project_root: Path, variant: str) -> tuple[list[str], Path]:
    train_script = project_path(project_root, args.train_script)
    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")

    data_root = project_path(project_root, args.data_root)
    train_file = data_root / variant / "train.jsonl"
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found for variant {variant}: {train_file}")

    output_dir = build_output_dir(args, project_root, variant)

    cmd = [
        args.python,
        str(train_script),
        "--model",
        args.model,
        "--train-file",
        str(train_file),
        "--output-dir",
        str(output_dir),
        "--method",
        args.method,
        "--dataset-format",
        args.dataset_format,
        "--max-length",
        str(args.max_length),
        "--num-train-epochs",
        str(args.num_train_epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--lr-scheduler-type",
        args.lr_scheduler_type,
        "--per-device-train-batch-size",
        str(args.per_device_train_batch_size),
        "--per-device-eval-batch-size",
        str(args.per_device_eval_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--eval-ratio",
        str(args.eval_ratio),
        "--logging-steps",
        str(args.logging_steps),
        "--save-strategy",
        args.save_strategy,
        "--save-total-limit",
        str(args.save_total_limit),
        "--seed",
        str(args.seed),
        "--report-to",
        args.report_to,
        "--attn-implementation",
        args.attn_implementation,
    ]
    if args.save_strategy == "steps":
        cmd.extend(["--save-steps", str(args.save_steps)])
    if args.eval_strategy != "auto":
        cmd.extend(["--eval-strategy", args.eval_strategy])
        if args.eval_strategy == "steps":
            cmd.extend(["--eval-steps", str(args.eval_steps)])
    if extra_args:
        cmd.extend(extra_args)
    return cmd, output_dir


def quoted_command(env: dict[str, str], cmd: list[str]) -> str:
    env_parts = []
    for key in ["CUDA_VISIBLE_DEVICES", "PYTHONNOUSERSITE"]:
        value = env.get(key)
        if value is not None:
            env_parts.append(f"{key}={shlex.quote(value)}")
    return " ".join(env_parts + [shlex.join(cmd)])


def main() -> int:
    args, extra_args = parse_args()
    project_root = Path(__file__).resolve().parent
    variants = split_variants(args.variants)

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if not args.allow_user_site:
        env["PYTHONNOUSERSITE"] = "1"

    failures: list[tuple[str, int]] = []

    for index, variant in enumerate(variants, start=1):
        cmd, output_dir = build_command(args, extra_args, project_root, variant)
        adapter_path = output_dir / "adapter_model.safetensors"
        header = f"[{index}/{len(variants)}] {variant}"

        if args.skip_existing and adapter_path.exists():
            print(f"{header} skip existing: {adapter_path}", flush=True)
            continue

        print(f"{header} output -> {output_dir}", flush=True)
        print(quoted_command(env, cmd), flush=True)

        if args.dry_run:
            continue

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(cmd, cwd=str(project_root), env=env)
        if completed.returncode != 0:
            failures.append((variant, completed.returncode))
            print(f"{header} failed with exit code {completed.returncode}", flush=True)
            if not args.continue_on_error:
                break
        else:
            print(f"{header} done", flush=True)

    if failures:
        print("Batch training finished with failures:", flush=True)
        for variant, returncode in failures:
            print(f"  - {variant}: exit_code={returncode}", flush=True)
        return 1

    print("Batch training finished successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
