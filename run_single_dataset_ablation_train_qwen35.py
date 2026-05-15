#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_DATA_ROOT_PARENT = "meta_incontext_data_variants_by_dataset"
DEFAULT_OUTPUT_ROOT_PARENT = "outputs_single_dataset_ablation"
DEFAULT_VARIANTS = "0-5-shot"
DEFAULT_OUTPUT_TEMPLATE = "{output_root_parent}/{dataset}/qwen35_{model_label}_{method}_{variant}"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Batch-train one meta-ICL LoRA model per dataset for single-dataset ablation."
    )
    parser.add_argument("--model", default="4b", help="model alias or absolute model path")
    parser.add_argument("--datasets", default=None, help="comma-separated dataset names; default is all dataset dirs")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS, help="comma-separated variant names")
    parser.add_argument("--method", choices=["lora", "qlora"], default="lora")
    parser.add_argument("--data-root-parent", default=DEFAULT_DATA_ROOT_PARENT)
    parser.add_argument("--output-root-parent", default=DEFAULT_OUTPUT_ROOT_PARENT)
    parser.add_argument("--output-template", default=DEFAULT_OUTPUT_TEMPLATE)
    parser.add_argument("--python", default=sys.executable or "python3")
    parser.add_argument("--progressive-script", default="run_progressive_train_qwen35.py")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--allow-user-site", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, extra_args = parser.parse_known_args()
    return args, extra_args


def parse_dataset_names(raw_value: str | None, data_root_parent: Path) -> list[str]:
    if raw_value is None:
        dataset_dirs = sorted(path.name for path in data_root_parent.iterdir() if path.is_dir())
        if not dataset_dirs:
            raise SystemExit(f"No dataset dirs found under {data_root_parent}")
        return dataset_dirs
    names = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not names:
        raise SystemExit("No datasets specified.")
    return names


def parse_variant_names(raw_value: str) -> list[str]:
    names = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not names:
        raise SystemExit("No variants specified.")
    return names


def sanitize_label(value: str) -> str:
    name = Path(value).name if any(sep in value for sep in ("/", "\\")) else value
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name.strip())
    return cleaned or "model"


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
    data_root_parent = (project_root / args.data_root_parent).resolve()
    output_root_parent = (project_root / args.output_root_parent).resolve()
    progressive_script = (project_root / args.progressive_script).resolve()
    if not progressive_script.exists():
        raise FileNotFoundError(f"Progressive training script not found: {progressive_script}")
    if not data_root_parent.exists():
        raise FileNotFoundError(f"Data root parent not found: {data_root_parent}")

    datasets = parse_dataset_names(args.datasets, data_root_parent)
    variants = parse_variant_names(args.variants)

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if not args.allow_user_site:
        env["PYTHONNOUSERSITE"] = "1"

    failures: list[tuple[str, int]] = []
    model_label = sanitize_label(args.model)

    for index, dataset in enumerate(datasets, start=1):
        data_root = data_root_parent / dataset
        if not data_root.exists():
            raise FileNotFoundError(f"Per-dataset data root not found: {data_root}")
        output_template = args.output_template.format(
            output_root_parent=str(output_root_parent),
            dataset=dataset,
            model=args.model,
            model_label=model_label,
            method=args.method,
            variant="{variant}",
        )
        variant_output_dirs = [Path(output_template.replace("{variant}", variant)).resolve() for variant in variants]
        adapter_paths = [path / "adapter_model.safetensors" for path in variant_output_dirs]
        header = f"[{index}/{len(datasets)}] {dataset}"

        if args.skip_existing and adapter_paths and all(path.exists() for path in adapter_paths):
            print(f"{header} skip existing: {variant_output_dirs[0].parent}", flush=True)
            continue

        cmd = [
            args.python,
            str(progressive_script),
            "--model",
            args.model,
            "--data-root",
            str(data_root),
            "--output-template",
            output_template,
            "--variants",
            args.variants,
            "--method",
            args.method,
        ]
        if extra_args:
            cmd.extend(extra_args)

        print(f"{header} data_root -> {data_root}", flush=True)
        print(f"{header} output_template -> {output_template}", flush=True)
        print(quoted_command(env, cmd), flush=True)

        if args.dry_run:
            continue

        for variant_output_dir in variant_output_dirs:
            variant_output_dir.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(cmd, cwd=str(project_root), env=env)
        if completed.returncode != 0:
            failures.append((dataset, completed.returncode))
            print(f"{header} failed with exit code {completed.returncode}", flush=True)
            if not args.continue_on_error:
                break
        else:
            print(f"{header} done", flush=True)

    if failures:
        print("Single-dataset ablation training finished with failures:", flush=True)
        for dataset, returncode in failures:
            print(f"  - {dataset}: exit_code={returncode}", flush=True)
        return 1

    print("Single-dataset ablation training finished successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
