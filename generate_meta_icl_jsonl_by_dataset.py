#!/usr/bin/env python3

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

from generate_meta_icl_jsonl import DEFAULT_MAX_SHOTS, OUTPUT_ROOT, SELECTED_ROOT, generate_all, selected_dataset_dirs


DEFAULT_OUTPUT_PARENT = "meta_incontext_data_variants_by_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-dataset meta-ICL JSONL roots for single-dataset ablation training."
    )
    parser.add_argument("--input-root", default=str(SELECTED_ROOT), help="selected_balanced root directory")
    parser.add_argument(
        "--output-root-parent",
        default=DEFAULT_OUTPUT_PARENT,
        help="parent directory; each dataset will be written to <parent>/<dataset>/...",
    )
    parser.add_argument("--max-shots", type=int, default=DEFAULT_MAX_SHOTS)
    parser.add_argument("--retrieval-field", choices=["content", "normalized_content"], default="content")
    parser.add_argument("--output-format", choices=["instruction", "chat"], default="instruction")
    parser.add_argument("--query-mode", choices=["full", "lite"], default="full")
    parser.add_argument("--write-metadata", action="store_true")
    parser.add_argument(
        "--datasets",
        default=None,
        help="comma-separated dataset names; default is all datasets under selected_balanced",
    )
    return parser.parse_args()


def parse_dataset_names(raw_value: str | None) -> list[str] | None:
    if raw_value is None:
        return None
    names = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not names:
        raise SystemExit("No datasets specified.")
    return names


def build_dataset_args(args: argparse.Namespace, dataset_name: str, dataset_output_root: Path) -> Namespace:
    return Namespace(
        input_root=args.input_root,
        output_root=str(dataset_output_root),
        max_shots=args.max_shots,
        retrieval_field=args.retrieval_field,
        output_format=args.output_format,
        query_mode=args.query_mode,
        write_metadata=args.write_metadata,
        datasets=[dataset_name],
    )


def main() -> int:
    args = parse_args()
    dataset_names = parse_dataset_names(args.datasets)
    input_root = Path(args.input_root).resolve()
    output_root_parent = Path(args.output_root_parent).resolve()
    output_root_parent.mkdir(parents=True, exist_ok=True)

    dataset_dirs = selected_dataset_dirs(input_root, dataset_names)
    for index, dataset_dir in enumerate(dataset_dirs, start=1):
        dataset_name = dataset_dir.name
        dataset_output_root = output_root_parent / dataset_name
        print(
            f"[{index}/{len(dataset_dirs)}] generate {dataset_name} -> {dataset_output_root}",
            flush=True,
        )
        generate_all(build_dataset_args(args, dataset_name, dataset_output_root))

    print(f"Per-dataset meta-ICL roots written under {output_root_parent}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
