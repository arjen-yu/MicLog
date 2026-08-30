#!/usr/bin/env python3

"""Build a content-disjoint test split from LogHub full datasets."""

import argparse
import csv
from pathlib import Path


DEFAULT_DATASETS = [
    "Apache", "BGL", "Hadoop", "HDFS", "HealthApp", "HPC", "Linux",
    "Mac", "OpenSSH", "OpenStack", "Proxifier", "Spark", "Thunderbird", "Zookeeper",
]
REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_dataset_names(raw_value: str) -> list[str]:
    datasets = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not datasets:
        raise SystemExit("No datasets specified.")
    return datasets


def find_structured_csv(dataset_dir: Path) -> Path:
    paths = sorted(dataset_dir.glob("*_structured.csv"))
    if not paths:
        raise FileNotFoundError(f"No *_structured.csv found in {dataset_dir}")
    if len(paths) > 1:
        raise RuntimeError(f"Multiple *_structured.csv files found in {dataset_dir}")
    return paths[0]


def load_training_contents(path: Path) -> set[str]:
    contents: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "Content" not in (reader.fieldnames or []):
            raise RuntimeError(f"Missing Content column in {path}")
        for row in reader:
            contents.add(row.get("Content", ""))
    return contents


def build_dataset_test_split(full_path: Path, train_path: Path, output_path: Path) -> dict[str, int]:
    train_contents = load_training_contents(train_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_count = removed_count = test_count = 0
    with full_path.open("r", encoding="utf-8", newline="") as source, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames or []
        missing = sorted({"LineId", "Content", "EventTemplate"} - set(fieldnames))
        if missing:
            raise RuntimeError(f"Missing columns in {full_path}: {', '.join(missing)}")
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            full_count += 1
            if row.get("Content", "") in train_contents:
                removed_count += 1
                continue
            writer.writerow(row)
            test_count += 1
    return {
        "full_count": full_count,
        "unique_train_contents": len(train_contents),
        "removed_count": removed_count,
        "test_count": test_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a content-disjoint D_test = D_full \\ D_train split."
    )
    parser.add_argument("--full-root", default="loghub-2.0/full_dataset")
    parser.add_argument("--train-root", default="selected_balanced")
    parser.add_argument("--output-root", default="loghub-2.0/test_dataset")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    full_root = (REPO_ROOT / args.full_root).resolve()
    train_root = (REPO_ROOT / args.train_root).resolve()
    output_root = (REPO_ROOT / args.output_root).resolve()
    rows = []
    for dataset in parse_dataset_names(args.datasets):
        full_path = find_structured_csv(full_root / dataset)
        train_path = find_structured_csv(train_root / dataset)
        output_path = output_root / dataset / full_path.name
        stats = build_dataset_test_split(full_path, train_path, output_path)
        rows.append({"dataset": dataset, **stats})
        print(
            f"[{dataset}] full={stats['full_count']} removed={stats['removed_count']} "
            f"test={stats['test_count']} unique_train_contents={stats['unique_train_contents']}",
            flush=True,
        )

    summary_path = output_root / "split_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "full_count", "unique_train_contents", "removed_count", "test_count"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote test split under {output_root}", flush=True)
    print(f"Summary written to {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
