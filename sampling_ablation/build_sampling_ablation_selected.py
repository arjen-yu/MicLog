#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dedup_content_logs import (
    CLUSTERED_ROOT,
    ROOT,
    SELECTED_BALANCED_ROOT,
    balanced_keep_k,
    find_single_file,
    line_id_value,
    select_frequency_representative,
    set_csv_field_size_limit,
)


DEFAULT_OUTPUT_ROOT = ROOT / "selected_sampling_ablation"
EXTRA_FIELDNAMES = ["cluster_keep_k", "selected_rank", "selection_reason"]
STRATEGY_CHOICES = {
    "random_global_matched",
    "random_cluster_matched",
    "representative_only",
}


@dataclass(frozen=True)
class StrategyRun:
    strategy: str
    seed: int | None

    @property
    def run_name(self) -> str:
        if self.seed is None:
            return self.strategy
        return f"{self.strategy}_seed{self.seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sampling-ablation support roots with the same CSV schema as selected_balanced."
    )
    parser.add_argument("--input-root", default=str(CLUSTERED_ROOT), help="clustered root directory")
    parser.add_argument(
        "--baseline-root",
        default=str(SELECTED_BALANCED_ROOT),
        help="selected_balanced root used to match sample counts for random_global_matched",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="output root directory")
    parser.add_argument(
        "--strategies",
        default="random_global_matched,random_cluster_matched,representative_only",
        help="comma-separated strategies to generate",
    )
    parser.add_argument(
        "--random-global-seeds",
        default="42",
        help="comma-separated seeds for random_global_matched",
    )
    parser.add_argument(
        "--random-cluster-seeds",
        default="42,43,44",
        help="comma-separated seeds for random_cluster_matched",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="comma-separated dataset names; default is all dataset dirs under input-root",
    )
    return parser.parse_args()


def parse_name_list(raw_value: str | None) -> list[str] | None:
    if raw_value is None:
        return None
    names = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not names:
        raise SystemExit("No dataset names specified.")
    return names


def parse_int_list(raw_value: str) -> list[int]:
    values = []
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise SystemExit("Expected at least one integer seed.")
    return values


def parse_strategy_names(raw_value: str) -> list[str]:
    names = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not names:
        raise SystemExit("No strategies specified.")
    invalid = sorted(set(names) - STRATEGY_CHOICES)
    if invalid:
        raise SystemExit(f"Unsupported strategy name(s): {', '.join(invalid)}")
    return names


def dataset_dirs(input_root: Path, dataset_names: list[str] | None) -> list[Path]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
    if dataset_names is None:
        if not dirs:
            raise SystemExit(f"No dataset directories found under {input_root}")
        return dirs

    requested = set(dataset_names)
    dirs = [path for path in dirs if path.name in requested]
    missing = sorted(requested - {path.name for path in dirs})
    if missing:
        raise SystemExit(f"Requested dataset(s) not found under {input_root}: {', '.join(missing)}")
    return dirs


def load_csv_rows(dataset_dir: Path) -> tuple[Path, list[str], list[dict[str, str]]]:
    csv_path = find_single_file(dataset_dir, "_structured.csv")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    required = {
        "LineId",
        "Content",
        "dup_count",
        "normalized_unique_content_count",
        "cluster_id",
        "cluster_size_unique",
        "cluster_weight_dup_sum",
    }
    missing = sorted(required - set(fieldnames))
    if missing:
        raise RuntimeError(f"Missing required columns in {csv_path}: {', '.join(missing)}")
    return csv_path, fieldnames, rows


def baseline_selected_count(baseline_root: Path, dataset_name: str) -> int:
    csv_path = find_single_file(baseline_root / dataset_name, "_structured.csv")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _row in csv.DictReader(handle))


def group_rows_by_cluster(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["cluster_id"]].append(row)
    return dict(grouped)


def keep_count_for_cluster(rows: list[dict[str, str]]) -> int:
    return balanced_keep_k(rows[0], len(rows))


def random_global_matched(
    rows: list[dict[str, str]],
    target_count: int,
    seed: int,
) -> dict[str, list[tuple[dict[str, str], str]]]:
    if target_count > len(rows):
        raise RuntimeError(
            f"Target count {target_count} exceeds available candidate rows {len(rows)} for random_global_matched."
        )
    rng = random.Random(seed)
    chosen_rows = rng.sample(rows, target_count)
    grouped: dict[str, list[tuple[dict[str, str], str]]] = defaultdict(list)
    for row in chosen_rows:
        grouped[row["cluster_id"]].append((row, "random_global_sample"))
    return dict(grouped)


def random_cluster_matched(
    clusters: dict[str, list[dict[str, str]]],
    seed: int,
) -> dict[str, list[tuple[dict[str, str], str]]]:
    rng = random.Random(seed)
    grouped: dict[str, list[tuple[dict[str, str], str]]] = {}
    for cluster_id, rows in clusters.items():
        keep_k = keep_count_for_cluster(rows)
        grouped[cluster_id] = [(row, "random_cluster_sample") for row in rng.sample(rows, keep_k)]
    return grouped


def representative_only(
    clusters: dict[str, list[dict[str, str]]],
) -> dict[str, list[tuple[dict[str, str], str]]]:
    grouped: dict[str, list[tuple[dict[str, str], str]]] = {}
    for cluster_id, rows in clusters.items():
        grouped[cluster_id] = [(select_frequency_representative(rows), "frequency_representative")]
    return grouped


def sort_selected_rows(selected_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        selected_rows,
        key=lambda row: (
            row["cluster_id"],
            int(row["selected_rank"]),
            line_id_value(row),
        ),
    )


def materialize_selected_rows(
    grouped_selected: dict[str, list[tuple[dict[str, str], str]]],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    selected_rows: list[dict[str, str]] = []
    selected_dup_weight_sum = 0
    selected_unique_content_sum = 0
    clusters_with_1_sample = 0
    clusters_with_2_sample = 0
    clusters_with_3plus_sample = 0
    max_cluster_selected_count = 0

    for cluster_id, chosen_rows in grouped_selected.items():
        keep_k = len(chosen_rows)
        max_cluster_selected_count = max(max_cluster_selected_count, keep_k)
        if keep_k == 1:
            clusters_with_1_sample += 1
        elif keep_k == 2:
            clusters_with_2_sample += 1
        elif keep_k >= 3:
            clusters_with_3plus_sample += 1

        for rank, (row, reason) in enumerate(chosen_rows, start=1):
            output_row = dict(row)
            output_row["cluster_keep_k"] = str(keep_k)
            output_row["selected_rank"] = str(rank)
            output_row["selection_reason"] = reason
            selected_rows.append(output_row)
            selected_dup_weight_sum += int(row["dup_count"])
            selected_unique_content_sum += int(row.get("normalized_unique_content_count", "1"))

    stats = {
        "selected_log_count": len(selected_rows),
        "covered_cluster_count": len(grouped_selected),
        "selected_dup_weight_sum": selected_dup_weight_sum,
        "selected_unique_content_sum": selected_unique_content_sum,
        "clusters_with_1_sample": clusters_with_1_sample,
        "clusters_with_2_sample": clusters_with_2_sample,
        "clusters_with_3plus_sample": clusters_with_3plus_sample,
        "max_cluster_selected_count": max_cluster_selected_count,
    }
    return sort_selected_rows(selected_rows), stats


def write_selected_file(
    output_path: Path,
    fieldnames: list[str],
    selected_rows: list[dict[str, str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fieldnames, *EXTRA_FIELDNAMES])
        writer.writeheader()
        writer.writerows(selected_rows)


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_name",
        "strategy",
        "seed",
        "baseline_selected_log_count",
        "candidate_row_count",
        "total_cluster_count",
        "covered_cluster_count",
        "cluster_coverage_ratio",
        "clusters_with_0_sample",
        "clusters_with_1_sample",
        "clusters_with_2_sample",
        "clusters_with_3plus_sample",
        "max_cluster_selected_count",
        "selected_log_count",
        "selected_dup_weight_sum",
        "selected_unique_content_sum",
        "run_name",
        "output_root",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_runs(args: argparse.Namespace) -> list[StrategyRun]:
    strategy_names = parse_strategy_names(args.strategies)
    runs: list[StrategyRun] = []
    if "random_global_matched" in strategy_names:
        runs.extend(StrategyRun("random_global_matched", seed) for seed in parse_int_list(args.random_global_seeds))
    if "random_cluster_matched" in strategy_names:
        runs.extend(StrategyRun("random_cluster_matched", seed) for seed in parse_int_list(args.random_cluster_seeds))
    if "representative_only" in strategy_names:
        runs.append(StrategyRun("representative_only", None))
    return runs


def build_grouped_selection(
    run: StrategyRun,
    rows: list[dict[str, str]],
    clusters: dict[str, list[dict[str, str]]],
    target_count: int,
) -> dict[str, list[tuple[dict[str, str], str]]]:
    if run.strategy == "random_global_matched":
        assert run.seed is not None
        return random_global_matched(rows, target_count, run.seed)
    if run.strategy == "random_cluster_matched":
        assert run.seed is not None
        return random_cluster_matched(clusters, run.seed)
    if run.strategy == "representative_only":
        return representative_only(clusters)
    raise RuntimeError(f"Unsupported strategy: {run.strategy}")


def main() -> int:
    set_csv_field_size_limit()
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    baseline_root = Path(args.baseline_root).resolve()
    output_root = Path(args.output_root).resolve()
    runs = build_runs(args)
    datasets = dataset_dirs(input_root, parse_name_list(args.datasets))

    manifest_rows: list[dict[str, object]] = []
    summary_by_run: dict[str, list[dict[str, object]]] = defaultdict(list)

    for dataset_index, dataset_dir in enumerate(datasets, start=1):
        csv_path, fieldnames, rows = load_csv_rows(dataset_dir)
        clusters = group_rows_by_cluster(rows)
        target_count = baseline_selected_count(baseline_root, dataset_dir.name)
        total_cluster_count = len(clusters)

        print(
            f"[{dataset_index}/{len(datasets)}] {dataset_dir.name} candidates={len(rows)} "
            f"clusters={total_cluster_count} baseline_selected={target_count}",
            flush=True,
        )

        for run in runs:
            grouped_selected = build_grouped_selection(run, rows, clusters, target_count)
            selected_rows, stats = materialize_selected_rows(grouped_selected)
            run_root = output_root / run.run_name
            output_path = run_root / dataset_dir.name / csv_path.name
            write_selected_file(output_path, fieldnames, selected_rows)

            covered_cluster_count = stats["covered_cluster_count"]
            clusters_with_0_sample = total_cluster_count - covered_cluster_count
            summary_row: dict[str, object] = {
                "dataset_name": dataset_dir.name,
                "strategy": run.strategy,
                "seed": "" if run.seed is None else run.seed,
                "baseline_selected_log_count": target_count,
                "candidate_row_count": len(rows),
                "total_cluster_count": total_cluster_count,
                "covered_cluster_count": covered_cluster_count,
                "cluster_coverage_ratio": f"{covered_cluster_count / total_cluster_count:.6f}" if total_cluster_count else "0.000000",
                "clusters_with_0_sample": clusters_with_0_sample,
                "clusters_with_1_sample": stats["clusters_with_1_sample"],
                "clusters_with_2_sample": stats["clusters_with_2_sample"],
                "clusters_with_3plus_sample": stats["clusters_with_3plus_sample"],
                "max_cluster_selected_count": stats["max_cluster_selected_count"],
                "selected_log_count": stats["selected_log_count"],
                "selected_dup_weight_sum": stats["selected_dup_weight_sum"],
                "selected_unique_content_sum": stats["selected_unique_content_sum"],
                "run_name": run.run_name,
                "output_root": str(run_root),
            }
            summary_by_run[run.run_name].append(summary_row)
            manifest_rows.append(summary_row)

            print(
                f"  - {run.run_name}: selected={stats['selected_log_count']} "
                f"covered_clusters={covered_cluster_count}/{total_cluster_count} "
                f"output={output_path}",
                flush=True,
            )

    output_root.mkdir(parents=True, exist_ok=True)
    for run_name, rows in summary_by_run.items():
        run_root = output_root / run_name
        write_summary_csv(run_root / "summary.csv", rows)
        (run_root / "selection_config.json").write_text(
            json.dumps(
                {
                    "run_name": run_name,
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    write_summary_csv(output_root / "manifest.csv", manifest_rows)
    print(f"Sampling-ablation roots written to {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
