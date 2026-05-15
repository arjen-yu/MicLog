from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .types import LogRecord, ParsePrediction, ParseStats


def _find_structured_csv(dataset_dir: Path) -> Path:
    csv_paths = sorted(dataset_dir.glob("*_structured.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No *_structured.csv found in {dataset_dir}")
    if len(csv_paths) > 1:
        raise RuntimeError(f"Multiple *_structured.csv files found in {dataset_dir}")
    return csv_paths[0]


def _load_records_from_dir(root: Path, dataset: str) -> list[LogRecord]:
    dataset_dir = root / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    csv_path = _find_structured_csv(dataset_dir)
    records: list[LogRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"Empty CSV header: {csv_path}")
        for row in reader:
            line_id = str(row.get("LineId", ""))
            content = str(row.get("Content", ""))
            event_template = row.get("EventTemplate")
            extra = {
                key: value
                for key, value in row.items()
                if key not in {"LineId", "Content", "EventTemplate"}
            }
            records.append(
                LogRecord(
                    dataset=dataset,
                    line_id=line_id,
                    content=content,
                    event_template=event_template,
                    extra=extra,
                )
            )
    return records


def load_target_logs(root: str | Path, dataset: str, max_logs: int | None = None) -> list[LogRecord]:
    records = _load_records_from_dir(Path(root).resolve(), dataset)
    if max_logs is not None:
        return records[:max_logs]
    return records


def load_support_bank(root: str | Path, dataset: str) -> list[LogRecord]:
    return _load_records_from_dir(Path(root).resolve(), dataset)


def prediction_csv_fieldnames() -> list[str]:
    return list(ParsePrediction.__dataclass_fields__.keys())


class PredictionCsvWriter:
    def __init__(self, output_path: str | Path, flush_every: int = 10000) -> None:
        self.output_path = Path(output_path).resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = max(flush_every, 1)
        self._rows_written = 0
        self._handle = self.output_path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=prediction_csv_fieldnames())
        self._writer.writeheader()

    def write(self, prediction: ParsePrediction) -> None:
        self._writer.writerow(prediction.to_row())
        self._rows_written += 1
        if self._rows_written % self.flush_every == 0:
            self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()

    def __enter__(self) -> "PredictionCsvWriter":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.close()


def write_summary_json(output_path: str | Path, stats: ParseStats, extra: dict[str, Any] | None = None) -> None:
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = stats.to_dict()
    if extra:
        data.update(extra)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
