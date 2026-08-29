from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from miclog2.online import OnlineParserConfig, OnlineParserPipeline
from miclog2.online.cache import PATTERN_CACHE_VERSION_CHOICES


def _template_to_event_id(template: str) -> str:
    return hashlib.md5(template.encode("utf-8")).hexdigest()[:8]


def export_predictions_to_structured(predictions_csv: str | Path, output_csv: str | Path) -> None:
    predictions_csv = Path(predictions_csv).resolve()
    output_csv = Path(output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with predictions_csv.open("r", encoding="utf-8", newline="") as src, output_csv.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        if not reader.fieldnames:
            raise RuntimeError(f"Empty CSV header: {predictions_csv}")
        required_columns = {"line_id", "content", "predicted_template"}
        missing = sorted(required_columns - set(reader.fieldnames))
        if missing:
            raise RuntimeError(f"Missing required columns in {predictions_csv}: {missing}")

        writer = csv.DictWriter(dst, fieldnames=["LineId", "Content", "EventId", "EventTemplate"])
        writer.writeheader()
        for row in reader:
            template = str(row.get("predicted_template", ""))
            writer.writerow(
                {
                    "LineId": row.get("line_id", ""),
                    "Content": row.get("content", ""),
                    "EventId": "" if not template else _template_to_event_id(template),
                    "EventTemplate": template,
                }
            )


def load_parse_time_seconds(summary_json: str | Path) -> float | None:
    summary_json = Path(summary_json).resolve()
    if not summary_json.exists():
        return None
    data = json.loads(summary_json.read_text(encoding="utf-8"))
    total_latency_ms = data.get("total_latency_ms")
    if total_latency_ms is None:
        return None
    return float(total_latency_ms) / 1000.0


class LogParser:
    def __init__(
        self,
        *,
        indir: str,
        outdir: str,
        model_path: str,
        support_root: str,
        adapter_dir: str | None = None,
        shots: int = 5,
        retrieval_field: str = "Content",
        exact_cache_size: int = 50000,
        signature_cache_size: int = 10000,
        pattern_cache_size: int = 10000,
        pattern_cache_version: str = "v2",
        exclude_same_content: bool = True,
        max_generation_attempts: int = 1,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        attn_implementation: str = "sdpa",
        trust_remote_code: bool = True,
        force_cpu: bool = False,
        seed: int = 42,
        show_progress: bool = True,
        max_target_logs: int | None = None,
    ) -> None:
        self.indir = Path(indir).resolve()
        self.outdir = Path(outdir).resolve()
        self.model_path = model_path
        self.support_root = support_root
        self.adapter_dir = adapter_dir
        self.shots = shots
        self.retrieval_field = retrieval_field
        self.exact_cache_size = exact_cache_size
        self.signature_cache_size = signature_cache_size
        self.pattern_cache_size = pattern_cache_size
        if pattern_cache_version not in PATTERN_CACHE_VERSION_CHOICES:
            raise ValueError(
                f"Unsupported pattern cache version: {pattern_cache_version}. "
                f"Expected one of {PATTERN_CACHE_VERSION_CHOICES}."
            )
        self.pattern_cache_version = pattern_cache_version
        self.exclude_same_content = exclude_same_content
        self.max_generation_attempts = max_generation_attempts
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.attn_implementation = attn_implementation
        self.trust_remote_code = trust_remote_code
        self.force_cpu = force_cpu
        self.seed = seed
        self.show_progress = show_progress
        self.max_target_logs = max_target_logs

    def parse(self, log_file_basename: str) -> None:
        dataset = self.indir.name
        target_root = str(self.indir.parent)
        runtime_dir = self.outdir / "_online_runtime" / dataset
        config = OnlineParserConfig(
            dataset=dataset,
            model_path=self.model_path,
            adapter_dir=self.adapter_dir,
            target_root=target_root,
            support_root=self.support_root,
            output_dir=str(runtime_dir),
            shots=self.shots,
            retrieval_field=self.retrieval_field,
            exact_cache_size=self.exact_cache_size,
            signature_cache_size=self.signature_cache_size,
            pattern_cache_size=self.pattern_cache_size,
            pattern_cache_version=self.pattern_cache_version,
            exclude_same_content=self.exclude_same_content,
            max_generation_attempts=self.max_generation_attempts,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            attn_implementation=self.attn_implementation,
            trust_remote_code=self.trust_remote_code,
            force_cpu=self.force_cpu,
            seed=self.seed,
            max_target_logs=self.max_target_logs,
            show_progress=self.show_progress,
        )
        pipeline = OnlineParserPipeline(config)
        _stats = pipeline.run()
        parsed_path = self.outdir / f"{log_file_basename}_structured.csv"
        export_predictions_to_structured(runtime_dir / "predictions.csv", parsed_path)
