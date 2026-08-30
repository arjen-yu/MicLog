#!/usr/bin/env python3

"""Run a command while sampling host and NVIDIA GPU resource usage."""

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

import psutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="JSON path for the resource summary")
    parser.add_argument("--interval", type=float, default=1.0, help="sampling interval in seconds")
    parser.add_argument("--gpu-index", default=None, help="physical GPU index passed to nvidia-smi")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command to run after --")
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def gpu_samples(gpu_index: str | None) -> list[dict[str, float]]:
    query = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    if gpu_index is not None:
        query.extend(["--id", gpu_index])
    try:
        result = subprocess.run(query, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    samples = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        try:
            samples.append(
                {
                    "gpu_index": float(fields[0]),
                    "memory_used_mib": float(fields[1]),
                    "utilization_gpu_percent": float(fields[2]),
                    "power_draw_watts": float(fields[3]),
                }
            )
        except ValueError:
            continue
    return samples


def process_rss_bytes(root: psutil.Process) -> int:
    processes = [root]
    try:
        processes.extend(root.children(recursive=True))
    except psutil.Error:
        pass
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except psutil.Error:
            continue
    return total


def main() -> int:
    args = parse_args()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = args.command
    print(f"profiling command: {shlex.join(command)}", flush=True)

    started = time.perf_counter()
    child = subprocess.Popen(command, env=os.environ.copy())
    root = psutil.Process(child.pid)
    peak_rss = 0
    gpu_records: list[dict[str, float]] = []
    try:
        while child.poll() is None:
            peak_rss = max(peak_rss, process_rss_bytes(root))
            gpu_records.extend(gpu_samples(args.gpu_index))
            time.sleep(max(args.interval, 0.1))
    finally:
        peak_rss = max(peak_rss, process_rss_bytes(root))
        gpu_records.extend(gpu_samples(args.gpu_index))

    elapsed_seconds = time.perf_counter() - started
    by_gpu: dict[int, list[dict[str, float]]] = {}
    for record in gpu_records:
        by_gpu.setdefault(int(record["gpu_index"]), []).append(record)

    gpu_summary = []
    for index, records in sorted(by_gpu.items()):
        gpu_summary.append(
            {
                "gpu_index": index,
                "samples": len(records),
                "peak_memory_used_mib": max(record["memory_used_mib"] for record in records),
                "average_utilization_gpu_percent": sum(record["utilization_gpu_percent"] for record in records) / len(records),
                "peak_utilization_gpu_percent": max(record["utilization_gpu_percent"] for record in records),
                "average_power_draw_watts": sum(record["power_draw_watts"] for record in records) / len(records),
                "peak_power_draw_watts": max(record["power_draw_watts"] for record in records),
                "estimated_energy_Wh": sum(record["power_draw_watts"] for record in records) * args.interval / 3600.0,
            }
        )

    summary = {
        "command": command,
        "return_code": child.returncode,
        "elapsed_seconds": elapsed_seconds,
        "sampling_interval_seconds": args.interval,
        "peak_process_tree_rss_bytes": peak_rss,
        "peak_process_tree_rss_gib": peak_rss / (1024**3),
        "gpu": gpu_summary,
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
