#!/usr/bin/env python3

import argparse
import json
import site
import sys
from pathlib import Path
from typing import Any


_USER_SITE = site.getusersitepackages()
if _USER_SITE and _USER_SITE in sys.path:
    sys.path = [path for path in sys.path if path != _USER_SITE]


MODEL_ALIASES = {
    "0.8b": "/tempdisk2/yjb/Models/Qwen3.5-0.8B",
    "2b": "/tempdisk2/yjb/Models/Qwen3.5-2B",
    "4b": "/tempdisk2/yjb/Models/Qwen3.5-4B",
    "9b": "/tempdisk2/yjb/Models/Qwen3.5-9B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a trained LoRA adapter into a standalone Qwen3.5 model directory."
    )
    parser.add_argument("--adapter-dir", required=True, help="directory containing adapter_model.safetensors")
    parser.add_argument(
        "--model",
        default=None,
        help="model alias (0.8b/2b/4b/9b) or absolute model path; if omitted, infer from adapter metadata",
    )
    parser.add_argument("--output-dir", required=True, help="merged model output directory")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto", help="device_map passed to transformers, e.g. auto or cpu")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    return parser.parse_args()


def ensure_dependencies() -> dict[str, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise SystemExit(
            "Missing merge dependencies. Install them first, for example:\n"
            "  PYTHONNOUSERSITE=1 pip install -U -r requirements_qwen35.txt\n"
            f"Original import error: {type(exc).__name__}: {exc}"
        ) from exc

    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }


def resolve_model_path(model_arg: str | None, adapter_dir: Path) -> str:
    if model_arg:
        alias = model_arg.strip().lower()
        resolved = MODEL_ALIASES.get(alias, model_arg)
        path = Path(resolved)
        if not path.exists():
            raise FileNotFoundError(f"Model path does not exist: {resolved}")
        return str(path.resolve())

    run_config_path = adapter_dir / "run_config.json"
    if run_config_path.exists():
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        resolved = run_config.get("resolved_model_path")
        if resolved and Path(resolved).exists():
            return str(Path(resolved).resolve())

    adapter_config_path = adapter_dir / "adapter_config.json"
    if adapter_config_path.exists():
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        resolved = adapter_config.get("base_model_name_or_path")
        if resolved and Path(resolved).exists():
            return str(Path(resolved).resolve())

    raise SystemExit(
        "Could not infer the base model path from the adapter directory. "
        "Please pass --model explicitly."
    )


def resolve_dtype(torch_module: Any, dtype_name: str):
    if dtype_name == "float16":
        return torch_module.float16
    if dtype_name == "bfloat16":
        return torch_module.bfloat16
    if dtype_name == "float32":
        return torch_module.float32
    if torch_module.cuda.is_available() and torch_module.cuda.is_bf16_supported():
        return torch_module.bfloat16
    if torch_module.cuda.is_available():
        return torch_module.float16
    return torch_module.float32


def main() -> int:
    args = parse_args()
    mods = ensure_dependencies()
    torch_module = mods["torch"]
    PeftModel = mods["PeftModel"]
    AutoModelForCausalLM = mods["AutoModelForCausalLM"]
    AutoTokenizer = mods["AutoTokenizer"]

    adapter_dir = Path(args.adapter_dir).resolve()
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_model_path(args.model, adapter_dir)
    dtype = resolve_dtype(torch_module, args.torch_dtype)

    model_kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": model_path,
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": dtype,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map

    base_model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    tokenizer.save_pretrained(output_dir)

    merge_meta = {
        "base_model_path": model_path,
        "adapter_dir": str(adapter_dir),
        "merged_output_dir": str(output_dir),
        "torch_dtype": str(dtype),
        "device_map": args.device_map,
    }
    (output_dir / "merge_info.json").write_text(
        json.dumps(merge_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(merge_meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
