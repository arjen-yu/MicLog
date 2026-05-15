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
DEFAULT_SYSTEM_PROMPT = "You are a log parsing assistant that extracts templates from logs."
DEFAULT_INSTRUCTION = (
    "For each query log after the final <content> tag, try your best to extract one log template "
    "(substitute variable tokens in the log as <*> and remain constant tokens to construct the template) "
    "and put the template after the final <template> tag and between <START> and <END> tags. "
    "Use previous <content>/<template> pairs as in-context examples when they are provided."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a local Qwen3.5 base model plus a trained LoRA adapter."
    )
    parser.add_argument("--adapter-dir", required=True, help="directory containing adapter_model.safetensors")
    parser.add_argument(
        "--model",
        default=None,
        help="model alias (0.8b/2b/4b/9b) or absolute model path; if omitted, infer from adapter metadata",
    )
    parser.add_argument("--content", default=None, help="one raw log content string")
    parser.add_argument("--content-file", default=None, help="text file containing one raw log content")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--disable-system-prompt", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn-implementation", choices=["auto", "sdpa", "flash_attention_2", "eager"], default="sdpa")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--cpu", action="store_true", help="force CPU inference")
    parser.add_argument("--print-prompt", action="store_true", help="print the exact prompt before generation")
    return parser.parse_args()


def ensure_dependencies() -> dict[str, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    except Exception as exc:
        raise SystemExit(
            "Missing inference dependencies. Install them first, for example:\n"
            "  PYTHONNOUSERSITE=1 pip install -U -r requirements_qwen35.txt\n"
            f"Original import error: {type(exc).__name__}: {exc}"
        ) from exc

    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "set_seed": set_seed,
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


def read_content(args: argparse.Namespace) -> str:
    if bool(args.content) == bool(args.content_file):
        raise SystemExit("Use exactly one of --content or --content-file.")
    if args.content is not None:
        text = args.content.strip()
    else:
        text = Path(args.content_file).read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit("Input content is empty.")
    return text


def choose_dtype(torch_module: Any, use_cpu: bool):
    if use_cpu:
        return torch_module.float32
    if torch_module.cuda.is_available() and torch_module.cuda.is_bf16_supported():
        return torch_module.bfloat16
    if torch_module.cuda.is_available():
        return torch_module.float16
    return torch_module.float32


def build_user_text(instruction: str, content: str) -> str:
    return f"{instruction}\n\n<content>{content}\n<template>"


def extract_template(text: str) -> str:
    start_tag = "<START>"
    end_tag = "<END>"
    start = text.find(start_tag)
    end = text.find(end_tag, start + len(start_tag)) if start != -1 else -1
    if start != -1 and end != -1:
        return text[start + len(start_tag) : end].strip()
    return text.strip()


def main() -> int:
    args = parse_args()
    mods = ensure_dependencies()
    torch_module = mods["torch"]
    PeftModel = mods["PeftModel"]
    AutoModelForCausalLM = mods["AutoModelForCausalLM"]
    AutoTokenizer = mods["AutoTokenizer"]
    set_seed = mods["set_seed"]

    adapter_dir = Path(args.adapter_dir).resolve()
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    content = read_content(args)
    model_path = resolve_model_path(args.model, adapter_dir)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = choose_dtype(torch_module, args.cpu)
    attn_impl = None if args.attn_implementation == "auto" else args.attn_implementation

    model_kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": model_path,
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": dtype,
    }
    if attn_impl is not None:
        model_kwargs["attn_implementation"] = attn_impl
    if not args.cpu and torch_module.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    base_model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    messages: list[dict[str, str]] = []
    if not args.disable_system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": build_user_text(args.instruction, content)})
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if args.print_prompt:
        print("===== PROMPT BEGIN =====")
        print(prompt_text)
        print("===== PROMPT END =====")

    model_inputs = tokenizer(prompt_text, return_tensors="pt")
    if not args.cpu and torch_module.cuda.is_available():
        model_inputs = {key: value.to(model.device) for key, value in model_inputs.items()}

    do_sample = args.temperature > 0.0
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    with torch_module.no_grad():
        output_ids = model.generate(**model_inputs, **generation_kwargs)
    prompt_len = model_inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][prompt_len:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
    template = extract_template(decoded)

    print(f"BASE_MODEL: {model_path}")
    print(f"ADAPTER_DIR: {adapter_dir}")
    print(f"INPUT_LOG: {content}")
    print(f"RAW_OUTPUT: {decoded}")
    print(f"PREDICTED_TEMPLATE: {template}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
