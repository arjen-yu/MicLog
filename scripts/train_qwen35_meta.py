#!/usr/bin/env python3

import argparse
import json
import os
import site
import sys
from pathlib import Path
from typing import Any


# Keep user-site packages out of the training process. In mixed environments
# they can shadow the conda env with incompatible wheels, which breaks imports
# inside transformers/peft.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA / QLoRA fine-tuning for local causal language models on MicLog JSONL SFT data."
    )
    parser.add_argument("--model", required=True, help="model alias (0.8b/2b/4b/9b) or absolute model path")
    parser.add_argument("--train-file", required=True, help="training jsonl path")
    parser.add_argument("--output-dir", required=True, help="output directory for adapter checkpoints")
    parser.add_argument("--method", choices=["lora", "qlora"], default="lora")
    parser.add_argument("--dataset-format", choices=["auto", "instruction", "messages"], default="auto")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--eval-ratio", type=float, default=0.02)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-strategy", choices=["steps", "epoch"], default="epoch")
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--eval-strategy", choices=["no", "steps", "epoch"], default=None)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", default="none", help="none, tensorboard, wandb, or comma-separated list")
    parser.add_argument("--attn-implementation", choices=["auto", "sdpa", "flash_attention_2", "eager"], default="sdpa")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--disable-system-prompt", action="store_true")
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="all-linear",
        help="LoRA target modules. Use 'all-linear' or a comma-separated list.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def resolve_model_path(model_arg: str) -> str:
    alias = model_arg.strip().lower()
    resolved = MODEL_ALIASES.get(alias, model_arg)
    path = Path(resolved)
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {resolved}")
    return str(path.resolve())


def ensure_dependencies() -> dict[str, Any]:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except Exception as exc:
        raise SystemExit(
            "Missing fine-tuning dependencies. Install them first, for example:\n"
            "  PYTHONNOUSERSITE=1 pip install -U -r requirements.txt\n"
            "or\n"
            "  PYTHONNOUSERSITE=1 pip install -U \"torch>=2.4\" \"accelerate>=1.8.1\" \"datasets>=2.19\" \"peft>=0.19.0\" \"bitsandbytes>=0.43.0\" \"tokenizers>=0.22.0\" \"safetensors>=0.5.3\" \"sympy>=1.13\" \"mpmath>=1.3.0\" \"transformers>=4.53\"\n"
            f"Original import error: {type(exc).__name__}: {exc}"
        ) from exc

    return {
        "torch": torch,
        "load_dataset": load_dataset,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "set_seed": set_seed,
    }


def choose_precision(torch_module: Any) -> tuple[Any, bool, bool]:
    if torch_module.cuda.is_available() and torch_module.cuda.is_bf16_supported():
        return torch_module.bfloat16, True, False
    if torch_module.cuda.is_available():
        return torch_module.float16, False, True
    return torch_module.float32, False, False


def parse_report_to(value: str) -> str | list[str]:
    cleaned = value.strip().lower()
    if cleaned == "none":
        return "none"
    return [item.strip() for item in value.split(",") if item.strip()]


def target_modules_value(raw_value: str) -> str | list[str]:
    cleaned = raw_value.strip()
    if cleaned == "all-linear":
        return cleaned
    return [item.strip() for item in cleaned.split(",") if item.strip()]


def build_user_text(example: dict[str, Any]) -> str:
    instruction = str(example.get("instruction", "")).strip()
    input_text = str(example.get("input", "")).strip()
    if instruction and input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction or input_text


def detect_dataset_format(example: dict[str, Any], forced: str) -> str:
    if forced != "auto":
        return forced
    if {"instruction", "input", "output"}.issubset(example):
        return "instruction"
    if "messages" in example:
        return "messages"
    raise ValueError(f"Unsupported example keys: {sorted(example.keys())}")


def build_prompt_and_answer(
    example: dict[str, Any],
    dataset_format: str,
    system_prompt: str | None,
    tokenizer: Any,
) -> tuple[str, str]:
    if dataset_format == "instruction":
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": build_user_text(example)})
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        answer_text = str(example.get("output", ""))
        return prompt_text, answer_text

    messages = example.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages format requires a non-empty 'messages' list")
    if messages[-1].get("role") != "assistant":
        raise ValueError("messages format requires the last role to be assistant")
    prompt_messages = messages[:-1]
    if system_prompt and (not prompt_messages or prompt_messages[0].get("role") != "system"):
        prompt_messages = [{"role": "system", "content": system_prompt}] + prompt_messages
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    answer_text = str(messages[-1].get("content", ""))
    return prompt_text, answer_text


def make_preprocess_fn(tokenizer: Any, args: argparse.Namespace):
    eos_token = tokenizer.eos_token or ""
    system_prompt = None if args.disable_system_prompt else args.system_prompt

    def preprocess(example: dict[str, Any]) -> dict[str, Any]:
        dataset_format = detect_dataset_format(example, args.dataset_format)
        prompt_text, answer_text = build_prompt_and_answer(example, dataset_format, system_prompt, tokenizer)
        full_text = prompt_text + answer_text
        if eos_token and not full_text.endswith(eos_token):
            full_text += eos_token

        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_length,
        )["input_ids"]
        full_tokens = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_length,
        )
        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        supervised_tokens = sum(token != -100 for token in labels)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "supervised_tokens": supervised_tokens,
        }

    return preprocess


class SupervisedDataCollator:
    def __init__(self, tokenizer: Any, torch_module: Any) -> None:
        self.tokenizer = tokenizer
        self.torch = torch_module

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids = [self.torch.tensor(feature["input_ids"], dtype=self.torch.long) for feature in features]
        attention_mask = [self.torch.tensor(feature["attention_mask"], dtype=self.torch.long) for feature in features]
        labels = [self.torch.tensor(feature["labels"], dtype=self.torch.long) for feature in features]
        input_ids = self.torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = self.torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
        )
        labels = self.torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_model_and_tokenizer(mods: dict[str, Any], args: argparse.Namespace, model_path: str):
    torch_module = mods["torch"]
    AutoTokenizer = mods["AutoTokenizer"]
    BitsAndBytesConfig = mods["BitsAndBytesConfig"]
    AutoModelForCausalLM = mods["AutoModelForCausalLM"]
    LoraConfig = mods["LoraConfig"]
    get_peft_model = mods["get_peft_model"]
    prepare_model_for_kbit_training = mods["prepare_model_for_kbit_training"]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=args.trust_remote_code, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    torch_dtype, use_bf16, use_fp16 = choose_precision(torch_module)
    attn_impl = None if args.attn_implementation == "auto" else args.attn_implementation

    model_kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": model_path,
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if attn_impl is not None:
        model_kwargs["attn_implementation"] = attn_impl

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if args.method == "lora" and torch_module.cuda.is_available():
        model_kwargs["device_map"] = {"": local_rank} if world_size > 1 else {"": 0}

    if args.method == "qlora":
        if not torch_module.cuda.is_available():
            raise SystemExit("QLoRA requires CUDA because bitsandbytes 4-bit training is GPU-only.")
        quant_dtype = torch_module.bfloat16 if use_bf16 else torch_module.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=quant_dtype,
        )
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = {"": local_rank} if world_size > 1 else "auto"

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    if args.method == "qlora":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=not args.disable_gradient_checkpointing,
        )

    if not args.disable_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules_value(args.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer, use_bf16, use_fp16


def main() -> int:
    args = parse_args()
    mods = ensure_dependencies()
    torch_module = mods["torch"]
    load_dataset = mods["load_dataset"]
    Trainer = mods["Trainer"]
    TrainingArguments = mods["TrainingArguments"]
    set_seed = mods["set_seed"]

    model_path = resolve_model_path(args.model)
    train_file = Path(args.train_file).resolve()
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch_module.cuda.is_available():
        raise SystemExit(
            "CUDA is not available in the current PyTorch runtime. "
            "Your installed torch build appears incompatible with the NVIDIA driver, so training would fall back to CPU. "
            "Install a CUDA 12.x compatible PyTorch build first, then retry.\n"
            "Suggested fix for this machine: pip uninstall -y torch torchvision torchaudio && "
            "pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio"
        )

    set_seed(args.seed)
    model, tokenizer, use_bf16, use_fp16 = load_model_and_tokenizer(mods, args, model_path)

    raw_dataset = load_dataset("json", data_files=str(train_file))["train"]
    if args.max_train_samples is not None:
        raw_dataset = raw_dataset.select(range(min(args.max_train_samples, len(raw_dataset))))

    preprocess_fn = make_preprocess_fn(tokenizer, args)
    processed_dataset = raw_dataset.map(
        preprocess_fn,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing training data",
    )
    processed_dataset = processed_dataset.filter(
        lambda example: example["supervised_tokens"] > 0,
        desc="Dropping fully truncated samples",
    )

    if len(processed_dataset) == 0:
        raise SystemExit("All samples became empty after tokenization/truncation. Increase --max-length.")

    eval_dataset = None
    if args.eval_ratio > 0.0 and len(processed_dataset) >= 10:
        split = processed_dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed, shuffle=True)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        if args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))
    else:
        train_dataset = processed_dataset

    if args.eval_strategy is None:
        eval_strategy = "epoch" if eval_dataset is not None else "no"
    else:
        eval_strategy = args.eval_strategy
        if eval_strategy != "no" and eval_dataset is None:
            raise SystemExit("eval_strategy requires a non-empty evaluation split. Increase --eval-ratio.")

    run_config = {
        "resolved_model_path": model_path,
        "train_file": str(train_file),
        "output_dir": str(output_dir),
        "method": args.method,
        "max_length": args.max_length,
        "train_examples": len(train_dataset),
        "eval_examples": 0 if eval_dataset is None else len(eval_dataset),
        "dataset_format": args.dataset_format,
        "system_prompt_enabled": not args.disable_system_prompt,
        "system_prompt": None if args.disable_system_prompt else args.system_prompt,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps if eval_strategy == "steps" else None,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        save_total_limit=args.save_total_limit,
        bf16=use_bf16,
        fp16=use_fp16,
        dataloader_num_workers=2,
        report_to=parse_report_to(args.report_to),
        remove_unused_columns=False,
        label_names=["labels"],
        gradient_checkpointing=not args.disable_gradient_checkpointing,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SupervisedDataCollator(tokenizer, torch_module),
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    metrics = trainer.state.log_history
    (output_dir / "trainer_log_history.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
