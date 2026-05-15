from __future__ import annotations

from typing import Any


class HFTemplateGenerator:
    def __init__(
        self,
        model_path: str,
        adapter_dir: str | None,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        attn_implementation: str,
        trust_remote_code: bool,
        force_cpu: bool,
        seed: int,
    ) -> None:
        self.model_path = model_path
        self.adapter_dir = adapter_dir
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.attn_implementation = attn_implementation
        self.trust_remote_code = trust_remote_code
        self.force_cpu = force_cpu
        self.seed = seed

        self._torch = None
        self._model = None
        self._tokenizer = None
        self._load()

    def _ensure_dependencies(self) -> dict[str, Any]:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        except Exception as exc:
            raise RuntimeError(
                "Missing online inference dependencies. Install them first, for example:\n"
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

    def _choose_dtype(self, torch_module: Any):
        if self.force_cpu:
            return torch_module.float32
        if torch_module.cuda.is_available() and torch_module.cuda.is_bf16_supported():
            return torch_module.bfloat16
        if torch_module.cuda.is_available():
            return torch_module.float16
        return torch_module.float32

    def _load(self) -> None:
        mods = self._ensure_dependencies()
        torch_module = mods["torch"]
        PeftModel = mods["PeftModel"]
        AutoModelForCausalLM = mods["AutoModelForCausalLM"]
        AutoTokenizer = mods["AutoTokenizer"]
        set_seed = mods["set_seed"]

        set_seed(self.seed)
        self._torch = torch_module

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        dtype = self._choose_dtype(torch_module)
        attn_impl = None if self.attn_implementation == "auto" else self.attn_implementation
        model_kwargs: dict[str, Any] = {
            "pretrained_model_name_or_path": self.model_path,
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": dtype,
        }
        if attn_impl is not None:
            model_kwargs["attn_implementation"] = attn_impl
        if not self.force_cpu and torch_module.cuda.is_available():
            model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        if self.adapter_dir:
            model = PeftModel.from_pretrained(model, self.adapter_dir)
        model.eval()

        self._model = model
        self._tokenizer = tokenizer

    def generate(self, messages: list[dict[str, str]]) -> str:
        if self._model is None or self._tokenizer is None or self._torch is None:
            raise RuntimeError("Model runner is not initialized.")

        prompt_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self._tokenizer(prompt_text, return_tensors="pt")
        if not self.force_cpu and self._torch.cuda.is_available():
            model_inputs = {key: value.to(self._model.device) for key, value in model_inputs.items()}

        do_sample = self.temperature > 0.0
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p

        with self._torch.no_grad():
            output_ids = self._model.generate(**model_inputs, **generation_kwargs)
        prompt_len = model_inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][prompt_len:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
