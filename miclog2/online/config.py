from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_SYSTEM_PROMPT = "You are a log parsing assistant that extracts templates from logs."
DEFAULT_INSTRUCTION = (
    "For each query log after the final <content> tag, try your best to extract one log template "
    "(substitute variable tokens in the log as <*> and remain constant tokens to construct the template) "
    "and put the template after the final <template> tag and between <START> and <END> tags. "
    "Use previous <content>/<template> pairs as in-context examples when they are provided."
)


@dataclass(slots=True)
class OnlineParserConfig:
    dataset: str
    model_path: str
    target_root: str
    support_root: str
    output_dir: str
    adapter_dir: str | None = None
    shots: int = 5
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT
    instruction: str = DEFAULT_INSTRUCTION
    retrieval_field: str = "Content"
    exact_cache_size: int = 50000
    signature_cache_size: int = 10000
    pattern_cache_size: int = 10000
    pattern_cache_version: str = "v2"
    exclude_same_content: bool = True
    enable_retrieval_fallback: bool = True
    max_generation_attempts: int = 1
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    attn_implementation: str = "sdpa"
    trust_remote_code: bool = True
    force_cpu: bool = False
    seed: int = 42
    max_target_logs: int | None = None
    show_progress: bool = True

    @property
    def target_root_path(self) -> Path:
        return Path(self.target_root).resolve()

    @property
    def support_root_path(self) -> Path:
        return Path(self.support_root).resolve()

    @property
    def output_dir_path(self) -> Path:
        return Path(self.output_dir).resolve()

    @property
    def model_path_obj(self) -> Path:
        return Path(self.model_path).resolve()

    @property
    def adapter_dir_path(self) -> Path | None:
        if self.adapter_dir is None:
            return None
        return Path(self.adapter_dir).resolve()
