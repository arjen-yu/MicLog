from __future__ import annotations

from .config import DEFAULT_INSTRUCTION
from .types import RetrievedDemo


def template_answer(template: str) -> str:
    return f"<START>{template}<END>"


def build_input(query_content: str, demos: list[RetrievedDemo]) -> str:
    lines: list[str] = []
    if demos:
        lines.append("Examples:")
        for demo in demos:
            lines.append(f"<content>{demo.content}")
            lines.append(f"<template>{template_answer(demo.template)}")
            lines.append("")
        lines.append("Query:")
    lines.append(f"<content>{query_content}")
    lines.append("<template>")
    return "\n".join(lines).rstrip()


def build_user_text(query_content: str, demos: list[RetrievedDemo], instruction: str = DEFAULT_INSTRUCTION) -> str:
    return f"{instruction}\n\n{build_input(query_content, demos)}"


def build_messages(query_content: str, demos: list[RetrievedDemo], instruction: str, system_prompt: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": build_user_text(query_content, demos, instruction)})
    return messages
