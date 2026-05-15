from __future__ import annotations

import re


TEMPLATE_RE = re.compile(r"<START>\s*(.+?)\s*<END>", re.DOTALL)


def extract_template_from_response(text: str) -> str:
    if not text:
        return ""
    match = TEMPLATE_RE.search(text)
    if not match:
        return ""
    return normalize_template_whitespace(match.group(1))


def normalize_template_whitespace(template: str) -> str:
    return re.sub(r"\s+", " ", template.strip())


def validate_template_against_log(log_text: str, template: str) -> tuple[bool, list[str], list[str]]:
    template = normalize_template_whitespace(template)
    if not template:
        return False, ["<EMPTY_TEMPLATE>"], []
    if template == "<*>":
        return True, [], []

    template_parts = template.split("<*>")
    start_idx = 0
    errors: list[str] = []
    replacements: list[str] = []

    for part in template_parts:
        stripped = part.strip()
        if not stripped:
            continue
        idx = log_text.find(stripped, start_idx)
        if idx == -1:
            errors.append(stripped)
            continue
        if start_idx != idx:
            replacements.append(log_text[start_idx:idx].strip())
        start_idx = idx + len(stripped)

    if start_idx < len(log_text):
        replacements.append(log_text[start_idx:].strip())

    if errors:
        return False, errors, replacements
    return True, [], replacements
