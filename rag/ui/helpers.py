"""Shared UI helper functions."""

import hashlib

# Build a short context window for follow-up questions.

def build_context(messages: list, max_turns: int) -> str:
    if not messages:
        return ""
    recent = messages[-(max_turns * 2):]
    lines = []
    for msg in recent:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ["user", "assistant"] and content:
            lines.append(f"{role.title()}: {content}")
    return "\n".join(lines)


# Clean and shorten raw source text for display.

def clean_source_text(text: str, max_len: int = 600) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."


# Generate a stable key for source toggle buttons.

def source_toggle_key(prefix: str, msg_idx: int, src_idx: int, file_name: str, text: str) -> str:
    raw_id = f"{prefix}|{msg_idx}|{src_idx}|{file_name}|{text[:120]}".encode(
        "utf-8",
        errors="ignore",
    )
    return prefix + "_" + hashlib.md5(raw_id).hexdigest()
