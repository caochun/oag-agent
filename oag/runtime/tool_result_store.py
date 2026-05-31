"""Persistence helpers for oversized tool results."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path


def persist_large_tool_result(*, storage_dir: str | None,
                              session_id: str,
                              tool_name: str,
                              content: str,
                              preview_chars: int) -> str:
    safe_session = _safe_name(session_id or "default")
    safe_tool = _safe_name(tool_name)
    result_dir = _base_dir(storage_dir) / safe_session
    result_dir.mkdir(parents=True, exist_ok=True)

    path = result_dir / f"{safe_tool}.txt"
    if path.exists():
        stem = path.stem
        suffix = 2
        while path.exists():
            path = result_dir / f"{stem}-{suffix}.txt"
            suffix += 1

    path.write_text(content, encoding="utf-8")
    return json.dumps({
        "persisted": True,
        "path": str(path),
        "original_chars": len(content),
        "preview_chars": preview_chars,
        "preview": content[:preview_chars],
        "hint": "完整工具结果已保存到 path，当前仅返回预览。",
    }, ensure_ascii=False)


def _safe_name(value: str) -> str:
    value = value.strip() or "default"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:80]


def _base_dir(storage_dir: str | None) -> Path:
    if storage_dir:
        return Path(storage_dir) / "tool-results"
    return Path(tempfile.gettempdir()) / "oag-tool-results"
