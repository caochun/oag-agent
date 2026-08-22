"""Persistence helpers for oversized tool results."""

from __future__ import annotations

import json
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any


class ToolResultStore:
    """Keep large tool results behind opaque references.

    The filesystem is an implementation detail of OAG.  Models receive only
    ``result_ref`` values, so a tool cannot accidentally expose server paths.
    """

    def __init__(self, storage_dir: str | None = None):
        self.storage_dir = storage_dir
        self._refs: dict[str, Path] = {}

    def persist(self, *, session_id: str, tool_name: str, content: str,
                preview_chars: int, storage_dir: str | None = None) -> str:
        safe_session = _safe_name(session_id or "default")
        safe_tool = _safe_name(tool_name)
        result_dir = _base_dir(storage_dir or self.storage_dir) / safe_session
        result_dir.mkdir(parents=True, exist_ok=True)

        path = result_dir / f"{safe_tool}.txt"
        if path.exists():
            stem = path.stem
            suffix = 2
            while path.exists():
                path = result_dir / f"{stem}-{suffix}.txt"
                suffix += 1
        path.write_text(content, encoding="utf-8")

        result_ref = f"result:{uuid.uuid4().hex}"
        self._refs[result_ref] = path
        return json.dumps({
            "persisted": True,
            "result_ref": result_ref,
            "original_chars": len(content),
            "preview_chars": preview_chars,
            "preview": content[:preview_chars],
            "hint": "完整工具结果已保存；需要更多内容时调用 read_tool_result，并传入 result_ref。",
        }, ensure_ascii=False)

    def read(self, *, result_ref: str, max_chars: int = 12000) -> str:
        path = self._refs.get(str(result_ref or ""))
        if path is None:
            return _json_error("未找到 OAG 工具结果引用。")
        return _read_result_file(
            path,
            max_chars=_bounded_max_chars(max_chars),
            display_ref=result_ref,
        )


def _read_result_file(path: Path, *, max_chars: int,
                      display_ref: str) -> str:
    content = path.read_text(encoding="utf-8", errors="replace")
    truncated = len(content) > max_chars
    return json.dumps({
        "result_ref": display_ref,
        "chars": len(content),
        "returned_chars": min(len(content), max_chars),
        "truncated": truncated,
        "content": content[:max_chars],
        "hint": "这是 OAG 持久化的工具结果。",
    }, ensure_ascii=False)


def _safe_name(value: str) -> str:
    value = value.strip() or "default"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:80]


def _base_dir(storage_dir: str | None) -> Path:
    if storage_dir:
        return Path(storage_dir) / "tool-results"
    return Path(tempfile.gettempdir()) / "oag-tool-results"


def _bounded_max_chars(value: int) -> int:
    return max(1000, min(int(value or 12000), 50000))


def _json_error(message: str, **extra: Any) -> str:
    return json.dumps({"error": message, **extra}, ensure_ascii=False)
