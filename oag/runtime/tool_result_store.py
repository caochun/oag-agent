"""Persistence helpers for oversized tool results."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any


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
        "hint": "完整工具结果已保存到 path，当前仅返回预览。如需完整内容，请调用 read_tool_result；不要用领域 read_document 读取该 path。",
    }, ensure_ascii=False)


def read_persisted_tool_result(*, path: str,
                               max_chars: int = 12000,
                               storage_dir: str | None = None) -> str:
    requested = Path(path).expanduser().resolve()
    allowed_roots = [_base_dir(None).resolve()]
    if storage_dir:
        allowed_roots.append(_base_dir(storage_dir).resolve())
    if not any(_is_relative_to(requested, root) for root in allowed_roots):
        return _json_error(
            "只能读取 OAG 持久化工具结果目录下的文件，不能读取普通业务文档或任意本地文件。",
            path=str(requested),
            allowed_roots=[str(root) for root in allowed_roots],
        )
    if not requested.exists() or not requested.is_file():
        return _json_error("未找到持久化工具结果文件。", path=str(requested))
    max_chars = max(1000, min(int(max_chars or 12000), 50000))
    content = requested.read_text(encoding="utf-8", errors="replace")
    truncated = len(content) > max_chars
    return json.dumps({
        "path": str(requested),
        "chars": len(content),
        "returned_chars": min(len(content), max_chars),
        "truncated": truncated,
        "content": content[:max_chars],
        "hint": "这是通用持久化工具结果，不是领域文档；不要再用领域 read_document 读取该 path。",
    }, ensure_ascii=False)


def _safe_name(value: str) -> str:
    value = value.strip() or "default"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:80]


def _base_dir(storage_dir: str | None) -> Path:
    if storage_dir:
        return Path(storage_dir) / "tool-results"
    return Path(tempfile.gettempdir()) / "oag-tool-results"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _json_error(message: str, **extra: Any) -> str:
    return json.dumps({"error": message, **extra}, ensure_ascii=False)
