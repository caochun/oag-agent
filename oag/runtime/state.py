"""运行时状态容器。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunState:
    messages: list[dict]
    session_id: str
    user_question: str = ""
    allowed_tools: frozenset[str] | None = None
    turn_count: int = 0
    query_complete_retry_active: bool = False
    cache_namespace: str = ""


@dataclass(frozen=True)
class PendingConfirmation:
    session_id: str
    tool_name: str
    args: dict
    tool_call_id: str
    messages: list[dict]
    skipped_tool_calls: list[dict] | None = None
    expects_answer: bool = False
    user_question: str = ""
    allowed_tools: frozenset[str] | None = None
    turn_count: int = 0
    query_complete_retry_active: bool = False
    cache_namespace: str = ""


@dataclass(frozen=True)
class ToolUseContext:
    session_id: str = ""
    messages: list[dict] | None = None
    turn_count: int | None = None
    confirmed: bool = False
    source: str = "main"
    cancelled: bool = False
    storage_dir: str | None = None
    cache_namespace: str = ""
