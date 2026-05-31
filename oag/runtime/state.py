"""运行时状态容器。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunState:
    messages: list[dict]
    session_id: str
    user_question: str = ""
    turn_count: int = 0
    stop_hook_active: bool = False
    transition_reason: str | None = None


@dataclass(frozen=True)
class PendingConfirmation:
    session_id: str
    tool_name: str
    args: dict
    tool_call_id: str
    messages: list[dict]
    skipped_tool_calls: list[dict] | None = None
    user_question: str = ""
    turn_count: int = 0
    stop_hook_active: bool = False


@dataclass(frozen=True)
class ToolUseContext:
    session_id: str = ""
    messages: list[dict] | None = None
    confirmed: bool = False
    source: str = "main"
    agent_id: str | None = None
    allow_user_prompt: bool = True
    cancelled: bool = False
    storage_dir: str | None = None
