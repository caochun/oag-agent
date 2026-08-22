"""运行时 hook 系统。

Hooks 用于在工具调用前后、最终回答后执行横切策略。默认 hook 只覆盖写操作
确认和审计日志；领域或应用可以显式注册完成检查。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

logger = logging.getLogger(__name__)

HOOK_EVENTS = (
    "pre_tool_call",
    "post_tool_call",
    "query_complete",
)

HookHandler = Callable[[dict], "HookResult"]


@dataclass
class HookResult:
    action: str = "allow"  # allow / block / pause
    reason: str = ""


class HookRegistry:
    def __init__(self):
        self._hooks: dict[str, list[HookHandler]] = {}

    def register(self, event: str, handler: HookHandler):
        if event not in HOOK_EVENTS:
            raise ValueError(f"Unknown hook event: {event}. Must be one of {HOOK_EVENTS}")
        self._hooks.setdefault(event, []).append(handler)

    def unregister(self, event: str, handler: HookHandler):
        handlers = self._hooks.get(event, [])
        if handler in handlers:
            handlers.remove(handler)

    def fire(self, event: str, context: dict) -> HookResult:
        for handler in self._hooks.get(event, []):
            try:
                result = handler(context)
                if result.action == "block":
                    return result
                if result.action == "pause":
                    return result
            except Exception as e:
                logger.warning(f"Hook handler error on {event}: {e}")
        return HookResult(action="allow")

    def has_handlers(self, event: str) -> bool:
        return bool(self._hooks.get(event))


class AuditLog:
    def __init__(self):
        self._entries: list[dict] = []

    def record(self, entry: dict):
        entry["timestamp"] = datetime.now().isoformat()
        self._entries.append(entry)

    def get_entries(self, limit: int | None = None) -> list[dict]:
        if limit:
            return self._entries[-limit:]
        return list(self._entries)

    def clear(self):
        self._entries.clear()


def write_confirmation_hook(context: dict) -> HookResult:
    tool_meta = context.get("tool_meta")
    if tool_meta and tool_meta.policy.requires_confirmation:
        # 暂停而不是直接执行；用户确认后 ConfirmationFlow 会以 confirmed=True 恢复。
        return HookResult(
            action="pause",
            reason=f"函数 {context['tool_name']} 将修改数据，请确认执行",
        )
    return HookResult(action="allow")


def audit_log_hook(context: dict) -> HookResult:
    audit_log = context.get("audit_log")
    if audit_log:
        audit_log.record({
            "event": context.get("hook_event", "unknown"),
            "tool": context.get("tool_name", ""),
            "args": context.get("args", {}),
            "session_id": context.get("session_id", ""),
            "result_preview": str(context.get("result", ""))[:200],
        })
    return HookResult(action="allow")
