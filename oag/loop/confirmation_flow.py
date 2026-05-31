"""用户确认后的继续执行流程。

当工具调用因为写操作确认或 ask_user 问题暂停时，本模块负责把用户的批准、
拒绝或回答写回消息列表，然后把控制权交还给正常 QueryLoop。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Callable, Generator

from ..runtime import PendingConfirmation, RunState, ToolUseContext
from ..runtime.events import Event, TextEvent, ToolCallEvent

if TYPE_CHECKING:
    from ..harness import Harness


SaveMessages = Callable[[str, list[dict]], None]
RunLoop = Callable[[RunState], Generator[Event, None, None]]


class ConfirmationFlow:
    def __init__(self, harness: Harness, save_messages: SaveMessages,
                 run_loop: RunLoop):
        self.harness = harness
        self.save_messages = save_messages
        self.run_loop = run_loop

    def confirm(self, pending: PendingConfirmation | None, approved: bool,
                answer: str | None = None) -> Generator[Event, None, None]:
        if not pending:
            yield TextEvent(content="没有待确认的操作。")
            return

        messages = pending.messages

        if not approved:
            messages.append({
                "role": "tool",
                "tool_call_id": pending.tool_call_id,
                "content": json.dumps({"denied": True, "reason": "用户拒绝执行"}, ensure_ascii=False),
            })
            self._append_skipped_tool_results(messages, pending)
            messages.append({
                "role": "user",
                "content": f"[系统提示] 用户拒绝了 {pending.tool_name} 的执行",
            })
            self.save_messages(pending.session_id, messages)
            yield TextEvent(content=f"已取消 {pending.tool_name} 的执行。")
            return

        if pending.tool_name == "ask_user" and answer:
            messages.append({
                "role": "tool",
                "tool_call_id": pending.tool_call_id,
                "content": json.dumps({"answer": answer}, ensure_ascii=False),
            })
            self._append_skipped_tool_results(messages, pending)
            yield from self._continue(pending.session_id, messages, pending)
            return

        context = ToolUseContext(
            session_id=pending.session_id,
            messages=messages,
            confirmed=True,
        )
        result = self.harness.execute_tool(pending.tool_name, pending.args, context=context)

        yield ToolCallEvent(
            name=pending.tool_name,
            args=pending.args,
            result=result.content[:200],
        )

        messages.append({
            "role": "tool",
            "tool_call_id": pending.tool_call_id,
            "content": result.content,
        })
        self._append_skipped_tool_results(messages, pending)

        yield from self._continue(pending.session_id, messages, pending)

    def _append_skipped_tool_results(self, messages: list[dict],
                                     pending: PendingConfirmation) -> None:
        for skipped in pending.skipped_tool_calls or []:
            messages.append({
                "role": "tool",
                "tool_call_id": skipped["tool_call_id"],
                "content": skipped["content"],
            })

    def _continue(self, session_id: str, messages: list[dict],
                  pending: PendingConfirmation | None = None) -> Generator[Event, None, None]:
        state = RunState(
            messages=messages,
            session_id=session_id,
            user_question=pending.user_question if pending else "",
            turn_count=pending.turn_count if pending else 0,
            stop_hook_active=pending.stop_hook_active if pending else False,
        )
        yield from self.run_loop(state)
        self.save_messages(session_id, messages)
