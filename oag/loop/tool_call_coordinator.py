"""Coordinate one model response's tool calls and confirmation pause."""

from __future__ import annotations

import json
from typing import Callable, Generator

from ..runtime import RunState
from ..runtime.events import (
    ConfirmationEvent,
    Event,
    InteractionEvent,
    QuestionEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from ..tools.pipeline import ToolResult
from .tool_executor import ToolExecutor


class ToolExecutionPaused(Exception):
    """Internal control flow used when a tool needs user confirmation."""

    def __init__(self, events: list[Event]):
        super().__init__("tool execution paused")
        self.events = events


PendingConfirmationHandler = Callable[
    [str, str, dict, str, list[dict], RunState, list[dict] | None, bool], None
]


class ToolCallCoordinator:
    """Execute tool batches and append protocol-valid tool messages."""

    def __init__(self, harness, tool_executor: ToolExecutor,
                 on_pending_confirmation: PendingConfirmationHandler):
        self.harness = harness
        self.tool_executor = tool_executor
        self.on_pending_confirmation = on_pending_confirmation

    def execute_segment(
        self, state: RunState, messages: list[dict], message,
        executable_calls: list[tuple[int, object, dict]],
    ) -> Generator[Event, None, None]:
        if not executable_calls:
            return

        index_by_call_id = {call.id: index for index, call, _ in executable_calls}
        pending_events: list[Event] = []

        def handle_result(call, args, result):
            index = index_by_call_id[call.id]
            for event in self.handle_result(
                state, messages, message, index, call, args, result,
            ):
                pending_events.append(event)

        try:
            parsed = [(call, args) for _, call, args in executable_calls]
            for batch in self.tool_executor.partition_tool_calls(parsed):
                for call, args in batch:
                    yield ToolCallEvent(name=call.function.name, args=args)
                batch_results = self.tool_executor.execute_batch(batch, state)
                for call, args, result in batch_results:
                    handle_result(call, args, result)
                if any(
                    result.needs_confirmation or result.needs_user_input
                    for _, _, result in batch_results
                ):
                    break
        except ToolExecutionPaused:
            raise ToolExecutionPaused(pending_events)

        yield from pending_events

    def handle_result(
        self, state: RunState, messages: list[dict], message, index: int,
        call, args: dict, result: ToolResult,
    ) -> Generator[Event, None, None]:
        if result.needs_confirmation or result.needs_user_input:
            skipped = self.build_skipped_tool_calls(message.tool_calls[index + 1:])
            self.harness.trace.record(
                "agent_transition",
                session_id=state.session_id,
                turn_count=state.turn_count,
                reason=(
                    "user_input_required"
                    if result.needs_user_input
                    else "confirmation_required"
                ),
                tool_name=call.function.name,
            )
            self.on_pending_confirmation(
                state.session_id,
                call.function.name,
                args,
                call.id,
                messages,
                state,
                skipped,
                result.needs_user_input,
            )
            if result.needs_user_input:
                yield QuestionEvent(
                    question=args.get("question", ""),
                    options=args.get("options", []),
                    multi_select=args.get("multi_select", False),
                )
            else:
                yield ConfirmationEvent(
                    tool_name=call.function.name,
                    args=args,
                    reason=result.block_reason,
                )
            raise ToolExecutionPaused([])

        yield ToolResultEvent(
            name=call.function.name,
            result=result.content[:200],
            blocked=bool(result.blocked),
        )
        interaction = self.interaction_payload(call.function.name, result.content)
        if interaction is not None:
            yield InteractionEvent(name=call.function.name, payload=interaction)

        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": result.content,
        })
        if result.blocked:
            messages.append({
                "role": "user",
                "content": f"[系统提示] 工具 {call.function.name} 被阻止: {result.block_reason}",
            })

    def interaction_payload(self, tool_name: str, content: str) -> dict | None:
        tool = self.harness.tools.get(tool_name)
        if tool is None or tool.category != "interaction":
            return None
        try:
            value = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(value, dict):
            return None
        payload = value.get("interaction")
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def build_skipped_tool_calls(tool_calls: list) -> list[dict]:
        return [
            {
                "tool_call_id": call.id,
                "content": json.dumps({
                    "skipped": True,
                    "reason": "前一个工具调用需要用户确认，本调用未执行",
                }, ensure_ascii=False),
            }
            for call in tool_calls
        ]
