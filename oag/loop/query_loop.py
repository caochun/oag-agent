"""主 LLM 回合循环。

QueryLoop 负责把消息和工具发给模型、记录调试事件、执行模型请求的工具、
处理确认暂停，并在最终回答后触发通用完成 hook。它不直接实现工具策略。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Callable, Generator

from openai import APIStatusError, OpenAI

from ..llm.retry import call_llm_with_retry
from ..runtime import RunState
from ..runtime.events import (
    AssistantDeltaEvent,
    AssistantEndEvent,
    CompactEvent,
    DebugEvent,
    Event,
    ToolCallEvent,
)
from ..runtime.message_sanitizer import sanitize_messages
from ..tools.pipeline import ToolResult
from .response_parser import LlmResponseParser
from .tool_call_coordinator import ToolCallCoordinator, ToolExecutionPaused
from .tool_executor import ToolExecutor

if TYPE_CHECKING:
    from ..harness import Harness


PendingConfirmationHandler = Callable[[str, str, dict, str, list[dict], RunState, list[dict] | None, bool], None]
class QueryLoop:
    def __init__(self, harness: Harness, llm_client: OpenAI, model: str,
                 on_pending_confirmation: PendingConfirmationHandler):
        self.harness = harness
        self.client = llm_client
        self.model = model
        self.on_pending_confirmation = on_pending_confirmation
        self.tool_executor = ToolExecutor(harness)
        self.response_parser = LlmResponseParser()
        self.tool_coordinator = ToolCallCoordinator(
            harness,
            self.tool_executor,
            on_pending_confirmation,
        )

    def run(self, state: RunState) -> Generator[Event, None, None]:
        tools = self._filter_tools_for_run(self.harness.build_tools(), state.allowed_tools)
        visible_tool_names = {
            tool.get("function", {}).get("name")
            for tool in tools
            if tool.get("function", {}).get("name")
        }

        while True:
            # 一次循环对应一个模型回合，以及该回合触发的工具执行结果。
            state.turn_count += 1
            sanitized_messages, sanitized = sanitize_messages(
                state.messages,
                repair_missing_tool_results=True,
            )
            if sanitized:
                state.messages[:] = sanitized_messages
            messages = state.messages
            self.harness.trace.record(
                "agent_turn_start",
                session_id=state.session_id,
                turn_count=state.turn_count,
                message_count=len(messages),
                query_complete_retry_active=state.query_complete_retry_active,
                allowed_tool_count=len(visible_tool_names),
                allowed_tools=sorted(visible_tool_names) if state.allowed_tools is not None else None,
            )
            if state.turn_count > self.harness.config.max_turns:
                self.harness.trace.record(
                    "agent_transition",
                    session_id=state.session_id,
                    turn_count=state.turn_count,
                    reason="max_turns_reached",
                )
                yield from self._finalize_at_turn_limit(state)
                return

            if state.turn_count > 1 and state.turn_count % 5 == 0:
                messages, compacted = self.harness.maybe_compact(messages)
                state.messages = messages
                if compacted:
                    yield CompactEvent()

            yield from self._compact_before_request(state)
            messages = state.messages
            self._record_context_usage(state, tools)
            yield self._build_request_debug_event(state)

            try:
                response = call_llm_with_retry(
                    self.client,
                    **self._request_kwargs(
                        messages=messages,
                        tools=tools if tools else None,
                        stream=True,
                    ),
                )
            except Exception as exc:
                if not self._is_context_overflow_error(exc):
                    raise
                compacted = yield from self._compact_after_overflow(state)
                if not compacted:
                    raise
                messages = state.messages
                self._record_context_usage(state, tools)
                response = call_llm_with_retry(
                    self.client,
                    **self._request_kwargs(
                        messages=messages,
                        tools=tools if tools else None,
                        stream=True,
                    ),
                )
            msg = yield from self._consume_llm_response(response)

            if not msg.tool_calls:
                yield from self._handle_final_response(
                    state,
                    msg.content or "",
                )
                return

            self._record_assistant_response(
                state,
                msg.content or "",
                kind="progress",
                has_tool_calls=True,
            )
            yield AssistantEndEvent(kind="progress")

            # OpenAI tool protocol 要求先保存 assistant 的 tool_calls envelope，
            # 再追加每个 tool_call_id 对应的 tool 消息。
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            tool_calls_parsed = []
            for tc in msg.tool_calls:
                args, parse_error = self._parse_tool_args(tc.function.arguments)
                if parse_error:
                    tool_calls_parsed.append((tc, args, parse_error))
                else:
                    tool_calls_parsed.append((tc, args, None))

            executable_calls = []
            for index, (tc, args, parse_error) in enumerate(tool_calls_parsed):
                if parse_error:
                    try:
                        yield from self._execute_tool_call_segment(state, messages, msg, executable_calls)
                    except ToolExecutionPaused as paused:
                        for event in paused.events:
                            yield event
                        return
                    executable_calls = []
                    yield ToolCallEvent(name=tc.function.name, args=args)
                    yield from self.tool_coordinator.handle_result(
                        state, messages, msg, index, tc, args, parse_error,
                    )
                    continue
                if tc.function.name not in visible_tool_names:
                    result = ToolResult(
                        content=json.dumps(
                            {
                                "error": "工具不可用",
                                "tool": tc.function.name,
                                "details": "该工具未对当前领域助手开放，请改用可用工具。",
                            },
                            ensure_ascii=False,
                        ),
                        blocked=True,
                        block_reason="工具未对当前领域助手开放",
                    )
                    yield ToolCallEvent(name=tc.function.name, args=args)
                    yield from self.tool_coordinator.handle_result(
                        state, messages, msg, index, tc, args, result,
                    )
                    continue
                executable_calls.append((index, tc, args))
            try:
                yield from self._execute_tool_call_segment(state, messages, msg, executable_calls)
            except ToolExecutionPaused as paused:
                for event in paused.events:
                    yield event
                return

            state.query_complete_retry_active = False
            self.harness.trace.record(
                "agent_transition",
                session_id=state.session_id,
                turn_count=state.turn_count,
                reason="next_turn",
            )

    def _finalize_at_turn_limit(self, state: RunState) -> Generator[Event, None, None]:
        instruction = (
            "工具调用轮次已经用完。不要再调用任何工具，也不要提及轮次限制。"
            "请仅依据对话中已有的工具结果，直接给用户一个简洁、明确的最终答复。"
            "如果证据不足，说明已经确认的事实和仍缺少的信息，不要继续尝试查询。"
        )
        request_messages = [
            *state.messages,
            {"role": "system", "content": instruction},
        ]
        try:
            response = call_llm_with_retry(
                self.client,
                **self._request_kwargs(
                    messages=request_messages,
                    tools=None,
                    stream=True,
                ),
            )
            msg = yield from self._consume_llm_response(response)
            content = (msg.content or "").strip()
        except Exception as exc:
            self.harness.trace.record(
                "agent_turn_limit_finalize_error",
                session_id=state.session_id,
                turn_count=state.turn_count,
                error=str(exc),
            )
            content = "当前已完成可用信息的查询，但未能生成最终总结。请缩小问题范围后重试。"
            yield AssistantDeltaEvent(content=content)

        if not content:
            content = "当前没有足够的已确认信息来回答该问题，请缩小问题范围后重试。"
            yield AssistantDeltaEvent(content=content)
        state.messages.append({"role": "assistant", "content": content})
        self._record_assistant_response(state, content, kind="final")
        yield AssistantEndEvent(kind="final")
        self.harness.trace.record(
            "agent_transition",
            session_id=state.session_id,
            turn_count=state.turn_count,
            reason="max_turns_final_response",
        )

    def _execute_tool_call_segment(self, state: RunState, messages: list[dict], msg,
                                   executable_calls: list[tuple[int, object, dict]]) -> Generator[Event, None, None]:
        yield from self.tool_coordinator.execute_segment(
            state, messages, msg, executable_calls,
        )

    def _handle_final_response(self, state: RunState,
                               content: str) -> Generator[Event, None, None]:
        messages = state.messages
        messages.append({"role": "assistant", "content": content})

        completion_feedback = self.harness.run_query_complete_hooks(
            state.user_question,
            messages,
        )
        if completion_feedback and not state.query_complete_retry_active:
            self._record_assistant_response(
                state,
                content,
                kind="progress",
                classification_reason="query_complete_retry",
            )
            yield AssistantEndEvent(kind="progress")
            state.query_complete_retry_active = True
            self.harness.trace.record(
                "agent_transition",
                session_id=state.session_id,
                turn_count=state.turn_count,
                reason="query_complete_retry",
            )
            messages.append({"role": "user", "content": completion_feedback})
            yield from self.run(state)
            return

        self._record_assistant_response(state, content, kind="final")
        yield AssistantEndEvent(kind="final")
        self.harness.trace.record(
            "agent_transition",
            session_id=state.session_id,
            turn_count=state.turn_count,
            reason="final_response",
        )

    def _record_assistant_response(
        self,
        state: RunState,
        content: str,
        *,
        kind: str,
        has_tool_calls: bool = False,
        classification_reason: str = "",
    ) -> None:
        if not content and kind == "progress":
            return
        self.harness.trace.record(
            "assistant_response",
            session_id=state.session_id,
            turn_count=state.turn_count,
            kind=kind,
            has_tool_calls=has_tool_calls,
            content_chars=len(content),
            content=content,
            classification_reason=classification_reason,
        )

    def _build_request_debug_event(self, state: RunState) -> DebugEvent:
        debug_msgs = []
        for m in state.messages[-6:]:
            role = m.get("role", "")
            if role == "system":
                debug_msgs.append(f"[SYS] {(m.get('content', ''))[:200]}")
            elif role == "user":
                debug_msgs.append(f"[USR] {(m.get('content', ''))[:200]}")
            elif role == "assistant":
                tc_names = [tc["function"]["name"] for tc in m.get("tool_calls", []) if isinstance(tc, dict)]
                if tc_names:
                    debug_msgs.append(f"[LLM] 调用->{', '.join(tc_names)} {(m.get('content', ''))[:100]}")
                else:
                    debug_msgs.append(f"[LLM] {(m.get('content', ''))[:200]}")
            elif role == "tool":
                debug_msgs.append(f"[TOOL] {(m.get('content', ''))[:200]}")
        return DebugEvent(
            stage="request",
            content=f"Turn {state.turn_count}, {len(state.messages)} msgs\n" + "\n".join(debug_msgs),
        )

    def _record_context_usage(self, state: RunState, tools: list[dict]):
        usage = self.harness.collect_context_usage(state.messages, tools)
        self.harness.trace.record(
            "context_usage",
            session_id=state.session_id,
            turn_count=state.turn_count,
            model=usage["model"],
            total_tokens=usage["total_tokens"],
            context_window=usage["context_window"],
            percentage=usage["percentage"],
            free_tokens=usage["free_tokens"],
            message_count=usage["messages"]["count"],
            tool_count=usage["tools"]["count"],
            categories=usage["categories"],
            largest_tool_results=usage["messages"]["largest_tool_results"],
            largest_tools=usage["tools"]["largest_tools"],
        )

    def _request_kwargs(self, *, messages: list[dict], tools: list[dict] | None,
                        stream: bool) -> dict:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "temperature": 0.1,
            "max_tokens": self.harness.config.max_response_tokens,
            "stream": stream,
        }
        if self.harness.config.llm_extra_body:
            kwargs["extra_body"] = self.harness.config.llm_extra_body
        return kwargs

    @staticmethod
    def _filter_tools_for_run(tools: list[dict], allowed_tools: frozenset[str] | None) -> list[dict]:
        if allowed_tools is None:
            return tools
        return [
            tool for tool in tools
            if tool.get("function", {}).get("name") in allowed_tools
        ]

    def _consume_llm_response(self, response):
        message = yield from self.response_parser.consume(response)
        return message

    def _compact_before_request(self, state: RunState) -> Generator[Event, None, None]:
        messages, compacted = self.harness.maybe_compact(state.messages)
        state.messages = messages
        if compacted:
            yield CompactEvent()

    def _compact_after_overflow(self, state: RunState) -> Generator[Event, None, bool]:
        before = state.messages
        messages, compacted = self.harness.force_compact(before)
        state.messages = messages
        if compacted:
            yield CompactEvent()
            return True
        return False

    def _is_context_overflow_error(self, exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if isinstance(exc, APIStatusError) and status_code not in (400, 413):
            return False

        code = ""
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            err = body.get("error", body)
            if isinstance(err, dict):
                code = str(err.get("code") or err.get("type") or "")

        text = f"{code} {exc}".lower()
        return any(marker in text for marker in (
            "context_length_exceeded",
            "maximum context length",
            "context length",
            "prompt too long",
            "too many tokens",
            "request too large",
        ))

    def _parse_tool_args(self, raw_args: str) -> tuple[dict, ToolResult | None]:
        try:
            parsed = json.loads(raw_args or "{}")
        except json.JSONDecodeError as exc:
            reason = f"工具参数不是合法 JSON: {exc.msg}"
            return {}, ToolResult(
                content=json.dumps({"error": reason}, ensure_ascii=False),
                blocked=True,
                block_reason=reason,
            )
        if not isinstance(parsed, dict):
            reason = "工具参数必须是 JSON object"
            return {}, ToolResult(
                content=json.dumps({"error": reason}, ensure_ascii=False),
                blocked=True,
                block_reason=reason,
            )
        return parsed, None

    def _build_skipped_tool_calls(self, tool_calls: list) -> list[dict]:
        return [
            {
                "tool_call_id": tc.id,
                "content": json.dumps({
                    "skipped": True,
                    "reason": "前一个工具调用需要用户确认，本调用未执行",
                }, ensure_ascii=False),
            }
            for tc in tool_calls
        ]
