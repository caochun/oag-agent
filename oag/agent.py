"""面向调用方的会话级 Agent API。

Agent 负责聊天会话、待确认操作、流式/SSE 输出和历史持久化；具体的 LLM
回合循环交给 QueryLoop，工具执行和策略约束交给 Harness。
"""

from __future__ import annotations

import uuid
from pathlib import Path
from time import monotonic
from typing import Iterable, Generator

from openai import OpenAI

from .runtime.events import (
    Event, TextEvent, event_to_dict,
)
from .runtime.genai_trace import OTLP_STATUS_ERROR, OTLP_STATUS_OK
from .harness import Harness
from .loop.confirmation_flow import ConfirmationFlow
from .loop.query_loop import QueryLoop
from .runtime import PendingConfirmation, RunState
from .runtime.session_store import SessionStore


class Agent:
    def __init__(self, harness: Harness, llm_client: OpenAI, model: str,
                 db_dir: str = ".oag_data"):
        Path(db_dir).mkdir(parents=True, exist_ok=True)
        if not harness.config.trace_jsonl_path:
            harness.trace.jsonl_path = str(Path(db_dir) / f"trace_{harness.ontology.name}.jsonl")

        self.harness = harness
        self.client = llm_client
        self.model = model
        self._pending: dict[str, PendingConfirmation] = {}
        self.query_loop = QueryLoop(
            harness,
            llm_client,
            model,
            on_pending_confirmation=self._set_pending_confirmation,
        )
        self.confirmation_flow = ConfirmationFlow(
            harness,
            save_messages=self.sessions_save,
            run_loop=self._run_loop,
        )

        db_path = str(Path(db_dir) / f"chat_{harness.ontology.name}.db")
        self.sessions = SessionStore(db_path)

    def chat(self, message: str, session_id: str = "default",
             allowed_tools: Iterable[str] | None = None) -> str:
        result_parts = []
        for event in self.chat_stream(message, session_id, allowed_tools=allowed_tools):
            if isinstance(event, TextEvent):
                result_parts.append(event.content)
        return "".join(result_parts)

    def has_pending(self, session_id: str) -> bool:
        return session_id in self._pending

    def pending_tool_name(self, session_id: str) -> str | None:
        """Return the tool currently waiting for this session's response."""
        pending = self._pending.get(session_id)
        return pending.tool_name if pending else None

    def confirm_tool(self, session_id: str, approved: bool,
                     answer: str | None = None) -> Generator[Event, None, None]:
        pending = self._pending.pop(session_id, None)
        try:
            yield from self.confirmation_flow.confirm(pending, approved, answer)
        finally:
            if pending:
                self.harness.clear_tool_cache_namespace(
                    pending.cache_namespace,
                )

    def chat_stream(self, message: str, session_id: str = "default",
                    allowed_tools: Iterable[str] | None = None,
                    run_id: str = "") -> Generator[Event, None, None]:
        if session_id in self._pending:
            yield TextEvent(content="当前会话有待确认的操作，请先确认或取消后再继续。")
            return

        messages = self.sessions.get(session_id)

        if not messages:
            system_prompt = self.harness.build_system_prompt()
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": message})
        self.sessions.save(session_id, messages)

        cache_namespace = f"agent-run:{uuid.uuid4().hex}"
        state = RunState(
            messages=messages,
            session_id=session_id,
            cache_namespace=cache_namespace,
            user_question=message,
            allowed_tools=frozenset(allowed_tools) if allowed_tools is not None else None,
        )
        streamed_content = ""
        completed = False
        last_snapshot_at = monotonic()
        last_snapshot_len = 0
        with self.harness.genai_trace.invocation(
            session_id=session_id,
            user_message=message,
            run_id=run_id,
            model=self.model,
        ) as genai_span:
            if genai_span:
                state.genai_trace_id = genai_span.trace_id
                state.genai_root_span_id = genai_span.span_id
                state.genai_parent_span_id = genai_span.span_id
            try:
                for event in self._run_loop(state):
                    if isinstance(event, TextEvent) and event.content:
                        streamed_content += event.content
                        now = monotonic()
                        if (
                            len(streamed_content) - last_snapshot_len >= 512
                            or now - last_snapshot_at >= 1.0
                        ):
                            self._save_stream_snapshot(session_id, messages, streamed_content)
                            last_snapshot_at = now
                            last_snapshot_len = len(streamed_content)
                    yield event
                completed = True
                self.sessions.save(session_id, messages)
            except Exception as exc:
                if genai_span:
                    genai_span.set_attribute("error.type", type(exc).__name__)
                    genai_span.finish(status_code=OTLP_STATUS_ERROR, status_message=str(exc))
                raise
            finally:
                if genai_span:
                    output = self._last_assistant_message(messages) or streamed_content
                    if output:
                        genai_span.set_attribute("gen_ai.output.messages", [
                            {
                                "role": "assistant",
                                "parts": [{"type": "text", "content": output}],
                                "finish_reason": "stop" if completed else "error",
                            }
                        ])
                    genai_span.finish(status_code=OTLP_STATUS_OK if completed else OTLP_STATUS_ERROR)
                if not completed and streamed_content:
                    self._save_stream_snapshot(session_id, messages, streamed_content)
                self.harness.clear_tool_cache_namespace(cache_namespace)

    def _run_loop(self, state: RunState) -> Generator[Event, None, None]:
        yield from self.query_loop.run(state)

    def _save_stream_snapshot(self, session_id: str, messages: list[dict],
                              streamed_content: str):
        snapshot = [dict(message) for message in messages]
        if streamed_content:
            if snapshot and snapshot[-1].get("role") == "assistant":
                snapshot[-1]["content"] = streamed_content
            else:
                snapshot.append({"role": "assistant", "content": streamed_content})
        self.sessions.save(session_id, snapshot)

    @staticmethod
    def _last_assistant_message(messages: list[dict]) -> str:
        for message in reversed(messages):
            if message.get("role") == "assistant" and message.get("content"):
                return str(message.get("content") or "")
        return ""

    def _set_pending_confirmation(self, session_id: str, tool_name: str, args: dict,
                                  tool_call_id: str, messages: list[dict],
                                  state: RunState,
                                  skipped_tool_calls: list[dict] | None = None):
        self._pending[session_id] = PendingConfirmation(
            session_id=session_id,
            tool_name=tool_name,
            args=args,
            tool_call_id=tool_call_id,
            messages=messages,
            cache_namespace=state.cache_namespace,
            skipped_tool_calls=skipped_tool_calls,
            user_question=state.user_question,
            allowed_tools=state.allowed_tools,
            turn_count=state.turn_count,
            stop_hook_active=state.stop_hook_active,
            genai_trace_id=state.genai_trace_id,
            genai_root_span_id=state.genai_root_span_id,
            genai_parent_span_id=state.genai_parent_span_id,
        )

    def sessions_save(self, session_id: str, messages: list[dict]):
        self.sessions.save(session_id, messages)

    def chat_stream_sse(self, message: str, session_id: str = "default",
                        allowed_tools: Iterable[str] | None = None,
                        run_id: str = "") -> Generator[dict, None, None]:
        for event in self.chat_stream(message, session_id, allowed_tools=allowed_tools, run_id=run_id):
            yield event_to_dict(event)

    def get_history(self, session_id: str) -> list[dict]:
        messages = self.sessions.get(session_id)
        return [
            {"role": m["role"], "content": m.get("content", "")}
            for m in messages
            if m["role"] in ("user", "assistant") and m.get("content")
        ]

    def get_context_usage(self, session_id: str) -> dict:
        messages = self.sessions.get(session_id)
        return self.harness.collect_context_usage(messages)

    def list_sessions(self) -> list[dict]:
        return self.sessions.list_sessions()
