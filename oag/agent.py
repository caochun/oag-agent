"""面向调用方的会话级 Agent API。

Agent 负责聊天会话、待确认操作、流式/SSE 输出和历史持久化；具体的 LLM
回合循环交给 QueryLoop，工具执行和策略约束交给 Harness。
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

from openai import OpenAI

from .runtime.events import (
    Event, TextEvent, event_to_dict,
)
from .harness import Harness
from .loop.confirmation_flow import ConfirmationFlow
from .loop.query_loop import QueryLoop
from .runtime import PendingConfirmation, RunState
from .runtime.session_store import SessionStore


class Agent:
    def __init__(self, harness: Harness, llm_client: OpenAI, model: str,
                 db_dir: str = ".oag_data"):
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

        Path(db_dir).mkdir(parents=True, exist_ok=True)
        db_path = str(Path(db_dir) / f"chat_{harness.ontology.name}.db")
        self.sessions = SessionStore(db_path)

    def chat(self, message: str, session_id: str = "default") -> str:
        result_parts = []
        for event in self.chat_stream(message, session_id):
            if isinstance(event, TextEvent):
                result_parts.append(event.content)
        return "".join(result_parts)

    def has_pending(self, session_id: str) -> bool:
        return session_id in self._pending

    def confirm_tool(self, session_id: str, approved: bool,
                     answer: str | None = None) -> Generator[Event, None, None]:
        pending = self._pending.pop(session_id, None)
        yield from self.confirmation_flow.confirm(pending, approved, answer)

    def chat_stream(self, message: str, session_id: str = "default") -> Generator[Event, None, None]:
        if session_id in self._pending:
            yield TextEvent(content="当前会话有待确认的操作，请先确认或取消后再继续。")
            return

        messages = self.sessions.get(session_id)

        if not messages:
            system_prompt = self.harness.build_system_prompt()
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": message})

        state = RunState(messages=messages, session_id=session_id, user_question=message)
        yield from self._run_loop(state)
        self.sessions.save(session_id, messages)

    def _run_loop(self, state: RunState) -> Generator[Event, None, None]:
        yield from self.query_loop.run(state)

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
            skipped_tool_calls=skipped_tool_calls,
            user_question=state.user_question,
            turn_count=state.turn_count,
            stop_hook_active=state.stop_hook_active,
        )

    def sessions_save(self, session_id: str, messages: list[dict]):
        self.sessions.save(session_id, messages)

    def chat_stream_sse(self, message: str, session_id: str = "default") -> Generator[dict, None, None]:
        for event in self.chat_stream(message, session_id):
            yield event_to_dict(event)

    def get_history(self, session_id: str) -> list[dict]:
        messages = self.sessions.get(session_id)
        return [
            {"role": m["role"], "content": m.get("content", "")}
            for m in messages
            if m["role"] in ("user", "assistant") and m.get("content")
        ]

    def list_sessions(self) -> list[dict]:
        return self.sessions.list_sessions()
