from __future__ import annotations

import logging

from openai import OpenAI

from .runtime import HarnessConfig, ToolUseContext
from .runtime.components import build_harness_components
from .tools.pipeline import ToolResult
from .loop.worker import run_workers_parallel
from .ontology.registry import FunctionRegistry
from .ontology.schema import Ontology
from .ontology.store import Store

logger = logging.getLogger(__name__)


class Harness:
    def __init__(self, ontology: Ontology, store: Store,
                 registry: FunctionRegistry, llm_client: OpenAI,
                 model: str, config: HarnessConfig | None = None):
        self.ontology = ontology
        self.config = config or HarnessConfig()
        self._current_messages: list[dict] | None = None
        components = build_harness_components(
            ontology,
            store,
            registry,
            llm_client,
            model,
            self.config,
            set_current_messages=self._set_current_messages,
            get_current_messages=self._get_current_messages,
            dispatch_workers=self._dispatch_workers,
        )
        self.hooks = components.hooks
        self.audit = components.audit
        self.rule_engine = components.rule_engine
        self.context_mgr = components.context_mgr
        self.ont = components.ont
        self.data = components.data
        self.tools = components.tools
        self._cache = components.cache
        self.trace = components.trace
        self.tool_pipeline = components.tool_pipeline
        self.runtime_tools = components.runtime_tools

    def register_stop_hook(self, handler):
        self.hooks.register("query_complete", handler)

    def execute_tool(self, tool_name: str, args: dict,
                     session_id: str = "",
                     confirmed: bool = False,
                     messages: list[dict] | None = None,
                     context: ToolUseContext | None = None) -> ToolResult:
        context = self._normalize_tool_context(session_id, confirmed, messages, context)
        return self.tool_pipeline.execute(tool_name, args, context)

    def _normalize_tool_context(self, session_id: str, confirmed: bool,
                                messages: list[dict] | None,
                                context: ToolUseContext | None) -> ToolUseContext:
        if context:
            return context
        return ToolUseContext(
            session_id=session_id,
            messages=messages,
            confirmed=confirmed,
        )

    def _set_current_messages(self, messages: list[dict] | None):
        self._current_messages = messages

    def _get_current_messages(self) -> list[dict] | None:
        return self._current_messages

    def _dispatch_workers(self, tasks: list[str], context: str) -> list[dict]:
        return run_workers_parallel(
            self,
            self.context_mgr.client,
            self.context_mgr.model,
            tasks,
            context=context,
            max_workers=min(len(tasks), 4),
        )

    def build_tools(self) -> list[dict]:
        return self.tools.build_tools()

    def build_system_prompt(self, domain_context: str = "") -> str:
        prompt = self.ont.build_system_prompt(
            domain_context,
            progressive_context=self.config.enable_progressive_context,
        )
        if not self.config.enable_progressive_context:
            prompt += "\n\n" + self.ont.build_full_context()
        return prompt

    def maybe_compact(self, messages: list[dict]) -> tuple[list[dict], bool]:
        messages, compacted = self.context_mgr.maybe_compact(messages)
        if compacted and self.config.enable_progressive_context:
            self.ont.reset_context_shown()
        return messages, compacted

    def run_stop_check(self, user_question: str, messages: list[dict]) -> str | None:
        result = self.hooks.fire("query_complete", {
            "messages": messages,
            "user_question": user_question,
        })
        if result.action == "pause" and result.reason:
            return (
                f"[系统自检] 请检查你的回复是否完整回答了用户问题: \"{user_question}\"\n"
                f"发现的问题: {result.reason}\n"
                f"如果回复已完整，直接说'已确认回复完整'即可。否则请补充。"
            )
        return None
