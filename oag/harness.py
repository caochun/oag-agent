"""运行时 Harness 门面。

Harness 是模型外侧的执行边界：构建 prompt 和工具、通过工具管线执行调用、
持有 hooks/audit/trace，并向对话循环提供压缩和最终回答检查能力。
"""

from __future__ import annotations

import logging
from datetime import datetime

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
        self._static_prompt_cache: dict[str, list[str]] = {}
        self._tools_cache_version = -1
        self._tools_cache: list[dict] | None = None

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
        if (
            self._tools_cache is not None
            and self._tools_cache_version == self.tools.version
        ):
            return self._tools_cache
        self._tools_cache = self.tools.build_tools()
        self._tools_cache_version = self.tools.version
        return self._tools_cache

    def build_system_prompt(self, domain_context: str = "") -> str:
        sections = self.build_system_prompt_sections(domain_context)
        return "\n\n".join(sections)

    def build_system_prompt_sections(self, domain_context: str = "") -> list[str]:
        sections = self.build_static_prompt_sections(domain_context)

        runtime_context = self.build_runtime_context()
        if runtime_context:
            sections.append(runtime_context)

        if self.config.include_ontology_full_context:
            full_context = self.ont.build_full_context()
            if full_context:
                sections.append(full_context)

        if self.config.append_system_prompt.strip():
            sections.append(self.config.append_system_prompt.strip())

        return sections

    def build_static_prompt_sections(self, domain_context: str = "") -> list[str]:
        if domain_context in self._static_prompt_cache:
            return list(self._static_prompt_cache[domain_context])

        if self.config.custom_system_prompt is not None:
            base = self.config.custom_system_prompt.strip()
            sections = [base] if base else []
            sections.extend(self.ont.build_static_sections(domain_context)[1:])
        else:
            sections = self.ont.build_static_sections(domain_context)

        self._static_prompt_cache[domain_context] = list(sections)
        return list(sections)

    def build_runtime_context(self) -> str:
        lines = [
            "## 运行时上下文",
            f"- session_time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
            f"- mode: {'write_confirmation' if self.config.enable_write_confirmation else 'no_write_confirmation'}",
            f"- audit: {'enabled' if self.config.enable_audit else 'disabled'}",
            f"- max_turns: {self.config.max_turns}",
            "- ontology_details: 摘要常驻；完整函数、对象、规则定义请调用 inspect 获取",
        ]
        for key, value in self.config.runtime_context.items():
            clean_key = str(key).strip()
            clean_value = str(value).strip()
            if clean_key and clean_value:
                lines.append(f"- {clean_key}: {clean_value}")
        return "\n".join(lines)

    def build_worker_system_prompt(self, worker_id: str, context: str = "") -> str:
        sections = [
            f"你是 Worker {worker_id}，负责执行一个具体子任务。",
            self.ont.build_base_system_prompt(),
            self.ont.build_ontology_summary(),
            "## 背景信息（主 Agent 已获取）\n" + (context or "(无)"),
            "## 要求\n- 直接执行任务，不要重复查询主 Agent 已提供的信息\n- 需要完整定义时调用 inspect，不要依赖主 Agent 的完整历史\n- 完成后用 1-3 句话总结关键结果\n- 包含具体数据（等级、数值、状态）",
        ]
        return "\n\n".join(section for section in sections if section.strip())

    def build_ontology_full_context(self) -> str:
        return self.ont.build_full_context()

    def maybe_compact(self, messages: list[dict]) -> tuple[list[dict], bool]:
        return self.context_mgr.maybe_compact(messages)

    def force_compact(self, messages: list[dict]) -> tuple[list[dict], bool]:
        return self.context_mgr.force_compact(messages)

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
