"""Harness 组件装配入口。

这里是在线执行的 composition root：创建 hooks/audit、OntologyRuntime、
DataExecutor、ToolRegistry、ToolExecutionPipeline、RuntimeTools 和 TraceRecorder。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from openai import OpenAI

from ..llm.context import ContextManager
from ..ontology.data_executor import DataExecutor
from ..ontology.registry import FunctionRegistry
from ..ontology.rules import RuleEngine
from ..ontology.runtime import OntologyRuntime
from ..ontology.schema import Ontology
from ..ontology.store import Store
from ..tools.pipeline import ToolExecutionPipeline, ToolResult
from ..tools.registry import ToolRegistry
from ..tools.runtime_tools import RuntimeTools
from .config import HarnessConfig
from .hooks import AuditLog, HookRegistry, audit_log_hook, business_review_hook, write_confirmation_hook
from .stop_check import default_stop_hook
from .trace import TraceRecorder


GetMessages = Callable[[], list[dict] | None]
SetMessages = Callable[[list[dict] | None], None]
DispatchWorkers = Callable[[list[str], str], list[dict]]


@dataclass
class HarnessComponents:
    hooks: HookRegistry
    audit: AuditLog
    rule_engine: RuleEngine | None
    context_mgr: ContextManager
    ont: OntologyRuntime
    data: DataExecutor
    tools: ToolRegistry
    cache: dict[str, ToolResult]
    trace: TraceRecorder
    tool_pipeline: ToolExecutionPipeline
    runtime_tools: RuntimeTools


def build_harness_components(
    ontology: Ontology,
    store: Store,
    registry: FunctionRegistry,
    llm_client: OpenAI,
    model: str,
    config: HarnessConfig,
    *,
    set_current_messages: SetMessages,
    get_current_messages: GetMessages,
    dispatch_workers: DispatchWorkers,
) -> HarnessComponents:
    hooks = HookRegistry()
    audit = AuditLog()
    rule_engine = RuleEngine(ontology, store) if ontology.rules else None
    context_mgr = ContextManager(llm_client, model)
    ont = OntologyRuntime(ontology, store, registry, rule_engine)
    data = DataExecutor(store, registry)
    tools = ToolRegistry()
    cache: dict[str, ToolResult] = {}
    trace = TraceRecorder()
    # 工具 handler 尽量保持简单；统一的策略、校验、缓存、审计都在 pipeline 中完成。
    tool_pipeline = ToolExecutionPipeline(
        tools=tools,
        ontology_runtime=ont,
        hooks=hooks,
        audit=audit,
        cache=cache,
        trace=trace,
        set_current_messages=set_current_messages,
    )
    runtime_tools = RuntimeTools(
        context_mgr=context_mgr,
        get_current_messages=get_current_messages,
        dispatch_workers=dispatch_workers,
    )

    ont.register_tools(tools, data)
    runtime_tools.register(tools)
    register_default_hooks(hooks, config)

    return HarnessComponents(
        hooks=hooks,
        audit=audit,
        rule_engine=rule_engine,
        context_mgr=context_mgr,
        ont=ont,
        data=data,
        tools=tools,
        cache=cache,
        trace=trace,
        tool_pipeline=tool_pipeline,
        runtime_tools=runtime_tools,
    )


def register_default_hooks(hooks: HookRegistry, config: HarnessConfig):
    if config.enable_write_confirmation:
        hooks.register("pre_tool_call", write_confirmation_hook)
    if config.enable_audit:
        hooks.register("post_tool_call", audit_log_hook)
    hooks.register("post_tool_call", business_review_hook)
    hooks.register("query_complete", default_stop_hook)
