"""Harness 组件装配入口。

这里是在线执行的 composition root：创建 hooks/audit、OntologyRuntime、
DataExecutor、ToolRegistry、ToolExecutionPipeline、RuntimeTools 和 TraceRecorder。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from openai import OpenAI

from ..llm.context import ContextManager
from ..ontology.bindings import RuntimeBindings
from ..ontology.data_executor import DataExecutor
from ..ontology.repository import OntologyRepository
from ..ontology.rules import RuleEngine
from ..ontology.runtime import OntologyRuntime
from ..ontology.schema import Ontology
from ..tools.pipeline import ToolExecutionPipeline, ToolResult
from ..tools.registry import ToolRegistry
from ..tools.runtime_tools import RuntimeTools
from .config import HarnessConfig
from .hooks import AuditLog, HookRegistry, audit_log_hook, write_confirmation_hook
from .tool_result_store import ToolResultStore
from .trace import TraceRecorder

DispatchWorkers = Callable[[list[str], str], list[dict]]


@dataclass
class HarnessComponents:
    hooks: HookRegistry
    audit: AuditLog
    rule_engine: RuleEngine | None
    repository: OntologyRepository
    context_mgr: ContextManager
    ont: OntologyRuntime
    data: DataExecutor
    tools: ToolRegistry
    cache: dict[str, ToolResult]
    trace: TraceRecorder
    result_store: ToolResultStore
    tool_pipeline: ToolExecutionPipeline


def build_harness_components(
    ontology: Ontology,
    repository: OntologyRepository,
    bindings: RuntimeBindings,
    llm_client: OpenAI,
    model: str,
    config: HarnessConfig,
    *,
    dispatch_workers: DispatchWorkers | None = None,
) -> HarnessComponents:
    hooks = HookRegistry()
    audit = AuditLog()
    rule_engine = RuleEngine(ontology, repository) if ontology.rules else None
    context_mgr = ContextManager(llm_client, model)
    ont = OntologyRuntime(
        ontology,
        bindings,
        repository=repository,
        rule_engine=rule_engine,
    )
    data = DataExecutor(repository, bindings)
    tools = ToolRegistry()
    cache: dict[str, ToolResult] = {}
    trace = TraceRecorder(jsonl_path=config.trace_jsonl_path)
    result_store = ToolResultStore()
    # 工具 handler 尽量保持简单；统一的策略、校验、缓存、审计都在 pipeline 中完成。
    tool_pipeline = ToolExecutionPipeline(
        tools=tools,
        ontology_runtime=ont,
        hooks=hooks,
        audit=audit,
        cache=cache,
        trace=trace,
        result_store=result_store,
        persist_large_results=config.enable_tool_result_reader,
    )
    runtime_tools = RuntimeTools(
        dispatch_workers=dispatch_workers if config.enable_worker_dispatch else None,
        result_store=result_store if config.enable_tool_result_reader else None,
    )

    ont.register_tools(tools, data)
    runtime_tools.register(tools)
    ont.set_available_tools({
        item["function"]["name"]
        for item in tools.build_tools()
    })
    register_default_hooks(hooks, config)

    return HarnessComponents(
        hooks=hooks,
        audit=audit,
        rule_engine=rule_engine,
        repository=repository,
        context_mgr=context_mgr,
        ont=ont,
        data=data,
        tools=tools,
        cache=cache,
        trace=trace,
        result_store=result_store,
        tool_pipeline=tool_pipeline,
    )


def register_default_hooks(hooks: HookRegistry, config: HarnessConfig):
    if config.enable_write_confirmation:
        hooks.register("pre_tool_call", write_confirmation_hook)
    if config.enable_audit:
        hooks.register("post_tool_call", audit_log_hook)
