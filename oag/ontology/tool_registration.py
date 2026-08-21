"""本体相关工具注册器。

把对象查询、统计分析、mutate、规则、工作流和领域函数转换成 ToolDef。
这里同时决定工具策略：是否只读、是否需要确认、是否允许 worker 执行等。
"""

from __future__ import annotations

import json
from typing import Protocol

from .data_executor import DataExecutor
from .registry import FunctionRegistry
from .rules import RuleEngine
from .schema import Ontology
from ..tools.registry import ToolDef, ToolPolicy, ToolRegistry


JSON_SCHEMA_TYPE_MAP = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "dict": "object",
    "object": "object",
    "list": "array",
    "array": "array",
}


class OntologyToolRuntime(Protocol):
    def inspect(self, target: str) -> str: ...
    def start_workflow(self, args: dict) -> str: ...
    def check_sla(self, args: dict) -> str: ...
    def apply_rule(self, tool_name: str, args: dict) -> str: ...


class OntologyToolRegistrar:
    """Registers ontology/data/business tools into the harness tool registry."""

    def __init__(self, ontology: Ontology, registry: FunctionRegistry,
                 rule_engine: RuleEngine | None, runtime: OntologyToolRuntime,
                 enable_analysis_tools: bool = False):
        self.ontology = ontology
        self.registry = registry
        self.rule_engine = rule_engine
        self.runtime = runtime
        self.enable_analysis_tools = enable_analysis_tools

    def register_tools(self, tools: ToolRegistry, data: DataExecutor):
        obj_types = list(self.ontology.objects.keys())
        relation_types = list(self.ontology.relations.keys())

        tools.register(ToolDef(
            name="inspect", description="查看函数、业务操作、对象、规则、展示工具或策略的完整定义",
            parameters={"type": "object", "properties": {"name": {"type": "string", "description": "函数名、Action 名、对象类型名、规则名、展示工具名、事件类型或交互策略名"}}, "required": ["name"]},
            handler=lambda args: self.runtime.inspect(args.get("name", "")),
            category="inspect",
        ))

        tools.register(ToolDef(
            name="query",
            description="查询对象实例。filters支持后缀: __like模糊, __gt大于, __gte大于等于, __lt小于, __lte小于等于, __ne不等于",
            parameters={"type": "object", "properties": {"object_type": {"type": "string", "enum": obj_types}, "filters": {"type": "object", "description": "过滤条件"}, "order_by": {"type": "string", "description": "排序字段，-前缀降序"}, "limit": {"type": "integer"}, "offset": {"type": "integer"}}, "required": ["object_type"]},
            handler=lambda args: data.execute("query", args),
            category="query",
        ))

        if relation_types:
            tools.register(ToolDef(
                name="query_relations",
                description="查询本体关系实例，可按起点、终点和方向过滤",
                parameters={
                    "type": "object",
                    "properties": {
                        "relation_type": {"type": "string", "enum": relation_types},
                        "filters": {"type": "object"},
                        "from_id": {"type": "string", "description": "起点对象 ID"},
                        "to_id": {"type": "string", "description": "终点对象 ID"},
                        "direction": {"type": "string", "enum": ["out", "in", "both"]},
                        "order_by": {"type": "string"},
                        "limit": {"type": "integer"},
                        "offset": {"type": "integer"},
                    },
                    "required": ["relation_type"],
                },
                handler=lambda args: data.execute("query_relations", args),
                category="query",
            ))

        tools.register(ToolDef(
            name="count", description="统计对象数量",
            parameters={"type": "object", "properties": {"object_type": {"type": "string", "enum": obj_types}, "filters": {"type": "object"}}, "required": ["object_type"]},
            handler=lambda args: data.execute("count", args),
            category="query",
        ))

        tools.register(ToolDef(
            name="describe", description="统计摘要",
            parameters={"type": "object", "properties": {"object_type": {"type": "string", "enum": obj_types}, "column": {"type": "string"}}, "required": ["object_type"]},
            handler=lambda args: data.execute("describe", args),
            category="analysis",
        ))

        if self.enable_analysis_tools:
            tools.register(ToolDef(
                name="pivot", description="透视表分析",
                parameters={"type": "object", "properties": {"object_type": {"type": "string", "enum": obj_types}, "index": {"type": "string"}, "columns": {"type": "string"}, "values": {"type": "string"}, "aggfunc": {"type": "string", "enum": ["mean", "sum", "count", "min", "max"]}}, "required": ["object_type", "index", "columns", "values"]},
                handler=lambda args: data.execute("pivot", args),
                category="analysis",
            ))

            tools.register(ToolDef(
                name="distribution", description="分布直方图",
                parameters={"type": "object", "properties": {"object_type": {"type": "string", "enum": obj_types}, "column": {"type": "string"}, "bins": {"type": "integer"}}, "required": ["object_type", "column"]},
                handler=lambda args: data.execute("distribution", args),
                category="analysis",
            ))

        tools.register(ToolDef(
            name="mutate",
            description="创建/更新/删除对象实例。写操作需要用户确认。object_id 使用业务主键（如 event_id、drone_id），不是内部 _id。如果不确定字段名，先用 inspect 查看对象定义",
            parameters={"type": "object", "properties": {"operation": {"type": "string", "enum": ["create", "update", "delete"], "description": "操作类型"}, "object_type": {"type": "string", "enum": obj_types, "description": "对象类型"}, "object_id": {"type": "string", "description": "对象ID（update/delete必填）"}, "data": {"type": "object", "description": "要写入的字段（create/update时提供）"}}, "required": ["operation", "object_type"]},
            handler=lambda args: data.execute("mutate", args),
            category="action", is_read_only=False, requires_confirmation=True, max_result_chars=2000,
            policy=ToolPolicy(
                read_only=False,
                requires_confirmation=True,
                concurrency_safe=False,
                worker_allowed=False,
                idempotent=False,
                destructive=True,
            ),
        ))

        tools.register(ToolDef(
            name="search",
            description=self._generic_search_description(),
            parameters={"type": "object", "properties": {"keyword": {"type": "string", "description": "搜索关键词"}, "object_types": {"type": "array", "items": {"type": "string", "enum": obj_types}, "description": "限定搜索的对象类型（可选，不填搜索全部）"}, "limit": {"type": "integer", "description": "最大返回条数（默认20）"}}, "required": ["keyword"]},
            handler=lambda args: data.execute("search", args),
            category="query",
        ))

        workflow_names = list(self.ontology.workflows.keys()) if self.ontology.workflows else []
        if workflow_names:
            tools.register(ToolDef(
                name="start_workflow",
                description="启动或推进工作流。返回工作流定义、当前步骤和下一步指引",
                parameters={"type": "object", "properties": {"workflow_name": {"type": "string", "enum": workflow_names, "description": "工作流名称"}, "advance_to_step": {"type": "string", "description": "推进到指定步骤名（可选）"}}, "required": ["workflow_name"]},
                handler=self.runtime.start_workflow,
                category="action",
                policy=ToolPolicy(
                    read_only=False,
                    requires_confirmation=False,
                    concurrency_safe=False,
                    worker_allowed=False,
                    idempotent=False,
                ),
            ))

        has_sla = any(
            fdef and fdef.temporal_constraints
            for _, fdef in self.registry.list_functions()
        ) or any(
            step.sla for wdef in self.ontology.workflows.values() for step in wdef.steps
        )
        if has_sla:
            tools.register(ToolDef(
                name="check_sla",
                description="检查当前领域中定义的所有时间约束和SLA。返回各函数和工作流步骤的 deadline/SLA 定义，用于判断是否超时",
                parameters={"type": "object", "properties": {"event_id": {"type": "string", "description": "事件编号（可选，用于上下文）"}}, "required": []},
                handler=self.runtime.check_sla,
                category="query",
            ))

        if self.rule_engine:
            rule_names = list(self.ontology.rules.keys())
            applicable_types = sorted({t for r in self.ontology.rules.values() for t in r.applies_to})

            tools.register(ToolDef(
                name="apply_rule", description="对指定对象应用业务规则，返回确定性结果（无需LLM推理）",
                parameters={"type": "object", "properties": {"rule_name": {"type": "string", "description": "规则名称", "enum": rule_names}, "object_type": {"type": "string", "description": "对象类型", "enum": applicable_types}, "object_id": {"type": "string", "description": "对象ID"}}, "required": ["rule_name", "object_type", "object_id"]},
                handler=lambda args: self.runtime.apply_rule("apply_rule", args),
                category="rule",
            ))

            tools.register(ToolDef(
                name="apply_rule_batch", description="批量应用规则到多个对象",
                parameters={"type": "object", "properties": {"rule_name": {"type": "string", "enum": rule_names}, "object_type": {"type": "string", "enum": applicable_types}, "filters": {"type": "object", "description": "过滤条件（同 query）"}}, "required": ["rule_name", "object_type"]},
                handler=lambda args: self.runtime.apply_rule("apply_rule_batch", args),
                category="rule",
            ))

        self._register_action_tools(tools)

        for name, fdef in self.registry.list_functions():
            if not fdef:
                continue
            props = {}
            required = []
            for pname, pdef in fdef.params.items():
                props[pname] = {
                    "type": JSON_SCHEMA_TYPE_MAP.get(pdef.type, "string"),
                    "description": pdef.description,
                }
                if pdef.default is None:
                    required.append(pname)

            fn_name = name
            tools.register(ToolDef(
                name=fn_name,
                description=(fdef.summary or fdef.description or "").strip(),
                parameters={"type": "object", "properties": props, "required": required},
                handler=lambda args, _n=fn_name: data.execute(_n, args),
                usage_prompt=fdef.usage_prompt or fdef.hint,
                category="query",
                is_read_only=True,
                requires_confirmation=False,
                max_result_chars=12000,
                policy=ToolPolicy(
                    read_only=True,
                    requires_confirmation=False,
                    concurrency_safe=(
                        True if fdef.concurrency_safe is None else fdef.concurrency_safe
                    ),
                    worker_allowed=True,
                    idempotent=True,
                    destructive=False,
                    timeout_seconds=fdef.timeout_seconds,
                ),
            ))

    def _register_action_tools(self, tools: ToolRegistry) -> None:
        action_runtime = self.registry.get_action_runtime()
        if not self.ontology.actions or action_runtime is None:
            return

        action_ids = [
            action_id
            for action_id, definition in self.ontology.actions.items()
            if definition.user_visible
        ]
        if not action_ids:
            return
        catalog = "；".join(
            f"{action_id}={definition.display_name}"
            for action_id, definition in self.ontology.actions.items()
            if definition.user_visible
        )
        tools.register(ToolDef(
            name="get_available_actions",
            description="获取当前上下文允许执行的模型化业务操作及其前置条件状态。",
            parameters={
                "type": "object",
                "properties": {
                    "context_id": {
                        "type": "string",
                        "description": "可选的当前业务对象 ID",
                    },
                },
                "required": [],
            },
            handler=lambda args: json.dumps(
                action_runtime.list_actions(str(args.get("context_id", ""))),
                ensure_ascii=False,
                default=str,
            ),
            category="query",
        ))

        presentation = self.ontology.presentation_tools.get("ui_open_action_form")
        description = (
            presentation.description
            if presentation
            else "在前端打开模型驱动的业务操作表单，不执行业务数据变更。"
        )
        tools.register(ToolDef(
            name="ui_open_action_form",
            description=f"{description.strip()}\n\n当前可用 action_id：{catalog}",
            usage_prompt=presentation.usage_prompt if presentation else "",
            parameters={
                "type": "object",
                "properties": {
                    "action_id": {
                        "type": "string",
                        "enum": action_ids,
                        "description": "领域本体定义的业务操作 ID",
                    },
                    "context_id": {
                        "type": "string",
                        "description": "操作依赖当前对象时提供其 ID",
                    },
                    "initial_inputs": {
                        "type": "object",
                        "description": "用户已经明确给出的可选预填值",
                    },
                },
                "required": ["action_id"],
            },
            handler=lambda args: self._open_action_form(action_runtime, args),
            category=presentation.category if presentation else "ui",
            max_result_chars=2000,
            policy=ToolPolicy(
                read_only=True,
                requires_confirmation=False,
                concurrency_safe=False,
                worker_allowed=False,
                idempotent=False,
                destructive=False,
                timeout_seconds=presentation.timeout_seconds if presentation else 5.0,
            ),
        ))

    @staticmethod
    def _open_action_form(action_runtime, args: dict) -> str:
        try:
            prepared = action_runtime.prepare_action(
                action_id=str(args.get("action_id", "")),
                context_id=str(args.get("context_id", "")),
                initial_inputs=args.get("initial_inputs") or {},
            )
        except Exception as exc:
            return json.dumps({
                "error": "无法打开业务操作表单",
                "details": str(exc),
                "errors": getattr(exc, "errors", []),
            }, ensure_ascii=False)
        action = prepared.get("action") or {}
        return json.dumps({
            "message": f"已向用户打开{action.get('name', '业务操作')}表单，等待用户填写并确认。",
            "presentation": {"kind": "action_form", **prepared},
        }, ensure_ascii=False, default=str)

    def _generic_search_description(self) -> str:
        description = "跨对象类型全文搜索。在所有（或指定）对象类型的文本字段中搜索关键词"
        if self.ontology.tool_preferences.get("generic_search") == "fallback_only":
            description += (
                "。本领域将通用 search 作为兜底定位工具：当存在领域专用检索/问答函数时，"
                "优先使用领域函数；不要仅根据通用 search 的片段直接回答文档正文内容。"
            )
        return description
