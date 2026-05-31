"""本体相关工具注册器。

把对象查询、统计分析、mutate、规则、工作流和领域函数转换成 ToolDef。
这里同时决定工具策略：是否只读、是否需要确认、是否允许 worker 执行等。
"""

from __future__ import annotations

from typing import Protocol

from .data_executor import DataExecutor
from .registry import FunctionRegistry
from .rules import RuleEngine
from .schema import Ontology
from ..tools.registry import ToolDef, ToolPolicy, ToolRegistry


class OntologyToolRuntime(Protocol):
    def inspect(self, target: str) -> str: ...
    def start_workflow(self, args: dict) -> str: ...
    def check_sla(self, args: dict) -> str: ...
    def apply_rule(self, tool_name: str, args: dict) -> str: ...


class OntologyToolRegistrar:
    """Registers ontology/data/business tools into the harness tool registry."""

    def __init__(self, ontology: Ontology, registry: FunctionRegistry,
                 rule_engine: RuleEngine | None, runtime: OntologyToolRuntime):
        self.ontology = ontology
        self.registry = registry
        self.rule_engine = rule_engine
        self.runtime = runtime

    def register_tools(self, tools: ToolRegistry, data: DataExecutor):
        obj_types = list(self.ontology.objects.keys())

        tools.register(ToolDef(
            name="inspect", description="查看函数/对象/规则的完整定义",
            parameters={"type": "object", "properties": {"name": {"type": "string", "description": "函数名、对象类型名或规则名"}}, "required": ["name"]},
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

        tools.register(ToolDef(
            name="count", description="统计对象数量",
            parameters={"type": "object", "properties": {"object_type": {"type": "string", "enum": obj_types}, "filters": {"type": "object"}}, "required": ["object_type"]},
            handler=lambda args: data.execute("count", args),
            category="query",
        ))

        if self.ontology.links:
            tools.register(ToolDef(
                name="query_links", description="沿关系查询关联实例",
                parameters={"type": "object", "properties": {"source_type": {"type": "string"}, "source_id": {"type": "string"}, "link_name": {"type": "string", "enum": list(self.ontology.links.keys())}}, "required": ["source_type", "source_id", "link_name"]},
                handler=lambda args: data.execute("query_links", args),
                category="query",
            ))

        tools.register(ToolDef(
            name="describe", description="统计摘要",
            parameters={"type": "object", "properties": {"object_type": {"type": "string", "enum": obj_types}, "column": {"type": "string"}}, "required": ["object_type"]},
            handler=lambda args: data.execute("describe", args),
            category="analysis",
        ))

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
            description="跨对象类型全文搜索。在所有（或指定）对象类型的文本字段中搜索关键词",
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

        for name, fdef in self.registry.list_functions():
            if not fdef:
                continue
            props = {}
            required = []
            for pname, pdef in fdef.params.items():
                props[pname] = {
                    "type": pdef.type if pdef.type in ("string", "integer", "number") else "string",
                    "description": pdef.description,
                }
                if pdef.default is None:
                    required.append(pname)

            has_writes = bool(fdef.writes_to)
            is_business = fdef.function_type == "business"
            fn_name = name
            tools.register(ToolDef(
                name=fn_name,
                description=(fdef.summary or fdef.description or "").strip(),
                parameters={"type": "object", "properties": props, "required": required},
                handler=lambda args, _n=fn_name: data.execute(_n, args),
                usage_prompt=fdef.usage_prompt or fdef.hint,
                category="action" if has_writes else "query",
                is_read_only=not has_writes,
                requires_confirmation=has_writes or is_business,
                policy=ToolPolicy(
                    read_only=not has_writes,
                    requires_confirmation=has_writes or is_business,
                    concurrency_safe=not has_writes,
                    worker_allowed=not (has_writes or is_business),
                    idempotent=not has_writes,
                    destructive=has_writes or is_business,
                ),
            ))
