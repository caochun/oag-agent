"""Independent registrars for ontology-backed agent tools."""

from __future__ import annotations

import json
from typing import Protocol

from ..tools.registry import ToolDef, ToolPolicy, ToolRegistry
from .bindings import RuntimeBindings
from .data_executor import DataExecutor
from .rules import RuleEngine
from .schema import Ontology

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

    def apply_rule(self, tool_name: str, args: dict) -> str: ...


class ReadToolRegistrar:
    """Register generic object, relation, inspect, and search tools."""

    def __init__(self, ontology: Ontology, runtime: OntologyToolRuntime):
        self.ontology = ontology
        self.runtime = runtime

    def register(self, tools: ToolRegistry, data: DataExecutor) -> None:
        object_types = list(self.ontology.objects)
        relation_types = list(self.ontology.relations)

        tools.register(ToolDef(
            name="inspect",
            description="查看函数、业务操作、对象、关系、规则或交互策略的完整定义",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "函数名、Action 名、对象类型名、关系类型名、规则名或交互策略名",
                    },
                },
                "required": ["name"],
            },
            handler=lambda args: self.runtime.inspect(args.get("name", "")),
            category="inspect",
        ))

        tools.register(ToolDef(
            name="get_object",
            description=(
                "按对象类型和完整稳定 ID 精确读取一个对象。ID 必须原样传递，"
                "不得删除类型前缀或改写分隔符"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "object_type": {"type": "string", "enum": object_types},
                    "id": {
                        "type": "string",
                        "description": "完整稳定对象 ID，例如 opportunity:park_ops",
                    },
                },
                "required": ["object_type", "id"],
            },
            handler=lambda args: data.execute("get_object", args),
            category="query",
        ))

        tools.register(ToolDef(
            name="query",
            description=(
                "查询对象实例。filters 支持后缀: __like 模糊, __gt 大于, "
                "__gte 大于等于, __lt 小于, __lte 小于等于, __ne 不等于, "
                "__in 属于列表。查询一个已知 ID 时优先使用 get_object"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "object_type": {"type": "string", "enum": object_types},
                    "filters": {
                        "type": "object",
                        "description": (
                            "过滤条件。数组值必须使用字段名__in，例如 "
                            "{\"id__in\": [\"party:a\", \"party:b\"]}"
                        ),
                    },
                    "order_by": {"type": "string", "description": "排序字段，-前缀降序"},
                    "limit": {"type": "integer"},
                    "offset": {"type": "integer"},
                },
                "required": ["object_type"],
            },
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

        searchable_types = self.runtime.repository.searchable_object_types()
        if searchable_types:
            tools.register(ToolDef(
                name="search",
                description="跨对象类型全文搜索。在所有（或指定）对象类型的文本字段中搜索关键词",
                parameters={
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string", "description": "搜索关键词"},
                        "object_types": {
                            "type": "array",
                            "items": {"type": "string", "enum": searchable_types},
                            "description": "限定搜索的对象类型（可选，不填搜索全部）",
                        },
                        "limit": {"type": "integer", "description": "最大返回条数（默认20）"},
                    },
                    "required": ["keyword"],
                },
                handler=lambda args: data.execute("search", args),
                category="query",
            ))


class RuleToolRegistrar:
    """Register deterministic rule tools when the ontology declares rules."""

    def __init__(self, ontology: Ontology, rule_engine: RuleEngine | None,
                 runtime: OntologyToolRuntime):
        self.ontology = ontology
        self.rule_engine = rule_engine
        self.runtime = runtime

    def register(self, tools: ToolRegistry) -> None:
        if self.rule_engine is None:
            return

        rule_names = list(self.ontology.rules)
        applicable_types = sorted({
            object_type
            for rule in self.ontology.rules.values()
            for object_type in rule.applies_to
        })
        tools.register(ToolDef(
            name="apply_rule",
            description="对指定对象应用业务规则，返回确定性结果（无需 LLM 推理）",
            parameters={
                "type": "object",
                "properties": {
                    "rule_name": {"type": "string", "enum": rule_names},
                    "object_type": {"type": "string", "enum": applicable_types},
                    "object_id": {"type": "string", "description": "对象 ID"},
                },
                "required": ["rule_name", "object_type", "object_id"],
            },
            handler=lambda args: self.runtime.apply_rule("apply_rule", args),
            category="rule",
        ))
        tools.register(ToolDef(
            name="apply_rule_batch",
            description="批量应用规则到多个对象",
            parameters={
                "type": "object",
                "properties": {
                    "rule_name": {"type": "string", "enum": rule_names},
                    "object_type": {"type": "string", "enum": applicable_types},
                    "filters": {"type": "object", "description": "过滤条件（同 query）"},
                },
                "required": ["rule_name", "object_type"],
            },
            handler=lambda args: self.runtime.apply_rule("apply_rule_batch", args),
            category="rule",
        ))


class FunctionToolRegistrar:
    """Register provider-bound, side-effect-free ontology Functions."""

    def __init__(self, bindings: RuntimeBindings):
        self.bindings = bindings

    def register(self, tools: ToolRegistry, data: DataExecutor) -> None:
        for name, definition in self.bindings.list_functions():
            properties = {
                parameter_name: {
                    "type": JSON_SCHEMA_TYPE_MAP.get(parameter.type, "string"),
                    "description": parameter.description,
                }
                for parameter_name, parameter in definition.params.items()
            }
            required = [
                parameter_name
                for parameter_name, parameter in definition.params.items()
                if "default" not in parameter.model_fields_set
            ]
            tools.register(ToolDef(
                name=name,
                description=(definition.summary or definition.description).strip(),
                parameters={"type": "object", "properties": properties, "required": required},
                handler=lambda args, function_name=name: data.execute(function_name, args),
                usage_prompt=definition.usage_prompt,
                category="query",
                max_result_chars=12000,
                policy=ToolPolicy(
                    read_only=True,
                    concurrency_safe=(
                        True if definition.concurrency_safe is None
                        else definition.concurrency_safe
                    ),
                    worker_allowed=True,
                    timeout_seconds=definition.timeout_seconds,
                ),
            ))


class ActionToolRegistrar:
    """Register the generic Action catalog and interaction bridge."""

    def __init__(self, ontology: Ontology, bindings: RuntimeBindings):
        self.ontology = ontology
        self.bindings = bindings

    def register(self, tools: ToolRegistry) -> None:
        action_runtime = self.bindings.get_action_runtime()
        if not self.ontology.actions or action_runtime is None:
            return

        visible_actions = {
            action_id: action
            for action_id, action in self.ontology.actions.items()
            if action.user_visible
        }
        if not visible_actions:
            return

        action_ids = list(visible_actions)
        catalog = "；".join(
            f"{action_id}={action.display_name}"
            for action_id, action in visible_actions.items()
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
            policy=ToolPolicy(worker_allowed=False),
        ))
        tools.register(ToolDef(
            name="request_action_input",
            description=(
                "请求应用展示模型驱动的业务操作输入界面，不执行业务数据变更。"
                f"\n\n当前可用 action_id：{catalog}"
            ),
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
            category="interaction",
            max_result_chars=2000,
            policy=ToolPolicy(
                read_only=True,
                concurrency_safe=False,
                worker_allowed=False,
                timeout_seconds=5.0,
            ),
        ))

    @staticmethod
    def _open_action_form(action_runtime, args: dict) -> str:
        prepared = action_runtime.prepare_action(
            action_id=str(args.get("action_id", "")),
            context_id=str(args.get("context_id", "")),
            initial_inputs=args.get("initial_inputs") or {},
        )
        action = prepared.get("action") or {}
        return json.dumps({
            "message": f"已向用户打开{action.get('name', '业务操作')}表单，等待用户填写并确认。",
            "interaction": {"kind": "action_form", **prepared},
        }, ensure_ascii=False, default=str)
