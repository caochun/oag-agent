"""本体 YAML schema。

这里的 Pydantic 模型定义对象、关系、函数、规则和工作流的配置契约，
并提供业务 ID 字段、表名、适用规则等常用派生信息。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, model_validator


STANDARD_RUNTIME_TOOLS = frozenset({
    "inspect",
    "query",
    "count",
    "query_links",
    "describe",
    "pivot",
    "distribution",
    "mutate",
    "search",
    "apply_rule",
    "apply_rule_batch",
    "start_workflow",
    "check_sla",
    "summarize_progress",
    "ask_user",
    "read_tool_result",
    "dispatch_workers",
})


class PropertyDef(BaseModel):
    type: str = "str"
    required: bool = False
    description: str = ""
    default: Any = None


class ObjectConstraint(BaseModel):
    when: dict[str, Any]
    excluded_functions: list[str] = []
    reason: str = ""


class ObjectSourceDef(BaseModel):
    type: str = "table"  # table / resolver / json / http / sql
    table: str = ""
    resolver: str = ""
    id_field: str = ""
    capabilities: list[str] = []
    config: dict[str, Any] = {}


class ObjectAliasDef(BaseModel):
    terms: list[str] = []
    filters: dict[str, Any] = {}


class ObjectTypeDef(BaseModel):
    kind: str = "entity"  # entity / rule_table / lookup_table / config
    description: str = ""
    summary: str = ""
    display_name: str = ""
    aliases: list[ObjectAliasDef] = []
    countable: bool = False
    properties: dict[str, PropertyDef] = {}
    source: ObjectSourceDef | None = None
    status_transitions: dict[str, list[str]] = {}
    excluded_functions: list[str] = []
    constraints: list[ObjectConstraint] = []
    data_source: str = ""  # external_api / agent_generated / human_confirmed
    mutability: str = ""  # read_only / append_only / mutable


class LinkDef(BaseModel):
    source: str
    target: str
    join: dict[str, str]
    description: str = ""
    link_type: str = "contains"  # contains / causal / enables / prevents
    cardinality: str = ""  # 1..1 / 1..n / 0..n / 0..1


class FunctionParam(BaseModel):
    type: str = "str"
    description: str = ""
    default: Any = None


class Precondition(BaseModel):
    object: str
    field: str
    operator: str = "eq"  # eq / ne / in / exists / not_exists
    value: Any = None
    value_from_param: str = ""


class Effect(BaseModel):
    object: str
    field: str
    set_to: Any


class TemporalConstraint(BaseModel):
    when: dict[str, str] = {}
    deadline: str = ""
    sla: str = ""


class FunctionDef(BaseModel):
    description: str = ""
    summary: str = ""
    usage_prompt: str = ""
    group: str = ""
    user_visible: bool = True
    depends_on: list[str] = []
    hint: str = ""
    params: dict[str, FunctionParam] = {}
    function_type: str = ""  # business / lookup / get
    writes_to: list[str] = []
    involves_objects: list[str] = []
    preconditions: list[Precondition] = []
    effects: list[Effect] = []
    temporal_constraints: list[TemporalConstraint] = []


class RuleCondition(BaseModel):
    field: str
    operator: str = "eq"  # eq / ne / gt / gte / lt / lte / in / between / like
    value: Any = None
    result: Any = None


class RuleDef(BaseModel):
    description: str = ""
    rule_type: str = ""  # classification / judgment / qualification / threshold
    applies_to: list[str] = []
    conditions: list[RuleCondition] = []
    result_field: str = ""
    source: str = ""


class WorkflowStep(BaseModel):
    name: str
    function: str = ""
    description: str = ""
    next: str | dict[str, str] = ""
    sla: str = ""


class WorkflowDef(BaseModel):
    description: str = ""
    trigger: str = ""
    steps: list[WorkflowStep] = []
    involves_objects: list[str] = []


class IntentPolicyDef(BaseModel):
    keywords: list[str] = []
    context_keywords: list[str] = []
    tools: list[str] = []
    task_hint: str = ""


class InteractionPolicyDef(BaseModel):
    description: str = ""
    include_in_system_prompt: bool = False
    instructions: list[str] = []
    intents: dict[str, IntentPolicyDef] = {}


class PresentationToolDef(BaseModel):
    summary: str = ""
    description: str = ""
    usage_prompt: str = ""
    category: str = "ui"
    side_effect_scope: Literal["none", "frontend_map"] = "frontend_map"
    mutates_domain: bool = False
    object_scope: Literal["none", "mappable", "listed"] = "none"
    allowed_objects: list[str] = []
    requires_confirmation: bool = False
    concurrency_safe: bool = False
    worker_allowed: bool = False
    idempotent: bool = False
    destructive: bool = False
    timeout_seconds: float = 5.0

    @model_validator(mode="after")
    def validate_presentation_scope(self):
        if self.mutates_domain:
            raise ValueError("presentation tools must not mutate domain data")
        if self.object_scope == "listed" and not self.allowed_objects:
            raise ValueError("listed presentation tool requires allowed_objects")
        return self


class EventMapObjectDef(BaseModel):
    object_type: str
    filters: dict[str, Any] = {}
    label: str = ""
    fit: bool = False
    refresh: bool = False


class EventMapPolicyDef(BaseModel):
    mode: Literal["none", "when_relevant", "always"] = "none"
    tool: str = "ui_show_objects"
    objects: list[EventMapObjectDef] = []
    allowed_action_types: list[str] = []
    other_objects: Literal["none", "on_user_request", "allowed"] = "none"
    include_result_cards: bool = False


class EventPolicyDef(BaseModel):
    role: str = "领域智能体"
    description: str = ""
    allowed_tools: list[str] = []
    required_functions: list[str] = []
    forbidden_functions: list[str] = []
    instructions: list[str] = []
    automatic_map: EventMapPolicyDef = EventMapPolicyDef()
    conclusion_fields: list[str] = []

    @model_validator(mode="after")
    def validate_tool_policy(self):
        allowed = set(self.allowed_tools)
        missing = set(self.required_functions) - allowed
        if missing:
            raise ValueError(
                "required_functions must be included in allowed_tools: "
                + ", ".join(sorted(missing))
            )
        overlap = set(self.forbidden_functions) & allowed
        if overlap:
            raise ValueError(
                "forbidden_functions must not be included in allowed_tools: "
                + ", ".join(sorted(overlap))
            )
        if self.automatic_map.mode != "none" and self.automatic_map.tool not in allowed:
            raise ValueError(
                f"automatic map tool must be included in allowed_tools: {self.automatic_map.tool}"
            )
        if self.automatic_map.mode != "none" and not self.automatic_map.allowed_action_types:
            raise ValueError("automatic map policy requires allowed_action_types")
        return self


class Ontology(BaseModel):
    name: str
    description: str = ""
    tool_preferences: dict[str, Any] = {}
    excluded_tools: list[str] = []
    runtime_tools: list[str] = []
    objects: dict[str, ObjectTypeDef] = {}
    links: dict[str, LinkDef] = {}
    functions: dict[str, FunctionDef] = {}
    rules: dict[str, RuleDef] = {}
    workflows: dict[str, WorkflowDef] = {}
    interaction_policies: dict[str, InteractionPolicyDef] = {}
    presentation_tools: dict[str, PresentationToolDef] = {}
    event_policies: dict[str, EventPolicyDef] = {}

    @model_validator(mode="after")
    def validate_event_policy_objects(self):
        for tool_name, tool in self.presentation_tools.items():
            unknown_objects = set(tool.allowed_objects) - set(self.objects)
            if unknown_objects:
                raise ValueError(
                    f"presentation tool {tool_name} references unknown objects: "
                    + ", ".join(sorted(unknown_objects))
                )
        for event_type, policy in self.event_policies.items():
            for item in policy.automatic_map.objects:
                if item.object_type not in self.objects:
                    raise ValueError(
                        f"event policy {event_type} references unknown object: {item.object_type}"
                    )
            if policy.automatic_map.mode != "none":
                tool = self.presentation_tools.get(policy.automatic_map.tool)
                if not tool:
                    raise ValueError(
                        f"event policy {event_type} references unknown presentation tool: "
                        f"{policy.automatic_map.tool}"
                    )
                if (
                    tool.object_scope == "listed"
                    and any(
                        item.object_type not in tool.allowed_objects
                        for item in policy.automatic_map.objects
                    )
                ):
                    raise ValueError(
                        f"event policy {event_type} uses an object outside presentation tool scope"
                    )
            unknown_required = set(policy.required_functions) - set(self.functions)
            if unknown_required:
                raise ValueError(
                    f"event policy {event_type} references unknown required functions: "
                    + ", ".join(sorted(unknown_required))
                )
        known_tools = (
            set(STANDARD_RUNTIME_TOOLS)
            | set(self.runtime_tools)
            | set(self.functions)
            | set(self.presentation_tools)
        )
        for event_type, policy in self.event_policies.items():
            unknown_tools = set(policy.allowed_tools) - known_tools
            if unknown_tools:
                raise ValueError(
                    f"event policy {event_type} references unknown tools: "
                    + ", ".join(sorted(unknown_tools))
                )
        for policy_name, policy in self.interaction_policies.items():
            for intent_name, intent in policy.intents.items():
                unknown_tools = set(intent.tools) - known_tools
                if unknown_tools:
                    raise ValueError(
                        f"interaction policy {policy_name}.{intent_name} references unknown tools: "
                        + ", ".join(sorted(unknown_tools))
                    )
        return self

    @classmethod
    def load(cls, path: str | Path) -> Ontology:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)

    def get_id_column(self, object_type: str) -> str | None:
        obj = self.objects.get(object_type)
        if not obj:
            return None
        for name, prop in obj.properties.items():
            if prop.required:
                return name
        return None

    def table_name(self, object_type: str) -> str:
        obj = self.objects.get(object_type)
        if obj and obj.source and obj.source.table:
            return obj.source.table

        result = []
        for i, ch in enumerate(object_type):
            if ch.isupper() and i > 0:
                result.append("_")
            result.append(ch.lower())
        return "".join(result)

    def get_rules_for_object(self, object_type: str) -> dict[str, RuleDef]:
        return {
            k: v for k, v in self.rules.items()
            if object_type in v.applies_to
        }
