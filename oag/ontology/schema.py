"""本体 YAML schema。

这里的 Pydantic 模型定义对象、关系、函数、规则和工作流的配置契约，
并提供业务 ID 字段、表名、适用规则等常用派生信息。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


STANDARD_RUNTIME_TOOLS = frozenset({
    "inspect",
    "query",
    "count",
    "query_relations",
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
    "get_available_actions",
    "ui_open_action_form",
})


class PropertyDef(BaseModel):
    type: str = "str"
    required: bool = False
    display_name: str = ""
    description: str = ""
    default: Any = None
    deprecated: bool = False
    aliases: list[str] = []


class ObjectConstraint(BaseModel):
    when: dict[str, Any]
    excluded_functions: list[str] = []
    reason: str = ""


class DataSourceDef(BaseModel):
    """A named runtime data source, independent from ontology semantics."""

    type: str
    mode: Literal["read_only", "writable"] = "read_only"
    capabilities: list[str] = []
    config: dict[str, Any] = {}


class DataBindingDef(BaseModel):
    """Map one logical object or relation definition to a named source."""

    source: str
    selector: dict[str, Any] = {}
    mapping: dict[str, Any] = {}


class ObjectAliasDef(BaseModel):
    terms: list[str] = []
    filters: dict[str, Any] = {}


class ObjectTypeDef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = "entity"  # entity / rule_table / lookup_table / config
    description: str = ""
    summary: str = ""
    display_name: str = ""
    aliases: list[ObjectAliasDef] = []
    countable: bool = False
    type_policy: Literal["closed", "open"] = "closed"
    properties: dict[str, PropertyDef] = {}
    binding: DataBindingDef | None = None
    status_transitions: dict[str, list[str]] = {}
    excluded_functions: list[str] = []
    constraints: list[ObjectConstraint] = []
    data_source: str = ""  # external_api / agent_generated / human_confirmed
    mutability: str = ""  # read_only / append_only / mutable
    deprecated: bool = False


class RelationTypeDef(BaseModel):
    """First-class ontology relation with independently queryable instances."""

    model_config = ConfigDict(extra="forbid")

    description: str = ""
    summary: str = ""
    display_name: str = ""
    from_types: list[str] = []
    to_types: list[str] = []
    directed: bool = True
    cardinality: str = ""
    type_policy: Literal["closed", "open"] = "closed"
    properties: dict[str, PropertyDef] = {}
    binding: DataBindingDef | None = None
    data_source: str = ""
    mutability: str = ""
    aliases: list[str] = []
    deprecated: bool = False
    acyclic: bool = False


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


class TemporalConstraint(BaseModel):
    when: dict[str, str] = {}
    deadline: str = ""
    sla: str = ""


class FunctionDef(BaseModel):
    """A side-effect-free domain capability exposed as an agent tool."""

    model_config = ConfigDict(extra="forbid")

    description: str = ""
    summary: str = ""
    usage_prompt: str = ""
    group: str = ""
    user_visible: bool = True
    depends_on: list[str] = []
    hint: str = ""
    params: dict[str, FunctionParam] = {}
    timeout_seconds: float | None = 30.0
    concurrency_safe: bool | None = None
    reads_objects: list[str] = []
    reads_relations: list[str] = []
    preconditions: list[Precondition] = []
    temporal_constraints: list[TemporalConstraint] = []


class ActionInputDef(BaseModel):
    """Public input contract for a state-changing domain Action."""

    model_config = ConfigDict(extra="forbid")

    display_name: str = ""
    description: str = ""
    type: str = "str"
    required: bool = False
    default: Any = None
    object_types: list[str] = []
    options: list[Any] = []


class ActionSideEffectsDef(BaseModel):
    """Public side-effect summary; execution templates remain domain-private."""

    model_config = ConfigDict(extra="forbid")

    creates_objects: list[str] = []
    updates_objects: list[str] = []
    retires_objects: list[str] = []
    creates_relations: list[str] = []
    updates_relations: list[str] = []
    retires_relations: list[str] = []


class ActionDef(BaseModel):
    """A modeled business operation that may change domain state."""

    model_config = ConfigDict(extra="forbid")

    display_name: str
    description: str = ""
    summary: str = ""
    usage_prompt: str = ""
    icon: str = ""
    user_visible: bool = True
    available_on: list[str] = []
    context_input: str = ""
    inputs: dict[str, ActionInputDef] = {}
    preconditions: list[dict[str, Any]] = []
    side_effects: ActionSideEffectsDef = ActionSideEffectsDef()
    confirmation: str = ""
    idempotency: Literal["required", "optional", "none"] = "required"


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
    side_effect_scope: Literal[
        "none",
        "frontend_map",
        "frontend_editor",
    ] = "frontend_map"
    mutates_domain: bool = False
    object_scope: Literal["none", "mappable", "listed"] = "none"
    allowed_objects: list[str] = []
    requires_confirmation: bool = False
    concurrency_safe: bool = False
    worker_allowed: bool = False
    idempotent: bool = False
    destructive: bool = False
    timeout_seconds: float = 5.0
    wait_for_user: bool = False

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
    display_name: str = ""
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
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_id: str = Field("", alias="schema")
    name: str
    version: str = ""
    description: str = ""
    tool_preferences: dict[str, Any] = {}
    excluded_tools: list[str] = []
    runtime_tools: list[str] = []
    data_sources: dict[str, DataSourceDef] = {}
    objects: dict[str, ObjectTypeDef] = {}
    relations: dict[str, RelationTypeDef] = {}
    functions: dict[str, FunctionDef] = {}
    actions: dict[str, ActionDef] = {}
    rules: dict[str, RuleDef] = {}
    workflows: dict[str, WorkflowDef] = {}
    interaction_policies: dict[str, InteractionPolicyDef] = {}
    presentation_tools: dict[str, PresentationToolDef] = {}
    event_policies: dict[str, EventPolicyDef] = {}

    @model_validator(mode="after")
    def validate_event_policy_objects(self):
        known_objects = set(self.objects)
        for name, definition in self.objects.items():
            self._validate_binding(f"object {name}", definition.binding)
        for name, definition in self.relations.items():
            self._validate_binding(f"relation {name}", definition.binding)
            unknown_from = set(definition.from_types) - known_objects
            unknown_to = set(definition.to_types) - known_objects
            if unknown_from or unknown_to:
                unknown = sorted(unknown_from | unknown_to)
                raise ValueError(
                    f"relation {name} references unknown objects: {', '.join(unknown)}"
                )
        for name, action in self.actions.items():
            unknown_context = set(action.available_on) - known_objects - {"*"}
            unknown_inputs = {
                object_type
                for input_definition in action.inputs.values()
                for object_type in input_definition.object_types
                if object_type not in known_objects
            }
            effects = action.side_effects
            unknown_effect_objects = (
                set(effects.creates_objects)
                | set(effects.updates_objects)
                | set(effects.retires_objects)
            ) - known_objects
            unknown_effect_relations = (
                set(effects.creates_relations)
                | set(effects.updates_relations)
                | set(effects.retires_relations)
            ) - set(self.relations)
            unknown = (
                unknown_context
                | unknown_inputs
                | unknown_effect_objects
                | unknown_effect_relations
            )
            if unknown:
                raise ValueError(
                    f"action {name} references unknown ontology types: "
                    + ", ".join(sorted(unknown))
                )
        for name, function in self.functions.items():
            unknown_objects = set(function.reads_objects) - known_objects
            unknown_relations = set(function.reads_relations) - set(self.relations)
            if unknown_objects or unknown_relations:
                raise ValueError(
                    f"function {name} references unknown ontology types: "
                    + ", ".join(sorted(unknown_objects | unknown_relations))
                )
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
            | ({"get_available_actions", "ui_open_action_form"} if self.actions else set())
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

    def _validate_binding(
        self,
        label: str,
        binding: DataBindingDef | None,
    ) -> None:
        if binding is None:
            raise ValueError(f"{label} 必须声明 data binding")
        if binding.source not in self.data_sources:
            raise ValueError(
                f"{label} references unknown data source: {binding.source}"
            )

    @classmethod
    def load(cls, path: str | Path) -> Ontology:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)

    def get_id_column(self, object_type: str) -> str | None:
        obj = self.objects.get(object_type) or self.relations.get(object_type)
        if not obj:
            return None
        for name, prop in obj.properties.items():
            if prop.required:
                return name
        return None

    def get_rules_for_object(self, object_type: str) -> dict[str, RuleDef]:
        return {
            k: v for k, v in self.rules.items()
            if object_type in v.applies_to
        }
