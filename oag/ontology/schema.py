"""OAG ontology metamodel.

The provider supplies an already compiled ``Ontology``. File formats and
domain-specific compilation remain outside OAG.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OntologyModel(BaseModel):
    """Strict base model for the provider-facing OAG metamodel."""

    model_config = ConfigDict(extra="forbid")


class PropertyDef(OntologyModel):
    type: str = "str"
    required: bool = False
    display_name: str = ""
    description: str = ""
    default: Any = None
    aliases: list[str] = Field(default_factory=list)


class DataSourceDef(OntologyModel):
    """A named runtime data source, independent from ontology semantics."""

    type: str
    mode: Literal["read_only", "writable"] = "read_only"
    config: dict[str, Any] = Field(default_factory=dict)


class DataBindingDef(OntologyModel):
    """Map one logical object or relation definition to a named source."""

    source: str
    selector: dict[str, Any] = Field(default_factory=dict)
    mapping: dict[str, Any] = Field(default_factory=dict)


class ObjectAliasDef(OntologyModel):
    terms: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)


class ObjectTypeDef(OntologyModel):
    kind: str = "entity"  # entity / rule_table / lookup_table / config
    description: str = ""
    summary: str = ""
    display_name: str = ""
    aliases: list[ObjectAliasDef] = Field(default_factory=list)
    type_policy: Literal["closed", "open"] = "closed"
    properties: dict[str, PropertyDef] = Field(default_factory=dict)
    binding: DataBindingDef | None = None
    data_source: str = ""  # external_api / agent_generated / human_confirmed
    mutability: str = ""  # read_only / append_only / mutable


class RelationTypeDef(OntologyModel):
    """First-class ontology relation with independently queryable instances."""

    description: str = ""
    summary: str = ""
    display_name: str = ""
    from_types: list[str] = Field(default_factory=list)
    to_types: list[str] = Field(default_factory=list)
    directed: bool = True
    cardinality: str = ""
    type_policy: Literal["closed", "open"] = "closed"
    properties: dict[str, PropertyDef] = Field(default_factory=dict)
    binding: DataBindingDef | None = None
    data_source: str = ""
    mutability: str = ""
    aliases: list[str] = Field(default_factory=list)
    acyclic: bool = False


class FunctionParam(OntologyModel):
    type: str = "str"
    description: str = ""
    default: Any = None


class Precondition(OntologyModel):
    object: str
    field: str = ""
    operator: Literal["eq", "in", "exists", "not_exists"] = "eq"
    value: Any = None
    value_from_param: str = ""


    @model_validator(mode="after")
    def validate_operands(self):
        if self.operator in {"eq", "in"} and not self.field:
            raise ValueError(f"precondition operator {self.operator} requires field")
        if (
            self.operator in {"eq", "in"}
            and self.value is None
            and not self.value_from_param
        ):
            raise ValueError(
                f"precondition operator {self.operator} requires value or value_from_param"
            )
        if (
            self.operator in {"exists", "not_exists"}
            and self.field
            and self.value is None
            and not self.value_from_param
        ):
            raise ValueError(
                f"precondition operator {self.operator} with field requires value or value_from_param"
            )
        if (
            self.operator == "in"
            and not self.value_from_param
            and not isinstance(self.value, list)
        ):
            raise ValueError("precondition operator in requires a list value")
        return self


class TemporalConstraint(OntologyModel):
    """Descriptive timing metadata; OAG does not persist workflow progress."""

    when: dict[str, str] = Field(default_factory=dict)
    deadline: str = ""
    sla: str = ""


class FunctionDef(OntologyModel):
    """A side-effect-free domain capability exposed as an agent tool."""

    description: str = ""
    summary: str = ""
    usage_prompt: str = ""
    user_visible: bool = True
    params: dict[str, FunctionParam] = Field(default_factory=dict)
    timeout_seconds: float | None = 30.0
    concurrency_safe: bool | None = None
    reads_objects: list[str] = Field(default_factory=list)
    reads_relations: list[str] = Field(default_factory=list)
    preconditions: list[Precondition] = Field(default_factory=list)
    temporal_constraints: list[TemporalConstraint] = Field(default_factory=list)


class ActionInputDef(OntologyModel):
    """Public input contract for a state-changing domain Action."""

    display_name: str = ""
    description: str = ""
    type: str = "str"
    required: bool = False
    default: Any = None
    object_types: list[str] = Field(default_factory=list)
    options: list[Any] = Field(default_factory=list)


class ActionSideEffectsDef(OntologyModel):
    """Public side-effect summary; execution templates remain domain-private."""

    creates_objects: list[str] = Field(default_factory=list)
    updates_objects: list[str] = Field(default_factory=list)
    retires_objects: list[str] = Field(default_factory=list)
    creates_relations: list[str] = Field(default_factory=list)
    updates_relations: list[str] = Field(default_factory=list)
    retires_relations: list[str] = Field(default_factory=list)


class ActionDef(OntologyModel):
    """A modeled business operation that may change domain state."""

    display_name: str
    description: str = ""
    summary: str = ""
    usage_prompt: str = ""
    icon: str = ""
    user_visible: bool = True
    available_on: list[str] = Field(default_factory=list)
    context_input: str = ""
    inputs: dict[str, ActionInputDef] = Field(default_factory=dict)
    preconditions: list[dict[str, Any]] = Field(default_factory=list)
    side_effects: ActionSideEffectsDef = Field(default_factory=ActionSideEffectsDef)
    confirmation: str = ""
    idempotency: Literal["required", "optional", "none"] = "required"


class RuleCondition(OntologyModel):
    field: str
    operator: Literal[
        "eq", "ne", "gt", "gte", "lt", "lte", "in", "between", "like"
    ] = "eq"
    value: Any = None
    result: Any = None


class RuleDef(OntologyModel):
    description: str = ""
    rule_type: str = ""  # classification / judgment / qualification / threshold
    applies_to: list[str] = Field(default_factory=list)
    conditions: list[RuleCondition] = Field(default_factory=list)
    result_field: str = ""
    source: str = ""


class WorkflowStep(OntologyModel):
    name: str
    function: str = ""
    action: str = ""
    description: str = ""
    next: str | dict[str, str] = ""
    sla: str = ""

    @model_validator(mode="after")
    def validate_capability(self):
        if self.function and self.action:
            raise ValueError("workflow step cannot reference both function and action")
        return self


class WorkflowDef(OntologyModel):
    description: str = ""
    trigger: str = ""
    steps: list[WorkflowStep] = Field(default_factory=list)
    involves_objects: list[str] = Field(default_factory=list)


class InteractionPolicyDef(OntologyModel):
    description: str = ""
    include_in_system_prompt: bool = False
    instructions: list[str] = Field(default_factory=list)


class Ontology(OntologyModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_id: str = Field("", alias="schema")
    name: str
    version: str = ""
    description: str = ""
    excluded_tools: list[str] = Field(default_factory=list)
    data_sources: dict[str, DataSourceDef] = Field(default_factory=dict)
    objects: dict[str, ObjectTypeDef] = Field(default_factory=dict)
    relations: dict[str, RelationTypeDef] = Field(default_factory=dict)
    functions: dict[str, FunctionDef] = Field(default_factory=dict)
    actions: dict[str, ActionDef] = Field(default_factory=dict)
    rules: dict[str, RuleDef] = Field(default_factory=dict)
    workflows: dict[str, WorkflowDef] = Field(default_factory=dict)
    interaction_policies: dict[str, InteractionPolicyDef] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_references(self):
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
            if action.context_input and action.context_input not in action.inputs:
                raise ValueError(
                    f"action {name} context_input references unknown input: "
                    f"{action.context_input}"
                )
        for name, function in self.functions.items():
            unknown_objects = set(function.reads_objects) - known_objects
            unknown_relations = set(function.reads_relations) - set(self.relations)
            if unknown_objects or unknown_relations:
                raise ValueError(
                    f"function {name} references unknown ontology types: "
                    + ", ".join(sorted(unknown_objects | unknown_relations))
                )
            unknown_preconditions = {
                precondition.object
                for precondition in function.preconditions
                if precondition.object not in known_objects
            }
            if unknown_preconditions:
                raise ValueError(
                    f"function {name} references unknown precondition objects: "
                    + ", ".join(sorted(unknown_preconditions))
                )
            unknown_params = {
                precondition.value_from_param
                for precondition in function.preconditions
                if (
                    precondition.value_from_param
                    and precondition.value_from_param not in function.params
                )
            }
            if unknown_params:
                raise ValueError(
                    f"function {name} preconditions reference unknown params: "
                    + ", ".join(sorted(unknown_params))
                )
        for name, rule in self.rules.items():
            unknown_objects = set(rule.applies_to) - known_objects
            if unknown_objects:
                raise ValueError(
                    f"rule {name} references unknown objects: "
                    + ", ".join(sorted(unknown_objects))
                )
        for name, workflow in self.workflows.items():
            unknown_objects = set(workflow.involves_objects) - known_objects
            unknown_functions = {
                step.function
                for step in workflow.steps
                if step.function and step.function not in self.functions
            }
            unknown_actions = {
                step.action
                for step in workflow.steps
                if step.action and step.action not in self.actions
            }
            if unknown_objects or unknown_functions or unknown_actions:
                raise ValueError(
                    f"workflow {name} references unknown ontology members: "
                    + ", ".join(sorted(
                        unknown_objects | unknown_functions | unknown_actions
                    ))
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
