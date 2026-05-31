"""本体 YAML schema。

这里的 Pydantic 模型定义对象、关系、函数、规则和工作流的配置契约，
并提供业务 ID 字段、表名、适用规则等常用派生信息。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class PropertyDef(BaseModel):
    type: str = "str"
    required: bool = False
    description: str = ""
    default: Any = None


class ObjectConstraint(BaseModel):
    when: dict[str, Any]
    excluded_functions: list[str] = []
    reason: str = ""


class ObjectTypeDef(BaseModel):
    kind: str = "entity"  # entity / rule_table / lookup_table / config
    description: str = ""
    summary: str = ""
    properties: dict[str, PropertyDef] = {}
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


class Ontology(BaseModel):
    name: str
    description: str = ""
    objects: dict[str, ObjectTypeDef] = {}
    links: dict[str, LinkDef] = {}
    functions: dict[str, FunctionDef] = {}
    rules: dict[str, RuleDef] = {}
    workflows: dict[str, WorkflowDef] = {}

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
