"""本体定义的显式检查器。

inspect 工具使用这里的逻辑按需返回函数、对象或规则的完整定义。这是模型
主动请求的显式查询，不是自动 prompt 注入。
"""

from __future__ import annotations

import json
from typing import Any

from .registry import FunctionRegistry
from .schema import Ontology


class OntologyInspector:
    """Renders detailed ontology definitions for the inspect tool."""

    def __init__(self, ontology: Ontology, registry: FunctionRegistry):
        self.ontology = ontology
        self.registry = registry

    def inspect(self, target: str) -> str:
        if not target:
            return json.dumps({"error": "需要参数 name"}, ensure_ascii=False)

        fdef = self.registry.get_def(target)
        if fdef:
            return json.dumps({
                "kind": "function",
                "name": target,
                "summary": fdef.summary,
                "description": fdef.description,
                "group": fdef.group,
                "depends_on": fdef.depends_on,
                "hint": fdef.hint,
                "function_type": fdef.function_type,
                "writes_to": fdef.writes_to,
                "params": {
                    p: {"type": d.type, "description": d.description, "default": d.default}
                    for p, d in fdef.params.items()
                },
            }, ensure_ascii=False, default=str)

        obj = self.ontology.objects.get(target)
        if obj:
            info: dict[str, Any] = {
                "kind": "object",
                "name": target,
                "object_kind": obj.kind,
                "summary": obj.summary,
                "description": obj.description,
                "properties": {
                    p: {"type": d.type, "required": d.required, "description": d.description}
                    for p, d in obj.properties.items()
                },
            }
            if obj.status_transitions:
                info["status_transitions"] = obj.status_transitions
            if obj.excluded_functions:
                info["excluded_functions"] = obj.excluded_functions
            if obj.constraints:
                info["constraints"] = [
                    {"when": c.when, "excluded_functions": c.excluded_functions, "reason": c.reason}
                    for c in obj.constraints
                ]
            rules = self.ontology.get_rules_for_object(target)
            if rules:
                info["applicable_rules"] = {
                    rname: {"description": rdef.description, "rule_type": rdef.rule_type}
                    for rname, rdef in rules.items()
                }
            return json.dumps(info, ensure_ascii=False, default=str)

        rdef = self.ontology.rules.get(target)
        if rdef:
            return json.dumps({
                "kind": "rule",
                "name": target,
                "description": rdef.description,
                "rule_type": rdef.rule_type,
                "applies_to": rdef.applies_to,
                "conditions": [
                    {"field": c.field, "operator": c.operator, "value": c.value, "result": c.result}
                    for c in rdef.conditions
                ],
            }, ensure_ascii=False, default=str)

        return json.dumps({"error": f"未找到: {target}"}, ensure_ascii=False)
