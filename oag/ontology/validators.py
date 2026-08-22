"""领域函数的本体前置条件校验。"""

from __future__ import annotations

import json
from typing import Any

from .bindings import RuntimeBindings
from .repository import OntologyRepository
from .schema import Ontology


class OntologyValidator:
    """Enforce declared preconditions before a Function reads domain data."""

    def __init__(self, ontology: Ontology, data: OntologyRepository,
                 bindings: RuntimeBindings):
        self.ontology = ontology
        self.store = data
        self.bindings = bindings

    def check_constraints(self, tool_name: str, args: dict) -> str | None:
        fdef = self.bindings.get_def(tool_name)
        if not fdef:
            return None

        if fdef.preconditions:
            missing = []
            for pre in fdef.preconditions:
                if pre.operator == "exists":
                    filters = self._precondition_filters(pre, args)
                    rows = self.store.query_objects(pre.object, filters=filters, limit=1)
                    if not rows:
                        missing.append(f"{pre.object} 不存在匹配记录")
                elif pre.operator == "not_exists":
                    filters = self._precondition_filters(pre, args)
                    rows = self.store.query_objects(pre.object, filters=filters, limit=1)
                    if rows:
                        missing.append(f"{pre.object} 已存在匹配记录")
                elif pre.operator == "eq":
                    expected = self._precondition_value(pre, args)
                    rows = self.store.query_objects(pre.object, filters={pre.field: expected}, limit=1)
                    if not rows:
                        missing.append(f"{pre.object}.{pre.field} 需要为 {expected}")
                elif pre.operator == "in":
                    expected = self._precondition_value(pre, args)
                    found = False
                    for v in (expected or []):
                        if self.store.query_objects(pre.object, filters={pre.field: v}, limit=1):
                            found = True
                            break
                    if not found:
                        missing.append(f"{pre.object}.{pre.field} 需要为 {expected} 之一")
            if missing:
                return json.dumps({
                    "warning": "前置条件未满足",
                    "missing": missing,
                    "hint": "请先完成前置步骤",
                }, ensure_ascii=False)

        return None

    def _precondition_filters(self, pre: Any, args: dict) -> dict | None:
        if not pre.field:
            return None
        value = self._precondition_value(pre, args)
        return {pre.field: value} if value is not None else None

    def _precondition_value(self, pre: Any, args: dict) -> Any:
        raw = getattr(pre, "value", None)
        if raw is not None:
            return raw
        value_from_param = getattr(pre, "value_from_param", None)
        if value_from_param:
            return args.get(value_from_param)
        extra = getattr(pre, "model_extra", None) or {}
        value_from_param = extra.get("value_from_param")
        if value_from_param:
            return args.get(value_from_param)
        return None
