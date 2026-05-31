"""本体感知的运行时校验。

在工具真正执行前检查函数前置条件、对象可变性、状态流转和 mutate 字段，
把可读的错误返回给模型，而不是让底层 store 抛出模糊异常。
"""

from __future__ import annotations

import json
from typing import Any

from .registry import FunctionRegistry
from .schema import Ontology
from .store import Store


class OntologyValidator:
    """Enforces ontology constraints before tools mutate or depend on data."""

    def __init__(self, ontology: Ontology, store: Store, registry: FunctionRegistry):
        self.ontology = ontology
        self.store = store
        self.registry = registry

    def check_constraints(self, tool_name: str, args: dict) -> str | None:
        fdef = self.registry.get_def(tool_name)
        if not fdef:
            return None

        for obj_name, obj_def in self.ontology.objects.items():
            if tool_name in obj_def.excluded_functions:
                for pname in ("object_type", "event_type"):
                    if args.get(pname) == obj_name:
                        return json.dumps({
                            "error": f"{tool_name} 不适用于 {obj_name}",
                            "hint": f"{obj_name} 已排除 {tool_name}",
                        }, ensure_ascii=False)

        if fdef.preconditions:
            missing = []
            for pre in fdef.preconditions:
                if pre.operator == "exists":
                    rows = self.store.query(pre.object, limit=1)
                    if not rows:
                        missing.append(f"{pre.object} 不存在任何记录")
                elif pre.operator == "eq":
                    rows = self.store.query(pre.object, filters={pre.field: pre.value}, limit=1)
                    if not rows:
                        missing.append(f"{pre.object}.{pre.field} 需要为 {pre.value}")
                elif pre.operator == "in":
                    found = False
                    for v in (pre.value or []):
                        if self.store.query(pre.object, filters={pre.field: v}, limit=1):
                            found = True
                            break
                    if not found:
                        missing.append(f"{pre.object}.{pre.field} 需要为 {pre.value} 之一")
            if missing:
                return json.dumps({
                    "warning": "前置条件未满足",
                    "missing": missing,
                    "hint": "请先完成前置步骤",
                }, ensure_ascii=False)

        return None

    def validate_mutate(self, args: dict) -> str | None:
        operation = args.get("operation", "")
        object_type = args.get("object_type", "")
        data = args.get("data", {})
        object_id = args.get("object_id")

        obj_def = self.ontology.objects.get(object_type)
        if not obj_def:
            return json.dumps({"error": f"未知对象类型: {object_type}"}, ensure_ascii=False)
        if operation not in ("create", "update", "delete"):
            return json.dumps({"error": f"未知操作: {operation}"}, ensure_ascii=False)

        if obj_def.mutability == "read_only":
            return json.dumps({
                "error": f"{object_type} 是只读对象（{obj_def.data_source}），不可 {operation}",
            }, ensure_ascii=False)
        if obj_def.mutability == "append_only" and operation in ("update", "delete"):
            return json.dumps({
                "error": f"{object_type} 仅支持追加写入，不可 {operation}",
            }, ensure_ascii=False)

        if operation in ("update", "delete") and not object_id:
            return json.dumps({"error": f"{operation} 操作需要 object_id"}, ensure_ascii=False)

        existing = None
        if operation in ("update", "delete") and object_id:
            existing = self.store.query_by_id(object_type, object_id)
            if not existing:
                found_in = self._find_object_type(object_id)
                if found_in:
                    return json.dumps({
                        "error": f"在 {object_type} 中未找到 {object_id}",
                        "hint": f"该ID存在于 {found_in}，请改用 object_type=\"{found_in}\"",
                    }, ensure_ascii=False)
                return json.dumps({"error": f"在 {object_type} 中未找到 {object_id}"}, ensure_ascii=False)

        if operation == "update" and "status" in data and obj_def.status_transitions:
            existing = existing or self.store.query_by_id(object_type, object_id)
            if existing:
                old_status = existing.get("status", "")
                new_status = data["status"]
                allowed = obj_def.status_transitions.get(old_status, [])
                if allowed and new_status not in allowed:
                    return json.dumps({
                        "error": f"非法状态转换: {old_status} → {new_status}",
                        "allowed": allowed,
                        "hint": f"{object_type} 从 '{old_status}' 只能转换到: {', '.join(allowed)}",
                    }, ensure_ascii=False)

        if operation in ("create", "update"):
            errors = self._validate_data(obj_def, data, operation)
            if errors:
                available = {p: {"type": d.type, "description": d.description}
                             for p, d in obj_def.properties.items()}
                return json.dumps({"error": "数据校验失败", "details": errors,
                                   "available_fields": available}, ensure_ascii=False)
        return None

    def _find_object_type(self, object_id: Any) -> str | None:
        for type_name in self.ontology.objects:
            row = self.store.query_by_id(type_name, object_id)
            if row:
                return type_name
        return None

    def _validate_data(self, obj_def: Any, data: dict, operation: str) -> list[str]:
        errors: list[str] = []
        valid_props = obj_def.properties

        for key in data:
            if key not in valid_props and key != "_id":
                errors.append(f"未知字段: {key}")

        if operation == "create":
            for prop_name, prop_def in valid_props.items():
                if prop_def.required and prop_name not in data:
                    errors.append(f"缺少必填字段: {prop_name}")

        type_map = {"int": int, "float": float, "str": str}
        for key, value in data.items():
            if key in valid_props and value is not None:
                expected = valid_props[key].type
                validator = type_map.get(expected)
                if validator:
                    try:
                        validator(value)
                    except (ValueError, TypeError):
                        errors.append(f"字段 {key} 类型错误: 期望 {expected}")

        return errors
