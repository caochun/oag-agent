"""本体查询工具和领域函数执行器。"""

from __future__ import annotations

import json
from typing import Literal

from .bindings import RuntimeBindings
from .repository import OntologyRepository


class DataExecutor:

    def __init__(self, repository: OntologyRepository, bindings: RuntimeBindings):
        self.store = repository
        self.bindings = bindings

    def execute(self, name: str, args: dict) -> str:
        # Built-in query tools are handled here; registered domain Functions
        # share the same serialization path. Exceptions are handled once by the
        # outer ToolExecutionPipeline.
        if name == "get_object":
            record = self.store.get_object(args["object_type"], args["id"])
            if record is None:
                return json.dumps({
                    "result": None,
                    "note": (
                        f"{args['object_type']} 中不存在稳定 ID {args['id']}。"
                    ),
                }, ensure_ascii=False)
            return json.dumps(
                self._annotate_record("object", args["object_type"], record),
                ensure_ascii=False,
                default=str,
            )

        if name == "query":
            self._validate_filters(args.get("filters"))
            rows = self.store.query_objects(
                args["object_type"], args.get("filters"),
                args.get("limit"), args.get("order_by"), args.get("offset"),
            )
            if not rows:
                return json.dumps({
                    "results": [],
                    "note": f"{args['object_type']} 没有匹配记录。",
                }, ensure_ascii=False)
            return json.dumps(
                [self._annotate_record("object", args["object_type"], row) for row in rows],
                ensure_ascii=False,
                default=str,
            )

        if name == "query_relations":
            self._validate_filters(args.get("filters"))
            rows = self.store.query_relations(
                args["relation_type"],
                args.get("filters"),
                from_id=args.get("from_id"),
                to_id=args.get("to_id"),
                direction=args.get("direction", "out"),
                limit=args.get("limit"),
                order_by=args.get("order_by"),
                offset=args.get("offset"),
            )
            return json.dumps(
                [
                    self._annotate_record("relation", args["relation_type"], row)
                    for row in rows
                ],
                ensure_ascii=False,
                default=str,
            )

        if name == "search":
            return self._search(args)

        if self.bindings.has(name):
            return self.bindings.call_as_tool(name, args)

        raise ValueError(f"未知工具: {name}")

    @staticmethod
    def _validate_filters(filters) -> None:
        if filters is None:
            return
        if not isinstance(filters, dict):
            raise ValueError("filters 必须是 object")
        for field, value in filters.items():
            if isinstance(value, (list, tuple, set)) and not field.endswith("__in"):
                raise ValueError(
                    f"过滤字段 {field} 使用数组值时必须改为 {field}__in"
                )

    def _search(self, args: dict) -> str:
        keyword = args.get("keyword", "")
        object_types = args.get("object_types")
        limit = args.get("limit", 20)
        results = self.store.search_text(keyword, object_types, limit)
        annotated = [
            self._annotate_record("object", str(row.get("_object_type", "")), row)
            if row.get("_object_type") in self.store.ontology.objects
            else dict(row)
            for row in results
        ]
        return json.dumps(annotated, ensure_ascii=False, default=str)

    def _annotate_record(
        self,
        kind: Literal["object", "relation"],
        type_name: str,
        record: dict,
    ) -> dict:
        definitions = (
            self.store.ontology.objects
            if kind == "object"
            else self.store.ontology.relations
        )
        definition = definitions[type_name]
        values = {**record, **(record.get("properties") or {})}
        properties = {
            name: {
                "type": property_definition.type,
                "display_name": property_definition.display_name or name,
                "description": property_definition.description,
            }
            for name, property_definition in definition.properties.items()
            if name in values
        }
        annotated = dict(record)
        annotated["_semantics"] = {
            "kind": kind,
            "type": type_name,
            "display_name": definition.display_name or type_name,
            "description": definition.description or definition.summary,
            "properties": properties,
        }
        return annotated
