"""Logical repository boundary for ontology objects and relations."""

from __future__ import annotations

from typing import Any, Literal

from .schema import Ontology
from .source import (
    ObjectQuerySource,
    ObjectSearchSource,
    RelationSource,
    SourceManager,
)

RecordKind = Literal["object", "relation"]


class OntologyRepository:
    """Access logical ontology records through their declared bindings."""

    def __init__(self, ontology: Ontology, sources: SourceManager):
        self.ontology = ontology
        self.sources = sources

    def query_objects(self, object_type: str, filters=None, limit=None,
                      order_by=None, offset=None) -> list[dict]:
        definition = self._definition("object", object_type)
        source = self.sources.require(definition.binding.source, ObjectQuerySource)
        return source.query_objects(
            object_type, definition.binding, filters, limit, order_by, offset,
        )

    def get_object(self, object_type: str, id_value: Any) -> dict | None:
        definition = self._definition("object", object_type)
        source = self.sources.require(definition.binding.source, ObjectQuerySource)
        return source.get_object(object_type, definition.binding, id_value)

    def query_relations(
        self, relation_type: str, filters=None, *, from_id=None, to_id=None,
        direction: Literal["out", "in", "both"] = "out", limit=None,
        order_by=None, offset=None,
    ) -> list[dict]:
        if direction not in {"out", "in", "both"}:
            raise ValueError("direction 必须是 out、in 或 both")
        definition = self._definition("relation", relation_type)
        source = self.sources.require(definition.binding.source, RelationSource)
        return source.query_relations(
            relation_type, definition.binding, filters,
            from_id=from_id, to_id=to_id, direction=direction,
            limit=limit, order_by=order_by, offset=offset,
        )

    def get_relation(self, relation_type: str, id_value: Any) -> dict | None:
        definition = self._definition("relation", relation_type)
        source = self.sources.require(definition.binding.source, RelationSource)
        return source.get_relation(relation_type, definition.binding, id_value)

    def query_all_objects(self, filters=None, limit=None) -> list[dict]:
        rows: list[dict] = []
        for object_type in self.ontology.objects:
            remaining = None if limit is None else max(0, limit - len(rows))
            if remaining == 0:
                break
            rows.extend(self.query_objects(object_type, filters, remaining))
        return rows if limit is None else rows[:limit]

    def query_all_relations(self, filters=None, limit=None) -> list[dict]:
        rows: list[dict] = []
        for relation_type in self.ontology.relations:
            remaining = None if limit is None else max(0, limit - len(rows))
            if remaining == 0:
                break
            rows.extend(self.query_relations(relation_type, filters, limit=remaining))
        return rows if limit is None else rows[:limit]

    def get_object_any(self, id_value: Any) -> dict | None:
        for object_type in self.ontology.objects:
            record = self.get_object(object_type, id_value)
            if record is not None:
                return record
        return None

    def search_text(self, keyword: str, object_types=None, limit: int = 20) -> list[dict]:
        if not keyword:
            return []
        results: list[dict] = []
        for object_type in object_types or self.ontology.objects:
            definition = self._definition("object", object_type)
            source = self.sources.get(definition.binding.source)
            if not isinstance(source, ObjectSearchSource):
                continue
            results.extend(source.search_objects(
                object_type, definition.binding, keyword, limit - len(results),
            ))
            if len(results) >= limit:
                break
        return results[:limit]

    def searchable_object_types(self) -> list[str]:
        return [
            object_type
            for object_type, definition in self.ontology.objects.items()
            if isinstance(
                self.sources.get(definition.binding.source),
                ObjectSearchSource,
            )
        ]

    def close(self) -> None:
        self.sources.close()

    def _definition(self, kind: RecordKind, type_name: str):
        definitions = self.ontology.objects if kind == "object" else self.ontology.relations
        definition = definitions.get(type_name)
        if definition is None:
            label = "对象" if kind == "object" else "关系"
            raise ValueError(f"未知{label}类型: {type_name}")
        return definition
