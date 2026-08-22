from __future__ import annotations

import json

import pytest

from oag.ontology.bindings import RuntimeBindings
from oag.ontology.data_executor import DataExecutor
from oag.ontology.repository import OntologyRepository
from oag.ontology.schema import Ontology
from oag.ontology.source import SourceManager


class MemoryGraphSource:
    def __init__(self, **_):
        self.objects = [
            {"id": "C1", "type": "customer", "name": "客户一", "properties": {"status": "active"}},
            {"id": "K1", "type": "contract", "name": "合同一", "properties": {"amount": 100}},
        ]
        self.relations = [
            {"id": "R1", "type": "signed", "from": "C1", "to": "K1", "properties": {"role": "buyer"}},
        ]

    @staticmethod
    def _matches(record, filters):
        values = {**record, **record.get("properties", {})}
        for raw_field, expected in (filters or {}).items():
            field, op = raw_field.rsplit("__", 1) if "__" in raw_field else (raw_field, "eq")
            actual = values.get(field)
            if op == "in" and actual not in expected:
                return False
            if op == "gte" and (actual is None or actual < expected):
                return False
            if op == "eq" and actual != expected:
                return False
        return True

    def query_objects(self, object_type, binding, filters=None, limit=None,
                      order_by=None, offset=None):
        rows = [row for row in self.objects if self._matches(row, filters)]
        if order_by:
            field = order_by.lstrip("-")
            rows.sort(
                key=lambda row: {**row, **row.get("properties", {})}.get(field),
                reverse=order_by.startswith("-"),
            )
        rows = rows[offset or 0:]
        return rows if limit is None else rows[:limit]

    def get_object(self, object_type, binding, id_value):
        return next((row for row in self.objects if row["id"] == id_value), None)

    def search_objects(self, object_type, binding, keyword, limit=20):
        needle = keyword.lower()
        return [
            row for row in self.objects
            if needle in json.dumps(row, ensure_ascii=False).lower()
        ][:limit]

    def query_relations(self, relation_type, binding, filters=None, *,
                        from_id=None, to_id=None, direction="out", limit=None,
                        order_by=None, offset=None):
        rows = [row for row in self.relations if self._matches(row, filters)]
        if from_id is not None:
            endpoint = "to" if direction == "in" else "from"
            if direction == "both":
                rows = [row for row in rows if from_id in {row["from"], row["to"]}]
            else:
                rows = [row for row in rows if row[endpoint] == from_id]
        if to_id is not None:
            rows = [row for row in rows if row["to"] == to_id]
        rows = rows[offset or 0:]
        return rows if limit is None else rows[:limit]

    def get_relation(self, relation_type, binding, id_value):
        return next((row for row in self.relations if row["id"] == id_value), None)


def make_repository():
    ontology = Ontology.model_validate({
        "name": "Existing Graph",
        "data_sources": {"graph": {"type": "memory_graph"}},
        "objects": {"Entity": {"binding": {"source": "graph"}}},
        "relations": {
            "BusinessRelation": {
                "display_name": "业务关系",
                "description": "两个业务对象之间的显式关系",
                "from_types": ["Entity"],
                "to_types": ["Entity"],
                "properties": {
                    "role": {
                        "type": "str",
                        "display_name": "参与角色",
                        "description": "该关系中的业务角色",
                    },
                },
                "binding": {"source": "graph"},
            },
        },
    })
    sources = SourceManager(ontology)
    sources.register("memory_graph", MemoryGraphSource)
    return ontology, OntologyRepository(ontology, sources), RuntimeBindings()


def test_named_source_maps_first_class_objects_and_relations():
    ontology, repository, bindings = make_repository()
    try:
        assert list(ontology.relations) == ["BusinessRelation"]
        assert repository.get_object("Entity", "C1")["name"] == "客户一"
        relations = repository.query_relations(
            "BusinessRelation", from_id="C1", direction="out",
        )
        assert relations == [{
            "id": "R1",
            "type": "signed",
            "properties": {"role": "buyer"},
            "from": "C1",
            "to": "K1",
        }]
        assert repository.query_relations(
            "BusinessRelation", from_id="K1", direction="both",
        ) == relations
        assert [item["id"] for item in repository.query_objects(
            "Entity", filters={"amount__gte": 100}, order_by="-amount",
        )] == ["K1"]
        assert repository.search_text("active", ["Entity"])[0]["id"] == "C1"
        assert json.loads(DataExecutor(repository, bindings).execute(
            "query", {"object_type": "Entity", "filters": {"type": "customer"}},
        ))[0]["id"] == "C1"
        annotated = json.loads(DataExecutor(repository, bindings).execute(
            "query", {"object_type": "Entity", "filters": {"id__in": ["C1", "K1"]}},
        ))
        assert [
            {key: value for key, value in row.items() if key != "_semantics"}
            for row in annotated
        ] == repository.query_objects("Entity")
        assert all(row["_semantics"]["type"] == "Entity" for row in annotated)
        relation_result = json.loads(DataExecutor(repository, bindings).execute(
            "query_relations",
            {
                "relation_type": "BusinessRelation",
                "from_id": "C1",
                "direction": "out",
            },
        ))
        assert relation_result[0]["_semantics"] == {
            "kind": "relation",
            "type": "BusinessRelation",
            "display_name": "业务关系",
            "description": "两个业务对象之间的显式关系",
            "properties": {
                "role": {
                    "type": "str",
                    "display_name": "参与角色",
                    "description": "该关系中的业务角色",
                },
            },
        }
    finally:
        repository.close()


def test_inline_source_and_links_are_rejected():
    from pydantic import ValidationError

    from oag.ontology.schema import Ontology

    with pytest.raises(ValidationError):
        Ontology.model_validate({
            "name": "Invalid",
            "objects": {"A": {"source": {"type": "json_file"}}},
            "links": {"a_b": {"source": "A", "target": "B", "join": {}}},
        })
