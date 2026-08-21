from __future__ import annotations

import json
import sqlite3
import pytest

from oag.ontology.data_executor import DataExecutor
from oag.ontology.loader import load_domain


def test_named_sqlite_graph_maps_first_class_relations(tmp_path):
    database = tmp_path / "existing.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE nodes (
            node_id TEXT PRIMARY KEY,
            node_kind TEXT NOT NULL,
            title TEXT NOT NULL,
            attrs TEXT NOT NULL
        );
        CREATE TABLE edges (
            edge_id TEXT PRIMARY KEY,
            edge_kind TEXT NOT NULL,
            source_node TEXT NOT NULL,
            target_node TEXT NOT NULL,
            attrs TEXT NOT NULL
        );
        """
    )
    connection.executemany(
        "INSERT INTO nodes VALUES (?, ?, ?, ?)",
        [
            ("C1", "customer", "客户一", json.dumps({"status": "active"})),
            ("K1", "contract", "合同一", json.dumps({"amount": 100})),
        ],
    )
    connection.execute(
        "INSERT INTO edges VALUES (?, ?, ?, ?, ?)",
        ("R1", "signed", "C1", "K1", json.dumps({"role": "buyer"})),
    )
    connection.commit()
    connection.close()

    (tmp_path / "ontology.yaml").write_text(
        f"""
name: Existing Graph
data_sources:
  graph:
    type: sqlite_property_graph
    config:
      database: {database}
objects:
  Entity:
    binding:
      source: graph
      mapping: {{id: node_id, type: node_kind, name: title, properties: attrs}}
    properties:
      id: {{type: str, required: true}}
      type: {{type: str, required: true}}
      name: {{type: str}}
relations:
  BusinessRelation:
    from_types: [Entity]
    to_types: [Entity]
    binding:
      source: graph
      mapping: {{id: edge_id, type: edge_kind, from: source_node, to: target_node, properties: attrs}}
    properties:
      id: {{type: str, required: true}}
      type: {{type: str, required: true}}
      from: {{type: str, required: true}}
      to: {{type: str, required: true}}
""",
        encoding="utf-8",
    )
    (tmp_path / "functions.py").write_text(
        "def register(registry, repository, ontology):\n    pass\n",
        encoding="utf-8",
    )

    ontology, repository, registry = load_domain(tmp_path)
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
        assert json.loads(DataExecutor(repository, registry).execute(
            "count", {"object_type": "Entity", "filters": {"type": "customer"}},
        )) == {"count": 1}
        assert repository.source_location("object", "Entity") == database.resolve()
        assert repository.source_location("relation", "BusinessRelation") == database.resolve()
    finally:
        repository.close()


def test_inline_source_and_links_are_rejected():
    from pydantic import ValidationError
    from oag.ontology.schema import Ontology

    with pytest.raises(ValidationError):
        Ontology.model_validate({
            "name": "Legacy",
            "objects": {"A": {"source": {"type": "json_file"}}},
            "links": {"a_b": {"source": "A", "target": "B", "join": {}}},
        })
