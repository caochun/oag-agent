from __future__ import annotations

from typing import Protocol, runtime_checkable

import pytest

from oag.ontology.schema import Ontology
from oag.ontology.source import ObjectQuerySource, SourceManager


class MemoryObjectSource:
    location = "memory://objects"

    def __init__(self, **_):
        self.closed = False

    def query_objects(self, object_type, binding, filters=None, limit=None,
                      order_by=None, offset=None):
        return []

    def get_object(self, object_type, binding, id_value):
        return None

    def search_objects(self, object_type, binding, keyword, limit=20):
        return []

    def close(self):
        self.closed = True


@runtime_checkable
class CustomCapability(Protocol):
    def custom_operation(self) -> str: ...


def ontology() -> Ontology:
    return Ontology.model_validate({
        "name": "Sources",
        "data_sources": {
            "memory": {"type": "memory_objects", "mode": "read_only"},
        },
        "objects": {
            "customer": {
                "binding": {"source": "memory"},
            },
        },
    })


def test_source_manager_reuses_one_named_source_instance():
    manager = SourceManager(ontology())
    manager.register("memory_objects", MemoryObjectSource)

    first = manager.require("memory", ObjectQuerySource)
    second = manager.require("memory", ObjectQuerySource)

    assert first is second
    manager.close()
    assert first.closed is True


def test_source_manager_rejects_an_unsupported_extension_capability():
    manager = SourceManager(ontology())
    manager.register("memory_objects", MemoryObjectSource)

    with pytest.raises(TypeError, match="CustomCapability"):
        manager.require("memory", CustomCapability)


def test_source_manager_rejects_duplicate_source_type_registration():
    manager = SourceManager(ontology())
    manager.register("memory_objects", MemoryObjectSource)

    with pytest.raises(ValueError, match="已注册"):
        manager.register("memory_objects", MemoryObjectSource)
