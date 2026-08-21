"""Repository boundary for ontology objects and relations.

Every logical type is bound to a named data source. The repository keeps the
domain vocabulary separate from the physical source adapter and exposes only
object/relation operations to the rest of OAG.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, Protocol, runtime_checkable

from .registry import FunctionRegistry
from .schema import DataBindingDef, Ontology


RecordKind = Literal["object", "relation"]


@runtime_checkable
class SourceAdapter(Protocol):
    """Adapter for one named source that can expose objects and relations."""

    location: Any

    def query_records(
        self, kind: RecordKind, type_name: str, binding: DataBindingDef,
        filters: dict[str, Any] | None = None, limit: int | None = None,
        order_by: str | None = None, offset: int | None = None,
    ) -> list[dict]: ...

    def count_records(
        self, kind: RecordKind, type_name: str, binding: DataBindingDef,
        filters: dict[str, Any] | None = None,
    ) -> int: ...

    def query_record_by_id(
        self, kind: RecordKind, type_name: str, binding: DataBindingDef,
        id_value: Any,
    ) -> dict | None: ...

    def query_relations(self, relation_type: str, binding: DataBindingDef, **kwargs) -> list[dict]: ...
    def search_records(self, kind: RecordKind, type_name: str, binding: DataBindingDef,
                       keyword: str, limit: int = 20) -> list[dict]: ...
    def close(self) -> None: ...


@runtime_checkable
class WritableSource(Protocol):
    def create_record(self, kind: RecordKind, type_name: str,
                      binding: DataBindingDef, data: dict) -> dict: ...
    def update_record(self, kind: RecordKind, type_name: str,
                      binding: DataBindingDef, id_value: Any, data: dict) -> dict: ...
    def retire_record(self, kind: RecordKind, type_name: str,
                      binding: DataBindingDef, id_value: Any) -> dict: ...


@runtime_checkable
class BulkGraphSource(Protocol):
    def query_by_ids(self, kind: RecordKind, type_name: str, binding: DataBindingDef,
                     ids: Iterable[Any], *, include_retired: bool = False) -> list[dict]: ...
    def query_adjacent(self, relation_type: str, binding: DataBindingDef,
                       object_ids: Iterable[Any], *, include_retired: bool = False) -> list[dict]: ...
    def type_counts(self, kind: RecordKind, type_name: str, binding: DataBindingDef) -> dict[str, int]: ...
    def record_exists(self, kind: RecordKind, type_name: str,
                      binding: DataBindingDef, record_id: str) -> bool: ...
    def get_record_version(self, kind: RecordKind, type_name: str,
                           binding: DataBindingDef, record_id: str) -> dict[str, Any] | None: ...


@runtime_checkable
class AtomicGraphSource(Protocol):
    def apply_changeset(self, operations: list[dict[str, Any]], **kwargs) -> None: ...
    def replace_graph(self, objects: list[dict], relations: list[dict], **kwargs) -> None: ...


@runtime_checkable
class HistorySource(Protocol):
    def list_action_log(self, limit: int = 100) -> list[dict]: ...
    def list_record_history(self, kind: RecordKind, record_id: str,
                            limit: int = 100) -> list[dict]: ...


class OntologyRepository:
    """Unified access point for ontology object and relation data."""

    def __init__(self, ontology: Ontology, registry: FunctionRegistry):
        self.ontology = ontology
        self.registry = registry
        self._source_adapters: dict[str, SourceAdapter] = {}

    def query_objects(self, object_type: str, filters: dict[str, Any] | None = None,
                      limit: int | None = None, order_by: str | None = None,
                      offset: int | None = None) -> list[dict]:
        return self._query_records("object", object_type, filters, limit, order_by, offset)

    def count_objects(self, object_type: str,
                      filters: dict[str, Any] | None = None) -> int:
        return self._count_records("object", object_type, filters)

    def get_object(self, object_type: str, id_value: Any) -> dict | None:
        return self._get_record("object", object_type, id_value)

    def query_all_objects(self, filters: dict[str, Any] | None = None,
                          limit: int | None = None) -> list[dict]:
        """Query every concrete object type without reintroducing Object alias."""
        rows: list[dict] = []
        for object_type in self.ontology.objects:
            if object_type in {"Object"}:
                continue
            remaining = None if limit is None else max(0, limit - len(rows))
            if remaining == 0:
                break
            rows.extend(self.query_objects(object_type, filters, remaining))
        return rows if limit is None else rows[:limit]

    def query_all_relations(self, filters: dict[str, Any] | None = None,
                            limit: int | None = None) -> list[dict]:
        """Query every concrete relation type without reintroducing Relation alias."""
        rows: list[dict] = []
        for relation_type in self.ontology.relations:
            if relation_type in {"Relation"}:
                continue
            remaining = None if limit is None else max(0, limit - len(rows))
            if remaining == 0:
                break
            rows.extend(self.query_relations(relation_type, filters, limit=remaining))
        return rows if limit is None else rows[:limit]

    def get_object_any(self, id_value: Any) -> dict | None:
        for object_type in self.ontology.objects:
            if object_type == "Object":
                continue
            record = self.get_object(object_type, id_value)
            if record is not None:
                return record
        return None

    def query_relations(
        self,
        relation_type: str,
        filters: dict[str, Any] | None = None,
        *,
        from_id: Any = None,
        to_id: Any = None,
        direction: Literal["out", "in", "both"] = "out",
        limit: int | None = None,
        order_by: str | None = None,
        offset: int | None = None,
    ) -> list[dict]:
        if direction not in {"out", "in", "both"}:
            raise ValueError("direction 必须是 out、in 或 both")
        definition = self._definition("relation", relation_type)
        return self._source_adapter(definition.binding.source).query_relations(
            relation_type, definition.binding, filters=filters,
            from_id=from_id, to_id=to_id,
            direction=direction, limit=limit,
            order_by=order_by, offset=offset,
        )

    def count_relations(self, relation_type: str,
                        filters: dict[str, Any] | None = None) -> int:
        return self._count_records("relation", relation_type, filters)

    def get_relation(self, relation_type: str, id_value: Any) -> dict | None:
        return self._get_record("relation", relation_type, id_value)

    def create_object(self, object_type: str, data: dict) -> dict:
        return self._create_record("object", object_type, data)

    def update_object(self, object_type: str, id_value: Any, data: dict) -> dict:
        return self._update_record("object", object_type, id_value, data)

    def retire_object(self, object_type: str, id_value: Any) -> dict:
        return self._delete_record("object", object_type, id_value)

    def create_relation(self, relation_type: str, data: dict) -> dict:
        return self._create_record("relation", relation_type, data)

    def update_relation(self, relation_type: str, id_value: Any, data: dict) -> dict:
        return self._update_record("relation", relation_type, id_value, data)

    def retire_relation(self, relation_type: str, id_value: Any) -> dict:
        return self._delete_record("relation", relation_type, id_value)

    def search_text(self, keyword: str, object_types: list[str] | None = None,
                    limit: int = 20) -> list[dict]:
        if not keyword:
            return []
        types_to_search = object_types or list(self.ontology.objects)
        results: list[dict] = []
        for type_name in types_to_search:
            rows = self._search_records("object", type_name, keyword, limit - len(results))
            results.extend(rows)
            if len(results) >= limit:
                break
        return results[:limit]

    def query_by_ids(self, kind: RecordKind, type_name: str, ids: Iterable[Any],
                     *, include_retired: bool = False) -> list[dict]:
        definition = self._definition(kind, type_name)
        adapter = self._source_for(kind, type_name)
        if isinstance(adapter, BulkGraphSource):
            return adapter.query_by_ids(
                kind, type_name, definition.binding, ids,
                include_retired=include_retired,
            )
        return [
            record
            for record_id in ids
            if (record := self._get_record(kind, type_name, record_id)) is not None
        ]

    def query_adjacent(self, relation_type: str, object_ids: Iterable[Any],
                       *, include_retired: bool = False) -> list[dict]:
        definition = self._definition("relation", relation_type)
        adapter = self._source_for("relation", relation_type)
        if isinstance(adapter, BulkGraphSource):
            return adapter.query_adjacent(
                relation_type, definition.binding, object_ids,
                include_retired=include_retired,
            )
        rows: dict[str, dict] = {}
        for object_id in object_ids:
            for row in self.query_relations(
                relation_type, from_id=object_id, direction="both",
            ):
                rows[str(row.get("id"))] = row
        return list(rows.values())

    def type_counts(self, kind: RecordKind, type_name: str) -> dict[str, int]:
        definition = self._definition(kind, type_name)
        adapter = self._source_for(kind, type_name)
        if isinstance(adapter, BulkGraphSource):
            return adapter.type_counts(kind, type_name, definition.binding)
        counts: dict[str, int] = {}
        for record in self._query_records(kind, type_name):
            discriminator = str(record.get("type") or "unknown")
            counts[discriminator] = counts.get(discriminator, 0) + 1
        return counts

    def record_exists(self, kind: RecordKind, type_name: str, record_id: str) -> bool:
        definition = self._definition(kind, type_name)
        adapter = self._source_for(kind, type_name)
        if isinstance(adapter, BulkGraphSource):
            return bool(adapter.record_exists(kind, type_name, definition.binding, record_id))
        return self._get_record(kind, type_name, record_id) is not None

    def get_record_version(self, kind: RecordKind, type_name: str,
                           record_id: str) -> dict[str, Any] | None:
        definition = self._definition(kind, type_name)
        adapter = self._source_for(kind, type_name)
        if isinstance(adapter, BulkGraphSource):
            return adapter.get_record_version(kind, type_name, definition.binding, record_id)
        return None

    def apply_changeset(self, operations: list[dict[str, Any]], *,
                        object_type: str = "Object",
                        relation_type: str = "Relation", **kwargs) -> None:
        adapter = self._shared_graph_source(object_type, relation_type)
        if not isinstance(adapter, AtomicGraphSource):
            raise TypeError("数据源不支持原子 ChangeSet")
        adapter.apply_changeset(
            operations,
            object_binding=self.ontology.objects[object_type].binding,
            relation_binding=self.ontology.relations[relation_type].binding,
            **kwargs,
        )

    def replace_graph(self, objects: list[dict], relations: list[dict], *,
                      object_type: str = "Object",
                      relation_type: str = "Relation", **kwargs) -> None:
        adapter = self._shared_graph_source(object_type, relation_type)
        if not isinstance(adapter, AtomicGraphSource):
            raise TypeError("数据源不支持图重建")
        adapter.replace_graph(objects, relations, **kwargs)

    def list_action_log(self, *, object_type: str = "Object", limit: int = 100):
        adapter = self._source_for("object", object_type)
        if not isinstance(adapter, HistorySource):
            return []
        return adapter.list_action_log(limit=limit)

    def list_record_history(self, kind: RecordKind, record_id: str, *,
                            object_type: str = "Object", limit: int = 100):
        adapter = self._source_for("object", object_type)
        if not isinstance(adapter, HistorySource):
            raise TypeError("数据源不支持记录历史")
        return adapter.list_record_history(kind, record_id, limit=limit)

    def source_location(self, kind: RecordKind, type_name: str):
        adapter = self._source_for(kind, type_name)
        return adapter.location

    def close(self):
        for adapter in self._source_adapters.values():
            adapter.close()

    def _query_records(self, kind, type_name, filters=None, limit=None,
                       order_by=None, offset=None):
        definition = self._definition(kind, type_name)
        return self._source_adapter(definition.binding.source).query_records(
            kind, type_name, definition.binding,
            filters, limit, order_by, offset,
        )

    def _count_records(self, kind, type_name, filters=None):
        definition = self._definition(kind, type_name)
        return self._source_adapter(definition.binding.source).count_records(
            kind, type_name, definition.binding, filters,
        )

    def _get_record(self, kind, type_name, id_value):
        definition = self._definition(kind, type_name)
        return self._source_adapter(definition.binding.source).query_record_by_id(
            kind, type_name, definition.binding, id_value,
        )

    def _search_records(self, kind, type_name, keyword, limit=20):
        definition = self._definition(kind, type_name)
        adapter = self._source_for(kind, type_name)
        return adapter.search_records(kind, type_name, definition.binding, keyword, limit)

    def _create_record(self, kind, type_name, data):
        definition = self._definition(kind, type_name)
        self._assert_writable(definition.binding.source)
        adapter = self._source_adapter(definition.binding.source)
        if not isinstance(adapter, WritableSource):
            raise TypeError(f"数据源 {definition.binding.source} 不支持写入")
        return adapter.create_record(
            kind, type_name, definition.binding, data,
        )

    def _update_record(self, kind, type_name, id_value, data):
        definition = self._definition(kind, type_name)
        self._assert_writable(definition.binding.source)
        adapter = self._source_adapter(definition.binding.source)
        if not isinstance(adapter, WritableSource):
            raise TypeError(f"数据源 {definition.binding.source} 不支持写入")
        return adapter.update_record(
            kind, type_name, definition.binding, id_value, data,
        )

    def _delete_record(self, kind, type_name, id_value):
        definition = self._definition(kind, type_name)
        self._assert_writable(definition.binding.source)
        adapter = self._source_adapter(definition.binding.source)
        if not isinstance(adapter, WritableSource):
            raise TypeError(f"数据源 {definition.binding.source} 不支持写入")
        return adapter.retire_record(
            kind, type_name, definition.binding, id_value,
        )

    def _source_for(self, kind: RecordKind, type_name: str):
        definition = self._definition(kind, type_name)
        return self._source_adapter(definition.binding.source)

    def _source_adapter(self, source_name: str) -> SourceAdapter:
        if source_name in self._source_adapters:
            return self._source_adapters[source_name]
        source = self.ontology.data_sources.get(source_name)
        if source is None:
            raise ValueError(f"未知数据源: {source_name}")
        factory = self.registry.get_source_adapter_factory(source.type)
        if factory is None:
            raise ValueError(f"不支持的数据源类型: {source.type}")
        adapter = factory(
            ontology=self.ontology, registry=self.registry,
            source_name=source_name, source=source,
        )
        if not isinstance(adapter, SourceAdapter):
            raise TypeError(
                f"数据源适配器 {source.type} 未实现 SourceAdapter 协议"
            )
        self._source_adapters[source_name] = adapter
        return adapter

    def _definition(self, kind: RecordKind, type_name: str):
        definitions = self.ontology.objects if kind == "object" else self.ontology.relations
        definition = definitions.get(type_name)
        if definition is None:
            label = "对象" if kind == "object" else "关系"
            raise ValueError(f"未知{label}类型: {type_name}")
        return definition

    def _shared_graph_source(self, object_type: str, relation_type: str):
        object_binding = self._definition("object", object_type).binding
        relation_binding = self._definition("relation", relation_type).binding
        if object_binding is None or relation_binding is None:
            raise TypeError("图操作需要命名对象和关系绑定")
        if object_binding.source != relation_binding.source:
            raise ValueError("一个原子图操作不能跨越多个数据源")
        self._assert_writable(object_binding.source)
        return self._source_adapter(object_binding.source)

    def _assert_writable(self, source_name: str) -> None:
        if self.ontology.data_sources[source_name].mode != "writable":
            raise ValueError(f"数据源 {source_name} 是只读的")
