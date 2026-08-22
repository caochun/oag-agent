"""Named data-source lifecycle and minimal object/relation read protocols."""

from __future__ import annotations

from typing import Any, Callable, Protocol, TypeVar, runtime_checkable

from .schema import DataBindingDef, Ontology


@runtime_checkable
class ObjectQuerySource(Protocol):
    """Logical object query and lookup."""

    def query_objects(
        self, object_type: str, binding: DataBindingDef,
        filters: dict[str, Any] | None = None, limit: int | None = None,
        order_by: str | None = None, offset: int | None = None,
    ) -> list[dict]: ...

    def get_object(
        self, object_type: str, binding: DataBindingDef, id_value: Any,
    ) -> dict | None: ...


@runtime_checkable
class ObjectSearchSource(Protocol):
    """Optional text search capability for an object source."""

    def search_objects(
        self, object_type: str, binding: DataBindingDef,
        keyword: str, limit: int = 20,
    ) -> list[dict]: ...


@runtime_checkable
class RelationSource(Protocol):
    """Logical first-class relation query and lookup."""

    def query_relations(
        self, relation_type: str, binding: DataBindingDef,
        filters: dict[str, Any] | None = None, *,
        from_id: Any = None, to_id: Any = None, direction: str = "out",
        limit: int | None = None, order_by: str | None = None,
        offset: int | None = None,
    ) -> list[dict]: ...

    def get_relation(
        self, relation_type: str, binding: DataBindingDef, id_value: Any,
    ) -> dict | None: ...


SourceFactory = Callable[..., object]
SourceCapability = TypeVar("SourceCapability")


class SourceManager:
    """Create and share one source instance for each named ontology source."""

    def __init__(self, ontology: Ontology):
        self.ontology = ontology
        self._factories: dict[str, SourceFactory] = {}
        self._instances: dict[str, object] = {}

    def register(self, source_type: str, factory: SourceFactory) -> None:
        if not source_type:
            raise ValueError("数据源类型不能为空")
        if source_type in self._factories:
            raise ValueError(f"数据源类型已注册: {source_type}")
        self._factories[source_type] = factory

    def get(self, source_name: str) -> object:
        if source_name in self._instances:
            return self._instances[source_name]
        definition = self.ontology.data_sources.get(source_name)
        if definition is None:
            raise ValueError(f"未知数据源: {source_name}")
        factory = self._factories.get(definition.type)
        if factory is None:
            raise ValueError(f"不支持的数据源类型: {definition.type}")
        instance = factory(
            ontology=self.ontology,
            source_name=source_name,
            source=definition,
        )
        self._instances[source_name] = instance
        return instance

    def require(
        self, source_name: str, capability: type[SourceCapability],
    ) -> SourceCapability:
        instance = self.get(source_name)
        if not isinstance(instance, capability):
            raise TypeError(
                f"数据源 {source_name} 不支持 {capability.__name__} 能力"
            )
        return instance

    def validate_configuration(self) -> None:
        missing_types = sorted({
            definition.type
            for definition in self.ontology.data_sources.values()
            if definition.type not in self._factories
        })
        if missing_types:
            raise ValueError(
                "No source factory registered for: " + ", ".join(missing_types)
            )

    def close(self) -> None:
        for instance in self._instances.values():
            close = getattr(instance, "close", None)
            if callable(close):
                close()
        self._instances.clear()
