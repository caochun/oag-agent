"""本体子系统导出。

ontology 包定义 OAG 可理解的运行时元模型、Provider/Source/Repository 协议，
并把已编译的领域能力桥接为 prompt 和工具；它不读取领域 DSL 或实现具体存储。
"""

__all__ = [
    "DomainContext",
    "DomainProvider",
    "DataExecutor",
    "RuntimeBindings",
    "Ontology",
    "OntologyRepository",
    "ObjectQuerySource",
    "ObjectSearchSource",
    "RelationSource",
    "SourceManager",
    "OntologyRuntime",
    "RuleEngine",
    "load_domain",
]


def __getattr__(name: str):
    if name in {"DomainContext", "DomainProvider"}:
        from .domain import DomainContext, DomainProvider

        return {"DomainContext": DomainContext, "DomainProvider": DomainProvider}[name]
    if name == "DataExecutor":
        from .data_executor import DataExecutor

        return DataExecutor
    if name == "RuntimeBindings":
        from .bindings import RuntimeBindings

        return RuntimeBindings
    if name == "Ontology":
        from .schema import Ontology

        return Ontology
    if name == "OntologyRepository":
        from .repository import OntologyRepository

        return OntologyRepository
    if name in {"ObjectQuerySource", "ObjectSearchSource", "RelationSource", "SourceManager"}:
        from .source import (
            ObjectQuerySource,
            ObjectSearchSource,
            RelationSource,
            SourceManager,
        )

        return {
            "ObjectQuerySource": ObjectQuerySource,
            "ObjectSearchSource": ObjectSearchSource,
            "RelationSource": RelationSource,
            "SourceManager": SourceManager,
        }[name]
    if name == "OntologyRuntime":
        from .runtime import OntologyRuntime

        return OntologyRuntime
    if name == "RuleEngine":
        from .rules import RuleEngine

        return RuleEngine
    if name == "load_domain":
        from .loader import load_domain

        return load_domain
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
