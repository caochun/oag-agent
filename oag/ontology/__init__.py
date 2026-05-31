"""本体子系统导出。

ontology 包承载领域真相：YAML schema、对象存储、函数注册、规则执行、
prompt 构建、运行时验证、显式 inspect、工作流辅助和工具注册。
"""

__all__ = [
    "DataExecutor",
    "FunctionRegistry",
    "Ontology",
    "OntologyRuntime",
    "RuleEngine",
    "Store",
    "load_domain",
]


def __getattr__(name: str):
    if name == "DataExecutor":
        from .data_executor import DataExecutor

        return DataExecutor
    if name == "FunctionRegistry":
        from .registry import FunctionRegistry

        return FunctionRegistry
    if name == "Ontology":
        from .schema import Ontology

        return Ontology
    if name == "OntologyRuntime":
        from .runtime import OntologyRuntime

        return OntologyRuntime
    if name == "RuleEngine":
        from .rules import RuleEngine

        return RuleEngine
    if name == "Store":
        from .store import Store

        return Store
    if name == "load_domain":
        from .loader import load_domain

        return load_domain
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
