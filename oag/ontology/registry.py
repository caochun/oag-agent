"""领域函数注册表。

FunctionRegistry 负责把 ontology 中的 FunctionDef 元数据绑定到 Python
callable，并在调用前检查声明的依赖关系。
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .schema import FunctionDef


class FunctionRegistry:
    def __init__(self):
        self._functions: dict[str, Callable] = {}
        self._defs: dict[str, FunctionDef] = {}
        self._executed: set[str] = set()
        self._services: dict[str, Any] = {}
        self._source_adapter_factories: dict[str, Callable] = {}

    def register(self, name: str, fn: Callable, definition: FunctionDef | None = None):
        self._functions[name] = fn
        if definition:
            self._defs[name] = definition

    def call(self, name: str, **kwargs) -> Any:
        fn = self._functions.get(name)
        if not fn:
            raise ValueError(f"Function not found: {name}")
        self._ensure_deps(name)
        result = fn(**kwargs)
        self._executed.add(name)
        return result

    def _ensure_deps(self, name: str):
        fdef = self._defs.get(name)
        if not fdef:
            return
        for dep in fdef.depends_on:
            if dep not in self._executed and dep in self._functions:
                self._ensure_deps(dep)
                self._functions[dep]()
                self._executed.add(dep)

    def has(self, name: str) -> bool:
        return name in self._functions

    def get_def(self, name: str) -> FunctionDef | None:
        return self._defs.get(name)

    def list_functions(self) -> list[tuple[str, FunctionDef | None]]:
        return [(name, self._defs.get(name)) for name in self._functions]

    def register_service(self, name: str, service: Any):
        self._services[name] = service

    def get_service(self, name: str) -> Any:
        return self._services.get(name)

    def has_service(self, name: str) -> bool:
        return name in self._services

    def register_source_adapter(self, source_type: str, factory: Callable):
        """Register a factory that creates one adapter per named data source."""
        self._source_adapter_factories[source_type] = factory

    def get_source_adapter_factory(self, source_type: str) -> Callable | None:
        return self._source_adapter_factories.get(source_type)

    def call_as_tool(self, name: str, args: dict) -> str:
        try:
            result = self.call(name, **args)
            if isinstance(result, (dict, list)):
                return json.dumps(result, ensure_ascii=False)
            return str(result)
        except Exception as e:
            return json.dumps({
                "error": "函数执行错误",
                "tool": name,
                "details": str(e),
            }, ensure_ascii=False)
