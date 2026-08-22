"""Runtime implementations bound to a compiled ontology."""

from __future__ import annotations

import json
from typing import Any, Callable

from .action_runtime import ActionRuntime
from .schema import FunctionDef


class RuntimeBindings:
    """Bind declared Functions and the optional ActionRuntime implementation."""

    def __init__(self):
        self._functions: dict[str, Callable] = {}
        self._defs: dict[str, FunctionDef] = {}
        self._action_runtime: ActionRuntime | None = None

    def register(self, name: str, fn: Callable, definition: FunctionDef):
        if not name:
            raise ValueError("Function name cannot be empty")
        if not callable(fn):
            raise TypeError(f"Function implementation must be callable: {name}")
        if name in self._functions:
            raise ValueError(f"Function already registered: {name}")
        self._functions[name] = fn
        self._defs[name] = definition

    def call(self, name: str, **kwargs) -> Any:
        fn = self._functions.get(name)
        if not fn:
            raise ValueError(f"Function not found: {name}")
        return fn(**kwargs)

    def has(self, name: str) -> bool:
        return name in self._functions

    def get_def(self, name: str) -> FunctionDef | None:
        return self._defs.get(name)

    def list_functions(self) -> list[tuple[str, FunctionDef]]:
        return [(name, self._defs[name]) for name in self._functions]

    def register_action_runtime(self, runtime: ActionRuntime) -> None:
        if not isinstance(runtime, ActionRuntime):
            raise TypeError("Action runtime must implement ActionRuntime")
        if self._action_runtime is not None:
            raise ValueError("ActionRuntime already registered")
        self._action_runtime = runtime

    def get_action_runtime(self) -> ActionRuntime | None:
        return self._action_runtime

    def call_as_tool(self, name: str, args: dict) -> str:
        result = self.call(name, **args)
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)
