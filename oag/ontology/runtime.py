"""本体运行时 facade。

OntologyRuntime 只负责装配和转发 prompt、校验、inspect、规则执行和工具注册。
"""

from __future__ import annotations

import json

from ..tools.registry import ToolRegistry
from .bindings import RuntimeBindings
from .data_executor import DataExecutor
from .inspector import OntologyInspector
from .prompt_builder import OntologyPromptBuilder
from .repository import OntologyRepository
from .rules import RuleEngine
from .schema import Ontology
from .tool_registrars import (
    ActionToolRegistrar,
    FunctionToolRegistrar,
    ReadToolRegistrar,
    RuleToolRegistrar,
)
from .validators import OntologyValidator


class OntologyRuntime:
    """Facade that wires ontology capabilities into the agent harness."""

    def __init__(self, ontology: Ontology,
                 bindings: RuntimeBindings,
                 repository: OntologyRepository,
                 rule_engine: RuleEngine | None = None):
        self.ontology = ontology
        self.repository = repository
        self.rule_engine = rule_engine

        self._prompt_builder = OntologyPromptBuilder(ontology, bindings)
        self._validator = OntologyValidator(ontology, self.repository, bindings)
        self._inspector = OntologyInspector(ontology, bindings)
        self._read_tools = ReadToolRegistrar(ontology, self)
        self._rule_tools = RuleToolRegistrar(ontology, rule_engine, self)
        self._function_tools = FunctionToolRegistrar(bindings)
        self._action_tools = ActionToolRegistrar(ontology, bindings)

    def build_static_sections(self, domain_context: str = "") -> list[str]:
        return self._prompt_builder.build_static_sections(domain_context=domain_context)

    def build_base_system_prompt(self) -> str:
        return self._prompt_builder.build_base_system_prompt()

    def build_ontology_summary(self, *, include_actions: bool = True) -> str:
        return self._prompt_builder.build_ontology_summary(
            include_actions=include_actions,
        )

    def check_constraints(self, tool_name: str, args: dict) -> str | None:
        return self._validator.check_constraints(tool_name, args)

    def inspect(self, target: str) -> str:
        return self._inspector.inspect(target)

    def apply_rule(self, tool_name: str, args: dict) -> str:
        if self.rule_engine:
            return self.rule_engine.execute_tool(tool_name, args)
        return json.dumps({"error": "规则引擎未初始化"}, ensure_ascii=False)

    def register_tools(self, tools: ToolRegistry, data: DataExecutor):
        self._read_tools.register(tools, data)
        self._rule_tools.register(tools)
        self._function_tools.register(tools, data)
        self._action_tools.register(tools)

    def set_available_tools(self, names: set[str]) -> None:
        self._prompt_builder.set_available_tools(names)
