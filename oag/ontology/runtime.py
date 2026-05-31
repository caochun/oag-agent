"""本体运行时 facade。

OntologyRuntime 只负责装配和转发：prompt、校验、inspect、workflow/SLA、
规则执行和工具注册的具体逻辑分别放在更小的模块里。
"""

from __future__ import annotations

import json

from .data_executor import DataExecutor
from .inspector import OntologyInspector
from .prompt_builder import OntologyPromptBuilder
from .registry import FunctionRegistry
from .rules import RuleEngine
from .schema import Ontology
from .store import Store
from .tool_registration import OntologyToolRegistrar
from .validators import OntologyValidator
from .workflow_runtime import WorkflowRuntime
from ..tools.registry import ToolRegistry


class OntologyRuntime:
    """Facade that wires ontology capabilities into the agent harness."""

    def __init__(self, ontology: Ontology, store: Store,
                 registry: FunctionRegistry,
                 rule_engine: RuleEngine | None = None):
        self.ontology = ontology
        self.store = store
        self.registry = registry
        self.rule_engine = rule_engine

        self._prompt_builder = OntologyPromptBuilder(ontology, registry)
        self._validator = OntologyValidator(ontology, store, registry)
        self._inspector = OntologyInspector(ontology, registry)
        self._workflow_runtime = WorkflowRuntime(ontology, registry)
        self._tool_registrar = OntologyToolRegistrar(
            ontology=ontology,
            registry=registry,
            rule_engine=rule_engine,
            runtime=self,
        )

    def build_system_prompt(self, domain_context: str = "") -> str:
        return self._prompt_builder.build_system_prompt(domain_context=domain_context)

    def build_static_sections(self, domain_context: str = "") -> list[str]:
        return self._prompt_builder.build_static_sections(domain_context=domain_context)

    def build_system_sections(self, domain_context: str = "") -> list[str]:
        return self._prompt_builder.build_system_sections(domain_context=domain_context)

    def build_base_system_prompt(self) -> str:
        return self._prompt_builder.build_base_system_prompt()

    def build_ontology_summary(self) -> str:
        return self._prompt_builder.build_ontology_summary()

    def build_full_context(self) -> str:
        return self._prompt_builder.build_full_context()

    def check_constraints(self, tool_name: str, args: dict) -> str | None:
        return self._validator.check_constraints(tool_name, args)

    def validate_mutate(self, args: dict) -> str | None:
        return self._validator.validate_mutate(args)

    def inspect(self, target: str) -> str:
        return self._inspector.inspect(target)

    def start_workflow(self, args: dict) -> str:
        return self._workflow_runtime.start_workflow(args)

    def check_sla(self, args: dict) -> str:
        return self._workflow_runtime.check_sla(args)

    def apply_rule(self, tool_name: str, args: dict) -> str:
        if self.rule_engine:
            return self.rule_engine.execute_tool(tool_name, args)
        return json.dumps({"error": "规则引擎未初始化"}, ensure_ascii=False)

    def register_tools(self, tools: ToolRegistry, data: DataExecutor):
        self._tool_registrar.register_tools(tools, data)
