"""本体 prompt 构建器。

负责生成模型初始 system prompt、紧凑本体摘要，以及按需使用的完整定义。
"""

from __future__ import annotations

from .bindings import RuntimeBindings
from .schema import Ontology


class OntologyPromptBuilder:
    """Builds model-facing ontology prompts and full static context."""

    def __init__(self, ontology: Ontology, bindings: RuntimeBindings):
        self.ontology = ontology
        self.bindings = bindings
        self.available_tools: frozenset[str] | None = None

    def set_available_tools(self, names: set[str]) -> None:
        self.available_tools = frozenset(names)

    def _is_function_visible(self, name: str, fdef) -> bool:
        return bool(fdef.user_visible and name not in set(self.ontology.excluded_tools or []))

    def _is_tool_visible(self, name: str) -> bool:
        return (
            name not in set(self.ontology.excluded_tools or [])
            and (self.available_tools is None or name in self.available_tools)
        )

    def build_system_prompt(self, domain_context: str = "") -> str:
        return "\n\n".join(self.build_static_sections(domain_context))

    def build_static_sections(self, domain_context: str = "") -> list[str]:
        sections = [
            self.build_base_system_prompt(),
            self.build_ontology_summary(),
            self.build_tool_usage_rules(),
            self.build_interaction_policy_prompt(),
        ]
        if domain_context:
            sections.append(domain_context.strip())
        return [section for section in sections if section.strip()]

    def build_base_system_prompt(self) -> str:
        parts = []
        parts.append(f"你是 {self.ontology.name} 领域的智能助手。")
        if self.ontology.description:
            parts.append(f"\n## 领域说明\n{self.ontology.description}")
        return "\n".join(parts)

    def build_ontology_summary(self, *, include_actions: bool = True) -> str:
        parts = []
        parts.append("## 可用对象")
        for name, obj in self.ontology.objects.items():
            kind_label = f" [{obj.kind}]" if obj.kind != "entity" else ""
            line = (obj.summary or obj.description or "").strip().split("\n")[0]
            extras = []
            if obj.mutability:
                extras.append(f"{'🔒' if obj.mutability == 'read_only' else '📝'}{obj.mutability}")
            suffix = f" | {' | '.join(extras)}" if extras else ""
            parts.append(f"- {name}{kind_label}: {line}{suffix}")

        if self.ontology.relations:
            parts.append("\n## 可用关系")
            for name, relation in self.ontology.relations.items():
                endpoints = []
                if relation.from_types:
                    endpoints.append("from=" + ",".join(relation.from_types))
                if relation.to_types:
                    endpoints.append("to=" + ",".join(relation.to_types))
                suffix = f" [{'; '.join(endpoints)}]" if endpoints else ""
                parts.append(
                    f"- {name}: {(relation.summary or relation.description).strip().split(chr(10))[0]}{suffix}"
                )

        if self.ontology.rules:
            parts.append("\n## 可用规则（确定性，无需推理）")
            for rname, rdef in self.ontology.rules.items():
                applies = ", ".join(rdef.applies_to)
                parts.append(f"- {rname} [{rdef.rule_type}]: {rdef.description} (适用于: {applies})")
            parts.append("\n使用 apply_rule/apply_rule_batch 工具应用规则，不要自己推理规则逻辑。")

        if self.ontology.workflows:
            parts.append("\n## 工作流（复杂任务请按以下流程逐步执行）")
            for wname, wdef in self.ontology.workflows.items():
                parts.append(f"\n### {wname}: {wdef.description}")
                parts.append(f"触发条件: {wdef.trigger}")
                for i, ws in enumerate(wdef.steps):
                    if ws.function:
                        capability_label = f"调用 Function {ws.function}"
                    elif ws.action:
                        capability_label = f"执行 Action {ws.action}"
                    else:
                        capability_label = "人工步骤"
                    branch = ""
                    if isinstance(ws.next, dict):
                        branch = " → 分支: " + ", ".join(f"{k}→{v}" for k, v in ws.next.items())
                    elif ws.next:
                        branch = f" → {ws.next}"
                    desc = f" ({ws.description})" if ws.description else ""
                    sla_label = f" ⏱{ws.sla}" if ws.sla else ""
                    parts.append(
                        f"  {i+1}. {ws.name}: {capability_label}{desc}{branch}{sla_label}"
                    )
            parts.append("\n重要: 执行工作流时逐步调用工具，根据每步的实际结果决定下一步行动。"
                         "不要一次规划所有步骤——看到结果后再决定。"
                         "如果某步结果显示应走分支路径，就走分支。")

        if include_actions:
            action_lines = []
            for name, action in self.ontology.actions.items():
                if not action.user_visible or not self._is_tool_visible(name):
                    continue
                scope = f" [适用于: {', '.join(action.available_on)}]" if action.available_on else ""
                summary = (action.summary or action.description).strip().split(chr(10))[0]
                action_lines.append(f"- {name} ({action.display_name}){scope}: {summary}")
            if action_lines:
                parts.append("\n## 可用业务操作")
                parts.extend(action_lines)
                parts.append("业务操作会改变领域状态；通过 request_action_input 交给用户补充输入并确认。")

        fn_lines = []
        for name, fdef in self.bindings.list_functions():
            if not self._is_function_visible(name, fdef):
                continue
            fn_parts = [f"- {name}"]
            fn_parts.append(f": {(fdef.summary or '').strip().split(chr(10))[0]}")
            fn_lines.append("".join(fn_parts))
        if fn_lines:
            parts.append("\n## 可用函数")
            parts.extend(fn_lines)
            if self._is_tool_visible("inspect"):
                parts.append("(这里只提供摘要。需要函数、对象或规则完整定义时，调用 inspect。)")

        return "\n".join(parts)

    def build_tool_usage_rules(self) -> str:
        parts = []
        parts.append("\n## 工具使用规则")
        query_tools = [
            name for name in ("get_object", "query", "query_relations")
            if self._is_tool_visible(name)
        ]
        if query_tools:
            parts.append(f"- 查询数据: 使用 {'/'.join(query_tools)}")
        if self._is_tool_visible("get_object"):
            parts.append("- 已知对象类型和稳定 ID 时优先使用 get_object，并原样传递完整 ID")
        if self._is_tool_visible("apply_rule"):
            parts.append("- 应用规则: 使用 apply_rule（确定性，不要自己推理）")
        if self._is_tool_visible("inspect"):
            parts.append(
                "- 查看定义: 仅当常驻摘要不足以回答所问的属性、约束或能力细节时调用 inspect；"
                "不要为了复述已有摘要而调用"
            )
        if self.ontology.actions:
            parts.append("- 业务操作: 不确定操作时调用 get_available_actions；确定后调用 request_action_input")
            parts.append("- Function 均为只读能力；Action 才能改变业务状态")
        if self._is_tool_visible("search"):
            parts.append("- 全文搜索: 使用 search 跨类型关键词搜索")
        if self.ontology.workflows:
            parts.append("- 工作流: 按本体定义的步骤和分支执行，业务进度以领域数据为准")
        if self._is_tool_visible("ask_user"):
            parts.append("- 用户决策: 遇到多种可行方案或需要用户确认偏好时，使用 ask_user 提问")
        if self._is_tool_visible("dispatch_workers"):
            parts.append("- 并行执行: 当有多个相互独立的子任务可以同时进行时，使用 dispatch_workers 并行执行以提高效率")

        parts.append("\n## 重要行为规则")
        parts.extend([
            "- 严格区分三类内容：工具结果中的业务事实、本体定义中的语义、基于常识的推断",
            "- 业务事实只能来自工具结果或用户明确提供的信息；先前 assistant 的自然语言总结不能作为新的事实来源",
            "- ID、对象类型、属性名、关系类型和枚举/角色值必须按证据原样引用；中文解释应同时保留原始值，不得改写成另一个值",
            "- 工具结果未包含的属性不得擅自补充；尤其不能把未记录的状态、职责、流程或因果关系表述为已确认事实",
            "- 对象 ID 只能作为 ID 使用；除非工具结果返回了名称或其他属性，不得根据 ID 文本猜测名称、状态或含义",
            "- 查询记录随附的 _semantics 是字段和类型语义的直接证据，应优先使用；仍不足以解释约束或能力细节时再调用 inspect，不得依据字段名猜测",
            "- 多个对象与同一主体有关联，不代表这些对象彼此构成先后链路；只有显式关系或确定性规则才能证明对象间路径和因果",
            "- 解释单个对象时，默认逐项说明对象属性和直接关系的语义，不要按对象类型或 ID 命名拼接“典型链路”；用户要求追溯完整链路时，先查询对象间显式关系",
            "- 本体定义可用于解释类型和关系的业务语义，但不能证明某个实例具有定义之外的具体事实",
            "- 用户未明确要求行业背景、判断或推测时，不要主动补充行业常识或推断；用户明确要求时，必须用“一般情况下”或“推测”标注并与事实分开",
            "- 解释已知对象时，优先在同一回合并行获取对象和直接相关关系；已有证据足以回答后停止查询",
        ])
        if self._is_tool_visible("ask_user"):
            parts.append("- 当存在多个可行方案时，必须使用 ask_user 让用户选择，不要自行决定")
            parts.append("- 当任务涉及优先级或策略权衡时，使用 ask_user 确认用户偏好后再执行")
            parts.append("- 当关键参数有多种合理取值时，使用 ask_user 让用户确认")

        return "\n".join(parts)

    def build_interaction_policy_prompt(self) -> str:
        policies = [
            (name, policy)
            for name, policy in self.ontology.interaction_policies.items()
            if policy.include_in_system_prompt and policy.instructions
        ]
        if not policies:
            return ""
        parts = ["## 领域交互策略"]
        for name, policy in policies:
            if policy.description:
                parts.append(f"### {policy.description}")
            elif len(policies) > 1:
                parts.append(f"### {name}")
            parts.extend(f"- {instruction}" for instruction in policy.instructions)
        return "\n".join(parts)
