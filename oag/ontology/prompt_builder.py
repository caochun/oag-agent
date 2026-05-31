"""本体 prompt 构建器。

负责生成模型初始 system prompt，以及函数/对象的全量静态上下文。当前 OAG
采用全量前置注入，不再在工具执行中做渐进式 prompt 注入。
"""

from __future__ import annotations

from .registry import FunctionRegistry
from .schema import Ontology


class OntologyPromptBuilder:
    """Builds model-facing ontology prompts and full static context."""

    def __init__(self, ontology: Ontology, registry: FunctionRegistry):
        self.ontology = ontology
        self.registry = registry

    def build_system_prompt(self, domain_context: str = "") -> str:
        return "\n\n".join(self.build_static_sections(domain_context))

    def build_static_sections(self, domain_context: str = "") -> list[str]:
        sections = [
            self.build_base_system_prompt(),
            self.build_ontology_summary(),
            self.build_tool_usage_rules(),
        ]
        if domain_context:
            sections.append(domain_context.strip())
        return [section for section in sections if section.strip()]

    def build_system_sections(self, domain_context: str = "") -> list[str]:
        return self.build_static_sections(domain_context)

    def build_base_system_prompt(self) -> str:
        parts = []
        parts.append(f"你是 {self.ontology.name} 领域的智能助手。")
        if self.ontology.description:
            parts.append(f"\n## 领域说明\n{self.ontology.description}")
        return "\n".join(parts)

    def build_ontology_summary(self) -> str:
        parts = []
        parts.append("## 可用对象")
        for name, obj in self.ontology.objects.items():
            kind_label = f" [{obj.kind}]" if obj.kind != "entity" else ""
            line = (obj.summary or obj.description or "").strip().split("\n")[0]
            extras = []
            if obj.mutability:
                extras.append(f"{'🔒' if obj.mutability == 'read_only' else '📝'}{obj.mutability}")
            if obj.excluded_functions:
                extras.append(f"⛔{', '.join(obj.excluded_functions)}")
            suffix = f" | {' | '.join(extras)}" if extras else ""
            parts.append(f"- {name}{kind_label}: {line}{suffix}")

        if self.ontology.links:
            parts.append("\n## 关系")
            for lname, ldef in self.ontology.links.items():
                extras = []
                if ldef.link_type != "contains":
                    extras.append(ldef.link_type)
                if ldef.cardinality:
                    extras.append(ldef.cardinality)
                suffix = f" [{', '.join(extras)}]" if extras else ""
                parts.append(f"- {lname}: {ldef.source} → {ldef.target}{suffix}")

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
                    fn_label = f"调用 {ws.function}" if ws.function else "人工步骤"
                    branch = ""
                    if isinstance(ws.next, dict):
                        branch = " → 分支: " + ", ".join(f"{k}→{v}" for k, v in ws.next.items())
                    elif ws.next:
                        branch = f" → {ws.next}"
                    desc = f" ({ws.description})" if ws.description else ""
                    sla_label = f" ⏱{ws.sla}" if ws.sla else ""
                    parts.append(f"  {i+1}. {ws.name}: {fn_label}{desc}{branch}{sla_label}")
            parts.append("\n重要: 执行工作流时逐步调用工具，根据每步的实际结果决定下一步行动。"
                         "不要一次规划所有步骤——看到结果后再决定。"
                         "如果某步结果显示应走分支路径，就走分支。")

        fn_lines = []
        for name, fdef in self.registry.list_functions():
            if not fdef:
                continue
            fn_parts = [f"- {name}"]
            if fdef.function_type:
                fn_parts.append(f"[{fdef.function_type}]")
            fn_parts.append(f": {(fdef.summary or '').strip().split(chr(10))[0]}")
            fn_lines.append("".join(fn_parts))
        if fn_lines:
            parts.append("\n## 可用函数")
            parts.extend(fn_lines)
            parts.append("(这里只提供摘要。需要函数、对象或规则完整定义时，调用 inspect。)")

        return "\n".join(parts)

    def build_tool_usage_rules(self) -> str:
        parts = []
        parts.append("\n## 工具使用规则")
        parts.append("- 查询数据: 使用 query/count/query_links")
        parts.append("- 统计分析: 使用 describe/pivot/distribution")
        parts.append("- 应用规则: 使用 apply_rule（确定性，不要自己推理）")
        parts.append("- 查看详情: 使用 inspect 获取函数/对象/规则的完整定义；不要假设摘要里没有出现的字段或约束")
        parts.append("- 业务操作: 调用注册的业务函数")
        parts.append("- 数据变更: 使用 mutate 创建/更新/删除对象实例（需用户确认）")
        parts.append("- 全文搜索: 使用 search 跨类型关键词搜索")
        if self.ontology.workflows:
            parts.append("- 工作流: 使用 start_workflow 启动和跟踪工作流进度")
        parts.append("- 进度总结: 使用 summarize_progress 回顾对话进展")
        parts.append("- 用户决策: 遇到多种可行方案或需要用户确认偏好时，使用 ask_user 提问")
        parts.append("- 并行执行: 当有多个相互独立的子任务可以同时进行时，使用 dispatch_workers 并行执行以提高效率")

        parts.append("\n## 重要行为规则")
        parts.append("- 当存在多个可行方案时，必须使用 ask_user 让用户选择，不要自行决定")
        parts.append("- 当任务涉及优先级或策略权衡时，使用 ask_user 确认用户偏好后再执行")
        parts.append("- 当关键参数有多种合理取值时，使用 ask_user 让用户确认")

        return "\n".join(parts)

    def build_full_context(self) -> str:
        parts: list[str] = []

        fn_parts = self._build_all_function_details()
        if fn_parts:
            parts.append("## 函数完整定义")
            parts.extend(fn_parts)

        obj_parts = self._build_all_object_details()
        if obj_parts:
            parts.append("## 对象完整定义")
            parts.extend(obj_parts)

        return "\n\n".join(parts)

    def _build_all_function_details(self) -> list[str]:
        details: list[str] = []
        for fn_name, fdef in self.registry.list_functions():
            if not fdef:
                continue

            lines = [f"### 函数: {fn_name}"]
            if fdef.summary:
                lines.append(f"摘要: {fdef.summary.strip()}")
            if fdef.description:
                lines.append(f"说明: {fdef.description.strip()}")
            if fdef.usage_prompt:
                lines.append(f"使用说明: {fdef.usage_prompt.strip()}")
            if fdef.hint:
                lines.append(f"规则: {fdef.hint.strip()}")
            if fdef.params:
                params = ", ".join(
                    f"{p}({d.type}): {d.description}"
                    for p, d in fdef.params.items()
                )
                lines.append(f"参数: {params}")
            if fdef.preconditions:
                reqs = "; ".join(
                    f"{p.object}.{p.field} {p.operator} {p.value}"
                    for p in fdef.preconditions
                )
                lines.append(f"前置条件: {reqs}")
            if fdef.effects:
                effs = "; ".join(
                    f"{e.object}.{e.field} -> {e.set_to}"
                    for e in fdef.effects
                )
                lines.append(f"执行效果: {effs}")
            if fdef.temporal_constraints:
                slas = "; ".join(
                    f"{tc.sla}({tc.deadline})" if tc.deadline else tc.sla
                    for tc in fdef.temporal_constraints
                    if tc.sla
                )
                if slas:
                    lines.append(f"时间约束: {slas}")
            if fdef.writes_to:
                lines.append(f"写入对象: {', '.join(fdef.writes_to)}")
            if fdef.involves_objects:
                lines.append(f"涉及对象: {', '.join(fdef.involves_objects)}")

            details.append("\n".join(lines))
        return details

    def _build_all_object_details(self) -> list[str]:
        details: list[str] = []
        for obj_name, obj_def in self.ontology.objects.items():
            lines = [f"### 对象: {obj_name}"]
            if obj_def.summary:
                lines.append(f"摘要: {obj_def.summary.strip()}")
            if obj_def.description:
                lines.append(f"说明: {obj_def.description.strip()}")
            if obj_def.mutability:
                lines.append(f"可变性: {obj_def.mutability}")
            if obj_def.data_source:
                lines.append(f"数据来源: {obj_def.data_source}")
            if obj_def.excluded_functions:
                lines.append(f"不可调用: {', '.join(obj_def.excluded_functions)}")
            if obj_def.status_transitions:
                flows = "; ".join(
                    f"{k}->{'|'.join(v)}"
                    for k, v in obj_def.status_transitions.items()
                )
                lines.append(f"状态流转: {flows}")
            for c in obj_def.constraints:
                cond = ", ".join(f"{ck}={cv}" for ck, cv in c.when.items())
                lines.append(
                    f"约束({cond}): 不可调用 {', '.join(c.excluded_functions)}; 原因: {c.reason}"
                )
            if obj_def.properties:
                props = ", ".join(
                    f"{p}({d.type}{'*' if d.required else ''}): {d.description}"
                    for p, d in obj_def.properties.items()
                )
                lines.append(f"属性: {props}")

            rules = self.ontology.get_rules_for_object(obj_name)
            if rules:
                lines.append(
                    "适用规则: " + ", ".join(
                        f"{rname}({rdef.rule_type})"
                        for rname, rdef in rules.items()
                    )
                )

            details.append("\n".join(lines))
        return details
