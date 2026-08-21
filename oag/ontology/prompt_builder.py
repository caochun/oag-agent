"""本体 prompt 构建器。

负责生成模型初始 system prompt，以及函数/对象的全量静态上下文。当前 OAG
采用全量前置注入，不再在工具执行中做渐进式 prompt 注入。
"""

from __future__ import annotations

import json
from typing import Any

from .registry import FunctionRegistry
from .schema import Ontology


class OntologyPromptBuilder:
    """Builds model-facing ontology prompts and full static context."""

    def __init__(self, ontology: Ontology, registry: FunctionRegistry):
        self.ontology = ontology
        self.registry = registry

    def _is_function_visible(self, name: str, fdef) -> bool:
        return bool(fdef and fdef.user_visible and name not in set(self.ontology.excluded_tools or []))

    def _is_tool_visible(self, name: str) -> bool:
        return name not in set(self.ontology.excluded_tools or [])

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
            parts.append("业务操作会改变领域状态；通过 ui_open_action_form 交给用户补充输入并确认。")

        fn_lines = []
        for name, fdef in self.registry.list_functions():
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

        if self.ontology.presentation_tools:
            parts.append("\n## 展示工具")
            for name, tool in self.ontology.presentation_tools.items():
                parts.append(f"- {name} [{tool.side_effect_scope}]: {tool.summary}")

        return "\n".join(parts)

    def build_tool_usage_rules(self) -> str:
        parts = []
        parts.append("\n## 工具使用规则")
        query_tools = [
            name for name in ("query", "count", "query_relations")
            if self._is_tool_visible(name)
        ]
        if query_tools:
            parts.append(f"- 查询数据: 使用 {'/'.join(query_tools)}")
        if self._is_tool_visible("describe"):
            parts.append("- 统计摘要: 使用 describe")
        if self._is_tool_visible("apply_rule"):
            parts.append("- 应用规则: 使用 apply_rule（确定性，不要自己推理）")
        if self._is_tool_visible("inspect"):
            parts.append("- 查看详情: 使用 inspect 获取函数/对象/规则的完整定义；不要假设摘要里没有出现的字段或约束")
        if self.ontology.actions:
            parts.append("- 业务操作: 不确定操作时调用 get_available_actions；确定后调用 ui_open_action_form")
            parts.append("- Function 均为只读能力；Action 才能改变业务状态")
        if self._is_tool_visible("mutate"):
            parts.append("- 数据变更: 使用 mutate 创建/更新/删除对象实例（需用户确认）")
        if self._is_tool_visible("search"):
            parts.append("- 全文搜索: 使用 search 跨类型关键词搜索")
        if self.ontology.workflows:
            parts.append("- 工作流: 使用 start_workflow 启动和跟踪工作流进度")
        if self._is_tool_visible("summarize_progress"):
            parts.append("- 进度总结: 使用 summarize_progress 回顾对话进展")
        if self._is_tool_visible("ask_user"):
            parts.append("- 用户决策: 遇到多种可行方案或需要用户确认偏好时，使用 ask_user 提问")
        if self._is_tool_visible("dispatch_workers"):
            parts.append("- 并行执行: 当有多个相互独立的子任务可以同时进行时，使用 dispatch_workers 并行执行以提高效率")

        parts.append("\n## 重要行为规则")
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

    def build_event_prompt(self, event_type: str, event: dict[str, Any]) -> str:
        policy = self.ontology.event_policies.get(event_type)
        if not policy:
            raise KeyError(f"ontology event policy not found: {event_type}")

        event_display_name = policy.display_name or event_type
        parts = [
            "/no_think",
            f"你正在作为{policy.role}接收“{event_display_name}”后台领域事件。",
        ]
        if policy.display_name:
            parts.append(
                f"该事件的技术类型标识是 {event_type}。面向用户的结论必须使用中文名称"
                f"“{event_display_name}”称呼该事件；除非解释原始技术字段，否则不要用"
                f" {event_type} 作为标题或正文中的事件名称。"
            )
        if policy.description:
            parts.append(policy.description)
        if policy.required_functions:
            parts.append(
                "必须调用以下函数：" + "、".join(policy.required_functions) + "。"
            )
        parts.extend(policy.instructions)

        map_policy = policy.automatic_map
        if map_policy.mode != "none" and map_policy.objects:
            objects = [item.model_dump(exclude={"label"}) for item in map_policy.objects]
            for item, configured in zip(map_policy.objects, objects):
                if item.label:
                    configured["label"] = item.label
            map_args = json.dumps({"objects": objects}, ensure_ascii=False)
            if map_policy.mode == "always":
                prefix = "必须执行自动地图展示"
            else:
                prefix = "如果该事件需要在 GIS 上展示"
            parts.append(
                f"{prefix}，调用 {map_policy.tool}，且对象配置只能使用 {map_args}。"
            )
        if map_policy.other_objects == "on_user_request":
            parts.append(
                "分析结果中的其他业务对象只用于本次文字概况，不得在自动事件处理中展示；"
                "只有用户在普通对话中明确请求时才可展示。"
            )
        elif map_policy.other_objects == "none":
            parts.append("不得在本次事件处理中展示自动地图策略未声明的其他对象。")

        if policy.forbidden_functions:
            parts.append(
                "禁止调用以下函数：" + "、".join(policy.forbidden_functions) + "。"
            )
        if policy.conclusion_fields:
            parts.append(
                "请用简短结论说明：" + "、".join(policy.conclusion_fields) + "。"
            )
        parts.append(
            "原始事件如下：\n" + json.dumps(event, ensure_ascii=False, indent=2)
        )
        return "\n".join(parts)

    def build_full_context(self) -> str:
        parts: list[str] = []

        fn_parts = self._build_all_function_details()
        if fn_parts:
            parts.append("## 函数完整定义")
            parts.extend(fn_parts)

        action_parts = self._build_all_action_details()
        if action_parts:
            parts.append("## 业务操作完整定义")
            parts.extend(action_parts)

        obj_parts = self._build_all_object_details()
        if obj_parts:
            parts.append("## 对象完整定义")
            parts.extend(obj_parts)

        presentation_parts = self._build_all_presentation_tool_details()
        if presentation_parts:
            parts.append("## 展示工具完整定义")
            parts.extend(presentation_parts)

        return "\n\n".join(parts)

    def _build_all_presentation_tool_details(self) -> list[str]:
        details = []
        for name, tool in self.ontology.presentation_tools.items():
            lines = [f"### 展示工具: {name}"]
            if tool.summary:
                lines.append(f"摘要: {tool.summary}")
            if tool.description:
                lines.append(f"说明: {tool.description}")
            if tool.usage_prompt:
                lines.append(f"使用说明: {tool.usage_prompt}")
            lines.append(f"副作用范围: {tool.side_effect_scope}")
            lines.append(f"修改领域数据: {tool.mutates_domain}")
            lines.append(f"对象范围: {tool.object_scope}")
            if tool.allowed_objects:
                lines.append(f"允许对象: {', '.join(tool.allowed_objects)}")
            details.append("\n".join(lines))
        return details

    def _build_all_function_details(self) -> list[str]:
        details: list[str] = []
        for fn_name, fdef in self.registry.list_functions():
            if not self._is_function_visible(fn_name, fdef):
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
            if fdef.temporal_constraints:
                slas = "; ".join(
                    f"{tc.sla}({tc.deadline})" if tc.deadline else tc.sla
                    for tc in fdef.temporal_constraints
                    if tc.sla
                )
                if slas:
                    lines.append(f"时间约束: {slas}")
            if fdef.reads_objects:
                lines.append(f"读取对象: {', '.join(fdef.reads_objects)}")
            if fdef.reads_relations:
                lines.append(f"读取关系: {', '.join(fdef.reads_relations)}")

            details.append("\n".join(lines))
        return details

    def _build_all_action_details(self) -> list[str]:
        details: list[str] = []
        for action_name, action in self.ontology.actions.items():
            if not action.user_visible or not self._is_tool_visible(action_name):
                continue
            lines = [f"### 业务操作: {action_name} ({action.display_name})"]
            if action.description:
                lines.append(f"说明: {action.description.strip()}")
            if action.available_on:
                lines.append(f"适用对象: {', '.join(action.available_on)}")
            if action.inputs:
                inputs = ", ".join(
                    f"{name}({definition.type}{'*' if definition.required else ''})"
                    for name, definition in action.inputs.items()
                )
                lines.append(f"输入: {inputs}")
            effects = action.side_effects
            effect_parts = []
            for label, values in (
                ("创建对象", effects.creates_objects),
                ("更新对象", effects.updates_objects),
                ("退役对象", effects.retires_objects),
                ("创建关系", effects.creates_relations),
                ("更新关系", effects.updates_relations),
                ("退役关系", effects.retires_relations),
            ):
                if values:
                    effect_parts.append(f"{label}: {', '.join(values)}")
            if effect_parts:
                lines.append("公开副作用: " + "; ".join(effect_parts))
            if action.confirmation:
                lines.append(f"确认说明: {action.confirmation}")
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
            if obj_def.binding:
                lines.append(f"数据访问: {obj_def.binding.source}")
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
