from __future__ import annotations

import json
import logging
from typing import Any

from .data_executor import DataExecutor
from .registry import FunctionRegistry
from .rules import RuleEngine
from .schema import Ontology
from .store import Store
from .tool_registry import ToolDef, ToolRegistry

logger = logging.getLogger(__name__)


class OntologyRuntime:

    def __init__(self, ontology: Ontology, store: Store,
                 registry: FunctionRegistry,
                 rule_engine: RuleEngine | None = None):
        self.ontology = ontology
        self.store = store
        self.registry = registry
        self.rule_engine = rule_engine
        self._context_shown: set[str] = set()
        self._active_workflows: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def build_system_prompt(self, domain_context: str = "") -> str:
        parts = []
        parts.append(f"你是 {self.ontology.name} 领域的智能助手。")
        if self.ontology.description:
            parts.append(f"\n## 领域说明\n{self.ontology.description}")

        parts.append("\n## 可用对象")
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
            parts.append("(首次调用函数时系统会自动注入该函数的详细规则、前置条件和约束)")

        parts.append("\n## 工具使用规则")
        parts.append("- 查询数据: 使用 query/count/query_links")
        parts.append("- 统计分析: 使用 describe/pivot/distribution")
        parts.append("- 应用规则: 使用 apply_rule（确定性，不要自己推理）")
        parts.append("- 查看详情: 使用 inspect 获取函数/对象的完整定义")
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

        if domain_context:
            parts.append(f"\n{domain_context}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Constraint checking
    # ------------------------------------------------------------------

    def check_constraints(self, tool_name: str, args: dict) -> str | None:
        fdef = self.registry.get_def(tool_name)
        if not fdef:
            return None

        for obj_name, obj_def in self.ontology.objects.items():
            if tool_name in obj_def.excluded_functions:
                for pname in ("object_type", "event_type"):
                    if args.get(pname) == obj_name:
                        return json.dumps({
                            "error": f"{tool_name} 不适用于 {obj_name}",
                            "hint": f"{obj_name} 已排除 {tool_name}",
                        }, ensure_ascii=False)

        if fdef.preconditions:
            missing = []
            for pre in fdef.preconditions:
                if pre.operator == "exists":
                    rows = self.store.query(pre.object, limit=1)
                    if not rows:
                        missing.append(f"{pre.object} 不存在任何记录")
                elif pre.operator == "eq":
                    rows = self.store.query(pre.object, filters={pre.field: pre.value}, limit=1)
                    if not rows:
                        missing.append(f"{pre.object}.{pre.field} 需要为 {pre.value}")
                elif pre.operator == "in":
                    found = False
                    for v in (pre.value or []):
                        if self.store.query(pre.object, filters={pre.field: v}, limit=1):
                            found = True
                            break
                    if not found:
                        missing.append(f"{pre.object}.{pre.field} 需要为 {pre.value} 之一")
            if missing:
                return json.dumps({
                    "warning": "前置条件未满足",
                    "missing": missing,
                    "hint": "请先完成前置步骤",
                }, ensure_ascii=False)

        return None

    # ------------------------------------------------------------------
    # Mutate validation (ontology-aware)
    # ------------------------------------------------------------------

    def validate_mutate(self, args: dict) -> str | None:
        operation = args.get("operation", "")
        object_type = args.get("object_type", "")
        data = args.get("data", {})
        object_id = args.get("object_id")

        obj_def = self.ontology.objects.get(object_type)
        if not obj_def:
            return json.dumps({"error": f"未知对象类型: {object_type}"}, ensure_ascii=False)
        if operation not in ("create", "update", "delete"):
            return json.dumps({"error": f"未知操作: {operation}"}, ensure_ascii=False)

        if obj_def.mutability == "read_only":
            return json.dumps({
                "error": f"{object_type} 是只读对象（{obj_def.data_source}），不可 {operation}",
            }, ensure_ascii=False)
        if obj_def.mutability == "append_only" and operation in ("update", "delete"):
            return json.dumps({
                "error": f"{object_type} 仅支持追加写入，不可 {operation}",
            }, ensure_ascii=False)

        if operation in ("update", "delete") and not object_id:
            return json.dumps({"error": f"{operation} 操作需要 object_id"}, ensure_ascii=False)

        existing = None
        if operation in ("update", "delete") and object_id:
            existing = self.store.query_by_id(object_type, object_id)
            if not existing:
                found_in = self._find_object_type(object_id)
                if found_in:
                    return json.dumps({
                        "error": f"在 {object_type} 中未找到 {object_id}",
                        "hint": f"该ID存在于 {found_in}，请改用 object_type=\"{found_in}\"",
                    }, ensure_ascii=False)
                return json.dumps({"error": f"在 {object_type} 中未找到 {object_id}"}, ensure_ascii=False)

        if operation == "update" and "status" in data and obj_def.status_transitions:
            existing = existing or self.store.query_by_id(object_type, object_id)
            if existing:
                old_status = existing.get("status", "")
                new_status = data["status"]
                allowed = obj_def.status_transitions.get(old_status, [])
                if allowed and new_status not in allowed:
                    return json.dumps({
                        "error": f"非法状态转换: {old_status} → {new_status}",
                        "allowed": allowed,
                        "hint": f"{object_type} 从 '{old_status}' 只能转换到: {', '.join(allowed)}",
                    }, ensure_ascii=False)

        if operation in ("create", "update"):
            errors = self._validate_data(obj_def, data, operation)
            if errors:
                available = {p: {"type": d.type, "description": d.description}
                             for p, d in obj_def.properties.items()}
                return json.dumps({"error": "数据校验失败", "details": errors,
                                   "available_fields": available}, ensure_ascii=False)
        return None

    def _find_object_type(self, object_id: Any) -> str | None:
        for type_name in self.ontology.objects:
            row = self.store.query_by_id(type_name, object_id)
            if row:
                return type_name
        return None

    def _validate_data(self, obj_def: Any, data: dict, operation: str) -> list[str]:
        errors: list[str] = []
        valid_props = obj_def.properties

        for key in data:
            if key not in valid_props and key != "_id":
                errors.append(f"未知字段: {key}")

        if operation == "create":
            for prop_name, prop_def in valid_props.items():
                if prop_def.required and prop_name not in data:
                    errors.append(f"缺少必填字段: {prop_name}")

        type_map = {"int": int, "float": float, "str": str}
        for key, value in data.items():
            if key in valid_props and value is not None:
                expected = valid_props[key].type
                validator = type_map.get(expected)
                if validator:
                    try:
                        validator(value)
                    except (ValueError, TypeError):
                        errors.append(f"字段 {key} 类型错误: 期望 {expected}")

        return errors

    # ------------------------------------------------------------------
    # Inspect
    # ------------------------------------------------------------------

    def inspect(self, target: str) -> str:
        if not target:
            return json.dumps({"error": "需要参数 name"}, ensure_ascii=False)

        fdef = self.registry.get_def(target)
        if fdef:
            return json.dumps({
                "kind": "function",
                "name": target,
                "summary": fdef.summary,
                "description": fdef.description,
                "group": fdef.group,
                "depends_on": fdef.depends_on,
                "hint": fdef.hint,
                "function_type": fdef.function_type,
                "writes_to": fdef.writes_to,
                "params": {
                    p: {"type": d.type, "description": d.description, "default": d.default}
                    for p, d in fdef.params.items()
                },
            }, ensure_ascii=False, default=str)

        obj = self.ontology.objects.get(target)
        if obj:
            info: dict[str, Any] = {
                "kind": "object",
                "name": target,
                "object_kind": obj.kind,
                "summary": obj.summary,
                "description": obj.description,
                "properties": {
                    p: {"type": d.type, "required": d.required, "description": d.description}
                    for p, d in obj.properties.items()
                },
            }
            if obj.status_transitions:
                info["status_transitions"] = obj.status_transitions
            if obj.excluded_functions:
                info["excluded_functions"] = obj.excluded_functions
            if obj.constraints:
                info["constraints"] = [
                    {"when": c.when, "excluded_functions": c.excluded_functions, "reason": c.reason}
                    for c in obj.constraints
                ]
            rules = self.ontology.get_rules_for_object(target)
            if rules:
                info["applicable_rules"] = {
                    rname: {"description": rdef.description, "rule_type": rdef.rule_type}
                    for rname, rdef in rules.items()
                }
            return json.dumps(info, ensure_ascii=False, default=str)

        rdef = self.ontology.rules.get(target)
        if rdef:
            return json.dumps({
                "kind": "rule",
                "name": target,
                "description": rdef.description,
                "rule_type": rdef.rule_type,
                "applies_to": rdef.applies_to,
                "conditions": [
                    {"field": c.field, "operator": c.operator, "value": c.value, "result": c.result}
                    for c in rdef.conditions
                ],
            }, ensure_ascii=False, default=str)

        return json.dumps({"error": f"未找到: {target}"}, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Progressive context injection
    # ------------------------------------------------------------------

    def build_context_for_tool(self, fn_name: str) -> str | None:
        if fn_name in self._context_shown:
            return None

        fdef = self.registry.get_def(fn_name)
        if not fdef:
            return None

        has_detail = fdef.description or fdef.hint or fdef.preconditions or fdef.effects or fdef.temporal_constraints or fdef.writes_to
        if not has_detail:
            self._context_shown.add(fn_name)
            return None

        parts: list[str] = []

        if fdef.description:
            parts.append(f"说明: {fdef.description.strip()}")
        if fdef.hint:
            parts.append(f"规则: {fdef.hint.strip()}")
        if fdef.preconditions:
            reqs = "; ".join(f"{p.object}.{p.field} {p.operator} {p.value}" for p in fdef.preconditions)
            parts.append(f"前置条件: {reqs}")
        if fdef.effects:
            effs = "; ".join(f"{e.object}.{e.field} → {e.set_to}" for e in fdef.effects)
            parts.append(f"执行效果: {effs}")
        if fdef.temporal_constraints:
            slas = "; ".join(f"{tc.sla}({tc.deadline})" if tc.deadline else tc.sla for tc in fdef.temporal_constraints if tc.sla)
            if slas:
                parts.append(f"时间约束: {slas}")
        if fdef.writes_to:
            parts.append(f"写入对象: {', '.join(fdef.writes_to)}")

        related_objects = set(fdef.writes_to or [])
        for pre in fdef.preconditions:
            related_objects.add(pre.object)
        for eff in fdef.effects:
            related_objects.add(eff.object)
        for obj_name in fdef.involves_objects or []:
            related_objects.add(obj_name)

        obj_details = []
        for obj_name in related_objects:
            obj_def = self.ontology.objects.get(obj_name)
            if not obj_def:
                continue
            props = ", ".join(f"{p}({d.type}{'*' if d.required else ''})" for p, d in obj_def.properties.items())
            obj_info = f"  {obj_name}: {props}"
            if obj_def.status_transitions:
                flows = "; ".join(f"{k}→{'|'.join(v)}" for k, v in obj_def.status_transitions.items())
                obj_info += f"\n    状态流转: {flows}"
            for c in obj_def.constraints:
                cond = ", ".join(f"{ck}={cv}" for ck, cv in c.when.items())
                obj_info += f"\n    约束({cond}): ⛔{', '.join(c.excluded_functions)} — {c.reason}"
            obj_details.append(obj_info)

        if obj_details:
            parts.append("关联对象详情:\n" + "\n".join(obj_details))

        self._context_shown.add(fn_name)
        return "\n".join(parts) if parts else None

    def build_context_from_result(self, result: str) -> str | None:
        obj_parts = []
        for obj_name, obj_def in self.ontology.objects.items():
            ctx_key = f"obj:{obj_name}"
            if ctx_key in self._context_shown:
                continue
            if f'"{obj_name}"' not in result and f"'{obj_name}'" not in result and f"={obj_name}" not in result:
                continue

            has_detail = obj_def.description or obj_def.status_transitions or obj_def.constraints or obj_def.excluded_functions
            if not has_detail:
                continue

            lines = [f"[对象 {obj_name} 的完整定义]"]
            if obj_def.description:
                lines.append(obj_def.description.strip())
            if obj_def.excluded_functions:
                lines.append(f"⛔不可调用: {', '.join(obj_def.excluded_functions)}")
            if obj_def.status_transitions:
                flows = "; ".join(f"{k}→{'|'.join(v)}" for k, v in obj_def.status_transitions.items())
                lines.append(f"状态流转: {flows}")
            for c in obj_def.constraints:
                cond = ", ".join(f"{ck}={cv}" for ck, cv in c.when.items())
                lines.append(f"约束({cond}): ⛔{', '.join(c.excluded_functions)} — {c.reason}")
            props = ", ".join(f"{p}({d.type}{'*' if d.required else ''}): {d.description}" for p, d in obj_def.properties.items())
            lines.append(f"属性: {props}")

            obj_parts.append("\n".join(lines))
            self._context_shown.add(ctx_key)

        return "\n\n".join(obj_parts) if obj_parts else None

    def reset_context_shown(self):
        self._context_shown.clear()

    # ------------------------------------------------------------------
    # Workflow management
    # ------------------------------------------------------------------

    def start_workflow(self, args: dict) -> str:
        workflow_name = args.get("workflow_name", "")
        advance_to = args.get("advance_to_step", "")

        wf = self.ontology.workflows.get(workflow_name)
        if not wf:
            return json.dumps({"error": f"未知工作流: {workflow_name}"}, ensure_ascii=False)

        state = self._active_workflows.get(workflow_name)
        if not state:
            state = {"workflow_name": workflow_name, "current_step_index": 0}
            self._active_workflows[workflow_name] = state

        if advance_to:
            found = False
            for i, step in enumerate(wf.steps):
                if step.name == advance_to:
                    state["current_step_index"] = i
                    found = True
                    break
            if not found:
                return json.dumps({"error": f"未知步骤: {advance_to}"}, ensure_ascii=False)

        idx = state["current_step_index"]
        steps_info = []
        for i, step in enumerate(wf.steps):
            info: dict[str, Any] = {
                "index": i,
                "name": step.name,
                "description": step.description or "",
                "function": step.function or "",
                "is_current": i == idx,
            }
            if step.sla:
                info["sla"] = step.sla
            if isinstance(step.next, dict):
                info["branches"] = step.next
            elif step.next:
                info["next"] = step.next
            steps_info.append(info)

        current = wf.steps[idx] if idx < len(wf.steps) else None
        result: dict[str, Any] = {
            "workflow": workflow_name,
            "description": wf.description,
            "trigger": wf.trigger,
            "total_steps": len(wf.steps),
            "current_step_index": idx,
            "current_step": current.name if current else "completed",
            "steps": steps_info,
        }
        if current and current.function:
            result["next_action"] = f"调用 {current.function}"

        return json.dumps(result, ensure_ascii=False, default=str)

    # ------------------------------------------------------------------
    # SLA checking
    # ------------------------------------------------------------------

    def check_sla(self, args: dict) -> str:
        event_id = args.get("event_id", "")
        results = []

        for fname, fdef in self.registry.list_functions():
            if not fdef or not fdef.temporal_constraints:
                continue
            for tc in fdef.temporal_constraints:
                results.append({
                    "function": fname,
                    "condition": tc.when if tc.when else "所有情况",
                    "deadline": tc.deadline,
                    "sla": tc.sla,
                })

        for wname, wdef in self.ontology.workflows.items():
            for step in wdef.steps:
                if step.sla:
                    results.append({
                        "workflow": wname,
                        "step": step.name,
                        "sla": step.sla,
                    })

        if not results:
            return json.dumps({"message": "当前本体中未定义时间约束"}, ensure_ascii=False)

        return json.dumps({
            "sla_definitions": results,
            "note": "请结合事件的实际时间判断是否超时",
        }, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Rule engine delegation
    # ------------------------------------------------------------------

    def apply_rule(self, tool_name: str, args: dict) -> str:
        if self.rule_engine:
            return self.rule_engine.execute_tool(tool_name, args)
        return json.dumps({"error": "规则引擎未初始化"}, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def register_tools(self, tools: ToolRegistry, data: DataExecutor):
        obj_types = list(self.ontology.objects.keys())

        tools.register(ToolDef(
            name="inspect", description="查看函数/对象/规则的完整定义",
            parameters={"type": "object", "properties": {"name": {"type": "string", "description": "函数名、对象类型名或规则名"}}, "required": ["name"]},
            handler=lambda args: self.inspect(args.get("name", "")),
            category="inspect",
        ))

        tools.register(ToolDef(
            name="query",
            description="查询对象实例。filters支持后缀: __like模糊, __gt大于, __gte大于等于, __lt小于, __lte小于等于, __ne不等于",
            parameters={"type": "object", "properties": {"object_type": {"type": "string", "enum": obj_types}, "filters": {"type": "object", "description": "过滤条件"}, "order_by": {"type": "string", "description": "排序字段，-前缀降序"}, "limit": {"type": "integer"}, "offset": {"type": "integer"}}, "required": ["object_type"]},
            handler=lambda args: data.execute("query", args),
            category="query",
        ))

        tools.register(ToolDef(
            name="count", description="统计对象数量",
            parameters={"type": "object", "properties": {"object_type": {"type": "string", "enum": obj_types}, "filters": {"type": "object"}}, "required": ["object_type"]},
            handler=lambda args: data.execute("count", args),
            category="query",
        ))

        if self.ontology.links:
            tools.register(ToolDef(
                name="query_links", description="沿关系查询关联实例",
                parameters={"type": "object", "properties": {"source_type": {"type": "string"}, "source_id": {"type": "string"}, "link_name": {"type": "string", "enum": list(self.ontology.links.keys())}}, "required": ["source_type", "source_id", "link_name"]},
                handler=lambda args: data.execute("query_links", args),
                category="query",
            ))

        tools.register(ToolDef(
            name="describe", description="统计摘要",
            parameters={"type": "object", "properties": {"object_type": {"type": "string", "enum": obj_types}, "column": {"type": "string"}}, "required": ["object_type"]},
            handler=lambda args: data.execute("describe", args),
            category="analysis",
        ))

        tools.register(ToolDef(
            name="pivot", description="透视表分析",
            parameters={"type": "object", "properties": {"object_type": {"type": "string", "enum": obj_types}, "index": {"type": "string"}, "columns": {"type": "string"}, "values": {"type": "string"}, "aggfunc": {"type": "string", "enum": ["mean", "sum", "count", "min", "max"]}}, "required": ["object_type", "index", "columns", "values"]},
            handler=lambda args: data.execute("pivot", args),
            category="analysis",
        ))

        tools.register(ToolDef(
            name="distribution", description="分布直方图",
            parameters={"type": "object", "properties": {"object_type": {"type": "string", "enum": obj_types}, "column": {"type": "string"}, "bins": {"type": "integer"}}, "required": ["object_type", "column"]},
            handler=lambda args: data.execute("distribution", args),
            category="analysis",
        ))

        tools.register(ToolDef(
            name="mutate",
            description="创建/更新/删除对象实例。写操作需要用户确认。object_id 使用业务主键（如 event_id、drone_id），不是内部 _id。如果不确定字段名，先用 inspect 查看对象定义",
            parameters={"type": "object", "properties": {"operation": {"type": "string", "enum": ["create", "update", "delete"], "description": "操作类型"}, "object_type": {"type": "string", "enum": obj_types, "description": "对象类型"}, "object_id": {"type": "string", "description": "对象ID（update/delete必填）"}, "data": {"type": "object", "description": "要写入的字段（create/update时提供）"}}, "required": ["operation", "object_type"]},
            handler=lambda args: data.execute("mutate", args),
            category="action", is_read_only=False, requires_confirmation=True, max_result_chars=2000,
        ))

        tools.register(ToolDef(
            name="search",
            description="跨对象类型全文搜索。在所有（或指定）对象类型的文本字段中搜索关键词",
            parameters={"type": "object", "properties": {"keyword": {"type": "string", "description": "搜索关键词"}, "object_types": {"type": "array", "items": {"type": "string", "enum": obj_types}, "description": "限定搜索的对象类型（可选，不填搜索全部）"}, "limit": {"type": "integer", "description": "最大返回条数（默认20）"}}, "required": ["keyword"]},
            handler=lambda args: data.execute("search", args),
            category="query",
        ))

        workflow_names = list(self.ontology.workflows.keys()) if self.ontology.workflows else []
        if workflow_names:
            tools.register(ToolDef(
                name="start_workflow",
                description="启动或推进工作流。返回工作流定义、当前步骤和下一步指引",
                parameters={"type": "object", "properties": {"workflow_name": {"type": "string", "enum": workflow_names, "description": "工作流名称"}, "advance_to_step": {"type": "string", "description": "推进到指定步骤名（可选）"}}, "required": ["workflow_name"]},
                handler=self.start_workflow,
                category="action",
            ))

        has_sla = any(
            fdef and fdef.temporal_constraints
            for _, fdef in self.registry.list_functions()
        ) or any(
            step.sla for wdef in self.ontology.workflows.values() for step in wdef.steps
        )
        if has_sla:
            tools.register(ToolDef(
                name="check_sla",
                description="检查当前领域中定义的所有时间约束和SLA。返回各函数和工作流步骤的 deadline/SLA 定义，用于判断是否超时",
                parameters={"type": "object", "properties": {"event_id": {"type": "string", "description": "事件编号（可选，用于上下文）"}}, "required": []},
                handler=self.check_sla,
                category="query",
            ))

        if self.rule_engine:
            rule_names = list(self.ontology.rules.keys())
            applicable_types = sorted({t for r in self.ontology.rules.values() for t in r.applies_to})

            tools.register(ToolDef(
                name="apply_rule", description="对指定对象应用业务规则，返回确定性结果（无需LLM推理）",
                parameters={"type": "object", "properties": {"rule_name": {"type": "string", "description": "规则名称", "enum": rule_names}, "object_type": {"type": "string", "description": "对象类型", "enum": applicable_types}, "object_id": {"type": "string", "description": "对象ID"}}, "required": ["rule_name", "object_type", "object_id"]},
                handler=lambda args: self.apply_rule("apply_rule", args),
                category="rule",
            ))

            tools.register(ToolDef(
                name="apply_rule_batch", description="批量应用规则到多个对象",
                parameters={"type": "object", "properties": {"rule_name": {"type": "string", "enum": rule_names}, "object_type": {"type": "string", "enum": applicable_types}, "filters": {"type": "object", "description": "过滤条件（同 query）"}}, "required": ["rule_name", "object_type"]},
                handler=lambda args: self.apply_rule("apply_rule_batch", args),
                category="rule",
            ))

        for name, fdef in self.registry.list_functions():
            if not fdef:
                continue
            props = {}
            required = []
            for pname, pdef in fdef.params.items():
                props[pname] = {
                    "type": pdef.type if pdef.type in ("string", "integer", "number") else "string",
                    "description": pdef.description,
                }
                if pdef.default is None:
                    required.append(pname)

            has_writes = bool(fdef.writes_to)
            is_business = fdef.function_type == "business"
            fn_name = name
            tools.register(ToolDef(
                name=fn_name,
                description=(fdef.summary or fdef.description or "").strip(),
                parameters={"type": "object", "properties": props, "required": required},
                handler=lambda args, _n=fn_name: data.execute(_n, args),
                category="action" if has_writes else "query",
                is_read_only=not has_writes,
                requires_confirmation=has_writes or is_business,
            ))
