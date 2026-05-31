"""本体工作流辅助运行时。

这里只维护轻量 workflow 游标，并把 workflow/SLA 定义作为工具结果返回。
它不自动执行步骤；下一步调用仍由模型选择，Harness 负责约束执行。
"""

from __future__ import annotations

import json
from typing import Any

from .registry import FunctionRegistry
from .schema import Ontology


class WorkflowRuntime:
    """Tracks ontology workflow progress and exposes SLA definitions."""

    def __init__(self, ontology: Ontology, registry: FunctionRegistry):
        self.ontology = ontology
        self.registry = registry
        self.active_workflows: dict[str, dict] = {}

    def start_workflow(self, args: dict) -> str:
        workflow_name = args.get("workflow_name", "")
        advance_to = args.get("advance_to_step", "")

        wf = self.ontology.workflows.get(workflow_name)
        if not wf:
            return json.dumps({"error": f"未知工作流: {workflow_name}"}, ensure_ascii=False)

        state = self.active_workflows.get(workflow_name)
        if not state:
            state = {"workflow_name": workflow_name, "current_step_index": 0}
            self.active_workflows[workflow_name] = state

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

    def check_sla(self, args: dict) -> str:
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
