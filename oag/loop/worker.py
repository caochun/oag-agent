"""并行 Worker 智能体。

Worker 用于执行彼此独立的子任务：它只能看到父任务显式传入的 context，并且
会使用经过过滤的工具列表。所有工具调用仍然经过 Harness 策略约束。
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, TYPE_CHECKING

from openai import OpenAI

from ..llm.retry import call_llm_with_retry
from ..runtime import ToolUseContext

if TYPE_CHECKING:
    from ..harness import Harness

TOOL_ALLOWLIST: dict[str, set[str]] = {
    "inspect": {"inspect_facility", "inspect", "query", "lookup_damage_grade", "apply_rule"},
    "recon": {"plan_recon_mission", "check_compliance", "request_flight_approval",
              "dispatch_drone", "collect_recon_data", "get_drone", "get_drones_in_range",
              "get_operators_available", "inspect", "query", "lookup_drone_class",
              "lookup_operator_license_rule", "lookup_airspace_rule"},
    "plan": {"generate_clearance_plans", "score_plans", "lookup_clearance_technique",
             "lookup_bridge_type", "inspect", "query"},
    "dispatch": {"dispatch_resources", "get_depots_in_range", "get_rescue_teams_in_range",
                 "get_equipment_by_depot", "get_material_by_depot", "inspect", "query"},
    "report": {"generate_event_report", "query", "query_links", "inspect"},
}

TASK_KEYWORDS: list[tuple[str, str]] = [
    ("检查", "inspect"), ("评估", "inspect"), ("inspect", "inspect"),
    ("侦测", "recon"), ("无人机", "recon"), ("飞行", "recon"),
    ("方案", "plan"), ("抢通", "plan"),
    ("调度", "dispatch"), ("资源", "dispatch"),
    ("报告", "report"), ("报送", "report"),
]


def _classify_task(task: str) -> str:
    for keyword, category in TASK_KEYWORDS:
        if keyword in task:
            return category
    return ""


def _filter_tools(all_tools: list[dict], task: str) -> list[dict]:
    category = _classify_task(task)
    allowed = TOOL_ALLOWLIST.get(category)
    if not allowed:
        return all_tools
    return [t for t in all_tools if t["function"]["name"] in allowed]


class Worker:
    def __init__(self, harness: Any, llm_client: OpenAI, model: str,
                 worker_id: str = "", max_turns: int = 5,
                 context: str = ""):
        self.harness = harness
        self.client = llm_client
        self.model = model
        self.worker_id = worker_id
        self.max_turns = max_turns
        self.context = context

    def run(self, task: str) -> dict:
        system = self.harness.build_worker_system_prompt(self.worker_id, self.context)

        all_tools = self.harness.build_tools()
        tools = _filter_tools(all_tools, task)

        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]

        tool_calls_log: list[dict] = []

        for _ in range(self.max_turns):
            response = call_llm_with_retry(
                self.client,
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                temperature=0.1,
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                return {
                    "worker_id": self.worker_id,
                    "task": task,
                    "result": msg.content or "",
                    "tool_calls": tool_calls_log,
                    "status": "success",
                }

            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("工具参数必须是 JSON object")
                    result = self.harness.execute_tool(
                        tc.function.name,
                        args,
                        context=ToolUseContext(source="worker", confirmed=False),
                    )
                except (json.JSONDecodeError, ValueError) as exc:
                    args = {}
                    result = SimpleToolResult(
                        json.dumps({"error": f"工具参数无效: {exc}"}, ensure_ascii=False)
                    )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result.content,
                })
                tool_calls_log.append({"name": tc.function.name, "args": args})

        return {
            "worker_id": self.worker_id,
            "task": task,
            "result": "(达到最大轮次限制)",
            "tool_calls": tool_calls_log,
            "status": "max_turns",
        }


class SimpleToolResult:
    def __init__(self, content: str):
        self.content = content


def run_workers_parallel(harness: Any, llm_client: OpenAI, model: str,
                         tasks: list[str], context: str = "",
                         max_workers: int = 4) -> list[dict]:
    results: list[dict] = [None] * len(tasks)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for i, task in enumerate(tasks):
            worker = Worker(harness, llm_client, model,
                            worker_id=f"W{i+1}", max_turns=5,
                            context=context)
            future = pool.submit(worker.run, task)
            futures[future] = i

        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = {
                    "worker_id": f"W{idx+1}",
                    "task": tasks[idx],
                    "result": f"Worker 执行出错: {e}",
                    "tool_calls": [],
                    "status": "error",
                }

    return results
