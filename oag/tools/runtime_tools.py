"""Runtime control tools, with optional result and worker extensions."""

from __future__ import annotations

import json
from typing import Callable

from ..runtime.tool_result_store import ToolResultStore
from .registry import ToolDef, ToolPolicy, ToolRegistry

DispatchWorkers = Callable[[list[str], str], list[dict]]


class RuntimeTools:
    def __init__(self, *, dispatch_workers: DispatchWorkers | None = None,
                 result_store: ToolResultStore | None = None):
        self.dispatch_workers = dispatch_workers
        self.result_store = result_store

    def register(self, tools: ToolRegistry):
        tools.register(ToolDef(
            name="ask_user",
            description="向用户提问以收集决策。当存在多种可行方案、需要确认优先级或参数时使用。用户回答后会作为工具结果返回",
            parameters={"type": "object", "properties": {
                "question": {"type": "string", "description": "要问用户的问题"},
                "options": {"type": "array", "items": {"type": "object", "properties": {"label": {"type": "string", "description": "选项标签"}, "description": {"type": "string", "description": "选项说明"}}, "required": ["label"]}, "description": "可选项列表（2-5个）"},
                "multi_select": {"type": "boolean", "description": "是否允许多选（默认单选）"},
            }, "required": ["question", "options"]},
            handler=lambda args: json.dumps({"question": args.get("question", ""), "options": args.get("options", [])}, ensure_ascii=False),
            usage_prompt="当关键参数、优先级、策略偏好存在多个合理选择时使用。问题要具体，options 通常提供 2-5 个互斥选项；不要询问可以通过只读工具直接查到的信息。",
            category="ask",
            policy=ToolPolicy(
                read_only=True,
                requires_confirmation=False,
                requires_user_input=True,
                concurrency_safe=False,
                worker_allowed=False,
            ),
        ))

        if self.result_store is not None:
            tools.register(ToolDef(
                name="read_tool_result",
                description="读取 OAG 持久化的大型工具结果。默认只读取有限片段；仅用于读取工具结果中的 result_ref。",
                parameters={"type": "object", "properties": {
                    "result_ref": {"type": "string", "description": "工具结果中的不透明 result_ref"},
                    "max_chars": {"type": "integer", "description": "最多返回字符数，默认12000，最多50000；只有确需完整核对时才显式加大"},
                }, "required": ["result_ref"]},
                handler=lambda args: self.result_store.read(
                    result_ref=args.get("result_ref", ""),
                    max_chars=args.get("max_chars", 12000) or 12000,
                ),
                usage_prompt="仅当某个工具结果返回 persisted=true 且预览不足以回答时调用。优先读取默认长度；需要更多证据时先定向查询缩小范围。",
                category="query",
                max_result_chars=52000,
                policy=ToolPolicy(
                    read_only=True,
                    requires_confirmation=False,
                    concurrency_safe=True,
                    worker_allowed=True,
                ),
            ))

        if self.dispatch_workers is not None:
            tools.register(ToolDef(
                name="dispatch_workers",
                description="并行派遣多个 Worker 执行独立子任务。每个 Worker 是独立的智能体，有自己的工具和上下文。Worker 只能看到 context 中提供的信息。",
                parameters={"type": "object", "properties": {
                    "tasks": {"type": "array", "items": {"type": "string"}, "description": "自包含的子任务描述列表；Worker 看不到主对话历史"},
                    "context": {"type": "string", "description": "传递给所有 Worker 的共享背景信息"},
                }, "required": ["tasks"]},
                handler=lambda args: self._dispatch_workers_handler(args),
                usage_prompt="仅用于可并行、相互独立的只读子任务。tasks 必须自包含必要 ID 和条件；context 应放入共享背景。不要派发需要用户确认、写入或依赖主会话隐含历史的任务。",
                category="orchestration",
                policy=ToolPolicy(
                    read_only=True,
                    requires_confirmation=False,
                    concurrency_safe=False,
                    worker_allowed=False,
                ),
            ))

    def _dispatch_workers_handler(self, args: dict) -> str:
        tasks = args.get("tasks", [])
        if not tasks:
            return json.dumps({"error": "tasks 列表不能为空"}, ensure_ascii=False)

        context = args.get("context", "")
        results = self.dispatch_workers(tasks, context)

        summary = []
        for r in results:
            tools_used = ", ".join(tc["name"] for tc in r.get("tool_calls", []))
            summary.append({
                "worker": r["worker_id"],
                "task": r["task"],
                "status": r["status"],
                "tools_used": tools_used,
                "result": r["result"][:500],
            })
        return json.dumps(summary, ensure_ascii=False, default=str)
