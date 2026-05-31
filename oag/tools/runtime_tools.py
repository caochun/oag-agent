"""运行时内置工具。

这些不是领域函数，而是 agent 控制原语：总结进展、向用户提问、派发独立
worker 子任务。
"""

from __future__ import annotations

import json
from typing import Callable

from ..llm.context import ContextManager
from .registry import ToolDef, ToolPolicy, ToolRegistry


GetMessages = Callable[[], list[dict] | None]
DispatchWorkers = Callable[[list[str], str], list[dict]]


class RuntimeTools:
    def __init__(self, *,
                 context_mgr: ContextManager,
                 get_current_messages: GetMessages,
                 dispatch_workers: DispatchWorkers):
        self.context_mgr = context_mgr
        self.get_current_messages = get_current_messages
        self.dispatch_workers = dispatch_workers

    def register(self, tools: ToolRegistry):
        tools.register(ToolDef(
            name="summarize_progress",
            description="总结当前对话进展。返回已完成的操作摘要、使用的工具统计。适合长对话中回顾进度",
            parameters={"type": "object", "properties": {}},
            handler=lambda args: self._summarize_progress_handler(args),
            usage_prompt="只在长对话、上下文复杂或用户询问进度时调用。不要用它替代最终回答。",
            category="query",
        ))

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
            requires_confirmation=True,
            policy=ToolPolicy(
                read_only=True,
                requires_confirmation=True,
                concurrency_safe=False,
                worker_allowed=False,
                idempotent=False,
            ),
        ))

        tools.register(ToolDef(
            name="dispatch_workers",
            description="并行派遣多个 Worker 执行独立子任务。每个 Worker 是独立的智能体，有自己的工具和上下文。Worker 只能看到 context 中提供的信息。",
            parameters={"type": "object", "properties": {
                "tasks": {"type": "array", "items": {"type": "string"}, "description": "子任务描述列表。每条须包含完整信息（事件ID、设施ID等），Worker 看不到你的对话历史"},
                "context": {"type": "string", "description": "传递给所有 Worker 的背景信息，如事件详情、已查到的设施列表等"},
            }, "required": ["tasks"]},
            handler=lambda args: self._dispatch_workers_handler(args),
            usage_prompt="仅用于可并行、相互独立的只读子任务。tasks 必须自包含必要 ID 和条件；context 应放入共享背景。不要派发需要用户确认、写入或依赖主会话隐含历史的任务。",
            category="action",
            policy=ToolPolicy(
                read_only=False,
                requires_confirmation=False,
                concurrency_safe=False,
                worker_allowed=False,
                idempotent=False,
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
            status_icon = "✓" if r["status"] == "success" else "✗"
            tools_used = ", ".join(tc["name"] for tc in r.get("tool_calls", []))
            summary.append({
                "worker": r["worker_id"],
                "task": r["task"],
                "status": status_icon,
                "tools_used": tools_used,
                "result": r["result"][:500],
            })
        return json.dumps(summary, ensure_ascii=False, default=str)

    def _summarize_progress_handler(self, args: dict) -> str:
        messages = self.get_current_messages()
        if not messages or len(messages) < 2:
            return json.dumps({"error": "对话历史过短，无需总结"}, ensure_ascii=False)

        tool_names_used: list[str] = []
        for m in messages:
            for tc in m.get("tool_calls", []):
                if isinstance(tc, dict):
                    tool_names_used.append(tc["function"]["name"])

        summary_text = self.context_mgr._summarize(messages[1:])
        return json.dumps({
            "summary": summary_text,
            "total_messages": len(messages),
            "tool_calls_count": len(tool_names_used),
            "tools_used": sorted(set(tool_names_used)),
        }, ensure_ascii=False)
