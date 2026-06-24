"""运行时内置工具。

这些不是领域函数，而是 agent 控制原语：总结进展、向用户提问、派发独立
worker 子任务。
"""

from __future__ import annotations

import json
from typing import Callable

from ..llm.context import ContextManager
from ..runtime.tool_result_store import read_persisted_tool_result
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
            name="read_tool_result",
            description="读取 OAG 持久化的大型工具结果文件。默认只读取有限片段，避免把大型工具结果整包塞回上下文；仅用于读取工具返回中 persisted=true 的 path，不用于读取业务文档。",
            parameters={"type": "object", "properties": {
                "path": {"type": "string", "description": "persisted 工具结果中的 path"},
                "max_chars": {"type": "integer", "description": "最多返回字符数，默认12000，最多50000；只有确需完整核对时才显式加大"},
            }, "required": ["path"]},
            handler=lambda args: read_persisted_tool_result(
                path=args.get("path", ""),
                max_chars=args.get("max_chars", 12000) or 12000,
            ),
            usage_prompt="仅当某个工具结果返回 persisted=true 且预览不足以回答时调用。优先读取默认长度或小片段；不要为了综合回答无条件读取完整大结果。需要更多证据时，先定向调用领域工具缩小范围，或显式设置较小 max_chars 分段读取。不要用领域 read_document 读取 /tmp/oag-tool-results 或 tool-results 路径；业务文档仍使用领域自己的读取工具。",
            category="query",
            max_result_chars=20000,
            policy=ToolPolicy(
                read_only=True,
                requires_confirmation=False,
                concurrency_safe=True,
                worker_allowed=True,
                idempotent=True,
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
