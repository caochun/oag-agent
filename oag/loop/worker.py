"""并行 Worker 智能体。

Worker 用于执行彼此独立的子任务：它只能看到父任务显式传入的 context，并且
会使用经过过滤的工具列表。所有工具调用仍然经过 Harness 策略约束。
"""

from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from typing import Any

from openai import OpenAI

from ..llm.retry import call_llm_with_retry
from ..runtime import ToolUseContext
from ..tools.pipeline import ToolResult


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
        self.cache_namespace = f"worker-run:{uuid.uuid4().hex}"

    def run(self, task: str) -> dict:
        system = self.harness.build_worker_system_prompt(self.worker_id, self.context)

        tools = self.harness.build_worker_tools()

        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]

        tool_calls_log: list[dict] = []

        for turn_count in range(1, self.max_turns + 1):
            request_kwargs = {
                "model": self.model,
                "messages": messages,
                "tools": tools if tools else None,
                "temperature": 0.1,
                "max_tokens": self.harness.config.max_response_tokens,
            }
            if self.harness.config.llm_extra_body:
                request_kwargs["extra_body"] = self.harness.config.llm_extra_body
            response = call_llm_with_retry(self.client, **request_kwargs)
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
                        context=ToolUseContext(
                            source="worker",
                            turn_count=turn_count,
                            confirmed=False,
                            cache_namespace=self.cache_namespace,
                        ),
                    )
                except (json.JSONDecodeError, ValueError) as exc:
                    args = {}
                    result = ToolResult(
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
            future = pool.submit(copy_context().run, worker.run, task)
            futures[future] = (i, worker.cache_namespace)

        for future in as_completed(futures):
            idx, cache_namespace = futures[future]
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
            finally:
                harness.clear_tool_cache_namespace(cache_namespace)

    return results
