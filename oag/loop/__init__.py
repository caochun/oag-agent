"""对话循环包。

loop 包包含在线编排路径：主查询循环、工具调用排序和并发、用户确认后的
继续执行，以及面向独立子任务的并行 worker。这里不直接实现领域逻辑。
"""

__all__ = [
    "ConfirmationFlow",
    "QueryLoop",
    "ToolExecutor",
    "Worker",
    "run_workers_parallel",
]


def __getattr__(name: str):
    if name == "ConfirmationFlow":
        from .confirmation_flow import ConfirmationFlow

        return ConfirmationFlow
    if name == "QueryLoop":
        from .query_loop import QueryLoop

        return QueryLoop
    if name == "ToolExecutor":
        from .tool_executor import ToolExecutor

        return ToolExecutor
    if name == "Worker":
        from .worker import Worker

        return Worker
    if name == "run_workers_parallel":
        from .worker import run_workers_parallel

        return run_workers_parallel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
