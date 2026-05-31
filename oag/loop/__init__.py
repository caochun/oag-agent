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
