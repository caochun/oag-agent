"""Harness 运行时配置。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HarnessConfig:
    max_turns: int = 10
    max_response_tokens: int = 2048
    enable_audit: bool = True
    enable_write_confirmation: bool = True
    enable_worker_dispatch: bool = False
    enable_tool_result_reader: bool = False
    custom_system_prompt: str | None = None
    append_system_prompt: str = ""
    runtime_context: dict[str, str] = field(default_factory=dict)
    llm_extra_body: dict[str, object] = field(default_factory=dict)
    trace_jsonl_path: str = ""
