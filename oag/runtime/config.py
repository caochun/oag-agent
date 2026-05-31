"""Harness 运行时配置。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HarnessConfig:
    max_turns: int = 10
    max_tool_result_chars: int = 5000
    enable_audit: bool = True
    enable_write_confirmation: bool = True
    custom_system_prompt: str | None = None
    append_system_prompt: str = ""
    runtime_context: dict[str, str] = field(default_factory=dict)
    include_ontology_full_context: bool = False
