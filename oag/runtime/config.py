"""Harness 运行时配置。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HarnessConfig:
    max_turns: int = 10
    max_tool_result_chars: int = 5000
    enable_audit: bool = True
    enable_write_confirmation: bool = True
