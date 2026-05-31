"""工具元数据注册表。

ToolDef 同时描述模型可见的函数 schema 和 harness 策略；ToolRegistry 保存
这些定义，并转换为 OpenAI chat.completions 可用的 tools 规格。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolPolicy:
    read_only: bool = True
    requires_confirmation: bool = False
    concurrency_safe: bool = True
    worker_allowed: bool = True
    idempotent: bool = True
    destructive: bool = False
    timeout_seconds: float | None = 30.0


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict
    handler: Callable[[dict], str]
    usage_prompt: str = ""
    category: str = "query"
    is_read_only: bool = True
    requires_confirmation: bool = False
    max_result_chars: int = 5000
    policy: ToolPolicy | None = None

    def __post_init__(self):
        if self.policy is None:
            self.policy = ToolPolicy(
                read_only=self.is_read_only,
                requires_confirmation=self.requires_confirmation,
                concurrency_safe=self.is_read_only,
                worker_allowed=self.is_read_only,
                idempotent=self.is_read_only,
                destructive=not self.is_read_only,
            )
        else:
            self.is_read_only = self.policy.read_only
            self.requires_confirmation = self.policy.requires_confirmation


class ToolRegistry:

    def __init__(self):
        self._tools: dict[str, ToolDef] = {}
        self._version = 0
        self._built_tools_cache: list[dict] | None = None
        self._built_tools_version = -1

    def register(self, tool: ToolDef):
        self._tools[tool.name] = tool
        self._version += 1
        self._built_tools_cache = None

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    @property
    def version(self) -> int:
        return self._version

    def build_tools(self) -> list[dict]:
        if (
            self._built_tools_cache is not None
            and self._built_tools_version == self._version
        ):
            return self._built_tools_cache

        self._built_tools_cache = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": self._build_description(t),
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]
        self._built_tools_version = self._version
        return self._built_tools_cache

    def _build_description(self, tool: ToolDef) -> str:
        description = tool.description.strip()
        usage_prompt = tool.usage_prompt.strip()
        if not usage_prompt:
            return description
        if not description:
            return usage_prompt
        return f"{description}\n\n使用说明:\n{usage_prompt}"
