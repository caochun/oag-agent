"""Context window usage estimation.

This module is intentionally domain-agnostic. It only inspects the prompt,
tool schemas, and chat messages that the agent runtime is about to send to
the LLM.
"""

from __future__ import annotations

import json
from typing import Any

from .context import count_messages_tokens, estimate_tokens

DEFAULT_CONTEXT_WINDOW = 128000


def collect_context_usage(
    messages: list[dict],
    tools: list[dict] | None = None,
    *,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    model: str = "",
) -> dict[str, Any]:
    """Return a structured token estimate for the current LLM request."""

    tools = tools or []
    message_breakdown = _message_breakdown(messages)
    tool_schema_breakdown = _tool_schema_breakdown(tools)

    system_prompt_tokens = message_breakdown["system_message_tokens"]
    message_tokens = message_breakdown["total_tokens"]
    non_system_message_tokens = max(0, message_tokens - system_prompt_tokens)
    tool_schema_tokens = tool_schema_breakdown["total_tokens"]
    total_tokens = message_tokens + tool_schema_tokens
    free_tokens = max(0, context_window - total_tokens)

    categories = [
        {
            "name": "System prompt",
            "tokens": system_prompt_tokens,
        },
        {
            "name": "Tool schemas",
            "tokens": tool_schema_tokens,
        },
        {
            "name": "Messages",
            "tokens": non_system_message_tokens,
        },
        {
            "name": "Free space",
            "tokens": free_tokens,
        },
    ]

    return {
        "model": model,
        "context_window": context_window,
        "total_tokens": total_tokens,
        "free_tokens": free_tokens,
        "percentage": round((total_tokens / context_window) * 100, 2) if context_window else 0,
        "categories": categories,
        "messages": message_breakdown,
        "tools": tool_schema_breakdown,
    }


def _message_breakdown(messages: list[dict]) -> dict[str, Any]:
    total_tokens = count_messages_tokens(messages)
    role_tokens = {
        "system": 0,
        "user": 0,
        "assistant": 0,
        "tool": 0,
        "other": 0,
    }
    tool_call_tokens = 0
    tool_result_tokens = 0
    tool_results: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []

    for index, msg in enumerate(messages):
        role = msg.get("role", "other")
        if role not in role_tokens:
            role = "other"

        content = msg.get("content", "")
        content_tokens = estimate_tokens(content) if isinstance(content, str) else 0
        role_tokens[role] += content_tokens

        if msg.get("role") == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            tool_result_tokens += content_tokens
            tool_results.append({
                "index": index,
                "tool_call_id": tool_call_id,
                "tokens": content_tokens,
                "chars": len(content) if isinstance(content, str) else 0,
            })

        for tc in msg.get("tool_calls", []) or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", {})
            name = str(fn.get("name", ""))
            arguments = str(fn.get("arguments", ""))
            tokens = estimate_tokens(name + arguments)
            tool_call_tokens += tokens
            tool_calls.append({
                "name": name,
                "tokens": tokens,
                "argument_chars": len(arguments),
            })

    tool_results.sort(key=lambda item: item["tokens"], reverse=True)
    tool_calls.sort(key=lambda item: item["tokens"], reverse=True)

    return {
        "count": len(messages),
        "total_tokens": total_tokens,
        "system_message_tokens": role_tokens["system"],
        "user_message_tokens": role_tokens["user"],
        "assistant_message_tokens": role_tokens["assistant"],
        "tool_result_tokens": tool_result_tokens,
        "tool_call_tokens": tool_call_tokens,
        "other_message_tokens": role_tokens["other"],
        "largest_tool_results": tool_results[:5],
        "largest_tool_calls": tool_calls[:5],
    }


def _tool_schema_breakdown(tools: list[dict]) -> dict[str, Any]:
    tool_details = []
    total_tokens = 0

    for tool in tools:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = str(function.get("name", ""))
        schema_text = json.dumps(tool, ensure_ascii=False, sort_keys=True, default=str)
        tokens = estimate_tokens(schema_text)
        total_tokens += tokens
        tool_details.append({
            "name": name,
            "tokens": tokens,
            "chars": len(schema_text),
        })

    tool_details.sort(key=lambda item: item["tokens"], reverse=True)
    return {
        "count": len(tools),
        "total_tokens": total_tokens,
        "largest_tools": tool_details[:10],
    }
