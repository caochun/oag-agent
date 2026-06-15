"""Conversation history protocol repair utilities."""

from __future__ import annotations

import json
from copy import deepcopy


def sanitize_messages(messages: list[dict], *,
                      repair_missing_tool_results: bool = True) -> tuple[list[dict], bool]:
    """Repair message history so it remains valid for tool-calling APIs.

    The sanitizer is intentionally conservative: it only fixes protocol issues
    that can make the next model request fail. It does not rewrite normal
    conversational content.
    """

    repaired: list[dict] = []
    changed = False
    i = 0

    while i < len(messages):
        msg = deepcopy(messages[i])
        i += 1
        role = msg.get("role")

        if role == "assistant":
            tool_calls = _valid_tool_calls(msg.get("tool_calls"))
            content = msg.get("content", "")
            if tool_calls:
                msg["tool_calls"] = tool_calls
                repaired.append(msg)

                call_ids = {tc["id"] for tc in tool_calls}
                seen_ids: set[str] = set()
                while i < len(messages):
                    next_msg = deepcopy(messages[i])
                    if next_msg.get("role") != "tool":
                        break
                    tool_call_id = next_msg.get("tool_call_id")
                    if tool_call_id not in call_ids or tool_call_id in seen_ids:
                        changed = True
                        i += 1
                        continue
                    repaired.append(next_msg)
                    seen_ids.add(tool_call_id)
                    i += 1

                missing_ids = call_ids - seen_ids
                if repair_missing_tool_results and missing_ids:
                    repaired.extend(_missing_tool_results(tool_calls, missing_ids))
                    changed = True
                elif missing_ids:
                    changed = True
                continue

            if not str(content or "").strip():
                changed = True
                continue

            repaired.append(msg)
            continue

        if role == "tool":
            changed = True
            continue

        repaired.append(msg)

    return repaired, changed


def _valid_tool_calls(tool_calls) -> list[dict]:
    if not isinstance(tool_calls, list):
        return []
    valid = []
    for tc in tool_calls:
        if not isinstance(tc, dict) or not tc.get("id"):
            continue
        fn = tc.get("function")
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        valid.append(tc)
    return valid


def _missing_tool_results(tool_calls: list[dict],
                          missing_ids: set[str]) -> list[dict]:
    return [
        {
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": json.dumps({
                "skipped": True,
                "reason": "历史恢复时发现缺失的工具结果，已自动补齐",
            }, ensure_ascii=False),
        }
        for tc in tool_calls
        if tc["id"] in missing_ids
    ]
