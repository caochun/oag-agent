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
    known_tool_call_ids: set[str] = set()
    satisfied_tool_call_ids: set[str] = set()

    for raw_msg in messages:
        msg = deepcopy(raw_msg)
        role = msg.get("role")

        if role == "assistant":
            tool_calls = _valid_tool_calls(msg.get("tool_calls"))
            content = msg.get("content", "")
            if tool_calls:
                msg["tool_calls"] = tool_calls
                for tc in tool_calls:
                    known_tool_call_ids.add(tc["id"])
                repaired.append(msg)
                continue

            if not str(content or "").strip():
                changed = True
                continue

            repaired.append(msg)
            continue

        if role == "tool":
            tool_call_id = msg.get("tool_call_id")
            if not tool_call_id or tool_call_id not in known_tool_call_ids:
                changed = True
                continue
            if tool_call_id in satisfied_tool_call_ids:
                changed = True
                continue
            satisfied_tool_call_ids.add(tool_call_id)
            repaired.append(msg)
            continue

        repaired.append(msg)

    missing_ids = known_tool_call_ids - satisfied_tool_call_ids
    if repair_missing_tool_results and missing_ids:
        repaired = _append_missing_tool_results(repaired, missing_ids)
        changed = True

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


def _append_missing_tool_results(messages: list[dict],
                                 missing_ids: set[str]) -> list[dict]:
    result: list[dict] = []
    pending_missing = set(missing_ids)
    i = 0

    while i < len(messages):
        msg = messages[i]
        result.append(msg)
        i += 1

        if msg.get("role") != "assistant" or not _valid_tool_calls(msg.get("tool_calls")):
            continue

        tool_calls = _valid_tool_calls(msg.get("tool_calls"))
        call_ids = {tc["id"] for tc in tool_calls}
        existing_result_ids = set()

        while i < len(messages):
            next_msg = messages[i]
            if next_msg.get("role") != "tool" or next_msg.get("tool_call_id") not in call_ids:
                break
            existing_result_ids.add(next_msg["tool_call_id"])
            result.append(next_msg)
            i += 1

        for tc in tool_calls:
            tool_call_id = tc["id"]
            if tool_call_id not in pending_missing or tool_call_id in existing_result_ids:
                continue
            result.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps({
                    "skipped": True,
                    "reason": "历史恢复时发现缺失的工具结果，已自动补齐",
                }, ensure_ascii=False),
            })
            pending_missing.remove(tool_call_id)

    return result
