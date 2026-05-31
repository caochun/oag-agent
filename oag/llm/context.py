"""对话上下文管理。

本模块负责估算消息长度、截断过大的工具结果，并在上下文过长时压缩历史。
压缩时会尽量保留 system prompt 和最近若干轮，旧消息才会被摘要。
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from .retry import call_llm_with_retry

COMPACT_PROMPT = """\
请将以下对话历史浓缩为一段简洁的摘要，保留关键信息：
- 用户询问了什么
- 系统执行了哪些操作
- 得到了什么关键结果和数据
- 任何重要的上下文信息

对话历史：
{history}

请用中文输出摘要，300字以内："""


def estimate_tokens(text: str) -> int:
    chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)


def count_messages_tokens(messages: list[dict]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            if isinstance(tc, dict):
                fn = tc.get("function", {})
                total += estimate_tokens(fn.get("name", "") + fn.get("arguments", ""))
    return total


def truncate_tool_result(result: str, max_chars: int = 5000) -> str:
    if len(result) <= max_chars:
        return result
    return result[:max_chars] + f"\n[... 截断，原始长度 {len(result)} 字符]"


class ContextManager:
    def __init__(self, llm_client: OpenAI, model: str,
                 context_window: int = 128000):
        self.client = llm_client
        self.model = model
        self.context_window = context_window
        self.micro_threshold = int(context_window * 0.60)
        self.compact_threshold = int(context_window * 0.75)

    def maybe_compact(self, messages: list[dict]) -> tuple[list[dict], bool]:
        token_count = count_messages_tokens(messages)

        # 先尝试本地轻量压缩，避免不必要的 LLM 摘要调用。
        if token_count >= self.micro_threshold and token_count < self.compact_threshold:
            messages = self._micro_compact(messages)
            if count_messages_tokens(messages) < self.micro_threshold:
                return messages, True

        if token_count < self.compact_threshold:
            return messages, False

        messages = self._micro_compact(messages)
        if count_messages_tokens(messages) < self.compact_threshold:
            return messages, True

        if len(messages) <= 4:
            return messages, False

        system_msg = messages[0] if messages[0].get("role") == "system" else None
        start = 1 if system_msg else 0
        split_index = max(start, len(messages) - min(6, len(messages) - 1))
        split_index = self._adjust_split_for_tool_pairs(messages, split_index, start)
        old_messages = messages[start:split_index]
        recent_messages = messages[split_index:]

        if not old_messages:
            return messages, False

        # system prompt 是长期约束，压缩时只摘要旧对话，不改写 system prompt。
        summary = self._summarize(old_messages)
        compacted = []
        if system_msg:
            compacted.append(system_msg)
        compacted.append({
            "role": "user",
            "content": f"[前置对话摘要]\n{summary}",
        })
        compacted.append({
            "role": "assistant",
            "content": "好的，我已了解前面的对话内容。请继续。",
        })
        compacted.extend(recent_messages)

        return compacted, True

    def force_compact(self, messages: list[dict]) -> tuple[list[dict], bool]:
        messages = self._micro_compact(messages)
        if len(messages) <= 4:
            return messages, False

        system_msg = messages[0] if messages[0].get("role") == "system" else None
        start = 1 if system_msg else 0
        split_index = max(start, len(messages) - min(4, len(messages) - 1))
        split_index = self._adjust_split_for_tool_pairs(messages, split_index, start)

        old_messages = messages[start:split_index]
        recent_messages = messages[split_index:]
        if not old_messages:
            return messages, False

        summary = self._summarize(old_messages)
        compacted = []
        if system_msg:
            compacted.append(system_msg)
        compacted.append({
            "role": "user",
            "content": f"[前置对话摘要]\n{summary}",
        })
        compacted.append({
            "role": "assistant",
            "content": "好的，我已了解前面的对话内容。请继续。",
        })
        compacted.extend(recent_messages)
        return compacted, True

    def _micro_compact(self, messages: list[dict]) -> list[str]:
        tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        protect = set(tool_indices[-3:]) if len(tool_indices) >= 3 else set(tool_indices)

        for i in tool_indices:
            if i in protect:
                continue
            content = messages[i].get("content", "")
            if len(content) > 500:
                messages[i] = {
                    **messages[i],
                    "content": content[:200] + "\n[... 结果已压缩]",
                }
        return messages

    def _adjust_split_for_tool_pairs(self, messages: list[dict],
                                     split_index: int,
                                     min_index: int) -> int:
        needed_ids = {
            m.get("tool_call_id")
            for m in messages[split_index:]
            if m.get("role") == "tool" and m.get("tool_call_id")
        }
        if not needed_ids:
            return split_index

        present_ids = self._assistant_tool_call_ids(messages[split_index:])
        missing_ids = needed_ids - present_ids
        adjusted = split_index

        for i in range(split_index - 1, min_index - 1, -1):
            if not missing_ids:
                break
            ids = self._assistant_tool_call_ids([messages[i]])
            if ids & missing_ids:
                adjusted = i
                missing_ids -= ids

        return adjusted

    def _assistant_tool_call_ids(self, messages: list[dict]) -> set[str]:
        ids: set[str] = set()
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls", []) or []:
                if isinstance(tc, dict) and tc.get("id"):
                    ids.add(tc["id"])
        return ids

    def _summarize(self, messages: list[dict]) -> str:
        history_parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                if role == "tool":
                    content = content[:500] + ("..." if len(content) > 500 else "")
                history_parts.append(f"[{role}] {content}")

        history_text = "\n".join(history_parts)
        if len(history_text) > 8000:
            history_text = history_text[:8000] + "\n[... 更早的历史已省略]"

        prompt = COMPACT_PROMPT.format(history=history_text)

        try:
            response = call_llm_with_retry(
                self.client, max_retries=3,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )
            return response.choices[0].message.content or "(摘要生成失败)"
        except Exception as e:
            return f"(摘要生成失败: {e})"

    def needs_compact(self, messages: list[dict]) -> bool:
        return count_messages_tokens(messages) >= self.compact_threshold
