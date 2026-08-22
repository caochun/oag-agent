"""Normalize complete and streaming OpenAI-compatible model responses."""

from __future__ import annotations

from types import SimpleNamespace

from ..runtime.events import (
    AssistantDeltaEvent,
    DebugEvent,
    ReasoningEvent,
)

MAX_REASONING_CHARS = 5000


class LlmResponseParser:
    """Turn provider responses into events and one normalized message object."""

    def consume(self, response):
        if hasattr(response, "choices") and response.choices:
            message = response.choices[0].message
            yield self.build_debug_event(message)
            reasoning = self.extract_reasoning(message)
            if reasoning:
                yield ReasoningEvent(content=reasoning)
            if message.content:
                yield AssistantDeltaEvent(content=message.content)
            return message

        content_parts: list[str] = []
        reasoning_chars = 0
        tool_call_parts: dict[int, dict] = {}

        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = self.extract_reasoning(delta)
            if reasoning:
                remaining = MAX_REASONING_CHARS - reasoning_chars
                if remaining > 0:
                    emitted = reasoning[:remaining]
                    reasoning_chars += len(emitted)
                    yield ReasoningEvent(content=emitted)
                    if len(reasoning) > remaining:
                        yield ReasoningEvent(content="\n[... reasoning 已截断]")
                        reasoning_chars = MAX_REASONING_CHARS

            if delta.content:
                content_parts.append(delta.content)
                yield AssistantDeltaEvent(content=delta.content)

            for tool_call_delta in delta.tool_calls or []:
                index = tool_call_delta.index
                entry = tool_call_parts.setdefault(
                    index,
                    {"id": "", "type": "function", "name": "", "arguments": []},
                )
                if tool_call_delta.id:
                    entry["id"] = tool_call_delta.id
                if tool_call_delta.type:
                    entry["type"] = tool_call_delta.type
                if tool_call_delta.function:
                    if tool_call_delta.function.name:
                        entry["name"] = tool_call_delta.function.name
                    if tool_call_delta.function.arguments:
                        entry["arguments"].append(tool_call_delta.function.arguments)

        content = "".join(content_parts)
        tool_calls = [
                SimpleNamespace(
                    id=entry["id"],
                    type=entry["type"],
                    function=SimpleNamespace(
                        name=entry["name"],
                        arguments="".join(entry["arguments"]),
                    ),
                )
                for _, entry in sorted(tool_call_parts.items())
                if entry["name"]
            ] or None
        message = SimpleNamespace(
            content=content,
            tool_calls=tool_calls,
        )
        yield self.build_debug_event(message)
        return message

    @staticmethod
    def extract_reasoning(message) -> str:
        reasoning = getattr(message, "reasoning_content", None)
        if not reasoning:
            extra = getattr(message, "model_extra", None) or {}
            reasoning = extra.get("reasoning_content")
        return str(reasoning) if reasoning else ""

    @staticmethod
    def build_debug_event(message) -> DebugEvent:
        summary = ""
        if message.tool_calls:
            calls = [
                f"{call.function.name}({call.function.arguments[:80]})"
                for call in message.tool_calls
            ]
            summary = "LLM选择调用: " + "; ".join(calls)
        if message.content:
            summary += f"\nLLM文本: {message.content[:300]}"
        return DebugEvent(stage="response", content=summary)
