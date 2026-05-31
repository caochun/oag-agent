"""流式事件类型。

Agent.chat_stream 产出这些 dataclass 事件；chat_stream_sse 会把它们转换成
web 前端更容易消费的 dict。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Event:
    type: str


@dataclass
class TextEvent(Event):
    type: str = "text"
    content: str = ""


@dataclass
class ToolCallEvent(Event):
    type: str = "tool_call"
    name: str = ""
    args: dict = field(default_factory=dict)
    result: str = ""


@dataclass
class CompactEvent(Event):
    type: str = "compact"
    before_tokens: int = 0
    after_tokens: int = 0


@dataclass
class HookBlockedEvent(Event):
    type: str = "hook_blocked"
    hook_event: str = ""
    reason: str = ""


@dataclass
class ConfirmationEvent(Event):
    type: str = "confirmation_required"
    tool_name: str = ""
    args: dict = field(default_factory=dict)
    reason: str = ""


@dataclass
class QuestionEvent(Event):
    type: str = "question"
    question: str = ""
    options: list[dict] = field(default_factory=list)
    multi_select: bool = False


@dataclass
class DebugEvent(Event):
    type: str = "debug"
    stage: str = ""
    content: str = ""


@dataclass
class ReasoningEvent(Event):
    type: str = "reasoning"
    content: str = ""


def event_to_dict(event: Event) -> dict:
    result = {"type": event.type}
    for k, v in event.__dict__.items():
        if k == "type":
            continue
        if v or v == 0 or v is False:
            result[k] = v
    return result
