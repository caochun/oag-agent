"""内存执行轨迹记录器。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass(frozen=True)
class TraceEvent:
    event_type: str
    session_id: str = ""
    source: str = "main"
    turn_count: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class TraceRecorder:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._events: list[TraceEvent] = []
        self._lock = Lock()

    def record(self, event_type: str, *,
               session_id: str = "",
               source: str = "main",
               turn_count: int | None = None,
               **payload: Any) -> TraceEvent | None:
        if not self.enabled:
            return None

        event = TraceEvent(
            event_type=event_type,
            session_id=session_id,
            source=source,
            turn_count=turn_count,
            payload=payload,
        )
        with self._lock:
            self._events.append(event)
        return event

    def snapshot(self) -> list[TraceEvent]:
        with self._lock:
            return list(self._events)

    def clear(self):
        with self._lock:
            self._events.clear()
