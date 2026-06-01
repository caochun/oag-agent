"""内存执行轨迹记录器。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
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
    def __init__(self, enabled: bool = True, jsonl_path: str = ""):
        self.enabled = enabled
        self.jsonl_path = jsonl_path
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
            self._write_jsonl(event)
        return event

    def snapshot(self) -> list[TraceEvent]:
        with self._lock:
            return list(self._events)

    def clear(self):
        with self._lock:
            self._events.clear()

    def _write_jsonl(self, event: TraceEvent):
        if not self.jsonl_path:
            return

        path = Path(self.jsonl_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            "session_id": event.session_id,
            "source": event.source,
            "turn_count": event.turn_count,
            "payload": event.payload,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
