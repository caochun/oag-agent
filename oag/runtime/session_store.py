"""SQLite 会话历史存储。"""

from __future__ import annotations

import json
import sqlite3

from .message_sanitizer import sanitize_messages


class SessionStore:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS chat_history "
            "(session_id TEXT PRIMARY KEY, messages TEXT, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        self.conn.commit()
        self._cache: dict[str, list[dict]] = {}

    def get(self, session_id: str) -> list[dict]:
        if session_id in self._cache:
            return self._cache[session_id]
        row = self.conn.execute(
            "SELECT messages FROM chat_history WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row:
            messages = json.loads(row[0])
        else:
            messages = []
        messages, changed = sanitize_messages(messages)
        if changed:
            self.save(session_id, messages)
        self._cache[session_id] = messages
        return messages

    def save(self, session_id: str, messages: list[dict]):
        messages, _ = sanitize_messages(messages, repair_missing_tool_results=False)
        self._cache[session_id] = messages
        self.conn.execute(
            "INSERT OR REPLACE INTO chat_history (session_id, messages, updated_at) "
            "VALUES (?, ?, CURRENT_TIMESTAMP)",
            (session_id, json.dumps(messages, ensure_ascii=False, default=str)),
        )
        self.conn.commit()

    def list_sessions(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT session_id, updated_at FROM chat_history ORDER BY updated_at DESC"
        ).fetchall()
        return [{"session_id": r[0], "updated_at": r[1]} for r in rows]
