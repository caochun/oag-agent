"""Compatibility parser for models that emit tool calls as DSML text."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass


_TOKEN = r"(?:｜｜DSML｜｜|\|\|DSML\|\|)"
_TOOL_CALLS_RE = re.compile(
    rf"<{_TOKEN}tool_calls>\s*(?P<body>.*?)\s*</{_TOKEN}tool_calls>",
    re.DOTALL,
)
_INVOKE_RE = re.compile(
    rf"<{_TOKEN}invoke\b(?P<attrs>[^>]*)>(?P<body>.*?)</{_TOKEN}invoke>",
    re.DOTALL,
)
_PARAMETER_RE = re.compile(
    rf"<{_TOKEN}parameter\b(?P<attrs>[^>]*)>(?P<value>.*?)</{_TOKEN}parameter>",
    re.DOTALL,
)
_ATTRIBUTE_RE = re.compile(r'([\w:-]+)\s*=\s*"([^"]*)"')

DSML_TOOL_CALL_OPENERS = (
    "<｜｜DSML｜｜tool_calls>",
    "<||DSML||tool_calls>",
)


@dataclass(frozen=True)
class DSMLToolCall:
    id: str
    name: str
    arguments: str


def extract_dsml_tool_calls(content: str) -> tuple[str, list[DSMLToolCall]]:
    """Return user-facing text and parsed tool calls from a DSML response."""
    calls: list[DSMLToolCall] = []

    for tool_block in _TOOL_CALLS_RE.finditer(content):
        for invoke_index, invoke in enumerate(_INVOKE_RE.finditer(tool_block.group("body"))):
            invoke_attrs = _attributes(invoke.group("attrs"))
            name = invoke_attrs.get("name", "").strip()
            if not name:
                continue

            arguments: dict[str, object] = {}
            valid = True
            for parameter in _PARAMETER_RE.finditer(invoke.group("body")):
                parameter_attrs = _attributes(parameter.group("attrs"))
                parameter_name = parameter_attrs.get("name", "").strip()
                if not parameter_name:
                    valid = False
                    break
                raw_value = parameter.group("value").strip()
                if parameter_attrs.get("string", "").lower() == "true":
                    arguments[parameter_name] = raw_value
                    continue
                try:
                    arguments[parameter_name] = json.loads(raw_value)
                except json.JSONDecodeError:
                    valid = False
                    break

            if not valid:
                continue
            digest = hashlib.sha256(invoke.group(0).encode("utf-8")).hexdigest()[:12]
            calls.append(DSMLToolCall(
                id=f"dsml_{invoke_index}_{digest}",
                name=name,
                arguments=json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
            ))

    if not calls:
        return content, []
    return _TOOL_CALLS_RE.sub("", content).strip(), calls


class DSMLTextFilter:
    """Suppress DSML control blocks without delaying normal streamed text."""

    def __init__(self) -> None:
        self._pending = ""
        self._inside_tool_calls = False

    def feed(self, content: str) -> str:
        if self._inside_tool_calls:
            return ""
        self._pending += content

        for opener in DSML_TOOL_CALL_OPENERS:
            index = self._pending.find(opener)
            if index >= 0:
                visible = self._pending[:index]
                self._pending = ""
                self._inside_tool_calls = True
                return visible

        possible_start = self._pending.rfind("<")
        if possible_start >= 0:
            candidate = self._pending[possible_start:]
            if any(opener.startswith(candidate) for opener in DSML_TOOL_CALL_OPENERS):
                visible = self._pending[:possible_start]
                self._pending = candidate
                return visible

        visible = self._pending
        self._pending = ""
        return visible

    def flush(self) -> str:
        if self._inside_tool_calls:
            return ""
        visible = self._pending
        self._pending = ""
        return visible


def _attributes(value: str) -> dict[str, str]:
    return {name: content for name, content in _ATTRIBUTE_RE.findall(value)}
