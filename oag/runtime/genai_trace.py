"""OpenTelemetry/OTLP GenAI trace export for evaluation evidence."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Iterator

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExportResult, SpanExporter, SpanProcessor
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags, set_span_in_context
from opentelemetry.trace.status import Status, StatusCode


OTLP_STATUS_UNSET = 0
OTLP_STATUS_OK = 1
OTLP_STATUS_ERROR = 2


@dataclass
class GenAISpan:
    span: trace.Span
    operation_name: str

    @property
    def trace_id(self) -> str:
        return _format_trace_id(self.span.get_span_context().trace_id)

    @property
    def span_id(self) -> str:
        return _format_span_id(self.span.get_span_context().span_id)

    def set_attribute(self, key: str, value: Any) -> None:
        if value is None:
            return
        self.span.set_attribute(key, _attribute_value(value))

    def finish(self, *, status_code: int = OTLP_STATUS_OK,
               status_message: str = "") -> None:
        self.span.set_status(_status(status_code, status_message))
        self.span.end()


class JsonFileSpanExporter(SpanExporter):
    def __init__(self, recorder: "GenAITraceRecorder"):
        self.recorder = recorder

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:
        self.recorder._record_readable_spans(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class ImmediateSpanProcessor(SpanProcessor):
    def __init__(self, exporter: SpanExporter):
        self.exporter = exporter

    def on_start(self, span: trace.Span, parent_context: Any | None = None) -> None:
        return None

    def on_end(self, span: ReadableSpan) -> None:
        self.exporter.export([span])

    def shutdown(self) -> None:
        self.exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self.exporter.force_flush(timeout_millis)


class GenAITraceRecorder:
    """Collect GenAI spans and export them as validator-compatible OTLP JSON."""

    def __init__(self, *, enabled: bool = True, json_path: str = "",
                 service_name: str = "oag-agent",
                 scope_name: str = "oag.genai_trace",
                 scope_version: str = "0.1.0",
                 provider_name: str = "openai"):
        self.enabled = enabled
        self.json_path = json_path
        self.service_name = service_name
        self.scope_name = scope_name
        self.scope_version = scope_version
        self.provider_name = provider_name
        self._spans: list[ReadableSpan] = []
        self._lock = Lock()
        self._provider = TracerProvider(
            resource=Resource.create({"service.name": service_name}),
        )
        self._provider.add_span_processor(
            ImmediateSpanProcessor(JsonFileSpanExporter(self)),
        )
        self._tracer = self._provider.get_tracer(scope_name, scope_version)

    @contextmanager
    def invocation(self, *, session_id: str, user_message: str,
                   run_id: str = "", model: str = "") -> Iterator[GenAISpan | None]:
        span_cm = self._start_span_context(
            "invoke_agent oag",
            "invoke_agent",
            force_new_trace=True,
        )
        span = span_cm.__enter__()
        if span is None:
            yield None
            span_cm.__exit__(None, None, None)
            return

        span.set_attribute("gen_ai.agent.name", self.service_name)
        span.set_attribute("gen_ai.conversation.id", session_id)
        span.set_attribute("oag.session.id", session_id)
        span.set_attribute("oag.run.id", run_id)
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.input.messages", [_text_message("user", user_message)])

        try:
            yield span
        except Exception as exc:
            span.set_attribute("error.type", type(exc).__name__)
            span.finish(status_code=OTLP_STATUS_ERROR, status_message=str(exc))
            raise
        finally:
            if span.span.is_recording():
                span.finish()
            span_cm.__exit__(None, None, None)

    @contextmanager
    def context(self, *, trace_id: str, span_id: str = "") -> Iterator[None]:
        token = otel_context.attach(_parent_context(trace_id, span_id))
        try:
            yield
        finally:
            otel_context.detach(token)

    def current_trace_id(self) -> str:
        span = trace.get_current_span()
        context = span.get_span_context()
        if not context.is_valid:
            return ""
        return _format_trace_id(context.trace_id)

    @contextmanager
    def chat(self, *, messages: list[dict], tools: list[dict] | None,
             model: str, temperature: float | None = None,
             max_tokens: int | None = None, stream: bool | None = None,
             trace_id: str = "", parent_span_id: str = "") -> Iterator[GenAISpan | None]:
        span_cm = self._start_span_context(
            "chat completion",
            "chat",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
        )
        span = span_cm.__enter__()
        if span is None:
            yield None
            span_cm.__exit__(None, None, None)
            return

        span.set_attribute("gen_ai.provider.name", self.provider_name)
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.request.temperature", temperature)
        span.set_attribute("gen_ai.request.max_tokens", max_tokens)
        span.set_attribute("gen_ai.request.stream", stream)
        span.set_attribute("gen_ai.input.messages", normalize_messages(messages))
        if tools:
            span.set_attribute("gen_ai.tool.definitions", normalize_tool_definitions(tools))

        try:
            yield span
        except Exception as exc:
            span.set_attribute("error.type", type(exc).__name__)
            span.finish(status_code=OTLP_STATUS_ERROR, status_message=str(exc))
            raise
        finally:
            if span.span.is_recording():
                span.finish()
            span_cm.__exit__(None, None, None)

    @contextmanager
    def tool_call(self, *, tool_name: str, args: dict,
                  tool_call_id: str = "",
                  trace_id: str = "",
                  parent_span_id: str = "") -> Iterator[GenAISpan | None]:
        span_cm = self._start_span_context(
            f"execute_tool {tool_name}",
            "execute_tool",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
        )
        span = span_cm.__enter__()
        if span is None:
            yield None
            span_cm.__exit__(None, None, None)
            return

        span.set_attribute("gen_ai.tool.name", tool_name)
        span.set_attribute("gen_ai.tool.type", "function")
        span.set_attribute("gen_ai.tool.call.id", tool_call_id)
        span.set_attribute("gen_ai.tool.call.arguments", args)

        try:
            yield span
        except Exception as exc:
            span.set_attribute("error.type", type(exc).__name__)
            span.set_attribute("gen_ai.tool.call.result", {"error": str(exc)})
            span.finish(status_code=OTLP_STATUS_ERROR, status_message=str(exc))
            raise
        finally:
            if span.span.is_recording():
                span.finish()
            span_cm.__exit__(None, None, None)

    @contextmanager
    def _start_span_context(self, name: str, operation_name: str, *,
                            trace_id: str = "",
                            parent_span_id: str = "",
                            force_new_trace: bool = False) -> Iterator[GenAISpan | None]:
        if not self.enabled:
            yield None
            return
        parent = None
        if trace_id:
            parent = _parent_context(trace_id, parent_span_id)
        elif force_new_trace:
            parent = trace.set_span_in_context(trace.INVALID_SPAN)
        with self._tracer.start_as_current_span(
            name,
            context=parent,
            end_on_exit=False,
        ) as sdk_span:
            wrapped = GenAISpan(sdk_span, operation_name)
            wrapped.set_attribute("gen_ai.operation.name", operation_name)
            yield wrapped

    def snapshot(self) -> list[ReadableSpan]:
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()
            self._write_json()

    def export_otlp(self) -> dict[str, Any]:
        with self._lock:
            return self._export_otlp_unlocked()

    def _write_json(self) -> None:
        if not self.json_path:
            return
        path = Path(self.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._export_otlp_unlocked(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _export_otlp_unlocked(self) -> dict[str, Any]:
        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            _attribute("service.name", self.service_name),
                        ],
                    },
                    "scopeSpans": [
                        {
                            "scope": {
                                "name": self.scope_name,
                                "version": self.scope_version,
                            },
                            "spans": [_readable_span_to_otlp(span) for span in self._spans],
                        }
                    ],
                }
            ],
        }

    def _record_readable_spans(self, spans: list[ReadableSpan]) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._spans.extend(spans)
            self._write_json()


def normalize_messages(messages: list[dict]) -> list[dict]:
    return [_normalize_message(message) for message in messages]


def normalize_tool_definitions(tools: list[dict]) -> list[dict]:
    definitions: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        definition = {
            "type": "function",
            "name": name,
            "description": str(function.get("description") or ""),
        }
        parameters = function.get("parameters")
        if isinstance(parameters, dict):
            definition["parameters"] = parameters
        definitions.append(definition)
    return definitions


def finish_reason_for_message(message: Any) -> str:
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        return "tool_call"
    finish_reason = getattr(message, "finish_reason", None)
    return str(finish_reason or "stop")


def _normalize_message(message: dict) -> dict:
    role = str(message.get("role") or "user")
    result = {
        "role": role,
        "parts": _message_parts(message),
    }
    if name := message.get("name"):
        result["name"] = str(name)
    return result


def _message_parts(message: dict) -> list[dict]:
    parts: list[dict] = []
    content = message.get("content")
    if content not in (None, ""):
        parts.append({"type": "text", "content": str(content)})
    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        parts.append({
            "type": "tool_call",
            "id": str(tool_call.get("id") or ""),
            "name": str(function.get("name") or ""),
            "arguments": _json_or_raw(function.get("arguments")),
        })
    if message.get("role") == "tool":
        parts.append({
            "type": "tool_call_response",
            "id": str(message.get("tool_call_id") or ""),
            "response": _json_or_raw(content),
        })
    if not parts:
        parts.append({"type": "text", "content": ""})
    return parts


def _text_message(role: str, content: str, *, finish_reason: str | None = None) -> dict:
    message = {
        "role": role,
        "parts": [{"type": "text", "content": content}],
    }
    if finish_reason:
        message["finish_reason"] = finish_reason
    return message


def _json_or_raw(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _attribute(key: str, value: Any) -> dict[str, Any]:
    return {
        "key": key,
        "value": _any_value(value),
    }


def _any_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, (list, dict)):
        return {"stringValue": json.dumps(value, ensure_ascii=False, default=str)}
    return {"stringValue": str(value)}


def _attribute_value(value: Any) -> str | bool | int | float:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _readable_span_to_otlp(span: ReadableSpan) -> dict[str, Any]:
    context = span.get_span_context()
    status_code, status_message = _otlp_status(span.status)
    result = {
        "name": span.name,
        "traceId": _format_trace_id(context.trace_id),
        "spanId": _format_span_id(context.span_id),
        "startTimeUnixNano": str(span.start_time or 0),
        "endTimeUnixNano": str(span.end_time or span.start_time or 0),
        "status": {"code": status_code},
        "attributes": [
            _attribute(key, value)
            for key, value in dict(span.attributes or {}).items()
        ],
    }
    if status_message:
        result["status"]["message"] = status_message
    if span.parent and span.parent.is_valid:
        result["parentSpanId"] = _format_span_id(span.parent.span_id)
    return result


def _parent_context(trace_id: str, span_id: str = ""):
    trace_int = int(trace_id, 16) if trace_id else 0
    span_int = int(span_id, 16) if span_id else 0
    parent = SpanContext(
        trace_id=trace_int,
        span_id=span_int,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
        trace_state={},
    )
    return set_span_in_context(NonRecordingSpan(parent))


def _status(status_code: int, message: str = "") -> Status:
    if status_code == OTLP_STATUS_ERROR:
        return Status(StatusCode.ERROR, message or None)
    if status_code == OTLP_STATUS_OK:
        return Status(StatusCode.OK, message or None)
    return Status(StatusCode.UNSET, message or None)


def _otlp_status(status: Status) -> tuple[int, str]:
    if status.status_code == StatusCode.ERROR:
        return OTLP_STATUS_ERROR, status.description or ""
    if status.status_code == StatusCode.OK:
        return OTLP_STATUS_OK, status.description or ""
    return OTLP_STATUS_UNSET, status.description or ""


def _format_trace_id(trace_id: int) -> str:
    return f"{trace_id:032x}"


def _format_span_id(span_id: int) -> str:
    return f"{span_id:016x}"
