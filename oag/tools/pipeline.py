from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from ..llm.context import truncate_tool_result
from ..ontology.runtime import OntologyRuntime
from ..runtime import ToolUseContext, TraceRecorder
from ..runtime.hooks import AuditLog, HookRegistry, HookResult
from .registry import ToolDef, ToolRegistry


@dataclass
class ToolResult:
    content: str
    raw_content: str = ""
    truncated: bool = False
    blocked: bool = False
    block_reason: str = ""
    needs_confirmation: bool = False
    context_note: str = ""


class ToolExecutionPipeline:
    def __init__(self, *,
                 tools: ToolRegistry,
                 ontology_runtime: OntologyRuntime,
                 hooks: HookRegistry,
                 audit: AuditLog,
                 cache: dict[str, ToolResult],
                 trace: TraceRecorder,
                 progressive_context_enabled: Callable[[], bool],
                 set_current_messages: Callable[[list[dict] | None], None]):
        self.tools = tools
        self.ont = ontology_runtime
        self.hooks = hooks
        self.audit = audit
        self.cache = cache
        self.trace = trace
        self.progressive_context_enabled = progressive_context_enabled
        self.set_current_messages = set_current_messages

    def execute(self, tool_name: str, args: dict, context: ToolUseContext) -> ToolResult:
        tool = self.tools.get(tool_name)
        if not tool:
            self.trace.record(
                "tool_unknown",
                session_id=context.session_id,
                source=context.source,
                tool_name=tool_name,
            )
            return ToolResult(content=json.dumps({"error": f"未知工具: {tool_name}"}, ensure_ascii=False))

        self.trace.record(
            "tool_start",
            session_id=context.session_id,
            source=context.source,
            tool_name=tool_name,
            args=args,
            confirmed=context.confirmed,
        )

        if result := self._enforce_tool_policy(tool_name, tool, context):
            self._record_tool_result("tool_blocked", tool_name, context, result)
            return result

        if result := self._validate_mutation(tool_name, args, context):
            self._record_tool_result("tool_blocked", tool_name, context, result)
            return result

        if result := self._run_pre_tool_hooks(tool_name, args, tool, context):
            event_type = "tool_confirmation_required" if result.needs_confirmation else "tool_blocked"
            self._record_tool_result(event_type, tool_name, context, result)
            return result

        if result := self._maybe_pause_for_user_question(tool_name, args, tool, context):
            self._record_tool_result("tool_confirmation_required", tool_name, context, result)
            return result

        if result := self._get_cached_result(tool_name, args, tool, context):
            self._record_tool_result("tool_cache_hit", tool_name, context, result)
            return result

        if result := self._check_constraints(tool_name, args, context):
            self._record_tool_result("tool_blocked", tool_name, context, result)
            return result

        result = self._execute_handler(tool_name, args, tool, context)
        self._store_cache_result(tool_name, args, tool, result)

        if tool_name == "mutate" and not result.blocked:
            self.cache.clear()

        self._record_tool_result("tool_end", tool_name, context, result)
        return result

    def _enforce_tool_policy(self, tool_name: str, tool: ToolDef,
                             context: ToolUseContext) -> ToolResult | None:
        policy = tool.policy
        if context.source != "worker":
            return None

        if not policy.worker_allowed:
            reason = f"工具 {tool_name} 不允许由 Worker 执行"
            return ToolResult(
                content=json.dumps({"blocked": True, "reason": reason}, ensure_ascii=False),
                blocked=True,
                block_reason=reason,
            )

        if policy.requires_confirmation and not context.confirmed:
            reason = f"工具 {tool_name} 需要主会话确认，Worker 不可直接执行"
            return ToolResult(
                content=json.dumps({"blocked": True, "reason": reason}, ensure_ascii=False),
                blocked=True,
                block_reason=reason,
            )
        return None

    def _validate_mutation(self, tool_name: str, args: dict,
                           context: ToolUseContext) -> ToolResult | None:
        if tool_name != "mutate" or context.confirmed:
            return None

        pre_check = self.ont.validate_mutate(args)
        if not pre_check:
            return None
        return ToolResult(content=pre_check, blocked=True, block_reason=pre_check)

    def _run_pre_tool_hooks(self, tool_name: str, args: dict, tool: ToolDef,
                            context: ToolUseContext) -> ToolResult | None:
        if context.confirmed:
            return None

        pre_result = self.hooks.fire("pre_tool_call", {
            "tool_name": tool_name,
            "args": args,
            "tool_meta": tool,
            "session_id": context.session_id,
        })
        if pre_result.action == "block":
            return ToolResult(
                content=json.dumps({"blocked": True, "reason": pre_result.reason}, ensure_ascii=False),
                blocked=True,
                block_reason=pre_result.reason,
            )
        if pre_result.action == "pause":
            return ToolResult(
                content=json.dumps({"paused": True, "reason": pre_result.reason}, ensure_ascii=False),
                blocked=True,
                block_reason=pre_result.reason,
                needs_confirmation=True,
            )
        return None

    def _maybe_pause_for_user_question(self, tool_name: str, args: dict,
                                       tool: ToolDef, context: ToolUseContext) -> ToolResult | None:
        if not (tool.requires_confirmation and not context.confirmed and tool_name == "ask_user"):
            return None

        raw_result = tool.handler(args)
        return ToolResult(
            content=raw_result,
            blocked=True,
            block_reason=args.get("question", ""),
            needs_confirmation=True,
        )

    def _get_cached_result(self, tool_name: str, args: dict, tool: ToolDef,
                           context: ToolUseContext) -> ToolResult | None:
        if not tool.is_read_only:
            return None
        return self.cache.get(self._cache_key(tool_name, args))

    def _check_constraints(self, tool_name: str, args: dict,
                           context: ToolUseContext) -> ToolResult | None:
        constraint_error = self.ont.check_constraints(tool_name, args)
        if not constraint_error:
            return None
        return ToolResult(
            content=constraint_error,
            blocked=True,
            block_reason=constraint_error,
        )

    def _execute_handler(self, tool_name: str, args: dict, tool: ToolDef,
                         context: ToolUseContext) -> ToolResult:
        if tool_name == "summarize_progress":
            self.set_current_messages(context.messages)

        raw_result = tool.handler(args)
        context_note = self._build_progressive_context_note(tool_name, raw_result)
        truncated_result = truncate_tool_result(raw_result, tool.max_result_chars)
        was_truncated = len(truncated_result) < len(raw_result)

        post_result = self._run_post_tool_hooks(tool_name, args, tool, raw_result, context)
        review_notes = post_result.data.get("review_notes", [])
        if review_notes:
            truncated_result += "\n\n[⚠ 系统校验提示]\n" + "\n".join(f"- {n}" for n in review_notes)

        return ToolResult(
            content=truncated_result,
            raw_content=raw_result,
            truncated=was_truncated,
            context_note=context_note,
        )

    def _store_cache_result(self, tool_name: str, args: dict, tool: ToolDef,
                            result: ToolResult):
        if tool.is_read_only:
            self.cache[self._cache_key(tool_name, args)] = result

    def _build_progressive_context_note(self, tool_name: str, raw_result: str) -> str:
        if not self.progressive_context_enabled():
            return ""
        fn_context = self.ont.build_context_for_tool(tool_name) or ""
        result_context = self.ont.build_context_from_result(raw_result) or ""
        context_parts = [p for p in [fn_context, result_context] if p]
        return "\n\n".join(context_parts)

    def _run_post_tool_hooks(self, tool_name: str, args: dict, tool: ToolDef,
                             raw_result: str, context: ToolUseContext) -> HookResult:
        return self.hooks.fire("post_tool_call", {
            "tool_name": tool_name,
            "args": args,
            "tool_meta": tool,
            "result": raw_result,
            "session_id": context.session_id,
            "hook_event": "post_tool_call",
            "audit_log": self.audit,
        })

    def _record_tool_result(self, event_type: str, tool_name: str,
                            context: ToolUseContext, result: ToolResult):
        self.trace.record(
            event_type,
            session_id=context.session_id,
            source=context.source,
            tool_name=tool_name,
            blocked=result.blocked,
            needs_confirmation=result.needs_confirmation,
            truncated=result.truncated,
            block_reason=result.block_reason,
            content_preview=result.content[:300],
        )

    def _cache_key(self, tool_name: str, args: dict) -> str:
        return f"{tool_name}:{json.dumps(args, sort_keys=True)}"
