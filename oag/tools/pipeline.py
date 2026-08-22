"""工具执行管线。

这是每次工具调用的中心策略闸门：worker 权限、hooks/确认、
只读缓存、本体约束、结果截断、审计 hook 和 trace 都在这里统一处理。
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextvars import copy_context
from dataclasses import dataclass
from typing import Protocol

from ..runtime import ToolUseContext, TraceRecorder
from ..runtime.hooks import AuditLog, HookRegistry, HookResult
from ..runtime.tool_result_store import ToolResultStore
from .registry import ToolDef, ToolRegistry


@dataclass
class ToolResult:
    content: str
    raw_content: str = ""
    truncated: bool = False
    blocked: bool = False
    block_reason: str = ""
    needs_confirmation: bool = False
    needs_user_input: bool = False


class ToolPolicyRuntime(Protocol):
    def check_constraints(self, tool_name: str, args: dict) -> str | None: ...


class ToolExecutionPipeline:
    def __init__(self, *,
                 tools: ToolRegistry,
                 ontology_runtime: ToolPolicyRuntime,
                 hooks: HookRegistry,
                 audit: AuditLog,
                 cache: dict[str, ToolResult],
                 trace: TraceRecorder,
                 result_store: ToolResultStore | None = None,
                 persist_large_results: bool = True):
        self.tools = tools
        self.ont = ontology_runtime
        self.hooks = hooks
        self.audit = audit
        self.cache = cache
        self.trace = trace
        self.result_store = result_store or ToolResultStore()
        self.persist_large_results = persist_large_results

    def execute(self, tool_name: str, args: dict, context: ToolUseContext) -> ToolResult:
        tool = self.tools.get(tool_name)
        if not tool:
            reason = f"未知工具: {tool_name}"
            self.trace.record(
                "tool_unknown",
                session_id=context.session_id,
                source=context.source,
                turn_count=context.turn_count,
                tool_name=tool_name,
            )
            return ToolResult(
                content=json.dumps({"error": reason}, ensure_ascii=False),
                blocked=True,
                block_reason=reason,
            )

        # 执行顺序很重要：先做便宜的策略/校验，再触发 hooks，最后才执行 handler。
        self.trace.record(
            "tool_start",
            session_id=context.session_id,
            source=context.source,
            turn_count=context.turn_count,
            tool_name=tool_name,
            args=args,
            confirmed=context.confirmed,
        )

        if result := self._validate_tool_args(tool_name, args, tool):
            self._record_tool_result("tool_blocked", tool_name, context, result)
            return result

        if result := self._enforce_tool_policy(tool_name, tool, context):
            self._record_tool_result("tool_blocked", tool_name, context, result)
            return result

        if result := self._maybe_pause_for_user_question(tool_name, args, tool, context):
            self._record_tool_result("tool_user_input_required", tool_name, context, result)
            return result

        if result := self._run_pre_tool_hooks(tool_name, args, tool, context):
            event_type = "tool_confirmation_required" if result.needs_confirmation else "tool_blocked"
            self._record_tool_result(event_type, tool_name, context, result)
            return result

        if result := self._get_cached_result(tool_name, args, tool, context):
            self._run_post_tool_hooks(
                tool_name,
                args,
                tool,
                result.raw_content or result.content,
                context,
            )
            self._record_tool_result("tool_cache_hit", tool_name, context, result)
            return result

        if result := self._check_constraints(tool_name, args, context):
            self._record_tool_result("tool_blocked", tool_name, context, result)
            return result

        result = self._execute_handler(tool_name, args, tool, context)
        self._store_cache_result(tool_name, args, tool, result, context)

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
        if not (
            tool.policy.requires_user_input
            and not context.confirmed
        ):
            return None

        raw_result = tool.handler(args)
        return ToolResult(
            content=raw_result,
            blocked=True,
            block_reason=args.get("question", ""),
            needs_user_input=True,
        )

    def _get_cached_result(self, tool_name: str, args: dict, tool: ToolDef,
                           context: ToolUseContext) -> ToolResult | None:
        if not tool.policy.read_only:
            return None
        return self.cache.get(self._cache_key(tool_name, args, context))

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
        if context.cancelled:
            reason = f"工具 {tool_name} 已取消"
            return ToolResult(
                content=json.dumps({"error": reason}, ensure_ascii=False),
                blocked=True,
                block_reason=reason,
            )

        timeout_result = self._run_handler_with_timeout(tool_name, args, tool)
        if timeout_result.blocked:
            return timeout_result

        raw_result = timeout_result.raw_content
        visible_result, was_truncated = self._prepare_visible_result(
            tool_name,
            raw_result,
            tool,
            context,
        )

        self._run_post_tool_hooks(tool_name, args, tool, raw_result, context)

        return ToolResult(
            content=visible_result,
            raw_content=raw_result,
            truncated=was_truncated,
        )

    def _run_handler_with_timeout(self, tool_name: str, args: dict,
                                  tool: ToolDef) -> ToolResult:
        timeout = tool.policy.timeout_seconds if tool.policy else None
        if timeout is None or timeout <= 0:
            try:
                raw_result = tool.handler(args)
            except Exception as exc:
                return self._handler_error(tool_name, exc)
            return ToolResult(content=raw_result, raw_content=raw_result)

        pool = ThreadPoolExecutor(max_workers=1)
        context = copy_context()
        future = pool.submit(context.run, tool.handler, args)
        try:
            raw_result = future.result(timeout=timeout)
            return ToolResult(content=raw_result, raw_content=raw_result)
        except TimeoutError:
            future.cancel()
            reason = f"工具执行超时: {tool_name} 超过 {timeout:g}s"
            return ToolResult(
                content=json.dumps({"error": reason}, ensure_ascii=False),
                blocked=True,
                block_reason=reason,
            )
        except Exception as exc:
            return self._handler_error(tool_name, exc)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _handler_error(tool_name: str, exc: Exception) -> ToolResult:
        reason = f"工具执行失败: {tool_name}"
        return ToolResult(
            content=json.dumps({
                "error": reason,
                "details": str(exc),
            }, ensure_ascii=False),
            blocked=True,
            block_reason=reason,
        )

    def _prepare_visible_result(self, tool_name: str, raw_result: str,
                                tool: ToolDef,
                                context: ToolUseContext) -> tuple[str, bool]:
        max_chars = tool.max_result_chars
        if len(raw_result) <= max_chars:
            return raw_result, False

        if not self.persist_large_results:
            return json.dumps({
                "truncated": True,
                "original_chars": len(raw_result),
                "preview_chars": max_chars,
                "preview": raw_result[:max_chars],
                "hint": "结果过长，当前仅返回预览；请缩小查询范围。",
            }, ensure_ascii=False), True

        return self.result_store.persist(
            session_id=context.session_id,
            tool_name=tool_name,
            content=raw_result,
            preview_chars=max_chars,
            storage_dir=context.storage_dir,
        ), True

    def _store_cache_result(self, tool_name: str, args: dict, tool: ToolDef,
                            result: ToolResult, context: ToolUseContext):
        if tool.policy.read_only and not result.blocked:
            self.cache[self._cache_key(tool_name, args, context)] = result

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
            turn_count=context.turn_count,
            tool_name=tool_name,
            blocked=result.blocked,
            needs_confirmation=result.needs_confirmation,
            needs_user_input=result.needs_user_input,
            truncated=result.truncated,
            block_reason=result.block_reason,
            content_preview=result.content[:300],
        )

    def _cache_key(self, tool_name: str, args: dict,
                   context: ToolUseContext) -> str:
        # Read results may depend on live domain state. Cache only inside one
        # agent run; the session fallback isolates direct Harness callers.
        namespace = context.cache_namespace or context.session_id
        return json.dumps(
            [namespace, tool_name, args],
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def clear_cache_namespace(self, namespace: str) -> None:
        if not namespace:
            return
        for key in list(self.cache):
            try:
                cached_namespace = json.loads(key)[0]
            except (IndexError, TypeError, json.JSONDecodeError):
                continue
            if cached_namespace == namespace:
                self.cache.pop(key, None)

    def _validate_tool_args(self, tool_name: str, args: dict,
                            tool: ToolDef) -> ToolResult | None:
        errors = validate_json_schema_args(args, tool.parameters or {})
        if not errors:
            return None
        return ToolResult(
            content=json.dumps({
                "error": "工具参数校验失败",
                "tool": tool_name,
                "details": errors,
            }, ensure_ascii=False),
            blocked=True,
            block_reason="工具参数校验失败",
        )


def validate_json_schema_args(args: dict, schema: dict) -> list[str]:
    errors: list[str] = []
    if schema.get("type") == "object" and not isinstance(args, dict):
        return ["参数必须是 JSON object"]

    props = schema.get("properties", {}) or {}
    required = schema.get("required", []) or []

    for name in required:
        if name not in args or args[name] is None:
            errors.append(f"缺少必填字段: {name}")

    for name, value in args.items():
        prop = props.get(name)
        if not isinstance(prop, dict):
            continue

        expected = prop.get("type")
        if expected and not _matches_json_type(value, expected):
            errors.append(f"{name} 类型错误: 期望 {expected}")
            continue

        if "enum" in prop and value not in (prop.get("enum") or []):
            errors.append(f"{name} 取值非法: {value}，允许值: {prop.get('enum')}")

        if expected == "array" and isinstance(value, list):
            item_schema = prop.get("items")
            if isinstance(item_schema, dict):
                item_type = item_schema.get("type")
                for idx, item in enumerate(value):
                    if item_type and not _matches_json_type(item, item_type):
                        errors.append(f"{name}[{idx}] 类型错误: 期望 {item_type}")

    return errors


def _matches_json_type(value, expected: str | list[str]) -> bool:
    if isinstance(expected, list):
        return any(_matches_json_type(value, item) for item in expected)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "null":
        return value is None
    return True
