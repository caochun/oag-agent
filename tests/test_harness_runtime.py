from __future__ import annotations

import json
import time
from contextvars import ContextVar
from types import SimpleNamespace

import pytest

from oag.agent import Agent
from oag.harness import Harness, HarnessConfig
from oag.llm.context import ContextManager
from oag.llm.context_usage import collect_context_usage
from oag.loop.confirmation_flow import ConfirmationFlow
from oag.loop.query_loop import QueryLoop
from oag.loop.tool_executor import ToolExecutor
from oag.ontology.bindings import RuntimeBindings
from oag.ontology.repository import OntologyRepository
from oag.ontology.schema import (
    ActionDef,
    ActionInputDef,
    ActionSideEffectsDef,
    DataBindingDef,
    DataSourceDef,
    FunctionDef,
    FunctionParam,
    ObjectTypeDef,
    Ontology,
    Precondition,
    PropertyDef,
)
from oag.ontology.source import SourceManager
from oag.runtime import PendingConfirmation, RunState, ToolUseContext
from oag.runtime.events import AssistantDeltaEvent, AssistantEndEvent
from oag.runtime.hooks import HookResult
from oag.runtime.message_sanitizer import sanitize_messages
from oag.runtime.session_store import SessionStore
from oag.tools.registry import ToolDef, ToolPolicy


class DummyClient:
    pass


class FakeActionRuntime:
    def list_actions(self, context_id=""):
        return {"context": None, "actions": [{"id": "create_work_order", "executable": True}]}

    def prepare_action(self, action_id, context_id="", initial_inputs=None):
        return {
            "action": {"id": action_id, "name": "创建维修工单"},
            "context": None,
            "context_id": context_id,
            "initial_inputs": initial_inputs or {},
        }

    def preview_action(self, action_id, inputs=None, context_id=""):
        return {"valid": True, "preview_token": "preview-1"}

    def execute_action(self, preview_token, reason="", actor="", channel=""):
        return {"applied": True}


class MemorySource:
    def __init__(self, ontology: Ontology):
        self.ontology = ontology
        self.rows: dict[str, list[dict]] = {}
        self.location = None

    def query_objects(self, object_type, binding, filters=None, limit=None,
                      order_by=None, offset=None):
        rows = list(self.rows.get(object_type, []))
        for key, value in (filters or {}).items():
            field, op = key.split("__", 1) if "__" in key else (key, "eq")
            if op == "gte":
                rows = [row for row in rows if row.get(field) >= value]
            else:
                rows = [row for row in rows if row.get(field) == value]
        if order_by:
            reverse = order_by.startswith("-")
            field = order_by.lstrip("-")
            rows = sorted(rows, key=lambda row: row.get(field), reverse=reverse)
        if offset:
            rows = rows[offset:]
        if limit:
            rows = rows[:limit]
        return [dict(row) for row in rows]

    def get_object(self, object_type, binding, id_value):
        id_field = self.ontology.get_id_column(object_type)
        if not id_field:
            return None
        rows = self.query_objects(object_type, binding, {id_field: id_value}, limit=1)
        return rows[0] if rows else None

    def search_objects(self, object_type, binding, keyword, limit=20):
        obj_def = self.ontology.objects[object_type]
        text_cols = [name for name, prop in obj_def.properties.items() if prop.type == "str"]
        results = []
        for row in self.rows.get(object_type, []):
            matched = [col for col in text_cols if row.get(col) and keyword in str(row[col])]
            if matched:
                record = dict(row)
                record["_object_type"] = object_type
                record["_matched_field"] = ", ".join(matched)
                results.append(record)
            if len(results) >= limit:
                break
        return results

    def query_relations(self, *args, **kwargs):
        return []

    def get_relation(self, *args, **kwargs):
        return None

    def load_data(self, object_type, rows):
        self.rows.setdefault(object_type, []).extend(dict(row) for row in rows)

    def close(self):
        pass


def make_repository(ontology: Ontology) -> OntologyRepository:
    sources = SourceManager(ontology)
    sources.register(
        "memory",
        lambda ontology, **kw: MemorySource(ontology),
    )
    return OntologyRepository(ontology, sources)


def make_harness(config: HarnessConfig | None = None,
                 include_actions: bool = False) -> Harness:
    ontology = Ontology(
        name="TestDomain",
        description="Test domain",
        data_sources={"memory": DataSourceDef(type="memory", mode="writable")},
        objects={
            "Asset": ObjectTypeDef(
                summary="Asset summary",
                description="Asset full description",
                binding=DataBindingDef(source="memory"),
                data_source="external_api",
                mutability="read_only",
                properties={
                    "asset_id": PropertyDef(type="str", required=True, description="Asset id"),
                    "status": PropertyDef(type="str", description="Asset status"),
                },
            ),
            "WorkOrder": ObjectTypeDef(
                summary="Work order summary",
                description="Work order full description",
                binding=DataBindingDef(source="memory"),
                data_source="agent_generated",
                mutability="mutable",
                properties={
                    "order_id": PropertyDef(type="str", required=True, description="Order id"),
                    "status": PropertyDef(type="str", description="Order status"),
                },
            ),
            "AuditNote": ObjectTypeDef(
                summary="Agent note summary",
                description="Agent generated append-only note",
                binding=DataBindingDef(source="memory"),
                data_source="agent_generated",
                mutability="append_only",
                properties={
                    "note_id": PropertyDef(type="str", required=True, description="Note id"),
                    "status": PropertyDef(type="str", description="Note status"),
                },
            ),
        },
        functions={
            "lookup_asset": FunctionDef(
                summary="Lookup an asset",
                description="Lookup asset details",
                timeout_seconds=75,
                concurrency_safe=False,
                params={"asset_id": FunctionParam(type="str", description="Asset id")},
                reads_objects=["Asset"],
            ),
            "create_work_order": FunctionDef(
                summary="Create a work order",
                description="Create work order details",
                usage_prompt="创建前必须确认 asset_id 指向真实资产，并说明写入影响。",
                reads_objects=["Asset", "WorkOrder"],
                params={"asset_id": FunctionParam(type="str", description="Asset id")},
                preconditions=[
                    Precondition(
                        object="Asset",
                        field="asset_id",
                        operator="exists",
                        value_from_param="asset_id",
                    ),
                ],
            ),
            "create_audit_note": FunctionDef(
                summary="Create an audit note",
                description="Create append-only agent note",
                reads_objects=["AuditNote"],
                params={"asset_id": FunctionParam(type="str", description="Asset id")},
            ),
            "set_asset_threshold": FunctionDef(
                summary="Set asset threshold",
                params={
                    "asset_id": FunctionParam(type="str", description="Asset id"),
                    "threshold": FunctionParam(type="float", description="Threshold"),
                    "enabled": FunctionParam(type="bool", description="Enabled"),
                },
            ),
        },
        actions={
            "create_work_order": ActionDef(
                display_name="创建维修工单",
                description="为资产创建维修工单",
                available_on=["Asset"],
                context_input="asset_id",
                inputs={
                    "asset_id": ActionInputDef(
                        display_name="资产", required=True, object_types=["Asset"],
                    ),
                },
                side_effects=ActionSideEffectsDef(creates_objects=["WorkOrder"]),
                confirmation="创建维修工单",
            ),
        } if include_actions else {},
    )
    bindings = RuntimeBindings()
    if include_actions:
        bindings.register_action_runtime(FakeActionRuntime())
    repository = make_repository(ontology)
    repository.sources.get("memory").load_data(
        "Asset", [{"asset_id": "A1", "status": "ok"}],
    )
    bindings.register(
        "lookup_asset",
        lambda asset_id: {"asset_id": asset_id, "status": "ok"},
        ontology.functions["lookup_asset"],
    )
    bindings.register(
        "create_work_order",
        lambda asset_id: {"order_id": "WO1", "asset_id": asset_id, "status": "created"},
        ontology.functions["create_work_order"],
    )
    bindings.register(
        "create_audit_note",
        lambda asset_id: {"note_id": "N1", "asset_id": asset_id, "status": "created"},
        ontology.functions["create_audit_note"],
    )
    bindings.register(
        "set_asset_threshold",
        lambda asset_id, threshold, enabled: {
            "asset_id": asset_id,
            "threshold": threshold,
            "enabled": enabled,
        },
        ontology.functions["set_asset_threshold"],
    )

    return Harness(
        ontology,
        repository,
        bindings,
        DummyClient(),
        "dummy-model",
        config or HarnessConfig(enable_write_confirmation=False),
    )


def register_confirmed_write_tool(harness: Harness, name: str = "write_record") -> str:
    harness.tools.register(ToolDef(
        name=name,
        description="Test-only explicit write tool",
        parameters={"type": "object", "properties": {}},
        handler=lambda args: json.dumps({"written": True}),
        policy=ToolPolicy(
            read_only=False,
            requires_confirmation=True,
            concurrency_safe=False,
            worker_allowed=False,
        ),
    ))
    return name


def make_tool_call(name: str, tool_id: str = "tool_1") -> SimpleNamespace:
    return SimpleNamespace(
        id=tool_id,
        function=SimpleNamespace(name=name),
    )


def make_stream_chunk(*, content: str | None = None,
                      reasoning: str | None = None,
                      tool_call=None) -> SimpleNamespace:
    delta = SimpleNamespace(
        content=content,
        reasoning_content=reasoning,
        tool_calls=[tool_call] if tool_call else None,
    )
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def make_tool_delta(index: int, *,
                    tool_id: str | None = None,
                    name: str | None = None,
                    arguments: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        index=index,
        id=tool_id,
        type="function" if tool_id else None,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def make_response_message(content: str = "", tool_calls=None) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def make_response(tool_calls=None, content: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=make_response_message(content=content, tool_calls=tool_calls),
            ),
        ],
    )


def make_full_tool_call(name: str, tool_id: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=tool_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_prompt_uses_summary_context_and_inspect_for_details():
    harness = make_harness()

    prompt = harness.build_system_prompt()

    assert "## 可用函数" in prompt
    assert "- lookup_asset: Lookup an asset" in prompt
    assert "## 函数完整定义" not in prompt
    assert "### 函数: lookup_asset" not in prompt
    assert "## 对象完整定义" not in prompt
    assert "首次调用函数时系统会自动注入" not in prompt
    assert "摘要不足以回答属性、约束或能力细节时再调用 inspect" in prompt
    assert "ID、对象类型、属性名、关系类型和枚举/角色值必须按证据原样引用" in prompt
    assert "多个对象与同一主体有关联，不代表这些对象彼此构成先后链路" in prompt
    assert "不要按对象类型或 ID 命名拼接“典型链路”" in prompt
    assert "对象 ID 只能作为 ID 使用" in prompt
    assert "查询记录随附的 _semantics 是字段和类型语义的直接证据" in prompt
    assert "用户未明确要求行业背景、判断或推测时，不要主动补充" in prompt
    assert "不要为了复述已有摘要而调用" in prompt

    details = json.loads(harness.execute_tool("inspect", {"name": "create_work_order"}).content)

    assert details["usage_prompt"] == "创建前必须确认 asset_id 指向真实资产，并说明写入影响。"
    assert details["preconditions"][0]["object"] == "Asset"
    assert details["reads_objects"] == ["Asset", "WorkOrder"]


def test_generic_analysis_and_crud_tools_are_not_registered():
    harness = make_harness()

    assert not harness.tools.has("describe")
    assert not harness.tools.has("count")
    assert not harness.tools.has("pivot")
    assert not harness.tools.has("distribution")
    assert not harness.tools.has("mutate")


def test_get_object_uses_the_complete_stable_id():
    harness = make_harness()

    result = json.loads(harness.execute_tool(
        "get_object",
        {"object_type": "Asset", "id": "A1"},
    ).content)

    assert result["asset_id"] == "A1"
    assert result["status"] == "ok"
    assert result["_semantics"] == {
        "kind": "object",
        "type": "Asset",
        "display_name": "Asset",
        "description": "Asset full description",
        "properties": {
            "asset_id": {
                "type": "str",
                "display_name": "asset_id",
                "description": "Asset id",
            },
            "status": {
                "type": "str",
                "display_name": "status",
                "description": "Asset status",
            },
        },
    }
    schema = harness.tools.get("get_object").parameters
    assert schema["required"] == ["object_type", "id"]
    assert "完整稳定对象 ID" in schema["properties"]["id"]["description"]


def test_query_requires_in_suffix_for_array_filters():
    harness = make_harness()

    result = harness.execute_tool(
        "query",
        {"object_type": "Asset", "filters": {"asset_id": ["A1", "A2"]}},
    )

    assert result.blocked
    assert "asset_id__in" in result.content
    description = harness.tools.get("query").parameters["properties"]["filters"]["description"]
    assert "id__in" in description


def test_function_tool_uses_ontology_execution_policy():
    harness = make_harness()

    assert harness.tools.get("lookup_asset").policy.timeout_seconds == 75
    assert harness.tools.get("lookup_asset").policy.concurrency_safe is False
    details = json.loads(harness.execute_tool("inspect", {"name": "lookup_asset"}).content)
    assert details["timeout_seconds"] == 75
    assert details["concurrency_safe"] is False


def test_function_param_types_map_to_json_schema_types():
    harness = make_harness()

    params = harness.tools.get("set_asset_threshold").parameters["properties"]

    assert params["asset_id"]["type"] == "string"
    assert params["threshold"]["type"] == "number"
    assert params["enabled"]["type"] == "boolean"


def test_action_catalog_registers_generic_discovery_and_form_tools():
    harness = make_harness(include_actions=True)

    assert harness.tools.has("get_available_actions")
    form = harness.tools.get("request_action_input")
    assert form.parameters["properties"]["action_id"]["enum"] == ["create_work_order"]
    result = json.loads(form.handler({
        "action_id": "create_work_order",
        "context_id": "A1",
        "initial_inputs": {"asset_id": "A1"},
    }))
    assert result["interaction"]["kind"] == "action_form"
    assert result["interaction"]["initial_inputs"] == {"asset_id": "A1"}
    assert form.policy.read_only is True
    assert form.policy.worker_allowed is False
    visible_names = {
        item["function"]["name"] for item in harness.build_tools()
    }
    assert "request_action_input" in visible_names


def test_function_param_schema_rejects_string_for_float():
    harness = make_harness()

    result = harness.execute_tool(
        "set_asset_threshold",
        {"asset_id": "A1", "threshold": "2500", "enabled": True},
    )

    assert result.blocked
    assert "threshold 类型错误: 期望 number" in result.content


def test_function_tool_errors_are_structured_json():
    harness = make_harness()
    harness.data.bindings.register(
        "broken_lookup",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")),
        FunctionDef(summary="Broken lookup"),
    )
    harness.tools.register(ToolDef(
        name="broken_lookup",
        description="Broken lookup",
        parameters={"type": "object", "properties": {}},
        handler=lambda args: harness.data.execute("broken_lookup", args),
        category="action",
    ))

    result = harness.execute_tool("broken_lookup", {})
    payload = json.loads(result.content)

    assert payload == {
        "error": "工具执行失败: broken_lookup",
        "details": "boom",
    }
    assert result.blocked


def test_prompt_sections_are_layered_and_cached():
    harness = make_harness()

    sections = harness.build_system_prompt_sections()
    first = harness.build_system_prompt()
    second = harness.build_system_prompt()

    assert sections[0].startswith("你是 TestDomain 领域的智能助手。")
    assert "## 可用对象" in sections[1]
    assert "## 工具使用规则" in sections[2]
    assert "## 运行时上下文" in sections[-1]
    assert "## 函数完整定义" not in first
    assert first == second
    assert list(harness._static_prompt_cache) == [""]
    assert all("## 运行时上下文" not in s for s in harness._static_prompt_cache[""])


def test_custom_append_and_runtime_context_layers():
    harness = make_harness(HarnessConfig(
        enable_write_confirmation=False,
        custom_system_prompt="你是部署 A 的 OAG。",
        append_system_prompt="## 部署策略\n优先使用只读工具。",
        runtime_context={"tenant": "alpha"},
    ))

    sections = harness.build_system_prompt_sections()
    prompt = "\n\n".join(sections)

    assert sections[0] == "你是部署 A 的 OAG。"
    assert "## 可用对象" in sections[1]
    assert "tenant: alpha" in prompt
    assert prompt.rstrip().endswith("优先使用只读工具。")


def test_tool_schema_is_cached_and_invalidated_on_register():
    harness = make_harness()

    first = harness.build_tools()
    second = harness.build_tools()
    harness.tools.register(ToolDef(
        name="new_tool",
        description="New test tool",
        parameters={"type": "object", "properties": {}},
        handler=lambda args: "ok",
    ))
    third = harness.build_tools()

    assert first is second
    assert third is not first
    assert any(t["function"]["name"] == "new_tool" for t in third)


def test_tool_registry_rejects_duplicate_names():
    harness = make_harness()

    with pytest.raises(ValueError, match="Tool already registered"):
        harness.tools.register(ToolDef(
            name="query",
            description="Duplicate",
            parameters={"type": "object", "properties": {}},
            handler=lambda args: "duplicate",
        ))


def test_runtime_bindings_reject_duplicate_function_names():
    bindings = RuntimeBindings()
    definition = FunctionDef(summary="Lookup")
    bindings.register("lookup", lambda: "first", definition)

    with pytest.raises(ValueError, match="Function already registered"):
        bindings.register("lookup", lambda: "second", definition)


def test_function_usage_prompt_is_added_to_tool_description():
    harness = make_harness()

    tools = harness.build_tools()
    create_tool = next(t for t in tools if t["function"]["name"] == "create_work_order")

    assert "Create a work order" in create_tool["function"]["description"]
    assert "使用说明:" in create_tool["function"]["description"]
    assert "创建前必须确认 asset_id 指向真实资产" in create_tool["function"]["description"]


def test_runtime_tools_have_usage_prompts():
    harness = make_harness(HarnessConfig(
        enable_worker_dispatch=True,
        enable_tool_result_reader=True,
    ))

    tools = harness.build_tools()
    dispatch_tool = next(t for t in tools if t["function"]["name"] == "dispatch_workers")
    ask_tool = next(t for t in tools if t["function"]["name"] == "ask_user")

    assert "使用说明:" in dispatch_tool["function"]["description"]
    assert "相互独立的只读子任务" in dispatch_tool["function"]["description"]
    assert "不要询问可以通过只读工具直接查到的信息" in ask_tool["function"]["description"]


def test_default_prompt_does_not_advertise_disabled_runtime_extensions():
    prompt = make_harness().build_system_prompt()

    assert "dispatch_workers" not in prompt
    assert "read_tool_result" not in prompt


def test_worker_tool_visibility_uses_declared_policy_not_task_keywords():
    harness = make_harness(include_actions=True)
    register_confirmed_write_tool(harness)

    names = {
        tool["function"]["name"]
        for tool in harness.build_worker_tools()
    }

    assert "lookup_asset" in names
    assert "query" in names
    assert "ask_user" not in names
    assert "write_record" not in names
    assert "get_available_actions" not in names


def test_explicit_write_tool_needs_confirmation():
    harness = make_harness(HarnessConfig(enable_write_confirmation=True))
    register_confirmed_write_tool(harness)

    result = harness.execute_tool("write_record", {})

    assert result.blocked
    assert result.needs_confirmation


def test_explicit_write_tool_runs_after_confirmation():
    harness = make_harness(HarnessConfig(enable_write_confirmation=True))
    register_confirmed_write_tool(harness)

    result = harness.execute_tool("write_record", {}, confirmed=True)

    assert not result.blocked
    assert json.loads(result.content) == {"written": True}


def test_functions_are_always_read_only_and_do_not_need_confirmation():
    harness = make_harness(HarnessConfig(enable_write_confirmation=True))

    result = harness.execute_tool("create_audit_note", {"asset_id": "A1"})

    assert not result.blocked
    assert not result.needs_confirmation
    assert json.loads(result.content)["note_id"] == "N1"


def test_function_read_only_policy_does_not_depend_on_returned_business_shape():
    harness = make_harness(HarnessConfig(enable_write_confirmation=True))

    result = harness.execute_tool("create_work_order", {"asset_id": "A1"})

    assert not result.blocked
    assert not result.needs_confirmation


def test_worker_system_prompt_uses_summary_not_full_context():
    harness = make_harness()

    prompt = harness.build_worker_system_prompt("W1", "事件 E1")

    assert "你是 Worker W1" in prompt
    assert "## 可用对象" in prompt
    assert "事件 E1" in prompt
    assert "## 函数完整定义" not in prompt
    assert "需要完整定义时调用 inspect" in prompt


def test_worker_context_blocks_confirmation_and_non_worker_tools():
    harness = make_harness()
    register_confirmed_write_tool(harness)
    context = ToolUseContext(source="worker", confirmed=False)

    write_result = harness.execute_tool("write_record", {}, context=context)
    ask_user_result = harness.execute_tool(
        "ask_user",
        {"question": "Choose?", "options": [{"label": "A"}]},
        context=context,
    )
    read_fn_result = harness.execute_tool(
        "lookup_asset",
        {"asset_id": "A1"},
        context=context,
    )

    assert write_result.blocked
    assert ask_user_result.blocked
    assert not ask_user_result.needs_confirmation
    assert not read_fn_result.blocked


def test_ask_user_uses_user_input_protocol_not_write_confirmation():
    harness = make_harness()

    result = harness.execute_tool(
        "ask_user",
        {"question": "Choose?", "options": [{"label": "A"}]},
    )

    assert result.blocked
    assert result.needs_user_input
    assert not result.needs_confirmation


def test_trace_records_successful_tool_execution():
    harness = make_harness()

    result = harness.execute_tool("lookup_asset", {"asset_id": "A1"})
    events = harness.trace.snapshot()

    assert not result.blocked
    assert [event.event_type for event in events] == ["tool_start", "tool_end"]
    assert events[0].payload["tool_name"] == "lookup_asset"
    assert events[1].payload["content_preview"]


def test_trace_records_jsonl_when_configured(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    harness = make_harness(HarnessConfig(
        enable_write_confirmation=False,
        trace_jsonl_path=str(trace_path),
    ))

    harness.execute_tool("lookup_asset", {"asset_id": "A1"})

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert [record["event_type"] for record in records] == ["tool_start", "tool_end"]
    assert records[0]["session_id"] == ""
    assert records[0]["payload"]["tool_name"] == "lookup_asset"


def test_context_usage_breaks_down_messages_and_tools():
    harness = make_harness()
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Question?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "tool_1",
                "type": "function",
                "function": {
                    "name": "lookup_asset",
                    "arguments": "{\"asset_id\":\"A1\"}",
                },
            }],
        },
        {"role": "tool", "tool_call_id": "tool_1", "content": "{\"asset_id\":\"A1\",\"status\":\"ok\"}"},
    ]

    usage = collect_context_usage(
        messages,
        harness.build_tools(),
        context_window=1000,
        model="dummy-model",
    )

    assert usage["model"] == "dummy-model"
    assert usage["context_window"] == 1000
    assert usage["total_tokens"] > 0
    assert usage["tools"]["count"] == len(harness.build_tools())
    assert usage["tools"]["largest_tools"][0]["tokens"] > 0
    assert usage["messages"]["count"] == len(messages)
    assert usage["messages"]["tool_call_tokens"] > 0
    assert usage["messages"]["tool_result_tokens"] > 0
    assert usage["messages"]["largest_tool_results"][0]["tool_call_id"] == "tool_1"
    assert {category["name"] for category in usage["categories"]} == {
        "System prompt",
        "Tool schemas",
        "Messages",
        "Free space",
    }


def test_agent_exposes_session_context_usage(tmp_path):
    harness = make_harness()
    agent = Agent(harness, DummyClient(), "dummy-model", db_dir=str(tmp_path))
    agent.sessions.save("s1", [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Question?"},
    ])

    usage = agent.get_context_usage("s1")

    assert usage["messages"]["count"] == 2
    assert usage["tools"]["count"] == len(harness.build_tools())
    assert usage["total_tokens"] > 0


def test_agent_sets_default_trace_jsonl_path(tmp_path):
    harness = make_harness()

    Agent(harness, DummyClient(), "dummy-model", db_dir=str(tmp_path))

    assert harness.trace.jsonl_path == str(tmp_path / "trace_TestDomain.jsonl")


def test_agent_refreshes_persisted_system_prompt_for_new_user_turn(tmp_path):
    harness = make_harness()
    agent = Agent(harness, DummyClient(), "dummy-model", db_dir=str(tmp_path))
    agent.sessions.save("s1", [
        {"role": "system", "content": "stale prompt"},
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"},
    ])
    agent._run_loop = lambda state: iter(())

    list(agent.chat_stream("New question", session_id="s1"))

    messages = agent.sessions.get("s1")
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == harness.build_system_prompt()
    assert "stale prompt" not in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "New question"}


def test_agent_chat_returns_only_final_assistant_turn(tmp_path):
    harness = make_harness()
    agent = Agent(harness, DummyClient(), "dummy-model", db_dir=str(tmp_path))
    agent._run_loop = lambda state: iter([
        AssistantDeltaEvent(content="I will check. "),
        AssistantEndEvent(kind="progress"),
        AssistantDeltaEvent(content="Asset A1 "),
        AssistantDeltaEvent(content="is available."),
        AssistantEndEvent(kind="final"),
    ])

    result = agent.chat("Explain A1", session_id="s1")

    assert result == "Asset A1 is available."


def test_agent_sse_preserves_assistant_delta_boundaries(tmp_path):
    harness = make_harness()
    agent = Agent(harness, DummyClient(), "dummy-model", db_dir=str(tmp_path))
    agent._run_loop = lambda state: iter([
        AssistantDeltaEvent(content="First "),
        AssistantDeltaEvent(content="second"),
        AssistantEndEvent(kind="final"),
    ])

    events = list(agent.chat_stream_sse("Question?", session_id="s1"))

    assert events == [
        {"type": "assistant_delta", "content": "First "},
        {"type": "assistant_delta", "content": "second"},
        {"type": "assistant_end", "kind": "final"},
    ]


def test_tool_pipeline_records_cache_hit_for_repeated_read_tool():
    harness = make_harness()

    first = harness.execute_tool("lookup_asset", {"asset_id": "A1"})
    second = harness.execute_tool("lookup_asset", {"asset_id": "A1"})
    events = harness.trace.snapshot()

    assert first.content == second.content
    assert [event.event_type for event in events] == [
        "tool_start",
        "tool_end",
        "tool_start",
        "tool_cache_hit",
    ]


def test_tool_trace_preserves_originating_turn_count():
    harness = make_harness()

    harness.execute_tool(
        "lookup_asset",
        {"asset_id": "A1"},
        context=ToolUseContext(session_id="s1", turn_count=4),
    )

    events = harness.trace.snapshot()
    assert [event.turn_count for event in events] == [4, 4]


def test_tool_cache_does_not_cross_agent_run_namespaces():
    harness = make_harness()

    first = harness.execute_tool(
        "lookup_asset",
        {"asset_id": "A1"},
        context=ToolUseContext(
            session_id="same-session",
            cache_namespace="run-1",
        ),
    )
    second = harness.execute_tool(
        "lookup_asset",
        {"asset_id": "A1"},
        context=ToolUseContext(
            session_id="same-session",
            cache_namespace="run-2",
        ),
    )

    assert first.content == second.content
    assert [event.event_type for event in harness.trace.snapshot()] == [
        "tool_start",
        "tool_end",
        "tool_start",
        "tool_end",
    ]


def test_timeout_worker_inherits_context_variables():
    harness = make_harness()
    marker = ContextVar("test_tool_workspace", default="missing")
    tool = harness.tools.get("lookup_asset")
    tool.handler = lambda args: marker.get()
    token = marker.set("current-workspace")
    try:
        result = harness.execute_tool(
            "lookup_asset",
            {"asset_id": "A1"},
            context=ToolUseContext(cache_namespace="context-test"),
        )
    finally:
        marker.reset(token)

    assert result.content == "current-workspace"


def test_tool_handler_exception_is_returned_as_blocked_result():
    harness = make_harness()
    harness.tools.register(ToolDef(
        name="broken_handler",
        description="Broken handler",
        parameters={"type": "object", "properties": {}},
        handler=lambda args: (_ for _ in ()).throw(RuntimeError("boom")),
        policy=ToolPolicy(timeout_seconds=None),
    ))

    result = harness.execute_tool("broken_handler", {})

    assert result.blocked
    assert json.loads(result.content) == {
        "error": "工具执行失败: broken_handler",
        "details": "boom",
    }


def test_blocked_read_tool_result_is_not_cached():
    harness = make_harness()
    attempts = 0

    def flaky_handler(args):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary")
        return "recovered"

    harness.tools.register(ToolDef(
        name="flaky_read",
        description="Flaky read",
        parameters={"type": "object", "properties": {}},
        handler=flaky_handler,
        policy=ToolPolicy(timeout_seconds=None),
    ))
    context = ToolUseContext(cache_namespace="flaky-run")

    first = harness.execute_tool("flaky_read", {}, context=context)
    second = harness.execute_tool("flaky_read", {}, context=context)

    assert first.blocked
    assert second.content == "recovered"
    assert attempts == 2


def test_unknown_tool_is_a_blocked_result():
    result = make_harness().execute_tool("missing_tool", {})

    assert result.blocked
    assert result.block_reason == "未知工具: missing_tool"


def test_tool_pipeline_runs_post_hook_for_cached_result():
    harness = make_harness()
    captured = []
    harness.hooks.register(
        "post_tool_call",
        lambda context: (
            captured.append(context["result"])
            or HookResult(action="allow")
        ),
    )

    harness.execute_tool("lookup_asset", {"asset_id": "A1"})
    harness.execute_tool("lookup_asset", {"asset_id": "A1"})

    assert len(captured) == 2
    assert captured[0] == captured[1]


def test_pre_tool_hook_can_block_a_read_only_tool():
    harness = make_harness()
    harness.hooks.register(
        "pre_tool_call",
        lambda context: HookResult(action="block", reason="policy denied"),
    )

    result = harness.execute_tool("lookup_asset", {"asset_id": "A1"})

    assert result.blocked
    assert result.block_reason == "policy denied"


def test_action_form_prepare_error_is_a_blocked_tool_result():
    harness = make_harness(include_actions=True)
    harness.data.bindings.get_action_runtime().prepare_action = (
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("invalid context"))
    )

    result = harness.execute_tool(
        "request_action_input",
        {"action_id": "create_work_order", "context_id": "missing"},
    )

    assert result.blocked
    assert json.loads(result.content) == {
        "error": "工具执行失败: request_action_input",
        "details": "invalid context",
    }


def test_tool_pipeline_validates_missing_required_arg():
    harness = make_harness()

    result = harness.execute_tool("query", {})

    assert result.blocked
    assert "工具参数校验失败" in result.content
    assert "缺少必填字段: object_type" in result.content


def test_tool_pipeline_validates_enum_arg():
    harness = make_harness()

    result = harness.execute_tool("query", {"object_type": "UnknownType"})

    assert result.blocked
    assert "工具参数校验失败" in result.content
    assert "object_type 取值非法" in result.content


def test_tool_pipeline_validates_arg_type():
    harness = make_harness()

    result = harness.execute_tool("query", {"object_type": "Asset", "limit": "ten"})

    assert result.blocked
    assert "工具参数校验失败" in result.content
    assert "limit 类型错误" in result.content


def test_tool_pipeline_times_out_slow_tool():
    harness = make_harness()
    harness.tools.register(ToolDef(
        name="slow_tool",
        description="Slow test tool",
        parameters={"type": "object", "properties": {}},
        handler=lambda args: (time.sleep(0.05) or "done"),
        policy=ToolPolicy(timeout_seconds=0.01),
    ))

    result = harness.execute_tool("slow_tool", {})

    assert result.blocked
    assert "工具执行超时" in result.content


def test_tool_pipeline_persists_large_tool_result(tmp_path):
    harness = make_harness(HarnessConfig(enable_tool_result_reader=True))
    harness.tools.register(ToolDef(
        name="large_tool",
        description="Large test tool",
        parameters={"type": "object", "properties": {}},
        handler=lambda args: "x" * 200,
        max_result_chars=50,
    ))

    result = harness.execute_tool(
        "large_tool",
        {},
        context=ToolUseContext(session_id="s1", storage_dir=str(tmp_path)),
    )

    payload = json.loads(result.content)

    assert result.truncated
    assert payload["persisted"] is True
    assert payload["original_chars"] == 200
    assert payload["preview"] == "x" * 50
    assert (tmp_path / "tool-results" / "s1").exists()
    assert "x" * 200 in (tmp_path / "tool-results" / "s1" / "large_tool.txt").read_text()


def test_tool_pipeline_returns_preview_when_result_reader_is_disabled():
    harness = make_harness()
    harness.tools.register(ToolDef(
        name="large_temp_tool",
        description="Large temp test tool",
        parameters={"type": "object", "properties": {}},
        handler=lambda args: "y" * 80,
        max_result_chars=20,
    ))

    result = harness.execute_tool(
        "large_temp_tool",
        {},
        context=ToolUseContext(session_id="temp-session"),
    )

    payload = json.loads(result.content)

    assert payload["truncated"] is True
    assert payload["preview"] == "y" * 20
    assert "result_ref" not in payload


def test_read_tool_result_reads_persisted_default_temp_result():
    harness = make_harness(HarnessConfig(enable_tool_result_reader=True))
    harness.tools.register(ToolDef(
        name="large_temp_tool",
        description="Large temp test tool",
        parameters={"type": "object", "properties": {}},
        handler=lambda args: "z" * 1500,
        max_result_chars=20,
    ))

    result = harness.execute_tool(
        "large_temp_tool",
        {},
        context=ToolUseContext(session_id="read-tool-result-session"),
    )
    payload = json.loads(result.content)

    read_result = harness.execute_tool(
        "read_tool_result",
        {"result_ref": payload["result_ref"], "max_chars": 1000},
    )
    read_payload = json.loads(read_result.content)

    assert read_payload["result_ref"] == payload["result_ref"]
    assert read_payload["chars"] == 1500
    assert read_payload["returned_chars"] == 1000
    assert read_payload["truncated"] is True
    assert read_payload["content"] == "z" * 1000


def test_read_tool_result_defaults_to_limited_window_and_caps_max_chars():
    harness = make_harness(HarnessConfig(enable_tool_result_reader=True))
    harness.tools.register(ToolDef(
        name="large_tool",
        description="Large test tool",
        parameters={"type": "object", "properties": {}},
        handler=lambda args: "a" * 60000,
        max_result_chars=20,
    ))
    persisted = json.loads(harness.execute_tool("large_tool", {}).content)

    default_payload = json.loads(harness.execute_tool(
        "read_tool_result", {"result_ref": persisted["result_ref"]},
    ).content)
    assert default_payload["returned_chars"] == 12000
    assert default_payload["truncated"] is True

    capped_payload = json.loads(harness.execute_tool(
        "read_tool_result",
        {"result_ref": persisted["result_ref"], "max_chars": 100000},
    ).content)
    assert capped_payload["returned_chars"] == 50000
    assert capped_payload["truncated"] is True


def test_read_tool_result_rejects_non_tool_result_path(tmp_path):
    harness = make_harness(HarnessConfig(enable_tool_result_reader=True))
    ordinary_file = tmp_path / "ordinary.txt"
    ordinary_file.write_text("secret", encoding="utf-8")

    result = harness.execute_tool(
        "read_tool_result",
        {"result_ref": "result:missing", "max_chars": 1000},
    )
    payload = json.loads(result.content)

    assert "error" in payload
    assert "工具结果引用" in payload["error"]


def test_trace_records_worker_policy_block():
    harness = make_harness()

    result = harness.execute_tool(
        "ask_user",
        {"question": "Choose?", "options": [{"label": "A"}]},
        context=ToolUseContext(source="worker", confirmed=False),
    )
    events = harness.trace.snapshot()

    assert result.blocked
    assert [event.event_type for event in events] == ["tool_start", "tool_blocked"]
    assert events[-1].source == "worker"
    assert "不允许由 Worker 执行" in events[-1].payload["block_reason"]


def test_tool_executor_partitions_tool_calls_by_concurrency_policy():
    harness = make_harness()
    register_confirmed_write_tool(harness)
    executor = ToolExecutor(harness)

    batches = executor.partition_tool_calls([
        (make_tool_call("query", "t1"), {"object_type": "Asset"}),
        (make_tool_call("search", "t2"), {"keyword": "Asset"}),
        (make_tool_call("write_record", "t3"), {}),
        (make_tool_call("lookup_asset", "t4"), {"asset_id": "A1"}),
    ])

    assert [[tc.function.name for tc, _ in batch] for batch in batches] == [
        ["query", "search"],
        ["write_record"],
        ["lookup_asset"],
    ]


def test_tool_executor_runs_concurrency_safe_batch_in_parallel():
    harness = make_harness()
    executor = ToolExecutor(harness)

    def slow_result(label):
        def handler(args):
            time.sleep(0.08)
            return json.dumps({"label": label})
        return handler

    harness.tools.register(ToolDef(
        name="slow_a",
        description="Slow read tool A",
        parameters={"type": "object", "properties": {}},
        handler=slow_result("a"),
        policy=ToolPolicy(read_only=True, concurrency_safe=True),
    ))
    harness.tools.register(ToolDef(
        name="slow_b",
        description="Slow read tool B",
        parameters={"type": "object", "properties": {}},
        handler=slow_result("b"),
        policy=ToolPolicy(read_only=True, concurrency_safe=True),
    ))
    state = RunState(messages=[], session_id="s1", user_question="")

    start = time.perf_counter()
    results = executor.execute_tool_calls([
        (make_tool_call("slow_a", "t1"), {}),
        (make_tool_call("slow_b", "t2"), {}),
    ], state)
    elapsed = time.perf_counter() - start

    assert [tc.id for tc, _, _ in results] == ["t1", "t2"]
    assert [json.loads(result.content)["label"] for _, _, result in results] == ["a", "b"]
    assert elapsed < 0.14


def test_tool_executor_stops_before_later_calls_when_confirmation_needed():
    harness = make_harness(HarnessConfig(enable_write_confirmation=True))
    register_confirmed_write_tool(harness)
    executor = ToolExecutor(harness)
    state = RunState(messages=[], session_id="s1", user_question="")

    results = executor.execute_tool_calls([
        (make_tool_call("write_record", "t1"), {}),
        (make_tool_call("lookup_asset", "t2"), {"asset_id": "A1"}),
    ], state)

    assert [tc.id for tc, _, _ in results] == ["t1"]
    assert results[0][2].needs_confirmation


def test_tool_executor_stops_before_later_calls_when_user_input_needed():
    harness = make_harness()
    executor = ToolExecutor(harness)
    state = RunState(messages=[], session_id="s1", user_question="")

    results = executor.execute_tool_calls([
        (make_tool_call("ask_user", "t1"), {
            "question": "Choose?",
            "options": [{"label": "A"}],
        }),
        (make_tool_call("lookup_asset", "t2"), {"asset_id": "A1"}),
    ], state)

    assert [tc.id for tc, _, _ in results] == ["t1"]
    assert results[0][2].needs_user_input


def test_query_loop_records_final_response_transition(monkeypatch):
    harness = make_harness()

    def fake_call_llm_with_retry(*args, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="This is a complete final answer for the user.",
                        tool_calls=None,
                    ),
                ),
            ],
        )

    monkeypatch.setattr("oag.loop.query_loop.call_llm_with_retry", fake_call_llm_with_retry)
    pending = []
    loop = QueryLoop(
        harness,
        DummyClient(),
        "dummy-model",
        on_pending_confirmation=lambda *args: pending.append(args),
    )
    state = RunState(
        messages=[{"role": "system", "content": "System prompt"}, {"role": "user", "content": "Question?"}],
        session_id="s1",
        user_question="Question?",
    )

    events = list(loop.run(state))
    trace_events = harness.trace.snapshot()

    assert [event.type for event in events] == [
        "debug", "debug", "assistant_delta", "assistant_end",
    ]
    assert events[-2].content == "This is a complete final answer for the user."
    assert events[-1].kind == "final"
    assert pending == []
    assert any(event.event_type == "context_usage" for event in trace_events)
    assert trace_events[-1].event_type == "agent_transition"
    assert trace_events[-1].payload["reason"] == "final_response"


def test_query_loop_finalizes_without_tools_after_turn_limit(monkeypatch):
    harness = make_harness(HarnessConfig(max_turns=1))
    calls = []

    def fake_call_llm_with_retry(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return make_response(tool_calls=[
                make_full_tool_call("lookup_asset", "tool_1", '{"asset_id":"A1"}'),
            ])
        assert kwargs["tools"] is None
        assert "不要再调用任何工具" in kwargs["messages"][-1]["content"]
        return make_response(content="Based on the retrieved record, asset A1 is available.")

    monkeypatch.setattr("oag.loop.query_loop.call_llm_with_retry", fake_call_llm_with_retry)
    loop = QueryLoop(
        harness,
        DummyClient(),
        "dummy-model",
        on_pending_confirmation=lambda *args: None,
    )
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Look up asset A1"},
    ]
    state = RunState(messages=messages, session_id="s1", user_question="Look up asset A1")

    events = list(loop.run(state))
    trace_events = harness.trace.snapshot()

    assert len(calls) == 2
    assert events[-2].type == "assistant_delta"
    assert events[-2].content == "Based on the retrieved record, asset A1 is available."
    assert events[-1].type == "assistant_end"
    assert events[-1].kind == "final"
    assert all(
        "最大轮次" not in event.content
        for event in events
        if event.type == "assistant_delta"
    )
    assert messages[-1] == {
        "role": "assistant",
        "content": "Based on the retrieved record, asset A1 is available.",
    }
    assert trace_events[-1].payload["reason"] == "max_turns_final_response"


def test_query_loop_emits_reasoning_event_without_persisting_it(monkeypatch):
    harness = make_harness()

    def fake_call_llm_with_retry(*args, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="This is a complete final answer for the user.",
                        reasoning_content="Internal reasoning trace.",
                        tool_calls=None,
                    ),
                ),
            ],
        )

    monkeypatch.setattr("oag.loop.query_loop.call_llm_with_retry", fake_call_llm_with_retry)
    loop = QueryLoop(
        harness,
        DummyClient(),
        "dummy-model",
        on_pending_confirmation=lambda *args: None,
    )
    messages = [{"role": "system", "content": "System prompt"}, {"role": "user", "content": "Question?"}]
    state = RunState(messages=messages, session_id="s1", user_question="Question?")

    events = list(loop.run(state))

    assert [event.type for event in events] == [
        "debug", "debug", "reasoning", "assistant_delta", "assistant_end",
    ]
    assert events[2].content == "Internal reasoning trace."
    assert all("Internal reasoning trace." not in msg.get("content", "") for msg in messages)


def test_query_loop_streams_reasoning_and_text_without_duplicate_final_text(monkeypatch):
    harness = make_harness()

    def fake_call_llm_with_retry(*args, **kwargs):
        assert kwargs["stream"] is True
        return iter([
            make_stream_chunk(reasoning="Think "),
            make_stream_chunk(reasoning="step."),
            make_stream_chunk(content="This is a complete "),
            make_stream_chunk(content="final answer for the user."),
        ])

    monkeypatch.setattr("oag.loop.query_loop.call_llm_with_retry", fake_call_llm_with_retry)
    loop = QueryLoop(
        harness,
        DummyClient(),
        "dummy-model",
        on_pending_confirmation=lambda *args: None,
    )
    messages = [{"role": "system", "content": "System prompt"}, {"role": "user", "content": "Question?"}]
    state = RunState(messages=messages, session_id="s1", user_question="Question?")

    events = list(loop.run(state))

    assert [event.type for event in events] == [
        "debug",
        "reasoning",
        "reasoning",
        "assistant_delta",
        "assistant_delta",
        "debug",
        "assistant_end",
    ]
    assert [event.content for event in events if event.type == "reasoning"] == ["Think ", "step."]
    assert [event.content for event in events if event.type == "assistant_delta"] == [
        "This is a complete ",
        "final answer for the user.",
    ]
    assert [event.kind for event in events if event.type == "assistant_end"] == ["final"]
    assert messages[-1] == {
        "role": "assistant",
        "content": "This is a complete final answer for the user.",
    }


def test_query_loop_separates_tool_progress_from_final_text(monkeypatch):
    harness = make_harness()
    calls = 0

    def fake_call_llm_with_retry(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return iter([
                make_stream_chunk(content="I will look that up."),
                make_stream_chunk(tool_call=make_tool_delta(
                    0,
                    tool_id="tool_1",
                    name="lookup_asset",
                    arguments='{"asset_id":"A1"}',
                )),
            ])
        return iter([make_stream_chunk(content="Asset A1 is available.")])

    monkeypatch.setattr("oag.loop.query_loop.call_llm_with_retry", fake_call_llm_with_retry)
    loop = QueryLoop(
        harness,
        DummyClient(),
        "dummy-model",
        on_pending_confirmation=lambda *args: None,
    )
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Explain A1"},
    ]

    events = list(loop.run(RunState(
        messages=messages,
        session_id="s1",
        user_question="Explain A1",
    )))

    assert [event.content for event in events if event.type == "assistant_delta"] == [
        "I will look that up.",
        "Asset A1 is available.",
    ]
    assert [event.kind for event in events if event.type == "assistant_end"] == [
        "progress",
        "final",
    ]
    assert messages[-1] == {
        "role": "assistant",
        "content": "Asset A1 is available.",
    }
    response_traces = [
        event for event in harness.trace.snapshot()
        if event.event_type == "assistant_response"
    ]
    assert [event.payload["kind"] for event in response_traces] == [
        "progress", "final",
    ]
    assert response_traces[0].payload["content"] == "I will look that up."
    assert response_traces[1].payload["content"] == "Asset A1 is available."


def test_query_loop_does_not_trace_empty_tool_preface(monkeypatch):
    harness = make_harness()
    calls = 0

    def fake_call_llm_with_retry(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return make_response(tool_calls=[
                make_full_tool_call("lookup_asset", "tool_1", '{"asset_id":"A1"}'),
            ])
        return make_response(content="Asset A1 is available.")

    monkeypatch.setattr("oag.loop.query_loop.call_llm_with_retry", fake_call_llm_with_retry)
    loop = QueryLoop(
        harness,
        DummyClient(),
        "dummy-model",
        on_pending_confirmation=lambda *args: None,
    )
    messages = [{"role": "user", "content": "Explain A1"}]

    list(loop.run(RunState(
        messages=messages,
        session_id="s1",
        user_question="Explain A1",
    )))

    response_traces = [
        event for event in harness.trace.snapshot()
        if event.event_type == "assistant_response"
    ]
    assert len(response_traces) == 1
    assert response_traces[0].payload["kind"] == "final"


def test_query_loop_aggregates_streaming_tool_calls(monkeypatch):
    harness = make_harness()

    def fake_call_llm_with_retry(*args, **kwargs):
        return iter([
            make_stream_chunk(tool_call=make_tool_delta(0, tool_id="tool_1", name="lookup_asset", arguments='{"asset')),
            make_stream_chunk(tool_call=make_tool_delta(0, arguments='_id":"A1"}')),
        ])

    monkeypatch.setattr("oag.loop.query_loop.call_llm_with_retry", fake_call_llm_with_retry)
    loop = QueryLoop(
        harness,
        DummyClient(),
        "dummy-model",
        on_pending_confirmation=lambda *args: None,
    )
    messages = [{"role": "system", "content": "System prompt"}, {"role": "user", "content": "Question?"}]
    state = RunState(messages=messages, session_id="s1", user_question="Question?")

    events = list(loop.run(state))

    assert [event.type for event in events[:5]] == [
        "debug", "debug", "assistant_end", "tool_call", "tool_result",
    ]
    assert events[3].name == "lookup_asset"
    assert events[3].args == {"asset_id": "A1"}
    assert '"asset_id": "A1"' in events[4].result
    assert messages[2]["tool_calls"][0]["function"]["arguments"] == '{"asset_id":"A1"}'


def test_query_loop_emits_interaction_event_for_action_input(monkeypatch):
    harness = make_harness(include_actions=True)

    def fake_call_llm_with_retry(*args, **kwargs):
        if not any(message.get("role") == "tool" for message in kwargs["messages"]):
            return make_response(tool_calls=[
                make_full_tool_call(
                    "request_action_input",
                    "tool_interaction",
                    '{"action_id":"create_work_order","context_id":"A1"}',
                ),
            ])
        return make_response(content="Waiting for input.")

    monkeypatch.setattr("oag.loop.query_loop.call_llm_with_retry", fake_call_llm_with_retry)
    loop = QueryLoop(
        harness,
        DummyClient(),
        "dummy-model",
        on_pending_confirmation=lambda *args: None,
    )
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Create work order"},
    ]

    events = list(loop.run(RunState(
        messages=messages,
        session_id="s1",
        user_question="Create work order",
    )))

    interaction = next(event for event in events if event.type == "interaction")
    assert interaction.name == "request_action_input"
    assert interaction.payload["kind"] == "action_form"
    assert interaction.payload["context_id"] == "A1"


def test_query_loop_ignores_interaction_payload_from_non_interaction_tool(monkeypatch):
    harness = make_harness()
    harness.tools.register(ToolDef(
        name="ordinary_query",
        description="Return ordinary domain data",
        parameters={"type": "object", "properties": {}},
        handler=lambda args: json.dumps({"interaction": {"kind": "untrusted"}}),
        category="query",
    ))

    def fake_call_llm_with_retry(*args, **kwargs):
        if not any(message.get("role") == "tool" for message in kwargs["messages"]):
            return make_response(tool_calls=[
                make_full_tool_call("ordinary_query", "tool_query", "{}"),
            ])
        return make_response(content="Done.")

    monkeypatch.setattr("oag.loop.query_loop.call_llm_with_retry", fake_call_llm_with_retry)
    loop = QueryLoop(
        harness,
        DummyClient(),
        "dummy-model",
        on_pending_confirmation=lambda *args: None,
    )
    messages = [{"role": "system", "content": "System prompt"}, {"role": "user", "content": "Query"}]
    events = list(loop.run(RunState(messages=messages, session_id="s1", user_question="Query")))

    assert all(event.type != "interaction" for event in events)


def test_confirmation_required_stops_before_later_tool_calls(monkeypatch):
    harness = make_harness(HarnessConfig(enable_write_confirmation=True))
    register_confirmed_write_tool(harness)
    executed = []
    original_execute = harness.execute_tool

    def recording_execute(tool_name, args, **kwargs):
        executed.append(tool_name)
        return original_execute(tool_name, args, **kwargs)

    harness.execute_tool = recording_execute
    pending = []

    def fake_call_llm_with_retry(*args, **kwargs):
        return make_response(tool_calls=[
            make_full_tool_call("write_record", "tool_1", '{}'),
            make_full_tool_call("lookup_asset", "tool_2", '{"asset_id":"A1"}'),
        ])

    monkeypatch.setattr("oag.loop.query_loop.call_llm_with_retry", fake_call_llm_with_retry)
    loop = QueryLoop(
        harness,
        DummyClient(),
        "dummy-model",
        on_pending_confirmation=lambda *args: pending.append(args),
    )
    messages = [{"role": "system", "content": "System prompt"}, {"role": "user", "content": "Create work order"}]
    state = RunState(messages=messages, session_id="s1", user_question="Create work order")

    events = list(loop.run(state))

    assert [event.type for event in events] == [
        "debug", "debug", "assistant_end", "tool_call", "confirmation_required",
    ]
    assert events[3].name == "write_record"
    assert events[3].args == {}
    assert executed == ["write_record"]
    assert len(pending) == 1
    assert pending[0][6] == [{"tool_call_id": "tool_2", "content": '{"skipped": true, "reason": "前一个工具调用需要用户确认，本调用未执行"}'}]
    assert all(m.get("tool_call_id") != "tool_2" for m in messages)


def test_confirmation_flow_appends_skipped_tool_results_in_order():
    harness = make_harness()
    saved = []
    continued_states = []

    def run_loop(state):
        continued_states.append(state)
        return iter(())

    flow = ConfirmationFlow(
        harness,
        save_messages=lambda session_id, messages: saved.append((session_id, messages)),
        run_loop=run_loop,
    )
    messages = [{"role": "system", "content": "System prompt"}]
    pending = PendingConfirmation(
        session_id="s1",
        tool_name="ask_user",
        args={"question": "Choose?", "options": [{"label": "A"}]},
        tool_call_id="tool_1",
        messages=messages,
        skipped_tool_calls=[{"tool_call_id": "tool_2", "content": '{"skipped": true}'}],
        expects_answer=True,
        user_question="Original question",
        turn_count=3,
        query_complete_retry_active=True,
    )

    list(flow.confirm(pending, approved=True, answer="A"))

    assert [m.get("tool_call_id") for m in messages if m["role"] == "tool"] == ["tool_1", "tool_2"]
    assert continued_states[0].user_question == "Original question"
    assert continued_states[0].turn_count == 3
    assert continued_states[0].query_complete_retry_active is True


def test_query_loop_invalid_tool_json_returns_tool_error(monkeypatch):
    harness = make_harness()
    calls = []

    def fake_call_llm_with_retry(*args, **kwargs):
        calls.append(kwargs["messages"])
        if len(calls) == 1:
            return make_response(tool_calls=[
                make_full_tool_call("lookup_asset", "tool_1", '{"asset_id":'),
            ])
        return make_response(content="I handled the tool error.")

    monkeypatch.setattr("oag.loop.query_loop.call_llm_with_retry", fake_call_llm_with_retry)
    loop = QueryLoop(
        harness,
        DummyClient(),
        "dummy-model",
        on_pending_confirmation=lambda *args: None,
    )
    messages = [{"role": "system", "content": "System prompt"}, {"role": "user", "content": "Lookup"}]
    events = list(loop.run(RunState(messages=messages, session_id="s1", user_question="Lookup")))

    assert any(event.type == "tool_result" and "工具参数不是合法 JSON" in event.result for event in events)
    assert messages[3]["role"] == "tool"
    assert "工具参数不是合法 JSON" in messages[3]["content"]


def test_query_loop_compacts_before_every_request(monkeypatch):
    harness = make_harness()
    calls = []

    def fake_maybe_compact(messages):
        calls.append(len(messages))
        if len(calls) == 1:
            return messages + [
                {"role": "user", "content": "[前置对话摘要]\nsummary"},
                {"role": "assistant", "content": "好的，我已了解前面的对话内容。请继续。"},
            ], True
        return messages, False

    harness.maybe_compact = fake_maybe_compact

    def fake_call_llm_with_retry(*args, **kwargs):
        assert any(m.get("content") == "[前置对话摘要]\nsummary" for m in kwargs["messages"])
        return make_response(content="This compacted response fully answers the user question.")

    monkeypatch.setattr("oag.loop.query_loop.call_llm_with_retry", fake_call_llm_with_retry)
    loop = QueryLoop(
        harness,
        DummyClient(),
        "dummy-model",
        on_pending_confirmation=lambda *args: None,
    )
    messages = [{"role": "system", "content": "System prompt"}, {"role": "user", "content": "Question?"}]

    events = list(loop.run(RunState(messages=messages, session_id="s1", user_question="Question?")))

    assert [event.type for event in events[:2]] == ["compact", "debug"]
    assert calls == [2]


def test_query_loop_force_compacts_and_retries_on_context_overflow(monkeypatch):
    harness = make_harness()
    calls = []
    force_calls = []

    harness.maybe_compact = lambda messages: (messages, False)

    def fake_force_compact(messages):
        force_calls.append(messages)
        return [
            messages[0],
            {"role": "user", "content": "[前置对话摘要]\nsummary"},
            {"role": "assistant", "content": "好的，我已了解前面的对话内容。请继续。"},
            messages[-1],
        ], True

    harness.force_compact = fake_force_compact

    def fake_call_llm_with_retry(*args, **kwargs):
        calls.append(kwargs["messages"])
        if len(calls) == 1:
            raise ValueError("context_length_exceeded: maximum context length")
        assert any(m.get("content") == "[前置对话摘要]\nsummary" for m in kwargs["messages"])
        return make_response(content="This recovered response fully answers the user question.")

    monkeypatch.setattr("oag.loop.query_loop.call_llm_with_retry", fake_call_llm_with_retry)
    loop = QueryLoop(
        harness,
        DummyClient(),
        "dummy-model",
        on_pending_confirmation=lambda *args: None,
    )
    messages = [{"role": "system", "content": "System prompt"}, {"role": "user", "content": "Question?"}]

    events = list(loop.run(RunState(messages=messages, session_id="s1", user_question="Question?")))

    assert len(calls) == 2
    assert len(force_calls) == 1
    assert any(event.type == "compact" for event in events)
    assert events[-2].content == "This recovered response fully answers the user question."
    assert events[-1].kind == "final"


def test_context_compaction_preserves_tool_call_pairs(monkeypatch):
    mgr = ContextManager(DummyClient(), "dummy-model", context_window=100)
    monkeypatch.setattr(mgr, "_summarize", lambda messages: "summary")
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "old " * 100},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "tool_1", "type": "function", "function": {"name": "lookup_asset", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "tool_1", "content": "result"},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "more"},
        {"role": "assistant", "content": "more"},
        {"role": "user", "content": "final"},
    ]

    compacted, did_compact = mgr.maybe_compact(messages)

    assert did_compact
    tool_idx = next(i for i, m in enumerate(compacted) if m.get("role") == "tool")
    assert compacted[tool_idx - 1]["role"] == "assistant"
    assert compacted[tool_idx - 1]["tool_calls"][0]["id"] == "tool_1"


def test_confirmation_flow_handles_missing_pending():
    harness = make_harness()
    saved = []
    flow = ConfirmationFlow(
        harness,
        save_messages=lambda session_id, messages: saved.append((session_id, messages)),
        run_loop=lambda state: iter(()),
    )

    events = list(flow.confirm(None, approved=True))

    assert [event.type for event in events] == ["text"]
    assert events[0].content == "没有待确认的操作。"
    assert saved == []


def test_confirmation_flow_denial_saves_rejection_messages():
    harness = make_harness()
    saved = []
    flow = ConfirmationFlow(
        harness,
        save_messages=lambda session_id, messages: saved.append((session_id, messages)),
        run_loop=lambda state: iter(()),
    )
    messages = [{"role": "system", "content": "System prompt"}]
    pending = PendingConfirmation(
        session_id="s1",
        tool_name="write_record",
        args={},
        tool_call_id="tool_1",
        messages=messages,
    )

    events = list(flow.confirm(pending, approved=False))

    assert [event.type for event in events] == ["text"]
    assert events[0].content == "已取消 write_record 的执行。"
    assert saved == [("s1", messages)]
    assert messages[-2]["role"] == "tool"
    assert "用户拒绝执行" in messages[-2]["content"]
    assert messages[-1]["role"] == "user"
    assert "用户拒绝了 write_record" in messages[-1]["content"]


def test_session_store_persists_and_lists_sessions(tmp_path):
    store = SessionStore(str(tmp_path / "chat.db"))
    messages = [{"role": "user", "content": "hello"}]

    assert store.get("s1") == []

    store.save("s1", messages)
    sessions = store.list_sessions()

    assert store.get("s1") == messages
    assert sessions[0]["session_id"] == "s1"
    assert sessions[0]["updated_at"]


def test_message_sanitizer_repairs_missing_tool_results():
    messages = [
        {"role": "user", "content": "Lookup"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "tool_1", "type": "function", "function": {"name": "lookup_asset", "arguments": "{}"}},
            {"id": "tool_2", "type": "function", "function": {"name": "query", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "tool_1", "content": "{}"},
        {"role": "user", "content": "Continue"},
    ]

    repaired, changed = sanitize_messages(messages)

    assert changed
    tool_results = [m for m in repaired if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_results] == ["tool_1", "tool_2"]
    assert "历史恢复时发现缺失的工具结果" in tool_results[1]["content"]


def test_message_sanitizer_drops_orphan_tool_results_and_empty_assistant():
    messages = [
        {"role": "system", "content": "System"},
        {"role": "tool", "tool_call_id": "missing", "content": "{}"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "Hello"},
    ]

    repaired, changed = sanitize_messages(messages)

    assert changed
    assert [m["role"] for m in repaired] == ["system", "user"]


def test_session_store_sanitizes_loaded_history(tmp_path):
    store = SessionStore(str(tmp_path / "chat.db"))
    raw_messages = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "tool_1", "type": "function", "function": {"name": "lookup_asset", "arguments": "{}"}}
        ]},
    ]
    store.conn.execute(
        "INSERT OR REPLACE INTO chat_history (session_id, messages) VALUES (?, ?)",
        ("s1", __import__("json").dumps(raw_messages)),
    )
    store.conn.commit()

    loaded = store.get("s1")

    assert loaded[-1]["role"] == "tool"
    assert loaded[-1]["tool_call_id"] == "tool_1"


def test_query_complete_has_no_default_policy():
    harness = make_harness()
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "I will call a tool now."},
        {"role": "tool", "tool_call_id": "tool_1", "content": "{}"},
        {"role": "assistant", "content": ""},
    ]

    result = harness.run_query_complete_hooks("Question", messages)

    assert result is None


def test_query_loop_filters_tools_per_run_allowed_tools(monkeypatch):
    harness = make_harness()
    available = [
        tool["function"]["name"]
        for tool in harness.build_tools()
        if tool.get("function", {}).get("name")
    ]
    assert len(available) >= 2
    allowed = frozenset({available[0]})
    captured = {}

    def fake_call_llm_with_retry(*args, **kwargs):
        captured["tools"] = [
            tool["function"]["name"]
            for tool in kwargs.get("tools") or []
        ]
        return make_response(content="Done.")

    monkeypatch.setattr("oag.loop.query_loop.call_llm_with_retry", fake_call_llm_with_retry)
    loop = QueryLoop(
        harness,
        DummyClient(),
        "dummy-model",
        on_pending_confirmation=lambda *args: None,
    )
    state = RunState(
        messages=[{"role": "system", "content": "System prompt"}, {"role": "user", "content": "Question?"}],
        session_id="s1",
        user_question="Question?",
        allowed_tools=allowed,
    )

    events = list(loop.run(state))

    assert captured["tools"] == [available[0]]
    assert events[-2].type == "assistant_delta"
    assert events[-2].content == "Done."
    assert events[-1].type == "assistant_end"
    assert events[-1].kind == "final"


def test_query_complete_runs_caller_registered_hook():
    harness = make_harness()
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Incomplete"},
    ]
    captured = []
    harness.register_query_complete_hook(
        lambda context: (
            captured.append(context)
            or HookResult(action="pause", reason="missing evidence")
        ),
    )

    result = harness.run_query_complete_hooks("Question", messages)

    assert captured == [{"messages": messages, "user_question": "Question"}]
    assert "missing evidence" in result
