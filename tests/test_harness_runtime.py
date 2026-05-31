from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from oag.harness import Harness, HarnessConfig
from oag.loop.confirmation_flow import ConfirmationFlow
from oag.loop.query_loop import QueryLoop
from oag.loop.tool_executor import ToolExecutor
from oag.llm.context import ContextManager
from oag.runtime.message_sanitizer import sanitize_messages
from oag.ontology.registry import FunctionRegistry
from oag.runtime import PendingConfirmation, RunState, ToolUseContext
from oag.ontology.schema import (
    Effect,
    FunctionDef,
    FunctionParam,
    Ontology,
    ObjectTypeDef,
    Precondition,
    PropertyDef,
)
from oag.runtime.session_store import SessionStore
from oag.ontology.store import Store
from oag.tools.registry import ToolDef, ToolPolicy


class DummyClient:
    pass


def make_harness(config: HarnessConfig | None = None) -> Harness:
    ontology = Ontology(
        name="TestDomain",
        description="Test domain",
        objects={
            "Asset": ObjectTypeDef(
                summary="Asset summary",
                description="Asset full description",
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
                data_source="agent_generated",
                mutability="mutable",
                properties={
                    "order_id": PropertyDef(type="str", required=True, description="Order id"),
                    "status": PropertyDef(type="str", description="Order status"),
                },
            ),
        },
        functions={
            "lookup_asset": FunctionDef(
                summary="Lookup an asset",
                description="Lookup asset details",
                function_type="get",
                params={"asset_id": FunctionParam(type="str", description="Asset id")},
                involves_objects=["Asset"],
            ),
            "create_work_order": FunctionDef(
                summary="Create a work order",
                description="Create work order details",
                usage_prompt="创建前必须确认 asset_id 指向真实资产，并说明写入影响。",
                hint="Only create when user explicitly asks.",
                function_type="business",
                writes_to=["WorkOrder"],
                involves_objects=["Asset", "WorkOrder"],
                params={"asset_id": FunctionParam(type="str", description="Asset id")},
                preconditions=[
                    Precondition(object="Asset", field="asset_id", operator="exists"),
                ],
                effects=[
                    Effect(object="WorkOrder", field="status", set_to="created"),
                ],
            ),
        },
    )
    store = Store(ontology)
    store.create_tables()
    store.load_data("Asset", [{"asset_id": "A1", "status": "ok"}])

    registry = FunctionRegistry()
    registry.register(
        "lookup_asset",
        lambda asset_id: {"asset_id": asset_id, "status": "ok"},
        ontology.functions["lookup_asset"],
    )
    registry.register(
        "create_work_order",
        lambda asset_id: {"order_id": "WO1", "asset_id": asset_id, "status": "created"},
        ontology.functions["create_work_order"],
    )

    return Harness(
        ontology,
        store,
        registry,
        DummyClient(),
        "dummy-model",
        config or HarnessConfig(enable_write_confirmation=False),
    )


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
    assert "- lookup_asset[get]: Lookup an asset" in prompt
    assert "## 函数完整定义" not in prompt
    assert "### 函数: lookup_asset" not in prompt
    assert "## 对象完整定义" not in prompt
    assert "首次调用函数时系统会自动注入" not in prompt
    assert "完整函数、对象、规则定义请调用 inspect 获取" in prompt

    details = json.loads(harness.execute_tool("inspect", {"name": "create_work_order"}).content)

    assert details["usage_prompt"] == "创建前必须确认 asset_id 指向真实资产，并说明写入影响。"
    assert details["preconditions"][0]["object"] == "Asset"
    assert details["effects"][0]["set_to"] == "created"


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


def test_prompt_can_include_full_context_for_compatibility():
    harness = make_harness(HarnessConfig(
        enable_write_confirmation=False,
        include_ontology_full_context=True,
    ))

    prompt = harness.build_system_prompt()

    assert "## 函数完整定义" in prompt
    assert "### 函数: lookup_asset" in prompt


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


def test_function_usage_prompt_is_added_to_tool_description():
    harness = make_harness()

    tools = harness.build_tools()
    create_tool = next(t for t in tools if t["function"]["name"] == "create_work_order")

    assert "Create a work order" in create_tool["function"]["description"]
    assert "使用说明:" in create_tool["function"]["description"]
    assert "创建前必须确认 asset_id 指向真实资产" in create_tool["function"]["description"]


def test_runtime_tools_have_usage_prompts():
    harness = make_harness()

    tools = harness.build_tools()
    dispatch_tool = next(t for t in tools if t["function"]["name"] == "dispatch_workers")
    ask_tool = next(t for t in tools if t["function"]["name"] == "ask_user")

    assert "使用说明:" in dispatch_tool["function"]["description"]
    assert "相互独立的只读子任务" in dispatch_tool["function"]["description"]
    assert "不要询问可以通过只读工具直接查到的信息" in ask_tool["function"]["description"]


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
    context = ToolUseContext(source="worker", confirmed=False)

    mutate_result = harness.execute_tool(
        "mutate",
        {"operation": "create", "object_type": "WorkOrder", "data": {"order_id": "WO2"}},
        context=context,
    )
    ask_user_result = harness.execute_tool(
        "ask_user",
        {"question": "Choose?", "options": [{"label": "A"}]},
        context=context,
    )
    write_fn_result = harness.execute_tool(
        "create_work_order",
        {"asset_id": "A1"},
        context=context,
    )
    read_fn_result = harness.execute_tool(
        "lookup_asset",
        {"asset_id": "A1"},
        context=context,
    )

    assert mutate_result.blocked
    assert ask_user_result.blocked
    assert write_fn_result.blocked
    assert not read_fn_result.blocked


def test_trace_records_successful_tool_execution():
    harness = make_harness()

    result = harness.execute_tool("lookup_asset", {"asset_id": "A1"})
    events = harness.trace.snapshot()

    assert not result.blocked
    assert [event.event_type for event in events] == ["tool_start", "tool_end"]
    assert events[0].payload["tool_name"] == "lookup_asset"
    assert events[1].payload["content_preview"]


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
    harness = make_harness()
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


def test_tool_pipeline_persists_large_tool_result_to_system_temp_by_default():
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

    assert payload["path"].startswith(str(Path(tempfile.gettempdir()) / "oag-tool-results"))


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
    executor = ToolExecutor(harness)

    batches = executor.partition_tool_calls([
        (make_tool_call("query", "t1"), {"object_type": "Asset"}),
        (make_tool_call("count", "t2"), {"object_type": "Asset"}),
        (make_tool_call("mutate", "t3"), {"operation": "create", "object_type": "WorkOrder"}),
        (make_tool_call("lookup_asset", "t4"), {"asset_id": "A1"}),
    ])

    assert [[tc.function.name for tc, _ in batch] for batch in batches] == [
        ["query", "count"],
        ["mutate"],
        ["lookup_asset"],
    ]


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

    assert [event.type for event in events] == ["debug", "debug", "text"]
    assert events[-1].content == "This is a complete final answer for the user."
    assert pending == []
    assert trace_events[-1].event_type == "agent_transition"
    assert trace_events[-1].payload["reason"] == "final_response"


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

    assert [event.type for event in events] == ["debug", "debug", "reasoning", "text"]
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

    assert [event.type for event in events] == ["debug", "reasoning", "reasoning", "text", "text", "debug"]
    assert [event.content for event in events if event.type == "reasoning"] == ["Think ", "step."]
    assert [event.content for event in events if event.type == "text"] == [
        "This is a complete ",
        "final answer for the user.",
    ]
    assert messages[-1] == {
        "role": "assistant",
        "content": "This is a complete final answer for the user.",
    }


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

    assert [event.type for event in events[:3]] == ["debug", "debug", "tool_call"]
    assert events[2].name == "lookup_asset"
    assert '"asset_id": "A1"' in events[2].result
    assert messages[2]["tool_calls"][0]["function"]["arguments"] == '{"asset_id":"A1"}'


def test_confirmation_required_stops_before_later_tool_calls(monkeypatch):
    harness = make_harness(HarnessConfig(enable_write_confirmation=True))
    executed = []
    original_execute = harness.execute_tool

    def recording_execute(tool_name, args, **kwargs):
        executed.append(tool_name)
        return original_execute(tool_name, args, **kwargs)

    harness.execute_tool = recording_execute
    pending = []

    def fake_call_llm_with_retry(*args, **kwargs):
        return make_response(tool_calls=[
            make_full_tool_call("create_work_order", "tool_1", '{"asset_id":"A1"}'),
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

    assert [event.type for event in events] == ["debug", "debug", "confirmation_required"]
    assert executed == ["create_work_order"]
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
        user_question="Original question",
        turn_count=3,
        stop_hook_active=True,
    )

    list(flow.confirm(pending, approved=True, answer="A"))

    assert [m.get("tool_call_id") for m in messages if m["role"] == "tool"] == ["tool_1", "tool_2"]
    assert continued_states[0].user_question == "Original question"
    assert continued_states[0].turn_count == 3
    assert continued_states[0].stop_hook_active is True


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

    assert any(event.type == "tool_call" and "工具参数不是合法 JSON" in event.result for event in events)
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
    assert events[-1].content == "This recovered response fully answers the user question."


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
        tool_name="mutate",
        args={"operation": "create"},
        tool_call_id="tool_1",
        messages=messages,
    )

    events = list(flow.confirm(pending, approved=False))

    assert [event.type for event in events] == ["text"]
    assert events[0].content == "已取消 mutate 的执行。"
    assert saved == [("s1", messages)]
    assert messages[-2]["role"] == "tool"
    assert "用户拒绝执行" in messages[-2]["content"]
    assert messages[-1]["role"] == "user"
    assert "用户拒绝了 mutate" in messages[-1]["content"]


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
            {"id": "tool_2", "type": "function", "function": {"name": "count", "arguments": "{}"}},
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


def test_stop_check_treats_empty_latest_assistant_as_incomplete():
    harness = make_harness()
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "I will call a tool now."},
        {"role": "tool", "tool_call_id": "tool_1", "content": "{}"},
        {"role": "assistant", "content": ""},
    ]

    result = harness.run_stop_check("Question", messages)

    assert result is not None
    assert "未生成最终回答" in result
