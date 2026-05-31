from __future__ import annotations

from types import SimpleNamespace

from oag.harness import Harness, HarnessConfig
from oag.loop.confirmation_flow import ConfirmationFlow
from oag.loop.query_loop import QueryLoop
from oag.loop.tool_executor import ToolExecutor
from oag.ontology.registry import FunctionRegistry
from oag.runtime import PendingConfirmation, RunState, ToolUseContext
from oag.ontology.schema import FunctionDef, FunctionParam, Ontology, ObjectTypeDef, PropertyDef
from oag.runtime.session_store import SessionStore
from oag.ontology.store import Store


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
                hint="Only create when user explicitly asks.",
                function_type="business",
                writes_to=["WorkOrder"],
                involves_objects=["Asset", "WorkOrder"],
                params={"asset_id": FunctionParam(type="str", description="Asset id")},
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


def test_prompt_uses_full_context():
    harness = make_harness()

    prompt = harness.build_system_prompt()

    assert "## 函数完整定义" in prompt
    assert "### 函数: lookup_asset" in prompt
    assert "## 对象完整定义" in prompt
    assert "### 对象: Asset" in prompt
    assert "首次调用函数时系统会自动注入" not in prompt


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
