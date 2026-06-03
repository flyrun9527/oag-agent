from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from oag.agent import Agent
from oag.harness import Harness, HarnessConfig
from oag.llm.context import ContextManager
from oag.loop.confirmation_flow import ConfirmationFlow
from oag.loop.query_loop import QueryLoop
from oag.loop.tool_executor import ToolExecutor
from oag.runtime import PendingConfirmation, RunState
from oag.runtime.message_sanitizer import sanitize_messages
from oag.runtime.session_store import SessionStore
from oag.tools.registry import ToolDef, ToolPolicy


class DummyClient:
    pass


class FakeToolProvider:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def list_tools(self):
        return [
            _tool(
                "lookup_asset",
                {"asset_id": {"type": "string"}},
                required=["asset_id"],
                read_only=True,
            ),
            _tool(
                "create_work_order",
                {"asset_id": {"type": "string"}},
                required=["asset_id"],
                read_only=False,
                requires_confirmation=True,
            ),
            _tool(
                "mutate",
                {
                    "operation": {"type": "string", "enum": ["create", "update", "delete"]},
                    "object_type": {"type": "string", "enum": ["Asset"]},
                },
                required=["operation", "object_type"],
                read_only=False,
                requires_confirmation=True,
            ),
        ]

    def call_tool(self, name, arguments=None):
        args = arguments or {}
        self.calls.append((name, args))
        if name == "lookup_asset":
            return json.dumps({"asset_id": args.get("asset_id"), "status": "ok"})
        if name == "create_work_order":
            return json.dumps({"order_id": "WO1", "asset_id": args.get("asset_id")})
        return json.dumps({"tool": name, "args": args})


def _tool(name: str, properties: dict, *, required: list[str],
          read_only: bool, requires_confirmation: bool = False) -> dict:
    return {
        "name": name,
        "description": f"{name} description",
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
        "category": "query" if read_only else "action",
        "read_only": read_only,
        "requires_confirmation": requires_confirmation,
        "policy": {
            "read_only": read_only,
            "requires_confirmation": requires_confirmation,
            "concurrency_safe": read_only,
            "worker_allowed": read_only,
            "idempotent": read_only,
            "destructive": not read_only,
        },
    }


def make_harness(config: HarnessConfig | None = None) -> Harness:
    return Harness(
        FakeToolProvider(),
        DummyClient(),
        "dummy-model",
        config or HarnessConfig(enable_write_confirmation=False),
        domain_name="TestDomain",
        domain_description="Test tool domain",
    )


def make_tool_call(name: str, tool_id: str = "tool_1") -> SimpleNamespace:
    return SimpleNamespace(id=tool_id, function=SimpleNamespace(name=name))


def make_full_tool_call(name: str, tool_id: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=tool_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def make_response(tool_calls=None, content: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
            ),
        ],
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


def test_prompt_and_tools_are_provider_based():
    harness = make_harness()

    prompt = harness.build_system_prompt()
    names = {tool["function"]["name"] for tool in harness.build_tools()}

    assert "TestDomain" in prompt
    assert "MCP tools only" in prompt
    assert {"lookup_asset", "create_work_order", "mutate", "ask_user"} <= names


def test_tool_schema_validation_and_execution():
    harness = make_harness()

    missing = harness.execute_tool("lookup_asset", {})
    ok = harness.execute_tool("lookup_asset", {"asset_id": "A1"})

    assert missing.blocked
    assert "缺少必填字段: asset_id" in missing.content
    assert json.loads(ok.content) == {"asset_id": "A1", "status": "ok"}


def test_confirmation_policy_from_tool_metadata():
    harness = make_harness(HarnessConfig(enable_write_confirmation=True))

    result = harness.execute_tool("create_work_order", {"asset_id": "A1"})

    assert result.blocked
    assert result.needs_confirmation


def test_tool_executor_stops_before_later_calls_when_confirmation_needed():
    harness = make_harness(HarnessConfig(enable_write_confirmation=True))
    executor = ToolExecutor(harness)
    state = RunState(messages=[], session_id="s1", user_question="")

    results = executor.execute_tool_calls([
        (make_tool_call("create_work_order", "t1"), {"asset_id": "A1"}),
        (make_tool_call("lookup_asset", "t2"), {"asset_id": "A1"}),
    ], state)

    assert [tc.id for tc, _, _ in results] == ["t1"]
    assert results[0][2].needs_confirmation


def test_query_loop_records_final_response_transition(monkeypatch):
    harness = make_harness()

    def fake_call_llm_with_retry(*args, **kwargs):
        return make_response(content="This is a complete final answer for the user.")

    monkeypatch.setattr("oag.loop.query_loop.call_llm_with_retry", fake_call_llm_with_retry)
    loop = QueryLoop(harness, DummyClient(), "dummy-model", on_pending_confirmation=lambda *args: None)
    state = RunState(
        messages=[{"role": "system", "content": "System"}, {"role": "user", "content": "Question?"}],
        session_id="s1",
        user_question="Question?",
    )

    events = list(loop.run(state))
    trace_events = harness.trace.snapshot()

    assert events[-1].type == "text"
    assert events[-1].content == "This is a complete final answer for the user."
    assert trace_events[-1].payload["reason"] == "final_response"


def test_query_loop_aggregates_streaming_tool_calls(monkeypatch):
    harness = make_harness()

    def fake_call_llm_with_retry(*args, **kwargs):
        return iter([
            make_stream_chunk(tool_call=make_tool_delta(0, tool_id="tool_1", name="lookup_asset", arguments='{"asset')),
            make_stream_chunk(tool_call=make_tool_delta(0, arguments='_id":"A1"}')),
        ])

    monkeypatch.setattr("oag.loop.query_loop.call_llm_with_retry", fake_call_llm_with_retry)
    loop = QueryLoop(harness, DummyClient(), "dummy-model", on_pending_confirmation=lambda *args: None)
    messages = [{"role": "system", "content": "System"}, {"role": "user", "content": "Lookup"}]
    state = RunState(messages=messages, session_id="s1", user_question="Lookup")

    events = list(loop.run(state))

    assert any(event.type == "tool_call" and event.name == "lookup_asset" for event in events)
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
    loop = QueryLoop(harness, DummyClient(), "dummy-model", on_pending_confirmation=lambda *args: pending.append(args))
    messages = [{"role": "system", "content": "System"}, {"role": "user", "content": "Create"}]

    events = list(loop.run(RunState(messages=messages, session_id="s1", user_question="Create")))

    assert [event.type for event in events] == ["debug", "debug", "confirmation_required"]
    assert executed == ["create_work_order"]
    assert len(pending) == 1
    assert pending[0][6] == [{"tool_call_id": "tool_2", "content": '{"skipped": true, "reason": "前一个工具调用需要用户确认，本调用未执行"}'}]


def test_confirmation_flow_appends_skipped_tool_results_in_order():
    harness = make_harness()
    continued_states = []
    flow = ConfirmationFlow(
        harness,
        save_messages=lambda session_id, messages: None,
        run_loop=lambda state: continued_states.append(state) or iter(()),
    )
    messages = [{"role": "system", "content": "System"}]
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
    assert continued_states[0].turn_count == 3


def test_confirmation_flow_denial_saves_rejection_messages():
    harness = make_harness()
    saved = []
    flow = ConfirmationFlow(
        harness,
        save_messages=lambda session_id, messages: saved.append((session_id, messages)),
        run_loop=lambda state: iter(()),
    )
    messages = [{"role": "system", "content": "System"}]
    pending = PendingConfirmation(
        session_id="s1",
        tool_name="mutate",
        args={"operation": "create"},
        tool_call_id="tool_1",
        messages=messages,
    )

    events = list(flow.confirm(pending, approved=False))

    assert events[0].content == "已取消 mutate 的执行。"
    assert saved == [("s1", messages)]
    assert "用户拒绝执行" in messages[-2]["content"]


def test_agent_sets_default_trace_jsonl_path(tmp_path):
    harness = make_harness()

    Agent(harness, DummyClient(), "dummy-model", db_dir=str(tmp_path))

    assert harness.trace.jsonl_path == str(tmp_path / "trace_TestDomain.jsonl")


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


def test_context_compaction_preserves_tool_call_pairs(monkeypatch):
    mgr = ContextManager(DummyClient(), "dummy-model", context_window=20)
    monkeypatch.setattr(mgr, "_summarize", lambda messages: "summary")
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "old " * 100},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "older follow-up"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "tool_1", "type": "function", "function": {"name": "lookup_asset", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "tool_1", "content": "result"},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "next answer"},
        {"role": "user", "content": "more"},
        {"role": "assistant", "content": "more answer"},
        {"role": "user", "content": "final"},
    ]

    compacted, did_compact = mgr.maybe_compact(messages)

    assert did_compact
    tool_idx = next(i for i, message in enumerate(compacted) if message.get("role") == "tool")
    assert compacted[tool_idx - 1]["tool_calls"][0]["id"] == "tool_1"


def test_stop_check_blocks_success_claim_after_unhandled_tool_error():
    harness = make_harness()
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Create work order"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "tool_1", "type": "function", "function": {"name": "create_work_order", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "tool_1", "content": '{"error": "backend unavailable"}'},
        {"role": "assistant", "content": "处理完成，推荐方案如下。"},
    ]

    result = harness.run_stop_check("Create work order", messages)

    assert result is not None
    assert "有工具执行出错未处理" in result
