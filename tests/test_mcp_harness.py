from __future__ import annotations

import json
from pathlib import Path

from oag.harness import Harness, HarnessConfig
from oag.tools.mcp_remote import _tool_to_provider_dict


class DummyClient:
    pass


class FakeToolProvider:
    def __init__(self):
        self.calls = []

    def list_tools(self):
        return [
            {
                "name": "query",
                "description": "Query data",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "object_type": {"type": "string"},
                    },
                    "required": ["object_type"],
                },
                "category": "query",
                "read_only": True,
                "requires_confirmation": False,
                "policy": {
                    "read_only": True,
                    "requires_confirmation": False,
                    "concurrency_safe": True,
                    "worker_allowed": True,
                    "idempotent": True,
                    "destructive": False,
                },
            },
            {
                "name": "mutate",
                "description": "Mutate data",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string"},
                        "object_type": {"type": "string"},
                    },
                    "required": ["operation", "object_type"],
                },
                "category": "action",
                "read_only": False,
                "requires_confirmation": True,
                "policy": {
                    "read_only": False,
                    "requires_confirmation": True,
                    "concurrency_safe": False,
                    "worker_allowed": False,
                    "idempotent": False,
                    "destructive": True,
                },
            },
        ]

    def call_tool(self, name, arguments=None):
        self.calls.append((name, arguments or {}))
        return json.dumps({"tool": name, "args": arguments or {}}, ensure_ascii=False)


def test_agent_harness_consumes_tool_provider_without_core_runtime():
    provider = FakeToolProvider()
    harness = Harness(
        provider,
        DummyClient(),
        "dummy-model",
        HarnessConfig(enable_write_confirmation=False),
        domain_name="FakeDomain",
        domain_description="Fake tool domain",
    )

    tools = harness.build_tools()
    tool_names = {tool["function"]["name"] for tool in tools}

    assert {"query", "mutate", "ask_user", "dispatch_workers", "summarize_progress"} <= tool_names
    assert "FakeDomain" in harness.build_system_prompt()

    result = harness.execute_tool("query", {"object_type": "Asset"})

    assert json.loads(result.content) == {"tool": "query", "args": {"object_type": "Asset"}}
    assert provider.calls == [("query", {"object_type": "Asset"})]


def test_mutating_mcp_tool_pauses_for_confirmation_from_policy():
    harness = Harness(
        FakeToolProvider(),
        DummyClient(),
        "dummy-model",
        HarnessConfig(enable_write_confirmation=True),
    )

    result = harness.execute_tool("mutate", {
        "operation": "create",
        "object_type": "Asset",
    })

    assert result.needs_confirmation
    assert result.blocked


def test_agent_package_does_not_import_or_depend_on_oag_ontology():
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")

    assert "oag-ontology" not in pyproject
    for source in (root / "oag").rglob("*.py"):
        text = source.read_text(encoding="utf-8")
        assert "oag_ontology" not in text, source


def test_remote_mcp_tool_metadata_maps_to_tool_provider_shape():
    from mcp.types import Tool, ToolAnnotations

    tool = Tool(
        name="query",
        description="Query objects",
        inputSchema={"type": "object", "properties": {"object_type": {"type": "string"}}},
        annotations=ToolAnnotations(readOnlyHint=True),
        _meta={
            "oag": {
                "category": "query",
                "requires_confirmation": False,
                "policy": {"read_only": True},
            }
        },
    )

    mapped = _tool_to_provider_dict(tool)

    assert mapped["name"] == "query"
    assert mapped["input_schema"]["properties"]["object_type"]["type"] == "string"
    assert mapped["category"] == "query"
    assert mapped["read_only"] is True
    assert mapped["policy"] == {"read_only": True}
