"""Tool provider adapter for agent runtime."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Protocol

from .registry import ToolDef, ToolPolicy, ToolRegistry


class ToolProvider(Protocol):
    def list_tools(self) -> list[dict[str, Any]]: ...
    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str: ...


def register_provider_tools(tools: ToolRegistry, provider: ToolProvider):
    for tool_info in provider.list_tools():
        name = tool_info["name"]
        tools.register(ToolDef(
            name=name,
            description=tool_info.get("description", ""),
            parameters=tool_info.get("input_schema") or tool_info.get("parameters") or {},
            handler=lambda args, _name=name: provider.call_tool(_name, args),
            category=tool_info.get("category", "query"),
            is_read_only=bool(tool_info.get("read_only", True)),
            requires_confirmation=bool(tool_info.get("requires_confirmation", False)),
            policy=_policy_from_tool_info(tool_info),
        ))


def _policy_from_tool_info(tool_info: dict[str, Any]) -> ToolPolicy:
    raw_policy = dict(tool_info.get("policy") or {})
    if not raw_policy:
        raw_policy = {
            "read_only": bool(tool_info.get("read_only", True)),
            "requires_confirmation": bool(tool_info.get("requires_confirmation", False)),
        }

    allowed = {field.name for field in fields(ToolPolicy)}
    data = {key: value for key, value in raw_policy.items() if key in allowed}
    return ToolPolicy(**data)
