"""Remote MCP tool provider built on the official MCP client session."""

from __future__ import annotations

import logging
import anyio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal

logger = logging.getLogger(__name__)


class RemoteMcpToolProvider:
    """Agent-side adapter for remote MCP servers.

    The agent consumes the transport-neutral ToolProvider shape. MCP connection
    details stay here, and ontology-specific tool construction stays in the MCP
    server process.
    """

    def __init__(self, url: str, transport: Literal["streamable-http", "sse"] = "streamable-http"):
        self.url = url
        self.transport = transport

    def list_tools(self) -> list[dict[str, Any]]:
        return anyio.run(self._list_tools)

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        return anyio.run(self._call_tool, name, arguments or {})

    async def _list_tools(self) -> list[dict[str, Any]]:
        logger.info("MCP list_tools url=%s transport=%s", self.url, self.transport)
        async with self._session() as session:
            result = await session.list_tools()
            tools = [_tool_to_provider_dict(tool) for tool in result.tools]
            logger.info("MCP list_tools done url=%s tools=%d", self.url, len(tools))
            return tools

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        logger.info("MCP call_tool start url=%s tool=%s args=%s", self.url, name, _clip_for_log(arguments))
        async with self._session() as session:
            result = await session.call_tool(name, arguments)
            text = _content_to_text(result.content)
            logger.info("MCP call_tool done url=%s tool=%s result_chars=%d", self.url, name, len(text))
            return text

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[Any]:
        from mcp import ClientSession

        if self.transport == "streamable-http":
            from mcp.client.streamable_http import streamable_http_client

            async with streamable_http_client(self.url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
            return

        if self.transport == "sse":
            from mcp.client.sse import sse_client

            async with sse_client(self.url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
            return

        raise ValueError(f"Unsupported MCP transport: {self.transport}")




def _tool_to_provider_dict(tool: Any) -> dict[str, Any]:
    meta = getattr(tool, "meta", None) or {}
    oag_meta = meta.get("oag", {}) if isinstance(meta, dict) else {}
    annotations = getattr(tool, "annotations", None)
    read_only = getattr(annotations, "readOnlyHint", None) if annotations else None
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": tool.inputSchema,
        "category": oag_meta.get("category", "query"),
        "read_only": True if read_only is None else bool(read_only),
        "requires_confirmation": bool(oag_meta.get("requires_confirmation", False)),
        "policy": oag_meta.get("policy", {}),
    }


def _content_to_text(content: list[Any]) -> str:
    parts = []
    for item in content:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(text)
        else:
            parts.append(str(item))
    return "\n".join(parts)


def _clip_for_log(value: Any, limit: int = 600) -> str:
    text = str(value)
    return text if len(text) <= limit else f"{text[:limit]}..."
