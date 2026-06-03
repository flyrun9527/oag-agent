"""Agent-side tool metadata.

Agent only needs a transport-neutral view of available tools: schema, handler,
and execution policy. Ontology-specific tool construction lives outside this
package and is consumed through a tool provider.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass
class ToolPolicy:
    read_only: bool = True
    requires_confirmation: bool = False
    concurrency_safe: bool = True
    worker_allowed: bool = True
    idempotent: bool = True
    destructive: bool = False
    timeout_seconds: float | None = 30.0


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict
    handler: Callable[[dict], str]
    usage_prompt: str = ""
    category: str = "query"
    is_read_only: bool = True
    requires_confirmation: bool = False
    max_result_chars: int = 5000
    policy: ToolPolicy | None = None

    def __post_init__(self):
        if self.policy is None:
            self.policy = ToolPolicy(
                read_only=self.is_read_only,
                requires_confirmation=self.requires_confirmation,
                concurrency_safe=self.is_read_only,
                worker_allowed=self.is_read_only,
                idempotent=self.is_read_only,
                destructive=not self.is_read_only,
            )
        else:
            self.is_read_only = self.policy.read_only
            self.requires_confirmation = self.policy.requires_confirmation

    def to_provider_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
            "category": self.category,
            "read_only": self.is_read_only,
            "requires_confirmation": self.requires_confirmation,
            "policy": asdict(self.policy) if self.policy else {},
        }


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolDef] = {}
        self._version = 0
        self._built_tools_cache: list[dict] | None = None
        self._built_tools_version = -1

    def register(self, tool: ToolDef):
        self._tools[tool.name] = tool
        self._version += 1
        self._built_tools_cache = None

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def values(self):
        return self._tools.values()

    @property
    def version(self) -> int:
        return self._version

    def build_tools(self) -> list[dict]:
        if (
            self._built_tools_cache is not None
            and self._built_tools_version == self._version
        ):
            return self._built_tools_cache

        self._built_tools_cache = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": self._build_description(tool),
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]
        self._built_tools_version = self._version
        return self._built_tools_cache

    def _build_description(self, tool: ToolDef) -> str:
        description = tool.description.strip()
        usage_prompt = tool.usage_prompt.strip()
        if not usage_prompt:
            return description
        if not description:
            return usage_prompt
        return f"{description}\n\n使用说明:\n{usage_prompt}"
