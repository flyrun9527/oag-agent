from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolPolicy:
    read_only: bool = True
    requires_confirmation: bool = False
    concurrency_safe: bool = True
    worker_allowed: bool = True
    idempotent: bool = True
    destructive: bool = False


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict
    handler: Callable[[dict], str]
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


class ToolRegistry:

    def __init__(self):
        self._tools: dict[str, ToolDef] = {}

    def register(self, tool: ToolDef):
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def build_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]
