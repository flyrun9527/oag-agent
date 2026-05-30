from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


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
