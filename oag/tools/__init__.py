"""工具子系统导出。

tools 包包含工具元数据/策略定义、统一执行管线，以及 ask_user、
summarize_progress、dispatch_workers 等运行时内置工具。
"""

__all__ = [
    "RuntimeTools",
    "RemoteMcpToolProvider",
    "ToolDef",
    "ToolExecutionPipeline",
    "ToolProvider",
    "ToolPolicy",
    "ToolRegistry",
    "ToolResult",
]


def __getattr__(name: str):
    if name in {"ToolDef", "ToolPolicy", "ToolRegistry"}:
        from .registry import ToolDef, ToolPolicy, ToolRegistry

        return {
            "ToolDef": ToolDef,
            "ToolPolicy": ToolPolicy,
            "ToolRegistry": ToolRegistry,
        }[name]
    if name in {"ToolExecutionPipeline", "ToolResult"}:
        from .pipeline import ToolExecutionPipeline, ToolResult

        return {
            "ToolExecutionPipeline": ToolExecutionPipeline,
            "ToolResult": ToolResult,
        }[name]
    if name == "RuntimeTools":
        from .runtime_tools import RuntimeTools

        return RuntimeTools
    if name == "ToolProvider":
        from .provider import ToolProvider

        return ToolProvider
    if name == "RemoteMcpToolProvider":
        from .mcp_remote import RemoteMcpToolProvider

        return RemoteMcpToolProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
