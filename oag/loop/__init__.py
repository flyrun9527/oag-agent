__all__ = [
    "ConfirmationFlow",
    "QueryLoop",
    "ToolExecutor",
]


def __getattr__(name: str):
    if name == "ConfirmationFlow":
        from .confirmation_flow import ConfirmationFlow

        return ConfirmationFlow
    if name == "QueryLoop":
        from .query_loop import QueryLoop

        return QueryLoop
    if name == "ToolExecutor":
        from .tool_executor import ToolExecutor

        return ToolExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
