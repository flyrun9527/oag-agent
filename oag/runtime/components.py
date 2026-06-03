"""Harness component assembly for the agent package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from openai import OpenAI

from ..llm.context import ContextManager
from ..tools.provider import ToolProvider, register_provider_tools
from ..tools.pipeline import ToolExecutionPipeline, ToolResult
from ..tools.registry import ToolRegistry
from ..tools.runtime_tools import RuntimeTools
from .config import HarnessConfig
from .hooks import AuditLog, HookRegistry, audit_log_hook, business_review_hook, write_confirmation_hook
from .stop_check import default_stop_hook
from .trace import TraceRecorder


GetMessages = Callable[[], list[dict] | None]
SetMessages = Callable[[list[dict] | None], None]
DispatchWorkers = Callable[[list[str], str], list[dict]]


@dataclass
class HarnessComponents:
    hooks: HookRegistry
    audit: AuditLog
    context_mgr: ContextManager
    tools: ToolRegistry
    cache: dict[str, ToolResult]
    trace: TraceRecorder
    tool_pipeline: ToolExecutionPipeline
    runtime_tools: RuntimeTools


def build_harness_components(
    tool_provider: ToolProvider,
    llm_client: OpenAI,
    model: str,
    config: HarnessConfig,
    *,
    set_current_messages: SetMessages,
    get_current_messages: GetMessages,
    dispatch_workers: DispatchWorkers,
) -> HarnessComponents:
    hooks = HookRegistry()
    audit = AuditLog()
    context_mgr = ContextManager(llm_client, model)
    tools = ToolRegistry()
    cache: dict[str, ToolResult] = {}
    trace = TraceRecorder(jsonl_path=config.trace_jsonl_path)

    register_provider_tools(tools, tool_provider)

    tool_pipeline = ToolExecutionPipeline(
        tools=tools,
        ontology_runtime=None,
        hooks=hooks,
        audit=audit,
        cache=cache,
        trace=trace,
        set_current_messages=set_current_messages,
    )
    runtime_tools = RuntimeTools(
        context_mgr=context_mgr,
        get_current_messages=get_current_messages,
        dispatch_workers=dispatch_workers,
    )

    runtime_tools.register(tools)
    register_default_hooks(hooks, config)

    return HarnessComponents(
        hooks=hooks,
        audit=audit,
        context_mgr=context_mgr,
        tools=tools,
        cache=cache,
        trace=trace,
        tool_pipeline=tool_pipeline,
        runtime_tools=runtime_tools,
    )


def register_default_hooks(hooks: HookRegistry, config: HarnessConfig):
    if config.enable_write_confirmation:
        hooks.register("pre_tool_call", write_confirmation_hook)
    if config.enable_audit:
        hooks.register("post_tool_call", audit_log_hook)
    hooks.register("post_tool_call", business_review_hook)
    hooks.register("query_complete", default_stop_hook)
