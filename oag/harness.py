from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from openai import OpenAI

from .context import ContextManager
from .data_executor import DataExecutor
from .hooks import AuditLog, HookRegistry, HookResult, audit_log_hook, business_review_hook, write_confirmation_hook
from .ontology_runtime import OntologyRuntime
from .runtime import ToolUseContext, TraceRecorder
from .tool_registry import ToolDef, ToolPolicy, ToolRegistry
from .tool_pipeline import ToolExecutionPipeline, ToolResult
from .worker import run_workers_parallel
from .registry import FunctionRegistry
from .rules import RuleEngine
from .schema import Ontology
from .store import Store

logger = logging.getLogger(__name__)


@dataclass
class HarnessConfig:
    max_turns: int = 10
    max_tool_result_chars: int = 5000
    enable_audit: bool = True
    enable_write_confirmation: bool = True
    enable_progressive_context: bool = False


def _default_stop_hook(context: dict) -> HookResult:
    messages = context.get("messages", [])

    last_assistant: str | None = None
    tool_errors = []
    for m in reversed(messages):
        if m.get("role") == "assistant":
            last_assistant = m.get("content") or ""
            break
        if m.get("role") == "tool":
            content = m.get("content", "")
            if '"error"' in content or "不存在" in content:
                tool_errors.append(content[:100])

    incomplete_signals = ["正在进行", "下一步", "接下来", "即将", "稍后", "继续调用", "我将调用", "我将"]

    issues = []
    if last_assistant is None or not last_assistant.strip():
        issues.append("未生成最终回答（可能工具调用轮次用尽）")
    elif len(last_assistant) < 20:
        issues.append("回复过短，可能未完整回答")
    elif any(sig in last_assistant for sig in incomplete_signals):
        issues.append("回复暗示任务未完成（含'正在进行/下一步/我将调用'等表述），请继续执行或给出最终结论")
    if tool_errors:
        issues.append(f"有工具执行出错未处理: {'; '.join(tool_errors[:2])}")

    if issues:
        return HookResult(action="pause", reason="; ".join(issues))
    return HookResult()


class Harness:
    def __init__(self, ontology: Ontology, store: Store,
                 registry: FunctionRegistry, llm_client: OpenAI,
                 model: str, config: HarnessConfig | None = None):
        self.ontology = ontology
        self.config = config or HarnessConfig()
        self.hooks = HookRegistry()
        self.audit = AuditLog()
        self.rule_engine = RuleEngine(ontology, store) if ontology.rules else None
        self.context_mgr = ContextManager(llm_client, model)

        self.ont = OntologyRuntime(ontology, store, registry, self.rule_engine)
        self.data = DataExecutor(store, registry)
        self.tools = ToolRegistry()
        self._cache: dict[str, ToolResult] = {}
        self.trace = TraceRecorder()
        self.tool_pipeline = ToolExecutionPipeline(
            tools=self.tools,
            ontology_runtime=self.ont,
            hooks=self.hooks,
            audit=self.audit,
            cache=self._cache,
            trace=self.trace,
            progressive_context_enabled=lambda: self.config.enable_progressive_context,
            set_current_messages=self._set_current_messages,
        )

        self.ont.register_tools(self.tools, self.data)
        self._register_runtime_tools()

        if self.config.enable_write_confirmation:
            self.hooks.register("pre_tool_call", write_confirmation_hook)
        if self.config.enable_audit:
            self.hooks.register("post_tool_call", audit_log_hook)
        self.hooks.register("post_tool_call", business_review_hook)
        self.hooks.register("query_complete", _default_stop_hook)

    def _register_runtime_tools(self):
        self.tools.register(ToolDef(
            name="summarize_progress",
            description="总结当前对话进展。返回已完成的操作摘要、使用的工具统计。适合长对话中回顾进度",
            parameters={"type": "object", "properties": {}},
            handler=lambda args: self._summarize_progress_handler(args),
            category="query",
        ))

        self.tools.register(ToolDef(
            name="ask_user",
            description="向用户提问以收集决策。当存在多种可行方案、需要确认优先级或参数时使用。用户回答后会作为工具结果返回",
            parameters={"type": "object", "properties": {
                "question": {"type": "string", "description": "要问用户的问题"},
                "options": {"type": "array", "items": {"type": "object", "properties": {"label": {"type": "string", "description": "选项标签"}, "description": {"type": "string", "description": "选项说明"}}, "required": ["label"]}, "description": "可选项列表（2-5个）"},
                "multi_select": {"type": "boolean", "description": "是否允许多选（默认单选）"},
            }, "required": ["question", "options"]},
            handler=lambda args: json.dumps({"question": args.get("question", ""), "options": args.get("options", [])}, ensure_ascii=False),
            category="ask", requires_confirmation=True,
            policy=ToolPolicy(
                read_only=True,
                requires_confirmation=True,
                concurrency_safe=False,
                worker_allowed=False,
                idempotent=False,
            ),
        ))

        self.tools.register(ToolDef(
            name="dispatch_workers",
            description="并行派遣多个 Worker 执行独立子任务。每个 Worker 是独立的智能体，有自己的工具和上下文。Worker 只能看到 context 中提供的信息。",
            parameters={"type": "object", "properties": {
                "tasks": {"type": "array", "items": {"type": "string"}, "description": "子任务描述列表。每条须包含完整信息（事件ID、设施ID等），Worker 看不到你的对话历史"},
                "context": {"type": "string", "description": "传递给所有 Worker 的背景信息，如事件详情、已查到的设施列表等"},
            }, "required": ["tasks"]},
            handler=lambda args: self._dispatch_workers_handler(args),
            category="action",
            policy=ToolPolicy(
                read_only=False,
                requires_confirmation=False,
                concurrency_safe=False,
                worker_allowed=False,
                idempotent=False,
            ),
        ))

    def register_stop_hook(self, handler):
        self.hooks.register("query_complete", handler)

    def execute_tool(self, tool_name: str, args: dict,
                     session_id: str = "",
                     confirmed: bool = False,
                     messages: list[dict] | None = None,
                     context: ToolUseContext | None = None) -> ToolResult:
        context = self._normalize_tool_context(session_id, confirmed, messages, context)
        return self.tool_pipeline.execute(tool_name, args, context)

    def _normalize_tool_context(self, session_id: str, confirmed: bool,
                                messages: list[dict] | None,
                                context: ToolUseContext | None) -> ToolUseContext:
        if context:
            return context
        return ToolUseContext(
            session_id=session_id,
            messages=messages,
            confirmed=confirmed,
        )

    def _set_current_messages(self, messages: list[dict] | None):
        self._current_messages = messages

    def _dispatch_workers_handler(self, args: dict) -> str:
        tasks = args.get("tasks", [])
        if not tasks:
            return json.dumps({"error": "tasks 列表不能为空"}, ensure_ascii=False)

        context = args.get("context", "")
        results = run_workers_parallel(
            self, self.context_mgr.client, self.context_mgr.model,
            tasks, context=context,
            max_workers=min(len(tasks), 4),
        )

        summary = []
        for r in results:
            status_icon = "✓" if r["status"] == "success" else "✗"
            tools_used = ", ".join(tc["name"] for tc in r.get("tool_calls", []))
            summary.append({
                "worker": r["worker_id"],
                "task": r["task"],
                "status": status_icon,
                "tools_used": tools_used,
                "result": r["result"][:500],
            })
        return json.dumps(summary, ensure_ascii=False, default=str)

    def _summarize_progress_handler(self, args: dict) -> str:
        messages = getattr(self, "_current_messages", None)
        if not messages or len(messages) < 2:
            return json.dumps({"error": "对话历史过短，无需总结"}, ensure_ascii=False)

        tool_names_used: list[str] = []
        for m in messages:
            for tc in m.get("tool_calls", []):
                if isinstance(tc, dict):
                    tool_names_used.append(tc["function"]["name"])

        summary_text = self.context_mgr._summarize(messages[1:])
        return json.dumps({
            "summary": summary_text,
            "total_messages": len(messages),
            "tool_calls_count": len(tool_names_used),
            "tools_used": sorted(set(tool_names_used)),
        }, ensure_ascii=False)

    def build_tools(self) -> list[dict]:
        return self.tools.build_tools()

    def build_system_prompt(self, domain_context: str = "") -> str:
        prompt = self.ont.build_system_prompt(
            domain_context,
            progressive_context=self.config.enable_progressive_context,
        )
        if not self.config.enable_progressive_context:
            prompt += "\n\n" + self.ont.build_full_context()
        return prompt

    def maybe_compact(self, messages: list[dict]) -> tuple[list[dict], bool]:
        messages, compacted = self.context_mgr.maybe_compact(messages)
        if compacted and self.config.enable_progressive_context:
            self.ont.reset_context_shown()
        return messages, compacted

    def run_stop_check(self, user_question: str, messages: list[dict]) -> str | None:
        result = self.hooks.fire("query_complete", {
            "messages": messages,
            "user_question": user_question,
        })
        if result.action == "pause" and result.reason:
            return (
                f"[系统自检] 请检查你的回复是否完整回答了用户问题: \"{user_question}\"\n"
                f"发现的问题: {result.reason}\n"
                f"如果回复已完整，直接说'已确认回复完整'即可。否则请补充。"
            )
        return None
