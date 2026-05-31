"""主 LLM 回合循环。

QueryLoop 负责把消息和工具发给模型、记录调试事件、执行模型请求的工具、
处理确认暂停，并在最终回答前运行 stop check。它不直接实现工具策略。
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Callable, Generator

from openai import OpenAI

from ..runtime.events import (
    CompactEvent, ConfirmationEvent, DebugEvent, Event, QuestionEvent,
    ReasoningEvent, TextEvent, ToolCallEvent,
)
from ..llm.retry import call_llm_with_retry
from ..runtime import RunState
from .tool_executor import ToolExecutor

if TYPE_CHECKING:
    from ..harness import Harness


PendingConfirmationHandler = Callable[[str, str, dict, str, list[dict]], None]
MAX_REASONING_CHARS = 5000


class QueryLoop:
    def __init__(self, harness: Harness, llm_client: OpenAI, model: str,
                 on_pending_confirmation: PendingConfirmationHandler):
        self.harness = harness
        self.client = llm_client
        self.model = model
        self.tool_executor = ToolExecutor(harness)
        self.on_pending_confirmation = on_pending_confirmation

    def run(self, state: RunState) -> Generator[Event, None, None]:
        tools = self.harness.build_tools()

        while True:
            # 一次循环对应一个模型回合，以及该回合触发的工具执行结果。
            state.turn_count += 1
            messages = state.messages
            self.harness.trace.record(
                "agent_turn_start",
                session_id=state.session_id,
                turn_count=state.turn_count,
                message_count=len(messages),
                stop_hook_active=state.stop_hook_active,
            )
            if state.turn_count > self.harness.config.max_turns:
                self.harness.trace.record(
                    "agent_transition",
                    session_id=state.session_id,
                    turn_count=state.turn_count,
                    reason="max_turns_reached",
                )
                yield DebugEvent(stage="info", content=f"max_turns ({self.harness.config.max_turns}) reached")
                break

            if state.turn_count > 1 and state.turn_count % 5 == 0:
                messages, compacted = self.harness.maybe_compact(messages)
                state.messages = messages
                if compacted:
                    yield CompactEvent()

            yield self._build_request_debug_event(state)

            response = call_llm_with_retry(
                self.client,
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                temperature=0.1,
                stream=True,
            )
            msg = yield from self._consume_llm_response(response)

            if not msg.tool_calls:
                yield from self._handle_final_response(
                    state,
                    msg.content or "",
                    already_streamed=getattr(msg, "content_streamed", False),
                )
                return

            # OpenAI tool protocol 要求先保存 assistant 的 tool_calls envelope，
            # 再追加每个 tool_call_id 对应的 tool 消息。
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            tool_calls_parsed = [
                (tc, json.loads(tc.function.arguments)) for tc in msg.tool_calls
            ]
            results_ordered = self.tool_executor.execute_tool_calls(tool_calls_parsed, state)

            should_stop = False
            for tc, args, result in results_ordered:
                if result.needs_confirmation:
                    # 暂停当前循环并保存现场；用户响应后由 ConfirmationFlow 继续。
                    self.harness.trace.record(
                        "agent_transition",
                        session_id=state.session_id,
                        turn_count=state.turn_count,
                        reason="confirmation_required",
                        tool_name=tc.function.name,
                    )
                    self.on_pending_confirmation(
                        state.session_id,
                        tc.function.name,
                        args,
                        tc.id,
                        messages,
                    )
                    if tc.function.name == "ask_user":
                        yield QuestionEvent(
                            question=args.get("question", ""),
                            options=args.get("options", []),
                            multi_select=args.get("multi_select", False),
                        )
                    else:
                        yield ConfirmationEvent(
                            tool_name=tc.function.name,
                            args=args,
                            reason=result.block_reason,
                        )
                    should_stop = True
                    break

                preview_len = 5000 if tc.function.name == "dispatch_workers" else 200
                yield ToolCallEvent(
                    name=tc.function.name,
                    args=args,
                    result=result.content[:preview_len],
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result.content,
                })

                if result.blocked:
                    messages.append({
                        "role": "user",
                        "content": f"[系统提示] 工具 {tc.function.name} 被阻止: {result.block_reason}",
                    })

            if should_stop:
                return

            state.stop_hook_active = False
            state.transition_reason = "next_turn"
            self.harness.trace.record(
                "agent_transition",
                session_id=state.session_id,
                turn_count=state.turn_count,
                reason=state.transition_reason,
            )

    def _handle_final_response(self, state: RunState,
                               content: str,
                               already_streamed: bool = False) -> Generator[Event, None, None]:
        messages = state.messages
        messages.append({"role": "assistant", "content": content})
        if not already_streamed:
            yield TextEvent(content=content)

        stop_result = self.harness.run_stop_check(state.user_question, messages)
        if stop_result and not state.stop_hook_active:
            state.stop_hook_active = True
            state.transition_reason = "stop_hook_blocking"
            self.harness.trace.record(
                "agent_transition",
                session_id=state.session_id,
                turn_count=state.turn_count,
                reason=state.transition_reason,
            )
            messages.append({"role": "user", "content": stop_result})
            yield from self.run(state)
            return

        self.harness.trace.record(
            "agent_transition",
            session_id=state.session_id,
            turn_count=state.turn_count,
            reason="final_response",
        )

    def _build_request_debug_event(self, state: RunState) -> DebugEvent:
        debug_msgs = []
        for m in state.messages[-6:]:
            role = m.get("role", "")
            if role == "system":
                debug_msgs.append(f"[SYS] {(m.get('content', ''))[:200]}")
            elif role == "user":
                debug_msgs.append(f"[USR] {(m.get('content', ''))[:200]}")
            elif role == "assistant":
                tc_names = [tc["function"]["name"] for tc in m.get("tool_calls", []) if isinstance(tc, dict)]
                if tc_names:
                    debug_msgs.append(f"[LLM] 调用->{', '.join(tc_names)} {(m.get('content', ''))[:100]}")
                else:
                    debug_msgs.append(f"[LLM] {(m.get('content', ''))[:200]}")
            elif role == "tool":
                debug_msgs.append(f"[TOOL] {(m.get('content', ''))[:200]}")
        return DebugEvent(
            stage="request",
            content=f"Turn {state.turn_count}, {len(state.messages)} msgs\n" + "\n".join(debug_msgs),
        )

    def _build_response_debug_event(self, msg) -> DebugEvent:
        resp_summary = ""
        if msg.tool_calls:
            tc_list = [f"{tc.function.name}({tc.function.arguments[:80]})" for tc in msg.tool_calls]
            resp_summary = "LLM选择调用: " + "; ".join(tc_list)
        if msg.content:
            resp_summary += f"\nLLM文本: {msg.content[:300]}"
        return DebugEvent(stage="response", content=resp_summary)

    def _consume_llm_response(self, response) -> Generator[Event, None, SimpleNamespace]:
        if hasattr(response, "choices") and response.choices:
            msg = response.choices[0].message
            msg.content_streamed = False
            yield self._build_response_debug_event(msg)
            if reasoning := self._extract_reasoning_from_message(msg):
                yield ReasoningEvent(content=reasoning)
            return msg

        content_parts: list[str] = []
        reasoning_chars = 0
        tool_call_parts: dict[int, dict] = {}

        for chunk in response:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            if reasoning_delta := self._extract_reasoning_from_message(delta):
                remaining = MAX_REASONING_CHARS - reasoning_chars
                if remaining > 0:
                    emitted = reasoning_delta[:remaining]
                    reasoning_chars += len(emitted)
                    yield ReasoningEvent(content=emitted)
                    if len(reasoning_delta) > remaining:
                        yield ReasoningEvent(content="\n[... reasoning 已截断]")
                        reasoning_chars = MAX_REASONING_CHARS

            if delta.content:
                content_parts.append(delta.content)
                yield TextEvent(content=delta.content)

            for tc_delta in delta.tool_calls or []:
                index = tc_delta.index
                entry = tool_call_parts.setdefault(
                    index,
                    {"id": "", "type": "function", "name": "", "arguments": []},
                )
                if tc_delta.id:
                    entry["id"] = tc_delta.id
                if tc_delta.type:
                    entry["type"] = tc_delta.type
                if tc_delta.function:
                    if tc_delta.function.name:
                        entry["name"] = tc_delta.function.name
                    if tc_delta.function.arguments:
                        entry["arguments"].append(tc_delta.function.arguments)

        msg = SimpleNamespace(
            content="".join(content_parts),
            content_streamed=bool(content_parts),
            tool_calls=[
                SimpleNamespace(
                    id=entry["id"],
                    type=entry["type"],
                    function=SimpleNamespace(
                        name=entry["name"],
                        arguments="".join(entry["arguments"]),
                    ),
                )
                for _, entry in sorted(tool_call_parts.items())
                if entry["name"]
            ] or None,
        )
        yield self._build_response_debug_event(msg)
        return msg

    def _extract_reasoning_from_message(self, msg) -> str:
        # llama-server exposes reasoning as an OpenAI-compatible extra field.
        reasoning = getattr(msg, "reasoning_content", None)
        if not reasoning:
            extra = getattr(msg, "model_extra", None) or {}
            reasoning = extra.get("reasoning_content")
        return str(reasoning) if reasoning else ""
