"""默认最终回答完整性检查 hook。"""

from __future__ import annotations

import json

from .hooks import HookResult


def default_stop_hook(context: dict) -> HookResult:
    messages = context.get("messages", [])

    last_assistant: str | None = None
    current_user_index = _find_current_user_index(messages, context.get("user_question", ""))
    pending_tool_errors = _collect_pending_tool_errors(messages, current_user_index)

    for m in reversed(messages):
        if m.get("role") == "assistant":
            last_assistant = m.get("content") or ""
            break

    incomplete_signals = ["正在进行", "下一步", "接下来", "即将", "稍后", "继续调用", "我将调用", "我将"]
    failure_ack_signals = [
        "未完成", "失败", "出错", "错误", "无法", "不能", "未能", "没有完成",
        "需要重试", "需重试", "需要继续", "尚未", "仍需", "被阻止",
    ]
    success_signals = [
        "完成", "成功", "已处理", "处理完成", "推荐方案", "最终方案", "可行",
        "通过", "解决",
    ]

    issues = []
    if last_assistant is None or not last_assistant.strip():
        issues.append("未生成最终回答（可能工具调用轮次用尽）")
    elif len(last_assistant) < 20:
        issues.append("回复过短，可能未完整回答")
    elif any(sig in last_assistant for sig in incomplete_signals):
        issues.append("回复暗示任务未完成（含'正在进行/下一步/我将调用'等表述），请继续执行或给出最终结论")
    if pending_tool_errors:
        acknowledged = last_assistant and any(sig in last_assistant for sig in failure_ack_signals)
        claims_success = _claims_success(last_assistant or "", success_signals)
        if claims_success or not acknowledged:
            issues.append(
                "有工具执行出错未处理，最终回答不能宣称已完成/成功: "
                + "; ".join(pending_tool_errors[:2])
            )

    if issues:
        return HookResult(action="pause", reason="; ".join(issues))
    return HookResult()


def _find_current_user_index(messages: list[dict], user_question: str) -> int:
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if msg.get("role") == "user" and msg.get("content") == user_question:
            return idx
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "user" and not str(messages[idx].get("content", "")).startswith("[系统"):
            return idx
    return 0


def _collect_pending_tool_errors(messages: list[dict], start_index: int) -> list[str]:
    tool_call_names = _tool_call_names(messages)
    pending: dict[str, str] = {}
    fallback_index = 0

    for msg in messages[start_index + 1:]:
        if msg.get("role") != "tool":
            continue
        tool_call_id = msg.get("tool_call_id", "")
        tool_name = tool_call_names.get(tool_call_id) or f"tool#{fallback_index}"
        fallback_index += 1
        error = _extract_tool_error(msg.get("content", ""))
        if error:
            pending[tool_name] = f"{tool_name}: {error}"
        else:
            pending.pop(tool_name, None)

    return list(pending.values())


def _tool_call_names(messages: list[dict]) -> dict[str, str]:
    names = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            call_id = call.get("id")
            function = call.get("function") or {}
            name = function.get("name")
            if call_id and name:
                names[call_id] = name
    return names


def _extract_tool_error(content: str) -> str:
    try:
        data = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return _text_error(content)
    return _json_error(data)


def _json_error(data) -> str:
    if isinstance(data, dict):
        if data.get("error"):
            return str(data["error"])[:160]
        if data.get("blocked") and data.get("reason"):
            return str(data["reason"])[:160]
        if data.get("paused") and data.get("reason"):
            return str(data["reason"])[:160]
    return ""


def _text_error(content: str) -> str:
    text = str(content or "")
    if '"error"' in text or "工具执行错误" in text or "不存在" in text:
        return text[:160]
    return ""


def _claims_success(content: str, success_signals: list[str]) -> bool:
    text = content
    for phrase in [
        "未完成", "没有完成", "无法完成", "不能完成", "未能完成",
        "未成功", "没有成功", "不成功", "未处理", "没有处理",
    ]:
        text = text.replace(phrase, "")
    return any(sig in text for sig in success_signals)
