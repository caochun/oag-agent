"""默认最终回答完整性检查 hook。"""

from __future__ import annotations

from .hooks import HookResult


def default_stop_hook(context: dict) -> HookResult:
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
