"""精简 ReAct Agent。

无独立路由 LLM、无工具白名单、无前置条件硬校验、无确认状态机：
- 单个 ReAct LLM 完成意图理解 + 选工具；
- 工具顺序内聚在复合工具内部；
- 昂贵工具确认由复合工具的幂等守卫 + system prompt 完成。
"""

import json
import logging
from dataclasses import dataclass

from .state import AgentState, MAX_CONVERSATION_MESSAGES, KEEP_RECENT_MESSAGES
from .prompts import SYSTEM_PROMPT
from ..tools.arg_validator import validate_tool_args_strict

logger = logging.getLogger("hot_projects")

MAX_TOOL_CALLS_PER_TURN = 15


@dataclass
class ToolContext:
    state: AgentState
    provider: object
    db: dict
    progress_cb: object = None


class HotProjectAgent:
    def __init__(self, llm, registry, provider, db):
        self.llm = llm
        self.registry = registry
        self.state = AgentState(db=db)
        self.ctx = ToolContext(state=self.state, provider=provider, db=db)
        self.state.conversation.append({"role": "system", "content": SYSTEM_PROMPT})

    def chat(self, user_message: str, progress_cb=None) -> str:
        if len(user_message) > 2000:
            return "消息过长（超过 2000 字符），请缩短后重试。"

        # 进度回调（仅 WS 路径传入）：榜单复合工具执行期间逐阶段回传百分比
        self.ctx.progress_cb = progress_cb
        self._maybe_compress()
        self.state.conversation.append({"role": "user", "content": user_message})

        for _ in range(MAX_TOOL_CALLS_PER_TURN):
            resp = self.llm.chat(list(self.state.conversation), tools=self.registry.schemas())
            if resp is None:
                msg = "抱歉，LLM 调用失败（A/B 均不可用），请稍后重试。"
                self.state.conversation.append({"role": "assistant", "content": msg})
                return msg

            message = (resp.get("choices") or [{}])[0].get("message", {})
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                content = message.get("content") or "（未生成回复，请重试或换个问法。）"
                self.state.conversation.append({"role": "assistant", "content": content})
                return content

            self.state.conversation.append({
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": tool_calls,
            })

            for tc in tool_calls:
                name = tc.get("function", {}).get("name", "")
                raw = tc.get("function", {}).get("arguments", "{}")
                result = self._run_tool(name, raw)
                self.state.conversation.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": _serialize_result(result),
                })

        return "已达到单轮最大 Tool 调用次数，请尝试简化请求。"

    def _run_tool(self, name: str, raw_args: str) -> dict:
        try:
            args = json.loads(raw_args) if raw_args else {}
            if not isinstance(args, dict):
                raise ValueError
        except (json.JSONDecodeError, ValueError):
            return {"error": "tool arguments 必须是合法 JSON object", "raw": raw_args[:300]}

        validated, errs = validate_tool_args_strict(name, args)
        if errs:
            return {"error": "参数校验失败，请按 invalid_arguments 修正后重试。",
                    "invalid_arguments": errs, "retryable": True}

        logger.info("[Agent] Tool 调用: %s(%s)", name, validated)
        try:
            return self.registry.dispatch(name, self.ctx, validated)
        except Exception as e:  # noqa: BLE001
            logger.error("[Agent] Tool %s 执行异常: %s", name, e)
            return {"error": f"工具执行异常: {e}"}

    def _maybe_compress(self) -> None:
        conv = self.state.conversation
        if len(conv) <= MAX_CONVERSATION_MESSAGES:
            return
        system = next((m for m in conv if m.get("role") == "system"), {"content": SYSTEM_PROMPT})
        non_system = [m for m in conv if m.get("role") != "system"]
        if len(non_system) <= KEEP_RECENT_MESSAGES:
            return
        old, recent = non_system[:-KEEP_RECENT_MESSAGES], non_system[-KEEP_RECENT_MESSAGES:]
        summary = self._summarize(old)
        if summary:
            self.state.conversation_summary = summary
        # 前缀缓存友好：system[0] 保持字节不变（稳定前缀）；摘要作为其后的独立消息。
        rebuilt = [system if system.get("role") == "system" else {"role": "system", "content": SYSTEM_PROMPT}]
        if self.state.conversation_summary:
            rebuilt.append({"role": "user", "content": f"[对话历史摘要]\n{self.state.conversation_summary}"})
        rebuilt += recent
        self.state.conversation = rebuilt
        logger.info("[Agent] 对话历史已压缩: %d 旧消息 → 摘要，保留 %d 条。", len(old), len(recent))

    def _summarize(self, old_msgs: list[dict]) -> str:
        parts = []
        if self.state.conversation_summary:
            parts.append(f"[之前摘要]\n{self.state.conversation_summary}")
        for m in old_msgs:
            role, content = m.get("role", ""), (m.get("content") or "")
            if role == "user":
                parts.append(f"[用户] {content}")
            elif role == "assistant" and content:
                parts.append(f"[助手] {content[:400]}")
        prompt = ("将以下对话历史浓缩为不超过 400 字的中文摘要，保留用户意图、关键参数、"
                  "已执行操作与重要结论：\n\n" + "\n".join(parts))
        data = self.llm.chat([{"role": "user", "content": prompt}], lite=True,
                             max_tokens=600, temperature=0.1, enable_thinking=False)
        if not data:
            return self.state.conversation_summary
        return ((data.get("choices") or [{}])[0].get("message", {}).get("content") or "").strip() \
            or self.state.conversation_summary


def _serialize_result(result: dict, max_len: int = 8000) -> str:
    s = json.dumps(result, ensure_ascii=False, default=str)
    if len(s) <= max_len:
        return s
    return json.dumps({"truncated": True, "preview": s[:max_len]}, ensure_ascii=False)
