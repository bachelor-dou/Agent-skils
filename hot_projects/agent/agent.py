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

# 本轮对话内 Agent 的最大「步数」（=向 LLM 请求的次数/ReAct 轮次，每步可含多个 tool_calls）。
# 正常退出条件是模型不再返回 tool_calls（见循环内 return）；本上限只是防失控/护栏，
# 命中后不再给工具、强制模型基于已有观察收口，而非死循环或返回死胡同话术。
# Web 部署有全局工具锁 + WS 超时，故用有限步数护栏，而非 while True。
MAX_AGENT_STEPS = 15

# 工具结果卸载阈值：tool 结果串超过该字符数时，在下一轮开头替换为小存根（完整内容
# 暂存到 tool_state），减少大结果在后续轮次里被反复传输的 token 开销。当前回合内结果
# 始终完整（同回合推理不受影响），模型需要旧结果细节时用 recall_tool_result(ref) 取回。
OFFLOAD_THRESHOLD = 1200


@dataclass
class ToolContext:
    state: AgentState
    provider: object
    db: dict
    progress_cb: object = None
    user_id: str = ""  # 当前 Web 用户（用于按用户收藏）；CLI/无身份时为空


class HotProjectAgent:
    def __init__(self, llm, registry, provider, db):
        self.llm = llm
        self.registry = registry
        self.state = AgentState(db=db)
        self.ctx = ToolContext(state=self.state, provider=provider, db=db)
        self.state.conversation.append({"role": "system", "content": SYSTEM_PROMPT})
        self._model_id = ""  # 本会话当前选用的模型 id（网页可切换）；空=按配置顺序回退
        self._lite_id = ""   # 本会话选用的子模型 id（"平台id:模型名"）；空=跟随主模型平台

    def chat(self, user_message: str, progress_cb=None, user_id: str = "",
             model: str = "", lite: str = "") -> str:
        if len(user_message) > 2000:
            return "消息过长（超过 2000 字符），请缩短后重试。"

        # 进度回调（仅 WS 路径传入）：榜单复合工具执行期间逐阶段回传百分比
        self.ctx.progress_cb = progress_cb
        self.ctx.user_id = user_id
        self._model_id = model or ""
        self._lite_id = lite or ""
        self._offload_large_tool_results()  # 卸载上一轮的大结果为存根（本轮开头，不碰当前回合）
        self._maybe_compress()
        self.state.conversation.append({"role": "user", "content": user_message})

        for _ in range(MAX_AGENT_STEPS):
            resp = self.llm.chat(list(self.state.conversation), tools=self.registry.schemas(),
                                 model_id=self._model_id or None)
            if resp is None:
                msg = ("抱歉，所选模型调用失败，请稍后重试或在下方切换其他模型。"
                       if self._model_id else
                       "抱歉，LLM 调用失败（所有已配置模型均不可用），请稍后重试。")
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

        # 命中步数护栏：不再给工具，强制模型基于已有观察给出最终回答，而非死胡同话术。
        return self._finalize_without_tools()

    def _finalize_without_tools(self) -> str:
        logger.warning("[Agent] 已达最大步数 %d，强制无工具收口。", MAX_AGENT_STEPS)
        resp = self.llm.chat(list(self.state.conversation), model_id=self._model_id or None)
        content = ""
        if resp:
            content = ((resp.get("choices") or [{}])[0]
                       .get("message", {}).get("content") or "").strip()
        if not content:
            content = "这个问题步骤较多，我已多次取证但未能收敛。请把需求拆细一点或换个问法再试。"
        self.state.conversation.append({"role": "assistant", "content": content})
        return content

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

    def _offload_large_tool_results(self) -> None:
        """把历史里超阈值的 tool 结果替换为小存根，完整内容存入 tool_state。

        存根本身 < 阈值，故不会被二次卸载（无需额外幂等标记）。仅处理已存在的历史消息，
        当前正在进行的回合尚未开始，其结果不受影响。
        """
        store = self.state.tool_state.setdefault("offloaded", {})
        for m in self.state.conversation:
            if m.get("role") != "tool":
                continue
            content = m.get("content", "")
            if len(content) <= OFFLOAD_THRESHOLD:
                continue
            seq = self.state.tool_state.get("offload_seq", 0) + 1
            self.state.tool_state["offload_seq"] = seq
            ref = f"tr{seq}"
            store[ref] = content
            m["content"] = json.dumps({
                "offloaded": True,
                "ref": ref,
                "digest": content[:200],
                "note": f"完整结果较大已暂存；需要细看用 recall_tool_result(ref='{ref}')。",
            }, ensure_ascii=False)

    def _maybe_compress(self) -> None:
        conv = self.state.conversation
        if len(conv) <= MAX_CONVERSATION_MESSAGES:
            return
        system = next((m for m in conv if m.get("role") == "system"), {"content": SYSTEM_PROMPT})
        non_system = [m for m in conv if m.get("role") != "system"]
        if len(non_system) <= KEEP_RECENT_MESSAGES:
            return
        old, recent = non_system[:-KEEP_RECENT_MESSAGES], non_system[-KEEP_RECENT_MESSAGES:]
        # 边界安全（Claude Code 式压缩）：recent 不能以孤儿 tool 消息开头——它对应的
        # assistant/tool_calls 已被归入 old，OpenAI 兼容接口会拒绝这样的历史。
        # 优先把边界对齐到 recent 内最早的 user 消息（保留完整轮次）；
        # 若切片内没有 user 消息（上一轮工具调用过长），则仅剥离开头的孤儿 tool 消息。
        boundary = next((i for i, m in enumerate(recent) if m.get("role") == "user"), None)
        if boundary is None:
            boundary = 0
            while boundary < len(recent) and recent[boundary].get("role") == "tool":
                boundary += 1
        old, recent = old + recent[:boundary], recent[boundary:]
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
                             model_id=self._model_id or None,
                             lite_id=self._lite_id or None,
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
