"""会话状态与对话历史。

历史这件事看着简单,其实有三条会被 API 直接打回、或者悄悄烧钱的规矩:

1. **孤儿 tool 消息**。`role="tool"` 的消息必须紧跟在带 `tool_calls` 的 assistant 消息
   之后。压缩历史时如果切在中间,留下的开头就是一条孤儿 —— OpenAI 兼容接口一律 400。
2. **大结果反复外发**。一次 `repo_profile` 的结果几千字符,留在历史里的话之后每一轮
   都要重发一遍。超过阈值的换成小存根,完整内容留在本地,模型要细看再用
   `recall_tool_result` 取回。
3. **前缀要稳定**。system 消息必须逐字节不变,摘要作为它之后的独立消息插入 ——
   把摘要拼进 system 会让每次压缩之后的前缀缓存全部落空。

所以这块单独一个文件、单独测:它的错误都是静默的(400 会当成「模型调用失败」,
token 浪费根本没人看得见)。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from .prompts import SYSTEM_PROMPT

logger = logging.getLogger("hot_project")

# 超过这么多条消息就压缩,保留最近这么多条。
MAX_MESSAGES = 35
KEEP_RECENT = 10

# 工具结果超过这么多字符就换存根。存根本身远小于阈值,所以不会被二次卸载,
# 不需要额外的幂等标记。
OFFLOAD_THRESHOLD = 1200
DIGEST_CHARS = 200

# 单条工具结果发给模型的上限。超了只给前半截 —— 一条几万字符的结果会把上下文挤爆,
# 而后半截通常是重复的列表项。
RESULT_MAX_CHARS = 8000


@dataclass
class Session:
    """一次会话。工具通过 `Ctx.state` 拿到它。

    **没有 db 字段**:要读库的工具自己读,写库走事务。旧版把整个 DB 挂在这里,
    工具就地改字典再由别人找机会保存,丢过数据。
    """

    messages: list[dict] = field(default_factory=list)
    summary: str = ""
    # 工具私有的会话槽,agent 层不看内容 —— 榜单的待确认参数、卸载的大结果都放这儿
    tool_state: dict = field(default_factory=dict)
    active_repo: str | None = None
    pending_confirmation_signature: str | None = None

    def __post_init__(self) -> None:
        if not self.messages:
            self.messages.append({"role": "system", "content": SYSTEM_PROMPT})

    # ── 追加 ────────────────────────────────────────────────────

    def user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def assistant(self, text: str | None, tool_calls: list | None = None) -> None:
        message: dict = {"role": "assistant", "content": text}
        if tool_calls:
            message["tool_calls"] = tool_calls
        self.messages.append(message)

    def tool_result(self, call_id: str, result: dict) -> None:
        self.messages.append({"role": "tool", "tool_call_id": call_id,
                              "content": serialize(result)})

    # ── 卸载 ────────────────────────────────────────────────────

    def offload_old_results(self) -> int:
        """把历史里的大结果换成存根。返回卸载了几条。

        只处理**已经存在**的消息,在新一轮开头调用 —— 当前回合的结果必须完整,
        否则模型刚拿到工具输出就只剩摘要,同一回合内没法推理。
        """
        store = self.tool_state.setdefault("offloaded", {})
        done = 0
        for message in self.messages:
            content = message.get("content") or ""
            if message.get("role") != "tool" or len(content) <= OFFLOAD_THRESHOLD:
                continue
            seq = self.tool_state.get("offload_seq", 0) + 1
            self.tool_state["offload_seq"] = seq
            ref = f"tr{seq}"
            store[ref] = content
            message["content"] = json.dumps({
                "offloaded": True, "ref": ref, "digest": content[:DIGEST_CHARS],
                "note": f"完整结果较大已暂存;要细看用 recall_tool_result(ref='{ref}')。",
            }, ensure_ascii=False)
            done += 1
        if done:
            logger.info("[Agent] 卸载了 %d 条大结果。", done)
        return done

    # ── 压缩 ────────────────────────────────────────────────────

    def needs_compress(self) -> bool:
        return len(self.messages) > MAX_MESSAGES

    def compress(self, summarize) -> None:
        """把旧消息换成一段摘要。`summarize(messages) -> str`,返回空串表示总结失败。"""
        if not self.needs_compress():
            return
        system = next((m for m in self.messages if m.get("role") == "system"),
                      {"role": "system", "content": SYSTEM_PROMPT})
        rest = [m for m in self.messages if m.get("role") != "system"]
        if len(rest) <= KEEP_RECENT:
            return

        old, recent = split_at_safe_boundary(rest, KEEP_RECENT)
        if not old:
            return          # 边界推到头了(整段都是一轮工具调用),这次压不了

        if text := summarize(old):
            self.summary = text

        # system 逐字节不变 —— 见模块头部第 3 条
        rebuilt = [system]
        if self.summary:
            rebuilt.append({"role": "user", "content": f"[对话历史摘要]\n{self.summary}"})
        self.messages = rebuilt + recent
        logger.info("[Agent] 历史已压缩:%d 条 → 摘要,保留 %d 条。", len(old), len(recent))


def split_at_safe_boundary(messages: list[dict], keep: int) -> tuple[list[dict], list[dict]]:
    """在不产生孤儿 tool 消息的位置切开。返回 `(丢弃的, 保留的)`。

    先按 `keep` 划线,然后把线往后推到一个安全位置:

        优先   推到保留段里最早的 user 消息 —— 那是一轮对话的自然起点,保留段完整
        兜底   只剥掉开头连续的 tool 消息 —— 上一轮工具调得太多,保留段里没有 user

    兜底那条不够漂亮(保留段会以一条没有提问的 assistant 回复开头),但它至少是合法的;
    而漂亮的做法是多保留几条,那又会让压缩失去意义。
    """
    old, recent = messages[:-keep], messages[-keep:]
    boundary = next((i for i, m in enumerate(recent) if m.get("role") == "user"), None)
    if boundary is None:
        boundary = 0
        while boundary < len(recent) and recent[boundary].get("role") == "tool":
            boundary += 1
    return old + recent[:boundary], recent[boundary:]


def serialize(result: dict, max_chars: int = RESULT_MAX_CHARS) -> str:
    """工具结果 → 发给模型的字符串。太长只给前半截,并明说截断了。"""
    text = json.dumps(result, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return text
    return json.dumps({"truncated": True, "preview": text[:max_chars]}, ensure_ascii=False)
