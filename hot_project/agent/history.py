"""会话状态与对话历史。三条规矩,违反了都是静默出错:

1. `role="tool"` 必须紧跟带 `tool_calls` 的 assistant 消息。压缩切在中间会留下孤儿,
   OpenAI 兼容接口一律 400。
2. 超过阈值的大结果换成小存根,完整内容留在本地,模型要细看再用 `recall_tool_result` 取回。
3. system 消息必须逐字节不变,摘要作为它之后的独立消息插入 —— 拼进 system 会让每次压缩
   之后的前缀缓存全部落空。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from .prompts import SYSTEM_PROMPT

logger = logging.getLogger("hot_project")

MAX_MESSAGES = 35
KEEP_RECENT = 10

OFFLOAD_THRESHOLD = 1200
DIGEST_CHARS = 200

RESULT_MAX_CHARS = 8000


@dataclass
class Session:
    """一次会话。工具通过 `Ctx.state` 拿到它。

    没有 db 字段:要读库的工具自己读,写库走事务。
    """

    messages: list[dict] = field(default_factory=list)
    summary: str = ""
    tool_state: dict = field(default_factory=dict)
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

        只在新一轮开头调用:本回合的结果必须完整,否则模型同一回合内没法推理。
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

        rebuilt = [system]
        if self.summary:
            rebuilt.append({"role": "user", "content": f"[对话历史摘要]\n{self.summary}"})
        self.messages = rebuilt + recent
        dropped = self._drop_orphan_offloads()
        logger.info("[Agent] 历史已压缩:%d 条 → 摘要,保留 %d 条%s。", len(old), len(recent),
                    f",顺带清掉 {dropped} 份取不到的暂存结果" if dropped else "")

    def _drop_orphan_offloads(self) -> int:
        """丢掉那些 ref 已经不在上下文里的暂存结果,返回丢了几份。

        `recall_tool_result` 只认模型看得见的 ref,压缩掉之后谁都取不到,留着纯占内存。
        """
        store = self.tool_state.get("offloaded") or {}
        if not store:
            return 0
        alive = "".join(m.get("content") or "" for m in self.messages)
        orphans = [ref for ref in store if f'"{ref}"' not in alive]
        for ref in orphans:
            del store[ref]
        return len(orphans)


def split_at_safe_boundary(messages: list[dict], keep: int) -> tuple[list[dict], list[dict]]:
    """在不产生孤儿 tool 消息的位置切开。返回 `(丢弃的, 保留的)`。

    线优先推到保留段最早的 user 消息;没有 user 就只剥掉开头连续的 tool 消息。
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
