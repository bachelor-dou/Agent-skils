"""多平台客户端:决定这次调用按什么顺序试哪些平台。

两种模式,区别只在**失败之后**:

    内部调用(没指定 model_id)   按目录顺序逐个试,某个平台挂了顺延下一个。
                                周报、描述生成这些没人盯着,韧性优先。
    网页硬切换(指定了 model_id) 只用选中的那个,失败就返回 None。
                                用户明确选了 A,悄悄回退到 B 给出的答案是另一回事。

`lite` 是「用便宜的子模型」。子模型池跨平台共享:选中的主模型是 azure,lite 却可以借
硅基流动的 Qwen —— 因为主模型选择和「便宜活儿用谁干」本来就是两件事。
"""

from __future__ import annotations

import logging

from . import wire
from .schemes import Scheme

logger = logging.getLogger("hot_project")


class LLMClient:
    def __init__(self, schemes: list[Scheme]) -> None:
        self.schemes = list(schemes)

    def usable(self) -> list[Scheme]:
        return [s for s in self.schemes if s.usable]

    def configured(self) -> bool:
        """有没有任何一个平台能用。没有就别走 LLM 那条路,直接回退。"""
        return bool(self.usable())

    def resolve_lite(self, lite_id: str) -> tuple[Scheme, str] | None:
        """`"平台id:子模型名"` → `(平台, 子模型名)`。对不上返回 None。"""
        pid, _, name = lite_id.partition(":")
        sel = next((s for s in self.usable() if s.id == pid), None)
        return (sel, name) if sel and name in sel.lite_models else None

    def _lite_order(self, preferred: Scheme | None) -> list[tuple[Scheme, str]]:
        """lite「自动」的候选顺序。

        先用主模型所在平台的子模型,再按目录顺序借别家的。只用真配了子模型的平台
        (各取首个);全都没配子模型时才退回主模型 —— 那样虽然贵,但至少调得通。
        """
        usable = self.usable()
        ordered = ([preferred] + [s for s in usable if s.id != preferred.id]
                   if preferred is not None else usable)
        subs = [(s, s.lite_models[0]) for s in ordered if s.lite_models]
        return subs or [(s, s.model) for s in ordered]

    def _order(self, *, lite: bool, model_id: str | None,
               lite_id: str | None) -> list[tuple[Scheme, str]] | None:
        """这次调用要按什么顺序试。返回 None 表示指定的模型压根不可用。"""
        usable = self.usable()
        if lite and lite_id:
            resolved = self.resolve_lite(lite_id)
            if resolved is None:
                logger.warning("[LLM] 子模型 %s 不可用(没配置或不存在)。", lite_id)
                return None
            return [resolved]                                   # 硬选,不回退
        if lite:
            preferred = next((s for s in usable if s.id == model_id), None) if model_id else None
            return self._lite_order(preferred)
        if model_id:
            sel = next((s for s in usable if s.id == model_id), None)
            if sel is None:
                logger.warning("[LLM] 模型 %s 不可用(没配置或不存在)。", model_id)
                return None
            return [(sel, sel.model)]                           # 硬切换,不回退
        return [(s, s.model) for s in usable]                   # 内部调用,顺序回退

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        lite: bool = False,
        model_id: str | None = None,
        lite_id: str | None = None,
        max_tokens: int | None = 16384,
        temperature: float | None = 0.3,
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
        timeout: int = 300,
        attempts: int | None = None,
        on_delta=None,
    ) -> dict | None:
        order = self._order(lite=lite, model_id=model_id, lite_id=lite_id)
        if not order:
            return None

        # 「已经外发过就不能重来」这条不只管平台内的重试,也管换平台:p1 吐了半句才断线,
        # 回退到 p2 会让用户看到「半句 + 另一个完整答案」—— 和重复重试是同一种坏。
        # wire 那层只看得见自己这一次请求,看不见前一个平台外发过什么,所以得在这里记。
        emitted = False
        if on_delta is not None:
            inner = on_delta

            def on_delta(piece):        # noqa: F811 —— 有意遮蔽,下面只用包装后的
                nonlocal emitted
                emitted = True
                inner(piece)

        for scheme, model in order:
            data = wire.request(
                scheme, model, messages, tools=tools,
                max_tokens=max_tokens, temperature=temperature,
                enable_thinking=enable_thinking, thinking_budget=thinking_budget,
                timeout=timeout, attempts=attempts, on_delta=on_delta,
            )
            if data is not None:
                return data
            logger.warning("[LLM] %s(%s) 调用失败。", scheme.id, model)
            if emitted:
                logger.warning("[LLM] 已向前端外发过内容,不再换平台(否则文字会重复)。")
                return None
        return None

    def text(self, prompt: str, **kwargs) -> str:
        """单轮问答,只要正文。失败或空回复返回空串。

        绝大多数内部调用(描述生成、批量浓缩)都是这个形状:一段提示词进,一段文字出。
        旧代码里每个调用点都自己写一遍 `[{"role": "user", ...}]` 加一个
        `(data.get("choices") or [{}])[0].get("message", {}).get("content")` 的取值链,
        取值链写错一处就是静默返回空描述。
        """
        data = self.chat([{"role": "user", "content": prompt}], **kwargs)
        if not data:
            return ""
        return ((data.get("choices") or [{}])[0].get("message", {}).get("content") or "").strip()

    def ping(self, *, model_id: str | None = None, lite_id: str | None = None,
             timeout: int = 20) -> bool:
        """预检:给选中的模型发一次极小请求,确认链路真的通。

        `max_tokens` 留足余量 —— reasoning 模型思考也吃输出配额,给太小会回一段空正文,
        看起来就像「这个模型不可用」。
        """
        return self.chat(
            [{"role": "user", "content": "ping"}],
            lite=bool(lite_id), model_id=model_id, lite_id=lite_id,
            max_tokens=512, temperature=0.0, enable_thinking=False,
            timeout=timeout, attempts=1,        # 预检求快,不退避重试
        ) is not None
