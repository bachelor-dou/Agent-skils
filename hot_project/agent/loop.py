"""ReAct 循环:问模型 → 它要调工具就调、把结果喂回去 → 它不调工具了就收工。

选工具、确认、参数校验都在别处;这里只管三件事:

    步数护栏      到上限就撤掉工具,逼模型用手上的观察收口
    确认短路      昂贵工具的参数由服务端原样回显,不经模型转述 —— 用户确认的是屏幕上那份
    流式的分轮    每轮一个独立的增量回调,本轮第一片带 reset,免得过渡话粘上最终回答
"""

from __future__ import annotations

import json
import logging

from ..infra import llm
from ..provider.github import facade
from ..tools import Ctx, Registry, registry as default_registry
from .history import Session
from .prompts import SUMMARIZE_PROMPT

logger = logging.getLogger("hot_project")

# 一轮对话里最多问模型几次(每次可含多个工具调用);正常退出是模型不再要工具,这只是护栏。
MAX_STEPS = 15

MESSAGE_MAX_CHARS = 2000
SUMMARY_MAX_TOKENS = 600


class Agent:
    def __init__(self, client=None, tools: Registry | None = None, gh=None) -> None:
        self.llm = client or llm.get()
        self.tools = tools or default_registry()
        self.session = Session()
        self.ctx = Ctx(gh=gh if gh is not None else facade.get(), state=self.session)
        self._model_id = ""
        self._lite_id = ""
        self._on_delta = None

    # ── 一轮对话 ────────────────────────────────────────────────

    def chat(self, message: str, *, progress=None, user_id: str = "",
             model: str = "", lite: str = "", on_delta=None) -> str:
        if len(message) > MESSAGE_MAX_CHARS:
            return f"消息过长(超过 {MESSAGE_MAX_CHARS} 字符),请缩短后重试。"

        self.ctx.progress = progress
        self.ctx.user_id = user_id
        self._model_id = model or ""
        self._lite_id = lite or ""
        self._on_delta = on_delta

        self.session.offload_old_results()      # 卸载上一轮的大结果,不碰本轮
        self.session.compress(self._summarize)
        self.session.user(message)

        for _ in range(MAX_STEPS):
            reply = self._ask(tools=self.tools.schemas())
            if reply is None:
                return self._say(self._failure_message())

            calls = reply.get("tool_calls") or []
            if not calls:
                return self._say(reply.get("content") or "(没生成回复,请重试或换个问法。)")

            self.session.assistant(reply.get("content"), calls)
            if confirm := self._run_calls(calls):
                return self._say(confirm)       # 确认短路,等用户回「开始」

        return self._finalize()

    # ── 内部 ────────────────────────────────────────────────────

    def _ask(self, *, tools=None) -> dict | None:
        data = self.llm.chat(list(self.session.messages), tools=tools,
                             model_id=self._model_id or None,
                             on_delta=self._round_delta())
        if data is None:
            return None
        return (data.get("choices") or [{}])[0].get("message", {})

    def _run_calls(self, calls: list[dict]) -> str | None:
        """执行本轮全部工具调用。返回要短路的确认文案,没有就返回 None。"""
        confirm: str | None = None
        for call in calls:
            fn = call.get("function") or {}
            try:
                result = self.run_tool(fn.get("name", ""), fn.get("arguments", "{}"))
            except Exception as e:      # noqa: BLE001 —— 兜底,理由见下
                # 每条 tool_calls 必须无条件配一条 tool 回复:漏一条,该会话之后每次请求
                # 都被接口 400,只能等 TTL 过期。
                logger.exception("[Agent] 工具 %s 异常逃出 run_tool", fn.get("name", ""))
                result = {"error": f"工具调用异常:{e}", "retryable": True}
            self.session.tool_result(call.get("id", ""), result)
            if confirm is None and result.get("needs_confirmation"):
                confirm = result.get("message") or "请确认参数后回复『开始』。"
        return confirm

    def run_tool(self, name: str, raw_args: str) -> dict:
        """跑一个工具。任何失败都变成一个 dict 回给模型 —— 它能读懂并自己改。"""
        tool = self.tools.get(name)
        if tool is None:
            return {"error": f"没有这个工具:{name}", "available": self.tools.names()}
        try:
            args = json.loads(raw_args) if raw_args else {}
            if not isinstance(args, dict):
                raise ValueError("不是 object")
        except (json.JSONDecodeError, ValueError, TypeError):
            return {"error": "tool arguments 必须是合法的 JSON object",
                    "raw": str(raw_args)[:300], "retryable": True}

        clean, errors = tool.validate(args)
        if errors:
            return {"error": "参数校验失败,请按 invalid_arguments 修正后重试。",
                    "invalid_arguments": errors, "retryable": True}

        logger.info("[Agent] 调用 %s(%s)", name, clean)
        try:
            return tool.run(self.ctx, clean)
        except Exception as e:      # noqa: BLE001 —— 工具崩了要让模型知道,不能掀翻会话
            logger.exception("[Agent] 工具 %s 执行异常", name)
            return {"error": f"工具执行异常:{e}"}

    def _finalize(self) -> str:
        """撞上步数护栏:撤掉工具再问一次,逼它用手上的观察给个答案。"""
        logger.warning("[Agent] 到了 %d 步上限,撤掉工具强制收口。", MAX_STEPS)
        reply = self._ask() or {}
        return self._say((reply.get("content") or "").strip()
                         or "这个问题步骤较多,我取证了几轮仍没收敛。"
                            "把需求拆细一点或换个问法再试。")

    def _say(self, text: str) -> str:
        self.session.assistant(text)
        return text

    def _failure_message(self) -> str:
        return ("抱歉,所选模型调用失败,请稍后重试或在下方切换其他模型。"
                if self._model_id else
                "抱歉,LLM 调用失败(所有已配置模型都不可用),请稍后重试。")

    def _round_delta(self):
        """每轮一个新的增量回调,本轮第一片带 reset。"""
        outer = self._on_delta
        if not outer:
            return None
        first = [True]

        def emit(piece: str) -> None:
            outer(piece, first[0])
            first[0] = False

        return emit

    def _summarize(self, old: list[dict]) -> str:
        """总结旧消息。失败就返回原摘要 —— 压缩本身不能失败,否则历史会一直涨。"""
        parts = [f"[之前摘要]\n{self.session.summary}"] if self.session.summary else []
        for message in old:
            content = message.get("content") or ""
            if message.get("role") == "user":
                parts.append(f"[用户] {content}")
            elif message.get("role") == "assistant" and content:
                parts.append(f"[助手] {content[:400]}")

        text = self.llm.text(SUMMARIZE_PROMPT + "\n".join(parts), lite=True,
                             model_id=self._model_id or None,
                             lite_id=self._lite_id or None,
                             max_tokens=SUMMARY_MAX_TOKENS, temperature=0.1,
                             enable_thinking=False)
        return text or self.session.summary
