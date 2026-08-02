"""线上协议:请求头、请求体、一次请求(含流式)。

各家「兼容 OpenAI」的程度不一样,差异集中在这个文件里,别处不该再出现 `if backend ==`。
已知差异:azure(gpt-5.x)用 max_completion_tokens,不认 temperature 和 thinking 系参数。

发了对方不认的参数不是被忽略,是整个请求 400 —— 所以这里是白名单:新参数默认不发给
未知后端,由加参数的人显式列进去。
"""

from __future__ import annotations

import json
import logging
import time

import requests

logger = logging.getLogger("hot_project")

AZURE = "azure"
RETRY_BACKOFF = (1.0, 2.0, 4.0)


def headers(backend: str, key: str) -> dict:
    if backend == AZURE:
        return {"api-key": key, "Content-Type": "application/json"}
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def payload(
    backend: str,
    model: str,
    messages: list[dict],
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    enable_thinking: bool | None = None,
    thinking_budget: int | None = None,
    tools: list[dict] | None = None,
) -> dict:
    body: dict = {"model": model, "messages": messages}
    if backend == AZURE:
        if max_tokens is not None:
            body["max_completion_tokens"] = max_tokens
    else:
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        if enable_thinking is not None:
            body["enable_thinking"] = enable_thinking
        if thinking_budget is not None:
            body["thinking_budget"] = thinking_budget
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    return body


# ── 工具调用泄漏 ─────────────────────────────────────────────────────
# 有些推理模型偶尔把本该走 tool_calls 通道的调用当成 JSON 文本吐在正文开头,例如
# `{"tool_uses":[{"recipient_name":...}]}`。这段不能给用户看见,外发前剥掉。

_LEAK_KEYS = ('"tool_uses"', '"recipient_name"', '"tool_calls"', '"parameters"')


def strip_leaked_toolcall(text: str) -> str | None:
    """剥掉正文开头那段疑似工具调用泄漏的 JSON。

    三态,流式靠它决定要不要继续缓冲:剥完的余文 / 原文(不含泄漏特征,不能动)/
    None(花括号未闭合,判不了)。
    """
    s = text.lstrip()
    if not s.startswith("{"):
        return text
    depth = 0
    in_str = esc = False
    end = None
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None
    blob = s[:end]
    return text if not any(k in blob for k in _LEAK_KEYS) else s[end:].lstrip()


def merge_toolcall_fragment(acc: dict, frag: dict) -> None:
    """把一个流式 tool_call 增量片段按 index 合并进累加器。

    首片带 id/type/name,后续片只带 arguments 的一截。同一 chunk 里重复的 index
    (vLLM 推测解码会这么发)必须落进同一槽位,不能各成一项。
    """
    slot = acc.setdefault(frag.get("index", 0),
                          {"id": "", "type": "function",
                           "function": {"name": "", "arguments": ""}})
    if frag.get("id"):
        slot["id"] = frag["id"]
    if frag.get("type"):
        slot["type"] = frag["type"]
    fn = frag.get("function") or {}
    if fn.get("name"):
        slot["function"]["name"] += fn["name"]
    if fn.get("arguments"):
        slot["function"]["arguments"] += fn["arguments"]


class _HeadGate:
    """正文开头的闸门:只拦「以 `{` 起头、疑似工具泄漏」的那一段。

    散文和 Markdown 零延迟放行,否则每个回答的首字延迟都要为这个罕见情况买单。
    """

    def __init__(self, emit) -> None:
        self._emit = emit
        self._open = False
        self._buf = ""

    def feed(self, piece: str) -> None:
        if self._open:
            self._emit(piece)
            return
        self._buf += piece
        if not self._buf.strip():
            return                      # 目前只有空白,判不了,继续等
        result = strip_leaked_toolcall(self._buf)
        if result is None:
            return                      # 开头是还没闭合的 JSON,继续缓冲
        self._release(result)           # 散文在这里原样放行,不多等一片

    def flush(self) -> None:
        """收流结束仍卡在闸门里(开头的 `{` 始终没闭合):尽力剥一次再放行,不丢用户内容。"""
        if self._open or not self._buf.strip():
            return
        result = strip_leaked_toolcall(self._buf)
        self._release(self._buf if result is None else result)

    def _release(self, text: str) -> None:
        self._open = True
        self._buf = ""
        self._emit(text)


def _stream(scheme, model: str, body: dict, on_delta, timeout: int) -> tuple[dict | None, bool]:
    """一次 SSE 请求,边收边把正文增量喂给 `on_delta`。不重试。

    返回 `(data, emitted)`。`emitted` 为真就**不能重试**,否则前端会看到重复的文字。
    """
    body = dict(body, stream=True)
    parts: list[str] = []
    tool_acc: dict = {}
    finish_reason = None
    emitted = False
    resp = None
    started = time.time()
    first_at = None     # 首字耗时:分辨「转半天不吐字」是模型在思考还是流式假死
    pieces = 0          # 增量条数:接近字符数 = 逐 token 流;远小于 = 被网关整块缓冲

    def emit(text: str) -> None:
        nonlocal first_at, pieces, emitted
        if not text:
            return
        if first_at is None:
            first_at = time.time() - started
        pieces += 1
        parts.append(text)
        emitted = True
        on_delta(text)

    gate = _HeadGate(emit)
    try:
        resp = requests.post(scheme.url, headers=headers(scheme.backend, scheme.key),
                             json=body, timeout=timeout, stream=True)
        if resp.status_code != 200:
            logger.warning("[LLM] %s 流式 HTTP %s: %s",
                           scheme.id, resp.status_code, resp.text[:200])
            return None, emitted
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue                        # 空行、注释行、SSE 的 event: 行
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except ValueError:
                continue                        # 半个 JSON / 填充行
            choice = (obj.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            if piece := delta.get("content"):
                gate.feed(piece)
            for frag in delta.get("tool_calls") or []:
                merge_toolcall_fragment(tool_acc, frag)
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
    except requests.RequestException as e:
        logger.warning("[LLM] %s 流式请求异常: %s", scheme.id, e)
        return None, emitted
    finally:
        if resp is not None:
            resp.close()        # 非 200 早退、[DONE] break、异常,各路径都要归还连接

    gate.flush()

    if chars := sum(len(p) for p in parts):
        logger.info("[LLM] %s 流式: 首字 %.1fs, 增量 %d 条, 正文 %d 字, 总 %.1fs",
                    scheme.id, first_at or 0.0, pieces, chars, time.time() - started)

    tool_calls = [tool_acc[i] for i in sorted(tool_acc)]
    content = "".join(parts)
    if not (content.strip() or tool_calls):
        return None, emitted                    # 全空视为失败(没外发过就还能重试)
    message: dict = {"role": "assistant", "content": content or (None if tool_calls else "")}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message, "finish_reason": finish_reason}]}, emitted


def _blocking(scheme, body: dict, timeout: int) -> dict | None:
    """一次普通请求。成功且回复非空才算成功。"""
    try:
        resp = requests.post(scheme.url, headers=headers(scheme.backend, scheme.key),
                             json=body, timeout=timeout)
        if resp.status_code != 200:
            logger.warning("[LLM] %s HTTP %s: %s", scheme.id, resp.status_code, resp.text[:200])
            return None
        data = resp.json()
        msg = (data.get("choices") or [{}])[0].get("message", {})
        if (msg.get("content") or "").strip() or msg.get("tool_calls"):
            return data
        logger.warning("[LLM] %s 回复为空。", scheme.id)
    except requests.RequestException as e:
        logger.warning("[LLM] %s 请求异常: %s", scheme.id, e)
    except ValueError as e:
        logger.warning("[LLM] %s 响应不是 JSON: %s", scheme.id, e)
    return None


def request(
    scheme,
    model: str,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    enable_thinking: bool | None = None,
    thinking_budget: int | None = None,
    timeout: int = 300,
    attempts: int | None = None,
    on_delta=None,
) -> dict | None:
    """向一个平台发请求,自带退避重试。失败返回 None,由上层决定要不要换平台。"""
    if not scheme.usable:
        logger.warning("[LLM] %s 没配 url/key,跳过。", scheme.id)
        return None

    body = payload(scheme.backend, model, messages,
                   max_tokens=max_tokens, temperature=temperature,
                   enable_thinking=enable_thinking, thinking_budget=thinking_budget,
                   tools=tools)
    rounds = attempts or len(RETRY_BACKOFF)
    for i in range(rounds):
        if on_delta is not None:
            data, emitted = _stream(scheme, model, body, on_delta, timeout)
            if data is not None:
                return data
            if emitted:
                return None                     # 已经外发过,不能重试
        else:
            if (data := _blocking(scheme, body, timeout)) is not None:
                return data
        if i < rounds - 1:
            time.sleep(RETRY_BACKOFF[min(i, len(RETRY_BACKOFF) - 1)])
    return None
