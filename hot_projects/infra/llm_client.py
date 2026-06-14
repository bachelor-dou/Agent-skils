"""LLM 客户端：A/B 双后端 + 逐调用回退 + 按后端参数适配。

- 方案 A（主力，默认 Azure OpenAI）/ 方案 B（备选，默认 SiliconFlow）。
- 逐调用回退：每次调用先用 A，A 失败（自身重试耗尽/连接错误/HTTP 错误/空响应）
  则该次改用 B；下次调用仍优先 A（A 恢复后自动用回 A，不粘滞）。
- 按后端做参数白名单：
  - azure(gpt-5.x): 用 max_completion_tokens；不发 enable_thinking/thinking_budget；省略 temperature。
  - openai(SiliconFlow): 用 max_tokens/temperature/enable_thinking/thinking_budget。
"""

import logging
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger("hot_projects")

LLM_RETRY_BACKOFF_SECONDS = (1.0, 2.0, 4.0)


def build_headers(backend: str, key: str) -> dict:
    """按后端返回鉴权请求头。"""
    if backend == "azure":
        return {"api-key": key, "Content-Type": "application/json"}
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def build_payload(
    backend: str,
    model: str,
    messages: list[dict],
    max_tokens: int | None = None,
    temperature: float | None = None,
    enable_thinking: bool | None = None,
    thinking_budget: int | None = None,
    tools: list[dict] | None = None,
) -> dict:
    """按后端构造请求体（参数白名单）。"""
    payload: dict = {"model": model, "messages": messages}
    if backend == "azure":
        if max_tokens is not None:
            payload["max_completion_tokens"] = max_tokens
        # azure(gpt-5.x): 省略 temperature；不发 enable_thinking/thinking_budget
    else:
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if enable_thinking is not None:
            payload["enable_thinking"] = enable_thinking
        if thinking_budget is not None:
            payload["thinking_budget"] = thinking_budget
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    return payload


@dataclass
class LLMScheme:
    backend: str
    url: str
    key: str
    model: str
    lite_model: str


def _request_once(
    scheme: LLMScheme,
    model: str,
    *,
    messages: list[dict],
    tools: list[dict] | None,
    max_tokens: int | None,
    temperature: float | None,
    enable_thinking: bool | None,
    thinking_budget: int | None,
    timeout: int = 300,
) -> dict | None:
    """单次请求单个后端（含 3 次退避重试）；成功返回 data dict，失败返回 None。"""
    if not scheme.key or not scheme.url:
        logger.warning("[LLM] 方案 %s 未配置 url/key，跳过。", scheme.backend)
        return None
    headers = build_headers(scheme.backend, scheme.key)
    payload = build_payload(
        scheme.backend,
        model,
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
        tools=tools,
    )
    for attempt in range(len(LLM_RETRY_BACKOFF_SECONDS)):
        try:
            resp = requests.post(scheme.url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                msg = (data.get("choices") or [{}])[0].get("message", {})
                if (msg.get("content") or "").strip() or msg.get("tool_calls"):
                    return data
                logger.warning("[LLM] %s 空响应, attempt=%d", scheme.backend, attempt + 1)
            else:
                logger.warning(
                    "[LLM] %s HTTP %s: %s", scheme.backend, resp.status_code, resp.text[:200]
                )
        except requests.RequestException as e:
            logger.warning("[LLM] %s 请求异常: %s", scheme.backend, e)
        except ValueError as e:
            logger.warning("[LLM] %s 响应 JSON 解析失败: %s", scheme.backend, e)
        if attempt < len(LLM_RETRY_BACKOFF_SECONDS) - 1:
            time.sleep(LLM_RETRY_BACKOFF_SECONDS[attempt])
    return None


class LLMClient:
    """A/B 双后端，逐调用回退。"""

    def __init__(self, scheme_a: LLMScheme, scheme_b: LLMScheme) -> None:
        self.a = scheme_a
        self.b = scheme_b

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        lite: bool = False,
        max_tokens: int | None = 16384,
        temperature: float | None = 0.3,
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
    ) -> dict | None:
        """发起对话；先 A 后 B 逐调用回退。lite=True 时使用各方案的小模型。"""
        for scheme in (self.a, self.b):
            model = scheme.lite_model if lite else scheme.model
            data = _request_once(
                scheme,
                model,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget,
            )
            if data is not None:
                return data
            logger.warning("[LLM] 方案 %s 失败，尝试回退。", scheme.backend)
        return None


def client_from_config() -> LLMClient:
    """从全局 config 构造 LLMClient（A=主力，B=备选）。"""
    from .. import config as cfg

    a = LLMScheme(cfg.LLM_A_BACKEND, cfg.LLM_A_URL, cfg.LLM_A_KEY, cfg.LLM_A_MODEL, cfg.LLM_A_LITE_MODEL)
    b = LLMScheme(cfg.LLM_B_BACKEND, cfg.LLM_B_URL, cfg.LLM_B_KEY, cfg.LLM_B_MODEL, cfg.LLM_B_LITE_MODEL)
    return LLMClient(a, b)
