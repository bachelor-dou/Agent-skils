"""LLM 客户端：多平台顺序回退 + 硬切换 + 按后端参数适配。

- 多平台：config.LLM_MODELS 每条 = 一个平台（url + key）。usable() 过滤出配了 key 的。
- 内部调用（无 model_id/lite_id，如摘要/描述/定时任务）：按列表顺序逐个平台尝试，
  某平台失败（重试耗尽/连接错误/HTTP 错误/空响应）则顺延下一个，保留韧性（不粘滞）。
- 网页硬切换（指定 model_id 或 lite_id）：只用选中的那个，失败即返回 None，不回退。
- 按后端做参数白名单：
  - azure(gpt-5.x): 用 max_completion_tokens；不发 enable_thinking/thinking_budget；省略 temperature。
  - 其它(openai 兼容): 用 max_tokens/temperature/enable_thinking/thinking_budget。
"""

import json
import logging
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger("hot_projects")

LLM_RETRY_BACKOFF_SECONDS = (1.0, 2.0, 4.0)


def _is_truthy(value) -> bool:
    """健壮布尔：1/True/"1"/"true"/"yes"/"on" 为真；0/False/"0"/"false"/"" 为假。
    （裸真值判断会把字符串 "0" 当真，这里按字面纠正。）"""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def normalize_models(raw: list[dict]) -> list[dict]:
    """把手写的 config.LLM_MODELS 归一化为下游可依赖的稳定结构（配置与代码解耦）。

    - enabled 为假（0/False/"0"…）的条目剔除，缺省视为开启；
    - 缺字段补默认值、key 一律强转字符串（配置怎么写都不 KeyError；空 key 由 usable() 跳过）；
    - id 唯一（撞 id 直接报错）、缺 id 跳过——id 是选择链路的键，按平台命名而非模型名；
    - lite_model 逗号串 → lite_models（仅平台内去重；跨平台融合去重在 api_server 做，不污染内部回退）。
    """
    out: list[dict] = []
    seen_ids: set[str] = set()
    for m in raw:
        if not _is_truthy(m.get("enabled", True)):
            continue
        mid = str(m.get("id") or "").strip()
        if not mid:
            continue  # 没有 id 无法被前端选择 / 硬切换定位，跳过而非崩溃
        if mid in seen_ids:
            raise ValueError(f"LLM_MODELS 存在重复 id: {mid!r}（id 需唯一，按平台命名如 azure01/aliyun02）")
        seen_ids.add(mid)

        seen_here: set[str] = set()
        lite_models: list[str] = []
        for name in (s.strip() for s in str(m.get("lite_model") or "").split(",")):
            if name and name not in seen_here:
                seen_here.add(name)
                lite_models.append(name)

        out.append({
            "id": mid,
            "label": str(m.get("label") or mid),
            "backend": str(m.get("backend") or "openai"),
            "url": str(m.get("url") or ""),
            "model": str(m.get("model") or ""),
            "lite_model": str(m.get("lite_model") or ""),
            "lite_models": lite_models,
            "key": str(m.get("key") or ""),
            "enabled": True,
            "desc": str(m.get("desc") or ""),
        })
    return out


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
    lite_models: list[str]
    id: str = ""
    label: str = ""


def _merge_tool_call_fragment(acc: dict, frag: dict) -> None:
    """把一个流式 tool_call 增量片段按 index 合并进累加器 acc（index -> 槽位）。

    OpenAI 流式约定：首片带 id/type/function.name，后续片仅带 function.arguments 的
    局部 JSON 串，须按 index 拼接。逐片处理天然也修复了「同一 chunk 内重复 index」的坑
    （vLLM/推测解码会这样发），因为相同 index 落到同一槽位而非各成一项。
    """
    idx = frag.get("index", 0)
    slot = acc.setdefault(idx, {"id": "", "type": "function",
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


def _stream_once(
    scheme: LLMScheme,
    model: str,
    *,
    messages: list[dict],
    tools: list[dict] | None,
    max_tokens: int | None,
    temperature: float | None,
    enable_thinking: bool | None,
    thinking_budget: int | None,
    on_delta,
    timeout: int,
) -> tuple[dict | None, bool]:
    """单次流式请求（SSE），边收边把正文增量喂给 on_delta；不做重试。

    返回 (data, emitted)：data 为组装成的、与非流式同结构的响应 dict（失败为 None）；
    emitted 表示是否已向 on_delta 外发过内容——一旦外发，上层就不能再重试（否则前端重复）。
    """
    headers = build_headers(scheme.backend, scheme.key)
    payload = build_payload(
        scheme.backend, model, messages,
        max_tokens=max_tokens, temperature=temperature,
        enable_thinking=enable_thinking, thinking_budget=thinking_budget, tools=tools,
    )
    payload["stream"] = True
    content_parts: list[str] = []
    tool_acc: dict = {}
    finish_reason = None
    emitted = False
    resp = None
    try:
        resp = requests.post(scheme.url, headers=headers, json=payload,
                             timeout=timeout, stream=True)
        if resp.status_code != 200:
            logger.warning("[LLM] %s 流式 HTTP %s: %s",
                           scheme.backend, resp.status_code, resp.text[:200])
            return None, emitted
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue  # 跳过空行、注释行与 SSE event: 行
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except ValueError:
                continue  # 半个 JSON / 混淆填充行，忽略
            choice = (obj.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            piece = delta.get("content")
            if piece:
                content_parts.append(piece)
                emitted = True
                on_delta(piece)
            for frag in (delta.get("tool_calls") or []):
                _merge_tool_call_fragment(tool_acc, frag)
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
    except requests.RequestException as e:
        logger.warning("[LLM] %s 流式请求异常: %s", scheme.backend, e)
        return None, emitted
    finally:
        if resp is not None:
            resp.close()  # 归还连接：非 200 早退、[DONE] 提前 break、异常等各路径都释放

    message: dict = {"role": "assistant"}
    tool_calls = [tool_acc[i] for i in sorted(tool_acc)]
    if tool_calls:
        message["tool_calls"] = tool_calls
    message["content"] = "".join(content_parts) or (None if tool_calls else "")
    if not ("".join(content_parts).strip() or tool_calls):
        return None, emitted  # 全空：视为失败（未外发则可重试）
    return {"choices": [{"message": message, "finish_reason": finish_reason}]}, emitted


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
    max_attempts: int | None = None,
    on_delta=None,
) -> dict | None:
    """单次请求单个后端（含退避重试，默认 3 次）；成功返回 data dict，失败返回 None。

    传入 on_delta 时走 SSE 流式：正文增量实时回调外发。流式下的重试仅在「尚未外发任何
    内容」前允许——一旦开始外发就不再重试，避免前端出现重复文字（宁可返回已收部分）。
    """
    if not scheme.key or not scheme.url:
        logger.warning("[LLM] 方案 %s 未配置 url/key，跳过。", scheme.backend)
        return None
    attempts = max_attempts or len(LLM_RETRY_BACKOFF_SECONDS)
    if on_delta is not None:
        for attempt in range(attempts):
            data, emitted = _stream_once(
                scheme, model, messages=messages, tools=tools,
                max_tokens=max_tokens, temperature=temperature,
                enable_thinking=enable_thinking, thinking_budget=thinking_budget,
                on_delta=on_delta, timeout=timeout,
            )
            if data is not None:
                return data
            if emitted:
                return None  # 已外发过内容，不能重试
            logger.warning("[LLM] %s 流式空响应/失败, attempt=%d", scheme.backend, attempt + 1)
            if attempt < attempts - 1:
                time.sleep(LLM_RETRY_BACKOFF_SECONDS[min(attempt, len(LLM_RETRY_BACKOFF_SECONDS) - 1)])
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
    for attempt in range(attempts):
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
        if attempt < attempts - 1:
            time.sleep(LLM_RETRY_BACKOFF_SECONDS[min(attempt, len(LLM_RETRY_BACKOFF_SECONDS) - 1)])
    return None


class LLMClient:
    """多后端 LLM 客户端。

    - 指定 model_id（网页选模型）：硬切换，只用该模型，失败即返回 None（不回退）。
    - 未指定 model_id（内部调用如摘要/描述）：按列表顺序逐个回退，保留韧性。
    - lite 调用可指定 lite_id（"平台id:子模型名"）：子模型池跨平台共享，
      用子模型所属平台的 url/key/backend 发请求，与主模型选择解耦。
    """

    def __init__(self, schemes: list[LLMScheme]) -> None:
        self.schemes = list(schemes)

    def usable(self) -> list[LLMScheme]:
        """已配置（有 key 和 url）的模型。"""
        return [s for s in self.schemes if s.key and s.url]

    def resolve_lite(self, lite_id: str) -> tuple[LLMScheme, str] | None:
        """解析 "平台id:子模型名" 为 (所属平台 scheme, 子模型名)；无效返回 None。"""
        pid, _, name = lite_id.partition(":")
        sel = next((s for s in self.usable() if s.id == pid), None)
        if sel is None or name not in sel.lite_models:
            return None
        return sel, name

    def _lite_order(self, preferred: "LLMScheme | None") -> list[tuple["LLMScheme", str]]:
        """lite「自动」的候选顺序：优先主模型所在平台的子模型，其余按 config 顺序借其它
        平台的子模型。只用真正配了子模型的平台（各取其首个子模型），不退回主模型；
        仅当所有可用平台都没配子模型时，才最终退回主模型以保证还能调通。
        （usable() 已过滤无 key 的；config 已剔除 enabled=0 的，故这里天然只含开启且可用的平台。）
        """
        usable = self.usable()
        ordered = (
            [preferred] + [s for s in usable if s.id != preferred.id]
            if preferred is not None else usable
        )
        subs = [(s, s.lite_models[0]) for s in ordered if s.lite_models]
        return subs or [(s, s.model) for s in ordered]

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
        max_attempts: int | None = None,
        on_delta=None,
    ) -> dict | None:
        usable = self.usable()
        # (scheme, 实际请求的模型名) 的尝试顺序
        if lite and lite_id:
            resolved = self.resolve_lite(lite_id)
            if resolved is None:
                logger.warning("[LLM] 指定子模型 %s 不可用（未配置或不存在）。", lite_id)
                return None
            order = [resolved]  # 硬选子模型：只用选中的，不回退
        elif lite:
            # lite「自动」：优先主模型平台的子模型，没有则按 config 顺序借其它平台的子模型
            preferred = next((s for s in usable if s.id == model_id), None) if model_id else None
            order = self._lite_order(preferred)
        elif model_id:
            sel = next((s for s in usable if s.id == model_id), None)
            if sel is None:
                logger.warning("[LLM] 指定模型 %s 不可用（未配置或不存在）。", model_id)
                return None
            order = [(sel, sel.model)]  # 主模型硬切换，不回退
        else:
            order = [(s, s.model) for s in usable]  # 主模型：内部调用按 config 顺序回退

        for scheme, model in order:
            data = _request_once(
                scheme,
                model,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget,
                timeout=timeout,
                max_attempts=max_attempts,
                on_delta=on_delta,
            )
            if data is not None:
                return data
            logger.warning("[LLM] 模型 %s(%s) 调用失败。", scheme.id or scheme.backend, model)
        return None

    def test_model(self, *, model_id: str | None = None, lite_id: str | None = None,
                   timeout: int = 20) -> bool:
        """轻量预检：对选中的主模型或子模型发一次极小请求，验证真实可用。"""
        # max_tokens 留足余量：reasoning 模型（如 azure gpt-5.x）思考也消耗输出配额，
        # 给太小会导致 content 为空而误判不可用。
        data = self.chat(
            [{"role": "user", "content": "ping"}],  # 最小探针：只求非空回复证明链路通，不关心内容
            lite=bool(lite_id),
            model_id=model_id,
            lite_id=lite_id,
            max_tokens=512,
            temperature=0.0,
            enable_thinking=False,
            timeout=timeout,
            max_attempts=1,  # 预检求快：单次尝试，不退避重试
        )
        return data is not None


def client_from_config() -> LLMClient:
    """从全局 config.LLM_MODELS 构造 LLMClient。"""
    from .. import config as cfg

    # 全部走 .get 带默认值：即使某条配置未经归一化 / 缺字段，也不会 KeyError。
    schemes = [
        LLMScheme(
            backend=m.get("backend", "openai"), url=m.get("url", ""), key=m.get("key", ""),
            model=m.get("model", ""), lite_models=m.get("lite_models", []),
            id=m.get("id", ""), label=m.get("label", ""),
        )
        for m in cfg.LLM_MODELS
    ]
    return LLMClient(schemes)


_shared: LLMClient | None = None


def get_client() -> LLMClient:
    """进程内共享的 LLMClient（无状态，可安全复用）；agent 与描述/浓缩等内部调用共用。"""
    global _shared
    if _shared is None:
        _shared = client_from_config()
    return _shared
