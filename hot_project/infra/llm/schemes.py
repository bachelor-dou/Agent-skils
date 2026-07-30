"""模型目录 → 可调用的方案表。

`config.LLM_MODELS` 是手写的**声明**:字段可能缺、`enabled` 可能写成 `"0"`、key 只记了
环境变量名。这个模块把它变成下游可以无条件依赖的结构,好让客户端里不再出现一行
`m.get("backend", "openai")`。

**归一化在这里而不在 config 里。** 旧 `config.py:153` 为了归一化反过来
`from .infra.llm_client import normalize_models` —— 配置依赖实现,循环 import 的温床
(而且 import 顺序一变就炸,炸的方式还很难看:`AttributeError: partially initialized module`)。
现在方向单一:config 只声明,infra 读它。

**key 在这里才落地。** 目录里只有 `key_env`,真值到构造方案时才从环境变量取。这样
`config.LLM_MODELS` 整个可以安全地打日志、进 `/api` 响应、被异常回溯带出去。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...common.env import truthy


@dataclass
class Scheme:
    """一个可以直接拿去发请求的平台。

    `key` 在这里是明文,所以 `Scheme` **不该**被整体序列化或打日志。
    需要给前端列模型时用 `public()`。
    """

    id: str
    label: str
    backend: str
    url: str
    model: str
    key: str
    lite_models: list[str] = field(default_factory=list)
    desc: str = ""

    @property
    def usable(self) -> bool:
        """没有 key 或没有 url 就发不出请求 —— 跳过它,而不是发一个注定 401 的请求。"""
        return bool(self.key and self.url)

    def public(self) -> dict:
        """给前端的视图:没有 key。"""
        return {
            "id": self.id, "label": self.label, "model": self.model,
            "lite_models": list(self.lite_models), "desc": self.desc,
        }


def _lite_names(raw: object) -> list[str]:
    """`"a, b, a"` → `["a", "b"]`。平台内去重,顺序保留(第一个是「自动」时的首选)。

    跨平台的融合去重不在这里做:那是前端展示的事,混进来会污染内部回退顺序。
    """
    seen: set[str] = set()
    out: list[str] = []
    for name in (part.strip() for part in str(raw or "").split(",")):
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def build(catalog: list[dict], resolve_key) -> list[Scheme]:
    """模型目录 → 方案表。`resolve_key(env_name) -> str` 负责把机密取进来。

    `enabled=0` 的条目直接不出现在结果里 —— 关掉的平台既不该被前端选中,也不该在内部
    回退时被试到。缺 `id` 的跳过(没有 id 就无法被选择,留着也用不上);id 撞车直接报错,
    因为它是整条选择链路的键,重复会让「选 A 却调到 B」这种事静默发生。
    """
    out: list[Scheme] = []
    seen: set[str] = set()
    for entry in catalog:
        if not truthy(entry.get("enabled", True)):
            continue
        mid = str(entry.get("id") or "").strip()
        if not mid:
            continue
        if mid in seen:
            raise ValueError(
                f"LLM_MODELS 里 id 重复:{mid!r}。id 是选择链路的键,按平台命名"
                f"(azure01 / aliyun02),不要按模型名命名。"
            )
        seen.add(mid)
        out.append(Scheme(
            id=mid,
            label=str(entry.get("label") or mid),
            backend=str(entry.get("backend") or "openai"),
            url=str(entry.get("url") or ""),
            model=str(entry.get("model") or ""),
            key=resolve_key(str(entry.get("key_env") or "")),
            lite_models=_lite_names(entry.get("lite_model")),
            desc=str(entry.get("desc") or ""),
        ))
    return out
