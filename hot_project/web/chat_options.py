"""一轮对话的选项 —— HTTP body 和 WS query 收敛成同一组干净值。

两条传输各自解析、各自兜默认值,是这条链上重复最多的地方:加一个选项要动两处 Python
加两处 JS。更糟的是默认值曾经在 Agent 和 LLMClient 各有一套,而空档位在 azure 上等于
不思考 —— 回答只是变差,没有任何报错。

这里是那组选项唯一的主人。线上字段名保持 `thinking` 不动前端,包内一律叫 `effort`:
翻译只发生在这一处。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

from ..infra import llm


@dataclass(frozen=True)
class ChatOptions:
    """归一化之后的选项。字段名有意和 `Agent.chat` 的关键字参数一一对应,`kwargs()`
    才能直接展开 —— 改这里的字段名就要同时改那边的签名。
    """

    user_id: str = ""
    model: str | None = None
    lite: str | None = None
    effort: str = llm.EFFORT_DEFAULT

    def kwargs(self) -> dict:
        return asdict(self)


def parse(source: Mapping) -> ChatOptions:
    """从 HTTP body(`model_dump()`)或 WS 的 query 解析,两边同一套规矩。

    缺键、空串、错字都落到安全值:模型和子模型落成 None(=跟随默认平台),档位落回默认档。
    空串在这里就消掉,不让它流到下游 —— 它是传输层的产物(用户没点过那个开关),
    不该出现在 LLM 的接口上。
    """
    return ChatOptions(
        user_id=str(source.get("user_id") or ""),
        model=str(source.get("model") or "") or None,
        lite=str(source.get("lite") or "") or None,
        effort=llm.level(str(source.get("thinking") or "")),
    )
