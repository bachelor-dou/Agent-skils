"""工具契约:一个参数只声明一次。

一个 `Param` 同时长出两样东西 —— `json_schema()` 给模型看,`coerce()` 做校验。分成两份写
的话它们会漂移,而漂移是**静默**的:校验那份认得 `"all"`、模型那份的 enum 里没有,于是
模型永远请求不到它,谁都不报错。

只有严格校验,没有「宽松模式」:静默把越界值裁到边界会让模型永远学不会自己传错了。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

REQUIRED = object()     # default 取这个值 = 必填


@dataclass(frozen=True)
class Param:
    name: str
    kind: str                       # int / str / bool / enum / list_str
    desc: str
    default: Any = REQUIRED
    min: int | None = None
    max: int | None = None
    choices: tuple[str, ...] = ()

    @property
    def required(self) -> bool:
        return self.default is REQUIRED

    def json_schema(self) -> dict:
        """给模型看的那份。"""
        types = {"int": "integer", "bool": "boolean", "list_str": "array", "enum": "string"}
        out: dict[str, Any] = {"type": types.get(self.kind, "string"),
                               "description": self.desc}
        if self.kind == "list_str":
            out["items"] = {"type": "string"}
        if self.choices:
            out["enum"] = list(self.choices)
        return out

    def coerce(self, value: Any) -> tuple[Any, str | None]:
        """校验一个值。返回 `(值, 出错原因)`,原因为 None 表示通过。

        不做静默纠偏:越界就报错,让模型看到 `must_be_lte_200` 并自己改。
        """
        if self.kind == "int":
            # bool 是 int 的子类,不拦的话 `top_n=true` 会变成 top_n=1
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None, "expected_integer"
            # json.loads 认 `1e400` 和 `NaN`,而 `int()` 对它们抛 OverflowError/ValueError。
            # 异常从这里逃出去会让那轮的 tool_calls 配不上 tool 回复,会话之后一律 400。
            if isinstance(value, float) and not math.isfinite(value):
                return None, "expected_integer"
            number = int(value)
            if self.min is not None and number < self.min:
                return None, f"must_be_gte_{self.min}"
            if self.max is not None and number > self.max:
                return None, f"must_be_lte_{self.max}"
            return number, None
        if self.kind == "bool":
            return (value, None) if isinstance(value, bool) else (None, "expected_boolean")
        if self.kind == "enum":
            return ((value, None) if value in self.choices
                    else (None, f"must_be_one_of_{list(self.choices)}"))
        if self.kind == "list_str":
            if not isinstance(value, list) or any(not isinstance(i, str) for i in value):
                return None, "expected_array_of_strings"
            return value, None
        return (value, None) if isinstance(value, str) else (None, "expected_string")


@dataclass(frozen=True)
class Tool:
    name: str
    desc: str
    run: Callable[[Any, dict], dict]        # (ctx, args) -> dict
    params: tuple[Param, ...] = ()
    expensive: bool = False                 # 昂贵工具执行前要用户确认

    def __post_init__(self) -> None:
        # 重名参数在 schema 里会静默合并成一条(后来者胜),校验时却两条都跑 —— 模型看到的
        # 定义和实际生效的规则就不是同一条了。拦在构造时。
        names = [p.name for p in self.params]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"{self.name} 的参数重名:{dupes}")

    def schema(self) -> dict:
        """LLM function-calling schema。"""
        fn: dict[str, Any] = {
            "name": self.name,
            "description": self.desc,
            "parameters": {"type": "object",
                           "properties": {p.name: p.json_schema() for p in self.params}},
        }
        if required := [p.name for p in self.params if p.required]:
            fn["parameters"]["required"] = required
        return {"type": "function", "function": fn}

    def validate(self, args: dict) -> tuple[dict, list[dict]]:
        """校验模型给的参数。返回 `(干净参数, 错误列表)`,错误为空才能执行。"""
        clean: dict[str, Any] = {}
        errors: list[dict] = []
        for param in self.params:
            if param.name not in args:
                if param.required:
                    errors.append({"param": param.name, "reason": "missing_required"})
                elif param.default is not None:
                    clean[param.name] = param.default
                continue
            value, reason = param.coerce(args[param.name])
            if reason:
                errors.append({"param": param.name, "reason": reason,
                               "received": args[param.name]})
            else:
                clean[param.name] = value

        # 未知参数显式拒绝:吞掉会让模型以为幻觉参数生效了,然后一直这么调下去。
        known = {p.name for p in self.params}
        errors += [{"param": name, "reason": "unknown_parameter", "received": args[name]}
                   for name in sorted(set(args) - known)]
        return clean, errors


@dataclass
class Ctx:
    """一次工具调用能看到的全部外部世界。

    **刻意没有 `db` 字段**:DB 挂在 ctx 上供人就地改,「谁改了什么、什么时候落盘」就没人
    说得清了。要读的工具自己 `universe.load()`,要写的走 `universe.write_*` —— 每次写都是
    一个事务,范围写在函数名里。
    """

    gh: Any = None                      # provider.github 的同步门面;None = 不联网的场景
    state: Any = None                   # 会话状态,只有 agent 路径有
    user_id: str = ""
    progress: Callable[[int, str], None] | None = None


class Registry:
    """工具表。名字 → 工具,分发只此一处。"""

    def __init__(self, tools: Sequence[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.add(tool)

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具重名:{tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)
