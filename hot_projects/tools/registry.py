"""工具注册表：单点注册 + 单一分发。

新增工具/任务类型 = 加一条 ToolSpec 注册项，无需散点改代码。
"""

from dataclasses import dataclass
from typing import Callable


@dataclass
class ToolSpec:
    name: str
    schema: dict          # LLM function-calling schema
    handler: Callable      # (ctx, args) -> dict
    expensive: bool = False  # 昂贵工具（榜单）：执行前需用户确认


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict]:
        return [t.schema for t in self._tools.values()]

    def dispatch(self, name: str, ctx, args: dict):
        spec = self._tools.get(name)
        if spec is None:
            return {"error": f"未知 Tool: {name}"}
        return spec.handler(ctx, args)


def build_default_registry() -> ToolRegistry:
    """组装默认注册表：3 个复合榜单工具（昂贵）+ 4 个原子工具。"""
    from .schemas import AGENT_TOOL_SCHEMAS
    from .ranking_tools import make_ranking_handler
    from .atomic_tools import (
        repo_growth_handler, describe_project_handler,
        get_db_info_handler, fetch_trending_handler,
    )

    schema_by_name = {s["function"]["name"]: s for s in AGENT_TOOL_SCHEMAS}
    handlers = {
        "comprehensive_ranking": (make_ranking_handler("comprehensive"), True),
        "hot_new_ranking": (make_ranking_handler("hot_new"), True),
        "keyword_ranking": (make_ranking_handler("keyword"), True),
        "repo_growth": (repo_growth_handler, False),
        "describe_project": (describe_project_handler, False),
        "get_db_info": (get_db_info_handler, False),
        "fetch_trending": (fetch_trending_handler, False),
    }

    reg = ToolRegistry()
    for name, (handler, expensive) in handlers.items():
        reg.register(ToolSpec(name=name, schema=schema_by_name[name],
                              handler=handler, expensive=expensive))
    return reg
