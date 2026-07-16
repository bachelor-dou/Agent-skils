"""agent 包：精简 ReAct Agent 与工厂。"""

from .agent import HotProjectAgent, ToolContext
from .state import AgentState

__all__ = ["HotProjectAgent", "ToolContext", "AgentState", "build_agent"]


def build_agent() -> HotProjectAgent:
    """组装生产用 Agent：A/B LLM 客户端 + 默认注册表 + GitHubProvider + DB。"""
    from ..infra.llm_client import client_from_config
    from ..infra.db import load_db
    from ..datasource.github.token_pool import GitHubTokenPool
    from ..datasource.github.provider import GitHubProvider
    from ..tools.registry import build_default_registry

    token_mgr = GitHubTokenPool()
    db = load_db()
    return HotProjectAgent(
        llm=client_from_config(),
        registry=build_default_registry(),
        provider=GitHubProvider(token_mgr),
        db=db,
    )
