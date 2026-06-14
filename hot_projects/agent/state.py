"""Agent 会话状态（精简版）。"""

from dataclasses import dataclass, field

from ..pipeline.cache import RankingCache

# 对话历史压缩参数
MAX_CONVERSATION_MESSAGES = 40
KEEP_RECENT_MESSAGES = 10


@dataclass
class AgentState:
    db: dict = field(default_factory=dict)
    conversation: list[dict] = field(default_factory=list)
    conversation_summary: str = ""
    ranking_cache: RankingCache = field(default_factory=RankingCache)
    active_repo: str | None = None
    pending_confirmation_signature: str | None = None
