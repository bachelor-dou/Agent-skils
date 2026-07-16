"""Agent 会话状态（精简版）。"""

from dataclasses import dataclass, field

# 对话历史压缩参数
MAX_CONVERSATION_MESSAGES = 40
KEEP_RECENT_MESSAGES = 10


@dataclass
class AgentState:
    db: dict = field(default_factory=dict)
    conversation: list[dict] = field(default_factory=list)
    conversation_summary: str = ""
    # 工具私有的会话级状态槽：具体工具自行读写（如 ranking 的 RankingCache），
    # agent 层与具体工具解耦、不感知内容。
    tool_state: dict = field(default_factory=dict)
    active_repo: str | None = None
    pending_confirmation_signature: str | None = None
