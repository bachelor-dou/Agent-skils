"""Agent 状态模型。

从 agent.py 抽离路由/执行共享状态定义，减少单文件复杂度。
"""

import json
from dataclasses import dataclass, field

from .common.async_token_pool import GitHubTokenPool
from .common.config import GROWTH_CALC_DAYS, MIN_STAR, STAR_GROWTH_THRESHOLD
from .common.db import load_db

# 榜单型意图集合：需要完整候选收集→增长计算→排名流程的意图类型
_RANKING_INTENTS = {"comprehensive_ranking", "hot_new_ranking", "keyword_ranking"}

# 目前受工具约束的意图集合（仅用于上下文文本标记）
_CONSTRAINED_INTENTS = {"comprehensive_ranking", "hot_new_ranking", "keyword_ranking"}


@dataclass
class PendingRequest:
    """待确认请求：路由解析阶段的中间状态。"""

    turn_kind: str = "unknown"
    intent_family: str = "unknown"
    intent_label_zh: str = "未确定请求"
    target_repo: str = ""
    user_specified_params: dict[str, object] = field(default_factory=dict)
    unresolved_constraints: list[str] = field(default_factory=list)
    ambiguous_fields: list[str] = field(default_factory=list)
    suggested_tools: list[str] = field(default_factory=list)
    route_confidence: str = "medium"
    confirmation_text_zh: str = ""
    report_requested: bool = False
    should_execute_now: bool = False
    must_call_tool_before_reply: bool = False
    source_turn_id: int = 0

    def to_dict(self) -> dict[str, object]:
        """转换为字典格式，用于日志输出和上下文传递。"""
        return {
            "turn_kind": self.turn_kind,
            "intent_family": self.intent_family,
            "intent_label_zh": self.intent_label_zh,
            "target_repo": self.target_repo,
            "specified_params": self.user_specified_params,
            "unresolved_constraints": self.unresolved_constraints,
            "ambiguous_fields": self.ambiguous_fields,
            "suggested_tools": self.suggested_tools,
            "route_confidence": self.route_confidence,
            "report_requested": self.report_requested,
            "should_execute_now": self.should_execute_now,
            "must_call_tool_before_reply": self.must_call_tool_before_reply,
            "confirmation_text_zh": self.confirmation_text_zh,
            "source_turn_id": self.source_turn_id,
        }


@dataclass
class ResolvedRequest:
    """已确认请求：路由解析完成后的冻结执行参数。"""

    turn_kind: str = "unknown"
    intent_family: str = "unknown"
    intent_label_zh: str = "未确定请求"
    target_repo: str = ""
    resolved_params: dict[str, object] = field(default_factory=dict)
    user_specified_params: dict[str, object] = field(default_factory=dict)
    defaulted_params: dict[str, object] = field(default_factory=dict)
    suggested_tools: list[str] = field(default_factory=list)
    route_confidence: str = "medium"
    report_requested: bool = False
    must_call_tool_before_reply: bool = False
    confirmation_text_zh: str = ""

    def requires_full_collection(self) -> bool:
        """判断是否为榜单型意图，需要完整的候选收集流程。"""
        return self.intent_family in _RANKING_INTENTS

    def to_dict(self) -> dict[str, object]:
        """转换为字典格式，用于日志输出和上下文传递。"""
        return {
            "turn_kind": self.turn_kind,
            "intent_family": self.intent_family,
            "intent_label_zh": self.intent_label_zh,
            "target_repo": self.target_repo,
            "resolved_params": self.resolved_params,
            "user_specified_params": self.user_specified_params,
            "defaulted_params": self.defaulted_params,
            "suggested_tools": self.suggested_tools,
            "route_confidence": self.route_confidence,
            "report_requested": self.report_requested,
            "must_call_tool_before_reply": self.must_call_tool_before_reply,
            "confirmation_text_zh": self.confirmation_text_zh,
        }

    def to_execution_context(self) -> str:
        """生成执行上下文文本，注入到 system prompt 中指导 LLM 执行。"""
        tool_constraint = "constrained" if self.intent_family in _CONSTRAINED_INTENTS else "open"
        lines = [
            "[已确认请求]",
            f"turn_kind={self.turn_kind}",
            f"intent_family={self.intent_family}",
            f"intent_label_zh={self.intent_label_zh}",
            f"target_repo={self.target_repo or '未指定'}",
            f"route_confidence={self.route_confidence}",
            f"must_call_tool_before_reply={self.must_call_tool_before_reply}",
            f"tool_constraint={tool_constraint}",
            f"suggested_tools={self.suggested_tools}",
            f"resolved_params={json.dumps(self.resolved_params, ensure_ascii=False, sort_keys=True)}",
        ]
        return "\n".join(lines)


@dataclass
class AgentState:
    """Agent 运行时状态，在整个会话期间保持。"""

    token_mgr: GitHubTokenPool = field(default_factory=GitHubTokenPool)
    db: dict = field(default_factory=dict)
    conversation: list[dict] = field(default_factory=list)

    last_search_repos: list[dict] = field(default_factory=list)
    last_candidates: dict[str, dict] = field(default_factory=dict)
    last_candidate_days_since_created: int | None = None
    last_ranked: list[tuple[str, dict]] = field(default_factory=list)
    last_mode: str = "comprehensive"
    last_growth_calc_days: int = GROWTH_CALC_DAYS
    last_growth_threshold: int = STAR_GROWTH_THRESHOLD
    last_min_star: int = MIN_STAR
    seen_repos: set[str] = field(default_factory=set)

    current_user_turn: int = 0
    discovery_turn_id: int | None = None
    awaiting_confirmation: bool = False
    pending_request: PendingRequest | None = None
    last_confirmed_request: ResolvedRequest | None = None
    current_turn_tools: set[str] = field(default_factory=set)
    current_turn_tool_call_count: int = 0
    current_turn_requires_tool_call: bool = False
    active_repo: str | None = None
    recent_verified_claims: list[dict[str, object]] = field(default_factory=list)

    conversation_summary: str = ""

    def __post_init__(self):
        if not self.db:
            self.db = load_db()


# 对话历史压缩参数：超过上限时触发 LLM 摘要压缩
MAX_CONVERSATION_MESSAGES = 40
KEEP_RECENT_MESSAGES = 10


__all__ = [
    "PendingRequest",
    "ResolvedRequest",
    "AgentState",
    "MAX_CONVERSATION_MESSAGES",
    "KEEP_RECENT_MESSAGES",
]
