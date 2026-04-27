"""
ReAct Agent 核心
================
实现 Thought → Action → Observation 循环的自主推理 Agent。

Agent 接收自然语言指令，通过 LLM 自主规划步骤，
调用 Tool 执行操作，观察结果后决定下一步行动，
直到得出最终回复。

架构分层：
  - parsing/   — 输入解析层：意图识别、参数提取、Tool 参数规范化
  - agent.py   — Agent 层：ReAct 循环、Tool 路由、状态管理
    - scheduled_update.py — 批处理入口：内置 DiscoveryPipeline 编排

核心类：
  - HotProjectAgent: ReAct Agent 主体
  - AgentState:      Agent 运行状态（会话历史、候选缓存、DB 等）
"""

import json
import logging
from dataclasses import dataclass, field

import requests

from .common.config import (
    LLM_API_KEY,
    LLM_API_URL,
    LLM_MODEL,
    MIN_STAR,
    MAX_STAR,
    GROWTH_CALC_DAYS,
    STAR_GROWTH_THRESHOLD,
    HOT_PROJECT_COUNT,
    HOT_NEW_PROJECT_COUNT,
    DAYS_SINCE_CREATED,
)
from .agent_tools import (
    tool_search_by_keywords,
    tool_scan_star_range,
    tool_check_repo_growth,
    tool_batch_check_growth,
    tool_rank_candidates,
    tool_describe_project,
    tool_generate_report,
    tool_get_db_info,
    tool_fetch_trending,
    trending_repo_to_search_repo,
)
from .common.db import load_db, save_db_desc_only, get_db_age_days
from .common.token_manager import TokenManager
from .parsing.arg_validator import (
    validate_tool_args,
    validate_tool_args_strict,
    log_validated_params,
)
from .parsing.route_helpers import (
    extract_json_object,
    looks_like_structured_confirmation_text,
    normalize_intent_family,
    normalize_specified_params,
    normalize_tool_names,
    normalize_turn_kind,
    ordered_tool_names,
    sanitize_confirmation_fallback,
)
from .parsing.schema import TOOL_PARAM_SCHEMA, TOOL_SCHEMAS

logger = logging.getLogger("discover_hot")

# Agent 单轮最大 Tool 调用次数（防止无限循环）
MAX_TOOL_CALLS_PER_TURN = 15

# ──────────────────────────────────────────────────────────────
# 工具名称索引：从 TOOL_SCHEMAS 提取，用于快速查找/校验工具名
# ──────────────────────────────────────────────────────────────
ALL_TOOL_NAMES = [
    schema.get("function", {}).get("name")
    for schema in TOOL_SCHEMAS
    if schema.get("function", {}).get("name")
]
TOOL_SCHEMA_NAME_SET = set(ALL_TOOL_NAMES)
TOOL_SCHEMA_BY_NAME = {
    schema["function"]["name"]: schema
    for schema in TOOL_SCHEMAS
    if schema.get("function", {}).get("name")
}

# ──────────────────────────────────────────────────────────────
# 参数键索引：从 TOOL_PARAM_SCHEMA 提取全部参数名，
# 排序后拼接成文本，嵌入到路由提示词中告知 LLM 允许的参数键
# ──────────────────────────────────────────────────────────────
ALL_TOOL_PARAM_NAMES = {
    param_name
    for schema in TOOL_PARAM_SCHEMA.values()
    for param_name in schema.keys()
}
CANONICAL_ROUTE_PARAM_KEYS = sorted(ALL_TOOL_PARAM_NAMES)
CANONICAL_ROUTE_PARAM_KEYS_TEXT = ", ".join(CANONICAL_ROUTE_PARAM_KEYS)

# ══════════════════════════════════════════════════════════════════════════════════════
# 提示词（意图分类阶段）
# ══════════════════════════════════════════════════════════════════════════════════════

CONFIRMATION_PROMPT = f"""你是 GitHub 热门项目助手的轻量路由器，任务是准确识别用户输入的意图，并只输出结构化 JSON。

目标：
1) 识别用户意图（intent_family）
2) 抽取用户指定的明确参数（specified_params）
3) 存在关键歧义时返回 ambiguous_fields,
4) 给出 suggested_tools（优先建议，不是强制流程）

意图类型（选最匹配的一项）：
- hot_new_ranking：新项目的热榜查询（创建时间过滤）
- comprehensive_ranking：综合热榜查询
- keyword_ranking：按关键词的热榜查询
- trending_only：查看 Trending
- repo_info：单仓库综合查询（默认：描述+增长）
- repo_growth：仅增长
- repo_description：仅介绍仓库
- db_info：本地 DB 查询
- freeform_answer：基于上下文的解释/比较/质疑回复（可不调用工具）
- unknown：无法稳定归类

turn_kind（消息类型，选以下最匹配的一项）：
- new_request：全新的独立请求，与上一轮无直接关联（如首次查询、换话题）
- request_modification：对上一轮请求的参数调整（如"改成 top 10"、"换个关键词"）
- clarification_answer：用户回答上一轮的澄清问题（如"综合榜"、"7天窗口")
- execution_ack：用户确认执行（如"开始"、"是的"、"确认执行")
- fact_check：针对具体事实的核查请求（如"langchain 近7天增长多少")
- capability_query：询问助手能力范围（如"你能做什么"、"支持哪些查询")
- greeting：问候或寒暄（如"你好"、"在吗")
- unknown：无法稳定归类

参数语义最小规则：
- “近N天热榜/增长”优先映射为 growth_calc_days（增长统计窗口）
- “近N天内创建的新项目”映射为 days_since_created（创建时间窗口）
- repo_info 默认是综合查询；仅当用户明确说”只看增长/只看介绍”时改为单一意图
- specified_params 只能使用 canonical key，不允许自造参数名。
- canonical key 列表：{CANONICAL_ROUTE_PARAM_KEYS_TEXT}
- 对无法映射的参数表达，放入 unresolved_constraints，不要放进 specified_params。
- 当 unresolved_constraints 非空时，必须给出 confirmation_text_zh（自然语言澄清问题）。

输出 JSON（仅以下字段，不要额外字段）：
{{
        “turn_kind”: “...”,
        “intent_family”: “...”,
        “intent_label_zh”: “...”,
        “target_repo”: “owner/repo 或空字符串”,
        “specified_params”: {{}},
        “unresolved_constraints”: [],
        “ambiguous_fields”: [],
        "suggested_tools": ["tool_name"],
        "route_confidence": "high|medium|low",
        "report_requested": false,
        "should_execute_now": true,
        "must_call_tool_before_reply": false,
        "confirmation_text_zh": "仅在需要澄清时填写自然语言"
}}

约束：
- 如果 ambiguous_fields 为空，should_execute_now 应为 true
- fact_check 默认 should_execute_now=true 且 must_call_tool_before_reply=true
- confirmation_text_zh 不能是 JSON
- route_confidence 判断规则：
  - high：用户意图清晰，参数完整，无歧义
  - medium：能理解意图但部分参数不确定（默认值）
  - low：无法稳定理解用户输入，意图模糊或超出能力范围，需要用户重述
"""

# ──────────────────────────────────────────────────────────────
# 对话轮次类型白名单（用于规范化路由输出的 turn_kind）
# ──────────────────────────────────────────────────────────────
TURN_KINDS = {
        "new_request",
        "request_modification",
        "clarification_answer",
        "execution_ack",
        "fact_check",
        "capability_query",
        "greeting",
        "unknown",
}

# ══════════════════════════════════════════════════════════════════════════════════════
# 意图类型 + 工具映射（混合架构核心）
# ══════════════════════════════════════════════════════════════════════════════════════

# 意图中文标签（用于用户确认文本、日志输出与路由标准化白名单）。
# 注意：这不是“工具约束集合”。
# 即便当前仅榜单类做工具约束，trending/repo/db 等意图标签仍需保留，
# 否则这些请求会在 normalize_intent_family 阶段退化为 unknown。
INTENT_LABELS = {
    "comprehensive_ranking": "综合热榜",
    "hot_new_ranking": "新项目热榜",
    "keyword_ranking": "关键词热榜",       # 新增：只搜索指定关键词/类别
    "trending_only": "Trending热门",
    "repo_info": "单仓库综合查询",
    "repo_growth": "单仓库增长",
    "repo_description": "项目介绍",
    "db_info": "数据库查询",
    "freeform_answer": "自由回答",
    "unknown": "未确定请求",
}

# 意图别名映射：将简写/常见写法映射到标准意图名
INTENT_ALIASES = {
    "comprehensive": "comprehensive_ranking",
    "hot_new": "hot_new_ranking",
    "keyword": "keyword_ranking",
    "trending": "trending_only",
    "describe_project": "repo_description",
    "check_repo_growth": "repo_growth",
    "freeform": "freeform_answer",
}

# 仅对复杂榜单意图做工具约束；其他意图默认开放
CONSTRAINED_TOOLS_BY_INTENT: dict[str, set[str]] = {
    "comprehensive_ranking": {
        "search_by_keywords", "scan_star_range", "fetch_trending",
        "batch_check_growth", "rank_candidates", "generate_report",
    },
    "hot_new_ranking": {
        "search_by_keywords", "scan_star_range", "fetch_trending",
        "batch_check_growth", "rank_candidates", "generate_report",
    },
    "keyword_ranking": {
        "search_by_keywords",                 # 只需要关键词搜索
        "batch_check_growth", "rank_candidates",
    },
}

# 榜单型意图集合：需要完整候选收集→增长计算→排名流程的意图类型
RANKING_INTENTS = {"comprehensive_ranking", "hot_new_ranking", "keyword_ranking"}

# 建议的候选收集工具（仅作日志提示，不强制阻断执行）
SUGGESTED_COLLECTION_TOOLS_BY_INTENT: dict[str, set[str]] = {
    "comprehensive_ranking": {"search_by_keywords", "scan_star_range", "fetch_trending"},
    "hot_new_ranking": {"search_by_keywords", "scan_star_range", "fetch_trending"},
    "keyword_ranking": {"search_by_keywords"},
}


# 参数渲染器：将参数键值转换为用户可见的确认文本片段
PARAM_DISPLAYERS = {
    "categories": lambda v: f"关注方向为{'、'.join(v)}" if isinstance(v, list) and v else None,
    "min_star": lambda v: f"项目最低star为{v}",
    "growth_calc_days": lambda v: f"统计近{v}天的增长",
    "days_since_created": lambda v: f"只看近{v}天内创建的项目",
    "growth_threshold": lambda v: f"增长门槛为{v}",
    "top_n": lambda v: f"返回前{v}名",
    "trending_range": lambda v: f"Trending范围:{v}",
    "repo": lambda v: f"仓库为{v}" if v else None,
}


# ══════════════════════════════════════════════════════════════════════════════════════
# 路由状态模型：两阶段分离架构的数据载体
# ══════════════════════════════════════════════════════════════════════════════════════
#
# ┌─────────────────────────────────────────────────────────────────────────────────┐
# │                        路由阶段 → 执行阶段 数据流                                │
# │                                                                                 │
# │  用户消息                                                                       │
# │      │                                                                          │
# │      ▼                                                                          │
# │  _build_route_pending_request()                                                 │
# │      │ 调用路由 LLM                                                             │
# │      ▼                                                                          │
# │  ┌─────────────────────┐         ┌──────────────────────┐                       │
# │  │   PendingRequest    │ ──────► │   ResolvedRequest    │                       │
# │  │   (待确认状态)       │ 确认后  │   (已冻结参数)       │                       │
# │  │                     │         │                      │                       │
# │  │ - intent_family     │         │ - resolved_params    │                       │
# │  │ - user_params       │         │   (用户+默认合并)    │                       │
# │  │ - ambiguous_fields  │         │ - suggested_tools    │                       │
# │  │ - should_execute_now│         │                      │                       │
# │  └─────────────────────┘         └──────────────────────┘                       │
# │                                          │                                      │
# │                                          ▼                                      │
# │                                    Tool 执行阶段                                │
# │                                    (参数注入 + 工具调用)                         │
# └─────────────────────────────────────────────────────────────────────────────────┘
#
# PendingRequest：路由 LLM 的原始输出，可能包含歧义
# ResolvedRequest：参数已合并冻结，准备执行
#
# ══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class PendingRequest:
    """
    待确认请求：路由解析阶段的中间状态

    生命周期：
      1. 路由 LLM 解析用户消息 → 输出 PendingRequest
      2. 若有歧义（ambiguous_fields 非空）→ 等待用户澄清
      3. 用户确认后 → 转换为 ResolvedRequest（参数冻结）

    核心字段：
      - turn_kind: 轮次类型（new_request/follow_up/fact_check/...）
      - intent_family: 意图类型（comprehensive_ranking/keyword_ranking/...）
      - user_specified_params: 用户明确指定的参数
      - ambiguous_fields: 有歧义需要澄清的字段
      - should_execute_now: 是否可以直接执行（无歧义时为 True）

    注意：
      - 参数未合并默认值，只存储用户原始指定
      - 状态可变，等待用户澄清或确认
    """
    turn_kind: str = "unknown"  # 轮次类型：new_query/follow_up/fact_check/clarification
    intent_family: str = "unknown"  # 意图：comprehensive_ranking/keyword_ranking/repo_analysis 等
    intent_label_zh: str = "未确定请求"  # 意图的中文显示名称，用于生成确认文本
    target_repo: str = ""  # 目标仓库，如 "facebook/react"
    user_specified_params: dict[str, object] = field(default_factory=dict)  # 用户明确指定的参数（growth_calc_days/language 等）
    unresolved_constraints: list[str] = field(default_factory=list)  # 无法解析的约束，如 "无法映射参数名: xyz"
    ambiguous_fields: list[str] = field(default_factory=list)  # 存在歧义的字段，需要用户澄清
    suggested_tools: list[str] = field(default_factory=list)  # 路由建议调用的工具列表
    route_confidence: str = "medium"  # 路由置信度：high/medium/low
    confirmation_text_zh: str = ""  # 向用户展示的确认/澄清文本
    report_requested: bool = False  # 用户是否请求生成报告
    should_execute_now: bool = False  # 是否可以立即执行（无歧义时为 True）
    must_call_tool_before_reply: bool = False  # 是否必须在回复前调用工具获取数据
    source_turn_id: int = 0  # 来源轮次 ID，用于追踪请求来源

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
    """
    已确认请求：路由解析完成后的冻结执行参数

    生命周期：
      1. PendingRequest 用户确认后 → 调用 _resolve_pending_request()
      2. 合并用户参数 + 意图默认参数 → ResolvedRequest
      3. 注入到执行阶段（_call_llm 的 system prompt）

    核心字段：
      - resolved_params: 最终执行参数（用户参数 + 默认参数）
      - user_specified_params: 用户原始指定（用于区分参数来源）
      - defaulted_params: 默认填充的参数（用于区分参数来源）
      - suggested_tools: 建议调用的工具列表

    特点：
      - 参数已冻结，不再变化
      - 无歧义，可直接执行
      - 用于驱动 Tool 执行阶段的参数注入
    """
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
        return self.intent_family in RANKING_INTENTS

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
        tool_constraint = "constrained" if self.intent_family in CONSTRAINED_TOOLS_BY_INTENT else "open"
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

# ══════════════════════════════════════════════════════════════════════════════════════
# SYSTEM_PROMPT（LLM自主决策阶段 - 工具详情已通过API tools参数传递）
# ══════════════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = f"""你是GitHub热门项目发现助手。根据用户需求自主调用提供的工具来完成任务。
你以 ReAct 方式工作：先理解问题，再决定是否调用工具，基于观察继续决策，最后给出结论。

规则：
1. 涉及事实数据（star、增长、创建时间、Trending）时，不要编造，优先调用工具核查。
2. 路由提示是“优先建议”，不是硬限制；如果证据不足或工具报错，可调整工具选择。
3. 单仓库默认优先综合查询（describe_project + check_repo_growth）；用户明确“只看增长/只看介绍”时再单工具。
4. growth_calc_days=增长统计窗口；days_since_created=创建时间窗口，两者可同时存在且互不覆盖。
5. 工具返回参数错误时，先修正后重试一次；仍失败再向用户澄清。
6. 用户做解释/比较/质疑追问时，可直接回答；必要时再做最小化取证。

"""


@dataclass
class AgentState:
    """
    Agent 运行时状态，在整个会话期间保持。

    包含三类数据：
      - 基础设施：Token 管理器、DB、对话历史
      - 流程缓存：搜索结果、候选列表、排序结果（供多个 Tool 间共享）
      - 路由状态：确认门控、已解析请求、活跃仓库等
    """
    # ─── 基础设施 ───
    token_mgr: TokenManager = field(default_factory=TokenManager)
    db: dict = field(default_factory=dict)
    conversation: list[dict] = field(default_factory=list)

    # ─── 流程缓存：榜单构建各阶段的中间结果 ───
    last_search_repos: list[dict] = field(default_factory=list)  # 搜索阶段收集的仓库
    last_candidates: dict[str, dict] = field(default_factory=dict)  # 增长筛选后的候选
    last_candidate_days_since_created: int | None = None  # 增长筛选时使用的新项目窗口
    last_ranked: list[tuple[str, dict]] = field(default_factory=list)  # 排序后的 Top N
    last_mode: str = "comprehensive"  # 当前榜单模式
    last_growth_calc_days: int = GROWTH_CALC_DAYS  # 当前增长统计窗口
    last_growth_threshold: int = STAR_GROWTH_THRESHOLD  # 当前增长阈值
    last_min_star: int = MIN_STAR  # 当前最低 star 过滤值
    seen_repos: set[str] = field(default_factory=set)  # 已扫描仓库（用于去重）

    # ─── 路由状态：确认门控和请求追踪 ───
    current_user_turn: int = 0  # 当前用户轮次计数
    discovery_turn_id: int | None = None  # 当前榜单构建轮次（用于重置缓存）
    awaiting_confirmation: bool = False  # 是否等待用户确认
    pending_request: PendingRequest | None = None  # 待确认请求
    last_confirmed_request: ResolvedRequest | None = None  # 已确认请求
    current_turn_tools: set[str] = field(default_factory=set)  # 本轮已调用的工具
    current_turn_tool_call_count: int = 0  # 本轮 Tool 调用计数
    current_turn_requires_tool_call: bool = False  # 本轮是否必须先调用 Tool
    active_repo: str | None = None  # 当前活跃仓库（用于追问上下文）
    recent_verified_claims: list[dict[str, object]] = field(default_factory=list)  # 近期核查的事实

    # ─── 对话记忆 ───
    conversation_summary: str = ""  # 早期对话的语义摘要（压缩后保留）

    def __post_init__(self):
        """初始化后自动加载 DB（若未提供）。"""
        if not self.db:
            self.db = load_db()

# ──────────────────────────────────────────────────────────────
# 对话历史压缩参数：超过上限时触发 LLM 摘要压缩
# ──────────────────────────────────────────────────────────────
MAX_CONVERSATION_MESSAGES = 40  # 触发压缩的消息数上限
KEEP_RECENT_MESSAGES = 10  # 压缩后保留的最近消息数


class HotProjectAgent:
    """
    ReAct Agent: 自主规划 + Tool 调用 + 多轮对话。

    使用方式::

        agent = HotProjectAgent()
        reply = agent.chat("帮我找最近 AI Agent 方向的热门项目")
        print(reply)
        reply = agent.chat("把增长阈值降到 500 再搜一次")
        print(reply)
    """

    def __init__(self) -> None:
        """初始化 Agent 状态并注入系统提示词。"""
        self.state = AgentState()
        # 初始化会话，加入 System Prompt
        self.state.conversation.append({
            "role": "system",
            "content": SYSTEM_PROMPT,
        })
        logger.info("HotProjectAgent 初始化完成。")

    def chat(self, user_message: str) -> str:
        """
        处理用户消息，返回 Agent 回复。

        内部执行 ReAct 循环：
          1. 如果对话历史过长，触发压缩
          2. 将用户消息加入会话历史
          3. 调用路由门控判断是否拦截
          4. 调用 LLM（带 Tool 定义）
          5. 如果 LLM 选择调用 Tool → 执行 Tool → 将结果加入历史 → 回到 4
          6. 如果 LLM 直接回复 → 返回回复文本
        """
        # ── 步骤1：对话历史压缩检查 ──────────────────────────────────────────────────────
        #    超过上限时触发 LLM 摘要压缩，保留最近消息
        if len(self.state.conversation) > MAX_CONVERSATION_MESSAGES:
            self._compress_conversation()

        # ── 步骤2：用户消息校验与追加 ─────────────────────────────────────────────────────
        #    过长消息拦截，正常消息追加到 conversation
        if len(user_message) > 2000:
            return "消息过长（超过 2000 字符），请缩短后重试。"

        self.state.current_user_turn += 1
        self.state.conversation.append({"role": "user", "content": user_message})

        # ── 步骤3：路由门控判断 ──────────────────────────────────────────────────────────
        #    调用路由 LLM 判断意图，可能返回拦截文本或放行执行
        intercept_reply, execution_confirmed = self._maybe_handle_confirmation_gate(user_message)
        if intercept_reply is not None:
            self.state.conversation.append({"role": "assistant", "content": intercept_reply})
            return intercept_reply

        # ── 步骤4：初始化本轮执行状态 ─────────────────────────────────────────────────────
        #    清空工具计数、设置执行契约约束
        self.state.current_turn_tools = set()
        self.state.current_turn_tool_call_count = 0
        self.state.current_turn_requires_tool_call = bool(
            execution_confirmed
            and self.state.last_confirmed_request is not None
            and self.state.last_confirmed_request.must_call_tool_before_reply
        )
        contract_hint: str | None = None
        contract_retry_used = False
        if execution_confirmed:
            self._log_execution_overview()

        # ── 步骤5：ReAct 循环（最多 MAX_TOOL_CALLS_PER_TURN 次调用）──────────────────────
        for step in range(MAX_TOOL_CALLS_PER_TURN):
            response = self._call_llm(
                execution_confirmed=execution_confirmed,
                contract_hint=contract_hint,
            )
            if response is None:
                error_msg = "抱歉，LLM 调用失败，请稍后重试。"
                self.state.conversation.append({"role": "assistant", "content": error_msg})
                return error_msg

            message = response.get("choices", [{}])[0].get("message", {})

            # ── 分支5a：无 Tool 调用 → LLM 直接回复 ───────────────────────────────────────
            #    需检查执行契约：事实核查类请求必须先调用 Tool
            tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                content = message.get("content", "") or ""

                # 子分支：违反执行契约 → 强制重试一次
                if self._violates_execution_contract(content):
                    if not contract_retry_used:
                        contract_retry_used = True
                        contract_hint = (
                            "[执行契约] 当前请求属于事实核查，必须先调用至少一个 Tool 获取事实，"
                            "再给出回复。不要直接输出等待文案或结论。"
                        )
                        logger.warning("[Agent] 周中执行契约重试：本轮尚无 Tool 调用，触发强制二次规划。")
                        continue
                    safe_reply = self._build_contract_fallback_reply()
                    self.state.conversation.append({"role": "assistant", "content": safe_reply})
                    return safe_reply

                # 子分支：正常回复 → 返回给用户
                self.state.conversation.append({"role": "assistant", "content": content})
                return content if content else "（Agent 未生成回复，请重试或换个问法。）"

            # ── 分支5b：有 Tool 调用 → 依次执行 ────────────────────────────────────────────
            contract_hint = None

            self.state.conversation.append({
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": tool_calls,
            })

            for tc in tool_calls:
                tool_name = tc.get("function", {}).get("name", "")
                tool_args_str = tc.get("function", {}).get("arguments", "{}")
                tool_call_id = tc.get("id", "")

                # 子分支：参数解析失败 → 返回错误提示给 LLM
                try:
                    tool_args = json.loads(tool_args_str) if tool_args_str else {}
                    if not isinstance(tool_args, dict):
                        raise ValueError("Tool arguments 必须是 JSON object")
                except (json.JSONDecodeError, ValueError) as e:
                    raw_args_preview = tool_args_str[:500]
                    logger.warning(
                        "[Agent] Tool %s 参数解析失败: %s | raw=%s",
                        tool_name,
                        e,
                        raw_args_preview,
                    )
                    result = {
                        "error": f"Tool arguments JSON 解析失败: {e}",
                        "raw_arguments": raw_args_preview,
                        "hint": "请返回合法 JSON object 作为 tool arguments，例如 {\"top_n\": 20}。",
                    }
                    result_str = self._serialize_result(result)
                    self.state.conversation.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": result_str,
                    })
                    continue

                # 子分支：参数解析成功 → 执行 Tool
                logger.info(f"[Agent] Tool 调用: {tool_name}({tool_args})")
                try:
                    result = self._execute_tool(tool_name, tool_args)
                except Exception as e:
                    logger.error(f"[Agent] Tool {tool_name} 执行异常: {e}")
                    result = {"error": f"工具执行异常: {e}"}

                # 记录成功调用的工具（用于执行契约检查）
                if not (isinstance(result, dict) and result.get("error")):
                    self.state.current_turn_tools.add(tool_name)
                    self.state.current_turn_tool_call_count += 1

                # 记录 Tool 结果到持久化状态（用于追问上下文）
                self._remember_tool_observation(tool_name, tool_args, result)

                # 序列化结果并追加到 conversation
                try:
                    result_str = self._serialize_result(result)
                except Exception as serialize_err:
                    logger.error(f"[Agent] Tool 结果序列化异常: {serialize_err}")
                    result_str = json.dumps({"error": f"序列化失败: {serialize_err}"})

                self.state.conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result_str,
                })

        # ── 步骤6：超过最大调用次数 → 返回提示 ───────────────────────────────────────────
        return "已达到单轮最大 Tool 调用次数，请尝试简化请求。"

    def _call_llm(
        self,
        execution_confirmed: bool = False,
        contract_hint: str | None = None,
    ) -> dict | None:
        """
        调用执行阶段 LLM（带 Tool 定义）。

        处理流程：
          1. 复制 conversation 作为消息基础
          2. 根据路由结果裁剪工具集
          3. 向 system prompt 注入执行上下文（ResolvedRequest、工具集、契约提示）
          4. 调用 LLM API

        参数：
          - execution_confirmed: 是否已通过路由确认，可直接执行
          - contract_hint: 执行契约提示（用于强制调用 Tool）
        """
        # ── 步骤1：准备消息基础 ───────────────────────────────────────────────────────────
        messages = list(self.state.conversation)

        # ── 步骤2：裁剪工具集 ─────────────────────────────────────────────────────────────
        selected_tools = self._select_tools_for_llm()
        selected_tool_names = [
            schema.get("function", {}).get("name")
            for schema in selected_tools
            if schema.get("function", {}).get("name")
        ]

        # ── 步骤3：注入执行上下文到 system prompt ────────────────────────────────────────
        #    包括：执行确认标记、ResolvedRequest、工具集列表、契约提示
        if messages and messages[0].get("role") == "system":
            extra_sections = []

            # 子分支：已确认执行 → 添加执行标记
            if execution_confirmed:
                extra_sections.append("[执行上下文] 当前请求已通过路由判定可执行。请直接执行对应 Tool，不要回到\"确认请回复开始\"。")

            # 子分支：有已确认请求 → 注入执行参数
            if self.state.last_confirmed_request is not None:
                extra_sections.append(self.state.last_confirmed_request.to_execution_context())

            # 子分支：添加工具集列表
            extra_sections.append(f"[本轮工具集] {selected_tool_names}")

            # 子分支：契约提示（强制调用 Tool）
            if contract_hint:
                extra_sections.append(contract_hint)

            # 合并到 system prompt
            if extra_sections:
                messages[0] = {
                    "role": "system",
                    "content": messages[0]["content"] + "\n\n" + "\n\n".join(extra_sections),
                }

        # ── 步骤4：调用 LLM API ───────────────────────────────────────────────────────────
        return self._request_llm(
            messages=messages,
            tools=selected_tools,
            temperature=0.3,
            max_tokens=16384,
            log_prefix="[Agent]",
            enable_thinking=False,
            thinking_budget=8192,
        )

    def _select_tools_for_llm(self) -> list[dict]:
        """
        根据路由结果动态裁剪可用工具集。

        裁剪规则：
          1. 无路由结果 → 开放全工具
                    2. 仅对复杂榜单意图做工具约束
                    3. 其他意图默认开放全工具

        目的：
                    - 在复杂流程场景提供必要护栏
                    - 在普通查询场景保留 ReAct 自主工具选择
        """
        resolved = self.state.last_confirmed_request

        # ── 分支1：无路由结果 → 开放全工具 ──────────────────────────────────────────────
        if resolved is None:
            return TOOL_SCHEMAS

        # ── 分支2：仅对受约束意图应用工具白名单 ──────────────────────────────────────────
        allowed = set(CONSTRAINED_TOOLS_BY_INTENT.get(resolved.intent_family, set()))

        # ── 分支3：非受约束意图默认开放全工具（兜底）────────────────────────────────────
        if not allowed:
            return TOOL_SCHEMAS

        # ── 分支4：过滤并返回工具 schema ─────────────────────────────────────────────────
        filtered = [TOOL_SCHEMA_BY_NAME[name] for name in ALL_TOOL_NAMES if name in allowed]
        return filtered or TOOL_SCHEMAS

    def _violates_execution_contract(self, content: str) -> bool:
        """
        执行契约检查：当前轮次是否违反"必须先调 Tool"的约束。

        触发条件：
          - current_turn_requires_tool_call = true（路由判定需要先调工具）
          - current_turn_tool_call_count = 0（本轮尚未调用任何工具）

        返回：
          - True: 违反契约，LLM 试图直接回复而非调用工具
          - False: 不违反契约，可以直接回复或已调用过工具

        目的：
          - 防止 fact_check 类请求跳过数据获取直接给出结论
          - 确保事实核查类回复基于真实数据
        """
        # ── 分支1：本轮不需要强制调用工具 → 不违反 ─────────────────────────────────────
        if not self.state.current_turn_requires_tool_call:
            return False

        # ── 分支2：本轮已调用过工具 → 不违反 ───────────────────────────────────────────
        if self.state.current_turn_tool_call_count > 0:
            return False

        # ── 分支3：需要调用但未调用 → 违反契约 ───────────────────────────────────────────
        return True

    def _build_contract_fallback_reply(self) -> str:
        """
        构建执行契约失败后的兜底回复。

        场景：
          - fact_check 类请求
          - 路由判定 must_call_tool_before_reply=true
          - LLM 尝试直接回复（违反契约）
          - 重试后仍违反 → 返回此兜底文本

        目的：
          - 引导用户补充具体核查点
          - 保证对话不中断
        """
        repo = self.state.active_repo
        if repo:
            return (
                f"为了保证准确性，我需要先调用数据工具核查 `{repo}` 的最新事实。"
                "请直接告诉我要核查的点（如创建时间、当前 star、近7天增长），我会立即执行查询。"
            )
        return "为了保证准确性，我需要先调用数据工具核查。请先提供仓库名（owner/repo），我会立即执行查询。"

    def _remember_tool_observation(self, tool_name: str, tool_args: dict, result: dict) -> None:
        """
        记录本轮 Tool 观测结果，供后续事实核查复用。

        记录内容：
          - active_repo: 从工具参数或结果中提取的仓库名
          - recent_verified_claims: check_repo_growth / get_db_info 的核查结果

        目的：
          - 支持追问时复用已核查的事实（fact_check 场景）
          - 避免重复查询相同仓库的相同数据

        存储限制：
          - recent_verified_claims 最多保留 20 条
        """
        # ── 分支1：结果无效 → 不记录 ────────────────────────────────────────────────────
        if not isinstance(result, dict) or result.get("error"):
            return

        # ── 分支2：提取仓库名（从参数或结果）────────────────────────────────────────────
        repo = None
        if isinstance(tool_args, dict):
            repo = tool_args.get("repo")
        if not repo:
            repo = result.get("repo")

        # ── 分支3：更新 active_repo（用于追问上下文）────────────────────────────────────
        if isinstance(repo, str) and repo:
            self.state.active_repo = repo

        # ── 分支4：记录 check_repo_growth 结果 ───────────────────────────────────────────
        #    包含：仓库、创建时间、当前 star、增长值、统计窗口
        if tool_name == "check_repo_growth":
            claim = {
                "repo": result.get("repo") or repo,
                "created_at": result.get("created_at"),
                "current_star": result.get("current_star"),
                "growth": result.get("growth"),
                "growth_calc_days": result.get("growth_calc_days"),
                "source_tool": tool_name,
                "turn": self.state.current_user_turn,
            }
            if claim.get("repo"):
                self.state.recent_verified_claims.append(claim)

        # ── 分支5：记录 get_db_info 结果 ────────────────────────────────────────────────
        elif tool_name == "get_db_info" and repo:
            info = result.get("info") if isinstance(result.get("info"), dict) else {}
            if info:
                self.state.recent_verified_claims.append(
                    {
                        "repo": repo,
                        "created_at": info.get("created_at"),
                        "current_star": info.get("star"),
                        "source_tool": tool_name,
                        "turn": self.state.current_user_turn,
                    }
                )

        # ── 分支6：限制 recent_verified_claims 长度 ─────────────────────────────────────
        if len(self.state.recent_verified_claims) > 20:
            self.state.recent_verified_claims = self.state.recent_verified_claims[-20:]

    def _request_llm(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 16384,
        log_prefix: str = "[Agent]",
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
    ) -> dict | None:
        """
        统一的 LLM API 请求入口。

        功能：
          - 支持带 Tool 定义（执行阶段）或纯文本（路由阶段）
          - 支持启用 thinking 模式（路由阶段）
          - 3 次重试机制
          - 诊断日志记录 token 用量

        参数：
          - messages: 对话消息列表
          - tools: 工具定义（可选）
          - temperature: 温度参数（路由阶段 0.1，执行阶段 0.3）
          - max_tokens: 最大输出 token
          - enable_thinking: 是否启用 thinking 模式
          - thinking_budget: thinking token 预算

        返回：
          - dict: LLM 响应数据（包含 choices、usage）
          - None: 调用失败（重试 3 次后）
        """
        # ── 步骤1：构建请求头和基础 payload ─────────────────────────────────────────────
        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # ── 步骤2：可选参数注入（thinking、tools）──────────────────────────────────────
        if enable_thinking is not None:
            payload["enable_thinking"] = enable_thinking
        if thinking_budget is not None:
            payload["thinking_budget"] = thinking_budget
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        # ── 步骤3：重试循环（最多 3 次）──────────────────────────────────────────────────
        for attempt in range(3):
            try:
                logger.info("%s 开始 LLM 调用: model=%s, attempt=%d", log_prefix, LLM_MODEL, attempt + 1)
                resp = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=300)

                # ── 分支3a：HTTP 200 → 解析响应 ───────────────────────────────────────────
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except (ValueError, Exception) as e:
                        logger.error(
                            "[Agent] LLM 响应 JSON 解析失败: %s, body=%s",
                            e, resp.text[:500],
                        )
                        continue

                    # 子分支：诊断日志（token 用量）
                    choice = (data.get("choices") or [{}])[0]
                    finish = choice.get("finish_reason", "unknown")
                    usage = data.get("usage", {})
                    detail = usage.get("completion_tokens_details", {})
                    logger.info(
                        "%s LLM 响应: finish=%s, prompt_tokens=%s, "
                        "completion_tokens=%s, reasoning_tokens=%s",
                        log_prefix,
                        finish,
                        usage.get("prompt_tokens"),
                        usage.get("completion_tokens"),
                        detail.get("reasoning_tokens"),
                    )

                    # 子分支：空响应检查（content + tool_calls 都为空）
                    msg = choice.get("message", {})
                    content = msg.get("content") or ""
                    has_tools = bool(msg.get("tool_calls"))
                    if not content and not has_tools:
                        logger.warning(
                            "%s LLM 返回空 content 且无 tool_calls "
                            "(finish=%s, reasoning_tokens=%s), attempt=%d",
                            log_prefix,
                            finish,
                            detail.get("reasoning_tokens"),
                            attempt + 1,
                        )
                        continue  # 重试，可能是 reasoning 耗尽 token

                    return data

                # ── 分支3b：HTTP 非 200 → 记录警告并重试 ───────────────────────────────────
                logger.warning(
                    "LLM 调用失败: status=%s, body=%s, attempt=%d",
                    resp.status_code, resp.text[:300], attempt + 1,
                )

            # ── 分支3c：请求异常 → 记录错误并重试 ─────────────────────────────────────────
            except requests.RequestException as e:
                logger.error(f"LLM 请求异常: {e}, attempt={attempt + 1}")

        # ── 步骤4：重试耗尽 → 返回 None ───────────────────────────────────────────────────
        return None

# ══════════════════════════════════════════════════════════════════════════════════════
    # 路由门禁：两阶段分离架构的核心入口
    # ══════════════════════════════════════════════════════════════════════════════════════
    #
    # 设计思想：
    #   路由阶段（本函数）→ 只做意图识别 + 参数抽取，不执行工具
    #   执行阶段（_call_llm）→ LLM 自主调用工具完成任务
    #
    # 返回值：(reply, confirmed)
    #   - reply ≠ None    → 拦截，返回澄清/确认文本给用户
    #   - reply = None    → 放行，进入执行阶段
    #   - confirmed = True → 已确认可执行，注入路由结果到执行阶段
    #
    # 状态流转：
    #   awaiting_confirmation = True  → 等待用户澄清/确认
    #   pending_request            → 待确认的请求（路由解析结果）
    #   last_confirmed_request     → 已确认的请求（参数已冻结，准备执行）
    #
    # ══════════════════════════════════════════════════════════════════════════════════════

    def _maybe_handle_confirmation_gate(self, user_message: str) -> tuple[str | None, bool]:
        """
        路由门禁：意图识别 + 参数抽取 + 确认门控

        流程决策树：
        ┌────────────────────────────────────────────────────────────────────────────┐
        │ 1. 消息为空 → 返回提示                                                       │
        │                                                                            │
        │ 2. awaiting_confirmation=True 且用户回复确认词（"是"/"开始"等）              │
        │    ├─ pending_request 不存在 → 返回提示（状态异常）                          │
        │    ├─ ambiguous_fields 非空 → 继续返回澄清文本（用户没解决歧义）              │
        │    └─ 无歧义 → 转为 ResolvedRequest，放行执行                               │
        │                                                                            │
        │ 3. awaiting_confirmation=False 且用户回复确认词 → 返回提示（无待确认请求）   │
        │                                                                            │
        │ 4. 其他情况（新消息或澄清内容）                                              │
        │    ├─ 调用路由 LLM 解析 → PendingRequest                                   │
        │    ├─ ambiguous_fields 非空 → awaiting=True，返回澄清文本                  │
        │    └─ 无歧义 → 转为 ResolvedRequest，放行执行                               │
        └────────────────────────────────────────────────────────────────────────────┘
        """
        text = (user_message or "").strip()

        # ── 分支1：空消息处理 ───────────────────────────────────────────────────────────
        if not text:
            self.state.awaiting_confirmation = False
            return "请直接告诉我你想看的 GitHub 热门项目需求。", False

        # ── 分支2：用户在"等待确认"状态下回复了确认词 ───────────────────────────────────
        #    场景：上一轮路由发现有歧义，返回了澄清文本，awaiting=True
        #    用户这一轮回复"是"/"开始"/"确认"等确认词
        if self.state.awaiting_confirmation and self._is_confirmation_ack(text):
            pending = self.state.pending_request

            # 子分支2a：pending_request 不存在（状态异常，可能是会话重置）
            if pending is None:
                self.state.awaiting_confirmation = False
                return "请直接描述要查询的 GitHub 热门项目需求。", False

            # 子分支2b：ambiguous_fields 非空 → 用户只回复了确认词，但没解决歧义
            #           继续返回澄清文本，引导用户补充具体信息
            #           例如：系统问"榜单类型是综合榜还是新项目榜？"，用户只回复"开始"
            if pending.ambiguous_fields:
                return self._render_clarification_message(pending), False

            # 子分支2c：无歧义 → 用户确认执行
            #           将 PendingRequest 转为 ResolvedRequest（参数冻结）
            #           清空等待状态，放行进入执行阶段
            self.state.awaiting_confirmation = False
            resolved = self._resolve_pending_request(pending)
            self.state.last_confirmed_request = resolved
            self.state.pending_request = None
            self._sync_active_repo_from_resolved_request(resolved)
            return None, True  # 放行，进入执行阶段

        # ── 分支3：用户回复确认词，但当前不在"等待确认"状态 ──────────────────────────────
        #    场景：用户误回复"开始"，但上一轮并没有待确认的请求
        if not self.state.awaiting_confirmation and self._is_confirmation_ack(text):
            self.state.awaiting_confirmation = False
            return "请先直接描述要查询的 GitHub 热门项目需求。", False

        # ── 分支4：新消息或澄清内容 → 调用路由 LLM 解析 ────────────────────────────────
        #    这是核心路由流程：调用 CONFIRMATION_PROMPT 让 LLM 识别意图和抽取参数
        pending = self._build_route_pending_request()
        self.state.pending_request = pending
        self.state.last_confirmed_request = None

        # 子分支4a：有歧义、路由判定不可立即执行、或置信度低 → 拦截，等待用户澄清
        if pending.ambiguous_fields or not pending.should_execute_now or pending.route_confidence == "low":
            self.state.awaiting_confirmation = True
            return self._render_clarification_message(pending), False

        # 子分支4b：无歧义，可直接执行 → 转为 ResolvedRequest，放行
        self.state.awaiting_confirmation = False
        resolved = self._resolve_pending_request(pending)
        self.state.last_confirmed_request = resolved
        self.state.pending_request = None
        self._sync_active_repo_from_resolved_request(resolved)
        return None, True  # 放行，进入执行阶段

    # ────────────────────────────────────────────────────────────────────────────────────
    # 确认词识别：判断用户消息是否为"确认执行"意图
    # ────────────────────────────────────────────────────────────────────────────────────
    #
    # 设计原则：
    #   1. 短文本（≤16字符）→ 更可能是确认词
    #   2. 无数字 → 避误判"top 20"、"30天内"等为确认词
    #   3. 精确匹配 + 可扩展后缀 → 支持"开始"/"开始吧"/"开始呀"等变体
    #
    # ────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_confirmation_ack(message: str) -> bool:
        """
        判断用户消息是否为"确认执行"意图

        匹配规则：
          - 精确匹配：是、是的、开始、确认、执行、继续、好的、好、可以、行、没问题、ok、okay、yes
          - 后缀扩展：支持语气词后缀（吧/呀/啊/哈/啦/了），如"开始吧"、"好的呀"

        过滤规则：
          - 长度 > 16 → 不匹配（避免误判正常查询）
          - 包含数字 → 不匹配（避免误判"top 20"、"30天"等）
        """
        normalized = message.strip().lower().rstrip("。！？!?")
        if not normalized:
            return False

        if len(normalized) > 16:
            return False
        if any(ch.isdigit() for ch in normalized):
            return False

        exact_keywords = {
            "是",
            "是的",
            "开始",
            "确认",
            "执行",
            "继续",
            "好的",
            "好",
            "可以",
            "行",
            "没问题",
            "ok",
            "okay",
            "yes",
        }
        if normalized in exact_keywords:
            return True

        suffixes = ("", "吧", "呀", "啊", "哈", "啦", "了")
        expandable_keywords = ("开始", "确认", "执行", "继续", "好的", "可以", "没问题", "ok", "okay", "yes", "是的")
        for kw in expandable_keywords:
            if any(normalized == f"{kw}{suffix}" for suffix in suffixes):
                return True
        return False

    # ────────────────────────────────────────────────────────────────────────────────────
    # 路由 LLM 调用：意图识别 + 参数抽取的核心流程
    # ────────────────────────────────────────────────────────────────────────────────────
    #
    # 输入：
    #   - CONFIRMATION_PROMPT（路由 prompt，定义意图类型和参数规则）
    #   - 上下文数据包（对话历史、活跃仓库、已验证事实等）
    #
    # 输出：
    #   - PendingRequest（结构化请求，包含意图、参数、歧义字段等）
    #
    # 特点：
    #   - 不传工具定义（tools=None），只做意图识别，不执行工具
    #   - 低温度（temperature=0.1），稳定输出结构化 JSON
    #   - 短输出（max_tokens=1024），节省 token
    #
    # ────────────────────────────────────────────────────────────────────────────────────

    def _build_route_pending_request(self) -> PendingRequest:
        """
        调用路由 LLM，解析用户意图和参数

        流程：
          1. 构建 payload：CONFIRMATION_PROMPT + 上下文数据包
          2. 调用 LLM（不传工具定义，只做意图识别）
          3. 解析输出 JSON → PendingRequest

        失败处理：
          - LLM 调用失败 → 返回默认 PendingRequest（ambiguous_fields 非空）
          - JSON 解析失败 → 返回默认 PendingRequest
        """
        payload_messages = [
            {"role": "system", "content": CONFIRMATION_PROMPT},
            {
                "role": "user",
                "content": json.dumps(self._build_parse_context_payload(), ensure_ascii=False),
            },
        ]

        response = self._request_llm(
            messages=payload_messages,
            tools=None,
            temperature=0.1,
            max_tokens=1024,
            log_prefix="[Agent-Route]",
            enable_thinking=False,
        )
        fallback = self._default_confirmation_message()
        if response is None:
            return PendingRequest(
                confirmation_text_zh=fallback,
                ambiguous_fields=["暂时无法完成语义，请重试或补充关键条件"],
                source_turn_id=self.state.current_user_turn,
            )

        message = response.get("choices", [{}])[0].get("message", {})
        content = (message.get("content") or "").strip()
        return self._parse_pending_request_content(content)

    def _build_parse_context_payload(self) -> dict[str, object]:
        """
        构建路由解析所需的紧凑上下文数据包

        包含内容：
          - current_user_turn: 当前用户轮次计数
          - recent_dialogue: 最近 8 条对话（user/assistant 角色）
          - active_repo: 当前活跃仓库（用于追问上下文）
          - recent_verified_claims: 近期核查的事实（用于 fact_check 场景）
          - pending_request: 上轮待确认请求（用于 clarification_answer 场景）

        不包含：
          - last_confirmed_request: 不传递，避免干扰路由 LLM 的输出格式
            （last_confirmed_request 仅在执行阶段注入到 system prompt）

        目的：
          - 提供足够上下文让路由 LLM 判断轮次类型和意图
          - 避免传递过多信息导致 token 浪费或输出干扰
        """
        recent_messages = [
            {"role": msg["role"], "content": msg.get("content") or ""}
            for msg in self.state.conversation[-8:]
            if msg.get("role") in {"user", "assistant"}
        ]

        return {
            "current_user_turn": self.state.current_user_turn,
            "recent_dialogue": recent_messages,
            "active_repo": self.state.active_repo,
            "recent_verified_claims": self.state.recent_verified_claims[-5:],
            "pending_request": self.state.pending_request.to_dict() if self.state.pending_request else None,
        }

    @staticmethod
    def _default_confirmation_message() -> str:
        """路由解析失败时的兜底澄清消息。"""
        return (
            "我暂时无法稳定判断你的请求。"
            "请直接补充关键条件（例如仓库名、时间窗口或榜单类型），我会继续执行。"
        )

    @staticmethod
    def _normalize_ambiguous_fields(raw_ambiguous: object) -> list[str]:
        """规范化歧义字段列表，保持插入顺序并去重。"""
        if not isinstance(raw_ambiguous, list):
            return []
        values = [str(item).strip() for item in raw_ambiguous if str(item).strip()]
        return list(dict.fromkeys(values))

    @staticmethod
    def _collect_unresolved_constraints(raw_unresolved: object, dropped_keys: list[str]) -> list[str]:
        """收集路由输出中的未解析约束和本地丢弃的参数键。"""
        unresolved: list[str] = []
        if isinstance(raw_unresolved, list):
            unresolved.extend(
                str(item).strip()
                for item in raw_unresolved
                if str(item).strip()
            )
        unresolved.extend(f"无法映射参数名: {key}" for key in dropped_keys)
        return list(dict.fromkeys(unresolved))

    def _parse_pending_request_content(self, content: str) -> PendingRequest:
        """
        解析路由 LLM 输出的 JSON，生成规范化的 PendingRequest

        处理流程：
          1. 提取 JSON 对象（extract_json_object）
          2. 归一化各字段（turn_kind、intent_family、specified_params 等）
          3. 收集未解析约束和歧义字段
          4. 计算 should_execute_now（无歧义且无未解析约束时为 True）
          5. 生成确认/澄清文本（confirmation_text_zh）

        字段归一化：
          - turn_kind: 映射到 TURN_KINDS 白名单
          - intent_family: 映射到 INTENT_LABELS + INTENT_ALIASES
          - specified_params: 过滤非法参数名，只保留 ALL_TOOL_PARAM_NAMES
          - suggested_tools: 过滤非法工具名，只保留 TOOL_SCHEMA_NAME_SET

        失败处理：
          - JSON 提取失败 → 返回默认 PendingRequest（ambiguous_fields 非空）
        """
        # 生成兜底澄清文本（用于 JSON 解析失败时）
        fallback = sanitize_confirmation_fallback(content, self._default_confirmation_message())

        # 从 LLM 输出中提取 JSON 对象
        payload = extract_json_object(content)

        # JSON 提取失败 → 返回默认 PendingRequest，标记歧义并拦截
        if not isinstance(payload, dict):
            return PendingRequest(
                confirmation_text_zh=fallback,
                ambiguous_fields=["我还需要确认你的具体目标（如仓库名、榜单类型或时间范围）"],
                should_execute_now=False,
                source_turn_id=self.state.current_user_turn,
            )

        # ── 字段归一化：将 LLM 输出映射到系统标准值 ──

        # turn_kind: 映射到白名单，非法值归为 "unknown"
        turn_kind = normalize_turn_kind(payload.get("turn_kind"), turn_kinds=TURN_KINDS)

        # intent_family: 别名映射 + 白名单校验，非法值归为 "unknown"
        intent_family = normalize_intent_family(
            payload.get("intent_family"),
            intent_aliases=INTENT_ALIASES,
            intent_labels=INTENT_LABELS,
        )

        # specified_params: 过滤非法参数名，只保留合法键；dropped_param_keys 记录被丢弃的键
        specified_params, dropped_param_keys, param_notes = normalize_specified_params(
            payload.get("specified_params"),
            allowed_param_names=ALL_TOOL_PARAM_NAMES,
        )
        if param_notes:
            logger.debug("[Agent-Route] 参数归一化: %s", " | ".join(param_notes))

        # 收集未解析约束（LLM 标记的 + 本地丢弃的参数键）
        unresolved_constraints = self._collect_unresolved_constraints(
            payload.get("unresolved_constraints"),
            dropped_param_keys,
        )

        # 规范化歧义字段列表（去重保序）
        normalized_ambiguous = self._normalize_ambiguous_fields(payload.get("ambiguous_fields"))

        # suggested_tools: 过滤非法工具名，只保留白名单内的
        suggested_tools = normalize_tool_names(
            payload.get("suggested_tools"),
            allowed_tool_names=TOOL_SCHEMA_NAME_SET,
        )

        # ── 置信度解析：默认 medium，只接受 high/medium/low ──
        route_confidence_raw = payload.get("route_confidence")
        route_confidence = "medium"
        if isinstance(route_confidence_raw, str):
            normalized_conf = route_confidence_raw.strip().lower()
            if normalized_conf in {"high", "medium", "low"}:
                route_confidence = normalized_conf

        # ── 目标仓库推断：多来源优先级 ──
        # 1. LLM 显式指定的 target_repo
        # 2. specified_params 中的 repo 参数
        # 3.追问类 turn_kind 时复用 active_repo
        raw_target_repo = payload.get("target_repo")
        target_repo = ""
        if isinstance(raw_target_repo, str) and raw_target_repo.strip():
            target_repo = raw_target_repo.strip()
        elif isinstance(specified_params.get("repo"), str):
            target_repo = str(specified_params.get("repo") or "").strip()
        elif turn_kind in {"fact_check", "request_modification", "clarification_answer", "execution_ack"}:
            target_repo = self.state.active_repo or ""

        # ── 是否可立即执行判断 ──
        # 条件：LLM 说可以 + 无歧义字段 + 无未解析约束
        should_execute_now_raw = payload.get("should_execute_now")
        has_unresolved = bool(unresolved_constraints)
        if isinstance(should_execute_now_raw, bool):
            should_execute_now = should_execute_now_raw and not normalized_ambiguous and not has_unresolved
        else:
            should_execute_now = not normalized_ambiguous and not has_unresolved

        # ── 是否必须先调工具 ──
        # fact_check 类型强制要求 must_call_tool_before_reply=true
        must_call_tool_raw = payload.get("must_call_tool_before_reply")
        must_call_tool_before_reply = bool(must_call_tool_raw)
        if turn_kind == "fact_check":
            must_call_tool_before_reply = True

        # ── 构建 PendingRequest 对象 ──
        pending = PendingRequest(
            turn_kind=turn_kind,
            intent_family=intent_family,
            intent_label_zh=str(payload.get("intent_label_zh") or INTENT_LABELS[intent_family]),
            target_repo=target_repo,
            user_specified_params=specified_params,
            unresolved_constraints=unresolved_constraints,
            ambiguous_fields=normalized_ambiguous,
            suggested_tools=suggested_tools,
            route_confidence=route_confidence,
            report_requested=bool(payload.get("report_requested")),
            should_execute_now=should_execute_now,
            must_call_tool_before_reply=must_call_tool_before_reply,
            source_turn_id=self.state.current_user_turn,
        )

        # ── 确认/澄清文本生成 ──
        # 优先用 LLM 提供的 confirmation_text_zh（非结构化时）
        # 否则根据歧义/未解析情况自动生成
        raw_confirmation = str(payload.get("confirmation_text_zh") or "").strip()
        if raw_confirmation and not looks_like_structured_confirmation_text(raw_confirmation):
            pending.confirmation_text_zh = raw_confirmation
        else:
            pending.confirmation_text_zh = (
                self._render_clarification_message(pending)
                if pending.ambiguous_fields or pending.unresolved_constraints
                else self._render_pending_request_text(pending)
            )
        return pending

    def _render_pending_request_text(self, pending: PendingRequest) -> str:
        """生成简洁的执行确认文本供用户确认。"""
        fragments = [pending.intent_label_zh]
        for key, value in pending.user_specified_params.items():
            renderer = PARAM_DISPLAYERS.get(key)
            if renderer is None:
                continue
            fragment = renderer(value)
            if fragment:
                fragments.append(fragment)
        if pending.report_requested:
            fragments.append("结果完成后生成报告")

        body = "，".join(fragment for fragment in fragments if fragment)
        if not body:
            body = "你的 GitHub 热门项目需求"
        return f"收到！我理解为：{body}。我会按这个方向继续执行；如果要改参数请直接告诉我。"

    # ────────────────────────────────────────────────────────────────────────────────────
    # 澄清文本生成：当路由发现歧义时，生成引导用户补充信息的文本
    # ────────────────────────────────────────────────────────────────────────────────────

    def _render_clarification_message(self, pending: PendingRequest) -> str:
        """
        生成用户可见的澄清文本

        触发场景：
          - route_confidence == "low" → LLM对理解不确定，引导用户重述
          - unresolved_constraints 非空 → 有无法解析的约束（如未知参数名）
          - ambiguous_fields 非空 → 有歧义字段需要用户澄清

        输出格式：
          - 置信度低："我不太确定你的需求，请重新描述一下。我可以帮你..."
          - 有 unresolved_constraints："我还需要你确认这些参数含义：xxx。确认后我再继续执行。"
          - 有 ambiguous_fields："我还需要确认这些点：xxx。请直接补充，我会继续执行。"
          - 无具体问题："我还需要补充关键条件后才能执行。请直接告诉我..."

        目的：
          - 引导用户提供具体信息解决歧义
          - 保持对话自然流畅，不中断用户体验
        """

        # 置信度低：无法稳定理解用户意图，返回功能介绍引导重述
        if pending.route_confidence == "low":
            return (
                "我不太确定你的需求，请重新描述一下。\n"
                "我可以帮你：\n"
                "- 查询热门项目榜单（综合榜、新项目榜、关键词榜、Trending）\n"
                "- 查看单个仓库的详情和增长数据\n"
                "- 核查项目的具体增长数值\n"
                "- 按增长时间窗口、关键词等条件筛选"
            )

        if pending.unresolved_constraints:
            questions: list[str] = []
            for item in pending.unresolved_constraints:
                content = str(item).strip()
                if not content:
                    continue
                if content.startswith("无法映射参数名:"):
                    raw_name = content.split(":", 1)[1].strip()
                    if raw_name:
                        questions.append(f"参数“{raw_name}”对应哪个筛选条件")
                        continue
                questions.append(content)

            if questions:
                unresolved_text = "；".join(questions)
                return f"我还需要你确认这些参数含义：{unresolved_text}。确认后我再继续执行。"

        if pending.ambiguous_fields:
            ambiguous_text = "；".join(pending.ambiguous_fields)
            return f"我还需要确认这些点：{ambiguous_text}。请直接补充，我会继续执行。"

        return "我还需要补充关键条件后才能执行。请直接告诉我你希望的仓库、时间窗口或榜单类型。"


    # ────────────────────────────────────────────────────────────────────────────────────
    # Pending → Resolved 转换：参数合并与冻结
    # ────────────────────────────────────────────────────────────────────────────────────
    #
    # PendingRequest（路由输出）→ 用户意图 + 用户指定参数（可能有歧义）
    # ResolvedRequest（执行输入）→ 合并后的完整参数（用户 + 默认值），无歧义
    #
    # 合并规则：
    #   1. 意图默认参数（_default_params_for_intent）作为基础
    #   2. 用户指定参数（user_specified_params）覆盖默认值
    #   3. target_repo 自动注入到 repo 参数
    #   4. 榜单型任务计算实际 growth_calc_days（根据 DB 年龄或默认值）
    #
    # ────────────────────────────────────────────────────────────────────────────────────

    def _resolve_pending_request(self, pending: PendingRequest) -> ResolvedRequest:
        """
        将 PendingRequest 转换为 ResolvedRequest

        参数合并顺序：
          1. defaults = _default_params_for_intent(intent_family) → 意图默认参数
          2. resolved_params = defaults → 初始化
          3. resolved_params.update(user_specified_params) → 用户参数覆盖默认值
          4. resolved_params["repo"] = target_repo → 目标仓库注入

        榜单型任务特殊处理：
          - 用户未指定 growth_calc_days → 使用 DB 年龄（若有效）或默认值
          - 确保确认文本与实际执行一致
        返回：
          ResolvedRequest（参数已冻结，准备执行）
        """
        defaults = self._default_params_for_intent(pending.intent_family)
        resolved_params = dict(defaults)
        resolved_params.update(pending.user_specified_params)
        if pending.target_repo and "repo" not in resolved_params:
            resolved_params["repo"] = pending.target_repo

        # ── 综合榜/关键词榜：用户未指定窗口时，提前确定实际窗口 ──
        # 若 DB 有效则用 DB 年龄，否则用默认值，确保确认文本与实际执行一致
        ranking_intents = {"comprehensive_ranking", "keyword_ranking"}
        if pending.intent_family in ranking_intents:
            user_specified_window = "growth_calc_days" in pending.user_specified_params
            if not user_specified_window:
                db_valid = self.state.db.get("valid", False)
                db_age = get_db_age_days(self.state.db)
                if db_valid and db_age is not None and db_age > 0:
                    # 用 DB 年龄作为实际窗口，与 batch_check_growth 逻辑保持一致
                    resolved_params["growth_calc_days"] = db_age

        defaulted_params = {
            key: value for key, value in defaults.items()
            if key not in pending.user_specified_params
        }

        suggested_tools = list(pending.suggested_tools)
        if not suggested_tools:
            intent_tools = CONSTRAINED_TOOLS_BY_INTENT.get(pending.intent_family, set())
            suggested_tools = ordered_tool_names(intent_tools, all_tool_names=ALL_TOOL_NAMES)

        if pending.turn_kind == "fact_check":
            suggested = set(suggested_tools)
            suggested.update({"check_repo_growth", "get_db_info", "describe_project", "fetch_trending"})
            suggested_tools = ordered_tool_names(suggested, all_tool_names=ALL_TOOL_NAMES)

        return ResolvedRequest(
            turn_kind=pending.turn_kind,
            intent_family=pending.intent_family,
            intent_label_zh=pending.intent_label_zh,
            target_repo=pending.target_repo,
            resolved_params=resolved_params,
            user_specified_params=dict(pending.user_specified_params),
            defaulted_params=defaulted_params,
            suggested_tools=suggested_tools,
            route_confidence=pending.route_confidence,
            report_requested=pending.report_requested,
            must_call_tool_before_reply=pending.must_call_tool_before_reply,
            confirmation_text_zh=pending.confirmation_text_zh,
        )

    def _sync_active_repo_from_resolved_request(self, resolved: ResolvedRequest) -> None:
        """
        从已解析请求中同步活跃仓库缓存。

        提取顺序：
          1. target_repo 字段（路由阶段识别的仓库）
          2. resolved_params["repo"]（参数中的仓库）

        目的：
          - 支持追问时复用仓库上下文（如 "查 langchain 的增长" → "它的 star 多少"）
        """
        # ── 分支1：从 target_repo 提取 ─────────────────────────────────────────────────
        repo = resolved.target_repo
        if not repo:
            # ── 分支2：从 resolved_params 提取 ───────────────────────────────────────────
            raw_repo = resolved.resolved_params.get("repo")
            if isinstance(raw_repo, str) and raw_repo.strip():
                repo = raw_repo.strip()

        # ── 分支3：更新 active_repo ────────────────────────────────────────────────────
        if repo:
            self.state.active_repo = repo

    @staticmethod
    def _default_params_for_intent(intent_family: str) -> dict[str, object]:
        """
        返回各意图类型在 Tool 执行前使用的默认参数槽位。

        用途：
          - 在 _resolve_pending_request 中与用户参数合并
          - 确保工具执行时有完整参数集

        各意图的默认值：
          - comprehensive_ranking: 综合榜默认配置
          - hot_new_ranking: 新项目榜默认配置（含创建时间窗口）
          - keyword_ranking: 关键词榜默认配置
          - trending_only: Trending 默认范围
          - repo_info / repo_growth: 单仓库默认增长窗口

        返回：
          - dict: 默认参数字典（空 dict 表示无默认值）
        """
        # ── 分支1：综合榜默认参数 ───────────────────────────────────────────────────────
        if intent_family == "comprehensive_ranking":
            return {
                "mode": "comprehensive",
                "top_n": HOT_PROJECT_COUNT,
                "growth_calc_days": GROWTH_CALC_DAYS,
                "growth_threshold": STAR_GROWTH_THRESHOLD,
                "min_star": MIN_STAR,
                "max_star": MAX_STAR,
                "trending_range": "all",
            }

        # ── 分支2：新项目榜默认参数 ─────────────────────────────────────────────────────
        if intent_family == "hot_new_ranking":
            return {
                "mode": "hot_new",
                "top_n": HOT_NEW_PROJECT_COUNT,
                "growth_calc_days": GROWTH_CALC_DAYS,
                "days_since_created": DAYS_SINCE_CREATED,
                "growth_threshold": STAR_GROWTH_THRESHOLD,
                "min_star": MIN_STAR,
                "max_star": MAX_STAR,
                "trending_range": "all",
            }

        # ── 分支3：Trending 默认参数 ────────────────────────────────────────────────────
        if intent_family == "trending_only":
            return {"trending_range": "weekly"}

        # ── 分支4：单仓库查询默认参数 ──────────────────────────────────────────────────
        if intent_family == "repo_info":
            return {"growth_calc_days": GROWTH_CALC_DAYS}
        if intent_family == "repo_growth":
            return {"growth_calc_days": GROWTH_CALC_DAYS}

        # ── 分支5：关键词榜默认参数 ─────────────────────────────────────────────────────
        if intent_family == "keyword_ranking":
            return {
                "mode": "keyword",
                "top_n": HOT_PROJECT_COUNT,
                "growth_calc_days": GROWTH_CALC_DAYS,
                "growth_threshold": STAR_GROWTH_THRESHOLD,
                "min_star": MIN_STAR,
            }

        # ── 分支6：其他意图 → 无默认参数 ───────────────────────────────────────────────
        return {}

    # ────────────────────────────────────────────────────────────────────────────────────
    # 参数注入：在 LLM 调用工具时，将路由解析的参数合并到工具参数
    # ────────────────────────────────────────────────────────────────────────────────────
    #
    # 合并时机：LLM 决定调用工具后，执行工具前
    # 合入来源：resolved_request.resolved_params（用户参数 + 意图默认参数）
    #
    # 合并规则：
    #   1. setdefault（不覆盖 LLM 已选择的参数）
    #   2. 单仓库工具自动注入 target_repo
    #   3. 榜单任务 fetch_trending 强制 trending_range="all"
    #   4. rank_candidates 根据 intent_family 设置 mode
    #
    # ────────────────────────────────────────────────────────────────────────────────────

    def _merge_request_defaults_into_tool_args(self, name: str, args: dict) -> dict:
        """
        将路由解析的参数注入到工具参数

        合入规则：
          1. resolved_params → setdefault（不覆盖 LLM 已选）
          2. 单仓库工具（check_repo_growth/describe_project/get_db_info）→ 注入 repo
          3. 榜单任务 fetch_trending → 强制 trending_range="all"
          4. rank_candidates → 根据 intent_family 设置 mode

        注意：
          - 使用 setdefault，LLM 显式选择的参数优先级更高
          - 硬编码工具名判断，扩展时需同步修改（架构改进点）
        """
        merged = dict(args)
        resolved_request = self.state.last_confirmed_request
        if resolved_request is None:
            return merged

        for key, value in resolved_request.resolved_params.items():
            merged.setdefault(key, value)

        if resolved_request.target_repo and name in {"check_repo_growth", "describe_project", "get_db_info"}:
            merged.setdefault("repo", resolved_request.target_repo)

        # 榜单型任务：fetch_trending 使用 trending_range="all"
        if resolved_request.requires_full_collection() and name == "fetch_trending":
            merged["trending_range"] = "all"

        if name == "rank_candidates":
            mode = "hot_new" if resolved_request.intent_family == "hot_new_ranking" else "comprehensive"
            merged.setdefault("mode", mode)

        return merged

    def _log_execution_overview(self) -> None:
        """
        在执行开始时打印本轮执行参数总览。

        输出内容：
          - turn: 当前轮次
          - turn_kind: 轮次类型
          - intent: 意图类型（英文 + 中文）
          - route_confidence: 路由置信度
          - persistence_policy: 持久化策略
          - params: 解析后的参数

        用途：
          - 调试追踪执行流程
          - 确认路由解析结果正确
        """
        resolved_request = self.state.last_confirmed_request

        # ── 分支1：无已确认请求 → 打印提示 ─────────────────────────────────────────────
        if resolved_request is None:
            logger.info(
                "[Agent] 运行参数总览: turn=%s | 当前无已确认请求（将按 LLM/工具默认参数执行）。",
                self.state.current_user_turn,
            )
            return

        # ── 分支2：有已确认请求 → 打印完整参数 ─────────────────────────────────────────
        mode = resolved_request.resolved_params.get("mode")
        mode_text = mode if isinstance(mode, str) else None
        persistence_policy = self._persistence_policy_for_request(mode=mode_text)

        logger.info(
            "[Agent] 运行参数总览: turn=%s | turn_kind=%s | intent=%s(%s) | route_confidence=%s | persistence_policy=%s | params=%s",
            self.state.current_user_turn,
            resolved_request.turn_kind,
            resolved_request.intent_family,
            resolved_request.intent_label_zh,
            resolved_request.route_confidence,
            persistence_policy,
            json.dumps(resolved_request.resolved_params, ensure_ascii=False, sort_keys=True, default=str),
        )

    def _check_suggested_collection_tools(self, tool_name: str) -> list[str]:
        """
        检查建议的候选收集工具是否已调用（仅作提示，不强制阻断）。

        设计原则：
          - LLM 自主决策为主
          - 硬编码只做建议提示
          - 不强制阻断执行

        检查时机：
          - 调用 batch_check_growth 或 rank_candidates 时

        返回：
          - list[str]: 缺失的建议工具名称列表
          - []: 无缺失或不需要检查
        """
        resolved_request = self.state.last_confirmed_request

        # ── 分支1：无已确认请求 或 不需要完整收集 → 跳过检查 ─────────────────────────────
        if resolved_request is None or not resolved_request.requires_full_collection():
            return []

        # ── 分支2：不是候选处理工具 → 跳过检查 ───────────────────────────────────────────
        if tool_name not in {"batch_check_growth", "rank_candidates"}:
            return []

        # ── 分支3：获取建议工具列表 ───────────────────────────────────────────────────────
        suggested_tools = SUGGESTED_COLLECTION_TOOLS_BY_INTENT.get(resolved_request.intent_family, set())
        if not suggested_tools:
            return []

        # ── 分支4：计算缺失的工具 ─────────────────────────────────────────────────────────
        missing = suggested_tools - self.state.current_turn_tools
        return sorted(missing)

    def _persistence_policy_for_request(self, mode: str | None = None) -> str:
        """返回请求对应的 DB 持久化策略：`desc_only` 或 `none`。

        规则：
        - 榜单型任务（comprehensive/hot_new/keyword）→ desc_only
        - 其他通道 → none
        """
        resolved_request = self.state.last_confirmed_request

        # 榜单型任务：只写 desc_only
        if mode in {"comprehensive", "hot_new"}:
            return "desc_only"
        if resolved_request is not None and resolved_request.requires_full_collection():
            return "desc_only"

        return "none"

    def _execute_tool(self, name: str, args: dict) -> dict:
        """
        路由并执行 Tool 调用。

        参数校验分两层：
          1. 严格校验（LLM 显式传入参数）：出错直接返回可重试错误
          2. 宽松校验（系统默认值注入 + 边界裁剪）

        执行流程：
          1. 严格校验 LLM 参数
          2. 合并默认参数
          3. 宽松校验并裁剪
          4. 重置榜单构建缓存（新一轮开始时）
          5. 调用具体 Tool 函数
          6. 更新状态缓存（last_search_repos、last_candidates 等）
        """
        state = self.state

        # ── 步骤1：严格校验 LLM 参数 ─────────────────────────────────────────────────────
        #    出错直接返回，要求 LLM 修正参数后重试
        _, strict_errors = validate_tool_args_strict(name, args)
        if strict_errors:
            logger.warning("[Agent] Tool %s 参数严格校验失败: %s", name, strict_errors)
            return {
                "error": "Tool 参数校验失败，请根据 invalid_arguments 修正后重试。",
                "error_code": "invalid_arguments",
                "tool_name": name,
                "invalid_arguments": strict_errors,
                "retryable": True,
            }

        # ── 步骤2：合并默认参数 + 宽松校验 ──────────────────────────────────────────────
        prepared_args = self._merge_request_defaults_into_tool_args(name, args)
        validated = validate_tool_args(name, prepared_args)
        log_validated_params(name, args, prepared_args, validated)

        # ── 步骤3：重置榜单构建缓存（新一轮开始时）─────────────────────────────────────
        self._maybe_reset_discovery_state(name, validated)

        # ── 步骤4：Tool 路由分发 ─────────────────────────────────────────────────────────

        # 分支4a：关键词搜索 → 更新 last_search_repos、seen_repos
        if name == "search_by_keywords":
            min_star = validated.get("min_star", MIN_STAR)
            result = tool_search_by_keywords(
                state.token_mgr,
                categories=validated.get("categories"),
                min_star=min_star,
                days_since_created=validated.get("days_since_created"),
            )
            raw_repos = result.pop("_raw_repos", [])
            state.last_search_repos = raw_repos
            state.seen_repos.update(r["full_name"] for r in raw_repos)
            state.last_min_star = min_star
            return result

        # 分支4b：star 范围扫描 → 追加到 last_search_repos
        elif name == "scan_star_range":
            result = tool_scan_star_range(
                state.token_mgr,
                min_star=validated.get("min_star", MIN_STAR),
                max_star=validated.get("max_star", MAX_STAR),
                seen_repos=state.seen_repos,
                days_since_created=validated.get("days_since_created"),
            )
            raw_repos = result.pop("_raw_repos", [])
            state.last_search_repos.extend(raw_repos)
            return result

        # 分支4c：单仓库增长核查 → 直接返回结果
        elif name == "check_repo_growth":
            repo = validated.get("repo")
            if not repo:
                return {"error": "缺少必需参数 repo（格式: owner/repo）"}
            return tool_check_repo_growth(
                state.token_mgr,
                repo=repo,
                db=state.db,
                growth_calc_days=validated.get("growth_calc_days", GROWTH_CALC_DAYS),
            )

        # 分支4d：批量增长计算 → 更新 last_candidates、持久化 desc
        elif name == "batch_check_growth":
            # 子分支：检查建议的候选收集工具（仅提示，不强制阻断）
            suggested_tools = self._check_suggested_collection_tools(name)
            if suggested_tools:
                suggested_text = "、".join(suggested_tools)
                logger.warning(
                    "[Agent] batch_check_growth: 建议先调用 %s 以最大化候选覆盖，但LLM可自主决策是否跳过。",
                    suggested_text,
                )

            # 子分支：检查前置条件（需要 search 结果）
            if not state.last_search_repos:
                return {"error": "没有搜索结果，请先调用 search_by_keywords"}

            # 子分支：提取参数并执行
            growth_calc_days = validated.get("growth_calc_days", GROWTH_CALC_DAYS)
            days_since_created = validated.get("days_since_created")
            growth_threshold = validated.get("growth_threshold", STAR_GROWTH_THRESHOLD)
            resolved_request = self.state.last_confirmed_request

            # 子分支：window_specified 判断（用户是否显式指定窗口）
            window_specified = "growth_calc_days" in args or (
                resolved_request is not None and (
                    "growth_calc_days" in resolved_request.user_specified_params
                    or "growth_calc_days" in resolved_request.resolved_params
                )
            )
            result = tool_batch_check_growth(
                state.token_mgr,
                repos=state.last_search_repos,
                db=state.db,
                growth_threshold=growth_threshold,
                days_since_created=days_since_created,
                growth_calc_days=growth_calc_days,
                window_specified=window_specified,
            )
            state.last_candidates = result.get("candidates", {})
            state.last_candidate_days_since_created = days_since_created
            state.last_growth_calc_days = result.get("growth_calc_days", growth_calc_days)
            state.last_growth_threshold = growth_threshold
            persistence_policy = self._persistence_policy_for_request()
            if result.get("db_updated", False) and persistence_policy == "desc_only":
                changed = save_db_desc_only(state.db)
                logger.info("[Agent] batch_check_growth 阶段仅持久化 desc 字段 (%d 个项目)。", changed)
            return result

        # 分支4e：候选排序 → 更新 last_ranked、last_mode
        elif name == "rank_candidates":
            # 子分支：检查建议的候选收集工具（仅提示）
            suggested_tools = self._check_suggested_collection_tools(name)
            if suggested_tools:
                suggested_text = "、".join(suggested_tools)
                logger.warning(
                    "[Agent] rank_candidates: 建议先调用 %s 以最大化候选覆盖，但LLM可自主决策是否跳过。",
                    suggested_text,
                )

            # 子分支：检查前置条件（需要 candidates）
            if not state.last_candidates:
                return {"error": "没有候选列表，请先调用 batch_check_growth"}

            # 子分支：提取参数并执行
            mode = validated.get("mode", "comprehensive")
            top_n = validated.get("top_n", HOT_PROJECT_COUNT if mode == "comprehensive" else HOT_NEW_PROJECT_COUNT)
            days_since_created = validated.get("days_since_created")
            result = tool_rank_candidates(
                state.last_candidates,
                top_n=top_n,
                mode=mode,
                db=state.db,
                days_since_created=days_since_created,
                prefiltered_days_since_created=state.last_candidate_days_since_created,
            )
            state.last_ranked = result.pop("_ordered_tuples", [])
            state.last_mode = mode
            return result

        # 分支4f：项目详情查询 → 直接返回
        elif name == "describe_project":
            repo = validated.get("repo")
            if not repo:
                return {"error": "缺少必需参数 repo（格式: owner/repo）"}
            return tool_describe_project(repo=repo, db=state.db, token_mgr=state.token_mgr)

        # 分支4g：报告生成 → 持久化 desc 字段
        elif name == "generate_report":
            # 子分支：检查前置条件（需要 ranked）
            if not state.last_ranked:
                return {"error": "没有排序结果，请先调用 rank_candidates"}

            # 子分支：执行报告生成
            report_new_project_days = state.last_candidate_days_since_created if state.last_mode == "hot_new" else None
            result = tool_generate_report(
                state.last_ranked,
                state.db,
                mode=state.last_mode,
                days_since_created=report_new_project_days,
                growth_calc_days=state.last_growth_calc_days,
                growth_threshold=state.last_growth_threshold,
                min_star=state.last_min_star,
            )

            # 子分支：持久化 desc 字段
            persistence_policy = self._persistence_policy_for_request(mode=state.last_mode)
            if persistence_policy == "desc_only":
                changed = save_db_desc_only(state.db)
                logger.info("[Agent] generate_report 阶段仅持久化 desc 字段 (%d 个项目)。", changed)
            return result

        # 分支4h：数据库信息查询 → 直接返回
        elif name == "get_db_info":
            return tool_get_db_info(db=state.db, repo=validated.get("repo"))

        # 分支4i：Trending 获取 → 补充到 last_search_repos
        elif name == "fetch_trending":
            result = tool_fetch_trending(
                trending_range=validated.get("trending_range", "weekly"),
            )
            raw_repos = result.pop("_raw_repos", [])
            for r in raw_repos:
                fn = r["full_name"]
                if fn not in state.seen_repos:
                    state.seen_repos.add(fn)
                    state.last_search_repos.append(trending_repo_to_search_repo(r))
            return result

        # 分支4j：未知 Tool → 返回错误
        else:
            return {"error": f"未知 Tool: {name}"}

    def _maybe_reset_discovery_state(self, tool_name: str, args: dict) -> None:
        """
        在新一轮榜单构建开始时，重置缓存状态。

        触发条件：
          - 调用 search_by_keywords 或 scan_star_range（榜单构建入口）
          - 调用 fetch_trending 且 trending_range="all"（全量补充）

        检查逻辑：
          - discovery_turn_id != current_user_turn → 新一轮，需要重置
          - discovery_turn_id == current_user_turn → 同一轮，跳过

        重置内容：
          - last_search_repos: 搜索结果缓存
          - last_candidates: 增长筛选后的候选
          - last_ranked: 排序结果
          - seen_repos: 已扫描仓库（去重）
          - discovery_turn_id: 标记本轮已初始化
        """
        # ── 分支1：判断是否为榜单构建入口工具 ───────────────────────────────────────────
        is_discovery_bootstrap = tool_name in {"search_by_keywords", "scan_star_range"}
        is_trending_supplement = tool_name == "fetch_trending" and args.get("trending_range") == "all"
        if not is_discovery_bootstrap and not is_trending_supplement:
            return

        # ── 分支2：检查是否同一轮（避免重复重置）──────────────────────────────────────
        current_turn = self.state.current_user_turn
        if self.state.discovery_turn_id == current_turn:
            return

        # ── 分支3：重置缓存状态 ─────────────────────────────────────────────────────────
        self.state.last_search_repos = []
        self.state.last_candidates = {}
        self.state.last_candidate_days_since_created = None
        self.state.last_ranked = []
        self.state.last_mode = "comprehensive"
        self.state.last_growth_calc_days = GROWTH_CALC_DAYS
        self.state.seen_repos.clear()
        self.state.discovery_turn_id = current_turn
        logger.info("[Agent] 检测到新一轮榜单构建，已重置候选、排序和去重状态。")

    @staticmethod
    def _serialize_result(result: dict, max_len: int = 8000) -> str:
        """
        将 Tool 结果序列化为 JSON 字符串（供 LLM 阅读）。

        截断策略：
          1. 正常长度 → 直接返回 JSON
          2. 超长 → 智能截断（保留摘要字段，截取列表前 N 项）
          3. 仍超长 → 硬截断

        目的：
          - 防止 Tool 结果过大导致 LLM context 溢出
          - 保留关键信息（error、摘要字段）
          - 标注截断信息让 LLM 知道数据不完整
        """
        # ── 步骤1：尝试直接序列化 ───────────────────────────────────────────────────────
        result_str = json.dumps(result, ensure_ascii=False, default=str)
        if len(result_str) <= max_len:
            return result_str

        # ── 步骤2：智能截断（逐步缩减列表/字典字段）────────────────────────────────────
        truncated = dict(result)
        list_keys = [k for k, v in truncated.items() if isinstance(v, (list, dict)) and k != "error"]

        # 子分支：尝试不同截断级别（50→30→20→10→5）
        for trim_count in (50, 30, 20, 10, 5):
            for key in list_keys:
                val = truncated[key]
                if isinstance(val, list) and len(val) > trim_count:
                    truncated[key] = val[:trim_count]
                    truncated[f"_{key}_note"] = f"已截取前{trim_count}项，原共{len(val)}项"
                elif isinstance(val, dict) and len(val) > trim_count:
                    items = list(val.items())[:trim_count]
                    truncated[key] = dict(items)
                    truncated[f"_{key}_note"] = f"已截取前{trim_count}项，原共{len(val)}项"
            s = json.dumps(truncated, ensure_ascii=False, default=str)
            if len(s) <= max_len:
                return s

        # ── 步骤3：兜底硬截断 ───────────────────────────────────────────────────────────
        return s[:max_len] + "\n...(结果已截断)"

    def _compress_conversation(self) -> None:
        """
        压缩对话历史：用 LLM 将早期消息语义摘要化，保留最近 KEEP_RECENT_MESSAGES 条。

        策略：
          1. 提取 system prompt（始终保留）
          2. 用 LLM 将旧消息浓缩为语义摘要
          3. 注入摘要到 system prompt 之后
          4. 保留最近的消息

        触发条件：
          - len(conversation) > MAX_CONVERSATION_MESSAGES（40 条）
        """
        conv = self.state.conversation

        # ── 步骤1：检查是否需要压缩 ─────────────────────────────────────────────────────
        if len(conv) <= MAX_CONVERSATION_MESSAGES:
            return

        # ── 步骤2：提取并清理 system prompt ─────────────────────────────────────────────
        #    移除旧的摘要标记，保留原始 system prompt
        system_message = next((m for m in conv if m.get("role") == "system"), None)
        system_content = (system_message or {}).get("content") or SYSTEM_PROMPT
        if "[对话历史摘要]" in system_content:
            system_content = system_content.split("\n\n[对话历史摘要]", 1)[0].strip() or SYSTEM_PROMPT

        # ── 步骤3：分离非 system 消息 ────────────────────────────────────────────────────
        non_system = [m for m in conv if m.get("role") != "system"]

        if len(non_system) <= KEEP_RECENT_MESSAGES:
            return

        # ── 步骤4：划分旧消息和近期消息 ────────────────────────────────────────────────
        old_msgs = non_system[:-KEEP_RECENT_MESSAGES]
        recent_msgs = non_system[-KEEP_RECENT_MESSAGES:]

        # ── 步骤5：生成语义摘要 ─────────────────────────────────────────────────────────
        llm_summary = self._generate_summary_with_llm(old_msgs)
        if llm_summary:
            self.state.conversation_summary = llm_summary
        else:
            # 子分支：LLM 调用失败 → fallback 简单截取
            summary_parts = []
            if self.state.conversation_summary:
                summary_parts.append(self.state.conversation_summary)
            for msg in old_msgs:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user" and content:
                    summary_parts.append(f"用户: {content[:200]}")
                elif role == "assistant" and content:
                    summary_parts.append(f"助手: {content[:200]}")
            self.state.conversation_summary = "\n".join(summary_parts[-20:])
            logger.warning("[Agent] LLM 摘要生成失败，fallback 到简单截取")

        # ── 步骤6：重建对话（system + 摘要 + 近期消息）──────────────────────────────────
        summary_text = (
            f"\n\n[对话历史摘要]\n"
            f"{self.state.conversation_summary}\n"
            f"[当前状态] 搜索结果: {len(self.state.last_search_repos)} 个, "
            f"候选仓库: {len(self.state.last_candidates)} 个, "
            f"已排序: {len(self.state.last_ranked)} 个, "
            f"已扫描: {len(self.state.seen_repos)} 个, "
            f"榜单模式: {self.state.last_mode}, "
            f"增长窗口: {self.state.last_growth_calc_days} 天, "
            f"创建窗口: {self.state.last_candidate_days_since_created if self.state.last_candidate_days_since_created is not None else '未启用'}"
            + (f"\n[上轮排序 Top 5] {', '.join(name for name, _ in self.state.last_ranked[:5])}" if self.state.last_ranked else "")
        )

        self.state.conversation = [{
            "role": "system",
            "content": system_content + summary_text,
        }] + recent_msgs

        logger.info(
            f"[Agent] 对话历史已压缩: {len(old_msgs)} 条旧消息 → LLM摘要, "
            f"保留 {len(recent_msgs)} 条近期消息"
        )

    def _generate_summary_with_llm(self, old_msgs: list[dict]) -> str | None:
        """
        用 LLM 将旧消息浓缩为语义摘要。

        Args:
            old_msgs: 要压缩的旧消息列表

        Returns:
            摘要文本，失败时返回 None

        摘要要求：
          - 保留用户核心意图和关键参数
          - 保留已完成的关键操作
          - 保留重要结果结论
          - 去除进度提示、重复内容、无关细节
        """
        SUMMARY_PROMPT = """你是对话历史压缩器。请将以下对话历史浓缩为简洁摘要。

要求：
1. 保留用户的核心意图和关键参数（类别、star范围、时间窗口、top_n等）
2. 保留已完成的关键操作（搜索、扫描、排序、生成报告）
3. 保留重要结果结论（如排名Top项目、增长数据）
4. 去除：进度提示、重复内容、无关细节、tool调用细节
5. 使用简洁的中文，不超过500字

格式示例：
- 用户意图：搜索AI Agent热门项目，top_n=20，时间窗口7天
- 已执行：搜索关键词、扫描star范围1000-5000、批量增长分析
- 关键结果：排名第一为langchain(增长2300 stars)，第二为autogen(增长1800 stars)
- 待跟进：用户曾提到想看新项目，但尚未执行

对话历史：
{conversation_history}

请输出摘要（不要输出任何额外内容）："""

        # ── 步骤1：构建对话历史文本 ─────────────────────────────────────────────────────
        #    简化格式，减少 token 消耗
        history_parts = []

        # 子分支：保留之前的摘要（如果有）
        if self.state.conversation_summary:
            history_parts.append(f"[之前的摘要]\n{self.state.conversation_summary}")

        # 子分支：遍历旧消息，按角色提取关键内容
        for msg in old_msgs:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                history_parts.append(f"[用户] {content}")
            elif role == "assistant":
                # assistant 消息可能很长，截取关键部分（最多500字符）
                history_parts.append(f"[助手] {content[:500] if content else '(tool调用)'}")
            elif role == "tool":
                # tool 结果保留关键信息提示，不保留完整数据
                history_parts.append(f"[Tool结果] (已执行)")

        history_text = "\n".join(history_parts)

        # ── 步骤2：调用 LLM（低成本配置）───────────────────────────────────────────────
        #    低温度、短输出、不需要 thinking
        response = self._request_llm(
            messages=[{"role": "user", "content": SUMMARY_PROMPT.format(conversation_history=history_text)}],
            temperature=0.1,
            max_tokens=600,
            log_prefix="[摘要生成]",
            enable_thinking=False,
        )

        # ── 步骤3：提取摘要文本 ───────────────────────────────────────────────────────────
        if response:
            choice = response.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "")
            if content:
                return content.strip()

        return None
