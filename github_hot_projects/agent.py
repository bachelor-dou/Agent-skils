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
    MIN_STAR_FILTER,
    SEARCH_KEYWORDS,
    STAR_RANGE_MIN,
    STAR_RANGE_MAX,
    TIME_WINDOW_DAYS,
    STAR_GROWTH_THRESHOLD,
    HOT_PROJECT_COUNT,
    HOT_NEW_PROJECT_COUNT,
    NEW_PROJECT_DAYS,
)
from .agent_tools import (
    tool_search_hot_projects,
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
from .common.db import load_db, save_db, save_db_desc_only
from .common.token_manager import TokenManager
from .parsing.arg_validator import validate_tool_args, log_validated_params
from .parsing.prompt_schema import PROMPT_PARAMETER_SCHEMA_CONTEXT
from .parsing.schema import TOOL_SCHEMAS

logger = logging.getLogger("discover_hot")

# Agent 单轮最大 Tool 调用次数（防止无限循环）
MAX_TOOL_CALLS_PER_TURN = 15

# ══════════════════════════════════════════════════════════════════════════════════════
# 提示词（意图分类阶段）
# ══════════════════════════════════════════════════════════════════════════════════════

CONFIRMATION_PROMPT = """你是GitHub热门项目助手。理解用户需求，输出结构化JSON。

""" + PROMPT_PARAMETER_SCHEMA_CONTEXT + """

意图类型（选择最匹配的一个）：
- comprehensive_ranking：综合热榜（全量候选收集，三源合一）
- hot_new_ranking：新项目热榜（全量收集+创建时间过滤）
- keyword_ranking：关键词热榜（只搜索指定关键词/类别，不需scan/trending）
- trending_only：查看Trending页面
- repo_growth：单个仓库增长趋势查询
- repo_description：单个项目功能介绍
- db_info：数据库概览查询

输出JSON（必须严格按此格式，不要输出其他字段）：
{
  "intent_family": "意图类型",
  "intent_label_zh": "中文任务名",
  "specified_params": {"用户明确指定的参数": "值"},
  "ambiguous_fields": ["需要澄清的点，无则为空"],
    "report_requested": true,
  "confirmation_text_zh": "收到！我理解为：……。确认请回复\"开始\"，或直接告诉修改点。"
}

注意：confirmation_text_zh 必须是自然语言文本，不能是JSON或其他结构化格式。
"""

CONFIRMATION_ACK_PROMPT = """判断用户回复是否表示确认执行当前待确认请求。

规则：
- 用户说"可以/执行/开始/继续/就按这个来"等 → {"is_ack": true}
- 用户补充条件、修改参数、提出问题 → {"is_ack": false}

只输出JSON，无其他文字。"""

# ══════════════════════════════════════════════════════════════════════════════════════
# 意图类型 + 工具映射（混合架构核心）
# ══════════════════════════════════════════════════════════════════════════════════════

INTENT_LABELS = {
    "comprehensive_ranking": "综合热榜",
    "hot_new_ranking": "新项目热榜",
    "keyword_ranking": "关键词热榜",       # 新增：只搜索指定关键词/类别
    "trending_only": "Trending热门",
    "repo_growth": "单仓库增长",
    "repo_description": "项目介绍",
    "db_info": "数据库查询",
    "unknown": "未确定请求",
}

# 意图 → 可用工具范围（LLM在此范围内自主选择）
AVAILABLE_TOOLS_BY_INTENT: dict[str, set[str]] = {
    "comprehensive_ranking": {
        "search_hot_projects", "scan_star_range", "fetch_trending",
        "batch_check_growth", "rank_candidates", "generate_report",
    },
    "hot_new_ranking": {
        "search_hot_projects", "scan_star_range", "fetch_trending",
        "batch_check_growth", "rank_candidates", "generate_report",
    },
    "keyword_ranking": {
        "search_hot_projects",                 # 只需要关键词搜索
        "batch_check_growth", "rank_candidates", "generate_report",
    },
    "trending_only": {"fetch_trending"},
    "repo_growth": {"check_repo_growth"},
    "repo_description": {"describe_project"},
    "db_info": {"get_db_info"},
    "unknown": set(),  # 空集，LLM可自主探索
}

# 榜单型意图（需要候选收集→增长计算→排名的完整流程）
RANKING_INTENTS = {"comprehensive_ranking", "hot_new_ranking", "keyword_ranking"}

# 建议的候选收集工具（仅作提示，不强制阻断）
SUGGESTED_COLLECTION_TOOLS_BY_INTENT: dict[str, set[str]] = {
    "comprehensive_ranking": {"search_hot_projects", "scan_star_range", "fetch_trending"},
    "hot_new_ranking": {"search_hot_projects", "scan_star_range", "fetch_trending"},
    "keyword_ranking": {"search_hot_projects"},
}

CONFIRMATION_MODIFICATION_MARKERS = (
    "并且",
    "另外",
    "再加",
    "加上",
    "改成",
    "改为",
    "改下",
    "换成",
    "顺便",
    "同时",
    "但是",
    "不过",
    "以及",
)

CONFIRMATION_QUESTION_MARKERS = ("?", "？", "吗", "怎么", "为何", "为什么", "是不是", "是否")

INTENT_ALIASES = {
    "comprehensive": "comprehensive_ranking",
    "hot_new": "hot_new_ranking",
    "keyword": "keyword_ranking",
    "trending": "trending_only",
    "describe_project": "repo_description",
    "check_repo_growth": "repo_growth",
}

# 参数显示（用于生成确认文本）
PARAM_DISPLAYERS = {
    "categories": lambda v: f"关注方向为{'、'.join(v)}" if isinstance(v, list) and v else None,
    "project_min_star": lambda v: f"关键词搜索最低star为{v}",
    "time_window_days": lambda v: f"统计近{v}天的增长",
    "new_project_days": lambda v: f"只看近{v}天内创建的项目",
    "growth_threshold": lambda v: f"增长门槛为{v}",
    "top_n": lambda v: f"返回前{v}名",
    "trending_range": lambda v: f"Trending范围:{v}",
    "repo": lambda v: f"仓库为{v}" if v else None,
}


@dataclass
class PendingRequest:
    intent_family: str = "unknown"
    intent_label_zh: str = "未确定请求"
    user_specified_params: dict[str, object] = field(default_factory=dict)
    ambiguous_fields: list[str] = field(default_factory=list)
    confirmation_text_zh: str = ""
    report_requested: bool = False
    source_turn_id: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "intent_family": self.intent_family,
            "intent_label_zh": self.intent_label_zh,
            "specified_params": self.user_specified_params,
            "ambiguous_fields": self.ambiguous_fields,
            "report_requested": self.report_requested,
            "confirmation_text_zh": self.confirmation_text_zh,
            "source_turn_id": self.source_turn_id,
        }


@dataclass
class ResolvedRequest:
    intent_family: str = "unknown"
    intent_label_zh: str = "未确定请求"
    resolved_params: dict[str, object] = field(default_factory=dict)
    user_specified_params: dict[str, object] = field(default_factory=dict)
    defaulted_params: dict[str, object] = field(default_factory=dict)
    report_requested: bool = False
    confirmation_text_zh: str = ""

    def requires_full_collection(self) -> bool:
        return self.intent_family in RANKING_INTENTS

    def to_dict(self) -> dict[str, object]:
        return {
            "intent_family": self.intent_family,
            "intent_label_zh": self.intent_label_zh,
            "resolved_params": self.resolved_params,
            "user_specified_params": self.user_specified_params,
            "defaulted_params": self.defaulted_params,
            "report_requested": self.report_requested,
            "confirmation_text_zh": self.confirmation_text_zh,
        }

    def to_execution_context(self) -> str:
        lines = [
            "[已确认请求]",
            f"intent_family={self.intent_family}",
            f"intent_label_zh={self.intent_label_zh}",
            f"可用工具={sorted(AVAILABLE_TOOLS_BY_INTENT.get(self.intent_family, set()))}",
            f"resolved_params={json.dumps(self.resolved_params, ensure_ascii=False, sort_keys=True)}",
        ]
        return "\n".join(lines)

# ══════════════════════════════════════════════════════════════════════════════════════
# SYSTEM_PROMPT（LLM自主决策阶段 - 工具详情已通过API tools参数传递）
# ══════════════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = f"""你是GitHub热门项目发现助手。根据用户需求自主调用提供的工具完成任务。

工具详情已通过API tools参数传递，请参考function定义中的description和parameters。

## 默认配置
- 时间窗口：{TIME_WINDOW_DAYS}天
- 增长阈值：{STAR_GROWTH_THRESHOLD} stars
- 综合榜Top N：{HOT_PROJECT_COUNT}
- 新项目榜Top N：{HOT_NEW_PROJECT_COUNT}
- 新项目窗口：{NEW_PROJECT_DAYS}天
- 搜索类别：{list(SEARCH_KEYWORDS.keys())}

## 调用流程建议（根据意图自主选择）
| 意图 | 推荐流程 | 说明 |
|------|----------|------|
| comprehensive_ranking | search→scan→trending→batch→rank→report | 三源合一，全量收集 |
| hot_new_ranking | search→scan→trending→batch→rank→report | 同上，batch/rank阶段会按new_project_days过滤 |
| keyword_ranking | search→batch→rank→report | 只搜索指定类别，不需要scan/trending |
| trending_only | fetch_trending | 直接获取Trending页面 |
| repo_growth | check_repo_growth | 查询单仓库增长，需repo参数 |
| repo_description | describe_project | 查询单项目功能介绍，需repo参数 |
| db_info | get_db_info | 查询本地DB状态 |

## 核心规则
1. **数据真实性**：必须基于GitHub API或Trending页面，不得编造数据
2. **报告生成**：用户要求"报告"时，排名完成后必须调用generate_report
3. **Trending参数**：未指定→weekly，指定"日榜/周榜/月榜"→daily/weekly/monthly，候选补充→all
4. **参数传递**：用户指定的参数（time_window_days/new_project_days等）必须传递给对应工具
5. **单仓库查询**：用户问"这个项目做什么/功能"→describe_project，问"增长趋势"→check_repo_growth
"""


@dataclass
class AgentState:
    """Agent 运行时状态，在整个会话期间保持。"""
    token_mgr: TokenManager = field(default_factory=TokenManager)
    db: dict = field(default_factory=dict)
    conversation: list[dict] = field(default_factory=list)
    # 缓存：搜索结果、候选列表，供多个 Tool 间共享
    last_search_repos: list[dict] = field(default_factory=list)
    last_candidates: dict[str, dict] = field(default_factory=dict)
    last_candidate_new_project_days: int | None = None
    last_ranked: list[tuple[str, dict]] = field(default_factory=list)
    last_mode: str = "comprehensive"
    last_time_window_days: int = TIME_WINDOW_DAYS
    seen_repos: set[str] = field(default_factory=set)
    current_user_turn: int = 0
    discovery_turn_id: int | None = None
    awaiting_confirmation: bool = False
    pending_request: PendingRequest | None = None
    last_confirmed_request: ResolvedRequest | None = None
    current_turn_tools: set[str] = field(default_factory=set)
    # 对话记忆：历史摘要
    conversation_summary: str = ""  # 早期对话的摘要（压缩后保留）

    def __post_init__(self):
        if not self.db:
            self.db = load_db()

# 对话历史最大消息数（超过后触发压缩）
MAX_CONVERSATION_MESSAGES = 40
# 压缩后保留最近的消息数
KEEP_RECENT_MESSAGES = 10


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
          3. 调用 LLM（带 Tool 定义）
          4. 如果 LLM 选择调用 Tool → 执行 Tool → 将结果加入历史 → 回到 3
          5. 如果 LLM 直接回复 → 返回回复文本
        """
        # 对话历史压缩
        if len(self.state.conversation) > MAX_CONVERSATION_MESSAGES:
            self._compress_conversation()

        if len(user_message) > 2000:
            return "消息过长（超过 2000 字符），请缩短后重试。"

        self.state.current_user_turn += 1
        self.state.conversation.append({"role": "user", "content": user_message})

        intercept_reply, execution_confirmed = self._maybe_handle_confirmation_gate(user_message)
        if intercept_reply is not None:
            self.state.conversation.append({"role": "assistant", "content": intercept_reply})
            return intercept_reply

        self.state.current_turn_tools = set()
        if execution_confirmed:
            self._log_execution_overview()

        for step in range(MAX_TOOL_CALLS_PER_TURN):
            response = self._call_llm(execution_confirmed=execution_confirmed)
            if response is None:
                error_msg = "抱歉，LLM 调用失败，请稍后重试。"
                self.state.conversation.append({"role": "assistant", "content": error_msg})
                if execution_confirmed:
                    self.state.awaiting_confirmation = True
                return error_msg

            message = response.get("choices", [{}])[0].get("message", {})

            # 检查是否有 Tool 调用
            tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                # LLM 直接回复用户
                content = message.get("content", "") or ""
                self.state.conversation.append({"role": "assistant", "content": content})
                return content if content else "（Agent 未生成回复，请重试或换个问法。）"

            # 有 Tool 调用 → 执行每个 Tool
            self.state.conversation.append({
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": tool_calls,
            })

            for tc in tool_calls:
                tool_name = tc.get("function", {}).get("name", "")
                tool_args_str = tc.get("function", {}).get("arguments", "{}")
                tool_call_id = tc.get("id", "")

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

                logger.info(f"[Agent] Tool 调用: {tool_name}({tool_args})")
                try:
                    result = self._execute_tool(tool_name, tool_args)
                except Exception as e:
                    logger.error(f"[Agent] Tool {tool_name} 执行异常: {e}")
                    result = {"error": f"工具执行异常: {e}"}
                if not (isinstance(result, dict) and result.get("error")):
                    self.state.current_turn_tools.add(tool_name)
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

        return "已达到单轮最大 Tool 调用次数，请尝试简化请求。"

    def _call_llm(self, execution_confirmed: bool = False) -> dict | None:
        """调用 LLM（带 Tool 定义）。"""
        messages = list(self.state.conversation)
        if messages and messages[0].get("role") == "system":
            extra_sections = []
            if execution_confirmed:
                extra_sections.append("[执行确认] 用户刚刚已经确认了最新请求中的参数。请直接执行对应的 Tool 流程，不要再次做执行前确认，也不要输出泛化能力介绍。")
            if self.state.last_confirmed_request is not None:
                extra_sections.append(self.state.last_confirmed_request.to_execution_context())
            if extra_sections:
                messages[0] = {
                    "role": "system",
                    "content": messages[0]["content"] + "\n\n" + "\n\n".join(extra_sections),
                }

        return self._request_llm(
            messages=messages,
            tools=TOOL_SCHEMAS,
            temperature=0.3,
            max_tokens=16384,
            log_prefix="[Agent]",
            enable_thinking=False,
            thinking_budget=8192,
        )

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
        """统一的 LLM 请求入口，支持带 Tool 或纯文本确认。"""
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
        if enable_thinking is not None:
            payload["enable_thinking"] = enable_thinking
        if thinking_budget is not None:
            payload["thinking_budget"] = thinking_budget
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        for attempt in range(3):
            try:
                resp = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=300)
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except (ValueError, Exception) as e:
                        logger.error(
                            "[Agent] LLM 响应 JSON 解析失败: %s, body=%s",
                            e, resp.text[:500],
                        )
                        continue
                    # 诊断日志：记录 finish_reason 和 token 用量
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
                logger.warning(
                    "LLM 调用失败: status=%s, body=%s, attempt=%d",
                    resp.status_code, resp.text[:300], attempt + 1,
                )
            except requests.RequestException as e:
                logger.error(f"LLM 请求异常: {e}, attempt={attempt + 1}")

        return None

    def _maybe_handle_confirmation_gate(self, user_message: str) -> tuple[str | None, bool]:
        """所有 Tool 执行前先走确认门禁。返回 (拦截回复, 是否已确认执行)。"""
        text = (user_message or "").strip()
        if not text:
            self.state.awaiting_confirmation = False
            return "请直接告诉我你想看的 GitHub 热门项目需求，我会先把识别到的参数发给你确认。", False

        confirmation_ack = False
        if self.state.awaiting_confirmation:
            confirmation_ack = self._is_confirmation_ack(text)
            if not confirmation_ack and self._should_try_llm_confirmation_ack(text):
                confirmation_ack = self._is_confirmation_ack_via_llm(text)

        if self.state.awaiting_confirmation and confirmation_ack:
            pending = self.state.pending_request
            if pending is not None and pending.ambiguous_fields:
                ambiguous_text = "；".join(pending.ambiguous_fields)
                return f"还有待确认的点：{ambiguous_text}。请先直接说明你的真实意思，我再继续。", False
            self.state.awaiting_confirmation = False
            if pending is not None:
                self.state.last_confirmed_request = self._resolve_pending_request(pending)
                self.state.pending_request = None
            return None, True

        if not self.state.awaiting_confirmation and self._is_confirmation_ack(text):
            self.state.awaiting_confirmation = False
            return "请先直接描述要查询的 GitHub 热门项目需求，我会先把识别到的参数发给你确认。", False

        if self._is_capability_or_greeting(text):
            self.state.awaiting_confirmation = False
            return self._scoped_capability_reply(), False

        confirm_message = self._build_confirmation_message()
        self.state.awaiting_confirmation = True
        return confirm_message, False

    @staticmethod
    def _is_confirmation_ack(message: str) -> bool:
        """判断用户是否在确认执行。"""
        normalized = message.strip().lower().rstrip("。！？!?")
        if not normalized:
            return False

        if len(normalized) > 16:
            return False
        if any(ch.isdigit() for ch in normalized):
            return False
        if HotProjectAgent._contains_confirmation_question_signal(normalized):
            return False
        if HotProjectAgent._looks_like_modification_reply(normalized):
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

    @staticmethod
    def _contains_confirmation_question_signal(text: str) -> bool:
        return any(token in text for token in CONFIRMATION_QUESTION_MARKERS)

    @staticmethod
    def _looks_like_modification_reply(text: str) -> bool:
        return any(marker in text for marker in CONFIRMATION_MODIFICATION_MARKERS)

    def _should_try_llm_confirmation_ack(self, message: str) -> bool:
        """是否触发确认词 LLM 语义兜底（混合门禁）。"""
        pending = self.state.pending_request
        if pending is None:
            return False

        text = (message or "").strip().lower()
        if not text:
            return False

        if len(text) > 24:
            return False

        if any(ch.isdigit() for ch in text):
            return False

        if self._contains_confirmation_question_signal(text):
            return False

        # 出现修改/补充迹象时，不走确认兜底，直接进入重新确认流程。
        if self._looks_like_modification_reply(text):
            return False

        return True

    def _is_confirmation_ack_via_llm(self, message: str) -> bool:
        """LLM 语义兜底：判断用户是否在确认执行。"""
        pending = self.state.pending_request
        if pending is None:
            return False

        payload = {
            "pending_request": pending.to_dict(),
            "user_reply": (message or "").strip(),
        }
        response = self._request_llm(
            messages=[
                {"role": "system", "content": CONFIRMATION_ACK_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            tools=None,
            temperature=0,
            max_tokens=256,
            log_prefix="[Agent-AckJudge]",
            enable_thinking=False,
        )
        if response is None:
            return False

        message_payload = response.get("choices", [{}])[0].get("message", {})
        content = (message_payload.get("content") or "").strip()
        parsed = self._extract_json_object(content)
        if isinstance(parsed, dict):
            for key in ("is_ack", "is_confirmation_ack"):
                value = parsed.get(key)
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    normalized = value.strip().lower()
                    if normalized in {"true", "yes", "1"}:
                        return True
                    if normalized in {"false", "no", "0"}:
                        return False

        normalized = content.lower()
        return normalized in {"true", "yes"}

    @staticmethod
    def _is_capability_or_greeting(message: str) -> bool:
        """识别问候或能力范围咨询，避免 LLM 生成泛化欢迎语。"""
        text = message.strip().lower()
        if not text:
            return False

        scope_keywords = ("github", "热榜", "热门", "trending", "增长", "仓库", "项目", "报告", "新项目")
        capability_keywords = ("你能做什么", "你会什么", "帮助", "help", "支持什么", "有哪些功能")
        greeting_keywords = ("你好", "您好", "hello", "hi", "在吗", "嗨")

        if any(keyword in text for keyword in capability_keywords):
            return True

        if any(text == keyword or text.startswith(f"{keyword}") for keyword in greeting_keywords):
            return not any(keyword in text for keyword in scope_keywords)

        return False

    @staticmethod
    def _scoped_capability_reply() -> str:
        """固定的域内欢迎语，避免能力边界漂移。"""
        return (
            "你好！我是 GitHub 热门项目助手，可以帮你查看综合热榜、新项目热榜、Trending、"
            "单个仓库近期增长、历史数据和报告。直接告诉我你的需求，我会先把识别到的参数发给你确认。"
        )

    def _build_confirmation_message(self) -> str:
        """调用确认解析器，生成并缓存结构化待确认请求。"""
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
            log_prefix="[Agent-Confirm]",
            enable_thinking=False,
        )
        fallback = self._default_confirmation_message()
        if response is None:
            self.state.last_confirmed_request = None
            self.state.pending_request = PendingRequest(
                confirmation_text_zh=fallback,
                source_turn_id=self.state.current_user_turn,
            )
            return fallback

        message = response.get("choices", [{}])[0].get("message", {})
        content = (message.get("content") or "").strip()
        pending = self._parse_pending_request_content(content)
        self.state.last_confirmed_request = None
        self.state.pending_request = pending
        return pending.confirmation_text_zh

    def _build_parse_context_payload(self) -> dict[str, object]:
        recent_messages = [
            {"role": msg["role"], "content": msg.get("content") or ""}
            for msg in self.state.conversation[-8:]
            if msg.get("role") in {"user", "assistant"}
        ]
        # 注意：不传递 last_confirmed_request，避免干扰 LLM 输出格式
        # last_confirmed_request 仅在执行阶段注入到 system prompt
        return {
            "current_user_turn": self.state.current_user_turn,
            "recent_dialogue": recent_messages,
            "pending_request": self.state.pending_request.to_dict() if self.state.pending_request else None,
        }

    @staticmethod
    def _default_confirmation_message() -> str:
        return (
            "收到！我会先按你刚才的 GitHub 热门项目需求整理参数。"
            "确认请回复\"开始\"，或直接告诉我需要修改的地方。"
        )

    @staticmethod
    def _looks_like_structured_confirmation_text(text: str) -> bool:
        stripped = (text or "").strip()
        if not stripped:
            return False

        lowered = stripped.lower()
        structured_keys = (
            '"intent_family"',
            '"intent_label_zh"',
            '"specified_params"',
            '"ambiguous_fields"',
            '"confirmation_text_zh"',
        )
        if any(key in lowered for key in structured_keys):
            return True

        return stripped.startswith("```") or (stripped.startswith("{") and stripped.endswith("}"))

    def _sanitize_confirmation_fallback(self, content: str) -> str:
        stripped = (content or "").strip()
        if not stripped:
            return self._default_confirmation_message()
        if self._looks_like_structured_confirmation_text(stripped):
            return self._default_confirmation_message()
        return stripped

    def _parse_pending_request_content(self, content: str) -> PendingRequest:
        fallback = self._sanitize_confirmation_fallback(content)
        payload = self._extract_json_object(content)
        if not isinstance(payload, dict):
            return PendingRequest(
                confirmation_text_zh=fallback,
                source_turn_id=self.state.current_user_turn,
            )

        intent_family = self._normalize_intent_family(payload.get("intent_family"))
        specified_params = payload.get("specified_params")
        if not isinstance(specified_params, dict):
            specified_params = {}
        ambiguous_fields = payload.get("ambiguous_fields")
        if not isinstance(ambiguous_fields, list):
            ambiguous_fields = []

        pending = PendingRequest(
            intent_family=intent_family,
            intent_label_zh=str(payload.get("intent_label_zh") or INTENT_LABELS[intent_family]),
            user_specified_params=specified_params,
            ambiguous_fields=[str(item) for item in ambiguous_fields if str(item).strip()],
            report_requested=bool(payload.get("report_requested")),
            source_turn_id=self.state.current_user_turn,
        )
        raw_confirmation = str(payload.get("confirmation_text_zh") or "").strip()
        if raw_confirmation and not self._looks_like_structured_confirmation_text(raw_confirmation):
            pending.confirmation_text_zh = raw_confirmation
        else:
            pending.confirmation_text_zh = self._render_pending_request_text(pending)
        return pending

    @staticmethod
    def _extract_json_object(content: str) -> dict | None:
        text = (content or "").strip()
        if not text:
            return None
        for candidate in (text, text[text.find("{"):] if "{" in text else ""):
            if not candidate:
                continue
            try:
                value, _ = json.JSONDecoder().raw_decode(candidate)
            except ValueError:
                continue
            if isinstance(value, dict):
                return value
        return None

    @staticmethod
    def _normalize_intent_family(raw_intent: object) -> str:
        if not isinstance(raw_intent, str) or not raw_intent.strip():
            return "unknown"
        normalized = raw_intent.strip().lower()
        normalized = INTENT_ALIASES.get(normalized, normalized)
        return normalized if normalized in INTENT_LABELS else "unknown"

    def _render_pending_request_text(self, pending: PendingRequest) -> str:
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
        if pending.ambiguous_fields:
            ambiguous_text = "；".join(pending.ambiguous_fields)
            body = f"{body}。另外还需要确认：{ambiguous_text}"
        if not body:
            body = "你的 GitHub 热门项目需求"
        return f"收到！我理解为：{body}。确认请回复\"开始\"，或直接告诉我需要修改的地方。"

    def _resolve_pending_request(self, pending: PendingRequest) -> ResolvedRequest:
        defaults = self._default_params_for_intent(pending.intent_family)
        resolved_params = dict(defaults)
        resolved_params.update(pending.user_specified_params)
        defaulted_params = {
            key: value for key, value in defaults.items()
            if key not in pending.user_specified_params
        }
        return ResolvedRequest(
            intent_family=pending.intent_family,
            intent_label_zh=pending.intent_label_zh,
            resolved_params=resolved_params,
            user_specified_params=dict(pending.user_specified_params),
            defaulted_params=defaulted_params,
            report_requested=pending.report_requested,
            confirmation_text_zh=pending.confirmation_text_zh,
        )

    @staticmethod
    def _default_params_for_intent(intent_family: str) -> dict[str, object]:
        if intent_family == "comprehensive_ranking":
            return {
                "mode": "comprehensive",
                "top_n": HOT_PROJECT_COUNT,
                "time_window_days": TIME_WINDOW_DAYS,
                "growth_threshold": STAR_GROWTH_THRESHOLD,
                "project_min_star": MIN_STAR_FILTER,
                "min_star": STAR_RANGE_MIN,
                "max_star": STAR_RANGE_MAX,
                "trending_range": "all",
            }
        if intent_family == "hot_new_ranking":
            return {
                "mode": "hot_new",
                "top_n": HOT_NEW_PROJECT_COUNT,
                "time_window_days": TIME_WINDOW_DAYS,
                "new_project_days": NEW_PROJECT_DAYS,
                "growth_threshold": STAR_GROWTH_THRESHOLD,
                "project_min_star": MIN_STAR_FILTER,
                "min_star": STAR_RANGE_MIN,
                "max_star": STAR_RANGE_MAX,
                "trending_range": "all",
            }
        if intent_family == "trending_only":
            return {"trending_range": "weekly"}
        if intent_family == "repo_growth":
            return {"time_window_days": TIME_WINDOW_DAYS}
        return {}

    def _merge_request_defaults_into_tool_args(self, name: str, args: dict) -> dict:
        merged = dict(args)
        resolved_request = self.state.last_confirmed_request
        if resolved_request is None:
            return merged

        for key, value in resolved_request.resolved_params.items():
            merged.setdefault(key, value)

        # 榜单型任务：fetch_trending 使用 trending_range="all"
        if resolved_request.requires_full_collection() and name == "fetch_trending":
            merged["trending_range"] = "all"

        if name == "rank_candidates":
            mode = "hot_new" if resolved_request.intent_family == "hot_new_ranking" else "comprehensive"
            merged.setdefault("mode", mode)

        return merged

    def _log_execution_overview(self) -> None:
        """在执行开始时打印本轮完整参数快照，便于排查。"""
        resolved_request = self.state.last_confirmed_request
        if resolved_request is None:
            logger.info(
                "[Agent] 运行参数总览: turn=%s | 当前无已确认请求（将按 LLM/工具默认参数执行）。",
                self.state.current_user_turn,
            )
            return

        mode = resolved_request.resolved_params.get("mode")
        mode_text = mode if isinstance(mode, str) else None
        persistence_policy = self._persistence_policy_for_request(mode=mode_text)

        logger.info(
            "[Agent] 运行参数总览: turn=%s | intent=%s(%s) | report_requested=%s | persistence_policy=%s",
            self.state.current_user_turn,
            resolved_request.intent_family,
            resolved_request.intent_label_zh,
            resolved_request.report_requested,
            persistence_policy,
        )
        logger.info(
            "[Agent] 运行参数(user_specified): %s",
            json.dumps(resolved_request.user_specified_params, ensure_ascii=False, sort_keys=True, default=str),
        )
        logger.info(
            "[Agent] 运行参数(resolved): %s",
            json.dumps(resolved_request.resolved_params, ensure_ascii=False, sort_keys=True, default=str),
        )

    def _check_suggested_collection_tools(self, tool_name: str) -> list[str]:
        """检查建议的候选收集工具是否已调用（仅作提示，不强制阻断）。

        LLM自主决策为主，硬编码只做建议提示：
        - 返回缺失的建议工具列表
        - 不强制阻断执行，只记录日志警告
        """
        resolved_request = self.state.last_confirmed_request
        if resolved_request is None or not resolved_request.requires_full_collection():
            return []
        if tool_name not in {"batch_check_growth", "rank_candidates"}:
            return []
        suggested_tools = SUGGESTED_COLLECTION_TOOLS_BY_INTENT.get(resolved_request.intent_family, set())
        if not suggested_tools:
            return []
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
        """路由并执行 Tool 调用。

        参数校验由 validate_tool_args 统一处理（类型/边界校验），
        不做语义纠偏，信任 LLM 的参数判断。
        """
        state = self.state
        prepared_args = self._merge_request_defaults_into_tool_args(name, args)
        validated = validate_tool_args(name, prepared_args)
        log_validated_params(name, args, prepared_args, validated)
        self._maybe_reset_discovery_state(name, validated)

        if name == "search_hot_projects":
            result = tool_search_hot_projects(
                state.token_mgr,
                categories=validated.get("categories"),
                project_min_star=validated.get("project_min_star", MIN_STAR_FILTER),
                max_pages=validated.get("max_pages", 3),
                new_project_days=validated.get("new_project_days"),
            )
            raw_repos = result.pop("_raw_repos", [])
            state.last_search_repos = raw_repos
            state.seen_repos.update(r["full_name"] for r in raw_repos)
            return result

        elif name == "scan_star_range":
            result = tool_scan_star_range(
                state.token_mgr,
                min_star=validated.get("min_star", STAR_RANGE_MIN),
                max_star=validated.get("max_star", STAR_RANGE_MAX),
                seen_repos=state.seen_repos,
                new_project_days=validated.get("new_project_days"),
            )
            raw_repos = result.pop("_raw_repos", [])
            state.last_search_repos.extend(raw_repos)
            return result

        elif name == "check_repo_growth":
            repo = validated.get("repo")
            if not repo:
                return {"error": "缺少必需参数 repo（格式: owner/repo）"}
            return tool_check_repo_growth(
                state.token_mgr,
                repo=repo,
                db=state.db,
                time_window_days=validated.get("time_window_days", TIME_WINDOW_DAYS),
            )

        elif name == "batch_check_growth":
            # 检查建议的候选收集工具（仅提示，不强制阻断）
            suggested_tools = self._check_suggested_collection_tools(name)
            if suggested_tools:
                suggested_text = "、".join(suggested_tools)
                logger.warning(
                    "[Agent] batch_check_growth: 建议先调用 %s 以最大化候选覆盖，但LLM可自主决策是否跳过。",
                    suggested_text,
                )
            if not state.last_search_repos:
                return {"error": "没有搜索结果，请先调用 search_hot_projects"}
            time_window_days = validated.get("time_window_days", TIME_WINDOW_DAYS)
            new_project_days = validated.get("new_project_days")
            resolved_request = self.state.last_confirmed_request
            window_specified = "time_window_days" in args or (
                resolved_request is not None and "time_window_days" in resolved_request.user_specified_params
            )
            result = tool_batch_check_growth(
                state.token_mgr,
                repos=state.last_search_repos,
                db=state.db,
                growth_threshold=validated.get("growth_threshold", STAR_GROWTH_THRESHOLD),
                new_project_days=new_project_days,
                time_window_days=time_window_days,
                window_specified=window_specified,
            )
            state.last_candidates = result.get("candidates", {})
            state.last_candidate_new_project_days = new_project_days
            state.last_time_window_days = result.get("time_window_days", time_window_days)
            persistence_policy = self._persistence_policy_for_request()
            if result.get("db_updated", False) and persistence_policy == "desc_only":
                changed = save_db_desc_only(state.db)
                logger.info("[Agent] batch_check_growth 阶段仅持久化 desc 字段 (%d 个项目)。", changed)
            return result

        elif name == "rank_candidates":
            # 检查建议的候选收集工具（仅提示，不强制阻断）
            suggested_tools = self._check_suggested_collection_tools(name)
            if suggested_tools:
                suggested_text = "、".join(suggested_tools)
                logger.warning(
                    "[Agent] rank_candidates: 建议先调用 %s 以最大化候选覆盖，但LLM可自主决策是否跳过。",
                    suggested_text,
                )
            if not state.last_candidates:
                return {"error": "没有候选列表，请先调用 batch_check_growth"}
            mode = validated.get("mode", "comprehensive")
            top_n = validated.get("top_n", HOT_PROJECT_COUNT if mode == "comprehensive" else HOT_NEW_PROJECT_COUNT)
            new_project_days = validated.get("new_project_days")
            result = tool_rank_candidates(
                state.last_candidates,
                top_n=top_n,
                mode=mode,
                db=state.db,
                new_project_days=new_project_days,
                prefiltered_new_project_days=state.last_candidate_new_project_days,
            )
            state.last_ranked = result.pop("_ordered_tuples", [])
            state.last_mode = mode
            return result

        elif name == "describe_project":
            repo = validated.get("repo")
            if not repo:
                return {"error": "缺少必需参数 repo（格式: owner/repo）"}
            return tool_describe_project(repo=repo, db=state.db, token_mgr=state.token_mgr)

        elif name == "generate_report":
            if not state.last_ranked:
                return {"error": "没有排序结果，请先调用 rank_candidates"}
            report_new_project_days = state.last_candidate_new_project_days if state.last_mode == "hot_new" else None
            result = tool_generate_report(
                state.last_ranked,
                state.db,
                mode=state.last_mode,
                new_project_days=report_new_project_days,
                time_window_days=state.last_time_window_days,
            )
            persistence_policy = self._persistence_policy_for_request(mode=state.last_mode)
            if persistence_policy == "desc_only":
                changed = save_db_desc_only(state.db)
                logger.info("[Agent] generate_report 阶段仅持久化 desc 字段 (%d 个项目)。", changed)
            return result

        elif name == "get_db_info":
            return tool_get_db_info(db=state.db, repo=validated.get("repo"))

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

        else:
            return {"error": f"未知 Tool: {name}"}

    def _maybe_reset_discovery_state(self, tool_name: str, args: dict) -> None:
        """在新一轮榜单构建开始前，清理上一轮的候选/去重状态。"""
        is_discovery_bootstrap = tool_name in {"search_hot_projects", "scan_star_range"}
        is_trending_supplement = tool_name == "fetch_trending" and args.get("trending_range") == "all"
        if not is_discovery_bootstrap and not is_trending_supplement:
            return

        current_turn = self.state.current_user_turn
        if self.state.discovery_turn_id == current_turn:
            return

        self.state.last_search_repos = []
        self.state.last_candidates = {}
        self.state.last_candidate_new_project_days = None
        self.state.last_ranked = []
        self.state.last_mode = "comprehensive"
        self.state.last_time_window_days = TIME_WINDOW_DAYS
        self.state.seen_repos.clear()
        self.state.discovery_turn_id = current_turn
        logger.info("[Agent] 检测到新一轮榜单构建，已重置候选、排序和去重状态。")

    @staticmethod
    def _serialize_result(result: dict, max_len: int = 8000) -> str:
        """将 Tool 结果序列化为 JSON 字符串（供 LLM 阅读）。

        超长时按结构智能截断：保留摘要字段，截取列表前 N 项。
        """
        result_str = json.dumps(result, ensure_ascii=False, default=str)
        if len(result_str) <= max_len:
            return result_str

        # 智能截断：找到列表/字典类型的大字段，逐步缩减
        truncated = dict(result)
        list_keys = [k for k, v in truncated.items() if isinstance(v, (list, dict)) and k != "error"]

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

        # 兜底：硬截断
        return s[:max_len] + "\n...(结果已截断)"

    def _compress_conversation(self) -> None:
        """
        压缩对话历史：用 LLM 将早期消息语义摘要化，保留最近 KEEP_RECENT_MESSAGES 条。

        策略：
          1. 提取 system prompt（始终保留）
          2. 用 LLM 将旧消息浓缩为语义摘要
          3. 注入摘要到 system prompt 之后
          4. 保留最近的消息
        """
        conv = self.state.conversation
        if len(conv) <= MAX_CONVERSATION_MESSAGES:
            return

        # 只压缩非 system 消息，system prompt 始终保留基座。
        system_message = next((m for m in conv if m.get("role") == "system"), None)
        system_content = (system_message or {}).get("content") or SYSTEM_PROMPT
        if "[对话历史摘要]" in system_content:
            system_content = system_content.split("\n\n[对话历史摘要]", 1)[0].strip() or SYSTEM_PROMPT

        non_system = [m for m in conv if m.get("role") != "system"]

        if len(non_system) <= KEEP_RECENT_MESSAGES:
            return

        # 要压缩的旧消息
        old_msgs = non_system[:-KEEP_RECENT_MESSAGES]
        recent_msgs = non_system[-KEEP_RECENT_MESSAGES:]

        # 用 LLM 生成语义摘要
        llm_summary = self._generate_summary_with_llm(old_msgs)
        if llm_summary:
            self.state.conversation_summary = llm_summary
        else:
            # fallback: 简单截取（LLM 调用失败时）
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

        # 重建对话：将摘要合并到初始 system prompt 内
        summary_text = (
            f"\n\n[对话历史摘要]\n"
            f"{self.state.conversation_summary}\n"
            f"[当前状态] 搜索结果: {len(self.state.last_search_repos)} 个, "
            f"候选仓库: {len(self.state.last_candidates)} 个, "
            f"已排序: {len(self.state.last_ranked)} 个, "
            f"已扫描: {len(self.state.seen_repos)} 个, "
            f"榜单模式: {self.state.last_mode}, "
            f"增长窗口: {self.state.last_time_window_days} 天, "
            f"创建窗口: {self.state.last_candidate_new_project_days if self.state.last_candidate_new_project_days is not None else '未启用'}"
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

        # 构建对话历史文本（简化格式，减少 token）
        history_parts = []
        if self.state.conversation_summary:
            history_parts.append(f"[之前的摘要]\n{self.state.conversation_summary}")

        for msg in old_msgs:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                history_parts.append(f"[用户] {content}")
            elif role == "assistant":
                # assistant 消息可能很长，截取关键部分
                history_parts.append(f"[助手] {content[:500] if content else '(tool调用)'}")
            elif role == "tool":
                # tool 结果保留关键信息提示，不保留完整数据
                history_parts.append(f"[Tool结果] (已执行)")

        history_text = "\n".join(history_parts)

        # 调用 LLM（低成本配置）
        response = self._request_llm(
            messages=[{"role": "user", "content": SUMMARY_PROMPT.format(conversation_history=history_text)}],
            temperature=0.1,  # 低温度，稳定输出
            max_tokens=600,
            log_prefix="[摘要生成]",
            enable_thinking=False,  # 不需要思考过程
        )

        if response:
            choice = response.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "")
            if content:
                return content.strip()

        return None
