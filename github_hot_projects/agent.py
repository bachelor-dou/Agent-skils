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

CONFIRMATION_PROMPT = f"""你是 GitHub 热门项目助手的第一阶段参数解析器。

你的任务只有两件事：
1. 从当前用户消息、最近对话、上一条待确认请求中，识别用户真正要执行的 GitHub 热门项目任务。
2. 输出结构化 JSON，既保留内部字段，也生成一条面向用户的中文确认语句。

严格要求：
1. 只处理这个 agent 支持的能力：综合热榜、新项目热榜、Trending、单仓库增长、仓库描述、数据库概览、报告。
2. 只提取用户明确指定或刚刚修改的参数；不要替用户补默认值。
3. 如果用户是在修改上一条待确认请求，要输出“合并后的最新请求”，不是只输出增量。
4. confirmation_text_zh 必须是自然中文，不要出现参数英文名、Tool 名、JSON 字段名。
5. 只输出一个 JSON object，不要输出 Markdown、代码块或额外解释。
6. 如果存在歧义，在 ambiguous_fields 中指出；confirmation_text_zh 里也要用中文把歧义说清楚。

{PROMPT_PARAMETER_SCHEMA_CONTEXT}

输出 JSON schema：
{{
  "intent_family": "comprehensive_ranking | hot_new_ranking | trending_only | repo_growth | repo_description | db_info | report_only | capability_or_greeting | unknown",
  "intent_label_zh": "给用户看的中文任务名称",
  "specified_params": {{"只放用户明确指定的参数": "值"}},
  "ambiguous_fields": ["需要澄清的点，若无则为空数组"],
  "report_requested": false,
  "confirmation_text_zh": "收到！我理解为：……。确认请回复“开始”，或直接告诉我需要修改的地方。"
}}
"""

CONFIRMATION_ACK_FALLBACK_PROMPT = """你是执行确认判定器。

任务：判断用户回复是否表示“按当前待确认请求直接执行，不再修改参数”。

判定规则：
1. 仅当用户明确表示“可以执行/就按这个来/继续”等同意执行意图，返回 true。
2. 若用户在补充条件、修改参数、提出问题、表达否定或含义不明确，返回 false。
3. 严格依据用户回复，不要臆测。
4. 只输出一个 JSON object，不要输出其他文字。

输出 JSON schema：
{
    "is_confirmation_ack": true
}
"""

INTENT_LABELS = {
    "comprehensive_ranking": "综合热榜",
    "hot_new_ranking": "新项目热榜",
    "trending_only": "Trending 热门",
    "repo_growth": "单仓库增长查询",
    "repo_description": "项目介绍",
    "db_info": "数据库查询",
    "report_only": "报告生成",
    "capability_or_greeting": "问候或能力咨询",
    "unknown": "未确定请求",
}

INTENT_ALIASES = {
    "comprehensive": "comprehensive_ranking",
    "hot_new": "hot_new_ranking",
    "trending": "trending_only",
    "repo_info": "db_info",
    "database_info": "db_info",
    "describe_project": "repo_description",
    "check_repo_growth": "repo_growth",
    "greeting": "capability_or_greeting",
}

RANKING_INTENTS = {"comprehensive_ranking", "hot_new_ranking"}
REQUIRED_COLLECTION_TOOLS = {"search_hot_projects", "scan_star_range", "fetch_trending"}

PARAM_DISPLAYERS = {
    "categories": lambda value: f"关注方向为{'、'.join(value)}" if isinstance(value, list) and value else None,
    "project_min_star": lambda value: f"关键词搜索最低 star 为 {value}",
    "min_star": lambda value: f"扫描最小 star 为 {value}",
    "max_star": lambda value: f"扫描最大 star 为 {value}",
    "time_window_days": lambda value: f"统计近 {value} 天的增长",
    "new_project_days": lambda value: f"只看近 {value} 天内创建的项目",
    "growth_threshold": lambda value: f"star 增长门槛为 {value}",
    "top_n": lambda value: f"返回前 {value} 名",
    "since": lambda value: f"查看 {value} Trending",
    "force_refresh": lambda value: "强制实时刷新" if value else None,
    "repo": lambda value: f"仓库为 {value}" if value else None,
    "include_all_periods": lambda value: "同时抓取日榜、周榜、月榜" if value else None,
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
            f"user_specified_params={json.dumps(self.user_specified_params, ensure_ascii=False, sort_keys=True)}",
            f"resolved_params={json.dumps(self.resolved_params, ensure_ascii=False, sort_keys=True)}",
            f"defaulted_params={json.dumps(self.defaulted_params, ensure_ascii=False, sort_keys=True)}",
        ]
        if self.requires_full_collection():
            lines.append(
                "执行约束：榜单型任务在调用 batch_check_growth 或 rank_candidates 前，必须先完成 search_hot_projects、scan_star_range、fetch_trending 三个候选收集工具；其中 fetch_trending 必须使用 include_all_periods=true。"
            )
        if self.report_requested:
            lines.append("执行约束：用户要求最终输出报告，完成排序后应调用 generate_report。")
        return "\n".join(lines)

# System Prompt：指导 LLM 如何使用 Tools
SYSTEM_PROMPT = f"""你是 GitHub 热门项目发现助手。你可以帮用户搜索、分析和发现近期增长最快的开源项目。

你的回复必须始终围绕这个 agent 的能力范围，不要宣称可以处理通用编程问题、通用问答或与 GitHub 热门项目无关的任务。

你拥有以下能力（通过 Tool 调用实现）：
1. **搜索项目**：按关键词类别搜索（search_hot_projects）或按 star 范围扫描（scan_star_range）
2. **增长分析**：单个仓库增长详情（check_repo_growth）或批量筛选（batch_check_growth）
3. **排序筛选**：两种评分模式 — comprehensive（综合排名）/ hot_new（新项目专榜）
4. **GitHub Trending**：获取 Trending 页面热门仓库（fetch_trending）
5. **描述生成**：调用 LLM 为项目生成详细中文描述（describe_project）
6. **报告输出**：生成 Markdown 格式的热门项目报告（generate_report）
7. **数据库查询**：查询历史数据和仓库信息（get_db_info）

## 当前默认配置
- 时间窗口：{TIME_WINDOW_DAYS} 天
- 增长阈值：>= {STAR_GROWTH_THRESHOLD} stars
- 默认 Top N：{HOT_PROJECT_COUNT}
- 新项目窗口：创建时间 <= {NEW_PROJECT_DAYS} 天
- 可搜索类别：{list(SEARCH_KEYWORDS.keys())}


### Tool 执行前：先确认参数再调用tool执行

## 注意事项
- 所有信息必须真实准确，基于 GitHub API 或者搜索总结得到的实际数据，不得编造或假设数据。(必须严格遵守)
- 搜索和增长计算需要较长时间，请告知用户正在处理
- 结果以结构化方式呈现，重点突出增长数据
- 对话中保持上下文，用户可以基于上次搜索结果继续操作
- 如果用户意图不明确（比如没说清楚排名模式、类别、数量），请先确认再执行
- 用户明确要求"报告"、"报告链接"、"榜单链接"、"HTML 链接"时，完成排名后必须调用 generate_report，再把可打开的报告文件名或链接返回给用户
- 用户直接查看 Trending 而未指定日/周/月时，默认返回 weekly；只有在综合/新项目工作流补源时才使用 include_all_periods=true 抓取三档并去重
- 用户说"新项目"、"新创建"、"新仓库"等明确提示词时，才进入 hot_new；若未明确提到新项目，则热榜默认走 comprehensive
- 用户说"近7天"、"近10天"、"近30天"等时间表述时，默认指增长统计窗口，而不是新项目创建窗口；只有"30天内创建的新项目"这类明确创建时间语义才映射为 new_project_days
- 用户明确要求"实时/最新/强制刷新/实时热榜"时，batch_check_growth 应传 force_refresh=true，跳过 DB 差值并刷新 DB 数据
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
            enable_thinking=True,
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

        keywords = (
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
        )
        if len(normalized) > 10:
            return False

        for kw in keywords:
            # 单字确认词仅接受精确匹配，避免把“是不是”误判为确认。
            if len(kw) == 1:
                if normalized == kw:
                    return True
                continue
            if normalized == kw or normalized.startswith(kw):
                return True
        return False

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

        if any(token in text for token in ("?", "？", "吗", "怎么", "为何", "为什么", "是不是", "是否")):
            return False

        # 出现修改/补充迹象时，不走确认兜底，直接进入重新确认流程。
        modification_markers = (
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
        if any(marker in text for marker in modification_markers):
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
                {"role": "system", "content": CONFIRMATION_ACK_FALLBACK_PROMPT},
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
            value = parsed.get("is_confirmation_ack")
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
        fallback = (
            "收到！我会先按你刚才的 GitHub 热门项目需求整理参数。"
            "确认请回复“开始”，或直接告诉我需要修改的地方。"
        )
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
        return {
            "current_user_turn": self.state.current_user_turn,
            "recent_dialogue": recent_messages,
            "pending_request": self.state.pending_request.to_dict() if self.state.pending_request else None,
            "last_confirmed_request": self.state.last_confirmed_request.to_dict() if self.state.last_confirmed_request else None,
        }

    def _parse_pending_request_content(self, content: str) -> PendingRequest:
        fallback = (
            content or
            "收到！我会先按你刚才的 GitHub 热门项目需求整理参数。确认请回复“开始”，或直接告诉我需要修改的地方。"
        )
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
        pending.confirmation_text_zh = (
            str(payload.get("confirmation_text_zh") or "").strip()
            or self._render_pending_request_text(pending)
        )
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
        return f"收到！我理解为：{body}。确认请回复“开始”，或直接告诉我需要修改的地方。"

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
                "since": "weekly",
                "include_all_periods": True,
                "force_refresh": False,
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
                "since": "weekly",
                "include_all_periods": True,
                "force_refresh": True,
            }
        if intent_family == "trending_only":
            return {"since": "weekly", "include_all_periods": False}
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

        if resolved_request.requires_full_collection() and name == "fetch_trending":
            merged["include_all_periods"] = True

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
        resolved_payload = dict(resolved_request.resolved_params)
        if resolved_payload.get("include_all_periods") is True:
            resolved_payload.setdefault("trending_periods", ["daily", "weekly", "monthly"])
        logger.info(
            "[Agent] 运行参数(resolved): %s",
            json.dumps(resolved_payload, ensure_ascii=False, sort_keys=True, default=str),
        )

    def _missing_required_collection_tools(self, tool_name: str) -> list[str]:
        resolved_request = self.state.last_confirmed_request
        if resolved_request is None or not resolved_request.requires_full_collection():
            return []
        if tool_name not in {"batch_check_growth", "rank_candidates"}:
            return []
        missing = REQUIRED_COLLECTION_TOOLS - self.state.current_turn_tools
        return sorted(missing)

    def _persistence_policy_for_request(self, mode: str | None = None) -> str:
        """返回请求对应的 DB 持久化策略：`full` 或 `desc_only`。"""
        if mode == "hot_new":
            return "desc_only"
        resolved_request = self.state.last_confirmed_request
        if resolved_request is not None and resolved_request.intent_family == "hot_new_ranking":
            return "desc_only"
        return "full"

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
            missing_tools = self._missing_required_collection_tools(name)
            if missing_tools:
                missing_text = "、".join(missing_tools)
                return {"error": f"当前榜单任务缺少必要数据源：{missing_text}。请先补齐这几个候选收集工具再继续。"}
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
                force_refresh=validated.get("force_refresh", False),
                window_specified=window_specified,
            )
            state.last_candidates = result.get("candidates", {})
            state.last_candidate_new_project_days = new_project_days
            state.last_time_window_days = result.get("time_window_days", time_window_days)
            if result.get("db_updated", False):
                if self._persistence_policy_for_request() == "full":
                    save_db(state.db)
                else:
                    changed = save_db_desc_only(state.db)
                    logger.info("[Agent] hot_new 实时模式: batch_check_growth 阶段仅持久化 desc 字段 (%d 个项目)。", changed)
            return result

        elif name == "rank_candidates":
            missing_tools = self._missing_required_collection_tools(name)
            if missing_tools:
                missing_text = "、".join(missing_tools)
                return {"error": f"当前榜单任务缺少必要数据源：{missing_text}。请先补齐这几个候选收集工具再继续。"}
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
                persist_db=self._persistence_policy_for_request(mode=mode) == "full",
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
            if self._persistence_policy_for_request(mode=state.last_mode) == "full":
                save_db(state.db)
            else:
                changed = save_db_desc_only(state.db)
                logger.info("[Agent] hot_new 实时模式: generate_report 阶段仅持久化 desc 字段 (%d 个项目)。", changed)
            return result

        elif name == "get_db_info":
            return tool_get_db_info(db=state.db, repo=validated.get("repo"))

        elif name == "fetch_trending":
            result = tool_fetch_trending(
                since=validated.get("since", "weekly"),
                include_all_periods=validated.get("include_all_periods", False),
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
        is_trending_supplement = tool_name == "fetch_trending" and args.get("include_all_periods", False)
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

        # 分离 system prompt（只保留初始 System Prompt，丢弃旧压缩摘要消息）
        initial_system = [m for m in conv if m.get("role") == "system" and "[对话历史摘要]" not in (m.get("content") or "")]
        non_system = [m for m in conv if m.get("role") != "system" or "[对话历史摘要]" in (m.get("content") or "")]

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

        if initial_system:
            initial_system[0] = {
                "role": "system",
                "content": SYSTEM_PROMPT + summary_text,
            }
        else:
            initial_system = [{"role": "system", "content": summary_text}]

        self.state.conversation = initial_system + recent_msgs
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
