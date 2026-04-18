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
  - execution/ — 执行层：Tool 实现、统一发现管道

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
    TOOL_SCHEMAS,
    coerce_non_negative_int,
    coerce_optional_positive_int,
    coerce_positive_int,
    coerce_ranking_mode,
    coerce_star_range,
    tool_search_hot_projects,
    tool_scan_star_range,
    tool_check_repo_growth,
    tool_batch_check_growth,
    tool_rank_candidates,
    tool_describe_project,
    tool_generate_report,
    tool_get_db_info,
    tool_fetch_trending,
)
from .common.db import load_db, save_db
from .common.token_manager import TokenManager
from .parsing import (
    is_new_project_workflow,
    is_comprehensive_ranking,
    is_realtime_refresh,
    latest_user_message,
    extract_time_window_days,
    extract_creation_window_days,
    has_explicit_creation_window,
    extract_top_n,
    has_explicit_top_n,
    normalize_tool_args,
    resolve_time_window_days,
    resolve_new_project_days,
    resolve_workflow_mode,
    log_effective_tool_params,
    log_request_summary,
    workflow_mode_source,
    time_window_source,
    creation_window_source,
    arg_source,
    mode_source,
    top_n_source,
    force_refresh_source,
)

logger = logging.getLogger("discover_hot")

# Agent 单轮最大 Tool 调用次数（防止无限循环）
MAX_TOOL_CALLS_PER_TURN = 15

# System Prompt：指导 LLM 如何使用 Tools
SYSTEM_PROMPT = f"""你是一个 GitHub 热门项目发现助手。你可以帮用户搜索、分析和发现近期增长最快的开源项目。

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

## 用户可自定义的参数（以下均有默认值，用户不指定则使用默认值）
| 参数 | Tool | 说明 |
|------|------|------|
| top_n | rank_candidates | 返回前 N 个项目（“前50”、“前20”） |
| time_window_days | check_repo_growth / batch_check_growth | 增长统计窗口（“近10天热榜”=10天） |
| growth_threshold | batch_check_growth | 增长阈值（“增长 >= 300”） |
| min_stars | search_hot_projects | 最低 star 过滤线（"至少 2000 star"） |
| min_star / max_star | scan_star_range | star 范围扫描区间（"5000-20000 star"） |
| new_project_days | search_hot_projects / scan_star_range / batch_check_growth / rank_candidates | 新项目创建窗口（仅在“30天内创建的新项目”这类明确创建时间语义下使用） |
| categories | search_hot_projects | 搜索类别（"AI Agent"、"Database"等） |
| since / language / spoken_language / include_all_periods | fetch_trending | Trending 时间范围、语言与多时间维度补源 |

## 两大核心工作流

### 工作流 1 — 综合热门排名（默认）
用户意图示例："帮我查近期GitHub热门项目"、"热门榜前50"
数据源：search_hot_projects(所有类别) + scan_star_range + fetch_trending，三源互补全量覆盖。
步骤：
  1. search_hot_projects → 关键词搜索全部类别
  2. scan_star_range → star 范围扫描补充覆盖
    3. fetch_trending(include_all_periods=true) → 抓取 daily / weekly / monthly 三档 Trending 去重补充
  4. batch_check_growth → 批量计算增长，筛选候选
  5. rank_candidates(mode="comprehensive") → 综合评分排序
    6. generate_report → 输出 Markdown 报告

### 工作流 2 — 新项目热度排名
用户意图示例："最近有什么新项目比较火"、"近一个月的新项目排名"
数据源与工作流 1 相同（三源全量收集），但在搜索/扫描阶段即前置过滤创建时间，只采集新项目创建窗口内的仓库（GitHub API created:>=date）。搜索和扫描的 star 范围统一使用默认值（STAR_RANGE_MIN={STAR_RANGE_MIN}），用户指定则用用户的。
步骤：
    1. search_hot_projects(new_project_days=N) → 关键词搜索，附加 created:>=date 过滤
    2. scan_star_range(new_project_days=N) → star 范围扫描，同样附加 created:>=date 过滤
    3. fetch_trending(include_all_periods=true) → 抓取 daily / weekly / monthly 三档 Trending 去重补充
    4. batch_check_growth(new_project_days=N, time_window_days=M) → 对采集到的新项目按指定增长窗口批量计算增长（仍会二次校验创建时间）
    5. rank_candidates(mode="hot_new", new_project_days=N) → 按增长量排序
说明：只有“新项目/新创建/30天内创建”等明确创建时间语义才进入 hot_new。用户说“近10天/近30天”默认指增长统计窗口，不直接等于新项目创建窗口；若未明确指定创建窗口，则 hot_new 默认使用 {NEW_PROJECT_DAYS} 天。

### 工作流 3 — Trending 浏览
用户意图示例："看看 GitHub Trending"、"今日热门 Python 项目"
步骤：fetch_trending(since="weekly" unless user explicitly requests daily/weekly/monthly, language / spoken_language) → 直接展示
说明：不做增长计算，直接展示 Trending 页面数据。用户未明确指定周期时，默认返回 weekly。

## 辅助功能

### 单项目信息查询
用户意图示例："查一下 vllm-project/vllm 的情况"、"这个项目最近增长怎么样"
- check_repo_growth(repo="owner/repo") → 返回实时 star、近期增长、项目基本信息及 LLM 生成的详细描述
- get_db_info(repo="owner/repo") → 查询本地 DB 中该仓库的历史信息

### 项目描述
- describe_project(repo="owner/repo") → LLM 生成 200-400 字中文详细描述（README 浓缩摘要）

### 数据库概览
- get_db_info() → DB 状态、项目总数、更新日期

## 注意事项
- 搜索和增长计算需要较长时间，请告知用户正在处理
- 结果以结构化方式呈现，重点突出增长数据
- 对话中保持上下文，用户可以基于上次搜索结果继续操作
- 如果用户意图不明确（比如没说清楚排名模式、类别、数量），请先确认再执行
- 用户明确要求"报告"、"报告链接"、"榜单链接"、"HTML 链接"时，完成排名后必须调用 generate_report，再把可打开的报告文件名或链接返回给用户
- 用户直接查看 Trending 而未指定日/周/月时，默认返回 weekly；只有在综合/新项目工作流补源时才使用 include_all_periods=true 抓取三档并去重
- 用户说"新项目"、"新创建"、"新仓库"等明确提示词时，才进入 hot_new；若未明确提到新项目，则热榜默认走 comprehensive
- 用户说"近7天"、"近10天"、"近30天"等时间表述时，默认指增长统计窗口，而不是新项目创建窗口；只有"30天内创建的新项目"这类明确创建时间语义才映射为 new_project_days
- 用户明确要求"实时/最新/强制刷新/实时热榜"时，batch_check_growth 应传 force_refresh=true，跳过 DB 差值与增长缓存
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
    # 对话记忆：历史摘要
    conversation_summary: str = ""  # 早期对话的摘要（压缩后保留）

    def __post_init__(self):
        if not self.db:
            self.db = load_db()

# 对话历史最大消息数（超过后触发压缩）
MAX_CONVERSATION_MESSAGES = 30
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

        self.state.current_user_turn += 1
        self.state.conversation.append({"role": "user", "content": user_message})

        for step in range(MAX_TOOL_CALLS_PER_TURN):
            response = self._call_llm()
            if response is None:
                return "抱歉，LLM 调用失败，请稍后重试。"

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
                result_str = self._serialize_result(result)

                self.state.conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result_str,
                })

        return "已达到单轮最大 Tool 调用次数，请尝试简化请求。"

    def _call_llm(self) -> dict | None:
        """调用 LLM（带 Tool 定义）。"""
        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": LLM_MODEL,
            "messages": self.state.conversation,
            "tools": TOOL_SCHEMAS,
            "tool_choice": "auto",
            "temperature": 0.3,
            "max_tokens": 16384,
        }

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
                        "[Agent] LLM 响应: finish=%s, prompt_tokens=%s, "
                        "completion_tokens=%s, reasoning_tokens=%s",
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
                            "[Agent] LLM 返回空 content 且无 tool_calls "
                            "(finish=%s, reasoning_tokens=%s), attempt=%d",
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

    def _execute_tool(self, name: str, args: dict) -> dict:
        """
        路由并执行 Tool 调用。

        根据 Tool 名分发到对应的实现函数，
        处理状态缓存（搜索结果、候选列表等在 Tool 间共享）。
        """
        state = self.state
        user_msg = self._latest_user_message()
        normalized_args = self._normalize_tool_args(name, args)
        self._maybe_reset_discovery_state(name, normalized_args)
        effective_new_project_days = self._resolve_new_project_days(name, normalized_args)
        effective_time_window_days = self._resolve_time_window_days(name, normalized_args)
        effective_workflow_mode = self._resolve_effective_workflow_mode(name, normalized_args)

        if name == "search_hot_projects":
            min_stars = coerce_positive_int(normalized_args.get("min_stars"), STAR_RANGE_MIN)
            max_pages = coerce_positive_int(normalized_args.get("max_pages"), 3)
            log_effective_tool_params(name, [
                ("workflow_mode", effective_workflow_mode, workflow_mode_source(effective_workflow_mode, user_msg)),
                ("min_stars", min_stars, arg_source(args, "min_stars", min_stars)),
                ("max_pages", max_pages, arg_source(args, "max_pages", max_pages)),
                ("creation_window_days", effective_new_project_days, creation_window_source(normalized_args, effective_new_project_days, user_msg)),
            ])
            result = tool_search_hot_projects(
                state.token_mgr,
                categories=normalized_args.get("categories"),
                min_stars=min_stars,
                max_pages=max_pages,
                new_project_days=effective_new_project_days,
            )
            # 缓存搜索结果
            raw_repos = result.pop("_raw_repos", [])
            state.last_search_repos = raw_repos
            state.seen_repos.update(r["full_name"] for r in raw_repos)
            return result

        elif name == "scan_star_range":
            min_star, max_star = coerce_star_range(
                normalized_args.get("min_star"),
                normalized_args.get("max_star"),
            )
            star_range_source = "default"
            if "min_star" in args or "max_star" in args:
                requested_min = args.get("min_star", STAR_RANGE_MIN)
                requested_max = args.get("max_star", STAR_RANGE_MAX)
                if requested_min == min_star and requested_max == max_star:
                    star_range_source = "tool_args"
                else:
                    star_range_source = "normalized"
            log_effective_tool_params(name, [
                ("workflow_mode", effective_workflow_mode, workflow_mode_source(effective_workflow_mode, user_msg)),
                ("star_range", f"{min_star}..{max_star}", star_range_source),
                ("creation_window_days", effective_new_project_days, creation_window_source(normalized_args, effective_new_project_days, user_msg)),
            ])
            result = tool_scan_star_range(
                state.token_mgr,
                min_star=min_star,
                max_star=max_star,
                seen_repos=state.seen_repos,
                new_project_days=effective_new_project_days,
            )
            raw_repos = result.pop("_raw_repos", [])
            state.last_search_repos.extend(raw_repos)
            return result

        elif name == "check_repo_growth":
            repo = normalized_args.get("repo")
            if not repo:
                return {"error": "缺少必需参数 repo（格式: owner/repo）"}
            log_effective_tool_params(name, [
                ("repo", repo, "tool_args"),
                ("growth_window_days", effective_time_window_days, time_window_source(normalized_args, user_msg)),
            ])
            return tool_check_repo_growth(
                state.token_mgr,
                repo=repo,
                db=state.db,
                time_window_days=effective_time_window_days,
            )

        elif name == "batch_check_growth":
            if not state.last_search_repos:
                return {"error": "没有搜索结果，请先调用 search_hot_projects"}
            force_refresh = bool(normalized_args.get("force_refresh", False)) or self._is_realtime_refresh_request()
            growth_threshold = coerce_non_negative_int(
                normalized_args.get("growth_threshold"),
                STAR_GROWTH_THRESHOLD,
            )
            log_effective_tool_params(name, [
                ("workflow_mode", effective_workflow_mode, workflow_mode_source(effective_workflow_mode, user_msg)),
                ("candidate_repo_count", len(state.last_search_repos), "state"),
                ("growth_threshold", growth_threshold, arg_source(args, "growth_threshold", growth_threshold)),
                ("growth_window_days", effective_time_window_days, time_window_source(normalized_args, user_msg)),
                ("creation_window_days", effective_new_project_days, creation_window_source(normalized_args, effective_new_project_days, user_msg)),
                ("force_refresh", force_refresh, force_refresh_source(normalized_args, force_refresh)),
            ])
            result = tool_batch_check_growth(
                state.token_mgr,
                repos=state.last_search_repos,
                db=state.db,
                growth_threshold=growth_threshold,
                new_project_days=effective_new_project_days,
                time_window_days=effective_time_window_days,
                force_refresh=force_refresh,
            )
            state.last_candidates = result.get("candidates", {})
            state.last_candidate_new_project_days = effective_new_project_days
            state.last_time_window_days = effective_time_window_days
            # 中间落盘
            save_db(state.db)
            return result

        elif name == "rank_candidates":
            if not state.last_candidates:
                return {"error": "没有候选列表，请先调用 batch_check_growth"}
            mode = coerce_ranking_mode(normalized_args.get("mode", "comprehensive"))
            requested_top_n = coerce_optional_positive_int(normalized_args.get("top_n"))
            explicit_top_n = self._extract_requested_top_n()
            if explicit_top_n is not None:
                effective_top_n = explicit_top_n
            elif requested_top_n is None:
                effective_top_n = HOT_NEW_PROJECT_COUNT if mode == "hot_new" else HOT_PROJECT_COUNT
            elif mode == "hot_new" and not self._user_explicitly_requested_top_n():
                # 避免模型在未被用户要求时擅自扩大新项目榜数量。
                effective_top_n = HOT_NEW_PROJECT_COUNT
            else:
                effective_top_n = requested_top_n

            log_effective_tool_params(name, [
                ("mode", mode, mode_source(args, mode, user_msg)),
                ("top_n", effective_top_n, top_n_source(args, effective_top_n, user_msg)),
                ("candidate_count", len(state.last_candidates), "state"),
                ("growth_window_days", state.last_time_window_days, "state"),
                ("creation_window_days", effective_new_project_days, creation_window_source(normalized_args, effective_new_project_days, user_msg)),
            ])

            result = tool_rank_candidates(
                state.last_candidates,
                top_n=effective_top_n,
                mode=mode,
                db=state.db,
                new_project_days=effective_new_project_days,
                prefiltered_new_project_days=state.last_candidate_new_project_days,
                time_window_days=state.last_time_window_days,
            )
            state.last_ranked = result.pop("_ordered_tuples", [])
            state.last_mode = mode
            return result

        elif name == "describe_project":
            repo = normalized_args.get("repo")
            if not repo:
                return {"error": "缺少必需参数 repo（格式: owner/repo）"}
            return tool_describe_project(
                repo=repo,
                db=state.db,
            )

        elif name == "generate_report":
            if not state.last_ranked:
                return {"error": "没有排序结果，请先调用 rank_candidates"}
            report_new_project_days = state.last_candidate_new_project_days if state.last_mode == "hot_new" else None
            log_effective_tool_params(name, [
                ("mode", state.last_mode, "state"),
                ("ranked_count", len(state.last_ranked), "state"),
                ("growth_window_days", state.last_time_window_days, "state"),
                ("creation_window_days", report_new_project_days, "state" if report_new_project_days is not None else "unused"),
            ])
            result = tool_generate_report(
                state.last_ranked,
                state.db,
                mode=state.last_mode,
                new_project_days=report_new_project_days,
                time_window_days=state.last_time_window_days,
            )
            save_db(state.db)
            return result

        elif name == "get_db_info":
            return tool_get_db_info(
                db=state.db,
                repo=normalized_args.get("repo"),
            )

        elif name == "fetch_trending":
            log_effective_tool_params(name, [
                ("since", normalized_args.get("since", "weekly"), arg_source(normalized_args, "since")),
                ("language", normalized_args.get("language", "") or "all", arg_source(normalized_args, "language")),
                ("spoken_language", normalized_args.get("spoken_language", "") or "all", arg_source(normalized_args, "spoken_language")),
                ("include_all_periods", normalized_args.get("include_all_periods", False), arg_source(normalized_args, "include_all_periods")),
            ])
            result = tool_fetch_trending(
                since=normalized_args.get("since", "weekly"),
                language=normalized_args.get("language", ""),
                spoken_language=normalized_args.get("spoken_language", ""),
                include_all_periods=normalized_args.get("include_all_periods", False),
            )
            # 路径 2：将 Trending 仓库加入 search_repos 缓存，后续可用于 batch_check_growth
            raw_repos = result.pop("_raw_repos", [])
            for r in raw_repos:
                fn = r["full_name"]
                if fn not in state.seen_repos:
                    state.seen_repos.add(fn)
                    state.last_search_repos.append({
                        "full_name": fn,
                        "star": r["star"],
                        "description": r.get("description", ""),
                        "language": r.get("language", ""),
                        "_raw": {
                            "full_name": fn,
                            "stargazers_count": r["star"],
                            "forks_count": r.get("forks", 0),
                            "description": r.get("description", ""),
                            "language": r.get("language", ""),
                            "topics": [],
                        },
                    })
            return result

        else:
            return {"error": f"未知 Tool: {name}"}

    def _latest_user_message(self) -> str:
        """返回最近一条用户消息的小写文本。"""
        return latest_user_message(self.state.conversation)

    # ── 以下方法委托给 parsing 层，保持 API 兼容 ──

    def _normalize_tool_args(self, tool_name: str, args: dict) -> dict:
        return normalize_tool_args(tool_name, args, self._latest_user_message())

    def _resolve_time_window_days(self, tool_name: str, args: dict) -> int:
        return resolve_time_window_days(tool_name, args, self._latest_user_message())

    def _resolve_new_project_days(self, tool_name: str, args: dict) -> int | None:
        return resolve_new_project_days(tool_name, args, self._latest_user_message())

    def _resolve_effective_workflow_mode(self, tool_name: str, args: dict) -> str | None:
        return resolve_workflow_mode(tool_name, args, self._latest_user_message())

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
        log_request_summary(self._latest_user_message())

    def _is_explicit_comprehensive_ranking_request(self) -> bool:
        return is_comprehensive_ranking(self._latest_user_message())

    def _extract_requested_time_window_days(self) -> int | None:
        return extract_time_window_days(self._latest_user_message())

    def _extract_requested_creation_window_days(self) -> int | None:
        return extract_creation_window_days(self._latest_user_message())

    def _has_explicit_creation_window_request(self) -> bool:
        return has_explicit_creation_window(self._latest_user_message())

    def _is_new_project_workflow_request(self) -> bool:
        return is_new_project_workflow(self._latest_user_message())

    def _is_realtime_refresh_request(self) -> bool:
        return is_realtime_refresh(self._latest_user_message())

    def _user_explicitly_requested_top_n(self) -> bool:
        return has_explicit_top_n(self._latest_user_message())

    def _extract_requested_top_n(self) -> int | None:
        return extract_top_n(self._latest_user_message())

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
        压缩对话历史：将早期消息摘要化，保留最近 KEEP_RECENT_MESSAGES 条。

        策略：
          1. 提取 system prompt（始终保留）
          2. 将中间的旧消息提取为摘要文本
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

        # 生成摘要：提取用户消息和关键 assistant 回复
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
            # tool 调用和结果省略

        self.state.conversation_summary = "\n".join(summary_parts[-20:])  # 保留最近 20 条

        # 重建对话
        summary_msg = {
            "role": "system",
            "content": (
                f"[对话历史摘要]\n"
                f"以下是之前对话的关键内容：\n{self.state.conversation_summary}\n"
                f"[当前状态] 搜索结果: {len(self.state.last_search_repos)} 个, "
                f"候选仓库: {len(self.state.last_candidates)} 个, "
                f"已排序: {len(self.state.last_ranked)} 个, "
                f"已扫描: {len(self.state.seen_repos)} 个, "
                f"榜单模式: {self.state.last_mode}, "
                f"增长窗口: {self.state.last_time_window_days} 天, "
                f"创建窗口: {self.state.last_candidate_new_project_days if self.state.last_candidate_new_project_days is not None else '未启用'}"
            ),
        }

        self.state.conversation = initial_system + [summary_msg] + recent_msgs
        logger.info(
            f"[Agent] 对话历史已压缩: {len(old_msgs)} 条旧消息 → 摘要, "
            f"保留 {len(recent_msgs)} 条近期消息"
        )
