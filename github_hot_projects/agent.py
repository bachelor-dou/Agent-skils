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
    TOOL_SCHEMAS,
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
from .parsing.arg_validator import validate_tool_args, log_validated_params
from .parsing.arg_validator import latest_user_message

logger = logging.getLogger("discover_hot")

# Agent 单轮最大 Tool 调用次数（防止无限循环）
MAX_TOOL_CALLS_PER_TURN = 15

CONFIRMATION_PROMPT = """你是 GitHub 热门项目助手的执行前确认器。

只做一件事：根据最近几轮对话，把用户最新要执行的 GitHub 热门项目相关请求整理成一条中文确认消息。

要求：
1. 只围绕这个 agent 的能力：综合热榜、新项目热榜、Trending、单仓库增长、仓库描述、数据库概览、报告。
2. 不要输出通用编程助手能力，不要出现参数英文名、Tool 名或 JSON。
3. 如果用户是在修改上一条待确认请求，要合并上下文后输出最新版本。
4. 输出必须是一句话，格式固定为：收到！我理解为：……。确认请回复“开始”，或直接告诉我需要修改的地方。
"""

# System Prompt：指导 LLM 如何使用 Tools
SYSTEM_PROMPT = f"""你是一个 GitHub 热门项目发现助手。你可以帮用户搜索、分析和发现近期增长最快的开源项目。

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

## 用户可自定义参数 — 完整语义说明（关键，必须理解）

以下是用户可以通过自然语言指定的全部参数。你在调用 Tool 时必须根据用户消息准确填写，不要遗漏。

| 参数 | 类型 | 约束 | 默认值 | 语义说明 | 用户表达示例 |
|------|------|------|--------|----------|-------------|
| categories | list[str] | 可选值见上方类别列表 | 全部类别 | 搜索的项目领域 | "AI Agent方向"、"数据库相关" |
| project_min_star | int | ≥0 | {MIN_STAR_FILTER} | 关键词搜索最低 star 门槛 | "至少2000star"、"1000星以上" |
| min_star / max_star | int | ≥1 | {STAR_RANGE_MIN}/{STAR_RANGE_MAX} | star范围扫描区间 | "5000到20000 star的项目" |
| time_window_days | int | ≥1 | {TIME_WINDOW_DAYS} | **增长统计窗口**：计算近N天的star增长量 | "近10天热榜"、"最近30天增长" |
| new_project_days | int | ≥1 | None（不过滤） | **创建时间窗口**：只看N天内创建的项目。与time_window_days完全独立 | "近20天内新创建的项目"、"一个月内的新项目" |
| growth_threshold | int | ≥0 | {STAR_GROWTH_THRESHOLD} | 增长量筛选门槛 | "增长>=300"、"增长超过500" |
| top_n | int | 1-200 | comprehensive:{HOT_PROJECT_COUNT}, hot_new:{HOT_NEW_PROJECT_COUNT} | 返回前N个项目 | "前10名"、"榜单前50" |
| mode | enum | comprehensive / hot_new | comprehensive | 排名模式 | "新项目榜"→hot_new；"综合榜"→comprehensive |
| since | enum | daily/weekly/monthly | weekly | Trending时间范围 | "今日热门"→daily；"本周"→weekly |
| force_refresh | bool | — | false | 强制实时刷新，跳过缓存 | "实时热榜"、"强制刷新" |
| repo | str | owner/repo格式 | — | 指定仓库 | "查一下vllm-project/vllm" |
| include_all_periods | bool | — | false | 抓取三档Trending | 仅在工作流1/2的候选补充阶段使用 |

### ⚠️ 关键区分：time_window_days vs new_project_days
- **time_window_days**（增长窗口）：统计最近N天的star增量。"近10天热榜"→time_window_days=10
- **new_project_days**（创建窗口）：只看创建时间在N天以内的项目。"近20天内新创建的项目"→new_project_days=20
- 两者**完全独立，可以同时指定**。例如"近20天内新创建项目的最近7天增长排名"→new_project_days=20, time_window_days=7
- 用户说"近N天热榜/近N天增长"→默认指time_window_days
- 用户说"最近N天内创建的项目/N天内的新项目/N天内新建的"→指new_project_days
- ⚠️ **歧义模式**："近N天内新项目的热榜"既可能是"N天内创建的新项目"也可能是"近N天增长+新项目模式"，**此时必须确认**

## 参数确认流程（必须遵守）

### 回复语言规范
**所有面向用户的回复中，禁止出现参数英文名称**（如 mode、hot_new、time_window_days、new_project_days、top_n 等）。
必须全部使用中文自然语言描述。例如：
- ❌ "模式 hot_new，time_window_days=10"
- ✅ "新项目热榜，统计近10天的增长"

### Tool 执行前：先确认参数再执行

任何**会触发 Tool 调用**的请求，都必须先用一句话总结你理解的关键参数，等用户确认后再开始调用 Tool。

**确认格式示例**：
> 收到！我将为您生成：**新项目热榜**，统计近 **{TIME_WINDOW_DAYS}天** 的增长（默认），只看 **45天** 内创建的新项目（默认），返回前 **10名**。确认请回复"开始"或直接说需要调整的地方。

> 收到！我将为您生成：**综合热榜**，统计近 **10天** 的增长（您指定的），返回前 **50名**。确认请回复"开始"或直接说需要调整的地方。

**确认规则**：
1. 用一句话列出关键参数：榜单类型、增长统计天数、创建时间限制（如有）、返回数量、特殊过滤条件
2. 括号标注"您指定的"或"默认"
3. "新项目热榜"本身已说明是新项目模式，不需要再额外提到"模式"
4. 等用户确认（"好的"/"确认"/"开始"/"没问题"等肯定回复）后，再调用第一个 Tool
5. 用户在确认时可以修改参数，你需要根据修改重新确认或直接执行

程序层也会拦截未确认的 Tool 请求，因此你不应跳过确认直接执行。

### 歧义时必须先确认

如果你无法确定某个参数值，**不要用默认值代替，在参数确认环节明确询问**。
需要确认的典型歧义场景：
a) "近N天内新项目的热榜"→"近N天"可能是增长统计天数也可能是创建时间限制
b) 用户说"热门项目"但不清楚要综合榜还是新项目榜
c) 参数之间可能有冲突时

**歧义确认消息格式**：
> 您说"近10天内新项目的热榜"，请确认您的意思是：
> (A) 统计近 **10天的增长** + 新项目热榜（创建时间限制用默认{NEW_PROJECT_DAYS}天）
> (B) 只看 **10天内创建的新项目** + 增长统计用默认{TIME_WINDOW_DAYS}天

### 不要遗漏用户明确提到的参数
- 用户说"前10名"→必须传 top_n=10
- 用户说"近20天内新创建"→必须传 new_project_days=20
- 用户说"近10天增长"→必须传 time_window_days=10

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
数据源与工作流 1 相同（三源全量收集），但在搜索/扫描阶段即前置过滤创建时间，只采集新项目创建窗口内的仓库（GitHub API created:>=date）。关键词搜索默认使用 project_min_star={MIN_STAR_FILTER}；Star 范围扫描默认使用 min_star={STAR_RANGE_MIN}、max_star={STAR_RANGE_MAX}；用户指定则用用户的。
步骤：
    1. search_hot_projects(new_project_days=N) → 关键词搜索，附加 created:>=date 过滤
    2. scan_star_range(new_project_days=N) → star 范围扫描，同样附加 created:>=date 过滤
    3. fetch_trending(include_all_periods=true) → 抓取 daily / weekly / monthly 三档 Trending 去重补充
    4. batch_check_growth(new_project_days=N, time_window_days=M) → 对采集到的新项目按指定增长窗口批量计算增长（仍会二次校验创建时间）
    5. rank_candidates(mode="hot_new", new_project_days=N) → 按增长量排序
说明：只有“新项目/新创建/30天内创建”等明确创建时间语义才进入 hot_new。用户说“近10天/近30天”默认指增长统计窗口，不直接等于新项目创建窗口；若未明确指定创建窗口，则 hot_new 默认使用 {NEW_PROJECT_DAYS} 天。

### 工作流 3 — Trending 浏览
用户意图示例："看看 GitHub Trending"、"今日热门项目"
步骤：fetch_trending(since="weekly" unless user explicitly requests daily/weekly/monthly) → 直接展示
说明：不做增长计算，直接展示 Trending 页面数据。用户未明确指定周期时，默认返回 weekly。

## 辅助功能

### 单项目查询 — 增长数据
用户意图示例："查一下 vllm-project/vllm 最近增长怎么样"、"这个仓库近7天涨了多少 star"
- check_repo_growth(repo="owner/repo") → 返回当前 star、近期增长量、增长率
- 仅适合查增长趋势数据，不适合回答"这个项目是做什么的"、"这个项目支持哪些功能"类问题

### 单项目查询 — 功能了解 / 项目介绍
用户意图示例："这个项目是做什么的"、"这个项目能不能用于某某场景"、"帮我介绍一下这个项目"、"这个项目支持哪些 CLI"
- describe_project(repo="owner/repo") → 基于 README 生成 200-400 字中文详细描述，包含功能、特色、适用场景
- ⚠️ 当用户问项目功能、兼容性、使用方式、适用场景时，必须用 describe_project 而不是 check_repo_growth

### 数据库查询
- get_db_info(repo="owner/repo") → 查询本地 DB 中该仓库的历史信息
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
    awaiting_confirmation: bool = False
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
        if execution_confirmed and messages and messages[0].get("role") == "system":
            messages[0] = {
                "role": "system",
                "content": messages[0]["content"] + "\n\n[执行确认] 用户刚刚已经确认了最新请求中的参数。请直接执行对应的 Tool 流程，不要再次做执行前确认，也不要输出泛化能力介绍。",
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

        if self.state.awaiting_confirmation and self._is_confirmation_ack(text):
            self.state.awaiting_confirmation = False
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
        return any(normalized == kw or normalized.startswith(f"{kw}") for kw in keywords)

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
        """调用纯文本 LLM 生成执行前确认消息。"""
        recent_messages = [
            {"role": msg["role"], "content": msg.get("content") or ""}
            for msg in self.state.conversation[-6:]
            if msg.get("role") in {"user", "assistant"}
        ]
        payload_messages = [{"role": "system", "content": CONFIRMATION_PROMPT}] + recent_messages

        response = self._request_llm(
            messages=payload_messages,
            tools=None,
            temperature=0.1,
            max_tokens=256,
            log_prefix="[Agent-Confirm]",
            enable_thinking=False,
        )
        if response is None:
            return (
                "收到！我会先按你刚才的 GitHub 热门项目需求整理参数。"
                "确认请回复“开始”，或直接告诉我需要修改的地方。"
            )

        message = response.get("choices", [{}])[0].get("message", {})
        content = (message.get("content") or "").strip()
        if content:
            return content

        return (
            "收到！我会先按你刚才的 GitHub 热门项目需求整理参数。"
            "确认请回复“开始”，或直接告诉我需要修改的地方。"
        )

    def _execute_tool(self, name: str, args: dict) -> dict:
        """路由并执行 Tool 调用。

        参数校验由 validate_tool_args 统一处理（类型/边界校验），
        不做语义纠偏，信任 LLM 的参数判断。
        """
        state = self.state
        validated = validate_tool_args(name, args)
        log_validated_params(name, args, validated)
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
            if not state.last_search_repos:
                return {"error": "没有搜索结果，请先调用 search_hot_projects"}
            time_window_days = validated.get("time_window_days", TIME_WINDOW_DAYS)
            new_project_days = validated.get("new_project_days")
            result = tool_batch_check_growth(
                state.token_mgr,
                repos=state.last_search_repos,
                db=state.db,
                growth_threshold=validated.get("growth_threshold", STAR_GROWTH_THRESHOLD),
                new_project_days=new_project_days,
                time_window_days=time_window_days,
                force_refresh=validated.get("force_refresh", False),
            )
            state.last_candidates = result.get("candidates", {})
            state.last_candidate_new_project_days = new_project_days
            state.last_time_window_days = time_window_days
            save_db(state.db)
            return result

        elif name == "rank_candidates":
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
                time_window_days=state.last_time_window_days,
            )
            state.last_ranked = result.pop("_ordered_tuples", [])
            state.last_mode = mode
            return result

        elif name == "describe_project":
            repo = validated.get("repo")
            if not repo:
                return {"error": "缺少必需参数 repo（格式: owner/repo）"}
            return tool_describe_project(repo=repo, db=state.db)

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
            save_db(state.db)
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

        # 重建对话：将摘要合并到初始 system prompt 内，避免产生多条独立 system 消息
        summary_text = (
            f"\n\n[对话历史摘要]\n"
            f"以下是之前对话的关键内容：\n{self.state.conversation_summary}\n"
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
            f"[Agent] 对话历史已压缩: {len(old_msgs)} 条旧消息 → 摘要, "
            f"保留 {len(recent_msgs)} 条近期消息"
        )
