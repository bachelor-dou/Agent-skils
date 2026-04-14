"""
ReAct Agent 核心
================
实现 Thought → Action → Observation 循环的自主推理 Agent。

Agent 接收自然语言指令，通过 LLM 自主规划步骤，
调用 Tool 执行操作，观察结果后决定下一步行动，
直到得出最终回复。

核心类：
  - HotProjectAgent: ReAct Agent 主体
  - AgentState:      Agent 运行状态（会话历史、候选缓存、DB 等）

设计特点：
  - 多轮对话：保持会话历史，支持追问和增量操作
  - 状态缓存：搜索结果、候选列表在会话内复用，避免重复请求
  - Tool 路由：LLM 通过 Function Calling 选择 Tool，Agent 执行并返回结果
  - 可扩展：新增 Tool 只需在 agent_tools.py 添加实现 + schema
"""

import json
import logging
from dataclasses import dataclass, field

import requests

from .config import (
    LLM_API_KEY,
    LLM_API_URL,
    LLM_MODEL,
    SEARCH_KEYWORDS,
    TIME_WINDOW_DAYS,
    STAR_GROWTH_THRESHOLD,
    HOT_PROJECT_COUNT,
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
    tool_full_discovery,
    tool_fetch_trending,
)
from .db import load_db, save_db
from .token_manager import TokenManager

logger = logging.getLogger("discover_hot")

# Agent 单轮最大 Tool 调用次数（防止无限循环）
MAX_TOOL_CALLS_PER_TURN = 15

# System Prompt：指导 LLM 如何使用 Tools
SYSTEM_PROMPT = f"""你是一个 GitHub 热门项目发现助手。你可以帮用户搜索、分析和发现近期增长最快的开源项目。

你拥有以下能力（通过 Tool 调用实现）：
1. **搜索项目**：按关键词类别搜索，或按 star 范围扫描
2. **增长分析**：计算单个仓库或批量仓库近期 star 增长
3. **排序筛选**：两种评分模式 — comprehensive（综合排名）/ hot_new（新项目专榜）
4. **GitHub Trending**：获取 Trending 页面热门仓库
5. **描述生成**：调用 LLM 为项目生成详细中文描述
6. **报告输出**：生成 Markdown 格式的热门项目报告
7. **数据库查询**：查询历史数据和仓库信息

当前配置：
- 时间窗口：{TIME_WINDOW_DAYS} 天
- 增长阈值：>= {STAR_GROWTH_THRESHOLD} stars
- 默认 Top N：{HOT_PROJECT_COUNT}
- 可搜索类别：{list(SEARCH_KEYWORDS.keys())}

## 三大核心工作流

### 工作流 1 — 综合热门排名（默认）
全量收集所有数据源，评出近期增长最快的项目综合排行。
步骤：search_hot_projects → fetch_trending → batch_check_growth → rank_candidates(mode="comprehensive") → generate_report
说明：默认搜索全部类别 + 自动包含 Trending 仓库 + comprehensive 评分（综合增长量和增长率，对数压缩，新项目平滑折扣）。
快捷方式：full_discovery（一键执行全部步骤含报告生成）。

### 工作流 2 — 新项目排名
发现最近新冒出来的爆款项目（创建时间 <= {NEW_PROJECT_DAYS} 天）。
步骤：search_hot_projects → fetch_trending → batch_check_growth → rank_candidates(mode="hot_new")
说明：以 star 范围扫描为主要发现手段（scan_star_range），关键词搜索和 Trending 作为补充。hot_new 模式仅筛选新项目，按增长量纯降序排列。

### 工作流 3 — Trending 浏览
直接查看 GitHub Trending 页面上的当前热门项目。
步骤：fetch_trending(since="daily/weekly/monthly", language=可选, spoken_language=可选) → 直接展示结果
说明：不做增长计算和评分，直接展示 Trending 页面数据（含 stars_today）。

## 辅助功能

- **查看类别/项目增长**：check_repo_growth(repo="owner/repo") 查询单个仓库近 {TIME_WINDOW_DAYS} 天增长
- **LLM 项目介绍**：describe_project(repo="owner/repo") 生成 200-400 字中文详细描述
- **数据库查询**：get_db_info(repo=可选) 查询 DB 状态或特定仓库历史信息

## 注意事项
- 搜索和增长计算需要时间，请告知用户正在处理
- 结果以结构化方式呈现，重点突出增长数据
- 对话中保持上下文，用户可以基于上次搜索结果继续操作
- 如果用户意图不明确（比如没说清楚要哪种排名、哪些类别、多少个），请先跟用户确认再执行
- 当你不确定用户说的"新项目"是指什么范围时，说明本系统的定义：创建时间 <= {NEW_PROJECT_DAYS} 天
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
    last_ranked: list[tuple[str, dict]] = field(default_factory=list)
    seen_repos: set[str] = field(default_factory=set)
    # 对话记忆：用户偏好 + 历史摘要
    user_preferences: dict = field(default_factory=dict)  # 用户本次会话中的偏好
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
                except json.JSONDecodeError:
                    tool_args = {}

                logger.info(f"[Agent] Tool 调用: {tool_name}({tool_args})")
                result = self._execute_tool(tool_name, tool_args)
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
            "max_tokens": 4096,
        }

        for attempt in range(3):
            try:
                resp = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=120)
                if resp.status_code == 200:
                    return resp.json()
                logger.warning(f"LLM 调用失败: status={resp.status_code}, attempt={attempt + 1}")
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

        if name == "search_hot_projects":
            result = tool_search_hot_projects(
                state.token_mgr,
                categories=args.get("categories"),
                min_stars=args.get("min_stars", 1000),
                max_pages=args.get("max_pages", 3),
            )
            # 缓存搜索结果
            raw_repos = result.pop("_raw_repos", [])
            state.last_search_repos = raw_repos
            state.seen_repos.update(r["full_name"] for r in raw_repos)
            return result

        elif name == "scan_star_range":
            result = tool_scan_star_range(
                state.token_mgr,
                min_star=args.get("min_star", 2000),
                max_star=args.get("max_star", 40000),
                seen_repos=state.seen_repos,
            )
            raw_repos = result.pop("_raw_repos", [])
            state.last_search_repos.extend(raw_repos)
            return result

        elif name == "check_repo_growth":
            return tool_check_repo_growth(
                state.token_mgr,
                repo=args["repo"],
                db=state.db,
            )

        elif name == "batch_check_growth":
            if not state.last_search_repos:
                return {"error": "没有搜索结果，请先调用 search_hot_projects"}
            result = tool_batch_check_growth(
                state.token_mgr,
                repos=state.last_search_repos,
                db=state.db,
                growth_threshold=args.get("growth_threshold", 800),
            )
            state.last_candidates = result.get("candidates", {})
            # 中间落盘
            save_db(state.db)
            return result

        elif name == "rank_candidates":
            if not state.last_candidates:
                return {"error": "没有候选列表，请先调用 batch_check_growth"}
            result = tool_rank_candidates(
                state.last_candidates,
                top_n=args.get("top_n", HOT_PROJECT_COUNT),
                mode=args.get("mode", "comprehensive"),
                token_mgr=state.token_mgr,
                db=state.db,
            )
            state.last_ranked = result.pop("_ordered_tuples", [])
            return result

        elif name == "describe_project":
            return tool_describe_project(
                repo=args["repo"],
                db=state.db,
            )

        elif name == "generate_report":
            if not state.last_ranked:
                return {"error": "没有排序结果，请先调用 rank_candidates"}
            result = tool_generate_report(state.last_ranked, state.db)
            save_db(state.db)
            return result

        elif name == "get_db_info":
            return tool_get_db_info(
                db=state.db,
                repo=args.get("repo"),
            )

        elif name == "full_discovery":
            result = tool_full_discovery(state.token_mgr)
            if result.get("status") == "completed":
                state.db = load_db()
                state.last_search_repos = []
                state.last_candidates = {}
                state.last_ranked = []
                state.seen_repos = set()
            return result

        elif name == "fetch_trending":
            result = tool_fetch_trending(
                since=args.get("since", "daily"),
                language=args.get("language", ""),
                spoken_language=args.get("spoken_language", ""),
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

        # 分离 system prompt
        system_msgs = [m for m in conv if m.get("role") == "system"]
        non_system = [m for m in conv if m.get("role") != "system"]

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
                f"[当前状态] 候选仓库: {len(self.state.last_candidates)} 个, "
                f"已排序: {len(self.state.last_ranked)} 个, "
                f"已扫描: {len(self.state.seen_repos)} 个"
            ),
        }

        self.state.conversation = system_msgs + [summary_msg] + recent_msgs
        logger.info(
            f"[Agent] 对话历史已压缩: {len(old_msgs)} 条旧消息 → 摘要, "
            f"保留 {len(recent_msgs)} 条近期消息"
        )
