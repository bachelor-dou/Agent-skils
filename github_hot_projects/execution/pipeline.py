"""
统一发现管道
============
封装 search → scan → trending → growth → rank → report 六步流程，
供定时任务（scheduled_update）和 Agent 复用。

数据源说明：
  综合热榜与新项目榜单均为 **三源数据合一** ：
    1. search_hot_projects — 关键词类别搜索
    2. scan_star_range     — star 范围扫描
    3. fetch_trending      — GitHub Trending 三档补源
  区别仅在筛选规则：
    - comprehensive : log-score 综合评分排序
    - hot_new       : created_at 过滤后按增长量排序
"""

import logging

from ..common.config import (
    HOT_PROJECT_COUNT,
    HOT_NEW_PROJECT_COUNT,
    STAR_GROWTH_THRESHOLD,
    TIME_WINDOW_DAYS,
)
from ..common.db import save_db
from ..common.token_manager import TokenManager
from ..agent_tools import (
    tool_search_hot_projects,
    tool_scan_star_range,
    tool_batch_check_growth,
    tool_rank_candidates,
    tool_generate_report,
    tool_fetch_trending,
)

logger = logging.getLogger("discover_hot")


class DiscoveryPipeline:
    """端到端项目发现管道。

    使用示例::

        pipeline = DiscoveryPipeline(token_mgr, db)
        result = pipeline.run(mode="comprehensive", top_n=100)
        print(result["report_path"])
    """

    def __init__(self, token_mgr: TokenManager, db: dict) -> None:
        self.token_mgr = token_mgr
        self.db = db

    def run(
        self,
        mode: str = "comprehensive",
        top_n: int | None = None,
        new_project_days: int | None = None,
        time_window_days: int = TIME_WINDOW_DAYS,
        growth_threshold: int = STAR_GROWTH_THRESHOLD,
    ) -> dict:
        """执行完整的 search → scan → trending → growth → rank → report 流程。

        Args:
            mode:              排名模式 ("comprehensive" | "hot_new")
            top_n:             取前 N 个项目，None 则按模式使用默认值
            new_project_days:  新项目创建窗口（天），仅 hot_new 模式有效
            time_window_days:  增长统计窗口（天）
            growth_threshold:  增长阈值

        Returns:
            {"report_path": str, "ranked_count": int, "candidates_count": int, ...}
        """
        if top_n is None:
            top_n = HOT_NEW_PROJECT_COUNT if mode == "hot_new" else HOT_PROJECT_COUNT

        logger.info(
            "[Pipeline] 启动: mode=%s, top_n=%s, new_project_days=%s, "
            "time_window_days=%s, growth_threshold=%s, 数据源=search+scan+trending 三源合一",
            mode, top_n, new_project_days, time_window_days, growth_threshold,
        )

        # ── Step 1: 三源搜索采集 ──
        all_repos, seen = self._collect_repos(new_project_days)
        if not all_repos:
            logger.error("[Pipeline] 搜索阶段未获取到任何仓库，终止。")
            return {"error": "搜索阶段无结果", "report_path": ""}

        # ── Step 2: 批量增长计算 ──
        logger.info("[Pipeline] Step 2: 批量增长计算 (%d 个仓库)", len(all_repos))
        growth_result = tool_batch_check_growth(
            self.token_mgr, all_repos, self.db,
            growth_threshold=growth_threshold,
            new_project_days=new_project_days,
            time_window_days=time_window_days,
        )
        candidates = growth_result.get("candidates", {})
        logger.info(
            "[Pipeline] 候选: %d / %d (阈值 >=%d)",
            len(candidates), growth_result.get("total_checked", 0), growth_threshold,
        )
        save_db(self.db)

        if not candidates:
            logger.warning("[Pipeline] 无候选项目，终止。")
            return {"error": "无候选项目", "report_path": "", "total_repos": len(all_repos)}

        # ── Step 3: 排序 ──
        logger.info("[Pipeline] Step 3: 排名 (mode=%s, top_n=%d)", mode, top_n)
        rank_result = tool_rank_candidates(
            candidates, top_n=top_n, mode=mode, db=self.db,
            new_project_days=new_project_days,
            time_window_days=time_window_days,
        )
        top_projects = rank_result.pop("_ordered_tuples", [])
        logger.info("[Pipeline] 排名完成: %d 个项目", len(top_projects))

        if not top_projects:
            logger.error("[Pipeline] 排名结果为空，终止。")
            return {"error": "排名结果为空", "report_path": "", "candidates_count": len(candidates)}

        # ── Step 4: 生成报告 ──
        logger.info("[Pipeline] Step 4: 生成报告")
        report_result = tool_generate_report(
            top_projects, self.db, mode=mode,
            new_project_days=new_project_days if mode == "hot_new" else None,
            time_window_days=time_window_days,
        )
        save_db(self.db)

        report_path = report_result.get("report_path", "")
        if report_path:
            logger.info("[Pipeline] 完成! 报告: %s", report_path)

        return {
            "report_path": report_path,
            "ranked_count": len(top_projects),
            "candidates_count": len(candidates),
            "total_repos": len(all_repos),
            "mode": mode,
        }

    def _collect_repos(self, new_project_days: int | None = None) -> tuple[list[dict], set[str]]:
        """三源搜索采集：search + scan + trending。

        综合榜和新项目榜使用相同的三源数据，
        new_project_days 控制是否在搜索阶段前置过滤创建时间。
        """
        seen: set[str] = set()
        all_repos: list[dict] = []

        # 1a. 关键词搜索
        logger.info("[Pipeline] Step 1a: 关键词搜索（全类别）")
        search_result = tool_search_hot_projects(
            self.token_mgr, new_project_days=new_project_days,
        )
        raw_repos = search_result.pop("_raw_repos", [])
        all_repos.extend(raw_repos)
        seen.update(r["full_name"] for r in raw_repos)
        logger.info("[Pipeline]   关键词搜索: %d 个仓库", len(raw_repos))

        # 1b. Star 范围扫描
        logger.info("[Pipeline] Step 1b: Star 范围扫描")
        scan_result = tool_scan_star_range(
            self.token_mgr, seen_repos=seen, new_project_days=new_project_days,
        )
        scan_repos = scan_result.pop("_raw_repos", [])
        for r in scan_repos:
            if r["full_name"] not in seen:
                seen.add(r["full_name"])
                all_repos.append(r)
        logger.info("[Pipeline]   范围扫描补充: %d 个, 累计 %d 个", len(scan_repos), len(all_repos))

        # 1c. Trending 三档补源
        logger.info("[Pipeline] Step 1c: Trending 补源（daily+weekly+monthly）")
        trending_result = tool_fetch_trending(include_all_periods=True)
        trending_repos = trending_result.pop("_raw_repos", [])
        added = 0
        for r in trending_repos:
            fn = r["full_name"]
            if fn not in seen:
                seen.add(fn)
                all_repos.append({
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
                added += 1
        logger.info("[Pipeline]   Trending 补充: %d 个, 最终 %d 个", added, len(all_repos))

        return all_repos, seen
