#!/usr/bin/env python
"""
定时更新脚本：后台自动执行综合热度搜索、增长计算、排名并生成报告。

无需 LLM 对话，直接执行内置 DiscoveryPipeline 编排流程。

用法：
    python scheduled_update.py [--top-n 100] [--growth-calc-days 7]
"""
# ============================================================
# 部署为定时任务（cron）
# ============================================================
#
# 1. 手动运行测试：
#    cd /root/code/Agent-skils/github_hot_projects
#    python scheduled_update.py --top-n 100
#
# 2. 编辑 crontab（每周五 23:00 自动执行）：
#    crontab -e
#    添加以下行：
#    0 23 * * 5 source ~/.bashrc && cd /root/code/Agent-skils/github_hot_projects && /usr/bin/python3 scheduled_update.py --top-n 100 --growth-calc-days 7
#
# 3. 或使用 systemd timer：
#    sudo cp scheduled_update.service /etc/systemd/system/
#    sudo cp scheduled_update.timer   /etc/systemd/system/
#    sudo systemctl enable --now scheduled_update.timer
#
# 日志：logs/scheduled-YYYY-MM-DD.log
# ============================================================
import argparse
import logging
import logging.handlers
import os
import sys
from datetime import datetime

# 确保可以 import 项目包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from github_hot_projects.common.config import (
    HOT_PROJECT_COUNT,
    LOG_DIR,
    STAR_GROWTH_THRESHOLD,
    GROWTH_CALC_DAYS,
)
from github_hot_projects.common.db import load_db, save_db
from github_hot_projects.common.token_manager import TokenManager
from github_hot_projects.agent_tools import (
    tool_batch_check_growth,
    tool_fetch_trending,
    tool_generate_report,
    tool_rank_candidates,
    tool_scan_star_range,
    tool_search_by_keywords,
    trending_repo_to_search_repo,
)


def setup_logging() -> str:
    """配置日志：同时输出到终端和文件，文件使用 RotatingFileHandler 防止过大。"""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(
        LOG_DIR,
        f"scheduled-{datetime.now().strftime('%Y-%m-%d')}.log",
    )
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=50 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            file_handler,
            logging.StreamHandler(),
        ],
        force=True,
    )
    return log_path


logger = logging.getLogger("scheduled_update")


class DiscoveryPipeline:
    """端到端项目发现管道。

    用于定时任务场景：search -> scan -> trending -> growth -> rank -> report。
    """

    def __init__(self, token_mgr: TokenManager, db: dict) -> None:
        self.token_mgr = token_mgr
        self.db = db

    def run(
        self,
        top_n: int | None = None,
        days_since_created: int | None = None,
        growth_calc_days: int = GROWTH_CALC_DAYS,
        growth_threshold: int = STAR_GROWTH_THRESHOLD,
        force_refresh: bool = False,
    ) -> dict:
        if top_n is None:
            top_n = HOT_PROJECT_COUNT

        logger.info(
            "[Pipeline] 启动: mode=%s, top_n=%s, days_since_created=%s, "
            "growth_calc_days=%s, growth_threshold=%s, 数据源=search+scan+trending 三源合一",
            "comprehensive", top_n, days_since_created, growth_calc_days, growth_threshold,
        )

        all_repos, seen = self._collect_repos(days_since_created)
        if not all_repos:
            logger.error("[Pipeline] 搜索阶段未获取到任何仓库，终止。")
            return {"error": "搜索阶段无结果", "report_path": ""}

        logger.info("[Pipeline] Step 2: 批量增长计算 (%d 个仓库)", len(all_repos))
        growth_result = tool_batch_check_growth(
            self.token_mgr,
            all_repos,
            self.db,
            growth_threshold=growth_threshold,
            days_since_created=days_since_created,
            growth_calc_days=growth_calc_days,
            force_refresh=force_refresh,
        )
        candidates = growth_result.get("candidates", {})
        logger.info(
            "[Pipeline] 候选: %d / %d (阈值 >=%d)",
            len(candidates),
            growth_result.get("total_checked", 0),
            growth_threshold,
        )
        save_db(self.db)

        if not candidates:
            logger.warning("[Pipeline] 无候选项目，终止。")
            return {"error": "无候选项目", "report_path": "", "total_repos": len(all_repos)}

        logger.info("[Pipeline] Step 3: 排名 (mode=%s, top_n=%d)", "comprehensive", top_n)
        rank_result = tool_rank_candidates(
            candidates,
            top_n=top_n,
            mode="comprehensive",
            db=self.db,
            days_since_created=days_since_created,
        )
        top_projects = rank_result.pop("_ordered_tuples", [])
        logger.info("[Pipeline] 排名完成: %d 个项目", len(top_projects))

        if not top_projects:
            logger.error("[Pipeline] 排名结果为空，终止。")
            return {"error": "排名结果为空", "report_path": "", "candidates_count": len(candidates)}

        logger.info("[Pipeline] Step 4: 生成报告")
        report_result = tool_generate_report(
            top_projects,
            self.db,
            mode="comprehensive",
            days_since_created=None,
            growth_calc_days=growth_calc_days,
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
            "mode": "comprehensive",
        }

    def _collect_repos(self, days_since_created: int | None = None) -> tuple[list[dict], set[str]]:
        seen: set[str] = set()
        all_repos: list[dict] = []

        logger.info("[Pipeline] Step 1a: 关键词搜索（全类别）")
        search_result = tool_search_by_keywords(
            self.token_mgr,
            days_since_created=days_since_created,
        )
        raw_repos = search_result.pop("_raw_repos", [])
        all_repos.extend(raw_repos)
        seen.update(r["full_name"] for r in raw_repos)
        logger.info("[Pipeline]   关键词搜索: %d 个仓库", len(raw_repos))

        logger.info("[Pipeline] Step 1b: Star 范围扫描")
        scan_result = tool_scan_star_range(
            self.token_mgr,
            seen_repos=seen,
            days_since_created=days_since_created,
        )
        scan_repos = scan_result.pop("_raw_repos", [])
        all_repos.extend(scan_repos)
        logger.info("[Pipeline]   范围扫描补充: %d 个, 累计 %d 个", len(scan_repos), len(all_repos))

        logger.info("[Pipeline] Step 1c: Trending 补源（daily+weekly+monthly）")
        trending_result = tool_fetch_trending(trending_range="all")
        trending_repos = trending_result.pop("_raw_repos", [])
        added = 0
        for repo in trending_repos:
            full_name = repo["full_name"]
            if full_name in seen:
                continue
            seen.add(full_name)
            all_repos.append(trending_repo_to_search_repo(repo))
            added += 1
        logger.info("[Pipeline]   Trending 补充: %d 个, 最终 %d 个", added, len(all_repos))

        return all_repos, seen


def log_update_summary(old_db: dict, new_db: dict) -> None:
    """输出本次更新的变化统计表格，确保准确且详细。"""
    old_projects = old_db.get("projects", {})
    new_projects = new_db.get("projects", {})

    old_count = len(old_projects)
    new_count = len(new_projects)

    # 区分新增项目 vs 已有项目
    new_added = set(new_projects.keys()) - set(old_projects.keys())
    removed = set(old_projects.keys()) - set(new_projects.keys())
    existing = set(old_projects.keys()) & set(new_projects.keys())

    # 统计：包含已有项目变化 + 新增项目的字段填充
    stats = {
        "refreshed_at": {"changed": 0, "new_filled": 0},
        "star": {"changed": 0, "increased": 0, "decreased": 0, "total_growth": 0},
        "forks": {"changed": 0, "increased": 0, "decreased": 0, "total_growth": 0},
        "short_desc": {
            "changed": 0,
            "empty_to_filled": 0,
            "filled_to_empty": 0,
            "content_changed": 0,
            "new_filled": 0,  # 新增项目填充
        },
        "desc": {
            "changed": 0,
            "empty_to_filled": 0,
            "filled_to_empty": 0,
            "content_changed": 0,
            "new_filled": 0,  # 新增项目填充
        },
        "topics": {"changed": 0, "empty_to_filled": 0, "new_filled": 0},
        "language": {"changed": 0, "empty_to_filled": 0, "new_filled": 0},
    }

    # 统计新增项目的字段填充
    for name in new_added:
        new_p = new_projects[name]
        stats["refreshed_at"]["new_filled"] += 1
        if new_p.get("short_desc"):
            stats["short_desc"]["new_filled"] += 1
        if new_p.get("desc"):
            stats["desc"]["new_filled"] += 1
        if new_p.get("topics"):
            stats["topics"]["new_filled"] += 1
        if new_p.get("language"):
            stats["language"]["new_filled"] += 1

    # 遍历已有项目，统计变化
    for name in existing:
        old_p = old_projects[name]
        new_p = new_projects[name]

        # refreshed_at
        if old_p.get("refreshed_at") != new_p.get("refreshed_at"):
            stats["refreshed_at"]["changed"] += 1

        # star - 详细统计增减
        old_star = old_p.get("star", 0)
        new_star = new_p.get("star", 0)
        if old_star != new_star:
            stats["star"]["changed"] += 1
            diff = new_star - old_star
            if diff > 0:
                stats["star"]["increased"] += 1
                stats["star"]["total_growth"] += diff
            else:
                stats["star"]["decreased"] += 1

        # forks - 详细统计增减
        old_forks = old_p.get("forks", 0)
        new_forks = new_p.get("forks", 0)
        if old_forks != new_forks:
            stats["forks"]["changed"] += 1
            diff = new_forks - old_forks
            if diff > 0:
                stats["forks"]["increased"] += 1
                stats["forks"]["total_growth"] += diff
            else:
                stats["forks"]["decreased"] += 1

        # short_desc - 区分变化类型
        old_sd = old_p.get("short_desc", "")
        new_sd = new_p.get("short_desc", "")
        if old_sd != new_sd:
            stats["short_desc"]["changed"] += 1
            if not old_sd and new_sd:
                stats["short_desc"]["empty_to_filled"] += 1
            elif old_sd and not new_sd:
                stats["short_desc"]["filled_to_empty"] += 1
            else:
                stats["short_desc"]["content_changed"] += 1

        # desc - 区分变化类型
        old_d = old_p.get("desc", "")
        new_d = new_p.get("desc", "")
        if old_d != new_d:
            stats["desc"]["changed"] += 1
            if not old_d and new_d:
                stats["desc"]["empty_to_filled"] += 1
            elif old_d and not new_d:
                stats["desc"]["filled_to_empty"] += 1
            else:
                stats["desc"]["content_changed"] += 1

        # topics
        old_topics = old_p.get("topics", [])
        new_topics = new_p.get("topics", [])
        if old_topics != new_topics:
            stats["topics"]["changed"] += 1
            if not old_topics and new_topics:
                stats["topics"]["empty_to_filled"] += 1

        # language
        old_lang = old_p.get("language", "")
        new_lang = new_p.get("language", "")
        if old_lang != new_lang:
            stats["language"]["changed"] += 1
            if not old_lang and new_lang:
                stats["language"]["empty_to_filled"] += 1

    # 输出详细统计
    logger.info("")
    logger.info("=" * 70)
    logger.info("【本次更新统计】")
    logger.info("=" * 70)

    # 项目数量变化
    logger.info(f"项目总数: {old_count} → {new_count}")
    if new_added:
        logger.info(f"  新增项目: {len(new_added)} 个")
    if removed:
        logger.info(f"  移除项目: {len(removed)} 个")
    logger.info(f"  已有项目: {len(existing)} 个")

    logger.info("-" * 70)
    logger.info("字段变化详情:")
    logger.info("-" * 70)

    # refreshed_at: 已有项目刷新 + 新增项目填充
    r = stats["refreshed_at"]
    total_r = r["changed"] + r["new_filled"]
    logger.info(f"  refreshed_at: {total_r} 个更新 (已有刷新 {r['changed']}, 新增 {r['new_filled']})")

    # star
    s = stats["star"]
    if s["changed"]:
        avg_growth = s["total_growth"] / s["increased"] if s["increased"] > 0 else 0
        logger.info(
            f"  star: {s['changed']} 个变化 "
            f"(↑{s['increased']} 个, ↓{s['decreased']} 个, "
            f"总增长 +{s['total_growth']}, 平均增长 +{avg_growth:.1f})"
        )
    else:
        logger.info("  star: 无变化")

    # forks
    f = stats["forks"]
    if f["changed"]:
        avg_growth = f["total_growth"] / f["increased"] if f["increased"] > 0 else 0
        logger.info(
            f"  forks: {f['changed']} 个变化 "
            f"(↑{f['increased']} 个, ↓{f['decreased']} 个, "
            f"总增长 +{f['total_growth']}, 平均增长 +{avg_growth:.1f})"
        )
    else:
        logger.info("  forks: 无变化")

    # short_desc: 已有项目变化 + 新增项目填充
    sd = stats["short_desc"]
    total_sd = sd["changed"] + sd["new_filled"]
    if total_sd:
        parts = [f"共 {total_sd} 个"]
        if sd["empty_to_filled"]:
            parts.append(f"已有空→有内容 {sd['empty_to_filled']}")
        if sd["new_filled"]:
            parts.append(f"新增项目填充 {sd['new_filled']}")
        if sd["filled_to_empty"]:
            parts.append(f"有内容→空 {sd['filled_to_empty']}")
        if sd["content_changed"]:
            parts.append(f"内容变更 {sd['content_changed']}")
        logger.info(f"  short_desc: {', '.join(parts)}")
    else:
        logger.info("  short_desc: 无变化")

    # desc: 已有项目变化 + 新增项目填充
    d = stats["desc"]
    total_d = d["changed"] + d["new_filled"]
    if total_d:
        parts = [f"共 {total_d} 个"]
        if d["empty_to_filled"]:
            parts.append(f"已有空→有内容 {d['empty_to_filled']}")
        if d["new_filled"]:
            parts.append(f"新增项目填充 {d['new_filled']}")
        if d["filled_to_empty"]:
            parts.append(f"有内容→空 {d['filled_to_empty']}")
        if d["content_changed"]:
            parts.append(f"内容变更 {d['content_changed']}")
        logger.info(f"  desc (LLM): {', '.join(parts)}")
    else:
        logger.info("  desc (LLM): 无变化")

    # topics
    t = stats["topics"]
    total_t = t["changed"] + t["new_filled"]
    if total_t:
        logger.info(f"  topics: {total_t} 个变化 (已有空→有内容 {t['empty_to_filled']}, 新增 {t['new_filled']})")
    else:
        logger.info("  topics: 无变化")

    # language
    l = stats["language"]
    total_l = l["changed"] + l["new_filled"]
    if total_l:
        logger.info(f"  language: {total_l} 个变化 (已有空→有内容 {l['empty_to_filled']}, 新增 {l['new_filled']})")
    else:
        logger.info("  language: 无变化")

    logger.info("=" * 70)
    logger.info("")


def run_update(
    top_n: int,
    growth_calc_days: int = GROWTH_CALC_DAYS,
) -> None:
    """执行完整的搜索→增长→排名→报告流程（委托给 DiscoveryPipeline）。"""
    token_mgr = TokenManager()
    db = load_db()

    # 保存旧 DB 快照用于对比
    import copy
    old_db = copy.deepcopy(db)

    logger.info(
        f"开始定时更新: mode=comprehensive, top_n={top_n}, growth_calc_days={growth_calc_days}, "
        f"DB projects={len(db.get('projects', {}))}, valid={db.get('valid')}"
    )

    pipeline = DiscoveryPipeline(token_mgr, db)
    result = pipeline.run(
        top_n=top_n,
        growth_calc_days=growth_calc_days,
        force_refresh=True,
    )

    report_path = result.get("report_path", "")
    if report_path:
        logger.info(f"定时更新完成! 报告: {report_path}")
        # 输出更新统计
        log_update_summary(old_db, db)
    elif result.get("error"):
        logger.error(f"定时更新失败: {result['error']}")
    else:
        logger.error("报告生成失败。")


def main():
    parser = argparse.ArgumentParser(
        description="定时更新：自动搜索、计算增长、排名并生成 GitHub 热门项目报告"
    )
    parser.add_argument(
        "--top-n", type=int, default=HOT_PROJECT_COUNT,
        help=f"取前 N 个项目 (默认: {HOT_PROJECT_COUNT})",
    )
    parser.add_argument(
        "--growth-calc-days", type=int, default=GROWTH_CALC_DAYS,
        help=f"增长计算窗口天数 (默认: {GROWTH_CALC_DAYS})",
    )
    args = parser.parse_args()

    log_path = setup_logging()
    logger.info(f"日志: {log_path}")

    try:
        run_update(
            top_n=args.top_n,
            growth_calc_days=args.growth_calc_days,
        )
    except Exception:
        logger.exception("定时更新异常终止")
        sys.exit(1)


if __name__ == "__main__":
    main()


