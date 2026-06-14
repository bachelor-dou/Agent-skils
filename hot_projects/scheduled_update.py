#!/usr/bin/env python
"""
定时更新脚本：后台自动执行综合热度搜索、增长计算、排名并生成报告。

无需 LLM 对话，直接执行内置 DiscoveryPipeline 编排流程。

用法：
    python scheduled_update.py --top-n 100 --growth-calc-days 7
"""
# ============================================================
# 部署为定时任务（cron）
# ============================================================
#
# 1. 手动运行测试：
#    cd /root/code/Agent-skils/github_hot_projects
#    python scheduled_update.py --top-n 100
#
# 2. 编辑 crontab（每周日 00:36 自动执行）：
#    crontab -e
#    添加以下行：
#    36 0 * * 7 source ~/.bashrc && cd /root/code/Agent-skils/github_hot_projects && /usr/bin/python3 scheduled_update.py --top-n 100 --growth-calc-days 7
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

from hot_projects.config import (
    HOT_PROJECT_COUNT,
    LOG_DIR,
    STAR_GROWTH_THRESHOLD,
    GROWTH_CALC_DAYS,
    MIN_STAR,
    MAX_STAR,
)
from hot_projects.infra.db import load_db, save_db
from hot_projects.providers.github.token_pool import GitHubTokenPool
from hot_projects.providers.github.provider import GitHubProvider
from hot_projects.pipeline.ranking_pipeline import run_ranking
from hot_projects.pipeline.cache import RankingCache


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
    # httpx/httpcore 会在 INFO 级别输出每条 HTTP 请求，定时日志里只保留业务日志。
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    return log_path


logger = logging.getLogger("scheduled_update")


class DiscoveryPipeline:
    """端到端项目发现管道（委托统一 ranking_pipeline）。

    定时任务场景：search -> scan -> trending -> growth -> rank -> report，
    与 Agent 复合榜单工具共用同一 run_ranking 实现。
    """

    def __init__(self, token_mgr: GitHubTokenPool, db: dict) -> None:
        self.provider = GitHubProvider(token_mgr)
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

        mode = "hot_new" if days_since_created is not None else "comprehensive"
        params = {
            "min_star": MIN_STAR,
            "max_star": MAX_STAR,
            "top_n": top_n,
            "growth_calc_days": growth_calc_days,
            "growth_threshold": growth_threshold,
            "days_since_created": days_since_created,
        }
        logger.info(
            "[Pipeline] 启动: mode=%s, top_n=%s, days_since_created=%s, "
            "growth_calc_days=%s, growth_threshold=%s, 数据源=search+scan+trending 三源合一",
            mode, top_n, days_since_created, growth_calc_days, growth_threshold,
        )

        result = run_ranking(
            self.provider, mode=mode, params=params, db=self.db,
            cache=RankingCache(), do_report=True, force_refresh=force_refresh,
        )
        save_db(self.db)

        report_path = result.get("report_path", "")
        ranked = result.get("ranked", [])
        if not ranked:
            logger.warning("[Pipeline] 无榜单结果。")
            return {"error": "无候选项目", "report_path": report_path,
                    "candidates_count": result.get("candidates_count", 0)}

        if report_path:
            logger.info("[Pipeline] 完成! 报告: %s", report_path)
        return {
            "report_path": report_path,
            "ranked_count": len(ranked),
            "candidates_count": result.get("candidates_count", 0),
            "mode": result.get("mode", mode),
        }


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
    # 定时窗口规则：取 max(指定值, 默认窗口)——指定值大于默认才覆盖，否则用默认。
    growth_calc_days = max(growth_calc_days, GROWTH_CALC_DAYS)
    token_mgr = GitHubTokenPool()
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


