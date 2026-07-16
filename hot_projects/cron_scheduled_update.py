#!/usr/bin/env python
"""
定时更新脚本：后台自动执行综合热度搜索、增长计算、排名并生成报告。

无需 LLM 对话，直接执行内置 DiscoveryPipeline 编排流程。

用法：
    python -m hot_projects.cron_scheduled_update --top-n 100 --growth-calc-days 7
"""
# ============================================================
# 部署为定时任务（cron）
# ============================================================
#
# 1. 手动运行测试：
#    cd /root/code/Agent-skils
#    python -m hot_projects.cron_scheduled_update --top-n 100
#
# 2. 编辑 crontab（每周日 00:36 自动执行）：
#    crontab -e
#    添加以下行：
#    36 0 * * 7 . /root/.hot_projects.env && cd /root/code/Agent-skils && /usr/bin/python3 -m hot_projects.cron_scheduled_update --top-n 100 --growth-calc-days 7
#
# 主日志：logs/YYYY-MM/cron-YYYY-MM-DD.log
# 调试日志：logs/YYYY-MM/debug/cron-YYYY-MM-DD.debug.log
# ============================================================
import argparse
import logging
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
from hot_projects.datasource.github.token_pool import GitHubTokenPool
from hot_projects.datasource.github.provider import GitHubProvider
from hot_projects.tools.tool.ranking import run_ranking, RankingCache


class _DebugOnlyFilter(logging.Filter):
    """只放行 DEBUG 级记录。

    INFO 及以上已写入主日志，debug 副日志只保留额外的 DEBUG 细节，避免重复。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == logging.DEBUG


def setup_logging() -> str:
    """配置定时任务日志：主日志（INFO，仅文件）+ debug 副日志（仅 DEBUG 细节，不重复 INFO）。

    按月归档：logs/YYYY-MM/cron-YYYY-MM-DD.log，调试日志在 logs/YYYY-MM/debug/ 下。
    """
    now = datetime.now()
    month_dir = os.path.join(LOG_DIR, now.strftime("%Y-%m"))
    os.makedirs(month_dir, exist_ok=True)
    debug_log_dir = os.path.join(month_dir, "debug")
    os.makedirs(debug_log_dir, exist_ok=True)

    main_level = logging.INFO
    log_date = now.strftime("%Y-%m-%d")
    log_path = os.path.join(month_dir, f"cron-{log_date}.log")
    file_handler = logging.FileHandler(
        log_path,
        encoding="utf-8",
    )
    file_handler.setLevel(main_level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )

    handlers: list[logging.Handler] = [
        file_handler,
    ]

    debug_path = os.path.join(debug_log_dir, f"cron-{log_date}.debug.log")
    debug_handler = logging.FileHandler(
        debug_path,
        encoding="utf-8",
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.addFilter(_DebugOnlyFilter())  # 只写 DEBUG 细节，INFO 已在主日志，避免重复
    debug_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handlers.append(debug_handler)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    # httpx/httpcore 会在 INFO 级别输出每条 HTTP 请求，定时日志里只保留业务日志。
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    return log_path


logger = logging.getLogger("scheduled_update")


def _run_comprehensive(token_mgr: GitHubTokenPool, db: dict, top_n: int,
                       growth_calc_days: int, force_refresh: bool) -> dict:
    """跑综合榜(search+scan+trending→增长→排序→报告),与 Agent 复合工具共用 run_ranking。"""
    params = {
        "min_star": MIN_STAR,
        "max_star": MAX_STAR,
        "top_n": top_n,
        "growth_calc_days": growth_calc_days,
        "growth_threshold": STAR_GROWTH_THRESHOLD,
        "days_since_created": None,
    }
    logger.info(
        "[Pipeline] 启动: mode=comprehensive, top_n=%s, growth_calc_days=%s, "
        "growth_threshold=%s, 数据源=search+scan+trending 三源合一",
        top_n, growth_calc_days, STAR_GROWTH_THRESHOLD,
    )

    result = run_ranking(
        GitHubProvider(token_mgr), mode="comprehensive", params=params, db=db,
        cache=RankingCache(), do_report=True, force_refresh=force_refresh,
    )

    report_path = result.get("report_path", "")
    if not result.get("ranked"):
        # 空跑（搜索/增长全失败等）不落库：save_db 会强制 date=今天，
        # 会把一次失败伪造成“新鲜基线”，误导下一轮 DB 差值判定。
        logger.warning("[Pipeline] 无榜单结果，跳过 DB 保存（避免伪造新鲜基线）。")
        return {"error": "无候选项目", "report_path": report_path,
                "candidates_count": result.get("candidates_count", 0)}

    save_db(db)
    if report_path:
        logger.info("[Pipeline] 完成! 报告: %s", report_path)
    return {
        "report_path": report_path,
        "ranked_count": len(result["ranked"]),
        "candidates_count": result.get("candidates_count", 0),
        "mode": result.get("mode", "comprehensive"),
        "funnel": result.get("funnel"),
        "ranked": result["ranked"],  # [(full_name, {growth, star}), ...] 供推送取 Top3
    }


def log_pipeline_funnel(funnel: dict | None) -> None:
    """输出本轮榜单漏斗：从收集到出榜的逐层数量，集中一处便于核对。"""
    if not funnel:
        return
    logger.info("")
    logger.info("=" * 70)
    logger.info("【本轮榜单漏斗】")
    logger.info("=" * 70)
    logger.info(f"收集并计算增长: {funnel.get('collected', 0)} 个仓库")
    logger.info(f"  · DB 差值(秒算): {funnel.get('db_diff', 0)}")
    logger.info(f"  · 实时 API 计算: {funnel.get('realtime', 0)}")
    logger.info(f"增长候选池(增长≥0): {funnel.get('growth_pool', 0)}")
    logger.info(f"达标(增长≥阈值) → 进入排名: {funnel.get('qualified', 0)}")
    logger.info(
        f"最近爆发探针: {funnel.get('recent_probe', 0)} 个"
        f"（爆发加成生效 {funnel.get('boost_applied', 0)}）"
    )
    logger.info(f"榜单输出: Top {funnel.get('ranked', 0)}")
    logger.info("=" * 70)


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
        "star": {"changed": 0, "increased": 0, "decreased": 0},
        "forks": {"changed": 0, "increased": 0, "decreased": 0},
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
    }

    # 统计新增项目的字段填充
    for name in new_added:
        new_p = new_projects[name]
        stats["refreshed_at"]["new_filled"] += 1
        if new_p.get("short_desc"):
            stats["short_desc"]["new_filled"] += 1
        if new_p.get("desc"):
            stats["desc"]["new_filled"] += 1

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
            if new_star - old_star > 0:
                stats["star"]["increased"] += 1
            else:
                stats["star"]["decreased"] += 1

        # forks - 详细统计增减
        old_forks = old_p.get("forks", 0)
        new_forks = new_p.get("forks", 0)
        if old_forks != new_forks:
            stats["forks"]["changed"] += 1
            if new_forks - old_forks > 0:
                stats["forks"]["increased"] += 1
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
        logger.info(f"  star: {s['changed']} 个变化 (↑{s['increased']} 个, ↓{s['decreased']} 个)")
    else:
        logger.info("  star: 无变化")

    # forks
    f = stats["forks"]
    if f["changed"]:
        logger.info(f"  forks: {f['changed']} 个变化 (↑{f['increased']} 个, ↓{f['decreased']} 个)")
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

    result = _run_comprehensive(
        token_mgr, db, top_n=top_n,
        growth_calc_days=growth_calc_days, force_refresh=True,
    )

    report_path = result.get("report_path", "")
    if report_path:
        logger.info(f"定时更新完成! 报告: {report_path}")
        # 先打印榜单漏斗（收集→出榜逐层数量），再打印 DB 字段变化统计
        log_pipeline_funnel(result.get("funnel"))
        log_update_summary(old_db, db)
        _push_weekly_digest(result)
    elif result.get("error"):
        logger.error(f"定时更新失败: {result['error']}")
    else:
        logger.error("报告生成失败。")


def _prev_report_added_removed(report_path: str, current_names: set[str]) -> tuple[int, int, str] | None:
    """对比上一份同名式报告，返回 (上新数, 移出数, 上期文件名)；无可比则 None。"""
    import glob
    import re
    from hot_projects.config import REPORT_DIR
    from hot_projects.tools.basic.report_parse import parse_structured_report

    cur = os.path.basename(report_path)
    m = re.match(r"^(\d{4}-\d{2}-\d{2})(.*)\.md$", cur)
    if not m:
        return None
    cur_date, suffix = m.group(1), m.group(2)
    prev = None
    for path in sorted(glob.glob(os.path.join(REPORT_DIR, "*.md")), reverse=True):
        pm = re.match(r"^(\d{4}-\d{2}-\d{2})(.*)\.md$", os.path.basename(path))
        if pm and pm.group(2) == suffix and pm.group(1) < cur_date:
            prev = path
            break
    if prev is None:
        return None
    try:
        parsed = parse_structured_report(open(prev, encoding="utf-8").read())
    except OSError:
        return None
    if not parsed:
        return None
    prev_names = {r["repo"] for r in parsed["repos"]}
    added = len(current_names - prev_names)
    removed = len(prev_names - current_names)
    return added, removed, os.path.basename(prev)


def _push_weekly_digest(result: dict) -> None:
    """周报生成成功后推一条微信摘要（Server酱）。未配 key 则静默跳过。

    含:①Top5 ②较上期上新/移出数。
    """
    from datetime import datetime, timezone
    from hot_projects.infra import notify

    ranked = result.get("ranked", [])
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    parts = [f"综合榜已更新，共 {len(ranked)} 个项目。"]

    # ② 较上期上新/移出
    diff = _prev_report_added_removed(result.get("report_path", ""), {n for n, _ in ranked})
    if diff:
        added, removed, prev_name = diff
        parts.append(f"较上期（{prev_name}）：上新 {added} · 移出 {removed}")

    # ① Top5
    top_lines = [
        f"{i}. {name} (+{info['growth']:,}, {info['star']:,}★)"
        for i, (name, info) in enumerate(ranked[:5], 1)
    ]
    if top_lines:
        parts.append("**Top5**\n" + "\n".join(top_lines))

    parts.append(f"报告：{os.path.basename(result.get('report_path', ''))}")
    notify.send(f"GitHub 周报 {today}", "\n\n".join(parts))


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


