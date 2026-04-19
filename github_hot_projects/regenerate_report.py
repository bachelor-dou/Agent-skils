#!/usr/bin/env python
"""
从历史日志恢复报告的独立脚本。

适用范围：
    - 这是当前仓库专用的恢复脚本，不是任意项目通用脚本。
    - 适用于本项目 agent / pipeline 产出的历史日志，要求日志里能找到：
            1. batch_check_growth 或 [Pipeline] 启动参数
            2. rank_candidates 或 [Pipeline] Step 3 排名参数
            3. [SEARCH] stargazers 查询 与 [GROWTH] 结果行
    - 如果日志格式和当前项目实现不一致，这个脚本不能保证可用。

做的事情：
    1. 从历史日志恢复最近一次运行的候选仓库。
    2. 复用现有 ranking.step2_rank_and_select 与 report.step3_generate_report。
    3. 先在内存里补齐缺失 desc，再生成报告，最后统一 save_db(db) 持久化。
    4. 通过 patch report.datetime 回放历史日期，不改动主逻辑代码。
    5. 生成出的报告文件名和报告标题日期都以日志中的历史日期为准，不以脚本执行当天为准。

可指定参数：
    --log                必填，历史日志路径，例如 logs/agent-2026-04-18.log
    --top-n              可选，覆盖日志里的 top_n
    --mode               可选，覆盖日志里的榜单模式，支持 comprehensive / hot_new
    --time-window-days   可选，覆盖日志里的增长统计窗口
    --growth-threshold   可选，覆盖日志里的增长阈值
    --new-project-days   可选，覆盖日志里的新项目窗口

用法示例：
    python regenerate_report.py --log logs/agent-2026-04-18.log
    python regenerate_report.py --log logs/agent-2026-04-18.log --top-n 130 --time-window-days 10
    python regenerate_report.py --log logs/scheduled-2026-04-18.log --mode comprehensive
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

if __package__:
    from .common.config import DEFAULT_SCORE_MODE, HOT_PROJECT_COUNT, STAR_GROWTH_THRESHOLD, TIME_WINDOW_DAYS
    from .common.db import load_db, save_db
    from .common.llm import call_llm_describe
    from .ranking import step2_rank_and_select
    from .report import step3_generate_report
else:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    common_config = importlib.import_module("github_hot_projects.common.config")
    common_db = importlib.import_module("github_hot_projects.common.db")
    common_llm = importlib.import_module("github_hot_projects.common.llm")
    ranking_module = importlib.import_module("github_hot_projects.ranking")
    report_module = importlib.import_module("github_hot_projects.report")

    DEFAULT_SCORE_MODE = common_config.DEFAULT_SCORE_MODE
    HOT_PROJECT_COUNT = common_config.HOT_PROJECT_COUNT
    STAR_GROWTH_THRESHOLD = common_config.STAR_GROWTH_THRESHOLD
    TIME_WINDOW_DAYS = common_config.TIME_WINDOW_DAYS
    load_db = common_db.load_db
    save_db = common_db.save_db
    call_llm_describe = common_llm.call_llm_describe
    step2_rank_and_select = ranking_module.step2_rank_and_select
    step3_generate_report = report_module.step3_generate_report

logger = logging.getLogger("regenerate_report")

_PIPELINE_START_RE = re.compile(
    r"\[Pipeline\] 启动: mode=(?P<mode>[^,]+), top_n=(?P<top_n>\d+), "
    r"new_project_days=(?P<new_project_days>[^,]+), time_window_days=(?P<time_window_days>\d+), "
    r"growth_threshold=(?P<growth_threshold>\d+)"
)
_PIPELINE_RANK_RE = re.compile(
    r"\[Pipeline\] Step 3: 排名 \(mode=(?P<mode>[^,]+), top_n=(?P<top_n>\d+)\)"
)
_SEARCH_STAR_RE = re.compile(
    r"\[SEARCH\] stargazers 查询: (?P<repo>[^ ]+) \(star=(?P<star>\d+)\)"
)
_GROWTH_WITH_VALUE_RE = re.compile(
    r"\[GROWTH\] (?P<repo>[^ ]+) .* growth=(?P<growth>-?\d+)"
)
_GROWTH_WINDOW_COUNT_RE = re.compile(
    r"\[GROWTH\] (?P<repo>[^ ]+) 采样精确: .* 窗口内 (?P<growth>\d+) 条"
)
_GROWTH_ESTIMATED_RE = re.compile(
    r"\[GROWTH\] (?P<repo>[^ ]+) .* estimated=(?P<growth>-?\d+)"
)


@dataclass(slots=True)
class RecoveryContext:
    start_idx: int
    rank_idx: int
    mode: str
    top_n: int
    time_window_days: int
    growth_threshold: int
    new_project_days: int | None
    report_date: str


def _parse_optional_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    value = raw.strip()
    if not value or value == "None":
        return None
    return int(value)


def _parse_pipe_params(line: str) -> dict[str, str]:
    if "|" not in line:
        return {}
    parsed: dict[str, str] = {}
    for part in [item.strip() for item in line.split("|")][1:]:
        if "=" not in part:
            continue
        key, raw_value = part.split("=", 1)
        parsed[key.strip()] = raw_value.split("(", 1)[0].strip()
    return parsed


def _find_last_index(lines: list[str], predicate) -> int:
    for idx in range(len(lines) - 1, -1, -1):
        if predicate(lines[idx]):
            return idx
    return -1


def _build_recovery_context(lines: list[str]) -> RecoveryContext:
    rank_idx = _find_last_index(
        lines,
        lambda line: "Tool 生效参数: rank_candidates" in line or "[Pipeline] Step 3: 排名" in line,
    )
    if rank_idx < 0:
        raise ValueError("日志中未找到 rank_candidates/Step 3 排名记录，无法恢复报告。")

    mode = DEFAULT_SCORE_MODE
    top_n = HOT_PROJECT_COUNT
    time_window_days = TIME_WINDOW_DAYS
    growth_threshold = STAR_GROWTH_THRESHOLD
    new_project_days: int | None = None

    rank_line = lines[rank_idx]
    if "Tool 生效参数: rank_candidates" in rank_line:
        rank_params = _parse_pipe_params(rank_line)
        mode = rank_params.get("mode", mode) or mode
        top_n = int(rank_params.get("top_n", str(top_n)))
        time_window_days = int(rank_params.get("growth_window_days", str(time_window_days)))
        new_project_days = _parse_optional_int(rank_params.get("creation_window_days"))
    else:
        match = _PIPELINE_RANK_RE.search(rank_line)
        if match:
            mode = match.group("mode").strip() or mode
            top_n = int(match.group("top_n"))

    start_idx = _find_last_index(
        lines[: rank_idx + 1],
        lambda line: "Tool 生效参数: batch_check_growth" in line or "[Pipeline] 启动:" in line,
    )
    if start_idx < 0:
        raise ValueError("日志中未找到 batch_check_growth/Pipeline 启动记录，无法恢复候选集。")

    start_line = lines[start_idx]
    if "Tool 生效参数: batch_check_growth" in start_line:
        batch_params = _parse_pipe_params(start_line)
        growth_threshold = int(batch_params.get("growth_threshold", str(growth_threshold)))
        time_window_days = int(batch_params.get("growth_window_days", str(time_window_days)))
        new_project_days = _parse_optional_int(batch_params.get("creation_window_days"))
    else:
        match = _PIPELINE_START_RE.search(start_line)
        if match:
            mode = match.group("mode").strip() or mode
            top_n = int(match.group("top_n"))
            time_window_days = int(match.group("time_window_days"))
            growth_threshold = int(match.group("growth_threshold"))
            new_project_days = _parse_optional_int(match.group("new_project_days"))

    return RecoveryContext(
        start_idx=start_idx,
        rank_idx=rank_idx,
        mode=mode,
        top_n=top_n,
        time_window_days=time_window_days,
        growth_threshold=growth_threshold,
        new_project_days=new_project_days,
        report_date=rank_line[:10],
    )


def _recover_candidates(lines: list[str], context: RecoveryContext, db: dict) -> dict[str, dict]:
    stars: dict[str, int] = {}
    growths: dict[str, int] = {}
    db_projects = db.get("projects", {})

    for line in lines[context.start_idx : context.rank_idx + 1]:
        match = _SEARCH_STAR_RE.search(line)
        if match:
            stars[match.group("repo")] = int(match.group("star"))
            continue

        match = _GROWTH_WINDOW_COUNT_RE.search(line)
        if match:
            growths[match.group("repo")] = int(match.group("growth"))
            continue

        match = _GROWTH_ESTIMATED_RE.search(line)
        if match:
            growths[match.group("repo")] = int(match.group("growth"))
            continue

        match = _GROWTH_WITH_VALUE_RE.search(line)
        if match:
            growths[match.group("repo")] = int(match.group("growth"))

    candidates: dict[str, dict] = {}
    for repo, growth in growths.items():
        if growth < context.growth_threshold:
            continue
        star = stars.get(repo) or int(db_projects.get(repo, {}).get("star", 0) or 0)
        if star <= 0:
            logger.warning("跳过缺少 star 的仓库: %s", repo)
            continue
        candidate = {"growth": growth, "star": star}
        created_at = db_projects.get(repo, {}).get("created_at", "")
        if created_at:
            candidate["created_at"] = created_at
        candidates[repo] = candidate

    return candidates


def _prefill_descs(top_projects: list[tuple[str, dict]], db: dict) -> None:
    pending = [repo for repo, _info in top_projects if not db.get("projects", {}).get(repo, {}).get("desc", "")]
    if not pending:
        logger.info("所有项目已有描述，无需额外补齐。")
        return

    logger.info("需要补齐描述 %d 个项目，先只写入内存，报告生成完成后再统一持久化。", len(pending))
    for idx, repo in enumerate(pending, 1):
        db_projects = db.get("projects", {})
        saved = db_projects.get(repo, {})
        html_url = f"https://github.com/{repo}"
        logger.info("[%d/%d] LLM 生成描述: %s", idx, len(pending), repo)
        desc = call_llm_describe(repo, saved, html_url)
        if not desc:
            logger.warning("  -> 描述生成失败: %s", repo)
            continue
        db_projects = db.get("projects", {})
        if repo not in db_projects:
            logger.warning("  -> %s 不在 DB 中，跳过 desc 持久化", repo)
            continue
        db_projects[repo]["desc"] = desc

    remaining = [repo for repo, _info in top_projects if not db.get("projects", {}).get(repo, {}).get("desc", "")]
    if remaining:
        logger.warning("仍有 %d 个项目缺少 desc，report 阶段会继续补齐。", len(remaining))
    else:
        logger.info("描述已全部预填，report 阶段应直接生成文件。")


def recover_report_from_log(
    log_path: str,
    *,
    top_n: int | None = None,
    mode: str | None = None,
    time_window_days: int | None = None,
    growth_threshold: int | None = None,
    new_project_days: int | None = None,
) -> dict:
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"日志文件不存在: {log_path}")

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    context = _build_recovery_context(lines)
    if top_n is not None:
        context.top_n = top_n
    if mode is not None:
        context.mode = mode
    if time_window_days is not None:
        context.time_window_days = time_window_days
    if growth_threshold is not None:
        context.growth_threshold = growth_threshold
    if new_project_days is not None:
        context.new_project_days = new_project_days

    db = load_db()
    candidates = _recover_candidates(lines, context, db)
    if not candidates:
        raise ValueError("日志中未恢复出满足阈值的候选仓库，无法生成报告。")

    sorted_all = step2_rank_and_select(
        candidates,
        mode=context.mode,
        db=db,
        new_project_days=context.new_project_days,
        prefiltered_new_project_days=context.new_project_days if context.mode == "hot_new" else None,
    )
    top_projects = sorted_all[: context.top_n]
    if not top_projects:
        raise ValueError("候选仓库恢复成功，但排序后为空。")

    _prefill_descs(top_projects, db)
    db["date"] = context.report_date

    report_dt = datetime.strptime(context.report_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    class _FixedDatetime:
        @staticmethod
        def now(tz=None):
            return report_dt if tz else report_dt.replace(tzinfo=None)

        @staticmethod
        def strptime(value: str, fmt: str):
            return datetime.strptime(value, fmt)

    with patch("github_hot_projects.report.datetime", _FixedDatetime):
        report_path = step3_generate_report(
            top_projects,
            db,
            mode=context.mode,
            new_project_days=context.new_project_days if context.mode == "hot_new" else None,
            time_window_days=context.time_window_days,
        )

    if not report_path:
        raise RuntimeError("报告生成失败。")

    save_db(db)
    return {
        "report_path": report_path,
        "report_date": context.report_date,
        "mode": context.mode,
        "top_n": len(top_projects),
        "candidates_count": len(candidates),
        "time_window_days": context.time_window_days,
        "growth_threshold": context.growth_threshold,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从历史日志恢复并重新生成 Markdown 报告")
    parser.add_argument("--log", required=True, help="日志文件路径，例如 logs/agent-2026-04-18.log")
    parser.add_argument("--top-n", type=int, help="覆盖日志中的 top_n")
    parser.add_argument("--mode", choices=["comprehensive", "hot_new"], help="覆盖日志中的榜单模式")
    parser.add_argument("--time-window-days", type=int, help="覆盖日志中的增长统计窗口")
    parser.add_argument("--growth-threshold", type=int, help="覆盖日志中的增长阈值")
    parser.add_argument("--new-project-days", type=int, help="覆盖日志中的新项目窗口")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    args = _parse_args()
    result = recover_report_from_log(
        args.log,
        top_n=args.top_n,
        mode=args.mode,
        time_window_days=args.time_window_days,
        growth_threshold=args.growth_threshold,
        new_project_days=args.new_project_days,
    )
    print(f"报告已生成: {result['report_path']}")
    print(
        "恢复摘要: "
        f"date={result['report_date']}, mode={result['mode']}, "
        f"ranked={result['top_n']}, candidates={result['candidates_count']}, "
        f"window={result['time_window_days']}d"
    )


if __name__ == "__main__":
    main()