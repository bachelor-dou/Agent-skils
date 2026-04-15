#!/usr/bin/env python
"""
日志恢复脚本：指定日志文件即可自动提取日期、恢复候选、生成对应日期的 Markdown 报告。

从日志文件名（agent-YYYY-MM-DD.log）自动提取日期，
若文件名无法匹配则从日志内容首行提取。

全部调用项目现有接口：
  - ranking.step2_rank_and_select  → 评分排序
  - common.llm.call_llm_describe   → 逐条生成描述（每条即时 save_db 防丢失）
  - report.step3_generate_report   → 生成报告（已有 desc 的不会重复调 LLM）

用法:
    cd github_hot_projects
    python regenerate_report.py --log logs/agent-2026-04-14.log [--top-n 100]
"""

import argparse
import logging
import os
import re
import sys
from datetime import datetime, timezone
from unittest.mock import patch

# ── 确保可以 import 项目包（向上两级到 Agent-skils/）──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from github_hot_projects.common.config import DATA_DIR
from github_hot_projects.common.db import load_db, save_db
from github_hot_projects.common.llm import call_llm_describe
from github_hot_projects.ranking import step2_rank_and_select
from github_hot_projects.report import step3_generate_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("regenerate_report")


# ══════════════════════════════════════════════════════════════
# 1. 从日志提取候选仓库 (growth + star)
# ══════════════════════════════════════════════════════════════

CANDIDATE_RE = re.compile(
    r"\[OK\] 候选: (?P<name>\S+) \| growth=(?P<growth>\d+) \| star=(?P<star>\d+)"
)

DATE_FROM_FILENAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
DATE_FROM_LOGLINE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")


def detect_date_from_log(log_path: str) -> str:
    """从日志文件名或首行自动提取日期（YYYY-MM-DD）。"""
    # 优先从文件名提取
    basename = os.path.basename(log_path)
    m = DATE_FROM_FILENAME_RE.search(basename)
    if m:
        return m.group(1)

    # 回退：从日志首行提取
    with open(log_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    m = DATE_FROM_LOGLINE_RE.search(first_line)
    if m:
        return m.group(1)

    logger.error(f"无法从日志文件名或内容中提取日期: {log_path}")
    sys.exit(1)


def parse_candidates_from_log(log_path: str) -> dict[str, dict]:
    """解析日志中所有 [OK] 候选 行，返回 {full_name: {growth, star}}."""
    candidates: dict[str, dict] = {}
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = CANDIDATE_RE.search(line)
            if m:
                name = m.group("name")
                candidates[name] = {
                    "growth": int(m.group("growth")),
                    "star": int(m.group("star")),
                }
    return candidates


# ══════════════════════════════════════════════════════════════
# 2. 逐条调用 LLM 并即时落盘（增强保护，其余调用原版接口）
# ══════════════════════════════════════════════════════════════

def prefill_descs(
    top_projects: list[tuple[str, dict]], db: dict
) -> None:
    """
    对 Top N 中 desc 为空的项目，逐条调用 call_llm_describe 生成描述，
    每条成功后立即 save_db 落盘，防止中途失败丢失已有描述。

    之后调用 step3_generate_report 时，这些项目已有 desc 缓存，
    不会重复调用 LLM。
    """
    db_projects = db.get("projects", {})

    need_llm = 0
    for full_name, _info in top_projects:
        saved = db_projects.get(full_name, {})
        if not saved.get("desc", ""):
            need_llm += 1

    if not need_llm:
        logger.info("所有项目已有描述，无需调用 LLM。")
        return

    logger.info(f"需要生成描述 {need_llm} 个项目，按顺序调用 LLM...")
    done = saved_count = 0

    for idx, (full_name, _info) in enumerate(top_projects):
        saved = db_projects.get(full_name, {})
        if saved.get("desc", ""):
            continue
        done += 1
        html_url = f"https://github.com/{full_name}"
        logger.info(f"[{done}/{need_llm}] LLM 生成描述: {full_name}")

        desc = call_llm_describe(full_name, saved, html_url)
        if desc:
            if full_name in db_projects:
                db_projects[full_name]["desc"] = desc
                # 每生成一条就落盘，防中途丢失
                save_db(db)
                saved_count += 1
                logger.info(f"  -> 已保存 ({saved_count}/{need_llm})")
            else:
                logger.warning(f"  -> {full_name} 不在 DB 中，描述未持久化")
        else:
            logger.warning(f"  -> 描述生成失败: {full_name}")

    logger.info(f"LLM 描述生成完毕: 成功 {saved_count}/{need_llm}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="从日志恢复候选数据，重新生成 LLM 描述和报告"
    )
    parser.add_argument(
        "--log",
        required=True,
        help="日志文件路径，日期从文件名自动提取 (如 logs/agent-2026-04-14.log)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="取前 N 个项目生成报告 (默认: 100)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.log):
        logger.error(f"日志文件不存在: {args.log}")
        sys.exit(1)

    # 0) 从日志自动提取日期
    report_date_str = detect_date_from_log(args.log)
    logger.info(f"检测到报告日期: {report_date_str}")

    # 1) 解析日志 → 候选列表
    logger.info(f"解析日志: {args.log}")
    candidates = parse_candidates_from_log(args.log)
    if not candidates:
        logger.error("未从日志中提取到候选仓库，请检查日志路径和格式。")
        sys.exit(1)
    logger.info(f"从日志提取 {len(candidates)} 个候选仓库。")

    # 2) 加载 DB
    db = load_db()
    db_projects = db.get("projects", {})
    logger.info(
        f"DB 加载: {len(db_projects)} 个项目, date={db.get('date')}, "
        f"valid={db.get('valid')}"
    )

    # 3) 调用原版 ranking 模块排序（评分逻辑与主流程完全一致）
    sorted_all = step2_rank_and_select(candidates, mode="comprehensive", db=db)
    top_projects = sorted_all[: args.top_n]
    logger.info(f"取 Top {args.top_n} 个项目生成报告。")

    # 4) 逐条调 LLM 生成描述 + 即时落盘（增强保护）
    prefill_descs(top_projects, db)

    # 5) 调用原版 step3_generate_report 生成报告
    #    此时所有 desc 已在 DB 中，不会重复调 LLM
    #    通过 mock datetime.now 让报告使用指定日期
    report_date = datetime.strptime(report_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    class _FixedDatetime:
        """让 datetime.now() 返回指定日期，其余行为不变。"""
        @staticmethod
        def now(tz=None):
            return report_date if tz else report_date.replace(tzinfo=None)

    with patch("github_hot_projects.report.datetime", _FixedDatetime):
        report_path = step3_generate_report(top_projects, db, mode="comprehensive")

    if report_path:
        # 最终保存 DB
        save_db(db)
        logger.info(f"完成! 报告: {report_path}")
    else:
        # 即使报告生成失败，LLM 描述已在步骤 4 中逐条落盘，不会丢失
        save_db(db)
        logger.error("报告生成失败，但 LLM 描述已保存到 DB，可重新运行本脚本。")
        sys.exit(1)


if __name__ == "__main__":
    main()
