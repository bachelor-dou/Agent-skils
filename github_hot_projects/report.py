"""
报告生成
========
为 Top N 项目生成 LLM 描述 + 输出 Markdown 报告。
"""

import logging
import os
from datetime import datetime, timezone

from .common.config import (
    MIN_STAR_FILTER,
    REPORT_DIR,
    STAR_GROWTH_THRESHOLD,
    TIME_WINDOW_DAYS,
)
from .common.llm import call_llm_describe

logger = logging.getLogger("discover_hot")


def step3_generate_report(
    top_projects: list[tuple[str, dict]], db: dict
) -> str:
    """
    为 Top N 项目生成 LLM 描述 + 输出 Markdown 报告。

    Returns:
        报告文件路径。
    """
    os.makedirs(REPORT_DIR, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    report_path = os.path.join(REPORT_DIR, f"{today}.md")
    db_projects = db.get("projects", {})

    need_llm: list[tuple[int, str, str, dict]] = []
    desc_results: dict[str, str] = {}

    for idx, (full_name, info) in enumerate(top_projects):
        saved = db_projects.get(full_name, {})
        existing_desc = saved.get("desc", "")
        html_url = f"https://github.com/{full_name}"
        if existing_desc:
            desc_results[full_name] = existing_desc
        else:
            need_llm.append((idx + 1, full_name, html_url, saved))

    if need_llm:
        logger.info(f"报告生成: 需要生成描述 {len(need_llm)} 个项目，按顺序调用 LLM...")
        for idx, full_name, html_url, saved in need_llm:
            logger.info(f"[{idx}/{len(top_projects)}] LLM 生成描述: {full_name}")
            desc = call_llm_describe(full_name, saved, html_url)
            if desc:
                desc_results[full_name] = desc
                if full_name in db_projects:
                    db_projects[full_name]["desc"] = desc
            else:
                desc_results.setdefault(full_name, "")

    lines: list[str] = [f"# GitHub 热门项目 — {today}\n"]
    lines.append(
        f"> 共 {len(top_projects)} 个项目 | "
        f"窗口期: {TIME_WINDOW_DAYS} 天 | "
        f"增长阈值: >={STAR_GROWTH_THRESHOLD} stars | "
        f"最低 star: >={MIN_STAR_FILTER}\n"
    )

    for idx, (full_name, info) in enumerate(top_projects, 1):
        growth = info["growth"]
        star = info["star"]
        html_url = f"https://github.com/{full_name}"
        detailed_desc = desc_results.get(full_name, "")

        lines.append(f"## {idx}. {full_name}（+{growth}，total {star}）\n")
        lines.append(f"链接: {html_url}\n")
        lines.append(f"{detailed_desc}\n")
        lines.append("---\n")

    report_content = "\n".join(lines)
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
    except IOError as e:
        logger.error(f"报告写入失败: {report_path}, {e}")
        return ""

    logger.info(f"报告生成完成: {report_path}")
    return report_path
