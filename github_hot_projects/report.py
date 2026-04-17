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
    NEW_PROJECT_DAYS,
    REPORT_DIR,
    STAR_GROWTH_THRESHOLD,
    TIME_WINDOW_DAYS,
)
from .common.llm import call_llm_describe

logger = logging.getLogger("discover_hot")

REPO_COPY_ICON_SVG = (
    '<span class="repo-copy-icon" aria-hidden="true">'
    '<svg viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M7.25 6.75H4.75C3.92157 6.75 3.25 7.42157 3.25 8.25V15.25C3.25 16.0784 3.92157 16.75 4.75 16.75H11.75C12.5784 16.75 13.25 16.0784 13.25 15.25V12.75" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
    '<path d="M8.25 4.75C8.25 3.92157 8.92157 3.25 9.75 3.25H15.25C16.0784 3.25 16.75 3.92157 16.75 4.75V10.25C16.75 11.0784 16.0784 11.75 15.25 11.75H9.75C8.92157 11.75 8.25 11.0784 8.25 10.25V4.75Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
    '</svg>'
    '</span>'
)


# 模式 → 文件后缀 / 标题映射
_MODE_META = {
    "comprehensive": ("", "GitHub 热门项目"),
    "hot_new":       ("_hot_new", "GitHub 新项目热度榜"),
}


def step3_generate_report(
    top_projects: list[tuple[str, dict]], db: dict,
    mode: str = "comprehensive",
    new_project_days: int | None = None,
) -> str:
    """
    为 Top N 项目生成 LLM 描述 + 输出 Markdown 报告。

    Args:
        mode: 排名模式，"comprehensive" 或 "hot_new"，影响文件名和标题。
        new_project_days: 新项目窗口天数。hot_new 模式下若提供，会写入文件后缀避免覆盖。

    Returns:
        报告文件路径。
    """
    os.makedirs(REPORT_DIR, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    suffix, title_prefix = _MODE_META.get(mode, ("", "GitHub 热门项目"))
    if mode == "hot_new" and new_project_days is not None:
        suffix = f"{suffix}_{new_project_days}d"
        title_prefix = f"{title_prefix}（近{new_project_days}天）"
    report_path = os.path.join(REPORT_DIR, f"{today}{suffix}.md")
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

    hot_new_window = new_project_days if new_project_days is not None else NEW_PROJECT_DAYS
    lines: list[str] = [f"# {title_prefix} — {today}\n"]
    summary_parts = [
        f"共 {len(top_projects)} 个项目",
        f"增长统计窗口: {TIME_WINDOW_DAYS} 天",
        f"增长阈值: >={STAR_GROWTH_THRESHOLD} stars",
        f"最低 star: >={MIN_STAR_FILTER}",
    ]
    if mode == "hot_new":
        summary_parts.insert(1, f"新项目创建窗口: <= {hot_new_window} 天")
    lines.append(f"> {' | '.join(summary_parts)}\n")

    for idx, (full_name, info) in enumerate(top_projects, 1):
        growth = info["growth"]
        star = info["star"]
        html_url = f"https://github.com/{full_name}"
        detailed_desc = desc_results.get(full_name, "")

        lines.append(
            f"## {idx}. {full_name}（+{growth}，⭐{star}） <button class=\"repo-copy-btn repo-copy-btn--title repo-copy-btn--icon\" type=\"button\" data-repo=\"{full_name}\" title=\"复制 {full_name}\" aria-label=\"复制 {full_name}\">{REPO_COPY_ICON_SVG}</button>\n"
        )
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
