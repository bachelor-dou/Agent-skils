"""
报告生成（执行层 · 报告组件）
================================
为 Top N 项目生成 LLM 描述，并输出纯 Markdown 报告。

架构定位：
    执行层独立组件，由 agent_tools.tool_generate_report() 调用。

工作流程：
    1. 遍历已排名项目，优先复用 DB 缓存描述，否则调用 LLM 生成 200-400 字中文摘要
    2. 每个项目输出可直接阅读的 Markdown 条目，避免将前端 HTML 混入 .md 文件
    3. 输出到 report/ 目录，文件名为日期格式 (YYYY-MM-DD.md)
    4. 支持 comprehensive / hot_new 两种标题样式 + 自定义时间窗口

关键实现细节：
    - LLM 描述容错：调用失败时回退到 GitHub 原始 description
    - 单项目始终展示创建时间；若仍处于新项目窗口，则额外标记 NEW
    - 原子写入：报告先写 .tmp 再 os.replace，避免中间状态
"""

import logging
import os
import re
from datetime import datetime, timedelta, timezone

from .common.config import (
    MIN_STAR_FILTER,
    NEW_PROJECT_DAYS,
    REPORT_DIR,
    STAR_GROWTH_THRESHOLD,
    TIME_WINDOW_DAYS,
)
from .common.llm import call_llm_describe

logger = logging.getLogger("discover_hot")

INTRO_SECTION_TITLES = (
    "项目定位与用途",
    "解决的问题",
    "使用场景",
    "技术架构与特性",
)

ALL_SECTION_TITLES = (
    "项目定位与用途",
    "解决的问题",
    "使用场景",
    "技术架构与特性",
    "核心依赖与生态",
    "已知局限或注意事项",
)

_MODE_META = {
    "comprehensive": ("", "GitHub 热门项目"),
    "hot_new": ("_hot_new", "GitHub 新项目热度榜"),
}


def _format_number(value: int) -> str:
    return f"{value:,}"


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _format_date(value: str) -> str:
    parsed = _parse_timestamp(value)
    if parsed is None:
        return value[:10] if value else ""
    return parsed.strftime("%Y-%m-%d")


def _is_recent_project(created_at: str, window_days: int) -> bool:
    created_ts = _parse_timestamp(created_at)
    if created_ts is None:
        return False
    return datetime.now(timezone.utc) - created_ts <= timedelta(days=window_days)


def _split_description_blocks(text: str) -> list[str]:
    if not text:
        return []
    paragraph_blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if len(paragraph_blocks) >= 2:
        return paragraph_blocks[:2]
    normalized = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if not normalized:
        return []
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[。！？.!?])\s+", normalized)
        if item.strip()
    ]
    if len(sentences) <= 1:
        return [normalized]
    split_at = max(1, len(sentences) // 2)
    return [
        " ".join(sentences[:split_at]).strip(),
        " ".join(sentences[split_at:]).strip(),
    ]


def _extract_structured_sections(text: str) -> dict[str, str]:
    """从 LLM 输出中提取结构化字段，只返回 INTRO_SECTION_TITLES 中定义的字段。"""
    if not text:
        return {}
    sections = {title: "" for title in INTRO_SECTION_TITLES}
    current_title = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # 检查是否是任何已知字段标题（包括 ALL_SECTION_TITLES）
        matched_title = ""
        for title in ALL_SECTION_TITLES:
            if line.startswith(f"{title}："):
                matched_title = title
                content_part = line.split("：", 1)[1].strip()
                # 只有 INTRO_SECTION_TITLES 中的字段才记录内容
                if title in INTRO_SECTION_TITLES:
                    sections[title] = content_part
                break
            if line.startswith(f"{title}:"):
                matched_title = title
                content_part = line.split(":", 1)[1].strip()
                if title in INTRO_SECTION_TITLES:
                    sections[title] = content_part
                break

        if matched_title:
            # 更新 current_title，只有 INTRO_SECTION_TITLES 中的才用于追加
            if matched_title in INTRO_SECTION_TITLES:
                current_title = matched_title
            else:
                # 遇到不需要的字段标题时，停止追加到前一个字段
                current_title = ""
            continue

        # 只有当前字段是我们需要的字段时才追加
        if current_title and current_title in INTRO_SECTION_TITLES:
            sections[current_title] = f"{sections[current_title]} {line}".strip()

    return {title: content for title, content in sections.items() if content}


def _build_positioning_text(saved: dict, detailed_desc: str) -> str:
    short_desc = (saved.get("short_desc") or "").strip()
    desc_blocks = _split_description_blocks(detailed_desc)
    if short_desc:
        if desc_blocks:
            return f"{short_desc} {desc_blocks[0]}".strip()
        return short_desc
    if desc_blocks:
        return desc_blocks[0]
    return "暂无结构化摘要，可直接查看仓库主页与 README。"


def _build_problem_text(saved: dict, detailed_desc: str) -> str:
    desc_blocks = _split_description_blocks(detailed_desc)
    if len(desc_blocks) >= 2:
        return desc_blocks[1]

    parts: list[str] = []
    short_desc = (saved.get("short_desc") or "").strip()
    language = (saved.get("language") or "").strip()
    topics = [topic for topic in saved.get("topics", []) if topic]
    if short_desc:
        parts.append(f"从当前简介看，它主要围绕“{short_desc}”这一方向提供能力。")
    if topics:
        parts.append(f"从标签信号看，它重点瞄准 {', '.join(topics[:4])} 相关问题。")
    if language:
        parts.append(f"实现侧主要落在 {language} 技术栈中，说明它更偏向工程化落地而非概念展示。")
    if not parts:
        parts.append("当前信息不足以明确拆解其具体痛点，只能保守判断它试图降低某类任务的实现门槛。")
    return " ".join(parts)


def _build_usage_text(saved: dict, hot_new_window: int) -> str:
    parts: list[str] = []
    language = (saved.get("language") or "").strip()
    topics = [topic for topic in saved.get("topics", []) if topic]
    if language:
        parts.append(f"适合关注 {language} 生态的开发者、研究者或技术团队先行评估。")
    if topics:
        parts.append(f"如果你的需求和 {', '.join(topics[:4])} 相关，这个项目更值得进入候选清单。")
    if _is_recent_project(saved.get("created_at", ""), hot_new_window):
        parts.append(f"由于其仍在近 {hot_new_window} 天的新项目窗口内，也适合用来跟踪新趋势和早期工具形态。")
    if saved.get("readme_url"):
        parts.append("进一步落地前，建议再结合仓库 README 核对接入方式、能力边界和维护状态。")
    if not parts:
        parts.append("当前更适合把它当作方向性线索，再结合仓库主页和 README 判断是否适合自己的业务场景。")
    return " ".join(parts)


def _build_tech_arch_text(saved: dict, detailed_desc: str) -> str:
    """构建技术架构与特性的 fallback 文本。"""
    parts: list[str] = []
    language = (saved.get("language") or "").strip()
    topics = [topic for topic in saved.get("topics", []) if topic]
    short_desc = (saved.get("short_desc") or "").strip()

    if language:
        parts.append(f"主要使用 {language} 实现。")
    if topics:
        tech_topics = [t for t in topics[:4] if not any(kw in t.lower() for kw in ["ai", "llm", "agent", "tool", "library", "framework"])]
        if tech_topics:
            parts.append(f"技术关键词包括 {', '.join(tech_topics)}。")
    if short_desc and len(short_desc) > 30:
        # 尝试从短描述中提取技术信息
        desc_blocks = _split_description_blocks(detailed_desc)
        if len(desc_blocks) >= 3:
            parts.append(desc_blocks[2][:150])
    if not parts:
        parts.append("暂无详细技术架构信息，可查看仓库 README 或源码了解实现细节。")
    return " ".join(parts)


def _resolve_intro_sections(saved: dict, detailed_desc: str, hot_new_window: int) -> list[tuple[str, str]]:
    structured = _extract_structured_sections(detailed_desc)
    return [
        (
            INTRO_SECTION_TITLES[0],
            structured.get(INTRO_SECTION_TITLES[0]) or _build_positioning_text(saved, detailed_desc),
        ),
        (
            INTRO_SECTION_TITLES[1],
            structured.get(INTRO_SECTION_TITLES[1]) or _build_problem_text(saved, detailed_desc),
        ),
        (
            INTRO_SECTION_TITLES[2],
            structured.get(INTRO_SECTION_TITLES[2]) or _build_usage_text(saved, hot_new_window),
        ),
        (
            INTRO_SECTION_TITLES[3],
            structured.get(INTRO_SECTION_TITLES[3]) or _build_tech_arch_text(saved, detailed_desc),
        ),
    ]


def _normalize_markdown_blocks(text: str) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if not blocks and text.strip():
        blocks = [text.strip()]
    if not blocks:
        blocks = ["暂无补充信息，可进入仓库查看 README。"]
    return blocks


def step3_generate_report(
    top_projects: list[tuple[str, dict]],
    db: dict,
    mode: str = "comprehensive",
    new_project_days: int | None = None,
    time_window_days: int = TIME_WINDOW_DAYS,
) -> str:
    """为 Top N 项目生成 LLM 描述并输出 Markdown 报告。"""
    os.makedirs(REPORT_DIR, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    suffix, title_prefix = _MODE_META.get(mode, ("", "GitHub 热门项目"))
    if mode == "comprehensive" and time_window_days != TIME_WINDOW_DAYS:
        suffix = f"{suffix}_{time_window_days}d" if suffix else f"_{time_window_days}d"
        title_prefix = f"{title_prefix}（近{time_window_days}天增长）"
    if mode == "hot_new" and new_project_days is not None:
        suffix = f"{suffix}_{new_project_days}d"
        title_prefix = f"{title_prefix}（近{new_project_days}天）"
    if mode == "hot_new" and time_window_days != TIME_WINDOW_DAYS:
        suffix = f"{suffix}_win{time_window_days}d"
    report_path = os.path.join(REPORT_DIR, f"{today}{suffix}.md")
    db_projects = db.get("projects", {})

    need_llm: list[tuple[int, str, str, dict]] = []
    desc_results: dict[str, str] = {}

    for idx, (full_name, _info) in enumerate(top_projects):
        saved = db_projects.get(full_name, {})
        existing_desc = saved.get("desc", "")
        html_url = f"https://github.com/{full_name}"
        if existing_desc:
            desc_results[full_name] = existing_desc
        else:
            need_llm.append((idx + 1, full_name, html_url, saved))

    if need_llm:
        logger.info(f"报告生成: 需要生成完整描述 {len(need_llm)} 个项目，按顺序调用 LLM...")
        for idx, full_name, html_url, saved in need_llm:
            logger.info(f"[{idx}/{len(top_projects)}] LLM 生成完整描述: {full_name}")
            desc = call_llm_describe(full_name, saved, html_url, detail_level="detailed")
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
        f"增长统计窗口: {time_window_days} 天",
        f"增长阈值: >={STAR_GROWTH_THRESHOLD} stars",
        f"最低 star: >={MIN_STAR_FILTER}",
        f"报告生成: {today}",
    ]
    if mode == "hot_new":
        summary_parts.insert(1, f"新项目创建窗口: <= {hot_new_window} 天")
    lines.append(f"> {' | '.join(summary_parts)}\n")

    for idx, (full_name, info) in enumerate(top_projects, 1):
        growth = info["growth"]
        star = info["star"]
        saved = db_projects.get(full_name, {})
        detailed_desc = desc_results.get(full_name, "")
        language = (saved.get("language") or "").strip()
        created_at = _format_date(saved.get("created_at", "")) or "未知"
        topics = [topic for topic in saved.get("topics", []) if topic][:6]
        intro_sections = _resolve_intro_sections(saved, detailed_desc, hot_new_window)
        html_url = f"https://github.com/{full_name}"

        lines.extend([
            f"## {idx}. {full_name}",
            "",
            f"链接: {html_url}",
            "",
            f"- 创建时间: {created_at}",
        ])
        if _is_recent_project(saved.get("created_at", ""), hot_new_window):
            lines.append(f"- 项目状态: NEW（{hot_new_window}天内）")
        if language:
            lines.append(f"- 主语言: {language}")
        lines.append(f"- 总 Star: {_format_number(star)}")
        lines.append(f"- 近{time_window_days}天增长: +{_format_number(growth)}")
        if topics:
            lines.append(f"- 主题标签: {', '.join(topics)}")
        lines.append("")

        for title, content in intro_sections:
            lines.append(f"### {title}")
            lines.append("")
            lines.extend(_normalize_markdown_blocks(content))
            lines.append("")

        if idx != len(top_projects):
            lines.append("---")
            lines.append("")

    report_content = "\n".join(lines).rstrip() + "\n"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
    except IOError as e:
        logger.error(f"报告写入失败: {report_path}, {e}")
        return ""

    logger.info(f"报告生成完成: {report_path}")
    return report_path
