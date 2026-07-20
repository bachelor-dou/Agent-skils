"""
LLM 调用模块
=============
调用 LLM（兼容 OpenAI /v1/chat/completions 格式）为项目生成结构化介绍。

调用方包括：报告生成（Top N 中 desc 为空的项目）、check_repo_growth、describe_project 等。
传入仓库元信息（description、language、topics、readme_url 等）供 LLM 综合总结。
"""

import logging

from .llm_client import get_client as _get_client

logger = logging.getLogger("hot_projects")


def _llm_configured() -> bool:
    """任一模型配置了 key 即视为可用。"""
    return bool(_get_client().usable())


def _extract_content(data: dict | None) -> str:
    if not data:
        return ""
    return ((data.get("choices") or [{}])[0].get("message", {}).get("content") or "").strip()


def _truncate_text(text: str, limit: int) -> str:
    clean = (text or "").strip()
    if len(clean) <= limit:
        return clean
    return clean[:limit] + "..."


def _format_recent_releases(releases: list[dict]) -> str:
    if not isinstance(releases, list) or not releases:
        return ""

    parts: list[str] = []
    for item in releases[:5]:
        if not isinstance(item, dict):
            continue
        tag = str(item.get("tag_name") or item.get("name") or "").strip()
        date = str(item.get("published_at") or "").strip()
        if not tag:
            continue
        state = []
        if item.get("prerelease"):
            state.append("prerelease")
        if item.get("draft"):
            state.append("draft")
        state_text = f"({'/'.join(state)})" if state else ""
        date_text = date[:10] if date else ""
        if date_text:
            parts.append(f"{tag}{state_text}@{date_text}")
        else:
            parts.append(f"{tag}{state_text}")
    return "; ".join(parts)


def _format_recent_commits(commits: list[dict]) -> str:
    if not isinstance(commits, list) or not commits:
        return ""

    parts: list[str] = []
    for item in commits[:8]:
        if not isinstance(item, dict):
            continue
        date = str(item.get("date") or "").strip()
        message = str(item.get("message") or "").strip()
        if not date and not message:
            continue
        date_text = date[:10] if date else ""
        message_text = _truncate_text(message, 60)
        if date_text and message_text:
            parts.append(f"{date_text}:{message_text}")
        elif date_text:
            parts.append(date_text)
        else:
            parts.append(message_text)
    return "; ".join(parts)


def call_llm_describe(repo_name: str, repo_info: dict, html_url: str,
                      detail_level: str = "standard") -> str:
    """
    调用 LLM 生成项目介绍。

    Args:
        repo_name: "owner/repo"
        repo_info: DB 中的仓库信息字典（含 short_desc / language / topics / readme_url）
        html_url:  项目 GitHub 页面 URL
        detail_level: "standard"=260-520字三段式, "detailed"=400-800字四段式

    Returns:
        LLM 生成的描述文本；单次调用失败（含 LLMClient 多平台顺序回退与重试）后返回空字符串。
    """
    if not _llm_configured():
        logger.warning("LLM 未配置，跳过描述生成。")
        return ""

    # 构建信息块供 LLM 参考
    info_parts = [f"项目名称: {repo_name}", f"项目地址: {html_url}"]
    if short_desc := repo_info.get("short_desc", ""):
        info_parts.append(f"官方简介: {short_desc}")
    if topics := repo_info.get("topics", []):
        info_parts.append(f"标签: {', '.join(topics)}")
    if readme_url := repo_info.get("readme_url", ""):
        info_parts.append(f"README链接（仅供标识，不能视为已读取内容）: {readme_url}")
    if readme_excerpt := repo_info.get("readme_excerpt", ""):
        info_parts.append(f"README摘录（已读取文本，可能截断）: {_truncate_text(str(readme_excerpt), 3200)}")
    if recent_releases := _format_recent_releases(repo_info.get("recent_releases", [])):
        info_parts.append(f"近期发布节奏: {recent_releases}")
    if recent_commits := _format_recent_commits(repo_info.get("recent_commits", [])):
        info_parts.append(f"近期提交线索: {recent_commits}")

    prompt = (
        "请基于以下已提供信息，用中文总结这个 GitHub 开源项目。\n"
        "输出要求：\n"
        "1. 只能基于下方明确提供的信息，不要把项目地址或 README 链接当作已读取内容。\n"
        "2. 不要补充未在输入中出现、且无法确认的外部知识；信息不足时使用保守表述。\n"
        "3. 如果输入中包含 README摘录、发布记录或提交记录，可以引用；若缺失，请明确说明信息不足。\n"
    )
    if detail_level == "detailed":
        prompt += (
            "4. 必须严格输出以下四个字段，每个字段单独成段，字段间用换行分隔：\n"
            "项目定位与用途：...（100-200字，说明是什么、做什么，简要介绍核心定位）\n"
            "解决的问题：...（100-200字，聚焦核心痛点，说明为什么需要这个项目）\n"
            "使用场景：...（100-200字，列举典型应用场景和目标用户）\n"
            "技术架构与特性：...（100-200字，关键技术栈、架构特点和核心特性）\n"
            "5. 总长度控制在 400-800 字，信息详实但不冗余。\n"
            "6. 不要使用列表、不要加 Markdown 标题、字段名后必须换行。\n\n"
        )
    else:
        prompt += (
            "4. 必须严格输出以下三个字段，字段名保持原样：\n"
            "项目定位与用途：...\n"
            "解决的问题：...\n"
            "使用场景：...\n"
            "5. 每个字段建议 80-160 字，总长度控制在 260-520 字。\n"
            "6. 不要使用列表、不要加 Markdown 标题、不要输出字段以外的说明。\n\n"
        )
    prompt += "\n".join(info_parts) + "\n"

    max_tokens = 2048 if detail_level == "detailed" else 1536
    data = _get_client().chat(
        [{"role": "user", "content": prompt}],
        lite=True,
        max_tokens=max_tokens,
        temperature=0.2,
        enable_thinking=False,
    )
    content = _extract_content(data)
    if content:
        return content

    logger.warning(f"LLM 描述生成失败（所有平台均失败），跳过描述: {repo_name}")
    return ""


def batch_condense_descriptions(repos: list[dict], max_chars: int = 70) -> list[str]:
    """
    用 LLM 批量浓缩项目描述，每个不超过 max_chars 字。

    Args:
        repos: [{"full_name": "owner/repo", "description": "..."}]
        max_chars: 每条描述的最大字符数

    Returns:
        与 repos 等长的浓缩描述列表；LLM 失败时回退截断原文。
    """
    if not _llm_configured():
        return [r.get("description", "")[:max_chars] for r in repos]

    if not repos:
        return []

    lines = []
    for i, r in enumerate(repos):
        desc = r.get("description", "").strip()
        if desc:
            lines.append(f"{i+1}. {r['full_name']}: {desc}")
        else:
            lines.append(f"{i+1}. {r['full_name']}: (无描述)")

    prompt = (
        f"请将以下 {len(repos)} 个 GitHub 项目的描述各浓缩为不超过{max_chars}字的中文简介。\n"
        f"要求：保留核心功能和用途，去掉修饰语，每行格式为「序号. 浓缩描述」，不要项目名。\n\n"
        + "\n".join(lines) + "\n"
    )

    import re

    data = _get_client().chat(
        [{"role": "user", "content": prompt}],
        lite=True,
        max_tokens=2048,
        temperature=0.1,
        enable_thinking=False,
    )
    content = _extract_content(data)
    if content:
        # 解析 "序号. 描述" 格式
        result = [""] * len(repos)
        for line in content.splitlines():
            m = re.match(r"(\d+)\.\s*(.+)", line.strip())
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx < len(repos):
                    result[idx] = m.group(2).strip()[:max_chars]
        filled = sum(1 for r in result if r)
        if filled >= len(repos) * 0.5:
            for i, r in enumerate(result):
                if not r:
                    result[i] = repos[i].get("description", "")[:max_chars]
            logger.info(
                f"LLM 批量浓缩项目简介完成: 成功解析 {filled}/{len(repos)} 条，"
                "未解析项已回退为原描述截断"
            )
            return result

    logger.warning("LLM 批量浓缩失败，回退截断")
    return [r.get("description", "")[:max_chars] for r in repos]
