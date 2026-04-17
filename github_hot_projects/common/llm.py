"""
LLM 调用模块
=============
调用 LLM（兼容 OpenAI /v1/chat/completions 格式）为项目生成结构化介绍。

仅对最终 Top N 中 desc 为空的项目调用，
传入仓库元信息（description、language、topics、readme_url）供 LLM 综合总结。
"""

import logging
import time

import requests

from .config import LLM_API_KEY, LLM_API_URL, LLM_MODEL

logger = logging.getLogger("discover_hot")


def call_llm_describe(repo_name: str, repo_info: dict, html_url: str) -> str:
    """
    调用 LLM 生成三段式项目介绍。

    Args:
        repo_name: "owner/repo"
        repo_info: DB 中的仓库信息字典（含 short_desc / language / topics / readme_url）
        html_url:  项目 GitHub 页面 URL

    Returns:
        LLM 生成的三段式描述文本；失败 3 次后返回空字符串。
    """
    if not LLM_API_URL or not LLM_API_KEY:
        logger.warning("LLM 未配置，跳过描述生成。")
        return ""

    # 构建信息块供 LLM 参考
    info_parts = [f"项目名称: {repo_name}", f"项目地址: {html_url}"]
    if short_desc := repo_info.get("short_desc", ""):
        info_parts.append(f"官方简介: {short_desc}")
    if language := repo_info.get("language", ""):
        info_parts.append(f"主要语言: {language}")
    if topics := repo_info.get("topics", []):
        info_parts.append(f"标签: {', '.join(topics)}")
    if readme_url := repo_info.get("readme_url", ""):
        info_parts.append(f"README链接（仅供标识，不能视为已读取内容）: {readme_url}")

    prompt = (
        f"请基于以下已提供信息，用中文总结这个 GitHub 开源项目。\n"
        f"输出要求：\n"
        f"1. 只能基于下方明确提供的信息，不要把项目地址或 README 链接当作已读取内容。\n"
        f"2. 不要补充未在输入中出现、且无法确认的外部知识；信息不足时使用保守表述。\n"
        f"3. 必须严格输出以下三个字段，字段名保持原样：\n"
        f"项目定位与用途：...\n"
        f"解决的问题：...\n"
        f"使用场景：...\n"
        f"4. 每个字段建议 70-160 字，总长度控制在 260-520 字。\n"
        f"5. 不要使用列表、不要加 Markdown 标题、不要输出字段以外的说明。\n\n"
        + "\n".join(info_parts) + "\n"
    )

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 1536,
        "enable_thinking": False,
    }

    for attempt in range(3):
        try:
            resp = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=300)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if not choices:
                        logger.warning(f"LLM 返回无 choices: {repo_name}")
                        continue
                    content = choices[0].get("message", {}).get("content", "")
                except (ValueError, KeyError, IndexError) as e:
                    logger.warning(f"LLM 响应解析失败: {repo_name}, {e}")
                    continue
                if content.strip():
                    return content.strip()
            else:
                logger.warning(
                    f"LLM 调用失败: {repo_name}, status={resp.status_code}, "
                    f"attempt={attempt + 1}/3"
                )
        except requests.RequestException as e:
            logger.error(f"LLM 请求异常: {repo_name}, attempt={attempt + 1}/3, {e}")
        time.sleep(2)

    logger.warning(f"LLM 3 次重试均失败，跳过描述: {repo_name}")
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
    if not LLM_API_URL or not LLM_API_KEY:
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

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 2048,
        "enable_thinking": False,
    }

    import re
    for attempt in range(2):
        try:
            resp = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    if content.strip():
                        # 解析 "序号. 描述" 格式
                        result = [""] * len(repos)
                        for line in content.strip().splitlines():
                            m = re.match(r"(\d+)\.\s*(.+)", line.strip())
                            if m:
                                idx = int(m.group(1)) - 1
                                if 0 <= idx < len(repos):
                                    result[idx] = m.group(2).strip()[:max_chars]
                        # 检查是否大部分都解析成功
                        filled = sum(1 for r in result if r)
                        if filled >= len(repos) * 0.5:
                            # 补全未解析的
                            for i, r in enumerate(result):
                                if not r:
                                    result[i] = repos[i].get("description", "")[:max_chars]
                            logger.info(
                                f"LLM 批量浓缩项目简介完成: 成功解析 {filled}/{len(repos)} 条，"
                                "未解析项已回退为原描述截断"
                            )
                            return result
            logger.warning(f"LLM 批量浓缩失败: status={resp.status_code}, attempt={attempt+1}")
        except requests.RequestException as e:
            logger.error(f"LLM 批量浓缩请求异常: attempt={attempt+1}, {e}")
        time.sleep(1)

    logger.warning("LLM 批量浓缩失败，回退截断")
    return [r.get("description", "")[:max_chars] for r in repos]
