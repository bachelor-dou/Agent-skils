"""
LLM 调用模块
=============
调用 LLM（兼容 OpenAI /v1/chat/completions 格式）为项目生成详细描述。

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
    调用 LLM 生成项目详细描述（200-400 字中文）。

    Args:
        repo_name: "owner/repo"
        repo_info: DB 中的仓库信息字典（含 short_desc / language / topics / readme_url）
        html_url:  项目 GitHub 页面 URL

    Returns:
        LLM 生成的描述文本；失败 3 次后返回空字符串。
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
        info_parts.append(f"README: {readme_url}")

    prompt = (
        f"请对以下 GitHub 开源项目进行详细介绍。\n"
        f"要求：\n"
        f"1. 综合利用下方提供的项目信息（官方简介、编程语言、标签、README 链接等），"
        f"同时结合你对该项目的已有知识进行补充\n"
        f"2. 重点说明：核心功能与解决的问题、关键技术特点、典型应用场景\n"
        f"3. 200-400 字，使用中文\n"
        f"4. 不要编造不确定的信息\n\n"
        + "\n".join(info_parts) + "\n"
    )

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": 1024,
    }

    for attempt in range(3):
        try:
            resp = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                content = (
                    resp.json()
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
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
