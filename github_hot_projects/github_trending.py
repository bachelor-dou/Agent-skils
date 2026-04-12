"""
GitHub Trending 爬虫
====================
爬取 GitHub Trending 页面，获取当前热门仓库列表。

两种使用路径：
  路径 1 — 直接展示：用户问"Trending 上有哪些项目"，直接返回列表
  路径 2 — 候选补充：将 Trending 仓库加入候选池，走正常评分流程

支持参数：
  since:           daily / weekly / monthly
  language:        编程语言筛选（如 python, javascript）
  spoken_language: 自然语言筛选（如 zh, en）
"""

import logging
import re

import requests

logger = logging.getLogger("discover_hot")

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def fetch_trending(
    since: str = "daily",
    language: str = "",
    spoken_language: str = "",
) -> list[dict]:
    """
    爬取 GitHub Trending 仓库列表。

    Args:
        since:           时间范围 ("daily" | "weekly" | "monthly")
        language:        编程语言筛选，如 "python"，"" 表示全部
        spoken_language: 自然语言代码，如 "zh"，"" 表示全部

    Returns:
        [{"full_name": "owner/repo",
          "star": int,
          "forks": int,
          "stars_today": int,
          "description": str,
          "language": str}, ...]
    """
    url = "https://github.com/trending"
    params: dict[str, str] = {}
    if since:
        params["since"] = since
    if language:
        params["language"] = language.lower()
    if spoken_language:
        params["spoken_language_code"] = spoken_language

    try:
        resp = requests.get(
            url,
            params=params,
            headers={"User-Agent": _USER_AGENT},
            timeout=30,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Trending 页面请求失败: {e}")
        return []

    return _parse_trending_html(resp.text, since)


def _parse_trending_html(html: str, since: str = "daily") -> list[dict]:
    """
    用正则解析 Trending HTML 页面。

    GitHub Trending 页面结构（2024-2026 稳定）：
      <article class="Box-row">
        <h2><a href="/owner/repo">...</a></h2>
        <p class="...">description</p>
        <span itemprop="programmingLanguage">Python</span>
        <a href="/owner/repo/stargazers">star N</a>
        <a href="/owner/repo/forks">fork N</a>
        <span>N stars today/this week/this month</span>
      </article>
    """
    repos: list[dict] = []

    # 按 <article> 分段
    articles = re.findall(
        r'<article\b[^>]*class="[^"]*Box-row[^"]*"[^>]*>(.*?)</article>',
        html,
        re.DOTALL,
    )

    for article in articles:
        # 仓库名：<h2> 内 <a href="/owner/repo">
        name_match = re.search(r'<h2[^>]*>.*?<a\s[^>]*href="/([^"]+)"', article, re.DOTALL)
        if not name_match:
            continue
        full_name = name_match.group(1).strip().strip("/")
        # 确保是 owner/repo 格式（排除 /stargazers 等子路径）
        parts = full_name.split("/")
        if len(parts) != 2:
            continue

        # 描述
        desc_match = re.search(r'<p[^>]*>(.*?)</p>', article, re.DOTALL)
        description = ""
        if desc_match:
            description = re.sub(r'<[^>]+>', '', desc_match.group(1)).strip()

        # 语言
        lang_match = re.search(
            r'itemprop="programmingLanguage"[^>]*>(.*?)<', article
        )
        language = lang_match.group(1).strip() if lang_match else ""

        # star 总数：stargazers 链接内 </svg> 后面的数字
        star = 0
        stargazer_section = re.search(r'stargazers.*?</a>', article, re.DOTALL)
        if stargazer_section:
            star_num = re.search(r'</svg>\s*([\d,]+)', stargazer_section.group(0))
            if star_num:
                star = _parse_number(star_num.group(1))

        # fork 总数：/forks 链接内 </svg> 后面的数字
        forks = 0
        fork_section = re.search(r'/forks.*?</a>', article, re.DOTALL)
        if fork_section:
            fork_num = re.search(r'</svg>\s*([\d,]+)', fork_section.group(0))
            if fork_num:
                forks = _parse_number(fork_num.group(1))

        # 今日/本周/本月 star 增长
        period_label = {"daily": "today", "weekly": "this week", "monthly": "this month"}
        label = period_label.get(since, "today")
        today_match = re.search(
            r'([\d,]+)\s+stars?\s+' + re.escape(label), article, re.IGNORECASE
        )
        stars_today = _parse_number(today_match.group(1)) if today_match else 0

        repos.append({
            "full_name": full_name,
            "star": star,
            "forks": forks,
            "stars_today": stars_today,
            "description": description[:500],
            "language": language,
            "since": since,
        })

    logger.info(f"Trending 解析完成: {len(repos)} 个仓库 (since={since})")
    return repos


def _parse_number(s: str) -> int:
    """解析带逗号的数字字符串，如 '12,345' → 12345。"""
    return int(s.replace(",", "").strip()) if s else 0
