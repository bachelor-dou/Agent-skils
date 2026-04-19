"""
GitHub Trending 爬虫（执行层 · 数据源组件）
=============================================
爬取 GitHub Trending 页面，获取当前热门仓库列表。

架构定位：
  执行层独立数据源，由 agent_tools.tool_fetch_trending() 调用。
  不依赖项目内部模块，仅使用 requests + 正则解析 HTML。

两种使用路径：
  路径 1 — 直接展示：用户问"Trending 上有哪些项目"，直接返回列表
  路径 2 — 候选补充：将 Trending 仓库加入候选池，走正常评分流程

支持参数：
    since:           daily / weekly / monthly

说明：
    仓库主语言会从 Trending 页面中解析出来并保留在返回结果里，
    但不再作为用户可指定的筛选条件。
"""

import logging
import re

import requests

logger = logging.getLogger("discover_hot")

TRENDING_PERIODS = ("daily", "weekly", "monthly")
DEFAULT_TRENDING_SINCE = "weekly"

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def fetch_trending(since: str = DEFAULT_TRENDING_SINCE) -> list[dict]:
    """
    爬取 GitHub Trending 仓库列表。

    Args:
        since:           时间范围 ("daily" | "weekly" | "monthly")，默认 weekly

    Returns:
        [{"full_name": "owner/repo",
          "star": int,
          "forks": int,
          "stars_today": int,
          "description": str,
          "language": str}, ...]
    """
    normalized_since = since if since in TRENDING_PERIODS else DEFAULT_TRENDING_SINCE

    url = "https://github.com/trending"
    params: dict[str, str] = {}
    if normalized_since:
        params["since"] = normalized_since

    try:
        resp = requests.get(
            url,
            params=params,
            headers={"User-Agent": _USER_AGENT},
            timeout=120,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Trending 页面请求失败: {e}")
        return []

    return _parse_trending_html(resp.text, normalized_since)


def fetch_trending_all() -> list[dict]:
    """
    抓取 daily / weekly / monthly 三个时间维度的 Trending，按仓库去重汇总。

    主要用于综合/新项目工作流中的候选补充阶段。
    """
    merged: dict[str, dict] = {}

    for since in TRENDING_PERIODS:
        repos = fetch_trending(since=since)
        for repo in repos:
            full_name = repo["full_name"]
            existing = merged.get(full_name)
            if existing is None:
                merged[full_name] = {
                    **repo,
                    "periods": [since],
                    "stars_by_period": {since: repo.get("stars_today", 0)},
                }
                continue

            if since not in existing["periods"]:
                existing["periods"].append(since)
                existing["periods"].sort(key=TRENDING_PERIODS.index)

            stars_by_period = existing.setdefault("stars_by_period", {})
            stars_by_period[since] = repo.get("stars_today", 0)
            existing["stars_today"] = max(existing.get("stars_today", 0), repo.get("stars_today", 0))
            existing["star"] = max(existing.get("star", 0), repo.get("star", 0))
            existing["forks"] = max(existing.get("forks", 0), repo.get("forks", 0))
            if not existing.get("description") and repo.get("description"):
                existing["description"] = repo["description"]
            if not existing.get("language") and repo.get("language"):
                existing["language"] = repo["language"]

    repos = list(merged.values())
    repos.sort(
        key=lambda item: (
            len(item.get("periods", [])),
            item.get("star", 0),
            item.get("stars_today", 0),
        ),
        reverse=True,
    )
    logger.info(
        "Trending 汇总完成: %d 个去重仓库 (periods=%s)",
        len(repos),
        ",".join(TRENDING_PERIODS),
    )
    return repos


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
        parts = full_name.split("/", 1)
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
