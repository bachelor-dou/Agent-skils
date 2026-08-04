"""GitHub Trending —— 抓页面并解析成仓库列表。

唯一不吃 token 的采集来源(网页,不占 Search 限额),能和另两个收集阶段完全并行,并补上
Search 按 star 排序永远排不到的盲区:总量不够门槛但这几天涨得凶的新项目。

正则解析的代价是页面改版会静默解析出 0 条,所以 `parse` 一并返回文章总数 —— 拿它和结果数
一比才分得清「今天榜单短」和「解析器坏了」,这两件事的处置方式完全相反。
"""

from __future__ import annotations

import logging
import re
from typing import Any, NamedTuple

import httpx

from ...infra.exceptions import RetryableError

logger = logging.getLogger("hot_project")

URL = "https://github.com/trending"
PERIODS = ("daily", "weekly", "monthly")
DEFAULT_PERIOD = "weekly"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
}

_PERIOD_LABEL = {"daily": "today", "weekly": "this week", "monthly": "this month"}

_ARTICLE = re.compile(
    r'<article\b[^>]*class="[^"]*Box-row[^"]*"[^>]*>(.*?)</article>', re.DOTALL
)
_NAME = re.compile(r'<h2[^>]*>.*?<a\s[^>]*href="/([^"]+)"', re.DOTALL)
_DESC = re.compile(r"<p[^>]*>(.*?)</p>", re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_LANG = re.compile(r'itemprop="programmingLanguage"[^>]*>(.*?)<')
_STARGAZERS = re.compile(r"stargazers.*?</a>", re.DOTALL)
_FORKS = re.compile(r"/forks.*?</a>", re.DOTALL)
_COUNT = re.compile(r"</svg>\s*([\d,]+)")


class Trending(NamedTuple):
    repos: list[dict[str, Any]]
    articles: int          # 页面上有多少个条目 —— 和 len(repos) 对不上就是解析器出问题了

    @property
    def looks_broken(self) -> bool:
        """解析成功率低于四分之三 = 页面结构变了,不是榜单变短了。"""
        return self.articles > 0 and len(self.repos) < self.articles * 0.75


def _number(text: str) -> int:
    try:
        return int(text.replace(",", "").strip())
    except (ValueError, AttributeError):
        return 0


def _count_in(section: re.Pattern[str], article: str) -> int:
    found = section.search(article)
    if not found:
        return 0
    number = _COUNT.search(found.group(0))
    return _number(number.group(1)) if number else 0


def parse(html: str, period: str = DEFAULT_PERIOD) -> Trending:
    """把 Trending 页面拆成仓库列表。纯函数,不联网。"""
    articles = _ARTICLE.findall(html)
    repos: list[dict[str, Any]] = []

    for article in articles:
        name = _NAME.search(article)
        if not name:
            continue
        full_name = name.group(1).strip().strip("/")
        if full_name.count("/") != 1:      # 排除 /owner/repo/stargazers 这类子路径
            continue

        desc = _DESC.search(article)
        lang = _LANG.search(article)
        label = _PERIOD_LABEL.get(period, "today")
        gained = re.search(rf"([\d,]+)\s+stars?\s+{re.escape(label)}", article, re.IGNORECASE)

        repos.append({
            "full_name": full_name,
            "star": _count_in(_STARGAZERS, article),
            "forks": _count_in(_FORKS, article),
            "stars_today": _number(gained.group(1)) if gained else 0,
            "description": (_TAG.sub("", desc.group(1)).strip()[:500] if desc else ""),
            "language": lang.group(1).strip() if lang else "",
            "since": period,
        })

    return Trending(repos=repos, articles=len(articles))


async def fetch_trending(
    client: httpx.AsyncClient, period: str = DEFAULT_PERIOD
) -> list[dict[str, Any]]:
    """抓一个周期的榜单。失败抛 `RetryableError`,由任务池决定重不重试。"""
    try:
        resp = await client.get(URL, params={"since": period}, headers=HEADERS, timeout=30.0)
    except httpx.HTTPError as e:
        raise RetryableError(f"Trending 请求失败:{e}") from e

    if resp.status_code != 200:
        raise RetryableError(f"Trending HTTP {resp.status_code}")

    result = parse(resp.text, period)
    if result.looks_broken:
        logger.warning(
            "Trending(%s) 页面有 %d 个条目却只解析出 %d 个 —— 页面结构可能改了。",
            period, result.articles, len(result.repos),
        )
    return result.repos
