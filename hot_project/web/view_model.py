"""报告页的视图模型:把「报告 + 上期对比 + DB 描述覆盖 + Trending 对照」算成纯数据。

只算事实,不产 HTML:排名升降、上新徽章、覆盖后的分段、附栏归属,都以普通数据结构给出,
「算得对不对」可以直接断言,不必到 HTML 字符串里找片段。拼标签、escape、URL 白名单在
render.py —— 那是输出编码,和这里的事实计算分开演化。
"""

from __future__ import annotations

import functools
import logging
import re
from typing import NamedTuple

from ..infra.data_access import reports, universe
from ..infra.data_access._file_io import StoreReadError
from ..service.describe import LEGACY_SECTIONS, SECTIONS

logger = logging.getLogger("hot_project")

# 侧栏序号淡蓝的分界。60 是八期历史里「周涨 2,000+」个数的稳定区间(51-64,中位 58),
# 61 名往后基本只剩存量大仓的慢涨。纯展示分区,不影响出榜数量。
NAV_HOT_CUTOFF = 60

TREND_ANCHOR = "trending-appendix"

_NON_ALNUM = re.compile(r"[^a-zA-Z0-9]+")
_BLANK_LINE = re.compile(r"\n\s*\n")
TOPIC_SEP = re.compile(r"[，,]")

_NO_DESC = "暂无补充信息，可进入仓库查看 README。"


def slug(text: str) -> str:
    """锚点 id。中文被整段吃掉是有意的:id 要能直接拼进 `href="#..."` 而不必再编码。
    清完为空时兜底成 section —— 空 id 的标题点不到。
    """
    return _NON_ALNUM.sub("-", text.lower()).strip("-") or "section"


def paragraphs(text: str) -> list[str]:
    return [block.strip() for block in _BLANK_LINE.split(text) if block.strip()]


def anchor(rank: int, repo: str) -> str:
    """详情面板的锚点 id。附栏靠它跳回正文,公式只能有一份。"""
    return f"repo-{rank}-{slug(repo)}"


# ══════════════════════════════════════════════════════════════
# 上期对比 —— 蓝色「上新」徽章 + 排名变化
# ══════════════════════════════════════════════════════════════

# 报告名 = 日期 + 类型/区间/方向尾缀:2026-07-07.md / 2026-07-07_NEW.md /
# 2026-07-07_KEY_10d.md / 2026-07-07_KEY_向量库.md(方向可含中文)
_NAME = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})(?P<suffix>.*)\.md$")


class Diff(NamedTuple):
    prev_name: str
    prev_ranks: dict[str, int]
    added: int
    removed: int


@functools.lru_cache(maxsize=16)
def _load_cached(name: str, mtime: float) -> reports.Report | None:
    """解析上一期报告,按 (文件名, mtime) 缓存。

    mtime 必须进 key(还只能当参数传,lru_cache 只认参数):同一天的报告会被重跑覆盖,
    只按文件名缓存会一直拿着旧一期的排名。
    """
    return reports.load(name)


def _mtime(name: str) -> float | None:
    try:
        return (reports.directory() / name).stat().st_mtime
    except OSError:
        return None


def _title_prefix(report: reports.Report) -> str:
    """标题去掉日期部分(「GitHub 热门项目 — 2026-07-01」→「GitHub 热门项目」)。"""
    return (report.title or "").split("—")[0].strip()


def _prev_report(
    name: str, current: reports.Report
) -> tuple[str, reports.Report] | None:
    """同尾缀(同类榜)且日期更早的最近一份报告。

    还要再比一次标题前缀:旧命名规则下关键词榜和综合榜的尾缀会混叠,拿错会算出一整页假「上新」。
    """
    m = _NAME.match(name)
    if not m:
        return None
    day, suffix = m.group("date"), m.group("suffix")

    earlier: list[tuple[str, str]] = []
    for item in reports.listing():
        pm = _NAME.match(item.name)
        if not pm or item.name == name:
            continue
        if pm.group("suffix") != suffix or pm.group("date") >= day:
            continue
        earlier.append((pm.group("date"), item.name))

    for _, fname in sorted(earlier, reverse=True):
        mtime = _mtime(fname)
        if mtime is None:
            continue
        report = _load_cached(fname, mtime)
        if report is None or _title_prefix(report) != _title_prefix(current):
            continue
        return fname, report
    return None


def diff_of(name: str, current: reports.Report) -> Diff | None:
    prev = _prev_report(name, current)
    if prev is None:
        return None
    prev_name, prev_report = prev
    prev_ranks = {e.repo: e.rank for e in prev_report.entries}
    now = {e.repo for e in current.entries}
    return Diff(
        prev_name,
        prev_ranks,
        added=sum(1 for repo in now if repo not in prev_ranks),
        removed=sum(1 for repo in prev_ranks if repo not in now),
    )


# ══════════════════════════════════════════════════════════════
# 描述与 DB 实时同步
# ══════════════════════════════════════════════════════════════

_DESC_HEAD = re.compile(
    r"^(?P<title>"
    + "|".join(re.escape(t) for t in SECTIONS + LEGACY_SECTIONS)
    + r")[:：]\s*(?P<body>.*)$"
)

_desc_cache: tuple[float, dict[str, str]] | None = None


def _desc_index() -> dict[str, str]:
    """{仓库: desc},只含非空的。

    DB 读不出来时退回上一份索引:描述只是覆盖层,不该让整页渲染失败。
    """
    global _desc_cache
    mtime = universe.mtime()
    if mtime is None:
        return {}
    if _desc_cache and _desc_cache[0] == mtime:
        return _desc_cache[1]

    try:
        records = universe.load()
    except StoreReadError as e:
        logger.warning("DB 描述索引加载失败:%s", e)
        return _desc_cache[1] if _desc_cache else {}

    index: dict[str, str] = {}
    for repo, info in records.items():
        desc = (info.get("desc") or "").strip() if isinstance(info, dict) else ""
        if desc:
            index[repo] = desc
    _desc_cache = (mtime, index)
    logger.info("DB 描述索引已刷新:%d 条非空描述", len(index))
    return index


def _desc_sections(desc: str) -> dict[str, str]:
    """DB 里「标题:内容」分段的 desc → {标题: 内容}。

    标题限定为已知小节名:正文里本来就有冒号(「支持 Python:3.10 以上」),放开会把正文切碎。
    """
    sections: dict[str, str] = {}
    current = ""
    for raw in desc.splitlines():
        line = raw.strip()
        if not line:
            continue
        if head := _DESC_HEAD.match(line):
            current = head.group("title")
            sections[current] = head.group("body").strip()
        elif current:
            sections[current] = f"{sections[current]} {line}".strip()
    return sections


def section_payload(desc: str) -> list[dict]:
    """一份 desc → 报告页那四段的 JSON,给「刷新介绍」按钮就地替换用。

    切分必须和整页渲染同一套,否则刷新前后排版会不一致。
    """
    sections = _desc_sections(desc)
    return [{"title": title, "paragraphs": paragraphs(body)}
            for title in SECTIONS if (body := sections.get(title, "").strip())]


def entry_topics(metadata: dict) -> list[str]:
    """「主题标签」元数据 → 主题列表。正文条目和附栏满卡同一份切法。"""
    return [t.strip() for t in TOPIC_SEP.split(metadata.get("主题标签", "")) if t.strip()]


def card_sections(entry: reports.Entry) -> list[dict]:
    """一个条目的介绍段落 → `[{"title", "paragraphs"}]`:DB 描述覆盖 + 分段 + 空兜底。

    正文条目和附栏满卡共用这一份 —— 同一个仓库在两处必须说同一套话,
    刷新过的介绍不能只盖正文不盖附栏。
    """
    overrides = _desc_sections(_desc_index().get(entry.repo, ""))
    return [
        {"title": s["title"],
         "paragraphs": paragraphs(overrides.get(s["title"]) or s["content"]) or [_NO_DESC]}
        for s in entry.sections
    ]


# ══════════════════════════════════════════════════════════════
# Trending 对照附栏
# ══════════════════════════════════════════════════════════════


class TrendRow(NamedTuple):
    entry: reports.Entry
    card: bool              # True = 未进本期榜单,渲染成完整卡片
    rank: int | None        # card=False 时的正文排名(报告里查不到就是 None)


class TrendView(NamedTuple):
    period: str
    label: str              # 周榜 / 月榜
    anchor: str
    rows: list[TrendRow]    # 保持报告里的原始顺序,卡片和锚点行交错
    listed: frozenset[str]  # 已在本期榜单的仓库,正文里的那些要挂 TRENDING 角标

    @property
    def total(self) -> int:
        return len(self.rows)

    @property
    def hits(self) -> int:
        return len(self.listed)


def trending_views(report: reports.Report) -> list[TrendView]:
    """报告里的每段附栏各出一块。顺序跟着报告走(周榜在前、月榜在后)。

    生成端只给「见正文 #N」那一行 = 已上榜,没有任何字段(metadata 为空)。
    """
    ranks = {e.repo: e.rank for e in report.entries}
    views: list[TrendView] = []
    for period, entries in report.trending.items():
        if not entries:
            continue
        label = reports.PERIOD_TEXT.get(period, (period, ""))[0]  # 榜名和生成端同源
        rows: list[TrendRow] = []
        listed: set[str] = set()
        for entry in entries:
            if entry.metadata:
                rows.append(TrendRow(entry, card=True, rank=None))
            else:
                listed.add(entry.repo)
                rows.append(TrendRow(entry, card=False, rank=ranks.get(entry.repo)))
        views.append(TrendView(period, label, f"{TREND_ANCHOR}-{period}",
                               rows, frozenset(listed)))
    return views


# ══════════════════════════════════════════════════════════════
# 正文条目
# ══════════════════════════════════════════════════════════════


class EntryView(NamedTuple):
    """一个正文条目在页面上要用到的全部事实,已算完,渲染层照抄即可。"""

    rank: int
    repo: str
    anchor: str
    link: str                   # 原始链接,输出前由渲染层过 URL 白名单
    star: str                   # 「总 Star」,缺失为空串(渲染层跳过该格)
    growth_label: str
    growth_value: str
    language: str
    created: str
    status: str                 # 「项目状态」,非空即挂 NEW 徽章
    topics: list[str]
    sections: list[dict]        # [{"title", "paragraphs"}],DB 描述已覆盖、空段已兜底
    delta: int                  # 较上期名次变化,正 = 上升;0 = 没变或没得比
    is_fresh: bool              # 上期报告里没有(蓝色「上新」)
    trend_labels: list[str]     # 同时在哪几个 Trending 榜上
    search_blob: str            # 侧栏搜索匹配串
    hot: bool                   # 排名在 NAV_HOT_CUTOFF 之内


def entry_views(report: reports.Report, diff: Diff | None,
                trends: list[TrendView] | None = None) -> list[EntryView]:
    """正文条目 → 视图模型。排名差、上新、描述覆盖、附栏角标,一轮算完。"""
    prev_ranks = diff.prev_ranks if diff else None
    on_trending: dict[str, list[str]] = {}      # 仓库 → 在哪几个榜上,同时上榜就两个都列
    for item in trends or []:
        for repo in item.listed:
            on_trending.setdefault(repo, []).append(item.label)

    views: list[EntryView] = []
    for entry in report.entries:
        metadata = entry.metadata
        topics = entry_topics(metadata)
        language = metadata.get("主语言", "")
        sections = card_sections(entry)

        is_fresh = prev_ranks is not None and entry.repo not in prev_ranks
        delta = prev_ranks[entry.repo] - entry.rank if prev_ranks is not None and not is_fresh else 0
        trend_labels = on_trending.get(entry.repo, [])
        search_blob = " ".join(
            [entry.repo, language] + topics
            + (["trending", *trend_labels] if trend_labels else [])).lower()

        views.append(EntryView(
            rank=entry.rank, repo=entry.repo,
            anchor=anchor(entry.rank, entry.repo),
            link=entry.link or f"https://github.com/{entry.repo}",
            star=metadata.get("总 Star", ""),
            growth_label=next((k for k in metadata if "增长" in k), ""),
            growth_value=reports.growth_of(metadata),
            language=language,
            created=metadata.get("创建时间", "未知"),
            status=metadata.get("项目状态", ""),
            topics=topics, sections=sections,
            delta=delta, is_fresh=is_fresh, trend_labels=trend_labels,
            search_blob=search_blob, hot=entry.rank <= NAV_HOT_CUTOFF,
        ))
    return views
