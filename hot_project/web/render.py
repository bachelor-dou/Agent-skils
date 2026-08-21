"""报告 HTML 输出(服务端渲染)。

只管把算好的事实拼成标签:上期对比、描述覆盖、附栏归属这些**数据**在 `view_model.py`
里算,这里照抄。不认识 HTTP:状态码、缓存头、路由由 `api_server` 决定。结构化榜单走
主从布局、每个字段单独 escape;其余 .md 退回通用 Markdown 渲染,那条路径要额外清洗
原始 HTML。

榜单内容全部来自外部:凡是拼进 HTML 的都过 `escape`,凡是进 href/src 的都过 `_is_safe_url`
协议白名单 —— 这是安全边界,不是代码风格。class 名和 DOM 形状被 report.css / report.js
按名字依赖,改名不会报错,只会静默错版。
"""

from __future__ import annotations

import functools
import hashlib
import logging
import re
from html import escape, unescape
from pathlib import Path

from markdown import Markdown

from .. import config
from ..infra.data_access import reports
from . import view_model as vm

logger = logging.getLogger("hot_project")

_REPORT_TEMPLATE = "report.html"
_EMPTY_TOC = '<p class="toc-empty">当前报告暂无可跳转目录。</p>'


# ══════════════════════════════════════════════════════════════
# 静态资源与模板
# ══════════════════════════════════════════════════════════════


def _asset_path(path_name: str) -> Path:
    """web 目录下的资源路径。越出目录的名字一律当「资源不存在」。

    挡住穿越只要两行,漏了它以后谁把请求里的文件名接进来,就变成任意文件读取。
    """
    root = config.WEB_DIR.resolve()
    path = (root / path_name).resolve()
    if path != root and root not in path.parents:
        raise FileNotFoundError(f"web 资源越界:{path_name}")
    return path


def asset_text(path_name: str) -> str:
    """读 web 目录里的文本资源。读不到就让 OSError 冒出去,由 `api_server` 翻成状态码。"""
    return _asset_path(path_name).read_text(encoding="utf-8")


@functools.lru_cache(maxsize=1)
def _asset_version() -> str:
    """web 目录的指纹,拼在 css/js 的 `?v=` 上,让浏览器只在文件真的改了之后才重新下载。

    一个进程只算一次,运行中改了静态文件就重启服务。子目录(vendor/)不参与。
    """
    digest = hashlib.md5(usedforsecurity=False)
    root = config.WEB_DIR
    for path in sorted(root.glob("*")) if root.is_dir() else []:
        if path.is_file():
            digest.update(f"{path.name}:{path.stat().st_mtime}".encode())
    return digest.hexdigest()[:10]


def page(path_name: str, replacements: dict[str, str]) -> str:
    """占位符模板 → 最终 HTML,`__ASSET_VER__` 总是自动补上(调用方给了则用调用方的)。

    不改调用方传进来的字典 —— 复用同一个字典时会带上上一次渲染的残留值。
    """
    document = asset_text(path_name)
    for placeholder, value in {"__ASSET_VER__": _asset_version(), **replacements}.items():
        document = document.replace(placeholder, value)
    return document


# ══════════════════════════════════════════════════════════════
# 链接白名单 —— 报告里的 URL 一律先过这里
# ══════════════════════════════════════════════════════════════

_SAFE_SCHEMES = frozenset({"http", "https", "mailto"})
_RELATIVE_PREFIXES = ("#", "/", "./", "../", "//")
_SCHEME = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*):")
_CTRL = re.compile(r"[\x00-\x20]+")
_URL_ATTR = re.compile(
    r'(?P<attr>\b(?:href|src))\s*=\s*'
    r'(?:(?P<quote>["\'])(?P<quoted>.*?)(?P=quote)|(?P<bare>[^\s>]+))',
    re.IGNORECASE,
)


def _is_safe_url(url: str) -> bool:
    """只放行 http / https / mailto 与站内相对链接,`javascript:` `data:` 一律拦掉。

    判之前先 unescape 再抹控制字符:`java&#09;script:alert(1)` 浏览器照样执行,只看原串会漏
    (抹掉只为判断,输出仍是原串)。取不出协议的一律放行。
    """
    normalized = unescape(url or "").strip()
    if not normalized:
        return False
    compact = _CTRL.sub("", normalized)
    if normalized.startswith(_RELATIVE_PREFIXES) or compact.startswith(_RELATIVE_PREFIXES):
        return True
    scheme = _SCHEME.match(compact)
    return scheme.group(1).lower() in _SAFE_SCHEMES if scheme else True


def _safe_href(url: str) -> str:
    return url if _is_safe_url(url) else "#"


def _sanitize_urls(html_text: str) -> str:
    """给渲染完的 HTML 再过一遍链接白名单。

    必须放在渲染**之后**:`[x](javascript:alert(1))` 在原文里长得像普通链接,变成 href 才认得出来。
    """

    def replace(match: re.Match[str]) -> str:
        quote = match.group("quote")
        value = match.group("quoted") if quote else match.group("bare")
        if _is_safe_url(value):
            return match.group(0)
        fallback = "#" if match.group("attr").lower() == "href" else ""
        return f'{match.group("attr")}="{fallback}"'

    return _URL_ATTR.sub(replace, html_text)


# ══════════════════════════════════════════════════════════════
# 文本零件
# ══════════════════════════════════════════════════════════════


def _stat(label: str, value: str, kind: str = "") -> str:
    class_name = f"repo-stat repo-stat--{kind}" if kind else "repo-stat"
    return (
        f'<div class="{class_name}">'
        '<div class="repo-stat__body">'
        f'<span class="repo-stat__label">{escape(label)}</span>'
        f'<strong class="repo-stat__value">{escape(value)}</strong>'
        '</div>'
        '</div>'
    )


# ══════════════════════════════════════════════════════════════
# 主从布局
# ══════════════════════════════════════════════════════════════

# 常见主语言的标识色(侧栏语言圆点),未收录语言回退灰色
_LANG_COLORS = {
    "python": "#3572A5", "typescript": "#3178c6", "javascript": "#f1e05a",
    "go": "#00ADD8", "rust": "#dea584", "c++": "#f34b7d", "c": "#555555",
    "java": "#b07219", "kotlin": "#A97BFF", "swift": "#F05138", "ruby": "#701516",
    "c#": "#178600", "html": "#e34c26", "css": "#563d7c", "shell": "#89e051",
    "jupyter notebook": "#DA5B0B", "dart": "#00B4AB", "php": "#4F5D95",
    "zig": "#ec915c", "lua": "#000080", "vue": "#41b883", "svelte": "#ff3e00",
}


# 附栏条目的字段是生成端按顺序写死的,这里按名字上色,认不出的当增长口径处理
_TREND_STAT_KINDS = {"总 Star": "star", "主语言": "language", "创建时间": "created"}


def _trend_card(entry: reports.Entry) -> str:
    """没上榜的附栏条目 → 一张完整卡片。复用正文的 stat / panel 样式,不另起一套。

    主题和介绍段落的事实来自 view_model(和正文条目同一份切分、同一层 DB 覆盖),
    这里只管 HTML 化。
    """
    stats = "".join(
        _stat(label, value,
              _TREND_STAT_KINDS.get(label, "growth" if "新增" in label else ""))
        for label, value in entry.metadata.items() if label != "主题标签"
    )
    topics = vm.entry_topics(entry.metadata)
    topics_html = (
        f'<div class="repo-detail__topics">'
        + "".join(f'<span class="repo-topic">{escape(t)}</span>' for t in topics[:6])
        + '</div>'
    ) if topics else ""
    sections = "".join(
        f'<section class="repo-panel"><h3>{escape(s["title"])}</h3>'
        + "".join(f"<p>{escape(p)}</p>" for p in s["paragraphs"])
        + '</section>'
        for s in vm.card_sections(entry)
    )
    link = _safe_href(entry.link or f"https://github.com/{entry.repo}")
    return (
        '<div class="trend-card">'
        '<div class="trend-card__head">'
        f'<span class="trend-row__no">T{entry.rank}</span>'
        f'<h3 class="trend-card__name">'
        f'<a href="{escape(link)}" target="_blank" rel="noreferrer">{escape(entry.repo)}</a></h3>'
        '<span class="trend-card__flag">未进本期榜单</span>'
        '</div>'
        f'<div class="repo-detail__stats">{stats}</div>'
        f'{topics_html}'
        f'<div class="repo-detail__grid">{sections}</div>'
        '</div>'
    )


def _trend_html(view: vm.TrendView) -> tuple[str, str, str]:
    """一个周期的附栏 → (面板, 侧栏入口, 头部 chip)。

    每个周期单独成一个面板,也不把附栏条目混进正文面板列表:`report.js` 是按下标把面板
    和侧栏项配对的,多塞或少塞一个就整页错位。已上榜的那些只渲染成一行锚点跳回正文,
    和 Markdown 里一致。
    """
    rows: list[str] = []
    for row in view.rows:
        if row.card:
            rows.append(f'<li class="trend-row trend-row--full">{_trend_card(row.entry)}</li>')
            continue
        target = (
            f'<a class="trend-row__hit" href="#{vm.anchor(row.rank, row.entry.repo)}">榜内 #{row.rank}</a>'
            if row.rank else '<span class="trend-row__hit">已入选本期榜单</span>'
        )
        rows.append(
            '<li class="trend-row">'
            f'<span class="trend-row__no">T{row.entry.rank}</span>'
            f'<span class="trend-row__name">{escape(row.entry.repo)}</span>'
            f'{target}</li>'
        )

    label, anchor, total, hits = view.label, view.anchor, view.total, view.hits
    panel = (
        f'<section class="repo-detail trend-panel" id="{anchor}">'
        '<header class="repo-detail__head">'
        '<span class="repo-detail__rank">附</span>'
        f'<h2>GitHub Trending {escape(label)}对照</h2>'
        '</header>'
        f'<p class="trend-note">共 {total} 个 · 已在本期榜单 {hits} 个 · '
        f'附栏补全 {total - hits} 个。已上榜的点一下跳回正文。</p>'
        f'<ol class="trend-list">{"".join(rows)}</ol>'
        '</section>'
    )
    nav = (
        f'<a class="repo-nav__item repo-nav__item--trend" href="#{anchor}" '
        f'data-panel="{anchor}" data-search="trending {escape(label)} 附">'
        '<span class="repo-nav__rank">附</span>'
        '<span class="repo-nav__body">'
        f'<span class="repo-nav__name">Trending {escape(label)}对照</span>'
        f'<span class="repo-nav__meta"><span class="repo-nav__trend">{total} 个</span></span>'
        '</span></a>'
    )
    chip = (
        '<span class="hero__chip hero__chip--trend">'
        f'Trending {escape(label)} {total}: 榜内 {hits} · 补全 {total - hits}'
        '</span>'
    )
    return panel, nav, chip


def _structured_html(
    views: list[vm.EntryView], trend_blocks: list[tuple[str, str]]
) -> tuple[str, str]:
    """返回 (详情面板集合, 侧栏项目列表)。

    两块 HTML 出自同一份 `EntryView`,拆开也不会算出不一样的结论 —— 事实都在
    view-model 里算好了,这里只挑格子、拼标签、escape。
    """
    article_parts: list[str] = []
    nav_items: list[str] = []

    for view in views:
        repo_link = _safe_href(view.link)
        readme_link = _safe_href(f"{repo_link}#readme") if repo_link != "#" else "#"

        stat_items: list[str] = []
        if view.star:
            stat_items.append(_stat("总 Star", view.star, "star"))
        if view.growth_label and view.growth_value:
            stat_items.append(_stat(view.growth_label, view.growth_value, "growth"))
        if view.language:
            stat_items.append(_stat("主语言", view.language, "language"))
        created_extra = (
            f' <span class="repo-stat__tag repo-stat__tag--new" title="{escape(view.status)}">NEW</span>'
            if view.status else ""
        )
        stat_items.append(
            '<div class="repo-stat repo-stat--created">'
            '<div class="repo-stat__body">'
            '<span class="repo-stat__label">创建时间</span>'
            f'<strong class="repo-stat__value">{escape(view.created)}{created_extra}</strong>'
            '</div>'
            '</div>'
        )

        section_items = [
            f'<section class="repo-panel" data-section="{escape(s["title"])}">'
            f'<h3 id="{view.anchor}-{vm.slug(s["title"])}">{escape(s["title"])}</h3>'
            + "".join(f"<p>{escape(p)}</p>" for p in s["paragraphs"])
            + '</section>'
            for s in view.sections
        ]

        topics_html = ""
        if view.topics:
            tags_html = "".join(f'<span class="repo-topic">{escape(t)}</span>'
                                for t in view.topics[:6])
            topics_html = f'<div class="repo-detail__topics">{tags_html}</div>'

        actions_html = (
            '<div class="repo-detail__actions">'
            f'<a class="repo-action" href="{escape(repo_link)}" target="_blank" rel="noreferrer">打开仓库 ↗</a>'
            f'<a class="repo-action repo-action--ghost" href="{escape(readme_link)}" target="_blank" rel="noreferrer">查看 README</a>'
            f'<button type="button" class="repo-action repo-action--ghost repo-trend-btn" data-repo="{escape(view.repo)}">📈 star 走势</button>'
            '</div>'
            '<div class="repo-trend" hidden></div>'
        )

        delta_html = nav_delta = ""
        if view.delta:
            way, word, arrow = ("up", "上升", "↑") if view.delta > 0 else ("down", "下降", "↓")
            magnitude = abs(view.delta)
            delta_html = (
                f'<span class="repo-detail__delta repo-detail__delta--{way}" '
                f'title="较上期{word} {magnitude} 名">{arrow}{magnitude}</span>'
            )
            nav_delta = (
                f'<span class="repo-nav__delta repo-nav__delta--{way}">{arrow}{magnitude}</span>'
            )
        fresh_badge = '<span class="repo-detail__fresh" title="上期报告中没有的项目">上新</span>' if view.is_fresh else ""
        new_badge = '<span class="repo-detail__new" title="新项目">NEW</span>' if view.status else ""
        fresh_attr = ' data-fresh="1"' if view.is_fresh else ""
        trend_badge = (
            f'<span class="repo-detail__trend" title="同时在 GitHub Trending '
            f'{escape("、".join(view.trend_labels))}上">TRENDING</span>'
            if view.trend_labels else ""
        )

        article_parts.append(
            f'<section class="repo-detail" id="{view.anchor}" data-repo="{escape(view.repo)}" data-rank="{view.rank}"{fresh_attr}>'
            '<header class="repo-detail__head">'
            f'<span class="repo-detail__rank">#{view.rank}</span>'
            f'{delta_html}'
            f'<h2>{escape(view.repo)}</h2>'
            f'{fresh_badge}'
            f'{new_badge}'
            f'{trend_badge}'
            '</header>'
            f'<div class="repo-detail__stats">{"".join(stat_items)}</div>'
            f'{topics_html}'
            f'<div class="repo-detail__grid">{"".join(section_items)}</div>'
            f'{actions_html}'
            '</section>'
        )

        lang_dot = ""
        if view.language:
            color = _LANG_COLORS.get(view.language.strip().lower(), "#8b94a7")
            lang_dot = f'<span class="repo-nav__lang"><i style="background:{color}"></i>{escape(view.language)}</span>'
        growth_chip = f'<span class="repo-nav__growth">{escape(view.growth_value)}</span>' if view.growth_value else ""
        new_chip = '<span class="repo-nav__new">NEW</span>' if view.status else ""
        fresh_chip = '<span class="repo-nav__fresh">上新</span>' if view.is_fresh else ""
        trend_chip = '<span class="repo-nav__trend">TRENDING</span>' if view.trend_labels else ""
        hot_class = " repo-nav__item--hot" if view.hot else ""
        nav_items.append(
            f'<a class="repo-nav__item{hot_class}" href="#{view.anchor}" data-panel="{view.anchor}" '
            f'data-repo="{escape(view.repo)}" data-search="{escape(view.search_blob)}"{fresh_attr}>'
            f'<span class="repo-nav__rank">{view.rank}</span>'
            '<span class="repo-nav__body">'
            f'<span class="repo-nav__name">{escape(view.repo)}</span>'
            f'<span class="repo-nav__meta">{fresh_chip}{new_chip}{trend_chip}'
            f'{growth_chip}{nav_delta}{lang_dot}</span>'
            '</span>'
            '</a>'
        )

    # 面板和侧栏项在 report.js 里是按下标配对的:附栏这两块只能一起加,也只能加在最后。
    for panel, nav in trend_blocks:
        article_parts.append(panel)
        nav_items.append(nav)

    toc_html = (
        f'<nav class="repo-nav" id="repo-nav">{"".join(nav_items)}</nav>'
        if nav_items else _EMPTY_TOC
    )
    article_html = "".join(article_parts) if article_parts else '<p>当前报告暂无项目内容。</p>'
    return article_html, toc_html


def _summary_chips(summary: str, extra_chips: list[str] | None = None) -> str:
    """把「共 N 个项目 | 窗口: 7 天 | …」形式的摘要拆成头部信息条。

    只有一段又没附加 chip 时按整句排版(单个 chip 看着像被截断的碎片)。
    `extra_chips` 是已渲染好的 HTML,不再过 escape。
    """
    text = (summary or "").strip()
    extra = "".join(extra_chips or [])
    if not text:
        return f'<div class="hero__chips">{extra}</div>' if extra else ""
    parts = [p.strip() for p in text.split("|") if p.strip()]
    if len(parts) <= 1 and not extra:
        return f'<p class="hero__summary">{escape(text)}</p>'
    chips = "".join(f'<span class="hero__chip">{escape(p)}</span>' for p in parts)
    return f'<div class="hero__chips">{chips}{extra}</div>'


# ══════════════════════════════════════════════════════════════
# 通用 Markdown 路径
# ══════════════════════════════════════════════════════════════
# ponytail: 黑名单清洗,不是白名单。撑得住已知向量(见 test_web.py 的探针集),但下一个
# 没想到的形态就是下一个洞。真正的修法是换 nh3/bleach 做白名单,代价是多一个依赖。

_RAW_TAG_PAIR = re.compile(
    r"<(script|iframe|object|embed|form|input|style)[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_RAW_TAG = re.compile(
    r"<(script|iframe|object|embed|form|input|style)[^>]*/?\s*>", re.IGNORECASE
)
_EVENT_ATTR = re.compile(r"\bon\w+\s*=", re.IGNORECASE)

_MAX_CLEAN_PASSES = 8


def _clean_raw_html(text: str) -> str:
    for _ in range(_MAX_CLEAN_PASSES):
        cleaned = _EVENT_ATTR.sub("", _RAW_TAG.sub("", _RAW_TAG_PAIR.sub("", text)))
        if cleaned == text:
            return cleaned
        text = cleaned
    return escape(text)     # 收敛不了就整篇转义:宁可显示成源码,也不把它交给浏览器


def _plain_page(name: str, markdown: str) -> str:
    lines = markdown.splitlines()
    title = next((ln[2:].strip() for ln in lines if ln.startswith("# ")), name)
    summary = next((ln[1:].strip() for ln in lines if ln.startswith(">")), "")

    text = _clean_raw_html(markdown)

    md = Markdown(extensions=["extra", "sane_lists", "toc", "nl2br"], output_format="html5")
    article_html = _sanitize_urls(md.convert(text))
    toc_html = _sanitize_urls(getattr(md, "toc", ""))

    return page(
        _REPORT_TEMPLATE,
        {
            "__REPORT_NAME__": escape(name),
            "__REPORT_TITLE__": escape(title),
            "__REPORT_SUMMARY_HTML__": _summary_chips(
                summary or "这是一份由服务器根据 Markdown 报告渲染出的可读网页。"
            ),
            "__REPORT_TOC_HTML__": toc_html if toc_html.strip() else _EMPTY_TOC,
            "__REPORT_ARTICLE_HTML__": article_html,
        },
    )


def _structured_page(name: str, report: reports.Report) -> str:
    diff = vm.diff_of(name, report)
    trends = vm.trending_views(report)
    views = vm.entry_views(report, diff, trends)

    rendered_trends = [_trend_html(t) for t in trends]
    article_html, toc_html = _structured_html(
        views, [(panel, nav) for panel, nav, _ in rendered_trends])

    extra_chips: list[str] = []
    if diff:
        extra_chips.append(
            '<span class="hero__chip hero__chip--fresh">'
            f'较上期 {escape(diff.prev_name.rsplit(".", 1)[0])}: '
            f'上新 {diff.added} · 移出 {diff.removed}'
            '</span>'
        )
    extra_chips.extend(chip for _, _, chip in rendered_trends)

    return page(
        _REPORT_TEMPLATE,
        {
            "__REPORT_NAME__": escape(name),
            "__REPORT_TITLE__": escape(report.title or name),
            "__REPORT_SUMMARY_HTML__": _summary_chips(report.summary, extra_chips),
            "__REPORT_TOC_HTML__": toc_html,
            "__REPORT_ARTICLE_HTML__": article_html,
        },
    )


def report_html(name: str, markdown: str) -> str:
    """一份报告的完整 HTML 页面。

    `name` 只用于展示和找上一期,不用来读文件 —— 原文由调用方读好传进来(路径穿越挡在
    `reports.resolve_name`)。
    """
    report = reports.parse(markdown)
    return _structured_page(name, report) if report else _plain_page(name, markdown)
