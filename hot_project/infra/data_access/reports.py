"""报告数据 —— `data/report/*.md` 的列举、读、写,以及 Markdown → 结构化的解析。

    data/report/2026-07-30.md          综合榜
    data/report/2026-07-30_NEW.md      新项目榜
    data/report/2026-07-30_KEY_向量库.md  关键词榜

**文件名是唯一的元数据载体。** 日期从文件名解析(`star_trend` 靠它拼时间序列);模式后缀
和关键词榜的方向名让同一天的多张榜互不覆盖 —— 盖掉之后没有任何痕迹。

解析的是 `service/report.py` 写出来的格式,两边必须同时改;守卫在 `test_report.py`:
生成一份再解析回来,字段对不上就红。
"""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path
from typing import Any, NamedTuple

from ... import config
from ...common.timeutil import parse_day
from . import _file_io

logger = logging.getLogger("hot_project")

_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_LATEST_WORDS = frozenset({"最新", "latest", "last", "newest"})

_UNSAFE = re.compile(r'[\s/\\:*?"<>|.]+')


class Listed(NamedTuple):
    name: str           # 文件名,如 2026-07-30_NEW.md
    title: str          # 报告里的一级标题
    day: date | None    # 从文件名解析的日期


def directory() -> Path:
    return config.REPORT_DIR


def safe_slug(text: str, limit: int = 6) -> str:
    """用户给的方向名 → 能安全放进文件名的片段。清理后为空则返回空串。"""
    return _UNSAFE.sub("", (text or "").strip())[:limit]


def day_of(filename: str) -> date | None:
    m = _DATE.search(filename)
    return parse_day(m.group(1)) if m else None


def _title_of(path: Path) -> str:
    """只读到一级标题就停 —— 列目录时没必要把每份报告整个读进内存。"""
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("# "):
                    return line[2:].strip()
    except OSError:
        pass
    return ""


def listing() -> list[Listed]:
    """按修改时间倒序列出报告。目录不存在 → 空列表(第一次跑本来就没有)。"""
    folder = directory()
    if not folder.is_dir():
        return []
    def mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return -1.0

    paths = sorted(folder.glob("*.md"), key=mtime, reverse=True)
    return [Listed(p.name, _title_of(p), day_of(p.name)) for p in paths]


def resolve_name(raw: str, available: list[Listed] | None = None) -> str | None:
    """用户输入 → 具体文件名。支持「最新」、省略 `.md`。找不到返回 None。

    先挡路径穿越:名字经模型的工具参数传进来,最终来自用户输入。
    """
    name = (raw or "").strip()
    if not name or "/" in name or "\\" in name or ".." in name:
        return None
    items = listing() if available is None else available
    if name.lower() in _LATEST_WORDS:
        return items[0].name if items else None
    if not name.endswith(".md"):
        name = f"{name}.md"
    return name if any(item.name == name for item in items) else None


def read(name: str) -> str | None:
    """读一份报告的原文。名字必须先过 `resolve_name`。"""
    path = directory() / name
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("报告读取失败:%s(%s)", path, e)
        return None


def load(name: str) -> Report | None:
    """读并解析。不是结构化榜单 → None。"""
    text = read(name)
    return parse(text) if text else None


def load_all() -> list[tuple[Listed, Report]]:
    """全部能解析的报告,按日期升序。`star_trend` 用它拼时间序列。

    同一天有多份时只留第一份:时间序列一天只能有一个点。
    """
    out: list[tuple[Listed, Report]] = []
    seen: set[date] = set()
    for item in sorted(listing(), key=lambda i: (i.day or date.min, i.name)):
        if item.day is None or item.day in seen:
            continue
        if (report := load(item.name)) is not None:
            seen.add(item.day)
            out.append((item, report))
    return out


def appearance_counts() -> tuple[dict[str, int], int]:
    """每个项目上过多少期**定时周报**,以及周报总期数。

    只数没有后缀的 `{日期}.md`(cron 每周产出的标准综合榜):带后缀的是临时跑的,计进去
    会让分子虚高而分母不涨。分子分母走同一次遍历。
    """
    counts: dict[str, int] = {}
    total = 0
    for item in listing():
        if item.day is None or item.name != f"{item.day}.md":
            continue
        if (report := load(item.name)) is None:
            continue
        total += 1
        for repo in {e.repo for e in report.entries}:    # 同一期内重复只算一次
            counts[repo] = counts.get(repo, 0) + 1
    return counts, total


def delete(name: str) -> bool:
    """删一份报告。名字必须先过 `resolve_name`。"""
    path = directory() / name
    try:
        path.unlink()
    except OSError as e:
        logger.warning("报告删除失败:%s(%s)", path, e)
        return False
    return True


def save(name: str, text: str) -> Path | None:
    """写一份报告。失败返回 None 而不是抛 —— 上游跑了两小时,不该被最后一步的磁盘错误吞掉。"""
    path = config.ensure_dir(directory()) / name
    try:
        _file_io.write_whole(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))
    except OSError as e:
        logger.error("报告写入失败:%s(%s)", path, e)
        return None
    logger.info("报告已写入:%s", path)
    return path


# ── 报告 Markdown 的格式:写与解析成对同源 ──
# 生成端(service/report.py)写标题必须经这里的函数,不许自己拼 f-string ——
# 附栏曾因写端加了 `## T1.` 而解析端不认识,整段在页面上静默消失。
# 每对「写函数 ↔ 正则」都有守卫:test_report.py::test_writers_and_parsers_agree。

# 周期 → (榜名, 增长字段名)。字段名必须写明是哪个窗口:Trending 的口径和我们的窗口增量
# 不同源,月榜的数字更不能顶着「本周新增」出现。
PERIOD_TEXT = {"weekly": ("周榜", "本周新增"), "monthly": ("月榜", "本月新增")}


def heading(rank: int, repo: str) -> str:
    """正文条目标题(和 `_HEADING` 成对)。"""
    return f"## {rank}. {repo}"


def trend_heading(rank: int, repo: str) -> str:
    """附栏条目标题(和 `_TREND_HEADING` 成对)。T 前缀让它避开正文统计。"""
    return f"## T{rank}. {repo}"


def appendix_mark(period: str) -> str:
    """附栏标题(和 `_APPENDIX` 成对)。幂等判断、渲染、解析分段共用这一份。"""
    label, _ = PERIOD_TEXT.get(period, PERIOD_TEXT["weekly"])
    return f"## 附:GitHub Trending {label}对照({period})"


_HEADING = re.compile(r"##\s+(?P<rank>\d+)\.\s+(?P<repo>[\w.-]+/[\w.-]+)\s*$")
# 附栏条目写成 `## T1. owner/repo`,和正文的纯数字排名分开,免得混进「上新/移出」、
# 出场次数、star 趋势的统计。解析成单独一份名单,不进 `entries`。
# 周榜和月榜的条目编号都从 T1 起,靠上一行的附栏标题 `(weekly)` / `(monthly)` 分段归位。
_TREND_HEADING = re.compile(r"##\s+T(?P<rank>\d+)\.\s+(?P<repo>[\w.-]+/[\w.-]+)\s*$")
_APPENDIX = re.compile(r"##\s+附[:：].*\((?P<period>[a-z]+)\)\s*$")
_META = re.compile(r"-\s*(?P<label>[^:：]+)[:：]\s*(?P<value>.+)")
_LINK = re.compile(r"链接[:：]\s*(?P<url>.+)")
_DIGITS = re.compile(r"[^\d-]")

_REQUIRED_META = ("创建时间", "总 Star")


class Entry(NamedTuple):
    rank: int
    repo: str
    link: str
    metadata: dict[str, str]
    sections: list[dict[str, str]]


class Report(NamedTuple):
    title: str
    summary: str
    entries: list[Entry]
    # {周期: 附栏条目},周期就是 Trending 的 weekly / monthly。只读;统计一律只看 entries
    trending: dict[str, list[Entry]] = {}

    def find(self, repo: str) -> Entry | None:
        """在报告里找一个仓库:先精确匹配,再子串兜底(说「找 langchain」不必打全名)。"""
        target = (repo or "").strip().lower()
        if not target:
            return None
        exact = next((e for e in self.entries if e.repo.lower() == target), None)
        return exact or next((e for e in self.entries if target in e.repo.lower()), None)


def growth_of(metadata: dict[str, str]) -> str:
    """取增长字段的值。

    字段名带着窗口天数(「近7天增长」),只能按「含『增长』」找;写死键名会在窗口一改时静默返回空。
    """
    label = next((k for k in metadata if "增长" in k), "")
    return metadata.get(label, "") if label else ""


def number_of(text: str) -> int | None:
    """「1,234」→ 1234;「+56」→ 56;取不出数字 → None。"""
    digits = _DIGITS.sub("", text or "").lstrip("-")
    return int(digits) if digits else None


def _parse_entry(lines: list[str], idx: int, rank: int, repo: str) -> tuple[Entry, int]:
    link = ""
    metadata: dict[str, str] = {}
    sections: list[dict[str, str]] = []

    while idx < len(lines):
        compact = lines[idx].strip()
        if compact.startswith("## "):
            break
        if compact == "---":
            idx += 1
            break
        if not compact:
            idx += 1
            continue
        if m := _LINK.match(compact):
            link = m.group("url").strip()
            idx += 1
            continue
        if m := _META.match(compact):
            metadata[m.group("label").strip()] = m.group("value").strip()
            idx += 1
            continue
        if compact.startswith("### "):
            title = compact[4:].strip()
            idx += 1
            body: list[str] = []
            while idx < len(lines):
                nxt = lines[idx].strip()
                if nxt.startswith(("### ", "## ")) or nxt == "---":
                    break
                body.append(lines[idx])
                idx += 1
            sections.append({"title": title, "content": "\n".join(body).strip()})
            continue
        idx += 1

    return Entry(rank, repo, link, metadata, sections), idx


def parse(markdown: str) -> Report | None:
    """解析一份报告。不是结构化榜单 → None。"""
    lines = markdown.splitlines()
    title = next((ln[2:].strip() for ln in lines if ln.startswith("# ")), "")
    summary = next((ln[1:].strip() for ln in lines if ln.startswith(">")), "")

    entries: list[Entry] = []
    trending: dict[str, list[Entry]] = {}
    period = ""
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if m := _HEADING.match(line):
            entry, idx = _parse_entry(lines, idx + 1, int(m.group("rank")), m.group("repo"))
            entries.append(entry)
        elif m := _APPENDIX.match(line):
            period = m.group("period")
            trending.setdefault(period, [])
            idx += 1
        elif period and (m := _TREND_HEADING.match(line)):
            entry, idx = _parse_entry(lines, idx + 1, int(m.group("rank")), m.group("repo"))
            trending[period].append(entry)
        else:
            idx += 1

    if not entries:
        return None
    if not any(all(e.metadata.get(k) for k in _REQUIRED_META) for e in entries):
        return None
    return Report(title, summary, entries, trending)


def as_dict(report: Report) -> dict[str, Any]:
    """给 Web / Agent 的 JSON 形状。`api_server` 直接把它序列化出去。"""
    return {
        "title": report.title,
        "summary": report.summary,
        "repos": [e._asdict() for e in report.entries],
    }
