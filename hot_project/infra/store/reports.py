"""报告目录 —— `report/*.md` 的列举、读、写。

    data/report/2026-07-30.md          综合榜
    data/report/2026-07-30_NEW.md      新项目榜
    data/report/2026-07-30_KEY_向量库.md  关键词榜

**文件名是唯一的元数据载体。** 日期从文件名解析(`star_trend` 靠它拼时间序列);模式后缀
和关键词榜的方向名让同一天的多张榜互不覆盖 —— 盖掉之后没有任何痕迹。
"""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path
from typing import NamedTuple

from ... import config
from ...common.timeutil import parse_day
from ...core import report_parse
from . import atomic

logger = logging.getLogger("hot_project")

_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_LATEST_WORDS = frozenset({"最新", "latest", "last", "newest"})

# 文件名里不能出现的字符。关键词榜的方向名来自用户输入,不清理会写出
# `report/../../etc/x.md` 这种路径 —— 而写报告这一步是有权限建目录的。
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
    # glob 和 stat 之间有个窗口:文件正好在这中间被删掉,裸 `p.stat()` 就抛
    # FileNotFoundError 让整个列表接口 500。取不到时间的排最后。
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


def load(name: str) -> report_parse.Report | None:
    """读并解析。不是结构化榜单 → None。"""
    text = read(name)
    return report_parse.parse(text) if text else None


def load_all() -> list[tuple[Listed, report_parse.Report]]:
    """全部能解析的报告,按日期升序。`star_trend` 用它拼时间序列。

    同一天有多份时只留第一份:时间序列一天只能有一个点。
    """
    out: list[tuple[Listed, report_parse.Report]] = []
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
        # 裸 write_text 不是原子的:写到一半被 kill(CI 超时、容器被回收)会留下一份截断的
        # 报告,而 report_parse 对截断内容照样返回对象、不返回 None —— 于是下一份周报会拿
        # 这半截当"上一期"算差异,页面上看不出任何异常。走 write_whole 就只有"旧的"或
        # "完整的新的"两种状态。
        atomic.write_whole(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))
    except OSError as e:
        logger.error("报告写入失败:%s(%s)", path, e)
        return None
    logger.info("报告已写入:%s", path)
    return path
