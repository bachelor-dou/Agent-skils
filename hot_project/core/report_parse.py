"""报告 Markdown → 结构化数据。纯解析,不碰文件系统。

四个调用方共用:Web 渲染、`analyze_report`、`star_trend`、周报的两期对比。

解析的是 `report.py` 自己写出来的格式,两边必须同时改。守卫在 `test_report.py`:
生成一份再解析回来,字段对不上就红。
"""

from __future__ import annotations

import re
from typing import Any, NamedTuple

# 「## 3. owner/repo」
_HEADING = re.compile(r"##\s+(?P<rank>\d+)\.\s+(?P<repo>[\w.-]+/[\w.-]+)\s*$")
# 「- 总 Star: 1,234」;中英文冒号都认,因为 LLM 写出来的两种都有
_META = re.compile(r"-\s*(?P<label>[^:：]+)[:：]\s*(?P<value>.+)")
_LINK = re.compile(r"链接[:：]\s*(?P<url>.+)")
_DIGITS = re.compile(r"[^\d-]")

# 这两个字段齐了才算「结构化榜单」,少了就是别的 md,不该被当报告解析。
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

    def find(self, repo: str) -> Entry | None:
        """在报告里找一个仓库:先精确匹配,再子串兜底(说「找 langchain」不必打全名)。"""
        target = (repo or "").strip().lower()
        if not target:
            return None
        exact = next((e for e in self.entries if e.repo.lower() == target), None)
        return exact or next((e for e in self.entries if target in e.repo.lower()), None)


def growth_of(metadata: dict[str, str]) -> str:
    """取增长字段的值。

    字段名带着窗口天数(「近7天增长」),所以只能按「含『增长』」找 —— 写死键名会在
    窗口一改时静默返回空。
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
    idx = 0
    while idx < len(lines):
        m = _HEADING.match(lines[idx].strip())
        if not m:
            idx += 1
            continue
        entry, idx = _parse_entry(lines, idx + 1, int(m.group("rank")), m.group("repo"))
        entries.append(entry)

    if not entries:
        return None
    if not any(all(e.metadata.get(k) for k in _REQUIRED_META) for e in entries):
        return None
    return Report(title, summary, entries)


def as_dict(report: Report) -> dict[str, Any]:
    """给 Web / Agent 的 JSON 形状。`api_server` 直接把它序列化出去。"""
    return {
        "title": report.title,
        "summary": report.summary,
        "repos": [e._asdict() for e in report.entries],
    }
