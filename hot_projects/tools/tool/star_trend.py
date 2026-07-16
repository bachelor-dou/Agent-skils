"""star_trend 工具：从历史报告推导某项目的多周 star 轨迹（本地读取，不联网）。

复用 report/*.md —— 每份周报都记录了当周 Top-N 项目的总 star，按日期拼成时间序列，
用于判断项目"在涨 / 见顶 / 退烧"。仅覆盖曾上过榜的项目；某周未上榜则该周缺点。
"""

import glob
import os
import re

from ...config import REPORT_DIR
from ..basic.report_parse import parse_structured_report

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _report_date(filename: str) -> str:
    m = _DATE_RE.search(filename)
    return m.group(1) if m else ""


def _to_int(text: str) -> int | None:
    digits = re.sub(r"[^\d]", "", text or "")
    return int(digits) if digits else None


def _growth_value(metadata: dict) -> str:
    label = next((k for k in metadata if "增长" in k), "")
    return metadata.get(label, "") if label else ""


def star_trend(repo: str) -> dict:
    """返回该项目按周的 star 轨迹（升序）。"""
    target = (repo or "").strip().lower()
    if not target:
        return {"error": "缺少 repo。"}

    by_date: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(REPORT_DIR, "*.md"))):
        date = _report_date(os.path.basename(path))
        if not date or date in by_date:
            # 同一天多份报告（如综合/关键词榜）只取一次
            if date in by_date:
                continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                parsed = parse_structured_report(f.read())
        except OSError:
            continue
        if not parsed:
            continue
        # 优先精确匹配，其次子串匹配
        match = next((r for r in parsed["repos"] if r["repo"].lower() == target), None)
        if match is None:
            match = next((r for r in parsed["repos"] if target in r["repo"].lower()), None)
        if match is None:
            continue
        md = match["metadata"]
        by_date[date] = {
            "date": date,
            "repo": match["repo"],
            "rank": match["rank"],
            "star": _to_int(md.get("总 Star", "")),
            "growth": _growth_value(md),
        }

    series = [by_date[d] for d in sorted(by_date)]
    if not series:
        return {"repo": repo, "points": 0,
                "message": "该项目未在历史报告中出现过，无法给出 star 轨迹。"}

    first, last = series[0]["star"], series[-1]["star"]
    delta = (last - first) if (first is not None and last is not None) else None
    return {
        "repo": series[-1]["repo"],
        "points": len(series),
        "span": f"{series[0]['date']} → {series[-1]['date']}",
        "star_change": delta,
        "series": series,
        "hint": "series 按周升序，star=当周总 star、rank=当周排名；据此判断在涨/见顶/退烧。缺周=该周未上榜。",
    }


def star_trend_handler(ctx, args: dict) -> dict:
    return star_trend(args.get("repo"))
