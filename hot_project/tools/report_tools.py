"""读历史报告的两个工具:整体分析、单项目 star 轨迹。都是本地读取,不联网。"""

from __future__ import annotations

import logging

from ..infra.data_access import reports
from ..service import repo_card
from .spec import Param, Tool

logger = logging.getLogger("hot_project")

TOPICS_IN_TABLE = 2     # 紧凑清单里每个项目展示几个标签(控 token)


def analyze_report(ctx, args: dict) -> dict:
    """不给 name → 列出报告;给 name → 紧凑清单;再给 repo → 该项目的完整分段。"""
    available = reports.listing()
    if not available:
        return {"error": "当前没有任何报告。可先生成榜单报告后再分析。"}

    if not (args.get("name") or "").strip():
        return {"reports": [{"name": i.name, "title": i.title} for i in available],
                "hint": "请指定要分析的报告 name(可用文件名或『最新』)。"}

    name = reports.resolve_name(args["name"], available)
    if name is None:
        return {"error": f"没找到报告 `{args['name']}`。",
                "available": [i.name for i in available]}

    report = reports.load(name)
    if report is None:
        return {"error": f"报告 `{name}` 不是结构化榜单,无法解析。"}

    if (repo := (args.get("repo") or "").strip()):
        entry = report.find(repo)
        if entry is None:
            return {"error": f"报告 `{name}` 里没有项目 `{repo}`。",
                    "repos_in_report": [e.repo for e in report.entries]}
        return {"name": name, "repo": entry.repo, "rank": entry.rank,
                "link": entry.link, "metadata": entry.metadata,
                "sections": entry.sections}

    rows = ["排名|仓库|总Star|增长|语言|主题"]
    for entry in report.entries:
        meta = entry.metadata
        topics = ",".join(t.strip() for t in
                          meta.get("主题标签", "").replace(",", ",").split(",")
                          if t.strip())
        rows.append(f"{entry.rank}|{entry.repo}|{meta.get('总 Star', '')}|"
                    f"{reports.growth_of(meta)}|{meta.get('主语言', '')}|"
                    f"{','.join(topics.split(',')[:TOPICS_IN_TABLE])}")

    return {"name": name, "title": report.title, "summary": report.summary,
            "project_count": len(report.entries), "projects_table": "\n".join(rows),
            "hint": "要深入某个项目就用相同 name 再调一次并传 repo=owner/repo,拿它的完整分段。"}


def star_trend(ctx, args: dict) -> dict:
    """star 轨迹。实现在 `service.repo_card`(和网页「star 走势」按钮同一份)。"""
    repo = (args.get("repo") or "").strip()
    if not repo:
        return {"error": "缺少 repo。"}
    return repo_card.trend(repo)


TOOLS = (
    Tool("analyze_report",
         "【报告分析】读取已生成的榜单报告并分析(本地读取,不联网)。"
         "不传 name→列出可用报告;传 name(文件名或『最新』)→返回该报告的项目清单"
         "(排名/仓库/Star/增长/语言/主题),用于整体分析与筛选;"
         "再带 repo=owner/repo→返回该项目在报告中的完整分段内容,用于针对单个项目追问。",
         analyze_report,
         (Param("name", "str", "报告文件名(如 2026-07-08.md)或『最新』;不传则列出全部报告。",
                default=None),
          Param("repo", "str", "可选,owner/repo;配合 name 获取该项目在报告中的完整分析分段。",
                default=None))),
    Tool("star_trend",
         "【star 轨迹】从历史周报推导某项目多周的总 star 与排名变化(本地读取,不联网),"
         "用于判断项目在涨/见顶/退烧。仅覆盖曾上过榜的项目;某周未上榜则该周缺点。",
         star_trend,
         (Param("repo", "str", "owner/repo(如 vllm-project/vllm);也可只给项目名,"
                               "按报告内名称匹配。"),)),
)
