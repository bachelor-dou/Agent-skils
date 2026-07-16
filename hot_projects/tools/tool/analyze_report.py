"""analyze_report 工具：读取已生成的榜单报告并分析（纯本地读取，不联网）。

- 无 name：列出可用报告；
- 有 name：返回该报告的紧凑项目清单（排名/仓库/Star/增长/语言/主题），供模型整体分析；
- name + repo：返回该项目在报告中的完整分段内容，供针对单个项目追问。
"""

import glob
import os

from ...config import REPORT_DIR
from ..basic.report_parse import parse_structured_report


def _growth_value(metadata: dict) -> str:
    label = next((k for k in metadata if "增长" in k), "")
    return metadata.get(label, "") if label else ""


def _read_markdown(name: str) -> str | None:
    path = os.path.join(REPORT_DIR, name)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def list_reports() -> list[dict]:
    """按修改时间倒序列出报告（名称 + 标题）。"""
    files = sorted(
        glob.glob(os.path.join(REPORT_DIR, "*.md")),
        key=os.path.getmtime,
        reverse=True,
    )
    out: list[dict] = []
    for path in files:
        title = ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("# "):
                        title = line[2:].strip()
                        break
        except OSError:
            pass
        out.append({"name": os.path.basename(path), "title": title})
    return out


def _resolve_name(name: str, available: list[dict]) -> str | None:
    """把用户输入解析为具体报告文件名。支持 '最新'/'latest'、省略 .md。"""
    raw = (name or "").strip()
    if raw in ("最新", "latest", "last", "newest"):
        return available[0]["name"] if available else None
    if not raw.endswith(".md"):
        raw = f"{raw}.md"
    names = {r["name"] for r in available}
    return raw if raw in names else None


def analyze_report(name: str | None = None, repo: str | None = None) -> dict:
    """读取报告供分析。name 为空→列出报告；给 repo→返回该项目完整分段。"""
    available = list_reports()
    if not available:
        return {"error": "当前没有任何报告。可先生成榜单报告后再分析。"}

    if not (name or "").strip():
        return {
            "reports": available,
            "hint": "请指定要分析的报告 name（可用文件名或『最新』）。",
        }

    # 路径穿越防护
    if "/" in name or "\\" in name or ".." in name:
        return {"error": "无效的报告名称。"}

    resolved = _resolve_name(name, available)
    if resolved is None:
        return {"error": f"未找到报告 `{name}`。", "available": [r["name"] for r in available]}

    parsed = parse_structured_report(_read_markdown(resolved) or "")
    if parsed is None:
        return {"error": f"报告 `{resolved}` 不是结构化榜单，无法解析。"}

    repos = parsed["repos"]

    # 针对单个项目追问：返回其完整分段内容
    if (repo or "").strip():
        target = (repo or "").strip().lower()
        match = next((r for r in repos if r["repo"].lower() == target), None)
        if match is None:
            match = next((r for r in repos if target in r["repo"].lower()), None)
        if match is None:
            return {
                "error": f"报告 `{resolved}` 中没有项目 `{repo}`。",
                "repos_in_report": [r["repo"] for r in repos],
            }
        return {
            "name": resolved,
            "repo": match["repo"],
            "rank": match["rank"],
            "link": match.get("link", ""),
            "metadata": match["metadata"],
            "sections": match["sections"],
        }

    # 整体分析：紧凑清单（一行一项，控制 token）
    header = "排名|仓库|总Star|增长|语言|主题"
    rows = [header]
    for r in repos:
        md = r["metadata"]
        topics = md.get("主题标签", "")
        topics_short = ",".join([t.strip() for t in topics.replace("，", ",").split(",") if t.strip()][:2])
        rows.append(
            f"{r['rank']}|{r['repo']}|{md.get('总 Star', '')}|"
            f"{_growth_value(md)}|{md.get('主语言', '')}|{topics_short}"
        )

    return {
        "name": resolved,
        "title": parsed.get("title", ""),
        "summary": parsed.get("summary", ""),
        "project_count": len(repos),
        "projects_table": "\n".join(rows),
        "hint": "针对某个项目深入分析时，用相同 name 再次调用并传 repo=owner/repo 获取其完整分段内容。",
    }


def analyze_report_handler(ctx, args: dict) -> dict:
    return analyze_report(name=args.get("name"), repo=args.get("repo"))
