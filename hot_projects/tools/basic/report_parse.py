"""报告 Markdown 结构化解析（Web 渲染与 Agent 读取共用）。

把 report/*.md 解析为 {title, summary, repos:[{rank, repo, link, metadata, sections}]}。
纯函数、无外部依赖，供 api_server 渲染和 analyze_report 工具共同调用。
"""

import re


def parse_structured_report(markdown_text: str) -> dict | None:
    """解析结构化报告；非结构化（无项目条目）返回 None。"""
    lines = markdown_text.splitlines()
    title = next((line[2:].strip() for line in lines if line.startswith("# ")), "")
    summary = next((line[1:].strip() for line in lines if line.startswith(">")), "")
    repos: list[dict] = []
    idx = 0

    while idx < len(lines):
        stripped = lines[idx].strip()
        heading_match = re.match(
            r"##\s+(?P<rank>\d+)\.\s+(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)\s*$", stripped
        )
        if not heading_match:
            idx += 1
            continue

        rank = int(heading_match.group("rank"))
        repo_name = heading_match.group("repo")
        idx += 1

        link = ""
        metadata: dict[str, str] = {}
        sections: list[dict[str, str]] = []

        while idx < len(lines):
            current = lines[idx].rstrip()
            compact = current.strip()
            if compact.startswith("## "):
                break
            if not compact:
                idx += 1
                continue
            if compact == "---":
                idx += 1
                break
            if compact.startswith("链接:") or compact.startswith("链接："):
                link = compact.split(":", 1)[1].strip() if ":" in compact else compact.split("：", 1)[1].strip()
                idx += 1
                continue

            meta_match = re.match(r"-\s*(?P<label>[^:：]+)[:：]\s*(?P<value>.+)", compact)
            if meta_match:
                metadata[meta_match.group("label").strip()] = meta_match.group("value").strip()
                idx += 1
                continue

            if compact.startswith("### "):
                section_title = compact[4:].strip()
                idx += 1
                block_lines: list[str] = []
                while idx < len(lines):
                    block_line = lines[idx]
                    block_compact = block_line.strip()
                    if block_compact.startswith("### ") or block_compact.startswith("## "):
                        break
                    if block_compact == "---":
                        break
                    block_lines.append(block_line)
                    idx += 1
                sections.append({
                    "title": section_title,
                    "content": "\n".join(block_lines).strip(),
                })
                continue

            idx += 1

        repos.append({
            "rank": rank,
            "repo": repo_name,
            "link": link,
            "metadata": metadata,
            "sections": sections,
        })

    if not repos:
        return None
    if not any(repo["metadata"].get("创建时间") and repo["metadata"].get("总 Star") for repo in repos):
        return None

    return {"title": title, "summary": summary, "repos": repos}
