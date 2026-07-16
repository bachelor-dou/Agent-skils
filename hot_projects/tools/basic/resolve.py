"""单仓库输入解析（公用）：把用户输入解析为确定的 owner/repo。

被 repo_growth / describe_project / repo_profile 等单仓库工具复用：
用户可给完整 owner/repo、仅项目名、或一句描述；能唯一定位即返回，
真有歧义才返回 disambiguation 候选交给 LLM 找用户确认。
"""


def resolve_repo(ctx, raw: str):
    """返回 (full_name, None) 表示已唯一定位；(None, payload) 表示需消歧或未找到。"""
    raw = (raw or "").strip()
    if not raw:
        return None, {"error": "缺少必需参数 repo"}

    # 形如 owner/repo：先精确查；命中即用，未命中则按 repo 名去搜相似项目
    if "/" in raw:
        if ctx.provider.repo_info(raw) is not None:
            return raw, None
        query = raw.split("/", 1)[1].strip() or raw
    else:
        query = raw

    cands = ctx.provider.search_similar(query, limit=5)
    if not cands:
        return None, {"error": f"未找到与 `{raw}` 匹配的仓库，也没有相似项目。请确认 owner/repo。"}

    # 唯一定位：只有一个候选，或恰有一个「项目名完全相同」的强匹配
    if len(cands) == 1:
        return cands[0].full_name, None
    exact = [c for c in cands if c.full_name.rsplit("/", 1)[-1].lower() == query.lower()]
    if len(exact) == 1:
        return exact[0].full_name, None

    return None, {
        "disambiguation": True,
        "message": f"`{raw}` 有多个相近项目，请向用户确认是哪一个（回复完整 owner/repo）。",
        "candidates": [
            {"full_name": c.full_name, "star": c.star, "desc": c.description} for c in cands
        ],
    }
