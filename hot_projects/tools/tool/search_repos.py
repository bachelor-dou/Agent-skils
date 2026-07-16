"""search_repos 工具：按描述/关键词查 GitHub，按 star 降序返回 Top N（找某个具体项目）。"""


def search_repos_handler(ctx, args: dict) -> dict:
    query = (args.get("query") or "").strip()
    if not query:
        return {"error": "缺少查询词 query。请把用户需求转成简洁的英文关键词。"}
    # 默认把匹配范围扩到 README，提升按描述找项目的命中率
    if "in:" not in query.lower():
        query = f"{query} in:name,description,readme"
    top_n = args.get("top_n", 5)
    min_star = args.get("min_star", 0)
    repos = ctx.provider.search_top_repos(query, top_n=top_n, min_star=min_star)
    if not repos:
        return {"query": query, "count": 0,
                "message": "没搜到匹配项目，可换一组关键词或放宽条件重试。"}
    return {
        "query": query,
        "count": len(repos),
        "results": [
            {
                "rank": i + 1,
                "repo": r.full_name,
                "star": r.star,
                "language": r.language,
                "description": r.description,
                "url": f"https://github.com/{r.full_name}",
            }
            for i, r in enumerate(repos)
        ],
    }
