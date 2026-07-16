"""fetch_trending 工具：获取 GitHub Trending 列表。"""


def fetch_trending_handler(ctx, args: dict) -> dict:
    return ctx.provider.fetch_trending(trending_range=args.get("trending_range", "weekly"))
