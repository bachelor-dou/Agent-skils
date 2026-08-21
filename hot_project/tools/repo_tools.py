"""单仓库工具:增长、介绍、画像、搜索、收藏。

五个都从同一步开始 —— 把用户说的东西变成确定的 `owner/repo`,所以放一个文件里。
"""

from __future__ import annotations

import logging

from .. import config
from ..common.timeutil import utc_today
from ..infra.data_access import favorites, universe
from ..service import describe
from ..service import favorites as favorite_service
from ..service import repo_card
from .spec import Param, Tool

logger = logging.getLogger("hot_project")

PROFILE_README_MAX = 5000
SEARCH_DEFAULT_N = 5
SEARCH_MAX_N = 20
DISAMBIGUATION_CANDIDATES = 5


# ── 把用户说的变成 owner/repo ────────────────────────────────────────

def resolve(ctx, raw: str) -> tuple[str | None, dict | None]:
    """返回 `(仓库全名, None)` 表示定位成功;`(None, 回给模型的话)` 表示要消歧或没找到。

    能唯一定位就直接用,真有歧义才交回候选让模型问 —— 猜错一个仓库比多问一句代价大得多。
    """
    raw = (raw or "").strip()
    if not raw:
        return None, {"error": "缺少必需参数 repo"}
    if ctx.gh is None or not ctx.gh.usable:
        return None, {"error": "没有可用的 GitHub token,无法查询仓库。"}

    if "/" in raw:
        if ctx.gh.info(raw) is not None:
            return raw, None
        query = raw.split("/", 1)[1].strip() or raw       # owner 可能拼错,拿仓库名再搜
    else:
        query = raw

    found = ctx.gh.search(query, limit=DISAMBIGUATION_CANDIDATES)
    if not found:
        return None, {"error": f"没找到与 `{raw}` 匹配的仓库,也没有相似项目。请确认 owner/repo。"}
    if len(found) == 1:
        return found[0]["full_name"], None

    exact = [r for r in found
             if r["full_name"].rsplit("/", 1)[-1].lower() == query.lower()]
    if len(exact) == 1:
        return exact[0]["full_name"], None

    return None, {
        "disambiguation": True,
        "message": f"`{raw}` 有多个相近项目,请向用户确认是哪一个(回复完整 owner/repo)。",
        "candidates": [{"full_name": r["full_name"], "star": r.get("stargazers_count", 0),
                        "desc": r.get("description") or ""} for r in found],
    }


# ── repo_growth ─────────────────────────────────────────────────────

def repo_growth(ctx, args: dict) -> dict:
    """增长算法在 `service.repo_card.live_growth`(和报告页刷新按钮同一份)。"""
    name, payload = resolve(ctx, args.get("repo"))
    if payload is not None:
        return payload

    days = args.get("growth_calc_days", config.GROWTH_CALC_DAYS)
    got = repo_card.live_growth(name, ctx.gh, days)
    if got["star"] is None:
        return {"error": f"拿不到 {name} 的当前 star。"}
    if got["growth"] is None:
        return {"repo": name, "star": got["star"], "growth": None,
                "message": f"最近 {days} 天的快照里一次都没测到过 {name}(多半是它刚进库),"
                           "这个窗口算不出增长 —— 不要当成零增长,过一天有了基线就能算。"}

    out = {"repo": name, **got}
    if got["growth_calc_days"] != days:
        out["note"] = (f"它最早的基线只到 {got['growth_calc_days']} 天前,"
                       f"所以这是 {got['growth_calc_days']} 天的增长(请求的是 {days} 天)。")
    return out


# ── describe_project ────────────────────────────────────────────────

def describe_project(ctx, args: dict) -> dict:
    name, payload = resolve(ctx, args.get("repo"))
    if payload is not None:
        return payload

    saved = universe.load().get(name, {})
    if cached := saved.get("desc"):
        return {"repo": name, "description": cached, "source": "cache"}

    facts = describe.merge_profile(
        saved, ctx.gh.profile(name, want=("info", "readme", "releases", "commits")))
    text = describe.describe(name, facts, describe.STANDARD)
    if not text:
        return {"error": f"生成 {name} 的介绍失败(LLM 未配置或全部平台不可用)。"}

    universe.write_descriptions(
        {name: {"desc": text, "desc_updated_at": str(utc_today())}})
    return {"repo": name, "description": text, "source": "llm"}


# ── repo_profile ────────────────────────────────────────────────────

def repo_profile(ctx, args: dict) -> dict:
    """一次给全原始证据,不做归纳 —— 品类判断、优缺点、活跃度由模型自己从这些事实里读。"""
    name, payload = resolve(ctx, args.get("repo"))
    if payload is not None:
        return payload

    pack = ctx.gh.profile(name)
    info = pack.get("info") or {}
    readme = pack.get("readme") or {}
    return {
        "repo": name,
        "html_url": info.get("html_url") or f"https://github.com/{name}",
        "description": info.get("description") or "",
        "language": info.get("language") or "",
        "topics": info.get("topics") or [],
        "star": info.get("stargazers_count", 0),
        "forks": info.get("forks_count", 0),
        "open_issues": info.get("open_issues_count", 0),
        "created_at": info.get("created_at") or "",
        "pushed_at": info.get("pushed_at") or "",
        "license": (info.get("license") or {}).get("spdx_id") or "",
        "archived": bool(info.get("archived")),
        "readme_excerpt": readme.get("text", "")[:PROFILE_README_MAX],
        "readme_truncated": bool(readme.get("truncated")),
        "recent_releases": pack.get("releases") or [],
        "recent_commits": (pack.get("commits") or [])[:5],
    }


# ── search_repos ────────────────────────────────────────────────────

def search_repos(ctx, args: dict) -> dict:
    query = (args.get("query") or "").strip()
    if not query:
        return {"error": "缺少查询词 query。请把用户需求转成简洁的英文关键词。"}
    if "in:" not in query.lower():
        query = f"{query} in:name,description,readme"
    if (min_star := args.get("min_star", 0)) > 0:
        query = f"{query} stars:>={min_star}"

    top_n = args.get("top_n", SEARCH_DEFAULT_N)
    found = ctx.gh.search(query, limit=top_n)
    if not found:
        return {"query": query, "count": 0,
                "message": "没搜到匹配项目,可换一组关键词或放宽条件重试。"}
    return {
        "query": query, "count": len(found),
        "results": [{"rank": i, "repo": r["full_name"],
                     "star": r.get("stargazers_count", 0),
                     "language": r.get("language") or "",
                     "description": r.get("description") or "",
                     "url": f"https://github.com/{r['full_name']}"}
                    for i, r in enumerate(found, 1)],
    }


# ── add_favorite ────────────────────────────────────────────────────

def add_favorite(ctx, args: dict) -> dict:
    if not favorites.valid_user_id(ctx.user_id):
        return {"error": "当前会话未登录,无法收藏。请在网页右上角登录后重试。"}

    name, payload = resolve(ctx, args.get("repo"))
    if payload is not None:
        return payload

    if universe.load().get(name) is None:
        info = ctx.gh.info(name)
        if not info:
            return {"error": f"没找到仓库 {name},无法收藏。"}
        universe.insert_discovered({name: {"star": info.get("stargazers_count", 0),
                                           "created_at": info.get("created_at", "")}})
        universe.refresh_display({name: {
            "language": info.get("language") or "",
            "topics": info.get("topics") or [], "gh_desc": info.get("description") or ""}})

    # 收藏的不变式(概要只给新收藏生成、不覆盖用户手写的那句、分类规则)统一走
    # service.favorites —— 网页 ☆ 和这里是同一条写路径。
    try:
        items, _total = favorite_service.update(ctx.user_id, name, "add")
    except ValueError as e:
        return {"error": str(e)}
    short = next((x.get("short_desc", "") for x in items if x.get("repo") == name), "")
    return {"ok": True, "repo": name, "short_desc": short,
            "message": f"已将 {name} 加入你的收藏。"}


# ── 契约 ────────────────────────────────────────────────────────────

_REPO = Param("repo", "str",
              "owner/repo;也可只给项目名、拼错、或一句描述,会自动检索匹配,"
              "有歧义时返回候选。")

TOOLS = (
    Tool("repo_growth",
         "【单仓库增长】查单个仓库近期 star 增长(实时 star 减去窗口内最早那份快照里的 star)。"
         "窗口内没测到过它时 growth 为 null 并附原因说明,不要当成零增长。"
         "若精确仓库查不到,会返回相似候选供用户选择。",
         repo_growth,
         (_REPO,
          Param("growth_calc_days", "int", f"增长统计窗口(天),默认{config.GROWTH_CALC_DAYS}",
                default=config.GROWTH_CALC_DAYS, min=1))),
    Tool("describe_project",
         "【项目介绍】生成单个仓库的中文功能介绍。精确查不到会返回相似候选供选择。",
         describe_project, (_REPO,)),
    Tool("repo_profile",
         "【项目画像取证】一次获取单仓库的原始证据:README 摘录、官方简介、topics、语言、"
         "star/forks/issues、创建/最近推送时间、release 节奏、近期提交(是否活跃维护)。"
         "只取证不归纳 —— 功能清单、场景覆盖、上手方式、优缺点、活跃度判断由你基于返回内容"
         "自行提炼。用于了解单个项目或同类项目对比(各调一次)。",
         repo_profile, (_REPO,)),
    Tool("search_repos",
         "【按描述找项目】把用户的自然语言需求转成简洁的英文搜索词,去 GitHub 按 star 降序找"
         " Top N 项目。适合『帮我找个手机远程控制 agent 的项目』这类『找到那个项目』的诉求"
         "—— 即时、轻量、不出榜单、不算增长。query 用 2-4 个核心英文关键词(可加引号词组),"
         "不要堆太多词以免零结果。结果不满意就换一组同义关键词再试。",
         search_repos,
         (Param("query", "str",
                "GitHub 搜索查询:由用户需求提炼的简洁英文关键词,"
                "如 'mobile remote control ai agent'。"),
          Param("top_n", "int", f"返回前 N 个,默认 {SEARCH_DEFAULT_N},最多 {SEARCH_MAX_N}。",
                default=SEARCH_DEFAULT_N, min=1, max=SEARCH_MAX_N),
          Param("min_star", "int", "可选最低 star 门槛,默认 0(不限制)。想只看有名气的可设 1000。",
                default=0, min=0))),
    Tool("add_favorite",
         "【收藏项目】把某项目加入当前用户收藏(需用户已登录)。适用于用户在对话中分析后"
         "确认『收藏/加入收藏』某项目时调用;DB 无此项目会自动拉取信息入库并生成一句话中文概要。",
         add_favorite, (_REPO,)),
)
