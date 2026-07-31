"""三张榜的 Agent 工具:综合、新项目、关键词。

出榜要先实时取全库当前 star、再为 Top N 逐个调模型写介绍,一次几分钟,所以首次调用只回显
参数、等用户回「开始」。回显和执行必须是同一份参数:参数存进会话,`confirm=true` 复调时用
存下的那份 —— 模型复述参数时会漂移(少个 `min_star`、把 top_n 从 20 写成 10),而用户确认
的是屏幕上那份。

关键词榜多一步搜索:「哪些仓库和向量数据库有关」快照和 DB 里都没有,只能真去搜一遍。
"""

from __future__ import annotations

import json
import logging

from .. import config
from . import ranking
from .spec import Param, Tool

logger = logging.getLogger("hot_project")

LABEL = {"comprehensive": "综合热榜", "hot_new": "新项目热榜", "keyword": "关键词热榜"}
KEYWORDS_SHOWN = 6      # 确认文案里列几个关键词


def _signature(mode: str, params: dict) -> str:
    return json.dumps({"mode": mode, **params}, ensure_ascii=False,
                      sort_keys=True, default=str)


def confirmation(mode: str, params: dict) -> str:
    """把生效参数原样拼成一句话。**展示即执行** —— 不经模型转述,不漏参数。"""
    bits: list[str] = []
    if words := params.get("keywords"):
        shown = "、".join(words[:KEYWORDS_SHOWN]) + ("…" if len(words) > KEYWORDS_SHOWN else "")
        bits.append(f"关键词 {len(words)} 个({shown})")
    if topic := params.get("topic"):
        bits.append(f"方向:{topic}")
    if (top_n := params.get("top_n")) is not None:
        bits.append(f"Top {top_n}")
    if (min_star := params.get("min_star")) is not None:
        bits.append(f"最低 star={min_star}")
    if (threshold := params.get("growth_threshold")) is not None:
        bits.append(f"增长阈值={threshold}"
                    + ("(不过滤增长)" if threshold == 0 else "(窗口内需涨够这么多才入选)"))
    bits.append(f"增长窗口={params.get('growth_days', config.GROWTH_CALC_DAYS)}天")
    if (created := params.get("created_days")) is not None:
        bits.append(f"新项目创建窗口={created}天")
    bits.append("生成报告文件(较慢,逐个项目写介绍)" if params.get("generate_report")
                else "不生成报告文件(榜单直接在对话里给)")
    return (f"将执行【{LABEL.get(mode, mode)}】,参数:" + ";".join(bits)
            + "。确认无误请回复『开始』;要改参数(如降低阈值、换关键词)直接说。")


def _keyword_pool(ctx, params: dict) -> dict[str, dict] | None:
    """关键词榜的候选名单:搜一遍,只留名字和创建时间。

    不带 star —— 那是 `ranking.run` 现取的活儿,搜索结果里那个值到出榜时已经旧了。
    `created_at` 优先用搜索结果的,DB 里那份兜底(Trending 来的条目没有它)。
    """
    words = params.get("keywords") or []
    if not words:
        # 不能返回 None —— 那在 `_run` 里的含义是「没有候选池」,`ranking.run` 会去排全库:
        # 用户要关键词榜,拿到一份综合榜,几分钟模型调用全花在不相关的项目上,还不报错。
        logger.warning("关键词榜收到空的 keywords,候选池按空处理(不退化成全库排名)。")
        return {}
    found = ctx.gh.keyword_sweep(words, params["min_star"])
    if not found:
        return {}
    saved = ranking.universe.load()
    pool = {
        name: {"created_at": (item.get("created_at")
                              or saved.get(name, {}).get("created_at", ""))}
        for name, item in found.items()
    }
    logger.info("关键词候选名单:搜到 %d 个。", len(pool))
    return pool


def _run(ctx, mode: str, params: dict) -> dict:
    pool = _keyword_pool(ctx, params) if mode == "keyword" else None
    result = ranking.run(
        mode=mode, min_star=params["min_star"],
        growth_threshold=params["growth_threshold"],
        growth_days=params.get("growth_days") or config.GROWTH_CALC_DAYS,
        created_days=params.get("created_days"), top_n=params.get("top_n"),
        topic=params.get("topic"), do_report=params.get("generate_report", False),
        gh=ctx.gh, progress=ctx.progress, pool=pool,
    )
    ranked = result["ranked"]
    return {
        "mode": mode, "ranked_count": len(ranked), "funnel": result["funnel"],
        "growth_calc_days": result["growth_days"],
        "report_path": result.get("report_path", ""),
        "ranked": [{"rank": i, "repo": name, "growth": info["growth"],
                    "star": info["star"]}
                   for i, (name, info) in enumerate(ranked, 1)],
    }


def make(mode: str):
    """造一张榜的 handler。三张榜除了模式之外完全一样。"""

    def handler(ctx, args: dict) -> dict:
        params = dict(args)
        confirm = bool(params.pop("confirm", False))
        signature = _signature(mode, params)

        pending = ctx.state.pending_confirmation_signature if ctx.state else None
        stored = (ctx.state.tool_state.get("pending_ranking") or {}) if ctx.state else {}
        # 确认必须认「是哪张榜」:只看 pending 非空的话,回显关键词榜、再拿 confirm=true
        # 调综合榜就能直接执行,走的还是模型这次传的参数。
        # 同签名复调也算确认 —— 有些模型不带 confirm,而是把参数原样再发一遍。
        if not (pending and stored.get("mode") == mode and (confirm or pending == signature)):
            if ctx.state is not None:
                ctx.state.pending_confirmation_signature = signature
                ctx.state.tool_state["pending_ranking"] = {"mode": mode, "params": params}
            return {"needs_confirmation": True, "mode": mode, "params": params,
                    "message": confirmation(mode, params)}

        if "params" in stored:              # mode 已在上面比过
            params = stored["params"]       # 用回显过的那份,见模块头部
        if ctx.state is not None:
            ctx.state.pending_confirmation_signature = None
            ctx.state.tool_state.pop("pending_ranking", None)
        return _run(ctx, mode, params)

    return handler


def _threshold(default: int, note: str = "") -> Param:
    return Param("growth_threshold", "int", f"增长入选阈值,默认 {default}。{note}".strip(),
                 default=default, min=0)


_COMMON = (
    Param("min_star", "int", f"最低 star 门槛,默认 {config.MIN_STAR}",
          default=config.MIN_STAR, min=1),
    Param("growth_days", "int",
          f"增长统计窗口(天),默认 {config.GROWTH_CALC_DAYS}。"
          "当天缺快照时会自动顺延到邻近的那天,返回值里会写实际天数。",
          default=config.GROWTH_CALC_DAYS, min=1),
    Param("confirm", "bool",
          "仅在用户已明确确认(回复『开始』『确认』『go』之类)时置 true;"
          "此时按上一轮回显的参数执行,参数无需重复。首次提出请求时不要设或设 false。",
          default=False),
    Param("generate_report", "bool",
          "是否额外产出一份 Markdown 报告文件,默认 false。仅当用户明确要报告"
          "(『生成报告』『出份报告』『发我一份』之类)时才置 true;用户只是想看榜单、"
          "找项目、问某方向有什么新东西,一律不要设 —— 出报告要为每个项目逐条调模型写介绍,"
          "慢得多也贵得多,而不出报告同样会把完整榜单返回给你,你直接在回复里讲清楚即可。",
          default=False),
)

TOOLS = (
    Tool("comprehensive_ranking",
         "【综合热榜·昂贵】按窗口内 star 增长对全库排名,输出综合 Top N。"
         "会实时取全库当前 star(几分钟),再和快照基线相减算增长。"
         "执行前请先回显参数并等用户确认『开始』。默认不产报告文件,只把榜单返回给你。",
         make("comprehensive"),
         (*_COMMON, _threshold(config.STAR_GROWTH_THRESHOLD),
          Param("top_n", "int", f"返回前 N,默认 {config.HOT_PROJECT_COUNT}",
                default=config.HOT_PROJECT_COUNT, min=1, max=200)),
         expensive=True),
    Tool("hot_new_ranking",
         "【新项目热榜·昂贵】只看近 created_days 天内创建的新项目,按增长排序。"
         "执行前先回显参数等用户确认『开始』。默认不产报告文件,只把榜单返回给你。",
         make("hot_new"),
         (*_COMMON, _threshold(config.STAR_GROWTH_THRESHOLD),
          Param("created_days", "int",
                f"新项目创建时间窗口(天),默认 {config.DAYS_SINCE_CREATED}",
                default=config.DAYS_SINCE_CREATED, min=1),
          Param("top_n", "int", f"返回前 N,默认 {config.HOT_NEW_PROJECT_COUNT}",
                default=config.HOT_NEW_PROJECT_COUNT, min=1, max=200)),
         expensive=True),
    Tool("keyword_ranking",
         "【关键词热榜·昂贵】按关键词搜 GitHub,再对搜到的项目算增长排序。"
         "挑词前先调 get_keyword_catalog 看预设分组表:从相关组挑关键词,"
         "并补充没覆盖到的英文搜索词,一起传进 keywords。"
         "执行前先回显参数等用户确认『开始』。默认不产报告文件,只把榜单返回给你。",
         make("keyword"),
         (*_COMMON,
          Param("keywords", "list_str",
                "要搜的英文关键词列表,如 [\"vector database\", \"voice assistant llm\"]。"),
          Param("topic", "str",
                "本次方向的简短中文概括,6 字以内,用于报告标题点明方向,"
                "如『向量数据库』『AI语音助手』。", default=None),
          _threshold(0, "关键词榜是细分定向搜索,套增长突刺阈值几乎必空,"
                        "所以默认不过滤、按增长量降序返回;一般无需设置。"),
          Param("top_n", "int", f"返回前 N,默认 {config.HOT_PROJECT_COUNT}",
                default=config.HOT_PROJECT_COUNT, min=1, max=200)),
         expensive=True),
)
