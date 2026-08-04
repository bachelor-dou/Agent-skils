"""每日快照:淘汰判定、采集任务、收集任务、Trending 解析。

用 httpx 的 MockTransport 假造 GitHub —— 任务、客户端、任务池、token 池全是真的,
只有网络那一层是假的。这样测到的是真实接线,而不是一堆互相配合的 mock。
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from hot_project import cron_daily_snapshot as cron
from hot_project.infra.exceptions import RateLimitError, RetryableError, TokenInvalidError
from hot_project.infra.tasks import TaskPool
from hot_project.provider.github import request as gh
from hot_project.provider.github import collect
from hot_project.provider.github import tokens as gh_tokens
from hot_project.provider.github import trending as gh_trending


# ──────────────────────────────────────────────────────────
# 淘汰:错了会一次性删掉大批活仓库
# ──────────────────────────────────────────────────────────
TRACKED = {"a/live", "a/small", "a/gone", "a/unasked"}


def test_github_says_gone_means_evict():
    plan = cron.decide(
        TRACKED, {"a/live": 900}, {"a/gone"}, star_floor=500,
    )
    assert plan.missing == ["a/gone"]
    assert "a/gone" in plan.all


def test_below_the_floor_means_evict():
    plan = cron.decide(
        TRACKED, {"a/live": 900, "a/small": 499}, set(), star_floor=500,
    )
    assert plan.too_small == ["a/small"]
    assert "a/live" not in plan.all


def test_phase_yield_counts_each_phase_and_overlaps():
    """一个新仓库被关键词和星段同时搜到,两个阶段都要算上它(不瓜分),
    所以三阶段相加会大于去重后的新增总数;只有 fresh 里的才计数。"""
    sources = {
        "kw:agent": {"a/x", "a/shared"},
        "kw:llm": {"a/shared"},          # 跨关键词重叠,阶段内只算一次
        "segment": {"a/shared", "a/seg"},
        "trending": {"a/trend", "a/old"},
    }
    fresh = {"a/x", "a/shared", "a/seg", "a/trend"}   # a/old 是老仓库,不算
    kw, seg, trend = cron.phase_yield(sources, fresh)
    assert (kw, seg, trend) == (2, 2, 1)             # 关键词:x+shared;星段:shared+seg;trending:trend
    assert kw + seg + trend > len(fresh)             # a/shared 被重复计入,相加大于去重总数


def test_exactly_at_the_floor_stays():
    """门槛是「涨过它就收进来」,所以等于门槛的必须留 —— 否则会和发现阶段打架:
    今天淘汰、明天发现又收回来,天天反复。"""
    plan = cron.decide(TRACKED, {"a/small": 500}, set(), star_floor=500)
    assert not plan.all


def test_a_whole_batch_failing_evicts_nobody():
    """整批限流失败 → 什么都不该淘汰。

    这是本文件里最要紧的一条。「GitHub 确认查不到」和「我们这次没问到」长得一样
    (都是「快照里没有这个键」),但把后者当前者,一次限流高峰就能删掉上万个活仓库。
    """
    plan = cron.decide(TRACKED, {}, set(), star_floor=500)
    assert plan.all == [], "没问到 ≠ 不存在"


def test_an_implausible_number_of_missing_repos_aborts_the_eviction(monkeypatch, caplog):
    """「确认查不到」本身也可能是错的,所以还要一道量级闸门。

    `decide` 只保证不把"没问到"当成"没了";它保证不了传进来的 confirmed_missing 是对的。
    GitHub 一次「200 + data 全 null」的事故会被 StarBatch 拆到单名批,每个都记成确认
    查不到 —— 于是整个库被判死刑,而这一步是不可逆的(记录里的 LLM desc 一起没)。
    """
    tracked = {f"a/r{i}" for i in range(1000)}
    harvest = collect.Harvest()
    harvest.missing.update(tracked)                 # 全库"查不到"
    harvest.stars.update({"a/r0": 499})             # 这个是真·掉到门槛下

    monkeypatch.setattr(cron.universe, "evict", lambda names: evicted.update(names))
    evicted: set[str] = set()

    removed = cron.retire(tracked, harvest, star_floor=500)

    assert "a/r0" in evicted, "star 掉到门槛下那部分来自成功测到的值,不该被连累"
    assert len(evicted) == 1, f"闸门没拦住,删了 {len(evicted)} 个"
    assert removed == ["a/r0"]
    assert any("系统性问题" in r.message for r in caplog.records)


def test_a_normal_day_of_metabolism_still_gets_evicted(monkeypatch):
    """闸门不能把正常代谢也拦掉 —— 每天几十个是常态,拦了库就只增不减。"""
    tracked = {f"a/r{i}" for i in range(1000)}
    harvest = collect.Harvest()
    harvest.missing.update({f"a/r{i}" for i in range(5)})

    evicted: set[str] = set()
    monkeypatch.setattr(cron.universe, "evict", lambda names: evicted.update(names))

    cron.retire(tracked, harvest, star_floor=500)
    assert len(evicted) == 5


def test_unasked_repos_are_left_alone():
    """部分失败时,只处置问到的那些。"""
    plan = cron.decide(
        TRACKED, {"a/live": 900, "a/small": 10}, {"a/gone"}, star_floor=500,
    )
    assert "a/unasked" not in plan.all
    assert set(plan.all) == {"a/small", "a/gone"}


def test_repos_not_in_db_are_not_reported():
    """采集结果里混进 DB 没跟踪的名字(比如刚被别的流程删掉),不该出现在淘汰名单里。"""
    plan = cron.decide(
        {"a/live"}, {"a/live": 900, "z/stranger": 3}, {"z/ghost"}, star_floor=500,
    )
    assert plan.all == []


# ──────────────────────────────────────────────────────────
# 采集任务:missing 与 failed 必须分开
# ──────────────────────────────────────────────────────────
def _pool(transport: httpx.MockTransport, tokens: int = 2) -> tuple[TaskPool, httpx.AsyncClient]:
    pool_tokens = gh_tokens.TokenPool([f"t{i}" for i in range(tokens)])
    client = httpx.AsyncClient(transport=transport)
    paces = {collect.SEARCH_TOKEN: gh_tokens.CORE, collect.CORE_TOKEN: gh_tokens.CORE}
    pool = TaskPool(
        lanes={collect.SEARCH_LANE: 2, collect.GRAPHQL_LANE: 2, collect.FREE_LANE: 1},
        leaser=lambda kind: pool_tokens.lease(paces[kind]),
    )
    return pool, client


async def _drain(transport: httpx.MockTransport, *tasks_of) -> None:
    pool, client = _pool(transport)
    async with pool:
        for make in tasks_of:
            pool.submit(make(client))
        await pool.join()
    await client.aclose()


def _graphql(stars_by_name: dict[str, int]):
    """假 GraphQL:名字在表里就给 star,不在就给 null + 一条 NOT_FOUND。

    那条 `errors` 不是装饰:真实 GitHub 对查不到的仓库一定会带 `type: "NOT_FOUND"`,
    而「有没有 NOT_FOUND」正是「真删了」和「服务出故障」的唯一区别。早先这个假响应
    只给 null 不给 errors,于是"故障"和"删除"在测试里长得一模一样。
    """
    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json
        query = _json.loads(request.content)["query"]
        data, errors = {}, []
        for line in query.splitlines():
            if ": repository(" not in line:
                continue
            alias = line.split(":", 1)[0].strip().removeprefix("query{").strip()
            owner = line.split('owner:"', 1)[1].split('"', 1)[0]
            name = line.split('name:"', 1)[1].split('"', 1)[0]
            star = stars_by_name.get(f"{owner}/{name}")
            data[alias] = {"stargazerCount": star} if star is not None else None
            if star is None:
                errors.append({"type": "NOT_FOUND", "path": [alias],
                               "message": f"Could not resolve to a Repository "
                                          f"with the name '{owner}/{name}'."})
        body = {"data": data}
        if errors:
            body["errors"] = errors
        return httpx.Response(200, json=body)
    return httpx.MockTransport(handler)


async def test_star_batch_separates_gone_from_unanswered():
    sink = collect.Harvest()
    names = ["a/one", "a/two", "a/gone"]
    transport = _graphql({"a/one": 100, "a/two": 200})

    await _drain(transport, lambda c: collect.StarBatch(sink, c, names))

    assert sink.stars == {"a/one": 100, "a/two": 200}
    assert sink.missing == {"a/gone"}, "响应回来了、只是没这个仓库 → 确认查不到"
    assert sink.failed == set()


async def test_a_batch_that_never_answers_goes_to_failed_not_missing():
    """一直限流 → 最终 failed,而且**会结束**。

    两件事各自要命:错记成 missing,淘汰会把这批活仓库全删了;不设上限,每日任务会一直
    转到 Actions 六小时超时,既不落盘也不报错(见 `Task.max_rate_limits`)。
    """
    sink = collect.Harvest()
    names = ["a/one", "a/two"]

    class Impatient(collect.StarBatch):
        max_rate_limits = 2         # 真值是 20,那要等 20 轮冷却,测不动

    def always_limited(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, headers={"Retry-After": "1"}, text="rate limited")

    await asyncio.wait_for(
        _drain(httpx.MockTransport(always_limited),
               lambda c: Impatient(sink, c, names)),
        timeout=30,
    )

    assert sink.failed == set(names)
    assert sink.missing == set(), "没问到的绝不能进淘汰名单"


async def test_a_degenerate_all_null_batch_splits_instead_of_evicting():
    """整批 null 是查询过大的退化,不是「这批仓库都没了」—— 必须拆开重问。

    没有这条,一次退化就能让整批仓库同时被判死刑。
    """
    sink = collect.Harvest()
    names = [f"a/r{i}" for i in range(8)]
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json
        query = _json.loads(request.content)["query"]
        aliases = [ln for ln in query.splitlines() if ": repository(" in ln]
        seen.append(len(aliases))
        if len(aliases) > 2:                       # 大于 2 个别名就"退化"成全 null
            return httpx.Response(200, json={"data": {
                f"r{i}": None for i in range(len(aliases))}})
        return httpx.Response(
            200, json={"data": {f"r{i}": {"stargazerCount": 700}
                                for i in range(len(aliases))}})

    await _drain(httpx.MockTransport(handler),
                       lambda c: collect.StarBatch(sink, c, names))

    assert len(sink.stars) == 8, "拆分后应该一个不漏"
    assert sink.missing == set(), "退化响应绝不能被当成「查不到」"
    assert max(seen) == 8 and min(seen) <= 2, "应该确实发生了对半拆分"


async def test_a_single_repo_that_stays_null_is_really_gone():
    """拆到只剩一个、且 GitHub 明说 NOT_FOUND —— 那它就是真的没了。"""
    sink = collect.Harvest()
    await asyncio.wait_for(
        _drain(_graphql({}), lambda c: collect.StarBatch(sink, c, ["a/ghost"])),
        timeout=10,
    )
    assert sink.missing == {"a/ghost"}


async def test_a_lone_null_without_not_found_counts_as_unanswered_not_as_deleted():
    """单名批的 null 不带 NOT_FOUND 时,必须记成「没问到」而不是「没了」。

    这是清库那条路的入口:`StarBatch` 会把整批 null 一路对半拆到单名为止,所以 GitHub
    一次「200 + data 全 null」的事故最终**全部**以单名批落地。早先单名批一律记 missing,
    于是几万个活仓库变成"确认查不到",下一步淘汰就把库删空 —— 而记录里连 LLM 写过的
    desc 一起没。

    也顺便钉住不能无限拆:单名批返回 None 会让 `1 // 2 == 0` 拆出空批加原样批。
    """
    sink = collect.Harvest()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {"r0": None}})

    await asyncio.wait_for(
        _drain(httpx.MockTransport(handler),
               lambda c: collect.StarBatch(sink, c, ["a/ghost"])),
        timeout=10,
    )
    assert sink.missing == set(), "没问到被当成了没了 —— 这条路通向清库"
    assert sink.failed == {"a/ghost"}


async def test_an_incident_shaped_error_also_counts_as_unanswered():
    """RATE_LIMITED / INTERNAL 之类的 errors 同样不足以断定仓库没了。"""
    sink = collect.Harvest()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "data": {"r0": None},
            "errors": [{"type": "INTERNAL", "message": "something went wrong"}]})

    await asyncio.wait_for(
        _drain(httpx.MockTransport(handler),
               lambda c: collect.StarBatch(sink, c, ["a/ghost"])),
        timeout=10,
    )
    assert sink.missing == set()
    assert sink.failed == {"a/ghost"}


# ──────────────────────────────────────────────────────────
# 收集任务
# ──────────────────────────────────────────────────────────
def _search(pages: dict[int, list[str]], total: int = 0):
    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params.get("page", 1))
        if request.url.params.get("per_page") == "1":
            return httpx.Response(200, json={"total_count": total, "items": []})
        names = pages.get(page, [])
        return httpx.Response(200, json={
            "total_count": total,
            "items": [{"full_name": n, "stargazers_count": 600} for n in names],
        })
    return httpx.MockTransport(handler)


async def test_keyword_search_follows_full_pages_and_stops_on_a_short_one():
    sink = collect.Discovered()
    full = [f"a/r{i}" for i in range(collect.PER_PAGE)]
    transport = _search({1: full, 2: ["a/last"], 3: ["a/never"]})

    await _drain(transport, lambda c: collect.KeywordPage(sink, c, "agent", 500))

    assert len(sink.repos) == collect.PER_PAGE + 1
    assert "a/never" not in sink.repos, "第 2 页没满就该收手,不再翻第 3 页"


async def test_search_stops_at_the_thousand_result_ceiling():
    """Search 只给前 1000 条,翻到第 11 页是 422。别去撞那堵墙。"""
    sink = collect.Discovered()
    full = {p: [f"a/p{p}n{i}" for i in range(collect.PER_PAGE)] for p in range(1, 13)}
    transport = _search(full)

    await _drain(transport, lambda c: collect.KeywordPage(sink, c, "agent", 500))

    assert len(sink.repos) == collect.MAX_PAGES * collect.PER_PAGE


async def test_a_fat_star_range_splits_before_it_is_scanned():
    """命中超过 1000 就得先劈开区间,否则超出的部分永远拿不到。"""
    sink = collect.Discovered()
    scanned: list[tuple[int, int]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        query = request.url.params["q"]
        lo, hi = (int(x) for x in query.removeprefix("stars:").split(".."))
        if request.url.params.get("per_page") == "1":
            return httpx.Response(200, json={"total_count": 100 if hi - lo < 250 else 5000})
        scanned.append((lo, hi))
        return httpx.Response(200, json={"items": [
            {"full_name": f"a/{lo}", "stargazers_count": lo}]})

    await _drain(httpx.MockTransport(handler),
                       lambda c: collect.SegmentProbe(sink, c, 500, 1500))

    assert scanned, "拆完之后必须真的去扫"
    assert all(hi - lo < 250 for lo, hi in scanned), "还装不下就不该开扫"
    covered = sorted(scanned)
    assert covered[0][0] == 500 and covered[-1][1] == 1500, "拆分不能漏掉边界"
    for (_, prev_hi), (next_lo, _) in zip(covered, covered[1:]):
        assert next_lo == prev_hi + 1, f"{prev_hi} 和 {next_lo} 之间漏了一段"


async def test_an_empty_star_range_costs_one_request():
    sink = collect.Discovered()
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json={"total_count": 0, "items": []})

    await _drain(httpx.MockTransport(handler),
                       lambda c: collect.SegmentProbe(sink, c, 500, 1500))
    assert len(calls) == 1, "空段探一次就该收手,不该再拆也不该去扫"


async def test_every_keyword_gets_credited_for_a_repo_it_returned():
    """一个仓库被两个关键词搜到,两边都要记上 —— 「谁先跑完」取决于并发,不能拿来记账。

    这份账是砍关键词的唯一依据:按先到先得记的话,同一个词今天 0 个明天 50 个,量不出
    任何东西,只会把还有用的词误砍掉。
    """
    sink = collect.Discovered()
    transport = httpx.MockTransport(
        lambda r: httpx.Response(200, json={"items": [{"full_name": "a/shared"}]}))

    for word in ("agent", "llm"):
        await _drain(transport, lambda c, w=word: collect.KeywordPage(sink, c, w, 500))

    assert len(sink.repos) == 1, "收集箱本身仍然去重"
    assert sink.sources == {"kw:agent": {"a/shared"}, "kw:llm": {"a/shared"}}


async def test_paging_does_not_split_one_keyword_into_several_sources():
    sink = collect.Discovered()
    full = [f"a/r{i}" for i in range(collect.PER_PAGE)]
    await _drain(_search({1: full, 2: ["a/last"]}),
                 lambda c: collect.KeywordPage(sink, c, "agent", 500))
    assert list(sink.sources) == ["kw:agent"]
    assert len(sink.sources["kw:agent"]) == collect.PER_PAGE + 1


async def test_a_failed_search_is_recorded_but_does_not_stop_the_others():
    """一个关键词挂了不能让整轮发现失败 —— DB 是累积的,今天漏的明天补。"""
    sink = collect.Discovered()

    def handler(request: httpx.Request) -> httpx.Response:
        if "bad" in request.url.params["q"]:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json={"items": [{"full_name": "a/good"}]})

    await asyncio.wait_for(_drain(
        httpx.MockTransport(handler),
        lambda c: collect.KeywordPage(sink, c, "bad", 500),
        lambda c: collect.KeywordPage(sink, c, "ok", 500),
    ), timeout=10)

    assert "a/good" in sink.repos
    assert len(sink.failures) == 1


# ──────────────────────────────────────────────────────────
# 出站客户端:HTTP 状态 → 异常(全项目唯一的翻译处)
# ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("status,expected", [
    (401, TokenInvalidError),
    (403, RateLimitError),
    (429, RateLimitError),
    (500, RetryableError),
    (502, RetryableError),
    (404, RuntimeError),      # 4xx:请求本身有问题,重试无用
])
def test_http_status_maps_to_the_right_exception(status, expected):
    resp = httpx.Response(status, text="x", request=httpx.Request("GET", "https://x"))
    with pytest.raises(expected):
        gh._classify(resp)


@pytest.mark.parametrize("headers, body, expected", [
    ({"x-ratelimit-remaining": "0"}, "{}", "主限额耗尽"),
    ({"retry-after": "60"}, '{"message": "You have exceeded a secondary rate limit"}',
     "二级限流"),
    ({"x-ratelimit-remaining": "0"},
     '{"message": "You have exceeded a secondary rate limit"}', "二级限流"),
    ({}, "{}", "未分类"),
])
def test_the_two_kinds_of_rate_limit_are_told_apart(headers, body, expected):
    """两者处置不同:主限额只是这个 token 这一分钟用完了,二级限流按认证身份计、要整体降速。

    分不清就只能一律冷却单个 token,而这正是「12 张 token 每 65 秒集体撞一次墙」时
    看不出病因的原因。
    """
    resp = httpx.Response(403, headers=headers, text=body,
                          request=httpx.Request("GET", "https://x"))
    assert expected in gh._limit_reason(resp)


def test_retry_after_wins_over_the_ratelimit_reset_header():
    """二级限流会同时带两个头,且说的不是一回事。

    X-RateLimit-Reset 是**主**限额那一分钟的窗口(与这次被拒无关),实测比 Retry-After 早
    22 秒。先读它就会提前重试、撞进没结束的罚时 —— 正是「每 65 秒全体撞一次墙」的成因。
    """
    now = time.time()
    resp = httpx.Response(403, headers={
        "retry-after": "60",
        "x-ratelimit-reset": str(int(now + 38)),
    }, text="{}", request=httpx.Request("GET", "https://x"))
    assert gh._reset_at(resp.headers) - now > 50


async def test_a_422_page_means_no_more_results_not_an_error():
    """翻过 1000 条上限时 Search 返回 422。那不是故障,是到底了。"""
    sink = collect.Discovered()
    transport = httpx.MockTransport(lambda r: httpx.Response(422, json={"message": "only 1000"}))
    await _drain(transport, lambda c: collect.KeywordPage(sink, c, "agent", 500))
    assert sink.failures == [], "422 不该被记成失败"


async def test_graphql_rate_limit_hides_inside_a_200():
    """GraphQL 的限流不走 403,而是 200 + errors 里写 RATE_LIMITED。"""
    async def go():
        transport = httpx.MockTransport(lambda r: httpx.Response(
            200, json={"errors": [{"type": "RATE_LIMITED"}]}))
        async with httpx.AsyncClient(transport=transport) as c:
            pool = gh_tokens.TokenPool(["t0"])
            async with pool.lease(gh_tokens.CORE) as lease:
                with pytest.raises(RateLimitError):
                    await gh.fetch_stars(c, lease, ["a/b"])

    await asyncio.wait_for(go(), timeout=10)


def test_repo_names_with_quotes_do_not_break_the_query():
    """仓库名进 GraphQL 前要转义,否则一个引号能把整批查询拼坏。"""
    query = gh._star_query(['ow"ner/re\\po'])
    assert '\\"' in query or "\\\\" in query
    assert query.count("repository(") == 1


# ──────────────────────────────────────────────────────────
# Trending 解析
# ──────────────────────────────────────────────────────────
_ARTICLE = """
<article class="Box-row">
  <h2><a href="/octocat/hello-world">octocat / hello-world</a></h2>
  <p class="col-9">Some <em>description</em> here</p>
  <span itemprop="programmingLanguage">Python</span>
  <a href="/octocat/hello-world/stargazers"></svg> 12,345 </a>
  <a href="/octocat/hello-world/forks"></svg> 678 </a>
  <span>90 stars this week</span>
</article>
"""


def test_trending_parses_a_row():
    result = gh_trending.parse(_ARTICLE, "weekly")
    assert len(result.repos) == 1
    repo = result.repos[0]
    assert repo["full_name"] == "octocat/hello-world"
    assert repo["star"] == 12345
    assert repo["forks"] == 678
    assert repo["stars_today"] == 90
    assert repo["description"] == "Some description here", "HTML 标签要剥掉"
    assert repo["language"] == "Python"


def test_a_short_list_is_not_a_broken_parser():
    """榜单本来就短 vs 解析器坏了 —— 长得一样,但一个不用管、一个要修代码。"""
    healthy = gh_trending.parse(_ARTICLE, "weekly")
    assert not healthy.looks_broken

    changed_layout = _ARTICLE.replace("<h2>", "<h9>").replace("</h2>", "</h9>")
    broken = gh_trending.parse(changed_layout, "weekly")
    assert broken.articles == 1 and broken.repos == []
    assert broken.looks_broken, "有条目却一个都没解析出来 = 页面结构变了"


def test_trending_ignores_period_it_does_not_know():
    assert gh_trending.parse(_ARTICLE, "yearly").repos[0]["stars_today"] == 0
