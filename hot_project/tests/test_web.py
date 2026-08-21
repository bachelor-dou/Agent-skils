"""网页端:HTTP 路由、会话池、安全中间件、报告渲染。

真起一个 app 打真实的 HTTP 请求(`TestClient` 不开端口,直接走 ASGI)。这一层出问题
不像业务逻辑那样有堆栈:404、路径穿越放行,都是「看起来正常运行」的故障。
"""

import time
from datetime import date

import pytest
from fastapi.testclient import TestClient

from hot_project import api_server, config
from hot_project.service import favorites as favorite_service
from hot_project.web import render, security, sessions

REPORT = """# GitHub 热门项目 — 2026-07-30

> 共 1 个项目 | 增长统计窗口: 7 天

## 1. openai/whisper

链接: https://github.com/openai/whisper

- 创建时间: 2022-09-16
- 主语言: Python
- 总 Star: 82,000
- 近7天增长: +1,500

### 项目定位与用途

语音识别模型。
"""


@pytest.fixture
def report_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORT_DIR", tmp_path)
    (tmp_path / "2026-07-30.md").write_text(REPORT, encoding="utf-8")
    return tmp_path


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(security, "_hits", {})
    return TestClient(api_server.app)


# ── 路由 ──────────────────────────────────────────────────────────

def test_the_service_reports_its_own_state(client):
    body = client.get("/api/status").json()
    assert body["status"] == "running"
    assert body["min_star"] == config.MIN_STAR


def test_the_chat_page_is_served(client):
    resp = client.get("/chat")
    assert resp.status_code == 200 and "<html" in resp.text.lower()


def test_pages_are_never_cached(client):
    """页面里注入了资源版本号。被缓存住的话,发版后用户拿到的是新 HTML 配旧 JS。"""
    assert "no-store" in client.get("/chat").headers["cache-control"]


def test_reports_are_listed_newest_first(client, report_dir):
    (report_dir / "2026-07-29.md").write_text(REPORT, encoding="utf-8")
    names = [r["name"] for r in client.get("/api/reports").json()["reports"]]
    assert names[0] == "2026-07-29.md" or names[0] == "2026-07-30.md"
    assert len(names) == 2


def test_the_listed_time_is_a_string_the_browser_parses_correctly(client, report_dir):
    """前端 `new Date(v)` 把裸数字当**毫秒**,给 st_mtime(秒)不会报错,只会把
    2026 年静悄悄显示成 1970-01-21 —— 页面照样打开,没人会去点开看那行小字。
    """
    item = client.get("/api/reports").json()["reports"][0]
    assert isinstance(item["modified_at"], str)
    assert item["modified_at"].startswith(str(date.today().year))


def test_a_report_renders_to_html(client, report_dir):
    resp = client.get("/api/reports/2026-07-30.md/html")
    assert resp.status_code == 200
    assert "openai/whisper" in resp.text
    assert "82,000" in resp.text


def test_a_report_can_be_fetched_as_markdown(client, report_dir):
    body = client.get("/api/reports/2026-07-30.md").json()
    assert body["content"].startswith("# GitHub 热门项目")


def test_an_unknown_report_is_a_404_not_a_500(client, report_dir):
    assert client.get("/api/reports/nope.md").status_code == 404


@pytest.mark.parametrize("evil", ["../config.py", "..%2f..%2fconfig.py", "a/b.md"])
def test_a_report_name_can_never_escape_the_directory(client, report_dir, evil):
    """名字直接来自 URL。放行一个就等于把整个文件系统开出去。"""
    assert client.get(f"/api/reports/{evil}").status_code in (404, 400)


def test_deleting_a_report_removes_the_file(client, report_dir):
    assert client.delete("/api/reports/2026-07-30.md").status_code == 200
    assert not (report_dir / "2026-07-30.md").exists()


def test_star_trend_rejects_a_malformed_repo_name(client):
    assert client.get("/api/star-trend", params={"repo": "not a repo"}).status_code == 400


def test_favorites_reject_a_malformed_user_id(client):
    assert client.get("/api/favorites", params={"user_id": "!!"}).status_code == 400


def test_a_favorite_post_answers_with_the_same_authoritative_list_as_get(
    client, monkeypatch, tmp_path,
):
    """前端拿 POST 响应直接对账、省掉一次 GET,前提是两者同形:
    条目带上榜统计、顶层带 report_total。这条钉的就是这份前后端契约。
    """
    monkeypatch.setattr(config, "FAVORITES_PATH", tmp_path / "favorites.json")
    posted = client.post("/api/favorites", json={
        "user_id": "tester", "repo": "owner/name",
        "action": "add", "short_desc": ""}).json()
    assert posted == client.get("/api/favorites",
                                params={"user_id": "tester"}).json()
    assert "report_total" in posted
    assert "report_count" in posted["favorites"][0]


def test_changing_the_category_neither_costs_an_llm_call_nor_eats_the_summary(
    client, monkeypatch, tmp_path,
):
    """只改分类时概要必须原样留着。

    重算一次要一次 LLM 调用(收藏栏拖一下就来一次),而且会把用户手写的概要冲掉 ——
    概要是可以手工编辑的,冲掉就没了。
    """
    monkeypatch.setattr(config, "FAVORITES_PATH", tmp_path / "favorites.json")
    generated = []
    monkeypatch.setattr(favorite_service, "short_desc",
                        lambda *a, **k: generated.append(a) or "机器写的")

    add = {"user_id": "tester", "repo": "owner/name", "action": "add"}
    client.post("/api/favorites", json=dict(add, short_desc="我手写的概要"))
    client.post("/api/favorites", json=dict(add, category="娱乐"))

    item = client.get("/api/favorites", params={"user_id": "tester"}).json()["favorites"][0]
    assert item["category"] == "娱乐"
    assert item["short_desc"] == "我手写的概要"
    assert generated == [], "改个分类而已,不该重新生成概要"

    client.post("/api/favorites", json=dict(add, short_desc=""))
    client.post("/api/favorites", json=dict(add, category="效率"))
    item = client.get("/api/favorites", params={"user_id": "tester"}).json()["favorites"][0]
    assert item["category"] == "效率"
    assert item["short_desc"] == ""
    assert generated == []


def test_refreshing_a_card_also_refreshes_its_star_and_growth(client, monkeypatch):
    """报告里的数字是出榜那天的。只换介绍不换数字,刷完卡片上仍是一份过期数据。

    顺带钉住「和 agent 同一份算法」:走的是 `live_growth`,所以报的是实际跨度(6 天),
    不是请求的 7 天。
    """
    from hot_project.infra.data_access import snapshots
    from hot_project.provider.github import client as github
    from hot_project.service import report

    class _GH:
        usable = True

        def info(self, name):
            return {"stargazers_count": 5200, "created_at": "2020-01-01T00:00:00Z"}

    monkeypatch.setattr(github, "shared", lambda: _GH())
    monkeypatch.setattr(report, "regenerate", lambda name, gh: "项目定位与用途: 干活的。")
    monkeypatch.setattr(snapshots, "earliest_in_window",
                        lambda days, today=None: snapshots.Baseline(
                            {"owner/name": 4000}, {"owner/name": 6}, date(2026, 7, 29), 6))

    body = client.post("/api/repo-describe", json={"repo": "owner/name"}).json()
    assert body["sections"], "介绍照旧要回来"
    assert body["star"] == 5200
    assert body["growth"] == 1200                                  # 实时 star 减最早那份快照
    assert body["growth_calc_days"] == 6                           # 实际跨度,不是请求的 7


def test_refresh_finds_the_baseline_of_a_renamed_repo_by_id(client, monkeypatch):
    """基线按 databaseId 键存,改过名的仓库只有按 id 才查得到。

    曾经的真 bug:刷新按钮只按名字查基线,改过名的仓库增长静默变 null,
    看起来像「缺数据」。键规则的唯一归属在 `Baseline.get`。
    """
    from hot_project.infra.data_access import snapshots
    from hot_project.provider.github import client as github
    from hot_project.service import report

    class _GH:
        usable = True

        def info(self, name):
            return {"stargazers_count": 5200, "id": 777,
                    "created_at": "2020-01-01T00:00:00Z"}

    monkeypatch.setattr(github, "shared", lambda: _GH())
    monkeypatch.setattr(report, "regenerate", lambda name, gh: "项目定位与用途: 干活的。")
    monkeypatch.setattr(snapshots, "earliest_in_window",
                        lambda days, today=None: snapshots.Baseline(
                            {777: 4000}, {777: 6}, date(2026, 7, 29), 6))

    body = client.post("/api/repo-describe", json={"repo": "newowner/newname"}).json()
    assert body["growth"] == 1200, "基线在 id 键下,必须按 id 查到"
    assert body["growth_calc_days"] == 6


def test_chat_page_injects_the_real_session_ttl(client):
    """前端的会话过期判断读页面注入的 TTL,必须和 sessions.TTL_SECONDS 同源。

    曾经的真缺陷:chat.js 里的占位符从没人替换(JS 是静态直出的),
    TTL 静默退化成硬编码 3600,服务端改了前端不知道。
    """
    html = client.get("/").text
    assert f"window.SESSION_TTL_SECONDS = {sessions.TTL_SECONDS};" in html
    assert "__SESSION_TTL_SECONDS__" not in html, "占位符残留说明注入断了"


def test_deleting_a_session_that_never_existed_is_a_404(client):
    assert client.delete("/api/sessions/nope").status_code == 404


# ── 安全 ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("path", ["/.env", "/v1/chat/completions", "/admin", "/.git/config"])
def test_scanner_paths_get_404_not_403(client, path):
    """403 等于告诉对方「这里有东西」。404 什么都没说。"""
    assert client.get(path).status_code == 404


def test_too_many_requests_from_one_ip_are_throttled(client, monkeypatch):
    monkeypatch.setattr(security, "RATE_LIMIT", 3)
    monkeypatch.setattr(security, "_hits", {})
    codes = [client.get("/api/status").status_code for _ in range(5)]
    assert 429 in codes


def test_the_window_slides_so_a_slow_client_is_never_blocked(monkeypatch):
    monkeypatch.setattr(security, "RATE_LIMIT", 2)
    monkeypatch.setattr(security, "RATE_WINDOW", 0.05)
    monkeypatch.setattr(security, "_hits", {})
    assert not security.rate_limited("1.2.3.4")
    assert not security.rate_limited("1.2.3.4")
    assert security.rate_limited("1.2.3.4")
    time.sleep(0.06)
    assert not security.rate_limited("1.2.3.4")


def test_one_noisy_ip_does_not_throttle_everyone_else(monkeypatch):
    monkeypatch.setattr(security, "RATE_LIMIT", 1)
    monkeypatch.setattr(security, "_hits", {})
    assert not security.rate_limited("1.1.1.1")
    assert security.rate_limited("1.1.1.1")
    assert not security.rate_limited("2.2.2.2")


def test_the_real_ip_is_taken_from_the_proxy_header():
    class _Req:
        headers = {"x-forwarded-for": "203.0.113.9, 10.0.0.1"}
        client = None
    assert security.client_ip(_Req()) == "203.0.113.9"


def test_a_rate_limited_client_cannot_open_a_websocket_either(client, monkeypatch):
    """安全中间件对 WebSocket **不生效** —— starlette 的 BaseHTTPMiddleware 见到非 http
    的 scope 就直接放行。于是 /ws/chat 曾是唯一没有限速的入口,而它恰好是唯一真会驱动
    agent 花钱的入口。HTTP 那几十条测试一条都抓不到这个,因为对 HTTP 它是好的。
    """
    monkeypatch.setattr(security, "RATE_LIMIT", 0)
    monkeypatch.setattr(security, "_hits", {})
    with pytest.raises(Exception):          # 被 close(1008) 掉,连不上
        with client.websocket_connect("/ws/chat/s1"):
            pass


def test_spoofed_forwarded_headers_do_not_grow_the_table_forever(monkeypatch):
    """`client_ip` 认 X-Forwarded-For,而那是客户端给的 —— 每换一个值就是一个新键。

    一个扫描器就能把这张表撑到几百万条。窗口空了就得删键:空 deque 不携带任何信息。
    """
    monkeypatch.setattr(security, "RATE_WINDOW", 0.01)
    monkeypatch.setattr(security, "_SWEEP_THRESHOLD", 4)
    monkeypatch.setattr(security, "_hits", {})
    for n in range(50):
        security.rate_limited(f"10.0.0.{n}")
    time.sleep(0.02)                        # 让所有窗口过期
    security.rate_limited("10.0.0.99")      # 这一次顺带触发清扫
    assert len(security._hits) < 10, f"表里还剩 {len(security._hits)} 条"


# ── 会话池 ────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_sessions(monkeypatch):
    monkeypatch.setattr(sessions, "_agents", {})
    monkeypatch.setattr(sessions, "_pending", {})
    monkeypatch.setattr(sessions, "build", lambda: object())


def test_the_same_session_id_keeps_the_same_agent():
    assert sessions.get("a") is sessions.get("a")


def test_different_sessions_do_not_share_an_agent():
    assert sessions.get("a") is not sessions.get("b")


def test_an_expired_session_is_swept_away(monkeypatch):
    first = sessions.get("a")
    monkeypatch.setattr(sessions, "TTL_SECONDS", -1)
    assert sessions.get("a") is not first


def test_the_oldest_session_is_evicted_at_the_cap(monkeypatch):
    """没有上限的话,爬虫拿随机 session_id 打几万次就能把内存撑爆。"""
    monkeypatch.setattr(sessions, "MAX_SESSIONS", 3)
    for name in "abc":
        sessions.get(name)
        time.sleep(0.001)
    sessions.get("d")
    assert sessions.count() == 3
    assert "a" not in sessions._agents


def test_touching_a_session_keeps_it_from_being_the_eviction_victim(monkeypatch):
    monkeypatch.setattr(sessions, "MAX_SESSIONS", 2)
    sessions.get("a")
    time.sleep(0.001)
    sessions.get("b")
    time.sleep(0.001)
    sessions.get("a")           # a 被用了一下,现在 b 才是最旧的
    sessions.get("c")
    assert "a" in sessions._agents and "b" not in sessions._agents


def test_a_reply_stashed_while_offline_is_delivered_once():
    """手机切后台会断连,而 agent 还在跑。跑完发不出去就存着,重连补推。"""
    sessions.stash("a", "回复")
    assert sessions.take("a") == ["回复"]
    assert sessions.take("a") == []


def test_dropping_a_session_also_drops_its_stashed_replies():
    sessions.get("a")
    sessions.stash("a", "回复")
    sessions.drop("a")
    assert sessions.take("a") == []


# ── 执行锁 ────────────────────────────────────────────────────────

def test_the_tool_lock_is_released_after_the_block():
    with sessions.hold_tool_lock("测试"):
        assert sessions.tool_lock.locked()
    assert not sessions.tool_lock.locked()


def test_a_held_lock_raises_busy_after_the_timeout(monkeypatch):
    monkeypatch.setattr(sessions, "TOOL_LOCK_TIMEOUT", 0.05)
    assert sessions.tool_lock.acquire()
    try:
        with pytest.raises(sessions.Busy):
            with sessions.hold_tool_lock("测试"):
                pass
    finally:
        sessions.tool_lock.release()


def test_the_lock_survives_an_exception_in_the_block():
    """漏了 release 的锁是永久的:下一个请求会等满超时然后 503,而且没有堆栈可查。"""
    with pytest.raises(RuntimeError):
        with sessions.hold_tool_lock("测试"):
            raise RuntimeError("炸")
    assert not sessions.tool_lock.locked()


def test_the_thinking_level_travels_from_the_request_to_the_agent(client, monkeypatch):
    """网页选的档位要真的落到这一轮对话上。这一跳只有关键字名对不上的风险,而拼错了
    没人报错 —— 要等用户发一条消息才 TypeError,所以钉在这里。"""
    seen = {}

    class _Agent:
        def chat(self, message, **kwargs):
            seen.update(kwargs)
            return "好"

    monkeypatch.setattr(sessions, "get", lambda _sid: _Agent())
    resp = client.post("/api/chat", json={"message": "hi", "thinking": "max"})
    assert resp.status_code == 200
    assert seen["effort"] == "max"


def test_the_two_transports_normalize_their_options_the_same_way():
    """HTTP 给的是 body(键都在,值可能是空串),WS 给的是 query(没选的键直接缺席)。
    两种形状必须规范化成同一个值,否则同一个选择走不同的路会有不同的行为。

    空串在这里就消掉:它是「用户没点过那个开关」的产物,不该流到下游被当成一个模型 id。
    """
    import inspect

    from hot_project.agent.loop import Agent
    from hot_project.infra import llm as llm_module
    from hot_project.web import chat_options

    from_http = chat_options.parse(
        {"user_id": "u1", "model": "azure01", "lite": "", "thinking": "max"})
    from_ws = chat_options.parse({"user_id": "u1", "model": "azure01", "thinking": "max"})
    assert from_http == from_ws              # 缺键和空串必须等价
    blank = chat_options.parse({"model": "", "lite": "", "thinking": ""})
    assert blank.model is None and blank.lite is None    # 空串不是一个模型 id

    assert chat_options.parse({}).effort == llm_module.EFFORT_DEFAULT
    assert chat_options.parse({"thinking": "错字"}).effort == llm_module.EFFORT_DEFAULT
    assert chat_options.parse({"thinking": "off"}).effort == llm_module.EFFORT_OFF

    # kwargs() 是直接展开给 Agent.chat 的,字段名对不上要等用户真发一条消息才 TypeError
    assert set(chat_options.parse({}).kwargs()) <= set(inspect.signature(Agent.chat).parameters)


def test_the_websocket_carries_the_options_into_the_conversation(client, monkeypatch):
    """浏览器默认走 WebSocket,HTTP 只是 socket 不可用时的回退 —— 而主路径此前一条断言都没有。
    连接时 query 里的选项要一路落到这一轮对话上。"""
    seen = {}

    class _Agent:
        def chat(self, message, **kwargs):
            seen.update(kwargs)
            return "好"

    monkeypatch.setattr(sessions, "get", lambda _sid: _Agent())
    with client.websocket_connect("/ws/chat/s1?thinking=max&model=azure01") as ws:
        ws.send_text("hi")
        for _ in range(20):                  # 进度帧可能先到,取到最终帧就停
            if ws.receive_json().get("type") in ("reply", "error"):
                break
    assert seen["effort"] == "max"
    assert seen["model"] == "azure01"


def test_the_model_list_tells_the_page_who_can_think_deeper(client, monkeypatch):
    """网页靠这个字段决定要不要画「更深思考」开关,空串就是不画。
    顺带钉住这个列表永远不带出 key —— 它是唯一把平台信息交给浏览器的地方。"""
    from hot_project.infra.llm import Api, LLMClient
    fake = LLMClient([
        Api(id="p1", label="P1", backend="azure", url="https://p1.test",
            model="m", key="secret-key-must-not-leak"),
        Api(id="p2", label="P2", backend="anthropic", url="https://p2.test",
            model="m", key="secret-key-must-not-leak"),
        Api(id="p3", label="P3", backend="foundry", url="https://p3.test",
            model="gpt-5.6-terra", key="secret-key-must-not-leak"),
    ])
    monkeypatch.setattr(api_server.llm, "get", lambda: fake)
    resp = client.get("/api/models")
    models = {m["id"]: m for m in resp.json()["models"]}
    assert models["p1"]["thinking_deeper"] == "max"
    assert models["p1"]["thinking_deeper_label"] == "xhigh"  # 显示用平台原名,发的仍是档位名
    assert models["p2"]["thinking_deeper"] == ""        # 没登记的后端不给这组选项
    assert models["p3"]["thinking_deeper"] == ""        # 5.5/5.6 对话不能思考,同样不给
    assert "secret-key-must-not-leak" not in resp.text


def test_a_busy_server_answers_chat_with_503(client, monkeypatch):
    """钉住 Busy → 503 的翻译层真的挂在 app 上,而不只是 guard 自己会抛。"""
    monkeypatch.setattr(sessions, "TOOL_LOCK_TIMEOUT", 0.05)
    assert sessions.tool_lock.acquire()
    try:
        resp = client.post("/api/chat", json={"message": "hi"})
    finally:
        sessions.tool_lock.release()
    assert resp.status_code == 503


# ── 渲染 ──────────────────────────────────────────────────────────

def test_a_javascript_url_in_a_report_is_defused(report_dir):
    html = render.report_html("x.md", "# 记\n\n[点我](javascript:alert(1))\n")
    assert "javascript:alert" not in html


def test_raw_script_tags_in_a_report_never_reach_the_page(report_dir):
    html = render.report_html("x.md", "# 记\n\n<script>alert(1)</script>\n")
    assert "<script>alert(1)</script>" not in html


@pytest.mark.parametrize("payload", [
    "<scr<script>ipt src=http://evil.example/x.js>",
    "<scr<scr<script>ipt>ipt src=http://evil.example/x.js>",
    "<a href=javascript:alert(1)>click</a>",
    "<img SRC=javascript:alert(1)>",
    '<a href="javascript:alert(1)">quoted</a>',
    "<a hREf=JaVaScRiPt:alert(1)>case</a>",
    '<a href=" java&#09;script:alert(1)">obfuscated</a>',
    "<img src=x onerror=alert(1)>",
])
def test_no_known_payload_survives_into_the_rendered_report(report_dir, payload):
    body = "\n".join(line for line in render.report_html("x.md", payload).splitlines()
                     if "/web/" not in line)      # 模板自己的 <script src="/web/..."> 不算
    lowered = body.lower()
    assert "evil.example" not in lowered
    assert "javascript:" not in lowered
    assert "onerror" not in lowered


def test_defusing_payloads_does_not_break_ordinary_links(report_dir):
    """清洗过头和清洗不足一样是 bug —— 报告里的项目链接是它唯一的用处。"""
    html = render.report_html("x.md", "[仓库](https://github.com/a/b)\n\n![图](/web/x.png)\n")
    assert 'href="https://github.com/a/b"' in html
    assert '/web/x.png' in html


def test_a_web_asset_name_cannot_escape_the_web_directory():
    with pytest.raises(OSError):
        render.asset_text("../config.py")


def test_client_side_limits_match_the_server_constants():
    """JS 里的限制是服务端常量的手抄副本(浏览器拿不到 Python 值),这里钉住不许漂移:
    岔开后前端会放行服务端要拒的输入,用户填完才报错。"""
    from hot_project.infra.data_access import favorites as store

    favorites_js = (config.WEB_DIR / "favorites.js").read_text(encoding="utf-8")
    user_js = (config.WEB_DIR / "user.js").read_text(encoding="utf-8")

    assert f"MAX_TAG_LEN = {store.MAX_CATEGORY_LEN};" in favorites_js
    assert f"DESC_MAX = {favorite_service.FAVORITE_DESC_MAX};" in favorites_js
    assert store.USER_ID_RE.pattern in user_js
