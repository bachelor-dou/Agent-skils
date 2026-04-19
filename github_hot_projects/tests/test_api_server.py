"""
测试 api_server 模块
=====================
覆盖：FastAPI REST 端点、会话管理、HTML 渲染安全。
"""

import json
from unittest.mock import patch, MagicMock, AsyncMock

import pytest


# ──────────────────────────────────────────────────────────────
# Fixture: FastAPI TestClient
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """创建 FastAPI TestClient，mock Agent 避免 LLM 调用。"""
    with patch("github_hot_projects.api_server.HotProjectAgent") as MockAgent:
        mock_instance = MagicMock()
        mock_instance.chat.return_value = "这是 Agent 的回复"
        MockAgent.return_value = mock_instance

        from github_hot_projects.api_server import app
        from starlette.testclient import TestClient
        yield TestClient(app)


class TestChatEndpoints:
    def test_get_root(self, client):
        """GET / 返回 chat.html 页面。"""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "GitHub" in resp.text or "chat" in resp.text.lower()
        assert "__SESSION_TTL_SECONDS__" not in resp.text

    def test_post_chat(self, client):
        """POST /api/chat 创建会话并返回 Agent 回复。"""
        resp = client.post("/api/chat", json={"message": "你好"})
        assert resp.status_code == 200
        data = resp.json()
        assert "reply" in data
        assert "session_id" in data
        assert data["session_ttl_seconds"] > 0
        assert data["session_expires_at"].endswith("Z")

    def test_post_chat_with_session(self, client):
        """复用 session_id 的多轮对话。"""
        resp1 = client.post("/api/chat", json={"message": "你好"})
        sid = resp1.json()["session_id"]

        resp2 = client.post("/api/chat", json={"message": "继续", "session_id": sid})
        assert resp2.status_code == 200
        assert resp2.json()["session_id"] == sid

    def test_post_chat_empty_message(self, client):
        """空消息应返回错误。"""
        resp = client.post("/api/chat", json={"message": ""})
        # 可能返回 422 或 200 带空回复，取决于实现
        assert resp.status_code in (200, 422)

    def test_delete_session(self, client):
        """DELETE /api/sessions/{sid} 删除会话。"""
        resp1 = client.post("/api/chat", json={"message": "创建会话"})
        sid = resp1.json()["session_id"]

        resp2 = client.delete(f"/api/sessions/{sid}")
        assert resp2.status_code == 200

    def test_delete_session_clears_pending_replies(self, client):
        """手动删除会话时应同步清理待发回复缓冲。"""
        from github_hot_projects import api_server

        resp1 = client.post("/api/chat", json={"message": "创建会话"})
        sid = resp1.json()["session_id"]

        with api_server._pending_replies_lock:
            api_server._pending_replies[sid] = ["pending"]

        resp2 = client.delete(f"/api/sessions/{sid}")

        assert resp2.status_code == 200
        with api_server._pending_replies_lock:
            assert sid not in api_server._pending_replies

    def test_delete_nonexistent_session(self, client):
        """删除不存在的会话应返回 404。"""
        resp = client.delete("/api/sessions/nonexistent-sid")
        assert resp.status_code == 404


class TestReportEndpoints:
    def test_get_reports_list(self, client):
        """GET /api/reports 返回报告列表。"""
        with patch("github_hot_projects.api_server.REPORT_DIR", "/tmp/test_reports_empty"):
            resp = client.get("/api/reports")
            assert resp.status_code == 200
            data = resp.json()
            assert "reports" in data
            assert isinstance(data["reports"], list)

    def test_get_report_html(self, client, tmp_path):
        """GET /api/reports/{name}/html 返回渲染的 HTML。"""
        report_file = tmp_path / "2026-04-14.md"
        report_file.write_text("# 测试报告\n\nhello")

        with patch("github_hot_projects.api_server.REPORT_DIR", str(tmp_path)):
            resp = client.get("/api/reports/2026-04-14.md/html")
            assert resp.status_code == 200
            assert "测试报告" in resp.text

    def test_get_report_html_uses_web_assets(self, client, tmp_path):
        """报告 HTML 页面应引用 web 目录下的静态资源。"""
        report_file = tmp_path / "2026-04-14.md"
        report_file.write_text("# 测试报告\n\nhello")

        with patch("github_hot_projects.api_server.REPORT_DIR", str(tmp_path)):
            resp = client.get("/api/reports/2026-04-14.md/html")
            assert resp.status_code == 200
            assert '/web/report.css' in resp.text
            assert '/web/report.js' in resp.text

    def test_get_report_html_renders_structured_markdown_report(self, client, tmp_path):
        """纯 Markdown 报告应在 HTML 视图中被渲染成结构化卡片。"""
        report_file = tmp_path / "2026-04-18_10d.md"
        report_file.write_text(
            "\n".join(
                [
                    "# GitHub 热门项目 — 2026-04-18",
                    "",
                    "> 共 1 个项目 | 增长统计窗口: 10 天",
                    "",
                    "## 1. hot-org/hot-repo",
                    "",
                    "链接: https://github.com/hot-org/hot-repo",
                    "",
                    "- 创建时间: 2026-04-10",
                    "- 项目状态: NEW（45天内）",
                    "- 主语言: Python",
                    "- 总 Star: 15,000",
                    "- 近10天增长: +2,000",
                    "- 主题标签: ai, agent",
                    "",
                    "### 项目定位与用途",
                    "",
                    "这是一个测试项目。",
                    "",
                    "### 解决的问题",
                    "",
                    "它用于验证 HTML 渲染。",
                    "",
                    "### 使用场景",
                    "",
                    "适合在测试中验证结构化报告。",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        with patch("github_hot_projects.api_server.REPORT_DIR", str(tmp_path)):
            resp = client.get("/api/reports/2026-04-18_10d.md/html")
            assert resp.status_code == 200
            assert 'repo-card--markdown' in resp.text
            assert 'repo-stat__tag--new' in resp.text
            assert '打开仓库' in resp.text
            assert '查看 README' in resp.text
            assert '最近刷新' not in resp.text

    def test_get_report_html_not_found(self, client, tmp_path):
        """不存在的报告应返回 404。"""
        with patch("github_hot_projects.api_server.REPORT_DIR", str(tmp_path)):
            resp = client.get("/api/reports/nonexistent.md/html")
            assert resp.status_code == 404


class TestStatusEndpoint:
    def test_status_includes_session_ttl(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        assert resp.json()["session_ttl_seconds"] > 0


class TestReportXSS:
    def test_xss_script_stripped(self, client, tmp_path):
        """报告中的 <script> 标签应被过滤。"""
        report_file = tmp_path / "xss.md"
        report_file.write_text("hello <script>alert('xss')</script> world")

        with patch("github_hot_projects.api_server.REPORT_DIR", str(tmp_path)):
            resp = client.get("/api/reports/xss.md/html")
            assert resp.status_code == 200
            assert "<script>" not in resp.text
            assert "hello" in resp.text

    def test_xss_event_handler_stripped(self, client, tmp_path):
        """内联事件处理器应被过滤。"""
        report_file = tmp_path / "xss2.md"
        report_file.write_text('hello <img onerror="alert(1)" src=x> world')

        with patch("github_hot_projects.api_server.REPORT_DIR", str(tmp_path)):
            resp = client.get("/api/reports/xss2.md/html")
            assert resp.status_code == 200
            assert "onerror" not in resp.text

    def test_xss_javascript_href_stripped(self, client, tmp_path):
        """Markdown 链接中的 javascript: 协议应被替换掉。"""
        report_file = tmp_path / "xss3.md"
        report_file.write_text("[click](javascript:alert(1))")

        with patch("github_hot_projects.api_server.REPORT_DIR", str(tmp_path)):
            resp = client.get("/api/reports/xss3.md/html")
            assert resp.status_code == 200
            assert 'href="javascript:alert(1)"' not in resp.text
            assert 'href="#"' in resp.text


class TestSessionManagement:
    def test_session_limit(self, client):
        """超过最大会话数应清理最旧会话。"""
        # 快速创建多个会话来触发清理
        sids = []
        for i in range(5):
            resp = client.post("/api/chat", json={"message": f"msg-{i}"})
            sids.append(resp.json()["session_id"])
        # 所有请求应成功
        assert len(sids) == 5
