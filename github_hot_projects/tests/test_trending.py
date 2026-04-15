"""
测试 github_trending 模块
=========================
覆盖：HTML 解析、数字解析、多周期汇总去重。
"""

from unittest.mock import patch, MagicMock

import pytest

from .conftest import TRENDING_HTML_FIXTURE


class TestTrendingParser:
    def test_parse_trending_html_basic(self):
        """解析标准 Trending HTML，提取仓库信息。"""
        from github_hot_projects.github_trending import _parse_trending_html
        repos = _parse_trending_html(TRENDING_HTML_FIXTURE, since="weekly")
        assert len(repos) == 2
        assert repos[0]["full_name"] == "trending-org/trending-repo"
        assert repos[0]["star"] == 12345
        assert repos[0]["forks"] == 1234
        assert repos[0]["stars_today"] == 567
        assert repos[0]["language"] == "Python"
        assert "AI" in repos[0]["description"]

    def test_parse_trending_html_second_repo(self):
        from github_hot_projects.github_trending import _parse_trending_html
        repos = _parse_trending_html(TRENDING_HTML_FIXTURE, since="weekly")
        assert repos[1]["full_name"] == "another-org/another-repo"
        assert repos[1]["star"] == 5678
        assert repos[1]["language"] == "Rust"

    def test_parse_trending_html_empty(self):
        """空 HTML 应返回空列表。"""
        from github_hot_projects.github_trending import _parse_trending_html
        repos = _parse_trending_html("<html><body></body></html>", since="daily")
        assert repos == []

    def test_parse_number(self):
        from github_hot_projects.github_trending import _parse_number
        assert _parse_number("12,345") == 12345
        assert _parse_number("0") == 0
        assert _parse_number("") == 0
        assert _parse_number("1,000,000") == 1000000

    def test_fetch_trending_success(self):
        """模拟 requests.get 返回 Trending HTML。"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = TRENDING_HTML_FIXTURE
        mock_resp.raise_for_status = MagicMock()

        with patch("github_hot_projects.github_trending.requests.get", return_value=mock_resp):
            from github_hot_projects.github_trending import fetch_trending
            repos = fetch_trending(since="weekly")
            assert len(repos) == 2
            assert repos[0]["full_name"] == "trending-org/trending-repo"

    def test_fetch_trending_network_error(self):
        """网络失败应返回空列表。"""
        import requests as req
        with patch("github_hot_projects.github_trending.requests.get", side_effect=req.RequestException("timeout")):
            from github_hot_projects.github_trending import fetch_trending
            repos = fetch_trending()
            assert repos == []

    def test_fetch_trending_invalid_since(self):
        """无效 since 参数应回退到默认 weekly。"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = TRENDING_HTML_FIXTURE
        mock_resp.raise_for_status = MagicMock()

        with patch("github_hot_projects.github_trending.requests.get", return_value=mock_resp) as mock_get:
            from github_hot_projects.github_trending import fetch_trending
            fetch_trending(since="invalid_period")
            call_args = mock_get.call_args
            assert call_args[1]["params"]["since"] == "weekly"


class TestTrendingAll:
    def test_fetch_trending_all_dedup(self):
        """多周期汇总应去重，保留多 periods。"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = TRENDING_HTML_FIXTURE
        mock_resp.raise_for_status = MagicMock()

        with patch("github_hot_projects.github_trending.requests.get", return_value=mock_resp):
            from github_hot_projects.github_trending import fetch_trending_all
            repos = fetch_trending_all()
            # 相同仓库被去重，但应有 3 个 periods（daily, weekly, monthly 都出现）
            assert len(repos) == 2
            first = repos[0]
            assert len(first["periods"]) == 3
            assert "daily" in first["periods"]
            assert "weekly" in first["periods"]
            assert "monthly" in first["periods"]
