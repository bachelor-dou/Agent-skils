"""
测试 common 子包模块
====================
覆盖：exceptions / token_manager / db / github_api / llm
"""

import json
import os
import tempfile
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pytest


# ──────────────────────────────────────────────────────────────
# 1. Exceptions
# ──────────────────────────────────────────────────────────────

class TestExceptions:
    def test_rate_limit_error(self):
        from github_hot_projects.common.exceptions import RateLimitError
        err = RateLimitError(token_idx=0, reset_time=1234567890.0)
        assert err.token_idx == 0
        assert err.reset_time == 1234567890.0

    def test_token_invalid_error(self):
        from github_hot_projects.common.exceptions import TokenInvalidError
        err = TokenInvalidError(token_idx=1, message="Token expired")
        assert err.token_idx == 1
        assert "Token expired" in str(err)

    def test_fatal_worker_error(self):
        from github_hot_projects.common.exceptions import FatalWorkerError
        err = FatalWorkerError("All tokens exhausted")
        assert isinstance(err, Exception)

    def test_retryable_error(self):
        from github_hot_projects.common.exceptions import RetryableError
        err = RetryableError(reset_time=9999.0, message="retry later")
        assert err.reset_time == 9999.0

    def test_rate_limit_is_retryable(self):
        from github_hot_projects.common.exceptions import RateLimitError, RetryableError
        err = RateLimitError(token_idx=0, reset_time=1234567890.0)
        assert isinstance(err, RetryableError)

    def test_token_invalid_is_fatal(self):
        from github_hot_projects.common.exceptions import TokenInvalidError, FatalWorkerError
        err = TokenInvalidError(token_idx=0)
        assert isinstance(err, FatalWorkerError)


# ──────────────────────────────────────────────────────────────
# 2. TokenManager
# ──────────────────────────────────────────────────────────────

class TestTokenManager:
    @patch("github_hot_projects.common.token_manager.GITHUB_TOKENS", ["ghp_aaa", "ghp_bbb"])
    def test_init_with_tokens(self):
        from github_hot_projects.common.token_manager import TokenManager
        mgr = TokenManager()
        assert len(mgr.tokens) == 2
        assert mgr.tokens[0] == "ghp_aaa"

    @patch("github_hot_projects.common.token_manager.GITHUB_TOKENS", [])
    def test_init_no_tokens_exits(self):
        from github_hot_projects.common.token_manager import TokenManager
        with pytest.raises(SystemExit):
            TokenManager()

    @patch("github_hot_projects.common.token_manager.GITHUB_TOKENS", ["ghp_test"])
    def test_rest_headers(self):
        from github_hot_projects.common.token_manager import TokenManager
        mgr = TokenManager()
        headers = mgr.get_rest_headers(0)
        assert headers["Authorization"] == "token ghp_test"
        assert "github.v3+json" in headers["Accept"]

    @patch("github_hot_projects.common.token_manager.GITHUB_TOKENS", ["ghp_test"])
    def test_star_headers(self):
        from github_hot_projects.common.token_manager import TokenManager
        mgr = TokenManager()
        headers = mgr.get_star_headers(0)
        assert "star+json" in headers["Accept"]

    @patch("github_hot_projects.common.token_manager.GITHUB_TOKENS", ["ghp_test"])
    def test_graphql_headers(self):
        from github_hot_projects.common.token_manager import TokenManager
        mgr = TokenManager()
        headers = mgr.get_graphql_headers(0)
        assert headers["Authorization"] == "bearer ghp_test"


# ──────────────────────────────────────────────────────────────
# 3. DB
# ──────────────────────────────────────────────────────────────

class TestDB:
    def test_load_db_file_not_exists(self, tmp_path):
        with patch("github_hot_projects.common.db.DB_FILE_PATH", str(tmp_path / "nonexistent.json")):
            from github_hot_projects.common.db import load_db
            db = load_db()
            assert db["date"] == ""
            assert db["valid"] is False
            assert db["projects"] == {}

    def test_load_db_valid(self, tmp_path):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        db_data = {"date": today, "valid": True, "projects": {"a/b": {"star": 100}}}
        db_file = tmp_path / "test_db.json"
        db_file.write_text(json.dumps(db_data))

        with patch("github_hot_projects.common.db.DB_FILE_PATH", str(db_file)):
            from github_hot_projects.common.db import load_db
            db = load_db()
            assert db["valid"] is True
            assert "a/b" in db["projects"]

    def test_load_db_expired(self, tmp_path):
        old_date = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
        db_data = {"date": old_date, "projects": {"a/b": {"star": 100}}}
        db_file = tmp_path / "test_db.json"
        db_file.write_text(json.dumps(db_data))

        with patch("github_hot_projects.common.db.DB_FILE_PATH", str(db_file)):
            from github_hot_projects.common.db import load_db
            db = load_db()
            assert db["valid"] is False
            # 数据不清空
            assert "a/b" in db["projects"]

    def test_save_db_atomic(self, tmp_path):
        db_file = tmp_path / "test_db.json"
        with patch("github_hot_projects.common.db.DB_FILE_PATH", str(db_file)):
            from github_hot_projects.common.db import save_db
            db = {"projects": {"x/y": {"star": 500}}}
            save_db(db)
            assert db_file.exists()
            loaded = json.loads(db_file.read_text())
            assert loaded["date"] == datetime.now(timezone.utc).strftime("%Y-%m-%d")
            assert "x/y" in loaded["projects"]

    def test_load_db_corrupt_json(self, tmp_path):
        db_file = tmp_path / "corrupt.json"
        db_file.write_text("{invalid json")
        with patch("github_hot_projects.common.db.DB_FILE_PATH", str(db_file)):
            from github_hot_projects.common.db import load_db
            db = load_db()
            assert db["projects"] == {}


# ──────────────────────────────────────────────────────────────
# 4. GitHub API
# ──────────────────────────────────────────────────────────────

class TestGitHubAPI:
    def test_search_github_repos_success(self, mock_token_mgr):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "total_count": 1,
            "items": [{"full_name": "org/repo", "stargazers_count": 5000}],
        }

        with patch("github_hot_projects.common.github_api.requests.get", return_value=mock_resp):
            from github_hot_projects.common.github_api import search_github_repos
            result = search_github_repos(mock_token_mgr, "ai agent", token_idx=0)
            assert result is not None
            assert len(result) == 1
            assert result[0]["full_name"] == "org/repo"

    def test_search_github_repos_rate_limit(self, mock_token_mgr):
        from github_hot_projects.common.exceptions import RateLimitError
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.headers = {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "9999999999"}
        mock_resp.text = "rate limit"

        with patch("github_hot_projects.common.github_api.requests.get", return_value=mock_resp):
            from github_hot_projects.common.github_api import search_github_repos
            with pytest.raises(RateLimitError):
                search_github_repos(mock_token_mgr, "test", token_idx=0)

    def test_search_github_repos_token_invalid(self, mock_token_mgr):
        from github_hot_projects.common.exceptions import TokenInvalidError
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Bad credentials"
        mock_resp.headers = {}

        with patch("github_hot_projects.common.github_api.requests.get", return_value=mock_resp):
            from github_hot_projects.common.github_api import search_github_repos
            with pytest.raises(TokenInvalidError):
                search_github_repos(mock_token_mgr, "test", token_idx=0)

    def test_search_github_repos_request_exception_returns_none(self, mock_token_mgr):
        import requests

        with patch(
            "github_hot_projects.common.github_api.requests.get",
            side_effect=requests.RequestException("network down"),
        ) as mock_get:
            with patch("github_hot_projects.common.github_api.time.sleep"):
                from github_hot_projects.common.github_api import search_github_repos

                result = search_github_repos(
                    mock_token_mgr,
                    "test",
                    token_idx=0,
                    worker_idx=2,
                )

        assert result is None
        assert mock_get.call_count == 3

    def test_get_stargazers_page_success(self, mock_token_mgr):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"starred_at": "2026-04-14T12:00:00Z", "user": {"login": "user1"}},
        ]

        with patch("github_hot_projects.common.github_api.requests.get", return_value=mock_resp):
            from github_hot_projects.common.github_api import get_stargazers_page
            result = get_stargazers_page(mock_token_mgr, "org", "repo", page=1, token_idx=0)
            assert result is not None
            assert len(result) == 1

    def test_get_stargazers_page_422_returns_none(self, mock_token_mgr):
        """大仓库高页码返回 422 → 应返回 None（降级信号）。"""
        mock_resp = MagicMock()
        mock_resp.status_code = 422
        mock_resp.text = "Unprocessable Entity"
        mock_resp.headers = {}

        with patch("github_hot_projects.common.github_api.requests.get", return_value=mock_resp):
            from github_hot_projects.common.github_api import get_stargazers_page
            result = get_stargazers_page(mock_token_mgr, "org", "repo", page=999, token_idx=0)
            assert result is None

    def test_graphql_stargazers_batch_success(self, mock_token_mgr):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": {
                "repository": {
                    "stargazers": {
                        "edges": [
                            {"starredAt": "2026-04-14T10:00:00Z", "cursor": "cursor1"},
                            {"starredAt": "2026-04-13T08:00:00Z", "cursor": "cursor2"},
                        ]
                    }
                }
            }
        }

        with patch("github_hot_projects.common.github_api.requests.post", return_value=mock_resp):
            from github_hot_projects.common.github_api import graphql_stargazers_batch
            timestamps, cursor = graphql_stargazers_batch(
                mock_token_mgr, "org", "repo", token_idx=0
            )
            assert len(timestamps) == 2
            assert cursor == "cursor1"

    def test_parse_starred_at(self):
        from github_hot_projects.common.github_api import parse_starred_at_from_entry
        entry = {"starred_at": "2026-04-14T12:30:00Z"}
        result = parse_starred_at_from_entry(entry)
        assert result is not None
        assert result.year == 2026
        assert result.month == 4

    def test_parse_starred_at_empty(self):
        from github_hot_projects.common.github_api import parse_starred_at_from_entry
        assert parse_starred_at_from_entry({}) is None
        assert parse_starred_at_from_entry({"starred_at": ""}) is None


# ──────────────────────────────────────────────────────────────
# 5. LLM
# ──────────────────────────────────────────────────────────────

class TestLLM:
    def test_call_llm_describe_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "这是一个AI项目的测试描述。"}}]
        }

        with patch("github_hot_projects.common.llm.requests.post", return_value=mock_resp):
            from github_hot_projects.common.llm import call_llm_describe
            desc = call_llm_describe(
                "test/repo",
                {"short_desc": "Test", "language": "Python", "topics": ["ai"]},
                "https://github.com/test/repo",
            )
            assert "测试描述" in desc

    def test_call_llm_describe_no_api_key(self):
        with patch("github_hot_projects.common.llm.LLM_API_KEY", ""):
            from github_hot_projects.common.llm import call_llm_describe
            desc = call_llm_describe("test/repo", {}, "https://github.com/test/repo")
            assert desc == ""

    def test_call_llm_describe_failure(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500

        with patch("github_hot_projects.common.llm.requests.post", return_value=mock_resp):
            with patch("github_hot_projects.common.llm.time.sleep"):
                from github_hot_projects.common.llm import call_llm_describe
                desc = call_llm_describe(
                    "test/repo",
                    {"short_desc": "Test"},
                    "https://github.com/test/repo",
                )
                assert desc == ""
