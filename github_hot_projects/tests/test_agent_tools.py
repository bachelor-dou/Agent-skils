"""
测试 agent_tools 模块
======================
覆盖：核心 Tool 函数的调用逻辑。
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import pytest


class TestToolCheckRepoGrowth:
    def _make_repo_item(self, full_name="org/repo", star=5000):
        return {
            "full_name": full_name,
            "stargazers_count": star,
            "description": "test",
            "language": "Python",
            "topics": ["ai"],
            "html_url": f"https://github.com/{full_name}",
            "created_at": "2025-01-01T00:00:00Z",
        }

    def test_single_repo_always_realtime_estimate(self, mock_token_mgr):
        """单仓库查询始终走实时估算，并实时生成描述（不复用 DB 缓存描述）。"""
        from github_hot_projects.agent_tools import tool_check_repo_growth

        refreshed_at = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
        db = {
            "valid": True,
            "date": (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d"),
            "projects": {
                "org/repo": {"star": 4800, "desc": "已有详细描述", "refreshed_at": refreshed_at},
            },
        }

        with patch("github_hot_projects.agent_tools.fetch_repo_info", return_value=self._make_repo_item()):
            with patch("github_hot_projects.agent_tools.estimate_star_growth_binary", return_value=200):
                with patch("github_hot_projects.agent_tools.call_llm_describe", return_value="实时描述"):
                    result = tool_check_repo_growth(mock_token_mgr, "org/repo", db=db)

        assert result["growth"] == 200
        assert "二分法" in result["method"]
        assert result["description"] == "实时描述"

    def test_stale_project_falls_back_to_estimate(self, mock_token_mgr):
        """仓库 refreshed_at 过旧 → 不走 DB 差值法，回退到二分法。"""
        from github_hot_projects.agent_tools import tool_check_repo_growth

        stale_refresh = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        db = {
            "valid": True,
            "date": (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d"),
            "projects": {
                "org/repo": {"star": 4800, "desc": "已有描述", "refreshed_at": stale_refresh},
            },
        }

        with patch("github_hot_projects.agent_tools.fetch_repo_info", return_value=self._make_repo_item()):
            with patch("github_hot_projects.agent_tools.estimate_star_growth_binary", return_value=650):
                result = tool_check_repo_growth(mock_token_mgr, "org/repo", db=db)

        assert result["growth"] == 650
        assert "二分法" in result["method"]

    def test_estimate_method_no_db(self, mock_token_mgr):
        """DB 为 None → 走二分法估算 + LLM 生成描述。"""
        from github_hot_projects.agent_tools import tool_check_repo_growth

        with patch("github_hot_projects.agent_tools.fetch_repo_info", return_value=self._make_repo_item()):
            with patch("github_hot_projects.agent_tools.estimate_star_growth_binary", return_value=800):
                with patch("github_hot_projects.agent_tools.call_llm_describe", return_value="描述"):
                    result = tool_check_repo_growth(mock_token_mgr, "org/repo", db=None)

        assert result["growth"] == 800
        assert "二分法" in result["method"]
        assert result["description"] == "描述"

    def test_custom_time_window(self, mock_token_mgr):
        """自定义时间窗口应体现在 method 中。"""
        from github_hot_projects.agent_tools import tool_check_repo_growth

        with patch("github_hot_projects.agent_tools.fetch_repo_info", return_value=self._make_repo_item()):
            with patch("github_hot_projects.agent_tools.estimate_star_growth_binary", return_value=650):
                result = tool_check_repo_growth(mock_token_mgr, "org/repo", db=None, growth_calc_days=10)

        assert result["growth"] == 650
        assert "10天窗口" in result["method"]
        assert result["growth_calc_days"] == 10

    def test_invalid_repo_format(self, mock_token_mgr):
        """非 owner/repo 格式应返回 error。"""
        from github_hot_projects.agent_tools import tool_check_repo_growth
        result = tool_check_repo_growth(mock_token_mgr, "invalid-format")
        assert "error" in result

    def test_repo_not_found(self, mock_token_mgr):
        """仓库不存在应返回 error。"""
        from github_hot_projects.agent_tools import tool_check_repo_growth
        with patch("github_hot_projects.agent_tools.fetch_repo_info", return_value=None):
            result = tool_check_repo_growth(mock_token_mgr, "org/nonexistent")
            assert "error" in result


class TestToolBatchCheckGrowth:
    def _repo_input(self) -> list[dict]:
        return [
            {
                "full_name": "org/repo",
                "star": 5000,
                "_raw": {
                    "full_name": "org/repo",
                    "stargazers_count": 5000,
                    "description": "A candidate repository",
                    "language": "Python",
                    "topics": ["ai", "agent"],
                    "created_at": "2026-04-01T00:00:00Z",
                },
            }
        ]

    def test_custom_window_generates_brief_desc_without_snapshot_refresh(self, mock_token_mgr):
        """指定非默认窗口时使用实时计算，不刷新 DB 快照。"""
        from github_hot_projects.agent_tools import tool_batch_check_growth

        repos = self._repo_input()
        db = {"valid": True, "date": "2026-04-01", "projects": {}}

        def fake_submit(_pool, _token_mgr, _raw_repos, _db, candidate_map, _growth_ctx):
            candidate_map["org/repo"] = {
                "growth": 900,
                "star": 5000,
                "created_at": "2026-04-01T00:00:00Z",
            }
            return {}

        with patch("github_hot_projects.agent_tools._submit_growth_tasks", side_effect=fake_submit):
            with patch("github_hot_projects.agent_tools.update_db_project") as mock_update_db:
                result = tool_batch_check_growth(
                    mock_token_mgr,
                    repos,
                    db,
                    growth_calc_days=10,
                )

        mock_update_db.assert_not_called()
        assert result["use_realtime_growth"] is True  # 指定窗口导致实时计算
        assert result["db_updated"] is False

    def test_force_refresh_seeds_snapshot_before_growth(self, mock_token_mgr):
        from github_hot_projects.agent_tools import tool_batch_check_growth

        repos = self._repo_input()
        db = {"valid": True, "date": "2026-04-01", "projects": {}}

        with patch("github_hot_projects.agent_tools._submit_growth_tasks", return_value={}):
            with patch("github_hot_projects.agent_tools.update_db_project") as mock_update_db:
                result = tool_batch_check_growth(
                    mock_token_mgr,
                    repos,
                    db,
                    force_refresh=True,
                )

        assert mock_update_db.call_count == 1
        assert result["seeded_snapshot_count"] == 1
        assert result["db_updated"] is True


class TestToolRankCandidates:
    def test_rank_returns_dict(self, sample_candidates):
        """tool_rank_candidates 应返回含 ranked_projects 的字典。"""
        from github_hot_projects.agent_tools import tool_rank_candidates
        result = tool_rank_candidates(sample_candidates)
        assert "ranked_projects" in result
        assert len(result["ranked_projects"]) > 0
        # new-org/new-repo 综合得分最高（高增长率）
        assert result["ranked_projects"][0]["repo"] == "new-org/new-repo"

    def test_rank_hot_new_mode(self, sample_candidates):
        """hot_new 模式排名。"""
        from github_hot_projects.agent_tools import tool_rank_candidates
        result = tool_rank_candidates(
            sample_candidates, mode="hot_new",
            days_since_created=45,
            prefiltered_days_since_created=45,
        )
        assert result["mode"] == "hot_new"

    def test_rank_invalid_numeric_and_mode_fall_back_to_defaults(self, sample_candidates):
        """非法 top_n / mode 不应静默产生截断错结果。"""
        from github_hot_projects.agent_tools import tool_rank_candidates

        result = tool_rank_candidates(sample_candidates, top_n=-1, mode="bad-mode")

        assert result["mode"] == "comprehensive"
        assert result["returned"] == 1


class TestToolGenerateReport:
    def test_generate_report_call(self, tmp_path):
        """tool_generate_report 应最终调用 step3_generate_report。"""
        from github_hot_projects.agent_tools import tool_generate_report

        top = [("org/repo", {"growth": 1000, "star": 5000})]
        db = {"projects": {}}

        with patch("github_hot_projects.agent_tools.step3_generate_report", return_value=str(tmp_path / "test.md")) as mock_gen:
            result = tool_generate_report(top, db)
            mock_gen.assert_called_once()
            assert result["report_path"].endswith("test.md")


class TestToolFetchTrending:
    def test_trending_returns_dict(self):
        """tool_fetch_trending 应返回含 repos 的字典。"""
        from github_hot_projects.agent_tools import tool_fetch_trending

        mock_repos = [
            {
                "full_name": "trending-org/trending-repo",
                "star": 12345,
                "forks": 1000,
                "description": "A cool project",
                "language": "Python",
                "stars_today": 567,
                "since": "weekly",
            }
        ]

        with patch("github_hot_projects.github_trending.fetch_trending", return_value=mock_repos):
            result = tool_fetch_trending()
            assert "repos" in result
            assert len(result["repos"]) == 1
            assert result["repos"][0]["full_name"] == "trending-org/trending-repo"
            assert "language" not in result


class TestToolScanStarRange:
    def test_retries_failed_pages_only(self, mock_token_mgr):
        from github_hot_projects.agent_tools import tool_scan_star_range

        calls = []
        page2_attempts = [0]

        def fake_search(_token_mgr, query, token_idx, page=1, **kwargs):
            calls.append((query, page, token_idx, kwargs.get("worker_idx")))
            if page == 1:
                return [{
                    "full_name": "org/repo-1",
                    "stargazers_count": 150,
                    "description": "repo1",
                    "language": "Python",
                    "created_at": "2026-04-01T00:00:00Z",
                }]
            if page == 2:
                page2_attempts[0] += 1
                if page2_attempts[0] == 1:
                    return None
                return [{
                    "full_name": "org/repo-2",
                    "stargazers_count": 160,
                    "description": "repo2",
                    "language": "Go",
                    "created_at": "2026-04-02T00:00:00Z",
                }]
            return []

        with patch("github_hot_projects.agent_tools.auto_split_star_range", return_value=[(100, 200)]):
            with patch("github_hot_projects.tasks.task.search_github_repos", side_effect=fake_search):
                with patch("github_hot_projects.tasks.task.time.sleep"):
                    result = tool_scan_star_range(mock_token_mgr, min_star=100, max_star=200)

        assert result["total"] == 2
        assert page2_attempts[0] == 2
        assert [page for _, page, _, _ in calls].count(1) == 1
        assert [page for _, page, _, _ in calls].count(2) == 2


class TestToolGetDBInfo:
    def test_db_info_overview(self):
        """tool_get_db_info 无 repo 参数返回概览。"""
        from github_hot_projects.agent_tools import tool_get_db_info

        db = {
            "date": "2026-04-14",
            "valid": True,
            "projects": {"a/b": {"star": 100}, "c/d": {"star": 200}},
        }

        result = tool_get_db_info(db)
        assert result["date"] == "2026-04-14"
        assert result["total_projects"] == 2

    def test_db_info_specific_repo(self):
        """tool_get_db_info 查询特定仓库。"""
        from github_hot_projects.agent_tools import tool_get_db_info

        db = {"projects": {"org/repo": {"star": 5000, "desc": "test"}}}
        result = tool_get_db_info(db, repo="org/repo")
        assert result["found"] is True
        assert result["info"]["star"] == 5000

    def test_db_info_repo_not_found(self):
        """tool_get_db_info 查询不存在仓库。"""
        from github_hot_projects.agent_tools import tool_get_db_info
        result = tool_get_db_info({"projects": {}}, repo="nonexistent/repo")
        assert result["found"] is False


class TestToolDescribeProject:
    def test_describe_with_api_enriched_llm(self, mock_token_mgr):
        """describe_project 拉取实时 API 上下文并传给 LLM，但完全不写 DB（其他通道只读不写）。"""
        from github_hot_projects.agent_tools import tool_describe_project

        db = {"projects": {"org/repo": {"star": 5000}}}

        repo_item = {
            "description": "A framework for building LLM apps",
            "language": "Python",
            "topics": ["llm", "agent"],
            "html_url": "https://github.com/org/repo",
        }

        with patch("github_hot_projects.agent_tools.fetch_repo_info", return_value=repo_item):
            with patch(
                "github_hot_projects.agent_tools.fetch_repo_readme_excerpt",
                return_value={"text": "README 摘录", "sha": "abc123"},
            ):
                with patch(
                    "github_hot_projects.agent_tools.fetch_repo_recent_releases",
                    return_value=[{"tag_name": "v1.2.0", "published_at": "2026-04-20T00:00:00Z"}],
                ):
                    with patch(
                        "github_hot_projects.agent_tools.fetch_repo_recent_commits",
                        return_value=[{"date": "2026-04-21T00:00:00Z", "message": "improve docs"}],
                    ):
                        with patch(
                            "github_hot_projects.agent_tools.call_llm_describe",
                            return_value="LLM生成的综合描述",
                        ) as mock_llm:
                            result = tool_describe_project("org/repo", db, token_mgr=mock_token_mgr)

        assert result["description"] == "LLM生成的综合描述"
        assert result["source"] == "LLM生成"
        assert "note" in result  # 包含"其他通道只读不写DB"的提示
        # 其他通道完全不写 DB（包括元数据）
        assert "desc" not in db["projects"]["org/repo"]
        assert "readme_sha" not in db["projects"]["org/repo"]

        llm_repo_info = mock_llm.call_args.args[1]
        assert llm_repo_info["readme_excerpt"] == "README 摘录"
        assert "language" not in llm_repo_info
        assert "languages_breakdown" not in llm_repo_info
        assert "roadmap_signals" not in llm_repo_info

    def test_describe_api_failure_falls_back_to_cached_desc(self, mock_token_mgr):
        """API 失败时若 DB 有缓存，回退到缓存描述。"""
        from github_hot_projects.agent_tools import tool_describe_project

        db = {"projects": {"org/repo": {"star": 5000, "desc": "已缓存描述"}}}

        with patch("github_hot_projects.agent_tools.fetch_repo_info", return_value=None):
            result = tool_describe_project("org/repo", db, token_mgr=mock_token_mgr)

        assert result["description"] == "已缓存描述"
        assert result["source"] == "DB缓存"

    def test_describe_api_failure_without_cache_returns_error(self, mock_token_mgr):
        """API 失败且无缓存时返回 error，避免凭空猜测。"""
        from github_hot_projects.agent_tools import tool_describe_project

        db = {"projects": {"org/repo": {"star": 5000, "desc": ""}}}

        with patch("github_hot_projects.agent_tools.fetch_repo_info", return_value=None):
            result = tool_describe_project("org/repo", db, token_mgr=mock_token_mgr)

        assert "error" in result

    def test_describe_without_token_mgr_keeps_legacy_path(self):
        """未传 token_mgr 时兼容旧行为：直接调用 LLM。"""
        from github_hot_projects.agent_tools import tool_describe_project

        db = {"projects": {"org/repo": {"star": 5000, "desc": ""}}}

        with patch("github_hot_projects.agent_tools.call_llm_describe", return_value="LLM生成的描述") as mock_llm:
            result = tool_describe_project("org/repo", db)

        mock_llm.assert_called_once()
        assert result["description"] == "LLM生成的描述"
        assert result["source"] == "LLM生成"
