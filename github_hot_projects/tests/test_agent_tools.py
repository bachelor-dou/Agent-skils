"""
测试 agent_tools 模块
======================
覆盖：核心 Tool 函数的调用逻辑。
"""

from unittest.mock import patch, MagicMock

import pytest


class TestToolCheckRepoGrowth:
    def test_db_diff_method(self, mock_token_mgr):
        """DB 有效且仓库在库中 → 用差值法计算增长。"""
        from github_hot_projects.agent_tools import tool_check_repo_growth

        db = {
            "valid": True,
            "projects": {
                "org/repo": {"star": 4800, "desc": "已有描述"},
            },
        }

        search_result = [{
            "full_name": "org/repo",
            "stargazers_count": 5000,
            "description": "test",
            "language": "Python",
            "topics": ["ai"],
            "html_url": "https://github.com/org/repo",
            "created_at": "2025-01-01T00:00:00Z",
        }]

        with patch("github_hot_projects.agent_tools.search_github_repos", return_value=search_result):
            result = tool_check_repo_growth(mock_token_mgr, "org/repo", db=db)
            assert result["growth"] == 200  # 5000 - 4800
            assert result["method"] == "DB差值法"
            assert result["description"] == "已有描述"

    def test_estimate_method(self, mock_token_mgr):
        """DB 无效 → 调用 estimate_star_growth_binary。"""
        from github_hot_projects.agent_tools import tool_check_repo_growth

        search_result = [{
            "full_name": "org/repo",
            "stargazers_count": 5000,
            "description": "test",
            "language": "Python",
            "topics": [],
            "html_url": "https://github.com/org/repo",
            "created_at": "2025-01-01T00:00:00Z",
        }]

        with patch("github_hot_projects.agent_tools.search_github_repos", return_value=search_result):
            with patch("github_hot_projects.agent_tools.estimate_star_growth_binary", return_value=800):
                with patch("github_hot_projects.agent_tools.call_llm_describe", return_value="描述"):
                    result = tool_check_repo_growth(mock_token_mgr, "org/repo", db=None)
                    assert result["growth"] == 800
                    assert "二分法" in result["method"]

    def test_invalid_repo_format(self, mock_token_mgr):
        """非 owner/repo 格式应返回 error。"""
        from github_hot_projects.agent_tools import tool_check_repo_growth
        result = tool_check_repo_growth(mock_token_mgr, "invalid-format")
        assert "error" in result

    def test_repo_not_found(self, mock_token_mgr):
        """仓库不存在应返回 error。"""
        from github_hot_projects.agent_tools import tool_check_repo_growth
        with patch("github_hot_projects.agent_tools.search_github_repos", return_value=[]):
            result = tool_check_repo_growth(mock_token_mgr, "org/nonexistent")
            assert "error" in result


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
            new_project_days=45,
            prefiltered_new_project_days=45,
        )
        assert result["mode"] == "hot_new"


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
    def test_describe_with_llm(self):
        """DB 无描述 → 调用 LLM。"""
        from github_hot_projects.agent_tools import tool_describe_project

        db = {"projects": {"org/repo": {"star": 5000, "desc": ""}}}

        with patch("github_hot_projects.agent_tools.call_llm_describe", return_value="LLM生成的描述"):
            result = tool_describe_project("org/repo", db)
            assert result["description"] == "LLM生成的描述"
            assert result["source"] == "LLM生成"

    def test_describe_uses_cache(self):
        """DB 已有 desc → 不调用 LLM。"""
        from github_hot_projects.agent_tools import tool_describe_project

        db = {"projects": {"org/repo": {"star": 5000, "desc": "已缓存描述"}}}

        with patch("github_hot_projects.agent_tools.call_llm_describe") as mock_llm:
            result = tool_describe_project("org/repo", db)
            mock_llm.assert_not_called()
            assert result["description"] == "已缓存描述"
            assert result["source"] == "DB缓存"
