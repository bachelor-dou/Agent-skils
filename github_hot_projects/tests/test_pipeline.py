"""测试 scheduled_update 内置 DiscoveryPipeline。"""

from unittest.mock import patch


class TestDiscoveryPipeline:
    def test_pipeline_passes_force_refresh_to_growth_stage(self, mock_token_mgr):
        from github_hot_projects.scheduled_update import DiscoveryPipeline

        pipeline = DiscoveryPipeline(mock_token_mgr, {"valid": True, "date": "2026-04-01", "projects": {}})
        raw_repo = {
            "full_name": "org/repo",
            "star": 5000,
            "_raw": {"full_name": "org/repo", "stargazers_count": 5000},
        }

        with patch(
            "github_hot_projects.scheduled_update.tool_search_hot_projects",
            return_value={"repos": [], "total": 1, "_raw_repos": [raw_repo]},
        ), patch(
            "github_hot_projects.scheduled_update.tool_scan_star_range",
            return_value={"repos": [], "total": 0, "_raw_repos": []},
        ), patch(
            "github_hot_projects.scheduled_update.tool_fetch_trending",
            return_value={"repos": [], "count": 0, "_raw_repos": []},
        ), patch(
            "github_hot_projects.scheduled_update.tool_batch_check_growth",
            return_value={"candidates": {"org/repo": {"growth": 600, "star": 5000}}, "total_checked": 1},
        ) as mock_batch, patch(
            "github_hot_projects.scheduled_update.tool_rank_candidates",
            return_value={"_ordered_tuples": [("org/repo", {"growth": 600, "star": 5000})]},
        ), patch(
            "github_hot_projects.scheduled_update.tool_generate_report",
            return_value={"report_path": "report.md"},
        ), patch("github_hot_projects.scheduled_update.save_db"):
            result = pipeline.run(force_refresh=True)

        assert result["report_path"] == "report.md"
        assert mock_batch.call_args.kwargs["force_refresh"] is True


def test_scheduled_update_uses_force_refresh_for_pipeline():
    from github_hot_projects.scheduled_update import run_update

    with patch("github_hot_projects.scheduled_update.TokenManager"), patch(
        "github_hot_projects.scheduled_update.load_db",
        return_value={"valid": True, "date": "2026-04-01", "projects": {}},
    ), patch("github_hot_projects.scheduled_update.DiscoveryPipeline") as mock_pipeline:
        mock_pipeline.return_value.run.return_value = {"report_path": "report.md"}

        run_update(20, "comprehensive")

    assert mock_pipeline.return_value.run.call_args.kwargs["force_refresh"] is True