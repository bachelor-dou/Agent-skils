"""
测试 growth_estimator 模块
==========================
覆盖：二分法增长估算、采样外推法、边界条件。
"""

import math
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock, call

import pytest


class TestGrowthBinary:
    """二分法增长估算测试。"""

    def test_below_threshold_returns_zero(self, mock_token_mgr):
        """star < STAR_GROWTH_THRESHOLD 直接返回 0。"""
        from github_hot_projects.growth_estimator import estimate_star_growth_binary
        result = estimate_star_growth_binary(mock_token_mgr, "org", "repo", total_stars=500)
        assert result == 0

    def test_growth_in_last_page(self, mock_token_mgr):
        """增长少，窗口边界在最后一页内 → 1 次请求精确计数。"""
        from github_hot_projects.growth_estimator import estimate_star_growth_binary
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=7)

        # 最后一页有 5 条，3 条在窗口内
        last_page = [
            {"starred_at": (cutoff - timedelta(hours=2)).isoformat()},  # 窗口外
            {"starred_at": (cutoff - timedelta(hours=1)).isoformat()},  # 窗口外
            {"starred_at": (cutoff + timedelta(hours=1)).isoformat()},  # 窗口内
            {"starred_at": (now - timedelta(days=3)).isoformat()},       # 窗口内
            {"starred_at": (now - timedelta(hours=1)).isoformat()},      # 窗口内
        ]

        with patch("github_hot_projects.growth_estimator.get_stargazers_page", return_value=last_page):
            result = estimate_star_growth_binary(mock_token_mgr, "org", "repo", total_stars=5000)
            assert result == 3

    def test_last_page_inaccessible_fallback_sampling(self, mock_token_mgr):
        """REST 返回 None → 降级为采样外推。"""
        from github_hot_projects.growth_estimator import estimate_star_growth_binary

        with patch("github_hot_projects.growth_estimator.get_stargazers_page", return_value=None):
            with patch("github_hot_projects.growth_estimator.estimate_by_sampling", return_value=1500) as mock_sampling:
                result = estimate_star_growth_binary(mock_token_mgr, "org", "repo", total_stars=80000)
                assert result == 1500
                mock_sampling.assert_called_once()

    def test_empty_last_page(self, mock_token_mgr):
        """最后一页为空列表 → 返回 0。"""
        from github_hot_projects.growth_estimator import estimate_star_growth_binary

        with patch("github_hot_projects.growth_estimator.get_stargazers_page", return_value=[]):
            result = estimate_star_growth_binary(mock_token_mgr, "org", "repo", total_stars=5000)
            assert result == 0


class TestGrowthSampling:
    """采样外推法测试。"""

    def test_sampling_precise_count(self, mock_token_mgr):
        """采样跨越窗口边界 → 精确计数窗口内 star。"""
        from github_hot_projects.growth_estimator import estimate_by_sampling
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=7)

        # 第 1 批：50 条在窗口外 + 50 条在窗口内
        batch1_ts = (
            [cutoff - timedelta(hours=i) for i in range(50, 0, -1)] +  # 窗口外
            [cutoff + timedelta(hours=i) for i in range(1, 51)]         # 窗口内
        )
        batch1_ts.sort()

        def mock_graphql(token_mgr, owner, repo, token_idx, last=100, before=None):
            if before is None:
                return batch1_ts, "cursor1"
            return [], None

        with patch("github_hot_projects.growth_estimator.graphql_stargazers_batch", side_effect=mock_graphql):
            with patch("github_hot_projects.growth_estimator.time.sleep"):
                result = estimate_by_sampling(mock_token_mgr, "org", "repo")
                assert result == 50  # 精确计数窗口内的

    def test_sampling_insufficient_data(self, mock_token_mgr):
        """采样不足 2 条 → 返回 UNRESOLVED。"""
        from github_hot_projects.growth_estimator import estimate_by_sampling, GROWTH_ESTIMATION_UNRESOLVED

        def mock_graphql(*args, **kwargs):
            return [datetime.now(timezone.utc)], None

        with patch("github_hot_projects.growth_estimator.graphql_stargazers_batch", side_effect=mock_graphql):
            result = estimate_by_sampling(mock_token_mgr, "org", "repo")
            assert result == GROWTH_ESTIMATION_UNRESOLVED

    def test_sampling_empty_batch(self, mock_token_mgr):
        """GraphQL 返回空 → 返回 UNRESOLVED。"""
        from github_hot_projects.growth_estimator import estimate_by_sampling, GROWTH_ESTIMATION_UNRESOLVED

        with patch("github_hot_projects.growth_estimator.graphql_stargazers_batch", return_value=([], None)):
            result = estimate_by_sampling(mock_token_mgr, "org", "repo")
            assert result == GROWTH_ESTIMATION_UNRESOLVED

    def test_sampling_extrapolation(self, mock_token_mgr):
        """全部采样在窗口内 → 分段加权外推。"""
        from github_hot_projects.growth_estimator import estimate_by_sampling
        now = datetime.now(timezone.utc)

        # 生成 200 条时间戳，全在最近 3 天内
        timestamps = [now - timedelta(hours=i * 0.36) for i in range(200)]
        timestamps.sort()

        call_count = [0]

        def mock_graphql(*args, **kwargs):
            if call_count[0] < 2:
                batch = timestamps[call_count[0] * 100: (call_count[0] + 1) * 100]
                call_count[0] += 1
                return batch, f"cursor{call_count[0]}"
            return [], None

        with patch("github_hot_projects.growth_estimator.graphql_stargazers_batch", side_effect=mock_graphql):
            with patch("github_hot_projects.growth_estimator.time.sleep"):
                result = estimate_by_sampling(mock_token_mgr, "org", "repo")
                # 外推结果应 > 200（因为采样只覆盖 3 天但窗口是 7 天）
                assert result > 200
