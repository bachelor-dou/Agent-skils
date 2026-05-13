"""
测试 tasks 子包
===============
覆盖：Task 基类、KeywordSearchTask、CalcGrowthTask、任务辅助函数。
"""

from datetime import datetime, timedelta, timezone
import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch, MagicMock, AsyncMock

import pytest


# ──────────────────────────────────────────────────────────────
# 1. Task Base
# ──────────────────────────────────────────────────────────────

class TestTaskBase:
    def test_abstract_task_cannot_instantiate(self):
        """Task 是抽象类（@dataclass + ABC），不能直接实例化。"""
        from github_hot_projects.tasks.task_base import Task
        with pytest.raises(TypeError):
            Task()

    def test_concrete_task_execute(self):
        """具体 Task 子类可正常执行。"""
        from github_hot_projects.tasks.task_base import Task

        @dataclass
        class DummyTask(Task):
            needs_github_token: bool = False
            def execute(self, token_idx=None):
                return "done"

        task = DummyTask()
        assert task.execute() == "done"
        assert task.needs_github_token is False

    def test_concrete_task_execute_async_bridge(self):
        """execute_async 默认桥接到同步 execute。"""
        from github_hot_projects.tasks.task_base import Task

        @dataclass
        class DummyTask(Task):
            needs_github_token: bool = False

            def execute(self, token_idx=None):
                return f"done:{token_idx}"

        task = DummyTask()
        result = asyncio.run(task.execute_async(token_idx=3))
        assert result == "done:3"


# ──────────────────────────────────────────────────────────────
# 2. KeywordSearchTask
# ──────────────────────────────────────────────────────────────

class TestKeywordSearchTask:
    def test_execute_basic(self, mock_token_mgr):
        """关键词搜索任务正常执行。"""
        mock_items = [
            {
                "full_name": "org/repo",
                "stargazers_count": 5000,
                "description": "test",
                "language": "Python",
                "created_at": "2026-01-01T00:00:00Z",
            }
        ]

        with patch(
            "github_hot_projects.tasks.task.search_github_repos",
            side_effect=[mock_items, [], []],
        ):
            with patch("github_hot_projects.tasks.task.time.sleep"):
                from github_hot_projects.tasks.task import KeywordSearchTask
                raw_repos = {}
                task = KeywordSearchTask(
                    keyword="ai agent",
                    category="AI-Agent",
                    keyword_idx=1,
                    total_keywords=1,
                    _raw_repos=raw_repos,
                    _token_mgr=mock_token_mgr,
                )
                result = task.execute(token_idx=0)
                assert len(result) == 1
                assert result[0]["full_name"] == "org/repo"

    def test_execute_async_uses_task_level_token(self, mock_token_mgr):
        """关键词搜索异步路径默认使用任务级 token。"""
        from github_hot_projects.tasks.task import KeywordSearchTask

        async def scenario() -> None:
            with patch(
                "github_hot_projects.tasks.task.async_search_github_repos",
                return_value=[],
            ) as mock_search:
                task = KeywordSearchTask(
                    keyword="ai agent",
                    category="AI-Agent",
                    keyword_idx=1,
                    total_keywords=1,
                    min_star=1500,
                    page_numbers=[1],
                    _raw_repos={},
                    _token_mgr=mock_token_mgr,
                )

                assert task.needs_github_token is True
                await task.execute_async(token_idx=7)

            mock_search.assert_awaited_once_with(
                mock_token_mgr,
                "ai agent",
                7,
                page=1,
                min_star=1500,
                client=None,
            )

        asyncio.run(scenario())

    def test_execute_async_rate_limit_marks_remaining_pages_for_retry(self, mock_token_mgr):
        """任务级 token 在中途限流时，只补偿当前页及后续页，不重跑已成功页。"""
        from github_hot_projects.common.exceptions import RateLimitError
        from github_hot_projects.tasks.task import KeywordSearchTask

        calls = []
        token_mgr = MagicMock()
        token_mgr.mark_rate_limited = AsyncMock()

        async def scenario() -> None:
            async def fake_search(_token_mgr, query, token_idx, page=1, **kwargs):
                calls.append(page)
                if page == 1:
                    return [{
                        "full_name": "org/repo-1",
                        "stargazers_count": 1800,
                        "description": "repo1",
                        "language": "Python",
                        "created_at": "2026-04-01T00:00:00Z",
                    }]
                raise RateLimitError(token_idx=token_idx, reset_time=time.time() + 60)

            with patch(
                "github_hot_projects.tasks.task.async_search_github_repos",
                side_effect=fake_search,
            ), patch("github_hot_projects.tasks.task.asyncio.sleep"):
                task = KeywordSearchTask(
                    keyword="ai agent",
                    category="AI-Agent",
                    keyword_idx=1,
                    total_keywords=1,
                    _raw_repos={},
                    _token_mgr=token_mgr,
                )

                result = await task.execute_async(token_idx=3)

            assert [repo["full_name"] for repo in result] == ["org/repo-1"]
            assert task.failed_pages == [2, 3]

        asyncio.run(scenario())
        assert calls == [1, 2]
        assert token_mgr.mark_rate_limited.await_count == 1
        mark_args = token_mgr.mark_rate_limited.await_args.args
        assert mark_args[0] == 3
        assert isinstance(mark_args[1], float)

    def test_on_result_populates_raw_repos(self, mock_token_mgr):
        """on_result 应将结果写入共享 raw_repos 字典。"""
        from github_hot_projects.tasks.task import KeywordSearchTask
        raw_repos = {}
        task = KeywordSearchTask(
            keyword="test",
            category="Test",
            keyword_idx=1,
            total_keywords=1,
            _raw_repos=raw_repos,
            _token_mgr=mock_token_mgr,
        )
        result = [{"full_name": "x/y", "star": 1000, "repo_item": {}, "created_at": ""}]
        task.on_result(result)
        assert "x/y" in raw_repos


class TestTrendingPeriodTask:
    def test_execute_fetches_single_period(self):
        from github_hot_projects.tasks.task import TrendingPeriodTask

        repos = [{"full_name": "org/repo", "star": 100, "forks": 10, "stars_today": 5}]

        with patch("github_hot_projects.tasks.task.fetch_trending", return_value=repos) as mock_fetch:
            task = TrendingPeriodTask(period="daily")
            period, result = task.execute(token_idx=None)

        assert period == "daily"
        assert result == repos
        mock_fetch.assert_called_once_with(since="daily")


class TestScanSegmentTask:
    def test_retry_pages_do_not_rescan_success_pages(self, mock_token_mgr):
        from github_hot_projects.tasks.task import ScanSegmentTask

        calls = []

        def fake_search(_token_mgr, query, token_idx, page=1, **kwargs):
            calls.append(page)
            if page == 2:
                return [{
                    "full_name": "org/repo-2",
                    "stargazers_count": 2000,
                    "description": "repo2",
                    "language": "Python",
                    "created_at": "2026-04-01T00:00:00Z",
                }]
            return []

        with patch("github_hot_projects.tasks.task.search_github_repos", side_effect=fake_search):
            with patch("github_hot_projects.tasks.task.time.sleep"):
                task = ScanSegmentTask(
                    seg_idx=1,
                    low=100,
                    high=200,
                    total_segments=1,
                    page_numbers=[2],
                    retry_round=1,
                    _raw_repos={},
                    _token_mgr=mock_token_mgr,
                )
                result = task.execute(token_idx=0)

        assert len(result) == 1
        assert calls == [2]

    def test_execute_async_uses_task_level_token(self, mock_token_mgr):
        """区间扫描异步路径默认使用任务级 token。"""
        from github_hot_projects.tasks.task import ScanSegmentTask

        async def scenario() -> None:
            with patch(
                "github_hot_projects.tasks.task.async_search_github_repos",
                return_value=[],
            ) as mock_search:
                task = ScanSegmentTask(
                    seg_idx=1,
                    low=100,
                    high=200,
                    total_segments=1,
                    page_numbers=[1],
                    _raw_repos={},
                    _token_mgr=mock_token_mgr,
                )

                assert task.needs_github_token is True
                await task.execute_async(token_idx=9)

            mock_search.assert_awaited_once_with(
                mock_token_mgr,
                "stars:100..200",
                9,
                page=1,
                sort="updated",
                min_star=0,
                client=None,
            )

        asyncio.run(scenario())

    def test_execute_async_token_invalid_marks_remaining_pages_for_retry(self, mock_token_mgr):
        """区间扫描中途 token 失效时，只补偿未完成页。"""
        from github_hot_projects.common.exceptions import TokenInvalidError
        from github_hot_projects.tasks.task import ScanSegmentTask

        calls = []
        token_mgr = MagicMock()
        token_mgr.mark_invalid = AsyncMock()

        async def scenario() -> None:
            async def fake_search(_token_mgr, query, token_idx, page=1, **kwargs):
                calls.append(page)
                if page == 1:
                    return [{
                        "full_name": "org/repo-1",
                        "stargazers_count": 2000,
                        "description": "repo1",
                        "language": "Python",
                        "created_at": "2026-04-01T00:00:00Z",
                    }]
                raise TokenInvalidError(token_idx=token_idx)

            with patch(
                "github_hot_projects.tasks.task.async_search_github_repos",
                side_effect=fake_search,
            ), patch("github_hot_projects.tasks.task.asyncio.sleep"):
                task = ScanSegmentTask(
                    seg_idx=1,
                    low=100,
                    high=200,
                    total_segments=1,
                    _raw_repos={},
                    _token_mgr=token_mgr,
                )

                result = await task.execute_async(token_idx=5)

            assert [repo["full_name"] for repo in result] == ["org/repo-1"]
            assert task.failed_pages == [2, 3, 4, 5, 6, 7, 8, 9, 10]

        asyncio.run(scenario())
        assert calls == [1, 2]
        token_mgr.mark_invalid.assert_awaited_once_with(5, "Token#5 invalid (401)")


# ──────────────────────────────────────────────────────────────
# 3. CalcGrowthTask
# ──────────────────────────────────────────────────────────────

class TestCalcGrowthTask:
    def test_execute_calls_estimator(self, mock_token_mgr):
        """CalcGrowthTask 应调用 estimate_star_growth_binary。"""
        with patch("github_hot_projects.tasks.task.estimate_star_growth_binary", return_value=1500):
            from github_hot_projects.tasks.task import CalcGrowthTask
            task = CalcGrowthTask(
                full_name="org/repo",
                current_star=10000,
                repo_item={"full_name": "org/repo", "stargazers_count": 10000},
                _ctx=None,
                _token_mgr=mock_token_mgr,
            )
            result = task.execute(token_idx=0)
            assert result == ("org/repo", 1500, 10000)

    def test_execute_async_calls_async_estimator(self, mock_token_mgr):
        """CalcGrowthTask 的异步路径应调用异步增长估算。"""
        from github_hot_projects.tasks.task import CalcGrowthTask

        async def scenario() -> None:
            with patch(
                "github_hot_projects.tasks.task.estimate_star_growth_binary_async",
                return_value=1200,
            ) as mock_estimator:
                task = CalcGrowthTask(
                    full_name="org/repo",
                    current_star=10000,
                    repo_item={"full_name": "org/repo", "stargazers_count": 10000},
                    _ctx=None,
                    _token_mgr=mock_token_mgr,
                )
                result = await task.execute_async(token_idx=1)

            assert result == ("org/repo", 1200, 10000)
            mock_estimator.assert_awaited_once()

        asyncio.run(scenario())

    def test_execute_invalid_format(self, mock_token_mgr):
        """非 owner/repo 格式应返回 -1。"""
        from github_hot_projects.tasks.task import CalcGrowthTask
        task = CalcGrowthTask(
            full_name="invalid",
            current_star=10000,
            repo_item={},
            _ctx=None,
            _token_mgr=mock_token_mgr,
        )
        result = task.execute(token_idx=0)
        assert result[1] == -1

    def test_submit_growth_tasks_stale_repo_skips_db_diff(self, mock_token_mgr):
        from github_hot_projects.tasks.task import _submit_growth_tasks, CalcGrowthTask

        class DummyPool:
            def __init__(self):
                self.submitted = []

            def submit(self, task):
                self.submitted.append(task)

        stale_refresh = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        raw_repos = {
            "org/repo": {
                "star": 5000,
                "created_at": "2026-04-01T00:00:00Z",
                "repo_item": {
                    "full_name": "org/repo",
                    "stargazers_count": 5000,
                    "created_at": "2026-04-01T00:00:00Z",
                },
            }
        }
        db = {
            "valid": True,
            "projects": {
                "org/repo": {
                    "star": 3200,
                    "refreshed_at": stale_refresh,
                }
            },
        }
        growth_ctx = {
            "checkpoint": None,
            "pending_created_at": {},
            "db_projects": db["projects"],
            "candidate_map": {},
            "growth_threshold": 800,
            "use_realtime_growth": False,
            "can_write_db": False,
            "growth_calc_days": 7,
            "window_specified": True,
            "unresolved_count": [0],
            "checkpoint_dirty": [False],
            "completed_since_save": [0],
        }
        pool = DummyPool()

        with patch("github_hot_projects.tasks.task_help._load_checkpoint", return_value={}), patch(
            "github_hot_projects.tasks.task_help._save_checkpoint"
        ):
            checkpoint = _submit_growth_tasks(pool, mock_token_mgr, raw_repos, db, {}, growth_ctx)

        assert checkpoint == {}
        assert len(pool.submitted) == 1
        assert isinstance(pool.submitted[0], CalcGrowthTask)

    def test_submit_growth_tasks_comprehensive_dynamic_window_uses_db_age(self, mock_token_mgr):
        from github_hot_projects.tasks.task import _submit_growth_tasks

        class DummyPool:
            def __init__(self):
                self.submitted = []

            def submit(self, task):
                self.submitted.append(task)

        db_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
        refreshed_at = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
        raw_repos = {
            "org/repo": {
                "star": 5000,
                "created_at": "2026-04-01T00:00:00Z",
                "repo_item": {
                    "full_name": "org/repo",
                    "stargazers_count": 5000,
                    "created_at": "2026-04-01T00:00:00Z",
                },
            }
        }
        db = {
            "valid": True,
            "date": db_date,
            "projects": {
                "org/repo": {
                    "star": 4200,
                    "refreshed_at": refreshed_at,
                }
            },
        }
        growth_ctx = {
            "checkpoint": None,
            "pending_created_at": {},
            "db_projects": db["projects"],
            "candidate_map": {},
            "growth_threshold": 500,
            "use_realtime_growth": False,
            "can_write_db": False,
            "window_specified": False,
            "growth_calc_days": 7,
            "days_since_created": None,
            "unresolved_count": [0],
            "checkpoint_dirty": [False],
            "completed_since_save": [0],
        }
        pool = DummyPool()

        with patch("github_hot_projects.tasks.task_help._load_checkpoint", return_value={}), patch(
            "github_hot_projects.tasks.task_help._save_checkpoint"
        ):
            checkpoint = _submit_growth_tasks(pool, mock_token_mgr, raw_repos, db, {}, growth_ctx)

        assert len(pool.submitted) == 0
        assert checkpoint["org/repo"]["growth"] == 800
        assert growth_ctx["effective_growth_calc_days"] == growth_ctx["growth_calc_days"]

    def test_submit_growth_tasks_comprehensive_specified_window_mismatch_falls_back(self, mock_token_mgr):
        from github_hot_projects.tasks.task import _submit_growth_tasks, CalcGrowthTask

        class DummyPool:
            def __init__(self):
                self.submitted = []

            def submit(self, task):
                self.submitted.append(task)

        db_date = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%d")
        refreshed_at = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
        raw_repos = {
            "org/repo": {
                "star": 5000,
                "created_at": "2026-04-01T00:00:00Z",
                "repo_item": {
                    "full_name": "org/repo",
                    "stargazers_count": 5000,
                    "created_at": "2026-04-01T00:00:00Z",
                },
            }
        }
        db = {
            "valid": True,
            "date": db_date,
            "projects": {
                "org/repo": {
                    "star": 4200,
                    "refreshed_at": refreshed_at,
                }
            },
        }
        growth_ctx = {
            "checkpoint": None,
            "pending_created_at": {},
            "db_projects": db["projects"],
            "candidate_map": {},
            "growth_threshold": 500,
            "use_realtime_growth": True,
            "can_write_db": False,
            "window_specified": True,
            "growth_calc_days": 7,
            "days_since_created": None,
            "unresolved_count": [0],
            "checkpoint_dirty": [False],
            "completed_since_save": [0],
        }
        pool = DummyPool()

        with patch("github_hot_projects.tasks.task_help._load_checkpoint", return_value={}), patch(
            "github_hot_projects.tasks.task_help._save_checkpoint"
        ):
            checkpoint = _submit_growth_tasks(pool, mock_token_mgr, raw_repos, db, {}, growth_ctx)

        assert checkpoint == {}
        assert len(pool.submitted) == 1
        assert isinstance(pool.submitted[0], CalcGrowthTask)

    def test_submit_growth_tasks_hot_new_always_realtime(self, mock_token_mgr):
        """新项目榜始终使用实时计算，不走 DB 差值。"""
        from github_hot_projects.tasks.task import _submit_growth_tasks, CalcGrowthTask

        class DummyPool:
            def __init__(self):
                self.submitted = []

            def submit(self, task):
                self.submitted.append(task)

        db_date = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
        raw_repos = {
            "org/repo": {
                "star": 5000,
                "created_at": "2026-04-01T00:00:00Z",
                "repo_item": {
                    "full_name": "org/repo",
                    "stargazers_count": 5000,
                    "created_at": "2026-04-01T00:00:00Z",
                },
            }
        }
        db = {
            "valid": True,
            "date": db_date,
            "projects": {
                "org/repo": {
                    "star": 4300,
                    "refreshed_at": (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            },
        }
        growth_ctx = {
            "checkpoint": None,
            "pending_created_at": {},
            "db_projects": db["projects"],
            "candidate_map": {},
            "growth_threshold": 500,
            "use_realtime_growth": True,  # 新项目榜始终实时
            "can_write_db": False,
            "window_specified": True,
            "growth_calc_days": 7,
            "days_since_created": 45,
            "is_hot_new": True,  # 新项目榜标记
            "use_checkpoint": False,  # 实时模式不使用 checkpoint
            "unresolved_count": [0],
            "checkpoint_dirty": [False],
            "completed_since_save": [0],
        }
        pool = DummyPool()

        with patch("github_hot_projects.tasks.task_help._load_checkpoint", return_value={}), patch(
            "github_hot_projects.tasks.task_help._save_checkpoint"
        ):
            checkpoint = _submit_growth_tasks(pool, mock_token_mgr, raw_repos, db, {}, growth_ctx)

        # 新项目榜必须提交实时计算任务
        assert len(pool.submitted) == 1
        assert isinstance(pool.submitted[0], CalcGrowthTask)

