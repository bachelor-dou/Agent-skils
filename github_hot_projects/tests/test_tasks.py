"""
测试 tasks 子包
===============
覆盖：Task 基类、KeywordSearchTask、CalcGrowthTask、WorkerPool。
"""

from datetime import datetime, timedelta, timezone
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch, MagicMock

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
            needs_token: bool = False
            def execute(self, token_idx=None):
                return "done"

        task = DummyTask()
        assert task.execute() == "done"
        assert task.needs_token is False


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

        with patch("github_hot_projects.tasks.task.search_github_repos", return_value=mock_items):
            with patch("github_hot_projects.tasks.task.time.sleep"):
                from github_hot_projects.tasks.task import KeywordSearchTask
                raw_repos = {}
                task = KeywordSearchTask(
                    keyword="ai agent",
                    category="AI-Agent",
                    keyword_idx=1,
                    total_keywords=1,
                    max_pages=1,
                    _raw_repos=raw_repos,
                    _token_mgr=mock_token_mgr,
                )
                result = task.execute(token_idx=0)
                assert len(result) == 1
                assert result[0]["full_name"] == "org/repo"

    def test_on_result_populates_raw_repos(self, mock_token_mgr):
        """on_result 应将结果写入共享 raw_repos 字典。"""
        from github_hot_projects.tasks.task import KeywordSearchTask
        raw_repos = {}
        task = KeywordSearchTask(
            keyword="test",
            category="Test",
            keyword_idx=1,
            total_keywords=1,
            max_pages=1,
            _raw_repos=raw_repos,
            _token_mgr=mock_token_mgr,
        )
        result = [{"full_name": "x/y", "star": 1000, "repo_item": {}, "created_at": ""}]
        task.on_result(result)
        assert "x/y" in raw_repos


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
            "unresolved_count": [0],
            "checkpoint_dirty": [False],
            "completed_since_save": [0],
        }
        pool = DummyPool()

        with patch("github_hot_projects.tasks.task._load_checkpoint", return_value={}), patch(
            "github_hot_projects.tasks.task._save_checkpoint"
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
            "time_window_days": 7,
            "new_project_days": None,
            "unresolved_count": [0],
            "checkpoint_dirty": [False],
            "completed_since_save": [0],
        }
        pool = DummyPool()

        with patch("github_hot_projects.tasks.task._load_checkpoint", return_value={}), patch(
            "github_hot_projects.tasks.task._save_checkpoint"
        ):
            checkpoint = _submit_growth_tasks(pool, mock_token_mgr, raw_repos, db, {}, growth_ctx)

        assert len(pool.submitted) == 0
        assert checkpoint["org/repo"]["growth"] == 800
        assert growth_ctx["effective_time_window_days"] == growth_ctx["time_window_days"]

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
            "time_window_days": 7,
            "new_project_days": None,
            "unresolved_count": [0],
            "checkpoint_dirty": [False],
            "completed_since_save": [0],
        }
        pool = DummyPool()

        with patch("github_hot_projects.tasks.task._load_checkpoint", return_value={}), patch(
            "github_hot_projects.tasks.task._save_checkpoint"
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
            "time_window_days": 7,
            "new_project_days": 45,
            "is_hot_new": True,  # 新项目榜标记
            "use_checkpoint": False,  # 实时模式不使用 checkpoint
            "unresolved_count": [0],
            "checkpoint_dirty": [False],
            "completed_since_save": [0],
        }
        pool = DummyPool()

        with patch("github_hot_projects.tasks.task._load_checkpoint", return_value={}), patch(
            "github_hot_projects.tasks.task._save_checkpoint"
        ):
            checkpoint = _submit_growth_tasks(pool, mock_token_mgr, raw_repos, db, {}, growth_ctx)

        # 新项目榜必须提交实时计算任务
        assert len(pool.submitted) == 1
        assert isinstance(pool.submitted[0], CalcGrowthTask)


# ──────────────────────────────────────────────────────────────
# 4. WorkerPool
# ──────────────────────────────────────────────────────────────

class TestWorkerPool:
    def test_basic_task_execution(self):
        """基本任务执行：提交 → 完成 → drain_results。"""
        from github_hot_projects.tasks.task_base import Task
        from github_hot_projects.tasks.worker_pool import TokenWorkerPool

        @dataclass
        class SimpleTask(Task):
            needs_token: bool = False
            result_value: Any = None
            def execute(self, token_idx=None):
                return 42
            def on_result(self, result):
                self.result_value = result

        pool = TokenWorkerPool(["token1"])
        pool.start()
        try:
            task = SimpleTask()
            pool.submit(task)
            pool.wait_all_done(timeout=5.0)
            pool.drain_results()
            assert task.result_value == 42
        finally:
            pool.shutdown()

    def test_multiple_tasks(self):
        """多任务并行执行。"""
        from github_hot_projects.tasks.task_base import Task
        from github_hot_projects.tasks.worker_pool import TokenWorkerPool

        results = []

        @dataclass
        class CountTask(Task):
            needs_token: bool = False
            n: int = 0
            def execute(self, token_idx=None):
                return self.n * 2
            def on_result(self, result):
                results.append(result)

        pool = TokenWorkerPool(["token1", "token2"])
        pool.start()
        try:
            for i in range(10):
                pool.submit(CountTask(n=i))
            pool.wait_all_done(timeout=10.0)
            pool.drain_results()
            assert len(results) == 10
            assert sorted(results) == [i * 2 for i in range(10)]
        finally:
            pool.shutdown()

    def test_task_with_exception(self):
        """任务抛出普通异常 → 调用 on_error，不崩溃 pool。"""
        from github_hot_projects.tasks.task_base import Task
        from github_hot_projects.tasks.worker_pool import TokenWorkerPool

        error_received = []

        @dataclass
        class FailTask(Task):
            needs_token: bool = False
            def execute(self, token_idx=None):
                raise ValueError("intentional error")
            def on_error(self, error):
                error_received.append(error)

        @dataclass
        class OKTask(Task):
            needs_token: bool = False
            done: bool = False
            def execute(self, token_idx=None):
                return "ok"
            def on_result(self, result):
                self.done = True

        pool = TokenWorkerPool(["token1"])
        pool.start()
        try:
            fail_task = FailTask()
            ok_task = OKTask()
            pool.submit(fail_task)
            pool.submit(ok_task)
            pool.wait_all_done(timeout=5.0)
            pool.drain_results()
            assert len(error_received) == 1
            assert isinstance(error_received[0], ValueError)
            assert ok_task.done is True
        finally:
            pool.shutdown()

    def test_fatal_error_requeues_to_other_worker(self):
        """FatalWorkerError 会让当前 worker 退出，并把任务回退给其他 worker。"""
        from github_hot_projects.common.exceptions import FatalWorkerError
        from github_hot_projects.tasks.task_base import Task
        from github_hot_projects.tasks.worker_pool import TokenWorkerPool

        execution_workers = []

        @dataclass
        class FatalOnceTask(Task):
            needs_token: bool = True
            attempts: int = 0
            result_value: Any = None

            def execute(self, token_idx=None):
                execution_workers.append(token_idx)
                self.attempts += 1
                if self.attempts == 1:
                    raise FatalWorkerError("token broken")
                return token_idx

            def on_result(self, result):
                self.result_value = result

        pool = TokenWorkerPool(["token1", "token2"])
        pool.start()
        try:
            task = FatalOnceTask()
            pool.submit(task)
            assert pool.wait_all_done(timeout=5.0) is True
            pool.drain_results()
            assert task.attempts == 2
            assert task.result_value is not None
            assert execution_workers == [execution_workers[0], task.result_value]
            assert execution_workers[0] != execution_workers[1]
        finally:
            pool.shutdown()

    def test_shutdown_idempotent(self):
        """多次 shutdown 不应报错。"""
        from github_hot_projects.tasks.worker_pool import TokenWorkerPool
        pool = TokenWorkerPool(["token1"])
        pool.start()
        pool.shutdown()
        pool.shutdown()  # 第二次不应报错
