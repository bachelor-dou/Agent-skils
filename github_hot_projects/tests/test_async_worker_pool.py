"""Integration-style tests for async worker pool dispatcher."""

import asyncio
from dataclasses import dataclass
from typing import Any

from github_hot_projects.common.async_token_pool import AsyncTokenPool
from github_hot_projects.tasks.async_worker_pool import AsyncTaskDispatcher
from github_hot_projects.tasks.task_base import Task


class TestAsyncTaskDispatcher:
    def test_tokenless_task_runs_without_token_pool(self):
        @dataclass
        class SimpleTask(Task):
            needs_github_token: bool = False
            result_value: Any = None

            def execute(self, token_idx=None):
                assert token_idx is None
                return "ok"

            def on_result(self, result):
                self.result_value = result

        async def scenario() -> None:
            dispatcher = AsyncTaskDispatcher(token_pool=None, worker_count=1)
            await dispatcher.start()
            try:
                task = SimpleTask()
                await dispatcher.submit(task)
                assert await dispatcher.wait_all_done(timeout=1.0) is True
                await dispatcher.drain_results()
                assert task.result_value == "ok"
            finally:
                await dispatcher.shutdown()

        asyncio.run(scenario())

    def test_basic_task_execution(self):
        @dataclass
        class SimpleTask(Task):
            needs_github_token: bool = False
            result_value: Any = None

            def execute(self, token_idx=None):
                assert token_idx is None
                return 42

            def on_result(self, result):
                self.result_value = result

        async def scenario() -> None:
            token_pool = AsyncTokenPool(["token1"], recovery_buffer_seconds=0.0)
            dispatcher = AsyncTaskDispatcher(token_pool=token_pool, worker_count=2)
            await dispatcher.start()
            try:
                task = SimpleTask()
                await dispatcher.submit(task)
                assert await dispatcher.wait_all_done(timeout=1.0) is True
                await dispatcher.drain_results()
                assert task.result_value == 42
            finally:
                await dispatcher.shutdown()

        asyncio.run(scenario())

    def test_idempotency_deduplicates_duplicate_tasks(self):
        @dataclass
        class DedupTask(Task):
            needs_github_token: bool = False
            key: str = "k"
            result_value: int | None = None
            run_counter: list[int] | None = None

            def execute(self, token_idx=None):
                if self.run_counter is not None:
                    self.run_counter[0] += 1
                return 1

            def on_result(self, result):
                self.result_value = result

            def idempotency_key(self) -> str:
                return self.key

        async def scenario() -> None:
            token_pool = AsyncTokenPool(["token1"], recovery_buffer_seconds=0.0)
            dispatcher = AsyncTaskDispatcher(token_pool=token_pool, worker_count=2)
            await dispatcher.start()
            try:
                counter = [0]
                task1 = DedupTask(key="same", run_counter=counter)
                task2 = DedupTask(key="same", run_counter=counter)
                await dispatcher.submit(task1)
                await dispatcher.submit(task2)
                assert await dispatcher.wait_all_done(timeout=1.0) is True
                await dispatcher.drain_results()
                metrics = await dispatcher.snapshot_metrics()
                assert counter[0] == 1
                assert metrics["deduplicated"] == 1
            finally:
                await dispatcher.shutdown()

        asyncio.run(scenario())
