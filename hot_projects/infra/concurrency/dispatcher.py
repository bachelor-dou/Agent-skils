"""
基于 PriorityQueue 的异步任务调度器。

核心行为：
- 使用 next_run_at 支持延迟重试任务
- 任务并发度与 token 数量解耦
- 回调行为与线程池版本保持一致（on_result/on_error）
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from collections import defaultdict
from typing import Any

from ...providers.github.token_pool import AsyncTokenPool
from ..exceptions import FatalWorkerError, RateLimitError, RetryableError, TokenInvalidError
from .task_base import Task

logger = logging.getLogger("discover_hot")

_SENTINEL = object()


class AsyncTaskDispatcher:
    """面向 Task 对象的协程调度器。"""

    def __init__(self, token_pool: AsyncTokenPool | None, worker_count: int) -> None:
        self.token_pool = token_pool
        self.worker_count = max(1, worker_count)
        self._queue: asyncio.PriorityQueue[tuple[float, int, Task | object]] = asyncio.PriorityQueue()
        self.result_queue: asyncio.Queue[tuple[Task, Any, Exception | None]] = asyncio.Queue()

        self._workers: list[asyncio.Task[None]] = []
        self._sequence = itertools.count()
        self._pending_count = 0
        self._pending_lock = asyncio.Lock()
        self._all_done = asyncio.Event()
        self._all_done.set()
        self._running = False
        self._active_task_keys: set[str] = set()
        self._task_type_pending: dict[str, int] = defaultdict(int)
        self._fairness_step_seconds = 0.0005
        self._fairness_max_delay_seconds = 0.02
        self._metrics: dict[str, int] = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "requeued_rate_limited": 0,
            "requeued_retryable": 0,
            "requeued_fatal": 0,
            "deduplicated": 0,
        }

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        for i in range(self.worker_count):
            self._workers.append(
                asyncio.create_task(self._worker_loop(i), name=f"AsyncWorker-{i}")
            )
        logger.info("AsyncTaskDispatcher 启动: workers=%d", self.worker_count)

    async def submit(self, task: Task, delay_seconds: float = 0.0) -> None:
        if not self._running:
            raise RuntimeError("AsyncTaskDispatcher is not started")

        task_key = task.idempotency_key()
        if task_key and task_key in self._active_task_keys:
            self._metrics["deduplicated"] += 1
            logger.info("幂等去重: 跳过重复任务 key=%s", task_key)
            return

        if task_key:
            self._active_task_keys.add(task_key)
        self._task_type_pending[task.__class__.__name__] += 1
        self._metrics["submitted"] += 1

        await self._increase_pending()
        await self._enqueue(task, delay_seconds=delay_seconds)

    async def wait_all_done(self, timeout: float | None = None) -> bool:
        try:
            if timeout is None:
                await self._all_done.wait()
            else:
                await asyncio.wait_for(self._all_done.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def drain_results(self) -> int:
        count = 0
        while not self.result_queue.empty():
            task, result, err = self.result_queue.get_nowait()
            count += 1
            try:
                if err is not None:
                    task.on_error(err)
                else:
                    task.on_result(result)
            except Exception as callback_err:  # pragma: no cover - defensive logging
                logger.error("异步任务回调异常 [%s]: %s", task, callback_err, exc_info=True)
        return count

    async def shutdown(self) -> None:
        if not self._running:
            return

        for _ in self._workers:
            await self._queue.put((0.0, next(self._sequence), _SENTINEL))

        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._running = False
        logger.info("AsyncTaskDispatcher 已关闭。")

    async def snapshot_metrics(self) -> dict[str, int]:
        """导出调度器运行指标，便于日志与压测统计。"""
        return dict(self._metrics)

    async def _worker_loop(self, worker_id: int) -> None:
        while True:
            run_at, _seq, item = await self._queue.get()

            if item is _SENTINEL:
                return

            task = item
            now = time.time()
            if run_at > now:
                await asyncio.sleep(run_at - now)

            token_idx: int | None = None
            try:
                # ── A模式（当前启用）：任务级 token 持有，任务开始时 acquire，结束时 release ──
                if task.needs_github_token:
                    if self.token_pool is None:
                        raise RuntimeError(
                            "Task requires GitHub token but dispatcher has no token pool"
                        )
                    token_idx = await self.token_pool.acquire()
                # ── B模式（已禁用）：请求级 token 借还，token_idx 保持 None，由增长链路内部自行管理 ──
                # if task.needs_github_token:
                #     pass  # token_idx 保持 None，不在此处 acquire

                result = await task.execute_async(token_idx)

                # A模式：任务完成后释放 token。
                if token_idx is not None:
                    await self.token_pool.release(token_idx)

                await self.result_queue.put((task, result, None))
                self._metrics["completed"] += 1
                self._finalize_task_tracking(task)
                await self._mark_task_done()

            except RateLimitError as e:
                if token_idx is not None:
                    await self.token_pool.mark_rate_limited(token_idx, e.reset_time, str(e))
                self._metrics["requeued_rate_limited"] += 1
                await self._enqueue(task, delay_seconds=0.0)

            except TokenInvalidError as e:
                if token_idx is not None:
                    await self.token_pool.mark_invalid(token_idx, str(e))
                self._metrics["requeued_fatal"] += 1
                await self._enqueue(task, delay_seconds=0.0)

            except RetryableError as e:
                if token_idx is not None:
                    await self.token_pool.release(token_idx)
                delay = max(0.0, e.reset_time - time.time())
                self._metrics["requeued_retryable"] += 1
                await self._enqueue(task, delay_seconds=delay)

            except FatalWorkerError as e:
                if token_idx is not None:
                    await self.token_pool.release(token_idx)
                logger.warning("Async worker-%d 命中 FatalWorkerError，任务回队: %s", worker_id, e)
                self._metrics["requeued_fatal"] += 1
                await self._enqueue(task, delay_seconds=0.0)

            except Exception as e:
                if token_idx is not None:
                    await self.token_pool.release(token_idx)
                await self.result_queue.put((task, None, e))
                self._metrics["failed"] += 1
                self._finalize_task_tracking(task)
                await self._mark_task_done()

    async def _enqueue(self, task: Task, delay_seconds: float = 0.0) -> None:
        run_at = time.time() + max(0.0, delay_seconds)
        # 软公平：同类任务短时间大量堆积时，追加微小时间片，降低单类任务饥饿风险。
        same_type_pending = self._task_type_pending.get(task.__class__.__name__, 0)
        if same_type_pending > 1:
            run_at += min(
                self._fairness_max_delay_seconds,
                (same_type_pending - 1) * self._fairness_step_seconds,
            )
        await self._queue.put((run_at, next(self._sequence), task))

    def _finalize_task_tracking(self, task: Task) -> None:
        task_key = task.idempotency_key()
        if task_key:
            self._active_task_keys.discard(task_key)
        task_type = task.__class__.__name__
        if task_type in self._task_type_pending:
            self._task_type_pending[task_type] = max(0, self._task_type_pending[task_type] - 1)

    async def _increase_pending(self) -> None:
        async with self._pending_lock:
            self._pending_count += 1
            self._all_done.clear()

    async def _mark_task_done(self) -> None:
        async with self._pending_lock:
            self._pending_count -= 1
            if self._pending_count <= 0:
                self._pending_count = 0
                self._all_done.set()
