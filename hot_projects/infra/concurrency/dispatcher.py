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

from ...datasource.github.token_pool import AsyncTokenPool
from ..exceptions import FatalWorkerError, RateLimitError, RetryableError, TokenInvalidError
from .task_base import Task

logger = logging.getLogger("hot_projects")

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

    def _run_callback(self, task: Task, result: Any, err: Exception | None) -> None:
        """同步执行任务回调（on_result/on_error）。

        on_result/on_error 为纯内存操作 + 同步落盘，无 await；在单线程事件循环里
        由 worker 内联调用是原子的（不会与其它协程交错），从而让 checkpoint/候选
        在运行过程中实时更新，而非堆到整轮结束——这样断点续传才真正有效。
        """
        try:
            if err is not None:
                task.on_error(err)
            else:
                task.on_result(result)
        except Exception as callback_err:  # pragma: no cover - defensive logging
            logger.error("异步任务回调异常 [%s]: %s", task, callback_err, exc_info=True)

    async def drain_results(self) -> int:
        """兜底排空 result_queue（正常路径下回调已在 worker 内联执行，此处通常为空）。"""
        count = 0
        while not self.result_queue.empty():
            task, result, err = self.result_queue.get_nowait()
            count += 1
            self._run_callback(task, result, err)
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
            # [DIAG-P0] 仅诊断：记录抢 token 等待时长与持有时长，不改变任何控制流。
            _diag_wait_start = time.time()
            _diag_acquired_at: float | None = None
            try:
                # ── A模式（当前启用）：任务级 token 持有，任务开始时 acquire，结束时 release ──
                if task.needs_github_token:
                    if self.token_pool is None:
                        raise RuntimeError(
                            "Task requires GitHub token but dispatcher has no token pool"
                        )
                    token_idx = await self.token_pool.acquire()
                    _diag_acquired_at = time.time()
                # ── B模式（已禁用）：请求级 token 借还，token_idx 保持 None，由增长链路内部自行管理 ──
                # if task.needs_github_token:
                #     pass  # token_idx 保持 None，不在此处 acquire

                result = await task.execute_async(token_idx)

                # A模式：任务完成后释放 token。
                if token_idx is not None:
                    await self.token_pool.release(token_idx)

                self._diag_log_token_usage(task, token_idx, _diag_wait_start, _diag_acquired_at, "ok")
                self._run_callback(task, result, None)  # 实时执行回调（checkpoint/候选即时落盘）
                self._metrics["completed"] += 1
                self._finalize_task_tracking(task)
                await self._mark_task_done()

            except RateLimitError as e:
                if token_idx is not None:
                    await self.token_pool.mark_rate_limited(token_idx, e.reset_time, str(e))
                self._diag_log_token_usage(task, token_idx, _diag_wait_start, _diag_acquired_at, "ratelimit")
                self._metrics["requeued_rate_limited"] += 1
                await self._enqueue(task, delay_seconds=0.0)

            except TokenInvalidError as e:
                if token_idx is not None:
                    # 401 不再无条件永久失效：交给 token 池按 strikes 判定（瞬时冷却 / 多次才永久）。
                    await self.token_pool.mark_auth_failed(token_idx, str(e))
                self._diag_log_token_usage(task, token_idx, _diag_wait_start, _diag_acquired_at, "token_invalid")
                self._metrics["requeued_fatal"] += 1
                await self._enqueue(task, delay_seconds=0.0)

            except RetryableError as e:
                if token_idx is not None:
                    await self.token_pool.release(token_idx)
                self._diag_log_token_usage(task, token_idx, _diag_wait_start, _diag_acquired_at, "retryable")
                task.requeue_count += 1
                if task.requeue_count > task.max_requeue:
                    # 瞬时故障重排超过上限（如持续性网络问题）：放弃本轮，按失败收尾，
                    # 避免无限重排自旋。任务的 on_error 会落 checkpoint，下一轮可重算。
                    logger.warning(
                        "任务 %s 重排超过上限(%d)，放弃本轮: %s",
                        task, task.max_requeue, e,
                    )
                    self._run_callback(task, None, e)
                    self._metrics["failed"] += 1
                    self._finalize_task_tracking(task)
                    await self._mark_task_done()
                else:
                    delay = max(0.0, e.reset_time - time.time())
                    self._metrics["requeued_retryable"] += 1
                    await self._enqueue(task, delay_seconds=delay)

            except FatalWorkerError as e:
                if token_idx is not None:
                    await self.token_pool.release(token_idx)
                self._diag_log_token_usage(task, token_idx, _diag_wait_start, _diag_acquired_at, "fatal")
                logger.warning("Async worker-%d 命中 FatalWorkerError，任务回队: %s", worker_id, e)
                self._metrics["requeued_fatal"] += 1
                await self._enqueue(task, delay_seconds=0.0)

            except Exception as e:
                if token_idx is not None:
                    await self.token_pool.release(token_idx)
                self._diag_log_token_usage(task, token_idx, _diag_wait_start, _diag_acquired_at, "error")
                self._run_callback(task, None, e)  # 实时执行回调
                self._metrics["failed"] += 1
                self._finalize_task_tracking(task)
                await self._mark_task_done()

    def _diag_log_token_usage(
        self,
        task: Task,
        token_idx: int | None,
        wait_start: float,
        acquired_at: float | None,
        outcome: str,
    ) -> None:
        """[DIAG-P0] 仅诊断：输出本次任务的 token 等待/持有时长与结果。

        wait  = 从开始抢 token 到拿到 token 的耗时（worker 饥饿信号）。
        hold  = 从拿到 token 到释放/标记的耗时（token 被占信号）。
        覆盖所有完成路径（含 growth=0 等不打 [GROWTH] 的快路径），消除按
        [GROWTH] 日志统计完成数的偏差。该方法不改变任何调度行为。
        """
        if acquired_at is None:
            return
        now = time.time()
        logger.debug(
            "[DIAG] task=%s token=%s wait=%.2fs hold=%.2fs outcome=%s",
            task, token_idx, acquired_at - wait_start, now - acquired_at, outcome,
        )

    async def _enqueue(self, task: Task, delay_seconds: float = 0.0) -> None:
        run_at = time.time() + max(0.0, delay_seconds)
        # 软公平：仅在「混合任务类型」场景下，给堆积的同类任务追加极小时间片，
        # 避免某一类任务长期插队饿死另一类。注意该增量被 _fairness_max_delay_seconds
        # 硬性封顶（当前 20ms），因此对「单一类型大批量」（如 5 万个增长任务全同类）
        # 几乎无影响，不是吞吐瓶颈，仅作防极端饥饿的占位。
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
