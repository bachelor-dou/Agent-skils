"""
统一 Worker Pool
=================
N 个 Worker 线程各绑定一个 GitHub Token，从共享任务队列中取任务执行。

核心机制：
  - Worker 绑定 Token：初始化时一次性绑定，无锁竞争
  - 统一任务队列：Phase 1（搜索/扫描）和 Phase 2（增长计算）复用同一个池
  - 限流处理：Worker 自行 sleep 到恢复时间，回退当前任务到队列
  - Token 失效：Worker 退出 + 回退任务（由其他 Worker 消费）+ 日志告警
  - 结果收集：result_queue 由主线程 wait_all_done 后单线程消费
"""

import logging
import threading
import time
from queue import Queue, Empty
from typing import Any

from ..common.exceptions import FatalWorkerError, RetryableError
from .task_base import Task

logger = logging.getLogger("discover_hot")


# ══════════════════════════════════════════════════════════════
# TokenWorkerPool
# ══════════════════════════════════════════════════════════════

_SENTINEL = None  # 停止信号


class TokenWorkerPool:
    """
    通用任务线程池：N 个 Worker，每个绑定一个 GitHub Token。

    使用方式::

        pool = TokenWorkerPool(tokens=["ghp_aaa", "ghp_bbb", "ghp_ccc"])
        pool.start()

        pool.submit(ScanSegmentTask(...))
        pool.submit(KeywordSearchTask(...))
        pool.wait_all_done()

        while not pool.result_queue.empty():
            task, result, err = pool.result_queue.get()
            ...

        pool.submit(CalcGrowthTask(...))
        pool.wait_all_done()
        ...

        pool.shutdown()
    """

    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self._task_queue: Queue[Task | None] = Queue()
        self.result_queue: Queue[tuple[Task, Any, Exception | None]] = Queue()
        self._workers: list[threading.Thread] = []
        self._active_workers = len(tokens)
        self._active_lock = threading.Lock()
        self._pending_count = 0
        self._pending_lock = threading.Lock()
        self._all_done = threading.Event()
        self._all_done.set()

    def start(self) -> None:
        """启动所有 Worker 线程。"""
        for i in range(len(self.tokens)):
            t = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                name=f"Worker-{i}",
                daemon=True,
            )
            self._workers.append(t)
            t.start()
        logger.info(f"TokenWorkerPool 启动: {len(self.tokens)} 个 Worker")

    def submit(self, task: Task) -> None:
        """提交单个任务到队列。"""
        with self._pending_lock:
            self._pending_count += 1
            self._all_done.clear()
        self._task_queue.put(task)

    def wait_all_done(self, timeout: float | None = None) -> bool:
        """阻塞等待所有已提交任务完成。返回 True 表示正常完成。"""
        return self._all_done.wait(timeout=timeout)

    def drain_results(self) -> int:
        """
        消费 result_queue 中所有结果，调用每个 Task 的 on_result/on_error。

        应在 wait_all_done() 之后调用（单线程安全）。
        返回消费的任务数量。
        """
        count = 0
        while not self.result_queue.empty():
            task, result, err = self.result_queue.get()
            count += 1
            try:
                if err:
                    task.on_error(err)
                else:
                    task.on_result(result)
            except Exception as callback_err:
                logger.error(
                    f"任务回调异常 [{task}]: {callback_err}",
                    exc_info=True,
                )
        return count

    def shutdown(self) -> None:
        """发送停止信号并等待所有 Worker 退出。"""
        for _ in self._workers:
            self._task_queue.put(_SENTINEL)
        for t in self._workers:
            t.join(timeout=10)
        logger.info("TokenWorkerPool 已关闭。")

    @property
    def active_workers(self) -> int:
        with self._active_lock:
            return self._active_workers

    # ────────── 内部方法 ──────────

    def _mark_task_done(self) -> None:
        with self._pending_lock:
            self._pending_count -= 1
            if self._pending_count <= 0:
                self._pending_count = 0
                self._all_done.set()

    def _worker_exit(self, worker_id: int, reason: Exception | None = None) -> None:
        """Worker 退出时调用，检查是否所有 Worker 都已退出。"""
        with self._active_lock:
            self._active_workers -= 1
            remaining = self._active_workers

        if remaining == 0:
            logger.critical("⚠️ 所有 Token 均已失效，无法继续执行！清空剩余任务。")
            dropped = 0
            while not self._task_queue.empty():
                try:
                    t = self._task_queue.get_nowait()
                    if t is not _SENTINEL:
                        dropped += 1
                        err = reason or FatalWorkerError("所有 Worker 已退出，任务被终止")
                        self.result_queue.put((t, None, err))
                        self._mark_task_done()
                except Empty:
                    break
            if dropped:
                logger.error(f"所有 Worker 退出后丢弃任务 {dropped} 个，已标记为失败。")
            self._all_done.set()

    def _worker_loop(self, worker_id: int) -> None:
        """Worker 主循环：取任务 → 执行 → 处理异常。"""
        token_idx = worker_id
        token = self.tokens[token_idx]
        logger.info(f"Worker-{worker_id} 启动，绑定 Token#{token_idx} ({token[:8]}...)")

        while True:
            task = self._task_queue.get()
            if task is _SENTINEL:
                logger.info(f"Worker-{worker_id} 收到停止信号，退出。")
                break

            try:
                idx = token_idx if task.needs_token else None
                result = task.execute(idx)
                self.result_queue.put((task, result, None))
                self._mark_task_done()

            except FatalWorkerError as e:
                logger.error(
                    f"Worker-{worker_id} 致命错误: {e}，"
                    f"退出并回退任务 [{task}]"
                )
                self._task_queue.put(task)
                self._worker_exit(worker_id, e)
                break

            except RetryableError as e:
                wait = max(0, e.reset_time - time.time()) + 5
                logger.warning(
                    f"Worker-{worker_id} 可重试错误: {e}，"
                    f"等待 {int(wait)}s，回退任务 [{task}]"
                )
                time.sleep(wait)
                self._task_queue.put(task)
                # 不 mark done，不 break — 恢复后继续取任务

            except Exception as e:
                logger.error(f"Worker-{worker_id} 任务 [{task}] 异常: {e}")
                self.result_queue.put((task, None, e))
                self._mark_task_done()
