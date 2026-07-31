"""任务池:每条道一个队列 + 固定数量的 worker。设计取舍写在 `__init__.py`。

「全部完成」用跨所有道的 `_pending` 计数,不能用「所有队列都空了」:任务可以派生任务,
父任务跑到一半时队列可能恰好是空的。提交时 +1,**终态**时 -1;重排不动它,派生任务在父任务
减一之前就已经加一,所以计数不会中途归零。

异常分三类:`RateLimitError` 回队且**不计入重试次数**(外部节流不是任务的错,计入的话一轮
限流高峰就能烧光所有任务的额度);`RetryableError` 回队并计入,超限按失败收尾;其他当场
失败、不重排。`CancelledError` 必须原样抛出,不能被「其他」那一支吞掉,否则关不掉池子。
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AbstractAsyncContextManager
from typing import Any, Callable

from ..exceptions import RateLimitError, RetryableError, TokenInvalidError
from .task import Ctx, Task

logger = logging.getLogger("hot_project")

# 给 leaser 的类型:传入任务自称的 token 种类,还回来一个异步上下文管理器。
Leaser = Callable[[str], AbstractAsyncContextManager[Any]]


class _NoToken:
    """`needs_token = False` 时用的空租约,省掉 worker 里的一个 if 分支。"""

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc: object) -> bool:
        return False


class TaskPool:
    """多道任务池。

        pool = TaskPool({"search": 6, "graphql": 4}, leaser=my_leaser)
        async with pool:
            pool.submit(task)
            await pool.join()
    """

    def __init__(self, lanes: dict[str, int], *, leaser: Leaser | None = None) -> None:
        if not lanes:
            raise ValueError("至少要有一条道")
        bad = [name for name, n in lanes.items() if n < 1]
        if bad:
            raise ValueError(f"这些道的 worker 数不是正整数:{bad}")

        self._sizes = dict(lanes)
        self._leaser = leaser
        self._queues: dict[str, asyncio.Queue[Task]] = {n: asyncio.Queue() for n in lanes}
        self._workers: list[asyncio.Task[None]] = []
        self._pending = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self.stats = {"submitted": 0, "done": 0, "failed": 0,
                      "retried": 0, "rate_limited": 0}

    # ── 生命周期 ──────────────────────────────────────────

    async def __aenter__(self) -> TaskPool:
        for lane, size in self._sizes.items():
            for i in range(size):
                self._workers.append(
                    asyncio.create_task(self._worker(lane), name=f"{lane}-{i}")
                )
        logger.info("任务池启动:%s。",
                    ", ".join(f"{k}×{v}" for k, v in self._sizes.items()))
        return self

    async def __aexit__(self, *exc: object) -> bool:
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("任务池关闭:%s。", self.stats)
        return False

    # ── 提交与等待 ────────────────────────────────────────

    def submit(self, task: Task) -> None:
        """投递一个任务。同步方法 —— 任务派生子任务时不该被 await 卡住。"""
        queue = self._queues.get(task.lane)
        if queue is None:
            raise KeyError(
                f"{task!r} 要走 {task.lane!r} 道,但池里只有 {sorted(self._queues)}"
            )
        self._pending += 1
        self._idle.clear()
        self.stats["submitted"] += 1
        queue.put_nowait(task)

    async def join(self) -> None:
        """等到所有任务(含派生出来的)都进入终态。"""
        await self._idle.wait()

    # ── worker ────────────────────────────────────────────

    async def _worker(self, lane: str) -> None:
        queue = self._queues[lane]
        while True:
            task = await queue.get()
            await self._run_once(task, queue)

    async def _run_once(self, task: Task, queue: asyncio.Queue[Task]) -> None:
        task.attempts += 1
        try:
            async with self._lease_for(task) as token:
                result = await task.run(Ctx(token=token, submit=self.submit))

        except asyncio.CancelledError:
            raise                                   # 关池子,别当成任务失败

        except RateLimitError as e:
            # 不计入重试次数:外部节流不是这个任务的问题。租约已经记过账并冷却了那个 token,
            # 直接回队即可,下次会拿到另一个。另记一本有界的账,理由见 `Task.max_rate_limits`。
            task.attempts -= 1
            task.rate_limits += 1
            self.stats["rate_limited"] += 1
            if task.rate_limits > task.max_rate_limits:
                logger.error("%r 连撞 %d 次限流,放弃。", task, task.rate_limits)
                self._finish(task, err=e)
            else:
                queue.put_nowait(task)

        except (RetryableError, TokenInvalidError) as e:
            # 401 也归这里:租约已按 strikes 处置过了,换个 token 多半就好。但**计入**次数,
            # 免得 token 真的集体坏掉时无限自旋。
            if task.attempts > task.max_retries:
                self._finish(task, err=e)
            else:
                self.stats["retried"] += 1
                queue.put_nowait(task)

        except Exception as e:                      # noqa: BLE001 - 兜底,不重排
            self._finish(task, err=e)

        else:
            self._finish(task, result=result)

    def _lease_for(self, task: Task) -> AbstractAsyncContextManager[Any]:
        if not task.needs_token:
            return _NoToken()
        if self._leaser is None:
            raise RuntimeError(f"{task!r} 需要 token,但池子没接 leaser")
        return self._leaser(task.token_kind)

    def _finish(self, task: Task, *, result: Any = None, err: BaseException | None = None) -> None:
        """终态收尾:跑回调、减计数。回调抛异常不能影响计数,否则 `join()` 永远不返回。"""
        try:
            if err is None:
                task.on_done(result)
            else:
                task.on_error(err)
        except Exception:                           # noqa: BLE001
            logger.exception("%r 的回调抛了异常", task)
        finally:
            self.stats["done" if err is None else "failed"] += 1
            self._pending -= 1
            if self._pending <= 0:
                self._idle.set()
