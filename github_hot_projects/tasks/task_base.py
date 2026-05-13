"""
Task 基类
=========
并行任务的抽象基类，定义 Worker Pool 任务接口。

扩展新任务类型只需：
  1. 继承 Task
    2. 按需设置 needs_github_token（是否需要 GitHub Token）
  3. 实现 execute(token_idx) 方法
  4. 可选实现 on_result() / on_error() 回调

execute() 可抛出的异常（由 Worker 统一处理）：
  - FatalWorkerError / TokenInvalidError → Worker 退出并回退任务
  - RetryableError / RateLimitError      → Worker sleep 后回退任务重试
  - 其他 Exception                       → 记录错误，标记任务完成（不回退）
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Task(ABC):
    """并行任务抽象基类。

    A模式：任务级 token 持有。
    当 needs_github_token=True 时，由 Dispatcher 在任务开始前 acquire，
    在任务返回后 release。当前默认协程任务优先使用这条链路。

    B模式：请求级 token 借还备份。
    任务本身不声明 needs_github_token，由下游 GitHub API helper 在每次
    HTTP 请求前临时 acquire，请求结束后立刻 release。相关调用在子类中
    以注释形式保留，便于需要时回切。
    """

    needs_github_token: bool = False  # A模式：任务级 token 持有
    _token_mgr: Any = None  # GitHubTokenPool 引用，由子类构造时设置

    @abstractmethod
    def execute(self, token_idx: int | None) -> Any:
        """
        执行任务。

        Args:
            token_idx: A模式下绑定的 token 索引；B模式下通常为 None，
                表示由请求级 helper 自行借还 token。

        Returns:
            任务结果，类型由子类定义。
        """
        ...

    async def execute_async(self, token_idx: int | None) -> Any:
        """异步执行入口（默认桥接到同步 execute）。"""
        return await asyncio.to_thread(self.execute, token_idx)

    def idempotency_key(self) -> str | None:
        """返回任务幂等键；默认不做去重。"""
        return None

    def on_result(self, result: Any) -> None:
        """结果处理回调，由主线程在 wait_all_done 后调用。子类可覆盖。"""
        pass

    def on_error(self, error: Exception) -> None:
        """错误处理回调，由主线程在 wait_all_done 后调用。子类可覆盖。"""
        pass

    def __str__(self) -> str:
        return self.__class__.__name__
