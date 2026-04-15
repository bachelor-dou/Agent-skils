"""
Task 基类
=========
并行任务的抽象基类，定义 Worker Pool 任务接口。

扩展新任务类型只需：
  1. 继承 Task
  2. 设置 needs_token（是否需要 GitHub Token）
  3. 实现 execute(token_idx) 方法
  4. 可选实现 on_result() / on_error() 回调

execute() 可抛出的异常（由 Worker 统一处理）：
  - FatalWorkerError / TokenInvalidError → Worker 退出并回退任务
  - RetryableError / RateLimitError      → Worker sleep 后回退任务重试
  - 其他 Exception                       → 记录错误，标记任务完成（不回退）
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Task(ABC):
    """并行任务抽象基类。"""

    needs_token: bool = True
    _token_mgr: Any = None  # TokenManager 引用，由子类构造时设置

    @abstractmethod
    def execute(self, token_idx: int | None) -> Any:
        """
        执行任务。

        Args:
            token_idx: 绑定的 token 索引。needs_token=False 时为 None。

        Returns:
            任务结果，类型由子类定义。
        """
        ...

    def on_result(self, result: Any) -> None:
        """结果处理回调，由主线程在 wait_all_done 后调用。子类可覆盖。"""
        pass

    def on_error(self, error: Exception) -> None:
        """错误处理回调，由主线程在 wait_all_done 后调用。子类可覆盖。"""
        pass

    def __str__(self) -> str:
        return self.__class__.__name__
