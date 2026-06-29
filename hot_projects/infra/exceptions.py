"""
自定义异常
==========
AsyncTaskDispatcher 与 GitHub API 共用的异常层级：

  RetryableError        — 任务回队重试
    └─ RateLimitError   — GitHub API 限流（403/429）
  FatalWorkerError      — 本次尝试失败，任务回队（worker 不退出）
    └─ TokenInvalidError — Token 已失效（401）
"""


class RetryableError(Exception):
    """任务应回退重试（Worker sleep 后重新入队）。"""

    def __init__(self, reset_time: float, message: str = ""):
        self.reset_time = reset_time
        super().__init__(message)


class FatalWorkerError(Exception):
    """本次尝试致命错误；dispatcher 将任务重新入队，worker 继续运行。"""

    pass


class RateLimitError(RetryableError):
    """GitHub API 限流（403/429）。"""

    def __init__(self, token_idx: int, reset_time: float):
        self.token_idx = token_idx
        super().__init__(reset_time, f"Token#{token_idx} rate limited until {reset_time}")


class TokenInvalidError(FatalWorkerError):
    """GitHub Token 已失效（401 Unauthorized）。"""

    def __init__(self, token_idx: int, message: str = ""):
        self.token_idx = token_idx
        super().__init__(message or f"Token#{token_idx} invalid (401)")
