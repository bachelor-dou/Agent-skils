"""concurrency — 并发任务系统（协程调度器 + Task 子类 + checkpoint）。

只 re-export 包外真正用到的名字（全部在 tools/basic/core.py）。
Task 基类、_upsert_candidate、_load_checkpoint 都只在包内部用：
前者由 dispatcher/tasks 直接从 task_base 引，后两者是 _resolve_growth 的内部步骤。
把它们摆在这里只会让人以为存在包外调用方。
"""

from .tasks import (
    KeywordSearchTask,
    ScanSegmentTask,
    TrendingPeriodTask,
)
from .task_help import (
    _save_checkpoint,
    _remove_checkpoint,
    _resolve_growth,
)
from .dispatcher import AsyncTaskDispatcher

__all__ = [
    "KeywordSearchTask",
    "ScanSegmentTask",
    "TrendingPeriodTask",
    "AsyncTaskDispatcher",
    "_save_checkpoint",
    "_remove_checkpoint",
    "_resolve_growth",
]
