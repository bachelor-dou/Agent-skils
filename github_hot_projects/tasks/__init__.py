"""
tasks — 任务系统子包
======================
Task 基类、数据采集任务子类、Worker Pool。
"""

from .task_base import Task
from .task import (
    KeywordSearchTask,
    ScanSegmentTask,
    CalcGrowthTask,
    _upsert_candidate,
    _load_checkpoint,
    _save_checkpoint,
    _remove_checkpoint,
    _submit_growth_tasks,
)
from .worker_pool import TokenWorkerPool

__all__ = [
    "Task",
    "KeywordSearchTask",
    "ScanSegmentTask",
    "CalcGrowthTask",
    "TokenWorkerPool",
    "_upsert_candidate",
    "_load_checkpoint",
    "_save_checkpoint",
    "_remove_checkpoint",
    "_submit_growth_tasks",
]
