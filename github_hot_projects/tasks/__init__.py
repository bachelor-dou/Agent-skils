"""
tasks — 任务系统子包
======================
Task 基类、数据采集任务子类、Worker Pool。

子模块：
  - task_base.py    — Task 抽象基类（execute / on_result / on_error）
  - task.py         — 3 个具体任务子类 + checkpoint 断点续传
                       · KeywordSearchTask  — 关键词搜索
                       · ScanSegmentTask    — Star 范围扫描（支持失败页级补偿）
                       · CalcGrowthTask     — 单仓库增长估算
  - worker_pool.py  — TokenWorkerPool：多 Token 绑定 Worker 线程池，
                       支持限流重试、Token 失效转移、超时防护
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
