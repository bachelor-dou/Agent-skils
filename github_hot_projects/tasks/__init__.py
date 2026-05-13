"""
tasks — 任务系统子包
======================
Task 基类、数据采集任务子类、异步调度器。

子模块：
  - task_base.py    — Task 抽象基类（execute / on_result / on_error）
    - task.py         — 4 个具体任务子类
                       · KeywordSearchTask  — 关键词搜索
                       · ScanSegmentTask    — Star 范围扫描（支持失败页级补偿）
                       · TrendingPeriodTask — Trending 单周期抓取
                       · CalcGrowthTask     — 单仓库增长估算
    - task_help.py    — checkpoint、候选管理、批量增长任务提交等辅助函数
    - async_worker_pool.py — AsyncTaskDispatcher：协程 + PriorityQueue 调度器
"""

from .task_base import Task
from .task import (
    KeywordSearchTask,
    ScanSegmentTask,
    TrendingPeriodTask,
    CalcGrowthTask,
)
from .task_help import (
    _upsert_candidate,
    _load_checkpoint,
    _save_checkpoint,
    _remove_checkpoint,
    _submit_growth_tasks,
)
from .async_worker_pool import AsyncTaskDispatcher

__all__ = [
    "Task",
    "KeywordSearchTask",
    "ScanSegmentTask",
    "TrendingPeriodTask",
    "CalcGrowthTask",
    "AsyncTaskDispatcher",
    "_upsert_candidate",
    "_load_checkpoint",
    "_save_checkpoint",
    "_remove_checkpoint",
    "_submit_growth_tasks",
]
