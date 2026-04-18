"""
执行层 (Execution Layer)
========================
统一的项目发现工作流管道，消除 scheduled_update.py 与 Agent 之间的逻辑重复。
"""

from .pipeline import DiscoveryPipeline

__all__ = ["DiscoveryPipeline"]
