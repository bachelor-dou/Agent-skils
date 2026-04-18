"""
执行层 (Execution Layer)
========================
统一的项目发现工作流管道，消除 scheduled_update.py 与 Agent 之间的逻辑重复。

子模块：
  - pipeline.py  — DiscoveryPipeline：端到端 search → scan → trending → growth → rank → report
                    6 步工作流，支持 comprehensive / hot_new 两种模式。

架构定位：
  输入解析层 (parsing/) → Agent 层 (agent.py) → 【执行层】 → 响应层 (api_server/cli)
  本层负责将 9 个 Tool 的调用编排为完整的发现流程，
  agent_tools.py 提供单个 Tool 实现，pipeline.py 提供维约。
"""

from .pipeline import DiscoveryPipeline

__all__ = ["DiscoveryPipeline"]
