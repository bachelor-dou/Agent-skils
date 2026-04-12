"""
github-hot-projects — GitHub 热门项目发现工具
=============================================
按多类别关键词搜索 + Star 范围扫描 GitHub 仓库，
根据近期 star 增长筛选候选，排序取 Top N 并调用 LLM 生成详细描述。

模块结构：
  config.py           — 全局配置（Token、阈值、路径、关键词）
  token_manager.py    — GitHub Token 轮换管理器（线程安全）
  github_api.py       — GitHub REST / GraphQL API 封装
  growth_estimator.py — Star 增长估算（二分法 + 采样外推）
  db.py               — DB 读写（Github_DB.json）
  llm.py              — LLM 描述生成
  pipeline.py         — 核心流程编排（step1/1b/2/3 + main）

用法：
  python -m github-hot-projects
"""

from .pipeline import main

__all__ = ["main"]
