"""
github-hot-projects — GitHub 热门项目发现工具
=============================================
按多类别关键词搜索 + Star 范围扫描 GitHub 仓库，
根据近期 star 增长筛选候选，排序取 Top N 并调用 LLM 生成详细描述。

模块结构：
  common/                 — 基础设施子包
    config.py             — 全局配置（Token、阈值、路径、关键词）
    db.py                 — DB 读写（Github_DB.json）
    token_manager.py      — GitHub Token 管理 + 请求头构建
    exceptions.py         — 自定义异常（限流、Token 失效等）
    github_api.py         — GitHub REST / GraphQL API 封装
    llm.py                — LLM 描述生成

  tasks/                  — 任务系统子包
    task_base.py          — Task 抽象基类
    task.py               — Task 子类（搜索/扫描/增长）+ 批量提交 + 断点续传
    worker_pool.py        — TokenWorkerPool 线程池

  ranking.py              — 评分排序算法
  report.py               — Markdown 报告生成
  growth_estimator.py     — Star 增长估算（二分法 + 采样外推）
  github_trending.py      — GitHub Trending 爬虫
  agent_tools.py          — 9 个 Tool 函数 + TOOL_SCHEMAS
  agent.py                — ReAct Agent 循环
  agent_cli.py            — Agent CLI 入口
  api_server.py           — FastAPI REST 接口

运行方式：
  Agent CLI:  python -m github_hot_projects.agent_cli
  API Server: uvicorn github_hot_projects.api_server:app
"""
