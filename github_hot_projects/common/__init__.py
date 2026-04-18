"""
common — 基础设施子包
======================
全局配置、数据库、Token 管理、异常定义、GitHub API、LLM 调用。

子模块：
  - config.py          — 全局参数集中管理（环境变量 + 默认值）
  - db.py              — Github_DB.json 的加载/校验/更新/保存（跨进程文件锁）
  - token_manager.py   — GitHub Token 多账号管理与锁定分配
  - github_api.py      — REST/GraphQL API 封装（搜索、Stargazer、自动分段）
  - llm.py             — LLM 调用接口（项目描述生成、批量浓缩）
  - exceptions.py      — 统一异常层级（RateLimitError / TokenInvalidError / FatalWorkerError）
"""
