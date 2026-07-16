"""tools/tool — 所有暴露给 LLM 的工具（一工具一文件，文件名 = 工具名）。

- ranking.py：复合榜单工具（综合/新项目/关键词），内部编排 basic 能力 + 缓存 + 确认守卫；
- 其余为独立工具：repo_growth / describe_project / repo_profile / search_repos /
  analyze_report / get_db_info / fetch_trending。

公用底层能力见 tools/basic；注册与 schema 见 tools/registry.py、tools/schemas.py。
"""
