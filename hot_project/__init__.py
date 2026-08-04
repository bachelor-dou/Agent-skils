"""hot_project —— GitHub 热门项目发现系统(服务端重构版)。设计与计划见 docs/superpowers/。

分层由 tests/test_layering.py 自动守卫,上层依赖下层,下层永不 import 上层:

    顶层入口脚本 → web → agent → tools → service → provider → infra → config → common

`common/` 只收零项目知识的小工具(出现 star / 仓库 / token 这类词就不许进),`infra/` 收
有状态但不懂产品的机制。顶层入口脚本是唯一知道全部接线的地方,它不该被解耦。
"""
