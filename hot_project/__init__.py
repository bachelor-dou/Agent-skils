"""hot_project —— GitHub 热门项目发现系统(服务端重构版)。

与旧包 `hot_projects/` 并存:旧包继续跑 CI 保证每日快照不断链,本包逐阶段建起底座,
影子并跑比对一致后再切换。设计与计划见 docs/superpowers/。

目录形状贴着旧包,只有 `common/` 是新增的:

    config.py              全局配置(顶层单文件,和旧包同位置,含搜索关键词表)
    api_server.py          HTTP 服务入口(只剩路由)
    agent_cli.py           命令行入口
    cron_*.py              定时任务入口
    common/            ← 新增:零项目知识的小工具(时间、环境变量、日志),换个项目能原样拷走
    core/                  纯算法:增长、打分、淘汰判定、报告解析(不碰 I/O)
    infra/                 机制:store(数据层) / tasks(任务池) / llm / notify / exceptions
    provider/            出站:base.py + github/
    tools/                 agent 可执行能力
    agent/                 对话核心:prompts / history / loop
    web/                   网页端:静态文件 + 渲染 / 会话池 / 安全中间件
    data/ report/ logs/ tests/

分层(上层依赖下层,下层永不 import 上层),由 tests/test_layering.py 自动守卫:

    顶层入口脚本 → web → agent → tools → provider → infra → core → config → common

`web` 压在 `agent` 之上而不是平级:网页端每个会话持有一个 Agent,反过来 agent 不该知道
有个网页存在 —— 它同样服务于 CLI。

没有 `entry/` 这种包:入口放顶层一眼能看见。

`common/` 是最容易长成杂物间的一格,所以它的成员判据被写成了可执行的测试:
**零项目知识** —— 出现 star / 仓库 / 快照 / token 这类词就不许进,
换个完全不同的项目它每个文件都能原样拷走。两个共用层的判据互不重叠:

    common   零项目知识              时间、环境变量
    infra    有状态的机制,不懂产品   文件锁、任务池、LLM 客户端

**纯算法只在两种情况下才提进 `core/`:多个调用方共用,或者需要这道边界挡住 I/O。**
增长相减属于前者(周报排名、爆发探针、单仓库查询用同一套),打分属于后者(旧版那个
打分函数里有一行 `db.get(...)`,于是验证一次排名要先造一个 DB)。只有一个调用方又不碰
I/O 的就留在原地:淘汰判定在 `cron_daily_snapshot.py`、Trending 解析在
`provider/github/trending.py` —— 为它们隔一层买到的边界是零,代价是读一件事要开两个文件。

顶层入口脚本是**唯一知道全部接线的地方**(建 token 池、注入给任务池、排屏障),
它不该被解耦 —— 否则接线逻辑会散回各层,重新长成旧包 `tools/basic/core.py` 那种
「既是工具实现、又是收集编排、又是任务池创建者、又是 DB 写入方」的上帝模块。
"""
