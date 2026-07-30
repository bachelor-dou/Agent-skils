"""tasks —— 全局任务池:**几条道**,每条一个队列 + 固定数量的协程 worker。

    pool = TaskPool({"search": 6, "graphql": 4, "free": 8}, leaser=...)
    async with pool:
        pool.submit(KeywordPage("llm framework", page=1))
        await pool.join()

外部只有 `submit` 和 `join` 两个入口。有真依赖的任务由调用方按屏障分批提交 ——
池本身不理解阶段。

## 为什么是分道,不是「一条队列 + 每类一个信号量」

信号量版会饿死人:12 个 worker 同时领到并发上限为 4 的 GraphQL 任务,其中 8 个堵在信号量
上**手里还攥着队列里取出来的任务**,于是没有 worker 去做搜索任务 —— 而搜索任务此刻明明
可以跑。要绕开就得让 worker 先窥探再放回,那是自旋。

分道之后:并发度**就是**这条道的 worker 数,没有信号量、没有 `max_concurrency` 属性、
没有三种制度的分支。少一个概念,还顺带换来最大的一笔提速 —— 搜索和 GraphQL 在 GitHub
那边是两份独立预算(2026-07-30 实测:779 个 GraphQL 批跑了 600 秒、一次限流都没有,
却被排在搜索之后干等),分道让它们能同时跑。

三种并发制度于是塌缩成一句话:**这条道开几个 worker,就是几个并发**。

| 道 | worker 数怎么定 |
|---|---|
| search | 够覆盖请求延迟即可;真正的节流在 token 池的配速上,再多也快不了 |
| graphql | 实测值(并发 3 顺畅,8 会让整批 token 被限) |
| free | 不吃 token 也不受限额的活(Trending 抓 HTML、纯本地计算),给宽点 |

## 本包不 import token 池

worker 要把租约递给任务,但 token 池是 GitHub 专属的(它懂 403/401 语义),住在
`provider/github/`。若这里 import 它,横轴设施就反向依赖了纵轴实现,分层守卫会红。

做法是本包只认一个 `Leaser` 可调用对象:给它一个字符串(任务自称需要哪种 token),
还回来一个异步上下文管理器。这个字符串对本包**没有含义** —— "search" 是什么意思由出站层
解释。顶层入口脚本负责把真池子接上;测试里换成假的。
"""

from .pool import TaskPool
from .task import Ctx, Task

__all__ = ["Ctx", "Task", "TaskPool"]
