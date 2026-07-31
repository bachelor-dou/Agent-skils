"""tasks —— 全局任务池:**几条道**,每条一个队列 + 固定数量的协程 worker。

    pool = TaskPool({"search": 6, "graphql": 4, "free": 8}, leaser=...)
    async with pool:
        pool.submit(KeywordPage("llm framework", page=1))
        await pool.join()

外部只有 `submit` 和 `join` 两个入口。池本身不理解阶段,有真依赖的任务由调用方按屏障分批
提交。

分道而不是「一条队列 + 每类一个信号量」:信号量版会饿死人 —— worker 堵在信号量上时**手里
还攥着已经出队的任务**,于是明明可以跑的搜索任务反而没人做。分道之后并发度**就是**这条道
的 worker 数,搜索和 GraphQL 各占 GitHub 的一份独立预算,能同时跑。

**本包不 import token 池。** 它是 GitHub 专属的(懂 403/401 语义)、住在 `provider/github/`,
横轴设施反向依赖纵轴实现会让分层守卫报红。这里只认一个 `Leaser` 可调用对象:给它一个字符串
(任务自称需要哪种 token),还回来一个异步上下文管理器;字符串对本包**没有含义**,由出站层
解释。真池子由顶层入口脚本接上,测试里换成假的。
"""

from .pool import TaskPool
from .task import Ctx, Task

__all__ = ["Ctx", "Task", "TaskPool"]
