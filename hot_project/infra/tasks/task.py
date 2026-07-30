"""任务基类。

子类要做的只有两件事:声明自己走哪条道、实现 `run`。

    class KeywordPage(Task):
        lane = "search"
        needs_token = True
        token_kind = "search"

        def __init__(self, word: str, page: int) -> None:
            self.word, self.page = word, page

        async def run(self, ctx):
            items = await search(ctx.token, self.word, self.page)
            if len(items) == 100 and self.page < 3:
                ctx.submit(KeywordPage(self.word, self.page + 1))   # 派生
            return items

**没有 `max_concurrency` 属性。** 并发度是「这条道开了几个 worker」,写在接线处而不是任务
类里 —— 同一个数字只该存在一份。旧包让任务自带 `max_concurrency = 12`(意思是「和 token
数一样」),那个 12 在 token 增减之后就是错的,而且和 token 池的容量重复记账。
"""

from __future__ import annotations

from typing import Any, Callable, ClassVar, NamedTuple


class Ctx(NamedTuple):
    """worker 递给任务的执行上下文。

    `submit` 显式传进来而不是让任务持有池引用:任务只能**提交**,不能启停池子、
    不能读队列长度、不能改并发度。能力越小,以后越不会有人顺手在任务里干调度的事。
    """

    token: Any                      # 租约对象;`needs_token = False` 的任务拿到 None
    submit: Callable[["Task"], None]


class Task:
    """一个可执行单元。**一个任务 = 一次外部请求**,不要在一个任务里循环发多页。

    分页应当靠派生(见类文档的例子)。旧包把「一个关键词的 1..3 页」塞进一个任务,
    于是这个任务从头到尾攥着同一个 token,实测中位持有 4.42 秒,大半时间是在等配速 ——
    别的任务只能干看着。拆成一页一个任务之后,每页各自借还,token 不会被长期占住。
    """

    lane: ClassVar[str] = "free"
    needs_token: ClassVar[bool] = False

    # 需要哪种 token。对本层是**不透明字符串**,由接线时的 leaser 解释
    # (出站层把它映射成 Search / GraphQL 的配速)。
    token_kind: ClassVar[str] = "default"

    # 瞬时故障最多重排几次。超了就按失败收尾,避免网络长期不通时无限自旋。
    max_retries: ClassVar[int] = 3

    # 撞限流最多回队几次。单独一本账、而且宽得多:限流不是这个任务的错,正常情况下
    # 等一会儿必然放行,不该占用瞬时故障的那三次。
    #
    # 但它**必须有界**。GitHub 的二级限流可能持续几十分钟返回 403,无界重排会让每日任务
    # 一直转到 Actions 六小时超时 —— 既没落盘也没报错,看起来像卡死。有界的话它会退化成
    # 「这批采集失败 → 覆盖率不足 → 拒绝落盘」,那是设计好的失败姿势。
    # 12 个 token 各冷却 60 秒的健康场景下,一个任务撞不到 20 次。
    max_rate_limits: ClassVar[int] = 20

    attempts: int = 0               # 已经跑过几次,由池维护
    rate_limits: int = 0            # 撞了几次限流,和 attempts 分开记

    async def run(self, ctx: Ctx) -> Any:
        raise NotImplementedError

    def on_done(self, result: Any) -> None:
        """成功回调。**在 worker 里同步内联执行**,所以不要在这里做慢的事。

        内联是有意的:落盘写在这里就是随算随落,而不是堆到整轮结束 ——
        中途被杀也保住了已完成的部分。
        """

    def on_error(self, err: BaseException) -> None:
        """最终失败(重试用尽或不可重试)回调,同样内联执行。"""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
