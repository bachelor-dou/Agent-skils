"""跨层共用的异常。

设计上只有一处和旧包不同,但它是本轮重构要解决的核心问题:**异常不再携带 `token_idx`。**

旧异常是 `RateLimitError(token_idx, reset_time)`、`TokenInvalidError(token_idx)`,于是每个
捕获点都得自己决定「拿这个索引去池里记什么账」。结果同一件事(一次 401)在代码库里长出了
四种处置:

    infra/concurrency/dispatcher.py:183   mark_auth_failed  → strikes 冷却   ✅
    infra/concurrency/tasks.py:69         mark_auth_failed  → strikes 冷却   ✅
    cron_daily_star_snapshot.py:168       mark_invalid      → **永久失效**   ❌
    datasource/github/api.py:400          mark_invalid      → **永久失效**   ❌

后两处意味着每日快照跑一半遇到一次瞬时 401(GitHub 在二级限流/鉴权抖动时会返回),
那个 token 就在本次运行里彻底烧掉了,而 12 个 token 烧几个就再也跑不完 7.8 万个仓库。

现在索引只存在于租约内部,调用方拿不到,也就没有「选错记账方式」这件事可做了:
异常类型 → 记账动作的映射只有 `tokens.Lease.__aexit__` 一处(见 `provider/github/tokens.py`)。
"""

from __future__ import annotations


class RetryableError(Exception):
    """瞬时故障:网络抖动、5xx、连接被重置 —— 同样的请求过一会儿多半就成了。

    和「限流」分开是因为两者计不计入重试次数不同(见 `infra/tasks/pool.py`):
    限流是外部节流、不该消耗任务的重试额度,瞬时故障则必须有次数上限,
    否则网络长期不通时任务会无限自旋。
    """


class GitHubError(Exception):
    """所有来自 GitHub 侧的可识别错误的基类。"""


class RateLimitError(GitHubError):
    """命中限流(403 / 429)。

    `reset_at` 是 GitHub `X-RateLimit-Reset` 给的 epoch 秒 —— 用它而不是「冷却 N 秒」,
    因为限流窗口是服务端的绝对时刻,本地估算会偏早,偏早就是再撞一次。

    待办(等每日快照跑顺之后):这里没有区分 403 的两种成因,而它们的处置是**相反**的 ——
    主限额耗尽(`x-ratelimit-remaining: 0`)该冷却这个 token,二级限流(封的是来源 IP)
    该降全局速率、冷却单个 token 毫无意义。实测证据记在 `provider/github/tokens.py`
    的 `SEARCH` 上方。
    """

    def __init__(self, reset_at: float, message: str = "") -> None:
        self.reset_at = reset_at
        super().__init__(message or f"rate limited until {int(reset_at)}")


class TokenInvalidError(GitHubError):
    """鉴权失败(401)。

    **不代表 token 一定坏了。** GitHub 对有效 token 也会返回瞬时 401,所以它的处置是
    strikes 冷却而非永久失效 —— 见本文件头部那两处 bug。
    """
