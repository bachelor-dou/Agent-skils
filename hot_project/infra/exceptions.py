"""跨层共用的异常。

**异常不携带 `token_idx`。** 索引只存在于租约内部,调用方拿不到,也就没有「选错记账方式」
这件事可做:异常类型 → 记账动作的映射只有 `tokens.Lease.__aexit__` 一处。带索引的话每个
捕获点都得自己决定拿它记什么账,同一次 401 就会长出好几种处置 —— 其中把瞬时 401 当成
token 永久失效的那种,能在一轮运行里把 token 逐个烧光。
"""

from __future__ import annotations


class RetryableError(Exception):
    """瞬时故障:网络抖动、5xx、连接被重置 —— 同样的请求过一会儿多半就成了。

    和「限流」分开是因为它**必须**计入重试次数,否则网络长期不通时任务会无限自旋。
    """


class GitHubError(Exception):
    """所有来自 GitHub 侧的可识别错误的基类。"""


class RateLimitError(GitHubError):
    """命中限流(403 / 429)。

    `reset_at` 用 GitHub `X-RateLimit-Reset` 给的绝对时刻而不是「冷却 N 秒」,本地估算会
    偏早,偏早就是再撞一次。`message` 带上是哪一种(见 `client._limit_reason`):两者处置
    相反 —— 主限额耗尽只影响这一个 token,冷却它就够;二级限流实测按**来源 IP** 计(12 张
    独立账号的 token 会在同一秒里一起被拒),冷却单张没有意义,得整体降速。
    """

    def __init__(self, reset_at: float, message: str = "") -> None:
        self.reset_at = reset_at
        super().__init__(message or f"rate limited until {int(reset_at)}")


class TokenInvalidError(GitHubError):
    """鉴权失败(401)。

    **不代表 token 一定坏了。** GitHub 对有效 token 也会返回瞬时 401,所以它的处置是
    strikes 冷却而非永久失效。
    """
