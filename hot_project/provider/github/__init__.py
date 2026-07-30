"""github —— GitHub 出站实现。

- `client` REST/GraphQL 请求
- `trending` 抓 github.com/trending 的 HTML(**不吃 token、不受 Search API 限流**,
  所以它在任务池里属「无限制」制度,可与其他发现任务完全并行)
- `tokens` token 池

token 池对每个 token 管三件事,「拿到一个可用 token」= 三者同时满足:
**占用**(未被借走)、**冷却**(限流/401 冷却已过)、**配速**(距该 token 上次请求已满最小间隔)。

第三条必须在池内。旧代码池里有正确的 async 配速但只有一处在用,另有四处散着同步
`time.sleep` 抄写;结果是并发等于 token 数时,12 个请求在 500ms 内打完、烧光每分钟配额,
然后集体撞二级限流(实测 2026-07-30:一轮跑出 143 个失败页、3 轮补偿)。
"""
