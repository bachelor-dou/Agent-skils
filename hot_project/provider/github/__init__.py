"""github —— GitHub 出站实现。

- `client` REST/GraphQL 请求
- `trending` 抓 github.com/trending 的 HTML(不吃 token、不受限流,可与发现任务完全并行)
- `tokens` token 池

「拿到一个可用 token」要同时满足三件事:**占用**(未被借走)、**冷却**(限流/401 冷却已过)、
**配速**(距该 token 上次请求已满最小间隔)。第三条必须在池内实现 —— 散在调用方各自 sleep
的话,12 个请求会在 500ms 内打完、烧光每分钟配额,然后集体撞二级限流。
"""
