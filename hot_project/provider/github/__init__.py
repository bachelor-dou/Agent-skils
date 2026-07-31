"""github —— GitHub 出站实现。自底向上一层压一层:

- `tokens`   token 池:借一张租约(占用 / 冷却 / 配速)。不发请求
- `request`  出站请求原语:发一次 REST/GraphQL,**不重试**,租约由调用方递进来
- `repo`     单仓库资料(REST `/repos/...`),自己借租约
- `trending` 抓 github.com/trending 的 HTML(不吃 token、不受限流,可与发现任务完全并行)
- `tasks`    并发任务单元,交给 `infra.tasks` 的任务池调度
- `client`   同步的 `GitHub` 客户端,组合以上全部 —— **包外只该 import 这一个**

`client` 和 `request` 不能合并:`repo` 与 `tasks` 都 import `request`,而 `client` 又
import 它们,合成一个模块就是循环 import。名字也别再对调 —— 栈顶那个才是"客户端"。

「拿到一个可用 token」要同时满足三件事:**占用**(未被借走)、**冷却**(限流/401 冷却已过)、
**配速**(距该 token 上次请求已满最小间隔)。第三条必须在池内实现 —— 散在调用方各自 sleep
的话,12 个请求会在 500ms 内打完、烧光每分钟配额,然后集体撞二级限流。
"""
