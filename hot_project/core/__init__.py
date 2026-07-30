"""core —— 纯算法层,零 I/O。

**进来的门槛是「隔这一层能买到东西」,不是纯不纯。** 够格的只有两种:

1. **多个调用方共用同一套算术。** `growth.py`(周报批量、单仓库查询、爆发探针)、
   `report_parse.py`(analyze_report / star_trend / api_server / 周报 diff)。
   各写一份的下场是同一个仓库在两个地方算出两个增长值。
2. **这段逻辑历史上被 I/O 污染过,需要守卫顶住。** `scoring.py` 只有排名一个调用方,
   但旧版打分函数里有一行 `db.get("projects", ...)` —— 一个纯打分公式伸手去读数据库,
   于是想验证一次排名要先造一个 DB。守卫把这条路堵死,污染就只能发生在调用方那边。

其余一律写在调用方里:淘汰判定就在 `cron_daily_snapshot.py`、Trending 解析就在
`provider/github/trending.py`。为它们隔一层买到的边界是零,代价是读一件事要开两个文件。

**契约**:只能 import `config`、`common` 和标准库里的纯计算部分。不许 import
`infra` / `provider` / `tools`,也不许 import 网络库或做文件读写。
由 `tests/test_layering.py` 自动守卫。

值在这层的意义:不用 token、不联网、不碰磁盘就能跑完整的排名回归测试。
旧包这些逻辑埋在 `tools/basic/` 里和有 I/O 的模块做邻居,想验证一次打分要先有 12 个 token。
"""
