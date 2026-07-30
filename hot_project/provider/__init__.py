"""provider —— 出站第三方数据源。

**契约**:对包外只暴露 Provider 协议;`github/` 下的具体请求实现是本包内部细节。
可 import `config` / `core` / `infra`;不许 import `tools` / `agent` 和顶层入口脚本。

为什么出站属这里、而不属工具层内部:**每日快照任务不是一个工具**,却需要 GitHub 收集。
若出站住在 `tools/` 下,批处理脚本就必须 import 工具层 —— 旧代码正是如此
(`cron_daily_star_snapshot.py` 里 `from ...tools.tool.ranking import _collect`),
于是一个定时脚本挂在了榜单逻辑上。放在这里,`tools/` 与顶层入口脚本就是平级的两个消费方。

命名注意:本包的 `github/client.py` 是**出站**(我们 → GitHub);顶层 `api_server.py` 是**入站**
(浏览器 → 我们)。旧代码把两者都叫 api(`provider/github/api.py` 与 `api_server.py`),
方向相反却同名。
"""
