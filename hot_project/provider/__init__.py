"""provider —— 出站第三方数据源。

**契约**:对包外只暴露 Provider 协议;`github/` 下的具体请求实现是本包内部细节。
可 import `config` / `core` / `infra`;不许 import `tools` / `agent` 和顶层入口脚本。

出站独立成包而不塞进工具层,是因为**每日快照任务不是一个工具**却需要 GitHub 收集:住在
`tools/` 下就意味着定时脚本得 import 榜单逻辑。放这里,`tools/` 和入口脚本是平级的消费方。

命名注意:`github/client.py` 是**出站**(我们 → GitHub),顶层 `api_server.py` 是**入站**
(浏览器 → 我们),别都叫 api。
"""
