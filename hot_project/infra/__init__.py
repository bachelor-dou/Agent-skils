"""infra —— 机制,不知道产品是什么。

**契约**:可 import `config`、`common`;不许 import `provider` / `service` / `tools` /
`agent` 和顶层入口脚本。

`data_access/` 数据文件读写(一个安全写入原语 + DB/快照/收藏/报告)、`tasks/` 全局任务池、
`llm/` LLM 客户端与 key 管理、`notify` 推送、`exceptions` 共享异常词汇。

判据:这里的东西换成「npm 包追踪器」照样能用。凡是知道 star、榜单、爆发探针的,
都该在 `service/` 或工具层,不在这里。
"""
