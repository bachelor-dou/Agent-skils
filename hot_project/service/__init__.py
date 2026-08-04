"""service —— 业务流水线,被 cron、web、agent 工具三方共用。

    growth.py    窗口增长算术(纯计算):榜单批量算与单仓库工具共用同一套规则
    ranking.py   榜单:实时取 star 减快照基线,边算边筛,打分排序出 Top N
    report.py    报告:Top N → 四段介绍 → Markdown(data_access.reports 负责读回)
    describe.py  LLM 文案:项目介绍(标准/四段)、批量浓缩短句

**契约**:可 import `config`、`common`、`infra`、`provider`;
不许 import `tools` / `agent` / `web` —— 工具是它的调用方,不是它的依赖。
"""
