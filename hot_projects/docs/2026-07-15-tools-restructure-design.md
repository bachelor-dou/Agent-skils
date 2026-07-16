# 工具层重构设计 — 消除 capabilities/pipeline/tools 三层散乱

日期:2026-07-15
状态:待实施
背景:项目多次迭代后,"工具"的定义散落在 `capabilities/_impl.py`(带 tool_ 前缀的历史遗留)、
`capabilities/*.py`(门面)、`pipeline/`(榜单编排)、`tools/`(agent 封装)四处;
一个工具的实现/门面/handler/schema 分居多文件,难以定位。本次重构统一为
"**一个 LLM 工具 = 一个文件,公用能力下沉 basic/**"的单一结构。

## 设计原则(与用户确认)

1. 工具只有两类:
   - **LLM 工具**:注册表暴露给模型的,一个工具一个文件、文件名 = 工具名;
     若内部需要多个基础工具的顺序编排(如榜单),编排就写在该工具文件内部;
   - **基础工具(basic/)**:严格"被 ≥2 个工具复用"才准入;新工具先写独立脚本,
     其内部逻辑被第二个消费者需要时才下沉 basic。
2. agent 与定时任务调用**同一个**榜单工具入口,不再有独立 pipeline 层。
3. `capabilities/`、`pipeline/` 目录删除。

## 目标结构

```
hot_projects/
├── tools/
│   ├── registry.py               # 唯一注册表(工具名单一览)
│   ├── schemas.py                # 唯一 schema
│   ├── arg_validator.py
│   ├── basic/                    # 公用基础工具(消费方 ≥2)
│   │   ├── search.py             # 关键词搜索/星段扫描      ← 三个榜单工具
│   │   ├── growth.py             # 单仓库/批量增长          ← 榜单 + repo_growth
│   │   ├── rank.py               # 评分排序(含原 scoring)   ← 三个榜单工具
│   │   ├── report.py             # 报告生成                 ← 三个榜单工具
│   │   ├── describe.py           # LLM 项目描述(含原 infra/llm.py 业务函数)
│   │   │                         #                          ← describe_project + report
│   │   └── report_parse.py       # 报告 Markdown 解析       ← analyze_report + Web 渲染
│   ├── ranking.py                # 综合榜/新项目榜/关键词榜(编排+分阶段缓存+确认守卫)
│   ├── repo_growth.py            # 单仓库增长(消歧 + basic/growth)
│   ├── describe_project.py       # 项目介绍(消歧 + basic/describe)
│   ├── repo_profile.py           # 项目画像(原 repo_overview + repo_activity 合并)
│   ├── search_repos.py           # 按描述找项目
│   ├── analyze_report.py         # 报告读取分析
│   ├── get_db_info.py            # DB 查询
│   └── fetch_trending.py         # Trending
├── agent/                        # ReAct 循环(state 增加通用 tool_state 槽)
├── datasource/                   # 数据源适配层(原 providers/,GitHub 平台细节)
├── infra/                        # 基础设施(llm_client/db/favorites/concurrency),llm.py 移出
├── data/                         # 数据存储(原 db/):Github_DB.json、favorites.json、断点
├── server_render.py              # 报告 HTML 渲染 + 上期对比(从 api_server 拆出)
├── api_server.py                 # 会话/安全/WS/收藏 API
└── cron_scheduled_update.py      # 直调 tools/ranking 综合榜工具(删 DiscoveryPipeline)
```

## 命名与路径调整(与用户确认)

- `providers/` → **`datasource/`**(数据源适配层,与 data/ 形成"源/储"对仗)。
- `db/`(磁盘目录) → **`data/`**;`config.py` 中路径**写死**,不引入环境变量覆盖
  (目录不需要运行时可配;`DATA_DIR` 环境变量覆盖机制一并移除,直接用包内固定路径)。
- 涉及改动:config 路径常量、.gitignore、README、以及 datasource 的全部 import 路径。

## 工具清单(重构后 9 个 LLM 工具)

| 工具 | 文件 | 说明 |
|---|---|---|
| comprehensive_ranking / hot_new_ranking / keyword_ranking | ranking.py | 一条参数化编排链的三个工具 |
| repo_growth | repo_growth.py | 薄封装 |
| describe_project | describe_project.py | 薄封装 |
| repo_profile | repo_profile.py | **合并**:README 摘录+元数据+release 节奏+近期提交+最近推送;README 预算 ~5000 字符、提交 5 条,保证不触发 8000 字符序列化截断;license 字段保留但 prompt 不引导用于对比 |
| search_repos | search_repos.py | 按描述找项目(star 降序 Top N) |
| analyze_report | analyze_report.py | 本地报告分析 |
| get_db_info | get_db_info.py | DB 查询 |
| fetch_trending | fetch_trending.py | Trending |

行为变化仅一处:repo_overview + repo_activity → repo_profile(注册表/schema/prompt 规则同步)。

## 其他模块审核结论

- **agent/state.py**:移除对 RankingCache 的直接依赖;新增通用 `tool_state: dict`,
  榜单缓存与确认签名由 ranking 工具存取,agent 层与具体工具解耦。
- **infra/llm.py**:业务函数(call_llm_describe / batch_condense_descriptions)并入
  tools/basic/describe.py;infra 只留 llm_client.py。
- **api_server.py**:报告渲染+上期对比(~400 行,含 mtime 解析缓存)拆出为 server_render.py;
  统一 logger 命名(现存 "hot_projects" 与 "discover_hot" 两棵树,Web 模式下前者日志丢失
  —— 遗留 bug,本次一并修复,统一为 "hot_projects")。
- **cron_scheduled_update.py**:删 DiscoveryPipeline 包装类,直调 ranking 工具;
  漏斗/字段变化统计日志保留。
- **providers/、web/、config.py、agent_cli.py、__main__.py**:不动。
  providers/github/api.py(1226 行)内聚但偏大,列为二期可选拆分。

## 记忆/缓存审核

| 缓存 | 处置 |
|---|---|
| 对话历史+压缩(AgentState) | 不动(边界安全此前已修复) |
| RankingCache 分阶段签名缓存 | 迁入 ranking.py,经 tool_state 挂到会话;cron 自建实例 |
| 昂贵工具确认签名 | 同上迁入 tool_state |
| Web 会话 TTL(_sessions) | 不动 |
| 上期报告解析缓存 | 随渲染迁 server_render.py |
| db/(Github_DB/favorites/断点) | 不动(此前已归档整理) |

## 详尽执行计划

约定:每步结束跑 `pytest hot_projects/tests/ -q` 必须全绿(基线 78 个)才进下一步;
每步为一个逻辑提交单元,失败可独立回滚。步骤内先建新文件、再改引用、最后删旧文件。

### 步骤 0 — 目录与路径改名(datasource/、data/)

- `providers/` 整目录 `git mv` → `datasource/`;全仓 `from ..providers`/`from .providers`
  → `datasource`(涉及 provider.py、token_pool、api、trending、growth_estimator、
  agent/__init__.py build_agent、tests)。
- `db/` 磁盘目录 → `data/`;`config.py`:`DB_DIR`→`DATA_DIR`(指向 data/),
  移除 `os.environ.get("DATA_DIR", ...)` 覆盖,写死 `PACKAGE_DIR/data`、report/、logs/;
  同步 `.gitignore`(`hot_projects/db/`→`hot_projects/data/`)。
- 验证:pytest 全绿 + `python -c "import hot_projects.api_server"` 无 ImportError。

### 步骤 1 — 建立 tools/basic/(公用能力下沉)

- 新建 `tools/basic/`,把 `capabilities/_impl.py` 按域拆分:
  - `search.py` ← tool_search_by_keywords / tool_scan_star_range(+ 并发/限流补偿 helper)
  - `growth.py` ← tool_check_repo_growth / tool_batch_check_growth
  - `rank.py`   ← tool_rank_candidates + `capabilities/scoring.py` 全部
  - `report.py` ← `capabilities/report.py`(报告生成)
  - `describe.py` ← tool_describe_project + `infra/llm.py`(call_llm_describe /
    batch_condense_descriptions 业务函数)
  - `report_parse.py` ← `capabilities/report_parse.py`(原样迁移)
  - `_shared.py` ← _impl 中的公共 helper(_run_coroutine_sync、_ensure_project_record、
    trending_repo_to_search_repo、异步调度器解析等)
- 去掉 `tool_` 前缀、"Tool N" 编号注释,函数名用能力语义命名。
- 暂不删 capabilities/;让其 __init__ 与门面改为从 tools/basic 转发(过渡壳),保证旧 import 不断。
- 验证:pytest 全绿。

### 步骤 2 — 合成 tools/ranking.py(唯一复合工具入口)

- 新建 `tools/ranking.py`,合并三处:`pipeline/ranking_pipeline.run_ranking`
  + `pipeline/cache.py`(RankingCache)+ `tools/ranking_tools.py`(make_ranking_handler
  确认守卫);内部 import tools/basic。
- `agent/state.py`:移除 `RankingCache` 依赖,`AgentState` 用通用
  `tool_state: dict`(+ 保留 `pending_confirmation_signature`);ranking 工具从
  `ctx.state.tool_state` 存取自己的缓存实例(键如 "ranking_cache")。
- `cron_scheduled_update.py`:删 `DiscoveryPipeline` 包装类,`run_update` 直接调
  ranking 综合榜执行函数(与 agent 同一入口);funnel/统计日志保留。
- 删除 `pipeline/` 目录。
- 验证:pytest 全绿(test_ranking_pipeline / test_ranking_cache 改为指向 tools.ranking);
  另跑 `python -m hot_projects.cron_scheduled_update --help` 确认入口可加载。

### 步骤 3 — 独立工具文件化 + repo_profile 合并

- `tools/atomic_tools.py` 拆为按工具命名的文件:`repo_growth.py`、`describe_project.py`、
  `search_repos.py`、`analyze_report.py`、`get_db_info.py`、`fetch_trending.py`;
  各文件含 handler + 复用的 `_resolve_repo` 消歧(提到 basic/_shared 或 tools 内共用模块)。
- `repo_profile.py`:合并原 `capabilities/evidence.py` 的 repo_overview + repo_activity
  为单个 `repo_profile` 工具;README 摘录预算 ~5000 字符、近期提交 5 条,防 8000 序列化截断;
  license 字段保留、prompt 不引导。
- `tools/schemas.py`:删 repo_overview/repo_activity 两条,加 repo_profile 一条;
  `TOOL_PARAM_SCHEMA` 同步。
- `tools/registry.py`:注册项更新(9 个工具);`agent/prompts.py` 第 8 条(对比工作法)
  改为用 repo_profile,第 10 条 search_repos 不变。
- 验证:pytest 全绿(test_evidence → test_repo_profile 改写;test_atomic_tools 拆分随文件走)。

### 步骤 4 — 清场

- 删除 `capabilities/` 目录与 `_impl.py`、过渡转发壳;删 `infra/llm.py`(已并入 basic/describe)。
- 全仓 grep 确认无 `capabilities`/`pipeline`/`_impl`/`infra.llm` 残留 import。
- 更新 README「项目结构」段为新结构图 + "找工具:registry 看名单 → 同名文件看实现"。
- 验证:pytest 全绿 + `import hot_projects.api_server` + `import hot_projects.agent`。

### 步骤 5 — api_server 减负 + 统一 logger

- 新建 `server_render.py`:迁入报告渲染 + 上期对比 diff + mtime 解析缓存 +
  语言色/摘要 chip 等(约 400 行);api_server 从中 import。
- 统一 logger 命名:全项目 `logging.getLogger("discover_hot")` → `"hot_projects"`
  (或反之择一);修 Web 模式下 agent/榜单日志不进 web.log 的遗留 bug。
- 验证:pytest 全绿;起服务 curl 报告页 200 且面板/侧栏渲染数正确。

### 步骤 6 — 全量验收

- pytest 全绿;`nohup ... -m hot_projects` 重启;
- 手工核验:报告页(上新标注 + 收藏 ★ + 侧栏过滤)、聊天里调 repo_growth / repo_profile /
  analyze_report / search_repos 各一次、cron 入口 dry 加载;
- 更新本设计文档状态为「已实施」。

## 风险与回滚

- 纯搬迁+合并,除 repo_profile 外零行为变化;每步 pytest 门禁,任一步失败即停并回滚该步;
- 现有 78 个测试为行为基线,涉及改名/合并的测试(ranking/cache/evidence/atomic)同步改写,
  数量不减;
- 数据零迁移风险:data/ 仅目录改名;报告/DB/收藏文件内容不动;
- 服务重启窗口秒级。
