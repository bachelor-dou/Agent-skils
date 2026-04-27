# GitHub 热门项目发现系统 — 设计文档

## 1. 设计目标与约束

### 1.1 核心需求
自动发现 GitHub 近期 star 增长最快的开源项目，生成结构化中文报告。

### 1.2 设计约束
- GitHub REST API 限制 5000 次/小时/Token，Search API 限制 30 次/分钟
- Search API 单次查询最多返回 1000 条结果
- Stargazers REST 分页在高 star 仓库（>40k）时返回 422
- 需支持多 Token 并行以提升吞吐
- 需支持交互式（CLI）、API（Web）、MCP 三种使用模式

### 1.3 技术选型
Python 3.10+, requests, FastAPI, JSON 文件存储, OpenAI 兼容 LLM, MCP SDK

---

## 2. 分层架构总览

```mermaid
graph TB
    subgraph L4["入口层 (Layer 4)"]
        CLI["agent_cli.py<br/>CLI REPL"]
        API["api_server.py<br/>REST/WS"]
        MCP["mcp_server.py<br/>MCP stdio"]
        MAIN["__main__.py<br/>默认入口"]
        SCHED["scheduled_update.py<br/>定时任务"]
        REGEN["regenerate_report.py<br/>报告恢复"]
    end

    subgraph L3_WEB["Web 展示"]
        WEB["web/<br/>chat.html · report.html · *.js · *.css"]
    end

    subgraph L2["Agent 层 (Layer 2)"]
        AGENT["agent.py<br/>ReAct 循环 + 状态管理 + 对话压缩"]
    end

    subgraph L1["输入解析层 (Layer 1)"]
        INTENT["parsing/intent_detector.py<br/>意图识别"]
        PARAM["parsing/param_extractor.py<br/>参数提取"]
        NORM["parsing/tool_arg_normalizer.py<br/>参数归一化"]
    end

    subgraph L3["执行层 (Layer 3)"]
        TOOLS["agent_tools.py<br/>⚙️ 工具中枢 · 9个Tool"]
        RANK["ranking.py<br/>评分排序"]
        REPORT["report.py<br/>报告生成"]
        GROWTH["growth_estimator.py<br/>增长估算"]
        TREND["github_trending.py<br/>Trending 爬虫"]
        PIPE["execution/pipeline.py<br/>统一流水线"]
        subgraph L3_TASKS["tasks/ 任务子系统"]
            TASK["task.py<br/>Task子类 + 候选管理"]
            TBASE["task_base.py<br/>Task ABC"]
            WPOOL["worker_pool.py<br/>线程池 + Token绑定"]
        end
    end

    subgraph L0["基础设施层 (Layer 0 · common/)"]
        CONFIG["config.py<br/>全局常量"]
        DB["db.py<br/>JSON DB + 文件锁"]
        GHAPI["github_api.py<br/>REST/GraphQL封装"]
        LLM["llm.py<br/>LLM 描述生成"]
        TOKEN["token_manager.py<br/>Token + 请求头构建"]
        EXC["exceptions.py<br/>自定义异常"]
    end

    CLI --> AGENT
    API --> AGENT
    MCP --> TOOLS
    MAIN --> API
    SCHED --> PIPE

    AGENT --> L1
    AGENT --> TOOLS

    TOOLS --> RANK
    TOOLS --> REPORT
    TOOLS --> GROWTH
    TOOLS --> TREND
    TOOLS --> TASK

    PIPE --> TOOLS

    TASK --> WPOOL
    TASK --> GROWTH
    RANK --> L0
    REPORT --> LLM
    GROWTH --> GHAPI
    TREND -.-> |"纯HTTP,不依赖内部模块"| TREND

    TOOLS --> L0
    WPOOL --> TOKEN
    DB --> CONFIG
    TOKEN --> CONFIG
```

### 2.1 分层说明

| 层级 | 名称 | 职责 | 包含文件 |
|------|------|------|---------|
| **L0** | 基础设施层 | 配置、存储、API 封装、Token 管理 | `common/` 全部 6 个模块 |
| **L1** | 输入解析层 | 意图识别、参数提取、归一化（纯函数） | `parsing/` 全部 3 个模块 |
| **L2** | Agent 层 | ReAct 推理循环、状态管理、对话压缩 | `agent.py` |
| **L3** | 执行层 | 9 个 Tool 实现 + 评分/报告/增长/爬虫/任务子系统 | `agent_tools.py`, `ranking.py`, `report.py`, `growth_estimator.py`, `github_trending.py`, `execution/`, `tasks/` |
| **L4** | 入口层 | CLI / API / MCP / 定时任务等启动入口 | `__main__.py`, `agent_cli.py`, `api_server.py`, `mcp_server.py`, `scheduled_update.py`, `regenerate_report.py` |

### 2.2 设计原则
- **模式统一**：CLI / API / MCP 三入口共享 agent_tools 中的 Tool 实现
- **Token 隔离**：每个 Worker 绑定唯一 Token，消除跨线程竞争
- **主线程回调**：所有数据写入（DB、候选集）在主线程 drain_results 中执行，无需额外锁
- **降级容错**：REST → GraphQL → 保守估算，每层失败都有退路
- **非侵入参数**：新增参数不传时行为等同参数添加前

---

## 3. 核心工具一览

```mermaid
graph LR
    subgraph TOOLS["agent_tools.py — 9 个核心 Tool"]
        direction TB
        T1["🔍 search_by_keywords<br/>关键词搜索"]
        T2["📊 scan_star_range<br/>Star 范围扫描"]
        T3["📈 check_repo_growth<br/>单仓库增长"]
        T4["⚡ batch_check_growth<br/>批量增长筛选"]
        T5["🏆 rank_candidates<br/>候选评分排序"]
        T6["📝 describe_project<br/>LLM 描述"]
        T7["📄 generate_report<br/>报告生成"]
        T8["💾 get_db_info<br/>DB 查询"]
        T9["🔥 fetch_trending<br/>Trending 爬取"]
    end

    T1 --> |"创建Pool"| POOL["TokenWorkerPool"]
    T2 --> POOL
    T4 --> POOL
    T5 --> SCORER["ranking.py"]
    T6 --> LLM["llm.py"]
    T7 --> RPT["report.py"]
    T3 --> GE["growth_estimator.py"]
    T9 --> GT["github_trending.py"]
```

| Tool | 创建 Pool | 写 DB | 说明 |
|------|----------|-------|------|
| search_by_keywords | ✅ | — | 25 类关键词多页搜索 |
| scan_star_range | ✅ | — | auto_split_star_range 递归二分 |
| check_repo_growth | — | — | 单仓库详情 + LLM 描述 |
| batch_check_growth | ✅ | 内存 | 批量增长计算 + checkpoint |
| rank_candidates | — | — | comprehensive / hot_new 排序 |
| describe_project | — | 内存 | 单项目 LLM 描述 |
| generate_report | — | 磁盘 | Markdown 报告 + DB 持久化 |
| get_db_info | — | — | 只读查询 |
| fetch_trending | — | — | HTML 爬虫，零 API 消耗 |

---

## 4. 端到端数据流

### 4.1 完整发现流程

```mermaid
flowchart TD
    START(["用户请求 / 定时触发"]) --> COLLECT

    subgraph COLLECT["Phase 1 · 数据收集"]
        direction TB
        KW["关键词搜索<br/>search_by_keywords<br/>25类 × 150+词 × 3页"]
        SCAN["Star 范围扫描<br/>scan_star_range<br/>auto_split ≤800条/段"]
        TREN["Trending 爬虫<br/>fetch_trending<br/>daily/weekly/monthly"]
        KW --> MERGE["去重合并 → raw_repos"]
        SCAN --> MERGE
        TREN --> MERGE
    end

    MERGE --> GROWTH_PHASE

    subgraph GROWTH_PHASE["Phase 2 · 增长量化"]
        direction TB
        DB_CHECK{"DB 有效?"}
        DB_DIFF["DB 差值法<br/>0 次 API"]
        REST["REST 二分法<br/>~5-10 次 API"]
        GQL["GraphQL 采样<br/>~30 次 API"]
        FILTER{"增长 ≥ 阈值?"}

        DB_CHECK -->|是| DB_DIFF --> FILTER
        DB_CHECK -->|否| REST
        REST -->|成功| FILTER
        REST -->|422| GQL --> FILTER
        FILTER -->|是| CAND["进入候选集"]
        FILTER -->|否| DROP["丢弃"]
    end

    CAND --> RANK_PHASE

    subgraph RANK_PHASE["Phase 3 · 评分排序"]
        direction TB
        MODE{"排序模式?"}
        COMP["comprehensive<br/>log(增长量) + log(增长率)<br/>新项目折扣"]
        HOT["hot_new<br/>created_at 过滤<br/>按增长量排序"]
        MODE -->|comprehensive| COMP --> TOPN["取 Top N"]
        MODE -->|hot_new| HOT --> TOPN
    end

    TOPN --> REPORT_PHASE

    subgraph REPORT_PHASE["Phase 4 · 报告生成"]
        direction TB
        DESC["LLM 生成描述<br/>200-400 字中文"]
        CARD["输出结构化卡片"]
        SAVE["保存 report/YYYY-MM-DD.md<br/>+ 写入 DB"]
        DESC --> CARD --> SAVE
    end

    SAVE --> DONE(["完成"])
```

### 4.2 三级增长降级详图

```mermaid
flowchart TD
    REPO["单仓库"] --> A{"DB有效<br/>且 star 非空?"}
    A -->|是| A1["A. DB差值法<br/>current - db_star<br/>精度: 精确 | 消耗: 0"]
    A -->|否| B["B. REST 二分法"]
    B --> B1{"最后页全在窗口内?"}
    B1 -->|是| B2["全量计数"]
    B1 -->|否| B3["二分搜索页码<br/>深度 ≤ 20"]
    B3 -->|成功| B4["精确计数边界页"]
    B3 -->|422| C["C. GraphQL 采样外推"]
    C --> C1{"采样跨越窗口?"}
    C1 -->|是| C2["精确计数"]
    C1 -->|否| C3["100条分段加权外推<br/>+ 覆盖率补偿<br/>+ burst 防护"]

    A1 --> RESULT["增长值"]
    B2 --> RESULT
    B4 --> RESULT
    C2 --> RESULT
    C3 --> RESULT
```

---

## 5. 并发模型

```mermaid
sequenceDiagram
    participant Main as 主线程
    participant Pool as TokenWorkerPool
    participant W1 as Worker-1 (Token-1)
    participant W2 as Worker-N (Token-N)

    Main->>Pool: 创建 Pool（N Worker ↔ N Token 1:1 绑定）
    Main->>Pool: submit(tasks)
    Pool->>W1: 取任务执行
    Pool->>W2: 取任务执行

    alt 200 OK
        W1-->>Pool: result → 结果队列
    else 403/429 限流
        W1-->>Pool: 任务回退 + sleep(reset)
    else 401 Token 失效
        W1-->>Pool: Worker 退出 + 任务回退
    end

    Main->>Pool: wait_all_done()
    Main->>Pool: drain_results()
    Pool-->>Main: 逐个回调 on_result / on_error<br/>（串行，无需锁）
    Main->>Pool: shutdown()
```

**关键决策**：
- **1:1 绑定**：消除跨线程 Token 争用
- **主线程回调**：drain_results 串行调用，数据写入无需锁
- **Pool 一次性**：每次 Tool 调用创建 → 销毁，不复用

**API Server 并发控制**：全局 `_tool_execution_lock` — GitHub Token 为全局资源，多会话并发 Pool 会超限。TTL=3600s 清理过期会话，超 100 个 LRU 淘汰。

---

## 6. Agent ReAct 循环

```mermaid
flowchart TD
    INPUT["用户输入"] --> COMPRESS{"消息数 > 30?"}
    COMPRESS -->|是| SUMMARY["LLM 摘要早期对话<br/>保留最近 10 条"]
    COMPRESS -->|否| SKIP["跳过"]
    SUMMARY --> APPEND
    SKIP --> APPEND["追加 user 消息"]
    APPEND --> LLM_CALL["LLM 推理<br/>（带 TOOL_SCHEMAS）"]
    LLM_CALL --> HAS_TC{"有 tool_calls?"}
    HAS_TC -->|否| REPLY["返回文本回复"]
    HAS_TC -->|是| EXEC["_execute_tool"]
    EXEC --> ROUTE["路由到 tool_* 函数"]
    ROUTE --> POOL_OP["创建 Pool → submit → wait → drain"]
    POOL_OP --> UPDATE["更新 AgentState 缓存"]
    UPDATE --> APPEND_RESULT["追加结果到 conversation"]
    APPEND_RESULT --> ROUND{"超 15 轮?"}
    ROUND -->|是| FORCE["强制返回当前结果"]
    ROUND -->|否| LLM_CALL
```

**AgentState 数据链**：
```
last_search_repos → last_candidates → last_ranked
         ↑                 ↑                ↑
    search/scan/       batch_check      rank_candidates
    trending 结果       筛选结果           排序结果
```

---

## 7. 入口层调用关系

```mermaid
graph TD
    CLI["agent_cli.py<br/>REPL 循环"] --> |"agent.chat()"| AGENT["agent.py"]
    API["api_server.py<br/>REST/WS"] --> |"agent.chat()"| AGENT
    MCP["mcp_server.py<br/>MCP stdio"] --> |"直接调用 tool_*()"| TOOLS["agent_tools.py"]
    SCHED["scheduled_update.py"] --> |"Pipeline.run()"| PIPE["execution/pipeline.py"]
    PIPE --> |"tool_*()"| TOOLS
    MAIN["__main__.py"] --> |"main()"| API

    AGENT --> |"_execute_tool()"| TOOLS
```

| 入口 | 命令 | 交互方式 | 经过 Agent? |
|------|------|---------|------------|
| API Server | `python -m github_hot_projects` | REST/WS | ✅ |
| CLI | `python -m github_hot_projects.agent_cli` | REPL | ✅ |
| MCP | `python -m github_hot_projects.mcp_server` | stdio | ❌ 直调 Tool |
| 定时更新 | `python -m github_hot_projects.scheduled_update` | 批处理 | ❌ Pipeline |
| 报告恢复 | `python -m github_hot_projects.regenerate_report --log ...` | 批处理 | ❌ |

---

## 8. 配置参数

### 8.1 全局常量（config.py）

| 常量 | 默认值 | 说明 |
|------|--------|------|
| `MIN_STAR` | 1200 | 项目最低 star 门槛（关键词搜索 + 范围扫描共用） |
| `MAX_STAR` | 45000 | 范围扫描上限 |
| `STAR_GROWTH_THRESHOLD` | 800 | 增长门槛 |
| `HOT_PROJECT_COUNT` | 100 | 综合榜默认 Top N |
| `HOT_NEW_PROJECT_COUNT` | 20 | 新项目榜默认 Top N |
| `GROWTH_CALC_DAYS` | 7 | 增长统计窗口（天），用户可通过 growth_calc_days 自定义 |
| `DAYS_SINCE_CREATED` | 45 | 新项目判定窗口 |
| `DATA_EXPIRE_DAYS` | GROWTH_CALC_DAYS + 1 | DB 有效期（动态计算） |
| `MAX_BINARY_SEARCH_DEPTH` | 20 | 二分法最大深度 |
| `SEARCH_REQUEST_INTERVAL` | 2.5s | Search API 请求间隔 |
| `SEARCH_KEYWORDS` | 25 类 × 150+ 词 | 搜索关键词字典 |

### 8.2 用户可自定义参数（Agent 模式）

| 参数 | 影响 Tool | 默认 | 说明 |
|------|----------|------|------|
| `categories` | search | 全部 25 类 | 搜索类别 |
| `min_star` | search, scan | 1200 | 项目最低 star 门槛 |
| `max_star` | scan | 45000 | 范围扫描上限 |
| `top_n` | rank | 100（综合）/ 20（新项目） | 返回前 N |
| `growth_calc_days` | check_repo_growth, batch_check_growth | 7 | 增长统计窗口 |
| `growth_threshold` | batch_check | 800 | 增长阈值 |
| `days_since_created` | search, scan, batch_check, rank | None | 新项目天数窗口 |
| `mode` | rank | comprehensive | comprehensive / hot_new |

---

## 9. 约束与不变量

### GitHub API 约束
| 编号 | 约束 | 应对 |
|------|------|------|
| C1 | REST 5000 次/小时/Token | 多 Token 轮换 |
| `STAR_GROWTH_THRESHOLD` | 800 | 默认增长门槛 |
| `HOT_PROJECT_COUNT` | 100 | 综合榜默认 Top N |
| `GROWTH_CALC_DAYS` | 7 | 默认增长窗口（Agent 对话可覆盖） |
| C5 | GraphQL 节点 500k/小时 | 每仓库最多 3000 条 |
| `DATA_EXPIRE_DAYS` | `GROWTH_CALC_DAYS + 1`（当前 8） | DB 有效期 |
### 线程安全不变量
| 编号 | 约束 |
|------|------|
| T1 | Worker 1:1 绑定 Token，不跨线程 |
| T2 | Task.execute() 不写 DB，结果通过 result_queue 传回主线程 |
| T3 | on_result/on_error 在主线程调用（drain_results 保证） |

说明：为避免文档与实现漂移，用户可见参数的**精确定义与默认行为**以 `agent.py` 中的 `SYSTEM_PROMPT` 和 `agent_tools.py` 中的 `TOOL_SCHEMAS` 为准；本节仅保留结构摘要。
| T4 | save_db() 使用 threading.Lock + fcntl.flock + 原子写入 |
| T5 | API Server _tool_execution_lock 防多会话并发 Tool |
| T6 | _sessions_lock 保护会话字典读写 |
| `categories` | search | 全部 25 类 | 搜索类别 |
| `min_star` | search / scan | 1200 | 项目最低 star 门槛 |
| `max_star` | scan | 45000 | 范围扫描上限 |
| `top_n` | rank | 综合榜 100 / 新项目榜 20 | 返回前 N 个项目 |
| `growth_calc_days` | check_repo_growth / batch_check / report | 7 | 增长统计窗口 |
| `growth_threshold` | batch_check | 800 | 候选筛选阈值 |
| `days_since_created` | search / scan / batch_check / rank | 仅 hot_new 默认为 45 | 新项目创建窗口 |
| `since` | fetch_trending | weekly | Trending 浏览参数 |
|------|---------|----------|
| 加搜索关键词 | `config.py` SEARCH_KEYWORDS | 无需改其他文件 |
| 加新 Tool | `agent_tools.py` + `agent.py` SYSTEM_PROMPT + TOOL_SCHEMAS + 本文档 | 4 处同步 |
| 改评分模型 | `ranking.py` _calc_score | 确认两种 mode 均正确 |
| 改报告格式 | `report.py` step3_generate_report | — |
| 改 Task 逻辑 | `tasks/task.py` | 确认 on_result 回调接口兼容 |
| 改增长算法 | `growth_estimator.py` | 确认 CalcGrowthTask 接口兼容 |
| 加用户参数 | config + agent_tools + agent + 本文档 | 满足非侵入约束 |

---

## 13. 备忘

**B1 Pool 生命周期**：每次 Tool 调用独立创建 → 使用 → 销毁，绝不复用。Agent 串行 Tool 调用天然保证同一时刻只有一个 Pool 活跃。

**B2 DB 差值法前提**：三条件全满足——db.valid==True + 仓库存在 + star 非 None。任一不满足则降级 REST 或 GraphQL。

**B3 REST→GraphQL 降级**：star 过高(>40k)时 REST 返回 422，自动转 GraphQL 采样。日志 "REST 422, fallback to sampling" 属正常。

**B4 checkpoint**：tool_batch_check_growth 每 CHECKPOINT_BATCH_SIZE 个结果批量写入 checkpoint（非逐个写入）。中断后重启自动恢复，完成后清理。

**B5 Trending HTML 依赖**：GitHub 改版可能导致正则匹配失败 → 返回空列表，不影响其他数据源。

**B6 LLM 容错**：3 次重试全部失败返回空串，不抛异常。报告中对应项目描述为空。

**B7 全局 Tool 互斥锁**：API Server 的 _tool_execution_lock 是有意设计——GitHub Token 全局共享，非 bug。

**B8 原子写入**：save_db 先写 .tmp 再 os.replace，崩溃只丢 .tmp 不损坏原文件。

---

## 附录：数据库结构

```json
{
  "date": "YYYY-MM-DD",
  "valid": true,
  "projects": {
    "owner/repo": {
      "star": 12345, "forks": 678,
      "created_at": "2025-01-15T10:30:00Z",
      "desc": "LLM 180-320字中文描述",
      "short_desc": "GitHub 原始 description",
      "language": "Python",
      "topics": ["llm", "agent"],
      "readme_url": "https://github.com/owner/repo#readme"
    }
  }
}
```
- valid = date 距今 ≤ DATA_EXPIRE_DAYS
- 更新策略：新仓库创建全字段，已有仓库更新 star + 补缺失字段（不覆盖已有 desc 等）
