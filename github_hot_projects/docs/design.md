# GitHub 热门项目发现系统 — 设计文档

## 1. 设计目标与约束

### 1.1 核心需求
自动发现 GitHub 近期 star 增长最快的开源项目，生成结构化中文报告。

### 1.2 设计约束
- GitHub REST API 限制 5000 次/小时/Token，Search API 限制 30 次/分钟
- Search API 单次查询最多返回 1000 条结果
- Stargazers REST 分页在高 star 仓库（>40k）时返回 422
- 需支持多 Token 并行以提升吞吐
- 需支持交互式（CLI）和 API（Web）两种使用模式

### 1.3 技术选型
Python 3.10+, requests, FastAPI, JSON 文件存储, OpenAI 兼容 LLM (SiliconFlow Qwen3.5-397B)

---

## 2. 架构设计

### 2.1 分层架构

```
┌─────────────────────────────────────────────────────────┐
│                      入口层                              │
│   agent_cli.py              api_server.py             │
├─────────────────────────────────────────────────────────┤
│                      Agent 层                            │
│   agent.py（ReAct 循环 + 状态管理 + 对话压缩）           │
├─────────────────────────────────────────────────────────┤
│                      工具层                              │
│   agent_tools.py（9 个 Tool 函数 + TOOL_SCHEMAS）       │
├─────────────────────────────────────────────────────────┤
│                      业务逻辑层                           │
│   tasks.py（Task子类 + 批量提交 + 断点续传）             │
│   scorer.py（评分排序）  report.py（报告生成）           │
├─────────────────────────────────────────────────────────┤
│                      核心能力层                           │
│   github_api.py    growth_estimator.py                  │
│   github_trending.py    llm.py                          │
├─────────────────────────────────────────────────────────┤
│                      基础设施层                           │
│   worker_pool.py   token_manager.py   db.py   config.py │
└─────────────────────────────────────────────────────────┘
```

### 2.2 设计原则
- **模式统一**：Agent CLI / API Server 两入口共享 agent_tools 中的 Tool 实现，核心逻辑无分歧
- **Token 隔离**：每个 Worker 绑定唯一 Token，消除跨线程竞争
- **主线程回调**：所有数据写入（DB、候选集）在主线程回调中执行，无需额外锁
- **降级容错**：REST → GraphQL → 保守估算，每层失败都有退路
- **非侵入参数**：新增参数不传时行为等同参数添加前

### 2.3 文件职责映射

| 层 | 文件 | 职责 |
|----|------|------|
| 入口 | `agent_cli.py` | CLI REPL 入口：用户输入 → agent.chat() → 输出 |
| 入口 | `api_server.py` | Web 服务：REST/WS 端点 + 会话管理（TTL+LRU）+ 全局 Tool 互斥锁 |
| Agent | `agent.py` | ReAct 循环 + AgentState + SYSTEM_PROMPT + 对话压缩 |
| 工具 | `agent_tools.py` | 9 个 tool_* 函数 + TOOL_SCHEMAS |
| 业务 | `tasks.py` | Task 子类（搜索/扫描/增长）+ 批量提交 + 断点续传 + 候选管理 |
| 业务 | `scorer.py` | 评分排序（comprehensive / hot_new） |
| 业务 | `report.py` | Markdown 报告生成（LLM 描述 + 文件输出） |
| 核心 | `github_api.py` | Search/Stargazers/GraphQL API 封装 + star 范围二分 |
| 核心 | `growth_estimator.py` | REST 二分法 + GraphQL 采样外推 |
| 核心 | `github_trending.py` | Trending HTML 爬虫（零 API 消耗） |
| 核心 | `llm.py` | LLM 描述生成（180-320 字中文，3 次重试） |
| 基础 | `worker_pool.py` | TokenWorkerPool + Task ABC + 异常控制（限流/失效） |
| 基础 | `token_manager.py` | Token 管理 + 三类 API 请求头构建（REST/Star/GraphQL） |
| 基础 | `db.py` | JSON 数据库加载/原子保存/记录更新 |
| 基础 | `config.py` | 全局常量 + 环境变量覆盖 + SEARCH_KEYWORDS |

> 注：token_manager.py 实际职责包含请求头构建，不仅是 Token 管理。

### 2.4 两种运行模式

| 模式 | 入口命令 | 交互方式 | 适用场景 |
|------|---------|---------|----------|
| Agent CLI | `python -m github_hot_projects.agent_cli` | REPL | 探索、调试 |
| API Server | `uvicorn ...api_server:app` | REST/WebSocket | Web 集成 |

Agent CLI 和 API Server 共享同一套 agent_tools 中的 Tool 实现，通过 ReAct 循环按需调用 Tool。

---

## 3. 数据收集设计

```
                          数据收集（Phase 1）
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │  关键词搜索   │    │ Star范围扫描  │    │ Trending爬虫  │
  │  Search API  │    │  Search API  │    │  纯HTTP爬取  │
  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
         │                   │                   │
  25类×150+关键词      auto_split_star_     daily/weekly/monthly
  × 最多3页/词         range 递归二分        HTML正则解析
         │             (每段≤800条)              │
         │                   │                   │
  自动附加 stars>=      按 updated 排序     weekly 且 stars_today
  STAR_RANGE_MIN       逐段扫描             ≥ 阈值?
         │                   │               ├─是→ 直入候选
  new_project_days?          │               └─否→ ↓
  ├─有→ +created:>=date      │                   │
  └─无→ 不变                 │                   │
         │                   │                   │
         └────────────────→ 去重合并 ←───────────┘
                               │
                               ▼
                           raw_repos
```

**设计理由**：关键词搜索按技术领域精准覆盖；Star 扫描兜底无关键词匹配的项目；Trending 零 API 消耗捕捉突发热度。三源互补不遗漏。

**Search 1000 条限制**：auto_split_star_range 递归二分至每段 ≤800 条或范围宽度 ≤50，Pool 启动前主线程串行调用。

---

## 4. 增长量化设计

```
单仓库增长估算（Phase 2）
  │
  ▼
DB有效(≤DATA_EXPIRE_DAYS) 且 仓库存在 且 star非空?
  │
  ├─ 是 → DB差值法: current_star - db_star
  │        精度: 精确 | API消耗: 0
  │        → 增长值
  │
  └─ 否 → REST二分法
            │
            取 stargazers 最后一页
            │
            最早点赞在窗口内?
            ├─ 是 → 全量计数 → 增长值
            │
            └─ 否 → 二分搜索页码（深度≤20）
                     定位 TIME_WINDOW 天前边界
                     │
                     ├─ 成功 → 精确计数边界页内记录 → 增长值
                     │
                     └─ 422(star>40k页码越界)
                          │
                          ▼
                     GraphQL采样外推
                     游标翻页采集≤2000条时间戳
                          │
                     跨越窗口边界?
                     ├─ 是 → 精确计数 → 增长值
                     └─ 否 → 100条分段加权外推
                              (越新权重越高)
                              + 覆盖率补偿
                              + burst防护
                              → 估算增长值
  │
  ▼
增长 ≥ STAR_GROWTH_THRESHOLD ?
  ├─ 是 → 进入候选 + 写DB + 写checkpoint
  └─ 否 → 丢弃
```

**三级降级设计理由**：DB 差值零开销最优先；REST 精确但有页码限制；GraphQL 无页码限制但只能估算。逐级退路确保任何仓库都能得到增长值。

---

## 5. 评分排序设计

```
候选排序（Phase 3）
  │
  ▼
mode = ?
  │
  ├─ comprehensive（综合）
  │    │
  │    计算 growth_score = log(1+growth) × 1000
  │    计算 rate_score = log(1+growth/star) / log(2) × 3000
  │    │
  │    growth_rate > 0.5 ?
  │    ├─ 是 → discount = 最多 0.85 折扣（防新项目虚高rate霸榜）
  │    └─ 否 → discount = 1.0
  │    │
  │    score = (growth_score + rate_score) × discount
  │    → 按 score 降序取 top_n
  │
  └─ hot_new（新项目发现）
       │
       created_at 距今 ≤ new_project_days ?
       ├─ 是 → 保留，按 growth 降序
       └─ 否 → 排除
       → 取 top_n
```

---

## 6. 并发设计

```
Tool 调用触发 Pool 执行
  │
  ▼
创建 TokenWorkerPool（N Worker ↔ N Token，1:1绑定）
  │
  ▼
submit(tasks) → 任务入队（共享队列）
  │
  ▼
各 Worker 线程取任务 → task.execute(token_idx)
  │
  ├─ 200 OK → result 入结果队列
  │
  ├─ 403/429 RateLimitError
  │    → 任务回退队列 + Worker sleep(reset_time) → 重试
  │
  └─ 401 TokenInvalidError
       → Worker 退出 + 任务回退给其他 Worker
  │
  ▼
wait_all_done → 阻塞至所有任务完成
  │
  ▼
drain_results → 主线程逐个调用 on_result / on_error
               （所有数据写入 DB/候选集 在此处，无需锁）
  │
  ▼
shutdown → 停止所有 Worker → Pool 销毁
```

**关键设计决策**：
- **1:1 绑定而非 Token 池**：消除跨线程 Token 争用，简化限流处理
- **主线程回调**：drain_results 串行调用回调，数据写入无需额外锁
- **生命周期独立**：每次 Tool 调用独立创建 → 销毁，绝不复用（Token 可能过期、Worker 可能已退出）

### API Server 并发控制
全局 `_tool_execution_lock` 互斥锁——所有会话的 Tool 执行串行化。**设计理由**：GitHub Token 是全局共享资源，多会话并发 Pool 会导致限流超标。TTL=3600s 自动清理过期会话，超 100 个 LRU 淘汰。

---

## 7. Agent 设计

### 7.1 ReAct 循环

```
用户输入
  │
  ▼
压缩检查 ←── 消息数 > 30 ? LLM 摘要早期对话(保留最近10条) : 跳过
  │
  ▼
追加 user 消息到 conversation
  │
  ▼
┌──────────────────────────────┐
│ LLM 推理（带 TOOL_SCHEMAS） │◄──────────────────────┐
└─────────────┬────────────────┘                       │
        ┌─────┴─────┐                                  │
        ▼           ▼                                  │
   有 tool_calls  无 tool_calls                        │
        │           │                                  │
        │           ▼                                  │
        │       返回文本回复                            │
        ▼                                              │
  _execute_tool                                        │
   ├─ 路由到 tool_* 函数                               │
   ├─ 创建 Pool → submit → wait → drain → shutdown    │
   ├─ 更新 AgentState 缓存字段                         │
   └─ 追加结果到 conversation ─────────────────────────┘
        │
        └─ 超 15 轮 → 强制返回当前结果
```

### 7.2 状态管理（AgentState）
- **会话级**：token_mgr, db, conversation, user_preferences
- **工具缓存**：last_search_repos → last_candidates → last_ranked（跨 Tool 共享数据链）
- **去重集**：seen_repos，跨轮次累积避免重复处理

### 7.3 SYSTEM_PROMPT 设计
f-string 模板，运行时渲染 config 常量值。包含角色定义、可用参数表、三种工作流说明（全量分析/快速查询/新项目发现）、辅助功能。

---

## 8. Tool 接口设计

9 个 Tool，每个封装 Pool 创建 → 提交 → 等待 → 回收 → 关闭的完整生命周期。

| Tool | 职责 | 写 DB | 创建 Pool |
|------|------|-------|----------|
| search_hot_projects | 关键词搜索仓库 | 否 | 是 |
| scan_star_range | Star 范围扫描仓库 | 否 | 是 |
| check_repo_growth | 单仓库增长查询+LLM描述 | 否 | 否 |
| batch_check_growth | 批量增长筛选 | 内存 | 是 |
| rank_candidates | 候选评分排序 | 否 | 否 |
| describe_project | LLM 描述生成 | 内存 | 否 |
| generate_report | 报告输出+保存 DB | 磁盘 | 否 |
| get_db_info | DB 查询 | 否 | 否 |
| fetch_trending | Trending 爬取 | 否 | 否 |

**TOOL_SCHEMAS 同步约束**：修改 Tool 签名时必须同步 4 处——(1) TOOL_SCHEMAS (2) _execute_tool 路由 (3) SYSTEM_PROMPT (4) 本文档

---

## 9. 业务逻辑模块（tasks.py / scorer.py / report.py）

从 agent_tools.py 拆分出的三个业务逻辑模块，由 Tool 函数内部调用：

**tasks.py** — Task 子类 + 数据收集辅助：
- **KeywordSearchTask**：单关键词多页搜索，on_result 去重汇入 raw_repos
- **ScanSegmentTask**：单 star 范围分段扫描（按 updated 排序），on_result 去重汇入
- **CalcGrowthTask**：单仓库增长计算，on_result 更新 DB + 候选集 + checkpoint
- **_submit_growth_tasks**：批量增长计算入队（DB差值 + checkpoint恢复 + Pool提交）
- **_upsert_candidate**：候选更新/插入（取更大 growth）
- **checkpoint 函数**：_save/_load/_remove，每 CHECKPOINT_BATCH_SIZE=10 批量落盘

三个 Task 子类均支持 created_after / min_stars_override 可选过滤（非侵入设计）。

**scorer.py** — 评分排序：
- comprehensive 模式：log(增长量) + log(增长率)，新项目平滑折扣（rate>0.5 线性衰减）
- hot_new 模式：先补全 created_at，再按增长量排序

**report.py** — 报告生成：
- 串行 LLM 补描述 → 写 report/YYYY-MM-DD.md

---

## 10. 配置与参数

### 10.1 全局常量（config.py）

| 常量 | 默认值 | 说明 |
|------|--------|------|
| `MIN_STAR_FILTER` | 1000 | 搜索最低 star |
| `STAR_GROWTH_THRESHOLD` | 800 | 增长门槛（10 天） |
| `HOT_PROJECT_COUNT` | 80 | 默认 Top N |
| `TIME_WINDOW_DAYS` | 10 | 增长窗口（不可用户自定义） |
| `NEW_PROJECT_DAYS` | 45 | 新项目判定窗口 |
| `DATA_EXPIRE_DAYS` | 11 | DB 有效期 |
| `STAR_RANGE_MIN` / `MAX` | 1300 / 45000 | Star 扫描范围 |
| `MAX_BINARY_SEARCH_DEPTH` | 20 | 二分法最大深度 |
| `SEARCH_REQUEST_INTERVAL` | 2.5s | Search API 请求间隔 |
| `SEARCH_KEYWORDS` | 25 类 × 150+ 词 | 搜索关键词字典 |

### 10.2 用户可自定义参数（Agent 模式）

| 参数 | 影响 Tool | 默认 | 说明 |
|------|----------|------|------|
| `categories` | search | 全部 25 类 | 搜索类别 |
| `min_stars` | search | 1000 | 最低 star |
| `top_n` | rank | 80 | 返回前 N |
| `growth_threshold` | batch_check | 800 | 增长阈值 |
| `new_project_days` | search, scan, batch_check, rank | None | 新项目天数窗口 |
| `min_star` / `max_star` | scan | 2000 / 45000 | star 扫描范围 |
| `mode` | rank | comprehensive | comprehensive / hot_new |

### 10.3 参数设计规则
- **非侵入**：任何新增参数 None/""/0 时跳过对应逻辑，等同参数添加前
- **new_project_days 联动**：设定时 → search/scan 附加 created 过滤，batch_check 按 created_at 预过滤，rank hot_new 按窗口筛选。star 范围统一使用 STAR_RANGE_MIN/MAX 默认值，用户指定则用用户的。不设定时四处分支均不进入

---

## 11. 约束与不变量

### GitHub API 约束
- **C1**: REST 5000 次/小时/Token → 多 Token 轮换
- **C2**: Search 30 次/分钟 → `SEARCH_REQUEST_INTERVAL=2.5s`
- **C3**: Search 结果上限 1000 条 → auto_split_star_range 递归分段 ≤800
- **C4**: Stargazers 页码上限 ~400 页 → star>40k 回退 GraphQL
- **C5**: GraphQL 节点 500k/小时 → 每仓库最多 2000 条

### 线程安全不变量
- **T1**: Worker 1:1 绑定 Token，不跨线程
- **T2**: Task.execute() 不写 DB，结果通过 result_queue 传回主线程
- **T3**: on_result/on_error 在主线程调用（drain_results 保证）
- **T4**: save_db() 使用 _db_lock + 原子写入（.tmp + os.replace）
- **T5**: API Server _tool_execution_lock 防多会话并发
- **T6**: _sessions_lock 保护会话字典读写

### DB 约束
- 有效期 > DATA_EXPIRE_DAYS → valid=false → DB 差值法不可用
- 写入：先 .tmp 再 os.replace() 原子替换，崩溃不损坏原文件

---

## 12. 扩展指南

| 场景 | 修改文件 | 注意事项 |
|------|---------|----------|
| 加搜索关键词 | `config.py` SEARCH_KEYWORDS | 无需改其他文件 |
| 加新 Tool | `agent_tools.py` + `agent.py` + 本文档 | 共 4 处同步（§8） |
| 改评分模型 | `scorer.py` _calc_score | 确认两种 mode 均正确 |
| 改报告格式 | `report.py` step3_generate_report | — |
| 改 Task 逻辑 | `tasks.py` | 确认 on_result 回调接口兼容 |
| 加用户参数 | config + agent_tools + agent + 本文档 | 满足非侵入约束 |
| 改增长算法 | `growth_estimator.py` | 确认 CalcGrowthTask 接口兼容 |

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
