# GitHub 热门项目发现 — 设计文档

## 1. 项目背景

### 1.1 问题

GitHub 上每天有大量开源项目产生和增长，但开发者很难系统性地发现近期真正在快速增长的热门项目。现有途径（GitHub Trending、社交媒体、技术博客）各有局限：

- **GitHub Trending**：仅展示 daily/weekly/monthly 快照，无法量化增长幅度，且受语言和地区偏好影响
- **手动跟踪**：依赖个人信息渠道，覆盖面窄，容易遗漏新领域
- **第三方排行**：更新滞后，通常只关注已知大项目

### 1.2 目标

构建一个自动化工具，能够：
1. 从多个数据源全面收集 GitHub 仓库（不依赖单一渠道）
2. 量化每个仓库近期（10 天窗口期）的 star 增长量
3. 通过公平的评分模型筛选出真正在增长的热门项目
4. 支持综合排名和新项目专榜两种视角
5. 提供对话式交互（Agent），让用户可以灵活探索

### 1.3 技术选型

| 组件 | 技术 | 说明 |
|------|------|------|
| 语言 | Python 3.10+ | 标准库 + requests |
| GitHub 数据 | REST API + GraphQL + Search API | 三种 API 对应不同场景 |
| Trending 数据 | HTML 正则解析 | 无官方 API，直接爬取 |
| LLM | OpenAI 兼容接口（SiliconFlow） | Qwen3.5-397B-A17B |
| Agent 框架 | 自研 ReAct 循环 | 基于 Function Calling |
| 存储 | JSON 本地文件 | 轻量无依赖 |

## 2. 逻辑方案设计

### 2.1 整体方案

采用四阶段流水线架构，每个阶段职责单一、可独立运行：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Phase 0 — 串行预处理                               │
│  auto_split_star_range(token_idx=0) → segments[]                        │
│  TokenWorkerPool(tokens).start() — N Worker 绑定 N Token               │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                          Phase 1 — 统一收集                             │
│  ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐  │
│  │  KeywordSearchTask  │ │  ScanSegmentTask   │ │  GitHub Trending   │  │
│  │  25 类 × 150+ 词    │ │  2000 ~ 45000      │ │  daily/weekly/     │  │
│  │  stars >= 1000      │ │  自动递归分段       │ │  monthly           │  │
│  │  (Worker Pool)      │ │  (Worker Pool)      │ │  (主线程)          │  │
│  └────────┬───────────┘ └────────┬───────────┘ └────────┬───────────┘  │
│           │                      │                      │               │
│           └──────────────────────┼──────────────────────┘               │
│                                  ↓                                      │
│  pool.wait_all_done() → pool.drain_results() → on_result 合并          │
│                          raw_repos (去重 Map)                           │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                       Phase 2 — 批量增长计算                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  DB 有效 + 已存在?  ──Yes──→  路径A: DB 差值法 (主线程, 0请求)    │  │
│  │       │                                                          │  │
│  │       No                                                         │  │
│  │       ↓                                                          │  │
│  │  CalcGrowthTask (Worker Pool)                                     │  │
│  │       ├─ 路径B: REST 二分法                                       │  │
│  │       └─ 422 → 路径C: GraphQL 采样外推                            │  │
│  │                                                                   │  │
│  │  pool.wait_all_done() → pool.drain_results()                      │  │
│  │  → CalcGrowthTask.on_result(): update_db + checkpoint + candidate │  │
│  │  → CalcGrowthTask.on_error(): 降级处理                             │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                        Phase 3 — 评分排序                               │
│  ┌──────────────────────────┐  ┌──────────────────────────┐            │
│  │  comprehensive 模式       │  │  hot_new 模式             │            │
│  │  log增长量 + log增长率    │  │  created_at ≤ 45天        │            │
│  │  × 平滑折扣               │  │  按 growth 纯降序         │            │
│  └────────────┬─────────────┘  └────────────┬─────────────┘            │
│               └──────────┬──────────────────┘                           │
│                          ↓                                              │
│                    Top N (默认80)                                        │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      Phase 4 — 报告生成                                 │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  desc 为空?  ──Yes──→  LLM 生成 200-400 字中文描述              │    │
│  │  输出 Markdown 报告 → report/YYYY-MM-DD.md                     │    │
│  └────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 三源收集策略

单一数据源无法覆盖全部场景，因此采用三源互补：

| 数据源 | 覆盖范围 | 独特价值 | API 限制应对 |
|--------|----------|----------|-------------|
| **关键词搜索** | 25 类别 × 150+ 关键词，`stars>=1000` | 补充 1000~2000 star 区间的垂直领域项目 | 每个关键词取前 3 页 |
| **Star 范围扫描** | 2000~45000 star | 全量覆盖中高 star 区间，不受关键词限制 | `auto_split_star_range` 递归分段使每段 ≤ 800 |
| **GitHub Trending** | daily / weekly / monthly | 捕捉实时热度，发现冷门关键词以外的项目 | HTML 爬虫，无 API 限制 |

Trending 特殊处理：
- **weekly** 且 `stars_today >= 800`：直接入选 `candidate_map`（7 天增长与 10 天窗口期接近）
- **daily / monthly**：进入 `raw_repos` 走正常增长计算

### 2.3 增长估算方案

目标：估算每个仓库近 `TIME_WINDOW_DAYS`（10 天）的 star 增长量。三条路径按成本递增排列：

**路径 A — DB 差值法（0 次请求）**

```
条件: DB valid=true 且仓库已存在于 DB
计算: growth = current_star - db_saved_star
原理: 上次运行时记录了 star 数，距今 ~10 天，差值即为近期增长
```

**路径 B — REST 二分法（~5-10 次请求）**

```
原理: stargazers API 按 starred_at 时间升序返回，每页 100 条
目标: 找到 "窗口起始时间" 所在的页码 boundary_page
算法:
  1. 获取最后一页 → 如果最老记录在窗口内 → 全部增长 = total_stars
  2. 否则二分搜索: lo=1, hi=total_pages
     → 取 mid 页，检查页内时间戳与窗口起始时间的关系
     → 收敛到 boundary_page
  3. growth = (total_pages - boundary_page) × 100 + 边界页内精确计数
```

**路径 C — GraphQL 采样外推（~5-20 次请求）**

```
触发: REST 二分法返回 HTTP 422（star > 40k 时 page 参数上限超出）
步骤:
  1. GraphQL 游标翻页：从最新 star 向前采集 ~2000 条 starred_at
  2. 若跨越窗口边界 → 精确计数直接返回
  3. 若全在窗口内 → 分段加权速率外推:
     a. 每 100 条一段，计算段速率 = 段内 star 数 / 段覆盖天数
     b. 线性加权：越新的段权重越高（段 i 的权重 = i+1）
     c. 加权平均速率 × 窗口天数 = 预估总增长

  保护机制:
  - 覆盖率检查: 采样覆盖 < 30% 窗口期 → 未覆盖部分使用最低段速率（保守估计）
  - 异常值处理: 最高段速率 > 中位数 × 3 → 使用中位数替代（防止短期 burst 干扰）
```

### 2.4 评分模型

**comprehensive 模式（综合排名）**：

```
growth_score = log(1 + growth) × 1000         # 增长量得分，对数压缩防止极端值主导
rate_score   = log(1 + rate) / log(2) × 3000  # 增长率得分，rate = growth / star
                                                # 上限约 3000（rate=1.0 时）

# 平滑折扣函数（防止新项目因 rate 过高而霸榜）
if rate > 0.5:
    discount = 1.0 - 0.15 × min((rate - 0.5) / 0.5, 1.0)
    # rate=0.5 → 1.0, rate=0.75 → 0.925, rate=1.0 → 0.85
else:
    discount = 1.0

score = (growth_score + rate_score) × discount
```

设计要点：
- **对数压缩**：`log(1+growth)` 使 growth=100 和 growth=10000 的差距从 100 倍缩小到 ~2 倍
- **增长率上限**：`rate_score` 最大 ~3000，不会无限放大小仓库的优势
- **平滑折扣**：从 rate=0.5 开始线性衰减到 0.85，消除原有 rate=0.95 处的断崖式跳变

**hot_new 模式（新项目专榜）**：
```
筛选: created_at 距今 ≤ NEW_PROJECT_DAYS (45天)
排序: growth 纯降序（不使用综合评分，因为新项目 rate 普遍偏高无法区分）
```

### 2.5 Agent 方案

采用 ReAct（Reasoning + Acting）模式：

```
┌─────────────────────────────────────────────────────┐
│                   ReAct Agent                        │
│                                                     │
│  用户消息 → 加入 conversation → 调用 LLM            │
│                                                     │
│  LLM 返回:                                          │
│  ├─ 有 tool_calls → _execute_tool() → Observation  │
│  │                   → 注入 conversation → 回到 LLM │
│  │                   (最多循环 15 次)                │
│  │                                                   │
│  └─ 无 tool_calls → 返回文本回复给用户              │
│                                                     │
│  记忆管理:                                           │
│  ├─ conversation > 30 条 → 压缩(摘要 + 保留最近10) │
│  └─ AgentState 跨轮次缓存候选/结果/偏好             │
└─────────────────────────────────────────────────────┘
```

10 个 Tool 覆盖完整工作流：搜索(2) + 增长计算(2) + 排序(1) + 描述(1) + 报告(1) + DB(1) + 一键执行(1) + Trending(1)。

## 3. 核心模块功能说明

### 3.1 config.py — 全局配置

集中管理所有可调参数，支持环境变量覆盖。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `GITHUB_TOKENS` | 3 个内置 | Token 列表，`GITHUB_TOKENS` 环境变量逗号分隔覆盖 |
| `LLM_API_URL` / `LLM_API_KEY` / `LLM_MODEL` | SiliconFlow / Qwen3.5-397B | LLM 接口 |
| `STAR_GROWTH_THRESHOLD` | 800 | 窗口期增长门槛（加入候选的最低增长量） |
| `MIN_STAR_FILTER` | 1000 | 搜索最低 star 数（过滤噪声小仓库） |
| `HOT_PROJECT_COUNT` | 80 | 最终输出 Top N |
| `TIME_WINDOW_DAYS` | 10 | 增长计算的时间窗口 |
| `NEW_PROJECT_DAYS` | 45 | hot_new 模式新项目判定窗口 |
| `DATA_EXPIRE_DAYS` | 11 | DB 数据过期天数（>11天 → DB差值不可信） |
| `CHECKPOINT_FILE_PATH` | `.pipeline_checkpoint.json` | 断点续传中间文件路径 |
| `DEFAULT_SCORE_MODE` | comprehensive | 默认评分模式 |
| `STAR_RANGE_MIN` / `MAX` | 2000 / 45000 | Star 范围扫描区间 |
| `MAX_BINARY_SEARCH_DEPTH` | 20 | 二分法最大迭代深度 |
| `SEARCH_REQUEST_INTERVAL` | 2.5s | Search API 请求最小间隔 |
| `SEARCH_KEYWORDS` | 25 类 × 150+ 词 | 搜索关键词词典 |

### 3.2 token_manager.py — Token 持有器

**类 `TokenManager`**：持有 GitHub Token 列表，仅负责构建请求头。Token 分配和限流处理由 `worker_pool.TokenWorkerPool` 管理（每个 Worker 绑定固定 Token）。

| 方法 | 说明 |
|------|------|
| `get_rest_headers(token_idx) → dict` | REST API 请求头 |
| `get_star_headers(token_idx) → dict` | stargazers 请求头（含 `Accept: application/vnd.github.v3.star+json`） |
| `get_graphql_headers(token_idx) → dict` | GraphQL 请求头 |

设计变更说明：
- **已移除**：`acquire_token`、`release_token`、`handle_rate_limit` 以及所有锁/轮换/限流逻辑
- **原因**：Token 绑定 Worker 后，不再需要运行时分配和回收；限流由 Worker 自行 sleep 处理

### 3.3 github_api.py — GitHub API 封装

| 函数 | 说明 |
|------|------|
| `search_github_repos(token_mgr, query, token_idx, page, per_page, sort, order, auto_star_filter)` | Search API 搜索（自动附加 `stars:>=1000`，3 次重试）。返回 `list[dict] \| None`（None 表示全部重试失败，[] 表示空结果） |
| `get_search_total_count(token_mgr, query, token_idx) → int` | 获取搜索结果总数（per_page=1 只查 count） |
| `auto_split_star_range(token_mgr, low, high, token_idx, max_results=800, min_span=50)` | **递归分段算法**：查询区间 total_count，超过 800 则二分拆成两个子区间继续递归 |
| `get_stargazers_page(token_mgr, owner, repo, page, token_idx, per_page)` | REST stargazers 分页查询 |
| `graphql_stargazers_batch(token_mgr, owner, repo, token_idx, last, before)` | GraphQL 批量获取 stargazers（`last+before` 逆序翻页） |

设计变更说明：
- **所有函数增加 `token_idx: int` 必选参数**：由调用方（Task.execute 或 Agent Tool）指定使用哪个 Token
- **移除内部 acquire/release**：不再在函数内部管理 Token 生命周期
- **新增 `_check_response(resp, token_idx)` 辅助函数**：401 → `TokenInvalidError`，403/429 → `RateLimitError`
- **`search_github_repos` 返回值变化**：`None` 表示网络失败（全部重试后），`[]` 表示正常空结果。调用方可据此区分"查无结果"和"请求失败"

**`auto_split_star_range` 递归分段**：
```
auto_split_star_range(2000, 45000)
  → total_count("stars:2000..45000") = 3200 > 800
  → mid = 23500
  → auto_split_star_range(2000, 23500)  → [(2000,10000), (10000,23500)]
  → auto_split_star_range(23501, 45000) → [(23501,45000)]
  → 结果: [(2000,10000), (10000,23500), (23501,45000)]
```

### 3.4 growth_estimator.py — 增长估算器

| 函数 | 说明 |
|------|------|
| `estimate_star_growth_binary(token_mgr, owner, repo, total_stars, token_idx=0) → int` | REST 二分法（路径 B） |
| `estimate_by_sampling(token_mgr, owner, repo, token_idx=0) → int` | GraphQL 采样外推（路径 C） |

设计变更：所有函数增加 `token_idx` 参数（默认 0），透传到 `get_stargazers_page` / `graphql_stargazers_batch`。

**二分法详细流程**：
```
1. total_pages = ceil(total_stars / 100)
2. 获取最后一页 last_page
   → 如果最老记录在窗口内 → return total_stars（全部是近期增长）
3. lo=1, hi=total_pages, 二分查找:
   → mid = (lo+hi)//2
   → 获取 mid 页最后一条的 starred_at
   → 在窗口内 → hi=mid  |  在窗口外 → lo=mid+1
4. boundary_page 确定后:
   → growth = (total_pages - boundary_page) × 100 + 边界页内精确计数
```

**采样外推详细计算**：
```
1. 采 ~2000 条 starred_at（每批 100 条 GraphQL 翻页）
2. 跨越窗口边界 → 精确计数
3. 全在窗口内 → 分段：
   段 0: star[0..99],   段 1: star[100..199], ...
   段速率 r[i] = 100 / (段首时间 - 段尾时间).total_seconds() × 86400
   加权速率 = Σ(r[i] × (i+1)) / Σ(i+1)

   保护：
   - 覆盖率 = 采样跨度 / 窗口天数
   - 覆盖率 < 0.3 → 未覆盖部分用 min(r[i]) 填补
   - max(r[i]) > median(r[i]) × 3 → 用 median 替代 max
```

### 3.5 github_trending.py — Trending 爬虫

| 函数 | 说明 |
|------|------|
| `fetch_trending(since="daily", language="", spoken_language="") → list[dict]` | 爬取 GitHub Trending |
| `_parse_trending_html(html, since) → list[dict]` | 正则按 `<article>` 分段提取 |

返回字段：`full_name`, `star`, `forks`, `stars_today`, `description`, `language`, `since`。

### 3.6 db.py — JSON 数据库

| 函数 | 说明 |
|------|------|
| `load_db() → dict` | 加载 `Github_DB.json`，距上次更新 > 11 天则 `valid=false` |
| `save_db(db)` | 保存并更新 `date` 字段为今天（线程安全 + 原子写入） |
| `update_db_project(db_projects, full_name, current_star, repo_item)` | 更新 star / forks / created_at 等字段 |

线程安全机制：
- **`_db_lock`**：全局 `threading.Lock()` 保护 `save_db()` 并发写入
- **原子写入**：先写 `.tmp` 文件，再 `os.replace()` 原子替换，防止崩溃导致 JSON 损坏

事务性设计：
- **全局 `date` + `valid`**：只在完整流程成功后才更新 `valid=true` 并刷新 `date`
- **中途崩溃不污染**：未完成的运行不会刷新 `date`，下次运行通过断点续传恢复已计算结果

DB 结构：
```json
{
  "date": "YYYY-MM-DD",
  "valid": true,
  "projects": {
    "owner/repo": {
      "star": 12345, "forks": 678,
      "created_at": "2025-01-15T10:30:00Z",
      "desc": "LLM 生成的描述",
      "short_desc": "GitHub 原始 description",
      "language": "Python",
      "topics": ["llm", "agent"],
      "readme_url": "https://github.com/owner/repo#readme"
    }
  }
}
```

### 3.7 llm.py — LLM 调用

| 函数 | 说明 |
|------|------|
| `call_llm_describe(repo_name, repo_info, html_url) → str` | 生成 200-400 字中文描述（3 次重试） |

传入仓库元信息（description / language / topics / readme_url），LLM 综合生成中文简介。

### 3.8 worker_pool.py — 统一 Worker 池

**异常层级**

| 异常 | 说明 | Worker 行为 |
|------|------|-------------|
| `RetryableError(reset_time)` | 通用可重试错误 | sleep 到 reset_time → 回退任务 → 继续 |
| `RateLimitError(token_idx, reset_time)` | GitHub 限流（继承 RetryableError） | 同上 |
| `FatalWorkerError` | 通用致命错误 | 回退任务 → Worker 退出 |
| `TokenInvalidError(token_idx)` | Token 失效（继承 FatalWorkerError） | 同上 |

**Task 基类（ABC + dataclass）**

| 字段/方法 | 说明 |
|-----------|------|
| `needs_token: bool = True` | 是否需要 GitHub Token（False 时 execute 收到 None） |
| `_token_mgr: Any = None` | TokenManager 引用 |
| `execute(token_idx) → Any` | 抽象方法，子类必须实现 |
| `on_result(result)` | 结果处理回调，主线程调用（可选覆盖） |
| `on_error(error)` | 错误处理回调，主线程调用（可选覆盖） |

**TokenWorkerPool**

| 方法 | 说明 |
|------|------|
| `__init__(tokens)` | 创建 N 个 Worker 绑定 N 个 Token |
| `start()` | 启动 daemon 线程 |
| `submit(task)` | 任务入队 |
| `wait_all_done(timeout)` | 阻塞等待所有任务完成 |
| `drain_results()` | 消费 result_queue，调用每个 Task 的 on_result/on_error，返回消费数 |
| `shutdown()` | 停止信号 + join |
| `active_workers` | 当前存活 Worker 数 |

Worker 主循环：
```
while True:
    task = queue.get()
    try:
        result = task.execute(token_idx)  # needs_token=False 时传 None
        → result_queue.put((task, result, None))
        → mark_done
    except FatalWorkerError:
        → queue.put(task)  # 回退
        → worker_exit + break
    except RetryableError as e:
        → sleep(reset_time + 5)
        → queue.put(task)  # 回退，不 mark_done
    except Exception:
        → result_queue.put((task, None, e))
        → mark_done
```

### 3.9 pipeline.py — 四阶段流水线

**Task 子类定义**

| Task | needs_token | execute 返回 | on_result 逻辑 |
|------|-------------|-------------|----------------|
| `KeywordSearchTask` | True | `list[dict]` | 合并到 `_raw_repos` |
| `ScanSegmentTask` | True | `list[dict]` | 合并到 `_raw_repos` |
| `CalcGrowthTask` | True | `(name, growth, star)` | 更新 checkpoint + DB + candidate_map |

**函数**

| 函数 | 阶段 | 说明 |
|------|------|------|
| `collect_from_trending(raw_repos, candidate_map)` | Phase 1 | weekly 直接入选 / 其余进 raw_repos |
| `_submit_growth_tasks(pool, token_mgr, raw_repos, db, candidate_map, growth_ctx)` | Phase 2 | DB 差值法主线程处理，其余提交 CalcGrowthTask |
| `_load_checkpoint() → dict` | Phase 2 | 加载断点续传文件 |
| `_save_checkpoint(completed)` | Phase 2 | 原子写入已完成的增长计算结果 |
| `_remove_checkpoint()` | Phase 2 | 流程成功后删除断点文件 |
| `_upsert_candidate(candidate_map, full_name, growth, ...)` | — | 候选更新/插入 |
| `step2_rank_and_select(candidate_map, mode) → list[tuple]` | Phase 3 | 评分排序 |
| `step3_generate_report(top_projects, db) → str` | Phase 4 | LLM 描述 + 报告 |
| `main()` | 全部 | 流水线入口 |

**main() 执行流程**

```
Phase 0: auto_split_star_range(token_idx=0)   # 串行分段
         pool = TokenWorkerPool(tokens)
         pool.start()

Phase 1: pool.submit(KeywordSearchTask × 164)  # 关键词搜索
         pool.submit(ScanSegmentTask × 56)      # Star 扫描
         collect_from_trending()                 # 主线程 HTML
         pool.wait_all_done()
         pool.drain_results()                    # on_result → raw_repos 合并

Phase 2: _submit_growth_tasks(pool, ..., growth_ctx)
         pool.wait_all_done()
         pool.drain_results()                    # on_result → checkpoint + DB + candidate_map

Phase 3: step2_rank_and_select(candidate_map)
Phase 4: step3_generate_report(top_projects, db)
         pool.shutdown()
```

### 3.10 agent_tools.py — 10 个 Agent Tool

| Tool | 参数 | 功能 |
|------|------|------|
| `search_hot_projects` | `categories, min_stars=1000, max_pages=3` | 按关键词类别搜索 |
| `scan_star_range` | `min_star=2000, max_star=45000` | Star 范围扫描 |
| `check_repo_growth` | `repo` | 单仓库增长查询 |
| `batch_check_growth` | `repos, growth_threshold=800` | 批量增长计算 |
| `rank_candidates` | `top_n=80, mode="comprehensive"` | 评分排序 |
| `describe_project` | `repo` | LLM 生成项目描述 |
| `generate_report` | — | 报告生成 |
| `get_db_info` | `repo=None` | DB 查询/统计 |
| `full_discovery` | — | 一键执行完整流水线 |
| `fetch_trending` | `since="daily", language, spoken_language` | Trending 爬虫 |

每个 Tool 通过 `TOOL_SCHEMAS` 定义 OpenAI Function Calling 格式 schema。

### 3.11 agent.py — ReAct Agent

**`AgentState`（dataclass）**：

| 字段 | 说明 |
|------|------|
| `token_mgr` | Token 管理器 |
| `db` | 数据库字典 |
| `conversation` | 完整对话历史 |
| `last_search_repos` | 最近搜索结果缓存 |
| `last_candidates` | 候选列表缓存 |
| `last_ranked` | 排序结果缓存 |
| `seen_repos` | 已扫描仓库集（避免重复） |
| `user_preferences` | 用户偏好 |
| `conversation_summary` | 早期对话压缩摘要 |

**`HotProjectAgent`**：

| 方法 | 说明 |
|------|------|
| `chat(user_message) → str` | 主入口：加入消息 → ReAct 循环 → 返回回复 |
| `_call_llm() → dict` | 调用 LLM（带 Tool 定义，3 次重试） |
| `_execute_tool(name, args) → dict` | 路由分发 10 个 Tool |
| `_compress_conversation()` | 对话 > 30 条时：LLM 摘要 + 保留最近 10 条 |

### 3.12 api_server.py — FastAPI Web 服务

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/status` | GET | 服务状态 |
| `/api/chat` | POST | 对话（`ChatRequest` → `ChatResponse`） |
| `/api/reports` | GET | 报告列表 |
| `/api/reports/{name}` | GET | 报告详情 |
| `/api/sessions/{sid}` | DELETE | 清除会话 |
| `/ws/chat/{sid}` | WebSocket | 实时对话（预留） |

内存会话管理 `_sessions`，按 `session_id` 隔离 Agent 实例。

## 4. 约束说明

### 4.1 GitHub API 限制

| 约束 | 影响 | 应对策略 |
|------|------|----------|
| REST API 5000 次/小时/Token | 全局请求预算 | 多 Token 轮换（3 个 = 15000 次/小时） |
| Search API 30 次/分钟 | 关键词搜索频率 | `SEARCH_REQUEST_INTERVAL=2.5s` 间隔控制 |
| Search 结果上限 1000 条/查询 | Star 范围扫描 | `auto_split_star_range` 递归分段 |
| Stargazers 页码上限 400 页 | 大仓库（>40k star） | 自动回退到 GraphQL 采样路径 |
| GraphQL 节点上限 500k/小时 | 采样外推的请求总量 | 最多采 2000 条（20 次请求） |

### 4.2 数据约束

| 约束 | 说明 |
|------|------|
| DB 有效期 11 天 | 超过 11 天未更新则 DB 差值不可信（`valid=false`） |
| Star ≥ 1000 过滤 | 低于 1000 star 的仓库数量过大且噪声高 |
| Growth ≥ 800 才入候选 | 10 天增长 800 约合日均 80，代表显著增长趋势 |
| 新项目 ≤ 45 天 | `created_at` 距今不超过 45 天才进入 hot_new 排名 |

### 4.3 评分约束

| 约束 | 原因 |
|------|------|
| 对数压缩增长量和增长率 | 防止极端值（刷星 / 爆发式增长）无限拉开差距 |
| rate_score 上限 ~3000 | 避免小仓库因增长率过高而霸榜 |
| 平滑折扣 rate>0.5 开始 | 消除原有 rate=0.95 断崖跳变，公平对待增长率连续体 |
| hot_new 不使用综合评分 | 新项目 rate 普遍偏高，使用综合评分无法有效区分 |

### 4.4 采样约束

| 约束 | 说明 |
|------|------|
| 覆盖率 < 30% 保守估计 | 采样只覆盖窗口期一小部分时，用最低段速率填补（宁可低估不高估） |
| 最高段 > 3×中位数 用中位数 | 防止短期 star burst（如 Hacker News 效应）导致外推过高 |
| 最多采 2000 条 | 平衡精度和 API 消耗 |

### 4.5 线程安全约束

| 约束 | 说明 |
|------|------|
| **Worker 绑定 Token** | 每个 Worker 固定使用一个 Token，无跨线程 Token 争用 |
| **Worker 不写 DB** | `CalcGrowthTask.execute()` 只做 API 调用和计算，返回结果 |
| **on_result/on_error 在主线程** | `pool.drain_results()` 在主线程消费 result_queue，回调中更新 DB、checkpoint、candidate_map 均无并发 |
| **异常层级驱动 Worker 行为** | `RetryableError` → 回退重试，`FatalWorkerError` → Worker 退出并回退任务（其他 Worker 接手） |
| `save_db()` 线程安全 | `_db_lock` 全局锁 + 原子写入（`.tmp` + `os.replace()`） |
| `api_server._sessions` 加锁 | `_sessions_lock` 保护会话字典的所有读写操作 |
| `_save_checkpoint()` 原子写入 | 先写 `.tmp` 再 `os.replace()`，防止半写入损坏 |

## 5. 数据流全景图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Phase 0: 串行预处理 (主线程)                               │
│  auto_split_star_range(token_idx=0) → segments[]                                │
│  pool = TokenWorkerPool(tokens); pool.start()                                   │
└─────────────────────────────────────────────────┬───────────────────────────────┘
                                                  │
                                                  ↓
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               Phase 1: 统一收集                                 │
│                                                                                 │
│  ┌──────────────────────────┐                                                   │
│  │  SEARCH_KEYWORDS (25类)  │                                                   │
│  │  → KeywordSearchTask     │──────────────────┐                                │
│  │    (TokenWorkerPool)     │                  │                                │
│  └──────────────────────────┘                  │                                │
│                                                 │                                │
│  ┌──────────────────────────┐                  ↓                                │
│  │  STAR_RANGE segments     │          ┌───────────────┐                        │
│  │  → ScanSegmentTask       │──────────→│  raw_repos    │                        │
│  │    (TokenWorkerPool)     │          │  {name: info} │                        │
│  └──────────────────────────┘          │  去重 + ≥1000 │                        │
│                                         └───────┬───────┘                        │
│  ┌──────────────────────────┐                  ↑                                │
│  │  GitHub Trending         │                  │                                │
│  │  → fetch_trending()      │──daily/monthly──→┘                                │
│  │  (主线程)                │──weekly≥800──→ candidate_map (直接入选)            │
│  └──────────────────────────┘                                                   │
│                                                                                 │
│  pool.wait_all_done() → pool.drain_results() → Task.on_result() 合并 raw_repos │
└─────────────────────────────────────────────────┬───────────────────────────────┘
                                                  │
                                                  ↓
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Phase 2: 批量增长计算                                  │
│                                                                                 │
│  raw_repos ──→ _submit_growth_tasks(pool, ..., growth_ctx)                      │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐        │
│  │  1. 加载 checkpoint → 恢复已计算项目 → save_db()                    │        │
│  │  2. DB valid + 已存在?                                              │        │
│  │     ├─ Yes → 路径A: current - db_star (主线程, 0请求)               │        │
│  │     └─ No  → CalcGrowthTask (TokenWorkerPool)                       │        │
│  │               ├─ 路径B: estimate_star_growth_binary()                │        │
│  │               └─ 422 → 路径C: estimate_by_sampling()                │        │
│  │  3. pool.wait_all_done()                                            │        │
│  │  4. pool.drain_results() → CalcGrowthTask.on_result():              │        │
│  │     → update_db + checkpoint + growth≥800 → candidate_map           │        │
│  │  5. 最终 save_checkpoint → 流程成功后删除                            │        │
│  └─────────────────────────────────────────────────────────────────────┘        │
│                                                                                 │
│  异常处理:                                                                       │
│  ├─ RateLimitError → Worker sleep 到 reset_time → 任务回退重试                  │
│  └─ TokenInvalidError → Worker 退出, 任务回退给其他 Worker                       │
│                                                                                 │
│  Github_DB.json ←──── save_db()  (仅在完整流程成功后更新 valid=true + date)     │
│  .pipeline_checkpoint.json ←──── 流程成功后删除                                 │
└─────────────────────────────────────────────────┬───────────────────────────────┘
                                                  │
                                                  ↓
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            Phase 3: 评分排序                                     │
│                                                                                 │
│  candidate_map ──→ step2_rank_and_select(mode)                                  │
│                                                                                 │
│  ┌──────────────────────────────┐   ┌──────────────────────────────┐            │
│  │  comprehensive               │   │  hot_new                     │            │
│  │  log(growth) + log(rate)     │   │  created_at ≤ 45天           │            │
│  │  × smooth_discount           │   │  按 growth 降序              │            │
│  └──────────────┬───────────────┘   └──────────────┬───────────────┘            │
│                 └───────────────┬──────────────────┘                             │
│                                 ↓                                                │
│                           Top N (默认80)                                         │
└─────────────────────────────────────────────────┬───────────────────────────────┘
                                                  │
                                                  ↓
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            Phase 4: 报告生成                                     │
│                                                                                 │
│  top_projects ──→ step3_generate_report()                                       │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────┐                        │
│  │  desc 为空? → call_llm_describe() → 200-400字描述   │                        │
│  │  输出 → report/YYYY-MM-DD.md                        │                        │
│  └─────────────────────────────────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────────────────┘
```
