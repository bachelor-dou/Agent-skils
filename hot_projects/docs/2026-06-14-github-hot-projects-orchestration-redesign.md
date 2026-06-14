# GitHub 热门项目发现系统 — 编排层重构设计

日期：2026-06-14
状态：已与用户确认设计方向，待进入实现计划（writing-plans）

---

## 1. 背景与目标

`github_hot_projects` 现有功能稳定，但因长期"打补丁"式开发，编排层（`agent.py` / `agent_tools.py` / `parsing/` / `scheduled_update.py` 的编排部分）逻辑复杂冗余、职责不清。核心症结：

- **强逻辑约束工具顺序**：榜单流程的正确性靠三套互相缠绕的机制硬约束——
  1. 工具白名单 `CONSTRAINED_TOOLS_BY_INTENT`
  2. `_execute_tool` 内"没有搜索结果，请先调用 search_by_keywords / 没有候选列表，请先调用 batch_check_growth / 没有排序结果，请先调用 rank_candidates"等前置错误回退
  3. `last_search_repos → last_candidates → last_ranked` 的全局状态手工串链 + 大量硬编码工具名特判（`_merge_request_defaults_into_tool_args`、`_maybe_reset_discovery_state`）
- **同一榜单流程存在两份实现**：LLM ReAct 驱动一份（脆弱），`scheduled_update.py` 的 `DiscoveryPipeline` 确定性一份（清晰）。
- **扩展成本高**：新增一个工具需在 7+ 处散点修改。

### 目标

1. 保证原有功能不变。
2. 工具函数、任务类型可扩展（新增 = 单点注册）。
3. 榜单类型"正确调用工具顺序"，但**不靠强逻辑约束**——通过把依赖顺序内聚到复合工具内部实现。
4. 顶层 ReAct 保留自由度：原子工具可自由搭配。
5. 在不牺牲抓取完整性/稳定性的前提下，做安全的性能改进。

### 范围边界（已确认）

- **只重写编排层**。下层稳定模块原样复用：`common/*`（config / github_api / async_token_pool / llm / db / exceptions）、`tasks/*`（Task 基类 + 子类 + dispatcher + checkpoint）、`ranking.py`、`report.py`、`growth_estimator.py`、`github_trending.py`、`web/*`。
- **不换语言**。该负载是 I/O 密集型，瓶颈在 GitHub API 速率限制与网络 I/O，换 Go 不会提速；提速根本杠杆是"加 token"与"减请求"。

---

## 2. 整体分层架构

```
入口层（薄）        api_server.py / agent_cli.py / scheduled_update.py
                          │                              │
编排层（重写）       agent.py(精简 ReAct) ── tool_registry ── agent_tools
                          │                                   │
                          │            ┌──────────────────────┤
                          │            │ 复合榜单工具          │ 原子工具
                          ▼            ▼                       ▼
                   ranking_pipeline(榜单唯一实现 + 分阶段缓存)  │
                          │                                   │
基础工具层          search / scan / batch_growth / rank / report / repo_growth /
（capabilities）     describe / db_info / trending   ← 纯函数
                          │
下层（复用）        common/* · tasks/* · ranking.py · report.py ·
                   growth_estimator.py · github_trending.py · web/*
```

- `scheduled_update` 与 Agent 的复合榜单工具**共用 `ranking_pipeline`**，消灭两份实现冗余。

---

## 3. 两层工具模型

### 3.1 基础工具层（不直接暴露给 LLM，彼此无强依赖）

由现有 `agent_tools.py` 的函数清理后保留，作为榜单流水线的内部步骤：
`search_by_keywords`、`scan_star_range`、`batch_check_growth`、`rank_candidates`、`generate_report`。

要求：保持纯函数形态（输入 `token_mgr`/`db`/参数，输出数据结构），不持有全局状态。

### 3.2 Agent 工具层（暴露给 LLM）

| 工具 | 类型 | 内部行为 | 昂贵(需确认) |
|------|------|----------|:-----------:|
| `comprehensive_ranking` 综合榜 | 复合 | collect(search+scan+trending)→growth→rank→report | 是 |
| `hot_new_ranking` 新项目榜 | 复合 | 同上，带 days_since_created 创建时间窗口 | 是 |
| `keyword_ranking` 关键词榜 | 复合 | search→growth→rank | 是 |
| `repo_growth` 单仓库增长 | 原子 | 精确查→查不到则 search 兜底返候选 | 否 |
| `describe_project` 项目介绍 | 原子 | 同上模糊消歧 | 否 |
| `get_db_info` DB 查询 | 原子 | 直接查 | 否 |
| `fetch_trending` Trending | 原子 | 直接抓 | 否 |

**收益**：依赖顺序内聚到复合工具内部（局部变量串链）。agent.py 删除：`CONSTRAINED_TOOLS_BY_INTENT`、`_select_tools_for_llm` 工具裁剪、`_execute_tool` 全部"请先调用 X"前置校验、`_check_suggested_collection_tools`、`_maybe_reset_discovery_state`，以及 `last_search_repos/last_candidates/last_ranked` 等全局状态串链。

---

## 4. 工具注册表（可扩展性）

```python
@dataclass
class ToolSpec:
    name: str
    schema: dict           # LLM function-calling schema
    param_schema: dict     # 校验规则（复用现有 validate_tool_args）
    handler: Callable       # 执行函数
    expensive: bool = False # 是否需要执行前确认
```

- 所有工具集中注册到一个 registry。
- `agent._execute_tool` 退化为"查注册表 → 校验参数 → 调 handler"的单一分发，**无任何工具名特判**。
- 新增工具/任务类型 = **只加一条注册项**。

---

## 5. 榜单流水线 + 分阶段缓存

把散落在 `AgentState` 的 9 个缓存字段（`last_search_repos`、`last_candidates`、`last_ranked`、`last_mode`、`last_growth_calc_days`、`last_min_star`、`last_candidate_days_since_created`、`seen_repos`、`discovery_turn_id`）收敛成**一个 `RankingCache`**，按"阶段 + 参数签名"缓存。

```
collect       依赖: categories, min_star, max_star, days_since_created, sources
growth_calc   依赖: collect 输出 + growth_calc_days, days_since_created   ← 昂贵(API)
threshold     依赖: growth_calc 输出 + growth_threshold                   ← 廉价(纯过滤)
rank          依赖: threshold 输出 + mode, top_n, days_since_created       ← 廉价
report        依赖: rank 输出 + 展示参数(growth_calc_days/growth_threshold/min_star...)
```

关键点：**把"计算增长值(昂贵 API)"与"按阈值过滤(廉价)"拆成两个阶段**，这样改阈值不会重算增长值。

**重跑策略**：从上往下逐阶段比对参数签名；签名未变的阶段复用缓存，从第一个变化的阶段往下重算。

- "增长阈值降到 500 再看看" → 仅 threshold 过滤 + rank 重算，**不重算增长值、不重新 search**。
- "换个增长窗口(growth_calc_days)" → 从 growth_calc 重算（需 API），不重新 collect。
- "换个关键词" → 从 collect 重跑。
- 取代 `discovery_turn_id` 手工重置 hack，行为可预测、可单测。

`RankingCache` 由会话持有，复合榜单工具读写它；`scheduled_update` 用一次性空缓存 + `force_refresh` 调同一流水线。

---

## 6. 轻量确认 + 单仓库模糊消歧

### 6.1 确认（已确认：prompt + 幂等守卫）

- 系统 prompt 约定："调用昂贵榜单工具前先回显参数并等用户『开始』"。
- 复合榜单工具入口加 **~10 行幂等确认守卫**：首次调用返回"请确认参数"并记录参数签名；用户确认后、同签名再调时才真正执行。这不是状态机，是工具内一个布尔/签名判断。
- 删除旧的 `awaiting_confirmation`/`pending_request`/`_is_confirmation_ack`/`_maybe_handle_confirmation_gate` 整套两阶段状态机。

> 注：若用户后续选择"纯 prompt 不加守卫"，则去掉守卫，仅靠 prompt 约束。

### 6.2 单仓库模糊消歧

`repo_growth` / `describe_project`：
1. 先按精确 `owner/repo` 查（现有 `fetch_repo_info`）。
2. 查不到 / 只给了 name / 拼错 → 自动 `search_github_repos`（现有）用名字搜，返回 Top N 相似候选。
3. 工具返回候选列表 → LLM 在 ReAct 循环里问用户"没找到 X，你是不是指：a/x、b/x、c/x？" → 用户选 → LLM 用全名重查。

无需新状态机，天然契合 ReAct。

---

## 7. 安全的性能改进（完整性优先）

| # | 改进 | 完整性风险 | 说明 |
|---|------|:---------:|------|
| 1 | `batch_check_growth` 的 `created_at` 补全从串行 sleep 循环改为异步调度器并行 | 无 | 纯提速，结果一致 |
| 2 | 更激进复用 DB 快照的 `created_at`/静态元数据，跳过重复 API | 无 | DB 对静态字段是权威源 |
| 3 | 关键词跨类别去重后再搜，减少重复请求 | 无 | 结果并集不变 |
| 4 | 增长二分查询改 GraphQL 批量采样 | 有 | 默认**不动**，仅记录为待评估项 |

**必须保留**：页级失败补偿（`failed_pages` 重试）——抓取完整性的保命机制。

提速根本杠杆仍是"加 token"。4 小时全量跑可接受；本节为顺手的零风险优化。

---

## 8. 会话状态（精简后）

新 `AgentState` 仅保留：
- `token_mgr`、`db`、`conversation`、`conversation_summary`（对话压缩沿用）
- `ranking_cache: RankingCache`（取代 9 个散乱缓存字段）
- `active_repo`（单仓库追问上下文）
- 昂贵工具的 `pending_confirmation_signature`（确认守卫用）

删除：路由门控相关全部字段、`current_turn_*` 执行契约字段、`last_*` 流水线缓存字段、`recent_verified_claims`（fact_check 取证改由 ReAct 自然完成）、`seen_repos`/`discovery_turn_id`。

---

## 9. 验证策略（已确认：TDD 重写编排层）

- 下层测试保持绿不动：`test_tasks`、`test_ranking`、`test_report`、`test_growth`、`test_common`、`test_async_token_pool`、`test_async_worker_pool`、`test_trending`。
- 编排层 TDD：先为以下写测试再实现——
  - `RankingCache` 分阶段复用与失效
  - 工具注册表分发
  - 复合工具内部顺序正确
  - 单仓库模糊消歧
  - 昂贵工具幂等确认守卫
- `test_agent.py` / `test_pipeline.py` / `test_agent_tools.py` / `test_api_server.py` 按新接口替换。

---

## 9b. ReAct 逻辑调整

1. **取消独立路由 LLM，合并为单 ReAct**（最大改善）：原设计每轮先跑路由 LLM（`CONFIRMATION_PROMPT`）做意图分类，再跑执行 LLM 做 ReAct；新设计由单个 ReAct LLM 同时完成意图理解 + 选工具。更简单、更省 token、更低延迟，澄清靠对话自然完成。
2. **保留**：工具结果截断 `_serialize_result`、对话压缩、坏参数重试一次、单轮最大调用次数护栏。
3. 昂贵工具：prompt + 幂等确认守卫（见 6.1）。
4. 单仓库：ReAct 内模糊消歧（见 6.2）。
5. **禁止**给工具结果塞"下一步提示"来引导顺序——顺序应内聚到复合工具，不得变相把强逻辑加回来。

### 刻意简化点（行为轻微变化，需知情）

- **双 LLM → 单 LLM**（见上）。
- **fact_check 硬契约移除**：原 `must_call_tool_before_reply` + `recent_verified_claims` 强制"先取证再回答"；新设计靠 system prompt 约束（"事实数据必须调工具核查"），略微放宽，换来删除整套契约机制。

---

## 9c. 多平台横向扩展（只"留好形状"）

引入数据源 Provider 边界，但本期只实现 `GitHubProvider`，不造第二个平台（YAGNI）。

- **`Provider` 接口**：声明 `search / scan / repo_info / growth / trending` 等能力方法。
- **归一化 `Repo` 模型**：取代当前裸 dict（`full_name`/`star`/`created_at`/...），作为跨层数据契约。
- **边界约束**：复合工具、ranking_pipeline、agent 层只依赖 `Provider` 接口与 `Repo` 模型，**GitHub 细节（REST/GraphQL/stargazers/trending HTML/token 限流）不得泄漏到这些层**。现有 `common/github_api`、`growth_estimator`、`github_trending`、`async_token_pool` 收敛为 `GitHubProvider` 内部实现。
- 未来加 GitLab 等：新增一个 Provider 实现即可，编排层零改动。

边界形状是低成本的；不做 token 池/限流的跨平台抽象（那属于过度设计，留到真有第二平台时）。

---

## 9d. 新项目落地（不改原项目）

新建自包含项目 `hot_projects`（原名去掉 github），**全部新代码写入新项目，绝不改动原 `github_hot_projects`**。下层稳定模块**原样复制**进新项目（不改算法逻辑），只重写编排层。

### 目标结构

```
hot_projects/
├── __main__.py / api_server.py / agent_cli.py / scheduled_update.py   # 入口（复用，改 import）
├── config.py                       # 配置（去 github 强绑定命名）
├── agent/        agent.py(精简 ReAct) · state.py(AgentState+RankingCache) · prompts.py
├── tools/        registry.py · ranking_tools.py(复合) · atomic_tools.py(原子+模糊消歧) · schemas.py
├── pipeline/     ranking_pipeline.py(唯一流水线) · cache.py(RankingCache 阶段签名)
├── capabilities/ 基础工具层（原 agent_tools 9 函数清理后，纯函数）
├── providers/    base.py(Provider 接口+归一化 Repo) · github/(api·token_pool·trending·growth_estimator)
├── infra/        db.py · llm.py · exceptions.py · concurrency/(dispatcher·task_base·tasks·checkpoint)
├── ranking.py  report.py · web/ · requirements.txt · tests/
```

### 复制时的整理原则（只动外壳，不改算法）

1. **命名去 GitHub 强绑定**：包名改 `hot_projects`；GitHub 专属实现收进 `providers/github/`，编排层只见 `Provider` + `Repo`。
2. **位置归类**：并发框架/db/llm 归 `infra`；下层 API 归 provider。
3. **删死代码/冗余参数**：清理 tasks 与 github_api 中成片注释的 "B 模式请求级 token 借还" 死代码及 `token_idx=None` 分支（当前仅用 A 模式）；清理仅服务旧路由阶段的参数。
4. **不动**：增长估算算法、二分/采样、token 轮换、报告渲染、DB 读写逻辑。

---

## 9e. LLM 接入层：A/B 双方案 + 逐调用回退 — 已实测确认

### 设计

- **两个自包含方案 A / B**，每个方案各带**一对模型**：ReAct 主对话模型 + 描述/总结小模型，外加该平台的鉴权方式与参数白名单。
- **方案 A（主力）= Azure OpenAI**；**方案 B（备选）= SiliconFlow**。
- **逐调用回退（per-call）**：每次 LLM 调用先用 A，A 失败（自身重试耗尽 / 连接错误 / HTTP 错误）则该次改用 B；下次调用仍优先 A（A 恢复后自动用回 A，不粘滞）。
  - 主对话调用：A.MODEL 失败 → B.MODEL；小模型调用：A.LITE 失败 → B.LITE。
- **按后端参数适配**：客户端按 `*_BACKEND` 套用对应鉴权头与参数白名单。
  - `azure`：`api-key` 头；用 `max_completion_tokens`；**不发** `enable_thinking`/`thinking_budget`；温度省略（默认）。
  - `openai`(SiliconFlow)：`Authorization: Bearer` 头；用 `max_tokens`/`temperature`/`enable_thinking`/`thinking_budget`。

### 配置（环境变量，注释标平台；key 绝不入代码/git）

```bash
# ===== 方案 A（主力）: Azure OpenAI =====
LLM_A_BACKEND=azure
LLM_A_URL=https://ceshi-001.openai.azure.com/openai/v1/chat/completions?api-version=preview
LLM_A_KEY=<azure-key>
LLM_A_MODEL=gpt-5.4              # ReAct 主对话模型（function calling 实测可用）
LLM_A_LITE_MODEL=gpt-5.4-mini   # 描述/总结小模型

# ===== 方案 B（备选）: SiliconFlow =====
LLM_B_BACKEND=openai
LLM_B_URL=https://api.siliconflow.cn/v1/chat/completions
LLM_B_KEY=<siliconflow-key>
LLM_B_MODEL=Pro/zai-org/GLM-5            # ReAct 主对话模型
LLM_B_LITE_MODEL=Qwen/Qwen3.5-35B-A3B   # 描述/总结小模型
```

### 实测结论（已验证）

- Azure 端点联通；`gpt-5.4` 基础对话 + **function calling** 实测可用；`gpt-5.4-mini` 对话可用。
- Azure 参数坑：`max_tokens` 被拒（须 `max_completion_tokens`）；`enable_thinking` 被拒。
- 本资源实际部署仅 9 个（gpt-4o / gpt-4o-mini / gpt-5.1-chat / gpt-5.3-chat / gpt-5-mini / gpt-5.4 / gpt-5.4-mini / gpt-5.4-mini-2 / text-embedding-3-small），`gpt-5.5` 未部署。

### 安全提醒

用户已在对话中明文暴露 Azure key，落地前应**轮换**。

---

## 10. 功能等价对照（保证"功能不变"）

| 现有能力 | 新实现承接点 |
|----------|-------------|
| 综合榜 / 新项目榜 / 关键词榜 | 三个复合工具 → ranking_pipeline |
| 单仓库增长 / 介绍 / DB 查询 / Trending | 四个原子工具 |
| 执行前参数确认 | prompt + 幂等守卫 |
| 增量调参追问 | RankingCache 分阶段复用 |
| 定时批处理 | scheduled_update → 同一 ranking_pipeline |
| 报告生成 / Web 渲染 / API / WS / 会话 TTL / 安全中间件 | 入口层与 web/ 不变 |
| 对话历史压缩 | 沿用 |
| 页级失败补偿 / token 轮换 / 增长估算 | 下层不变 |

---

## 10b. 必须小心搬运的隐藏行为（否则会丢功能）

1. **综合榜未指定窗口时自动采用 DB 年龄窗口**：现分散在 `agent._resolve_pending_request` + `task_help._submit_growth_tasks`，新设计须把参数解析统一搬进复合工具 / pipeline。
2. **持久化策略按调用方区分**：Agent 路径 `save_db_desc_only`；定时 `force_refresh` 全量 `save_db`。
3. **`prefiltered_days_since_created` 透传**：决定 hot_new 能否跳过排名阶段二次过滤，须从 growth 阶段透传到 rank 阶段。
4. **三路增长估算**（DB 差值 / REST 二分 / GraphQL 采样）与**断点续传**：均在下层，复合工具调用时参数须正确传入（`force_refresh`/`window_specified`/`days_since_created`/`growth_calc_days`）。

---

## 10c. 附录：下层基础计算缺口清单（本次不修，单独立项）

经全量审计，以下问题位于下层模块（`growth_estimator` / `github_api` / `tasks` / `config`），本次编排层重构**不处理**，留待后续单独修复（各自补测试）：

- **B2 最低星密集带截断**：`auto_split_star_range` 到 `min_span=50` 即停止细分，密集低星段（如 1200–1250）若 >1000 条，scan 10 页只取 1000，其余丢弃。
- **B4 created_at 缺失被丢弃**：hot_new 二次过滤时空 created_at 直接 skip，即使 API 补全失败也丢弃该新项目。
- **B5 超大仓库采样未决**：REST 取不到末页且 GraphQL 采样不足 → `unresolved`，该仓库当轮无增长值、不进候选。
- **C1 二分法空页低估**：空页/解析失败时 `lo=mid+1` 把空页当作窗口外，可能跳过真实边界导致增长低估。
- **C2 单仓库 <800 star 返回 0**：`estimate_star_growth_binary` 对 `total_stars < STAR_GROWTH_THRESHOLD` 直接返回 0，单仓库直查小而快增长的项目会误报 0。

产品阈值类（B1 `MIN_STAR=1200`、B3 `MAX_STAR=45000`）按需调参即可，非代码缺陷。

整体结论：基础计算逻辑**总体可靠**，`scan_star_range` 全星段扫描是完整性主干；上述为边缘缺口，不影响主流程正确性。

---

## 11. 待实现计划细化的开放点

- `RankingCache` 参数签名的精确字段划分（哪些参数归哪个阶段）需在计划阶段逐工具确认。
- 确认守卫的参数签名与"开始"识别如何与 LLM 自然语言协作（避免误判）。
- 性能改进 1/2/3 的具体落点与基准对比方式。
