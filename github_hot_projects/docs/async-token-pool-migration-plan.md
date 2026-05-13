# GitHub 热门项目发现系统 — 异步改造任务拆分清单

## [x] 拆分迁移方案为可执行任务清单

**完成内容**
- 已将原迁移设计文档重构为任务清单。
- 已按“先 Token 池、再 API 异步化、再调度器替换、再入口切换”的顺序排布。

## [x] 建立改造基线与验收指标

**目标**
- 固化改造前基线，避免优化后无对照。

**验收**
- 记录总耗时、成功请求数、限流等待时长、任务回队次数、失败率。

**涉及文件**
- docs/async-migration-baseline-and-benchmark.md

## [x] 新增 AsyncTokenPool 基础结构

**目标**
- 在 common 新建 token 状态中心，统一管理借出、归还、限流恢复、失效剔除。

**涉及文件**
- common/async_token_pool.py

## [x] 实现 Token 借用与归还接口

**目标**
- 提供 acquire/release；外部只拿到可用 token_idx，不直接操作 token 状态。

**验收**
- 仅“未借出 + 未失效 + 已过恢复时间”的 token 可被分配。

## [x] 实现限流与失效状态更新

**目标**
- 命中限流时 mark_rate_limited(available_at)；命中 401 时 mark_invalid。

**验收**
- 限流 token 可恢复后再参与分配；失效 token 永不再分配。

## [x] 实现无可用 Token 的最早恢复等待机制

**目标**
- 当无可用 token 时按最早恢复时间等待并可被提前唤醒。

**验收**
- 不忙轮询；支持按恢复时间自动唤醒重试。

## [x] 为 AsyncTokenPool 补充单元测试

**目标**
- 覆盖借出/归还/限流/失效/等待唤醒主路径。

**涉及文件**
- tests/test_async_token_pool.py

## [x] 引入异步 HTTP 依赖

**目标**
- 新增 httpx，作为 GitHub API 异步请求客户端。

**涉及文件**
- requirements.txt

## [x] 在 github_api 中增加异步请求实现（与同步并存）

**目标**
- 新增 async 版本 API，先不删除 requests 同步实现。

**涉及文件**
- common/github_api.py

## [x] 保持异常语义不变（401/403/429）

**目标**
- 异步链路继续抛 TokenInvalidError / RateLimitError，兼容现有任务语义。

**涉及文件**
- common/github_api.py
- common/exceptions.py

## [x] 增加 Task 异步执行入口（兼容迁移期）

**目标**
- Task 基类支持 async execute 入口；旧同步任务可桥接执行。

**涉及文件**
- tasks/task_base.py

## [x] 新增 AsyncTaskDispatcher

**目标**
- 以 PriorityQueue + 协程消费者替换线程绑定 token 的执行模型。

**涉及文件**
- tasks/async_worker_pool.py

## [x] 在调度层实现 needs_github_token 分流

**目标**
- 需要 token 的任务先 acquire；不需要 token 的任务直执行。

**验收**
- 无 token 任务不受 token 池影响。

## [x] 迁移 KeywordSearchTask 到异步链路

**目标**
- 保留现有业务行为与分页语义，逐步从桥接执行过渡到原生 async。

**涉及文件**
- tasks/task.py

## [x] 迁移 ScanSegmentTask 到异步链路

**目标**
- 保留失败页补偿逻辑，逐步从桥接执行过渡到原生 async。

**涉及文件**
- tasks/task.py

## [x] 迁移 CalcGrowthTask 到异步链路

**目标**
- 保留 checkpoint、candidate_map、unresolved 语义，逐步改为异步链路执行。

**涉及文件**
- tasks/task.py
- tasks/task_help.py

## [x] 切换 tool_search_by_keywords 到 AsyncTaskDispatcher

**涉及文件**
- agent_tools.py

## [x] 切换 tool_scan_star_range 到 AsyncTaskDispatcher

**涉及文件**
- agent_tools.py

## [x] 切换 tool_batch_check_growth 到 AsyncTaskDispatcher

**涉及文件**
- agent_tools.py

## [x] 导出并接入新调度器

**目标**
- 在 tasks 包导出 AsyncTaskDispatcher，并补齐引用位置。

**涉及文件**
- tasks/__init__.py

## [x] 增补异步调度集成测试

**目标**
- 覆盖任务回队、换 token 重试、结果一致性、基础回调行为。

**涉及文件**
- tests/test_async_worker_pool.py
- tests/test_tasks.py
- tests/test_agent_tools.py

## [x] 直接切换为协程主路径（不保留旧线程池代码）

**目标**
- 清理 thread 模式主路径调用并移除旧实现。

**涉及文件**
- agent_tools.py
- tasks/__init__.py
- tasks/worker_pool.py

## [x] 队列模型选型决策（本轮确认）

**结论**
- 默认并发度采用: min(max(token_count * 4, 8), 64)。
- 队列采用 PriorityQueue，统一处理即时任务与延迟重试任务。
- 4 个 Token 默认协程数 16；5 个 Token 默认协程数 20。

## [x] 实现 PriorityQueue 延迟重试调度（一步到位）

**目标**
- 以 next_run_at 为优先级，统一调度即时任务与重试任务。
- 消费者协程不执行长时间等待；到期任务由队列时钟驱动出队。

**验收**
- 高限流场景下消费者可持续处理其他可执行任务。
- 重试任务不会饿死新任务；新任务也不会无限压后重试任务。

## [x] 增加任务幂等键与重复执行保护

**目标**
- 为增长计算、候选写入、checkpoint 更新增加幂等保护，避免回队造成重复副作用。

## [x] 增加调度公平策略（防饥饿）

**目标**
- 针对任务类型或来源设置公平消费策略，避免单类任务长期占满队列。

## [x] 增加 Token 健康度与降级机制

**目标**
- 对短时间连续异常 token 进行降权或短暂隔离，恢复后再回流。

## [x] 增加异步链路可观测性指标

**目标**
- 统一上报 token_wait_seconds、requeue_count、retry_histogram、task_latency。

## [x] 固化发布回退预案（非双栈）

**目标**
- 不保留旧代码前提下，明确版本级回退路径（tag/release 回退），并定义触发条件。

**验收**
- 任一核心指标恶化超过阈值可在版本层面快速回退。

**涉及文件**
- docs/async-migration-rollback-plan.md

## [x] 进行压测与对比验收

**目标**
- 在高限流场景对比线程池与异步池吞吐和等待效率。

**验收**
- 吞吐提升或等待降低，且结果一致性不回归。

**涉及文件**
- tools/benchmark_async_dispatcher.py
- docs/async-migration-baseline-and-benchmark.md

## [x] 移除旧线程池实现与所有引用（立即执行）

**目标**
- 删除旧线程池实现及其引用，不保留兼容分支。

**涉及文件**
- tasks/worker_pool.py
- 相关调用入口与文档

## [x] 合并 Token 管理职责并清理重复文件

**目标**
- 将请求头构建能力并入 token pool，避免 TokenManager/TokenPool 双概念并存。

**结果**
- 使用 GitHubTokenPool 统一承载 token 生命周期与请求头构建能力。
- common/token_manager.py 已删除，链路仅保留统一实现。
