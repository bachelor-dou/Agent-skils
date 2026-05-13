# GitHub 热门项目发现系统：请求级 Token 借还改造方案

## 1. 背景

当前异步链路已经完成了线程池到协程调度器的迁移，但 Token 的占用粒度仍然是“任务级”。

也就是说：

1. 调度器先为任务获取一个 Token。
2. 整个任务执行期间一直持有该 Token。
3. 任务完成、失败、限流或失效时才释放或更新 Token 状态。

这种实现解决了线程阻塞和统一 Token 状态管理问题，但还没有完全发挥协程模型的优势。

## 2. 当前现象

### 2.1 Token 利用率受任务边界限制

当前模式下，Token 会贯穿整个任务生命周期，而不是只覆盖真实 GitHub 请求。

典型表现：

1. 任务内部存在“请求 -> 本地计算 -> 再请求”的阶段时，Token 在本地计算阶段也被占用。
2. 增长估算、分页扫描、二分探测这类多阶段任务，Token 空转时间偏长。
3. Token 并发上限更接近“同时运行的任务数”，而不是“同时进行的请求数”。

### 2.2 日志容易误导排障

以 scheduled-2026-05-10.log 为例，出现了这类现象：

1. 先看到某个关键词开始搜索，例如 ai cli。
2. 后续又看到同一个 query 的“异步搜索请求异常”。
3. 日志中的 worker 字段实际打印的是 token_idx，容易误以为多个 worker 复用了同一个执行上下文。
4. httpx.RequestError 在部分情况下字符串为空，导致日志只看到 error=，无法直接判断是超时、连接失败还是其他网络错误。

### 2.3 “开始搜索”和“后续报错”不是重复执行

这个现象的真实原因不是“同一个关键词被多个协程重复执行”，而是：

1. KeywordSearchTask 会按页循环请求 GitHub Search API。
2. 启动日志只打印一次，表示该关键词任务开始执行。
3. 后续 page=2 或 page=3 的请求如果超时或网络异常，会进入单页重试日志。
4. 因此它是“同一任务内的翻页请求失败”，不是“任务被重复调度”。

## 3. 当前实现的主要问题

### 3.1 调度层感知了 Token 生命周期

当前调度器负责 acquire/release Token，并将 token_idx 传入 Task。

这会导致：

1. 调度层同时承担“协程调度”和“资源借还”两种职责。
2. Task 被迫感知 Token 上下文。
3. 后续如果改成请求级借还，改动面会比较大。

### 3.2 API 层不是 Token 生命周期的唯一入口

虽然 Token 状态集中在池中，但 acquire/release 仍然由调度层控制，GitHub API 层只负责发请求和抛异常。

这会导致：

1. 资源管理和请求执行没有完全收口。
2. 每次想调整 Token 策略时，需要同时改调度器和 API 调用链。
3. 很难做到“一次 HTTP 请求对应一次 Token 借还”。

## 4. 改造目标

本轮改造的目标不是拆任务，而是改变 Token 占用粒度。

目标定义如下：

1. 保留现有任务边界，不拆分增长估算、分页扫描、关键词搜索等业务任务。
2. 调度器只负责协程执行、回队、重试和结果回调。
3. Token 生命周期完全下沉到 GitHub API 层统一管理。
4. 每一次真实 GitHub HTTP 请求前临时借用一个 Token。
5. 请求结束后立刻释放 Token；如果命中限流或失效，则先写回状态再释放或剔除。
6. 上层工具和任务对象不感知 Token 状态细节。

## 5. 目标执行模型

改造后的执行链应当是：

1. Dispatcher 从队列取出 Task。
2. Task 只执行自己的业务流程。
3. Task 内部调用 GitHub API helper 时，helper 临时向 TokenPool 借用 Token。
4. API helper 发起单次 REST/GraphQL 请求。
5. 请求完成后立刻 release Token。
6. 如果限流，则将 reset_time 写回 TokenPool。
7. 如果 Token 失效，则在 TokenPool 中标记 invalid 并移除。
8. Task 根据请求结果继续业务逻辑，下一次请求时再重新借用 Token。

最终资源单位从：

1. 一个任务占一个 Token

变成：

1. 一个请求占一个 Token

## 6. 修改方案

### 6.1 调度层去 Token 化

涉及文件：

1. tasks/async_worker_pool.py
2. tasks/task_base.py

改造方向：

1. Dispatcher 不再在 worker_loop 中调用 acquire/release。
2. Dispatcher 不再将 token_idx 作为任务执行上下文下发。
3. Dispatcher 仅负责执行 task.execute_async()。
4. 任务回队逻辑保留，但 Token 状态更新不再依赖 Dispatcher。

结果：

1. 调度层只保留“排队、并发、回调、重试”职责。
2. Token 生命周期彻底脱离 worker 生命周期。

### 6.2 GitHub API 层成为 Token 生命周期唯一入口

涉及文件：

1. common/github_api.py
2. common/async_token_pool.py

改造方向：

1. 在 GitHub API 层增加统一的“请求级 Token 执行器”。
2. 每次发 REST 或 GraphQL 请求前，从池中 acquire 可用 Token。
3. 根据该 Token 生成 headers。
4. 请求成功后立刻 release。
5. 限流时调用 mark_rate_limited(reset_time)。
6. 401 或明确的 Bad credentials 调用 mark_invalid()。
7. 普通网络异常则 release 后抛出，由上层决定是否重试任务。

结果：

1. 所有 Token 借还和状态回写都集中在 API 层。
2. Task 和 Tool 不再关心 Token 细节。

### 6.3 Task 改成纯业务编排者

涉及文件：

1. tasks/task.py
2. growth_estimator.py

改造方向：

1. KeywordSearchTask 只负责关键词、分页和结果聚合。
2. ScanSegmentTask 只负责区间分页扫描和失败页记录。
3. CalcGrowthTask 只负责二分、采样、回补等算法流程。
4. 真实请求全部交给 async GitHub API helper；Task 不再持有 token_idx。

结果：

1. Task 边界保持不变。
2. Token 利用率不再被 Task 生命周期绑住。

### 6.4 Growth 链路保持“单任务算法闭环”

增长估算是高风险链路，必须明确约束：

1. 不拆分为多个队列任务。
2. 不改变现有二分法、采样外推、checkpoint 的业务语义。
3. 只把每次请求时的 Token 获取下沉到 API 层。

这样可以同时满足：

1. 结果逻辑不变。
2. 协程资源利用率提升。
3. Token 不在本地计算阶段被空占。

### 6.5 日志与观测同步升级

涉及文件：

1. common/github_api.py
2. common/async_token_pool.py
3. tasks/task.py

改造方向：

1. 不再把 token_idx 打成 worker 标识。
2. 网络异常日志必须包含异常类型名。
3. 高频等待状态以 metrics 为主，日志只保留低频摘要。
4. 增加请求级借还指标，例如 borrow_total、release_total、current_waiter_count、average_hold_ms。

## 7. 建议实施顺序

### 阶段 1：建立请求级借还能力

1. 在 common/github_api.py 中建立统一请求包装层。
2. 在 async_token_pool.py 中补齐请求级观测指标。
3. 保留当前 dispatcher 代码，先不删除旧路径。

### 阶段 2：迁移低风险任务

1. 先迁移 KeywordSearchTask。
2. 再迁移 ScanSegmentTask。
3. 验证搜索、扫描、分页补偿链路结果一致。

### 阶段 3：迁移高收益链路

1. 迁移 CalcGrowthTask。
2. 迁移 growth_estimator.py 中的异步请求路径。
3. 验证增长值、checkpoint、unresolved 结果与旧链路一致。

### 阶段 4：收口架构

1. 删除 dispatcher 中的任务级 acquire/release。
2. 删除 needs_github_token 这类任务级 Token 分流逻辑。
3. 确认系统形成“调度层无 Token 感知、API 层统一借还”的最终结构。

## 8. 验收标准

### 8.1 行为正确性

1. 搜索、扫描、增长计算结果不回归。
2. 分页补偿、checkpoint、candidate_map 语义不变。
3. 限流和失效 Token 处理逻辑保持正确。

### 8.2 性能与资源利用

1. Token 空转时间下降。
2. 相同 Token 数下，总请求吞吐提升或等待时间下降。
3. 限流等待不会阻塞其他可执行协程。

### 8.3 可观测性

1. 日志能明确区分 task、page、token、异常类型。
2. 不再出现误导性的 worker=token 日志。
3. 对于异步请求异常，日志必须能直接看到异常类别。

## 9. 当前立即可执行的小修正

在进入完整请求级 Token 借还改造前，已经可以先做两类低风险修正：

1. 修正日志中的 worker/token 混用，避免排障误判。
2. 异步请求异常日志稳定输出异常类型，避免出现 error= 空白。

这两项修正不改变业务逻辑，但能明显提升线上排障效率。

## 10. 结论

当前系统已经完成“协程调度化”，但尚未完成“请求级资源化”。

下一阶段的核心不是继续增加协程数量，而是：

1. 保留任务边界。
2. 把 Token 占用粒度从任务级降到请求级。
3. 让调度层只做调度，让 API 层统一管理 Token 生命周期。

只有完成这一步，协程模型的资源复用优势才能真正释放出来。
