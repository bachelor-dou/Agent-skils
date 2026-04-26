# GitHub 热门项目发现系统重构设计

## 1. 重构目标

本次重构聚焦三个问题：

1. 输入解析不稳定，语义规则散落在提示词、正则、参数归一化和工具执行分支中。
2. 工具选择过于自由，当前更接近 ReAct 临场调用，缺少稳定的工作流边界。
3. 执行层承担了部分语义纠偏职责，导致职责交叉，后续难以维护和测试。

### 1.1 架构模式变更：从 ReAct 到结构化管道

当前系统是 **ReAct 循环**（Thought → Action → Observation → 循环），LLM 在每一轮自由决定下一步调什么工具。重构后采用 **结构化管道（Structured Pipeline）** 模式：

| 维度 | ReAct（当前） | 结构化管道（重构后） |
|------|--------------|-------------------|
| LLM 调用频率 | 每步都由 LLM 决定下一步，循环调用 | LLM 只调用 1-2 次（理解意图 + 结果总结） |
| 工具选择 | LLM 运行时自由选择任意 tool | 程序按 skeleton 预定义工具序列 |
| 执行控制 | LLM 观察结果后决定是否继续 | 执行器按计划顺序执行，无 LLM 参与 |
| 异常处理 | LLM 看到错误后自行决定策略 | 程序预定义失败策略（停止/跳过/降级） |

重构后 agent.py 的主方法变成线性调用链，不再有 ReAct 循环：

```
request = build_request(user_message)
parsed  = interpreter.parse(request)           # LLM 调用 1 次
resolved = resolver.resolve(parsed)            # 纯程序
confirm = confirm_rules.check(resolved)        # 纯程序
if confirm: return confirmation_response
plan    = planner.plan(resolved)               # 纯程序（骨架匹配）
validator.validate(plan)                       # 纯程序
result  = executor.execute(plan, resolved)     # 顺序执行工具
summary = summarizer.summarize(result)         # 可选 LLM 调用
```

### 1.2 目标架构

用户输入
-> 输入理解层（LLM 结构化解析，唯一的 LLM 调用点）
-> 程序约束与确认层（默认值、校验、歧义拦截）
-> 工作流规划层（骨架匹配，纯程序）
-> 计划校验层（合法性检查，纯程序）
-> 顺序执行层（按计划逐步执行工具）
-> 结果总结层（格式化输出，可选 LLM 润色）

核心原则：

- LLM 负责理解用户语义，不直接自由决定真实工具调用。
- 程序负责默认值、约束、确认、计划校验和执行边界。
- 执行器只执行结构化计划，不再重新理解自然语言。

---

## 2. 总体设计逻辑

### 2.1 新的数据流

建议将交互请求统一转换为以下链路：

1. 接收用户原始输入和会话上下文。
2. 调用 LLM 产出结构化 ParsedIntent。
3. 程序根据 schema 做类型校验、默认值回填、冲突处理和歧义识别。
4. 如存在关键歧义，返回 needs_confirmation 给前端，由用户补充。
5. 生成 ResolvedIntent，作为后续规划与执行的唯一权威输入。
6. 根据 ResolvedIntent 选择 workflow skeleton，并产出 ExecutionPlan。
7. 对 ExecutionPlan 做合法性和依赖校验。
8. 执行器按顺序执行步骤，产出 ExecutionResult。
9. 总结层根据结果生成对用户可见的摘要或报告。

### 2.2 重构后的职责边界

| 层 | 负责什么 | 不负责什么 |
|---|---|---|
| 输入理解层 | 理解用户意图、识别字段、标出歧义 | 设默认值、调用工具 |
| 程序约束层 | 默认值回填、范围校验、冲突处理、确认判断 | 理解开放语义 |
| 规划层 | 选 workflow skeleton、产出步骤序列 | 真正执行工具 |
| 校验层 | 校验计划完整性、依赖和边界 | 语义脑补 |
| 执行层 | 顺序执行步骤、收集结果、失败停止或降级 | 改写用户意图 |
| 总结层 | 输出结果摘要、报告、说明 | 推断执行逻辑 |

---

## 3. 用户可指定参数

建议将用户层输入收敛到以下 14 个参数。

| 参数 | 含义 | 用户常见表达 | 默认或备注 |
|---|---|---|---|
| workflow_mode | 榜单模式 | 综合榜、新项目榜、只看 Trending | 默认 comprehensive |
| repo | 指定单仓库 | 看一下 owner/repo | 单仓库查询时使用 |
| categories | 主题类别 | AI Agent、数据库、云原生 | 支持多类别 |
| top_n | 返回数量 | 前 10、前 20、前 50 | 按模式取默认值 |
| growth_calc_days | 增长统计窗口 | 近 7 天、近 14 天 | 默认 7 |
| creation_window_days | 新项目创建窗口 | 近 30 天新项目 | hot_new 模式核心参数 |
| growth_threshold | 最低增长阈值 | 增长至少 300 | 默认使用配置值 |
| project_min_star | 候选项目最低总 star | 只看总 star 超过 1000 的项目 | 默认使用配置值 |
| star_min | 搜索或扫描下界 | 扫描 1000 以上 | 默认使用配置值 |
| star_max | 搜索或扫描上界 | 扫描到 30000 为止 | 默认使用配置值 |
| since | Trending 时间范围 | 今日热门、本周热门 | 仅 trending_only 模式生效，默认 weekly |
| force_refresh | 强制跳过缓存 | 强制刷新、重新抓 | 默认 false |
| report_requested | 是否生成报告 | 生成报告、导出报告 | 默认 false |
| output_format | 输出形式 | 简要列表、详细说明、报告 | 枚举：chat_summary / detailed_list / markdown_report，默认 chat_summary |

说明：

- language 和 spoken_language 不再作为用户过滤条件，只保留为仓库信息字段。
- action_type 不建议作为用户输入参数，保留为内部 request_kind 或 intent_kind 更合适。
- include_all_periods、max_pages 等参数应留在内部规划层，不直接暴露给用户。

---

## 4. 核心领域对象

建议在重构后引入以下结构化对象。

### 4.1 UserRequest

表示一次用户输入及其会话上下文。

```python
@dataclass
class UserRequest:
    session_id: str
    message: str
    confirmed_fields: dict[str, object] = field(default_factory=dict)
    previous_intent_id: str | None = None
```

### 4.2 ParsedIntent

表示 LLM 对用户输入的结构化理解结果。

```python
@dataclass
class ParsedIntent:
    intent_type: str                       # "new"=新请求, "patch"=修改上一轮参数
    workflow_mode: str | None              # comprehensive / hot_new / trending_only / repo_inspect / db_query
    fields: dict[str, object]
    explicit_fields: set[str]
    ambiguous_fields: list[str]
    missing_required_fields: list[str]
    confidence: float | None
    rationale: str = ""
```

说明：
- `intent_type` 用于判断是全新请求还是对上一轮 ResolvedIntent 的局部修改（如"改成前50"）。
- `workflow_mode` 是唯一的意图分类字段，直接映射到 skeleton。不再设置独立的 `request_kind`，避免两个字段语义重叠。
- `workflow_mode` 枚举值：`comprehensive`（综合热榜）、`hot_new`（新项目榜）、`trending_only`（仅看 Trending）、`repo_inspect`（单仓库查询）、`db_query`（数据库查询）。

### 4.3 ResolvedIntent

表示程序约束后的正式执行输入。此对象应成为后续规划层和执行层的唯一输入。

```python
@dataclass
class ResolvedIntent:
    workflow_mode: str                   # comprehensive / hot_new / trending_only / repo_inspect / db_query
    repo: str | None
    categories: list[str] | None
    top_n: int | None
    growth_calc_days: int
    creation_window_days: int | None
    growth_threshold: int
    project_min_star: int
    star_min: int
    star_max: int
    since: str | None
    force_refresh: bool
    report_requested: bool
    output_format: str                   # chat_summary / detailed_list / markdown_report
```

说明：
- 不再有 `request_kind` 字段。`workflow_mode` 是唯一的意图标识，直接决定选择哪个 skeleton。
- resolver 处理顺序：先确定 `workflow_mode`（控制字段），再回填依赖 mode 的数值字段（如 `top_n` 的默认值）。

### 4.4 ConfirmationRequest

当请求存在关键歧义时，返回给前端做确认。

```python
@dataclass
class ConfirmationRequest:
    session_id: str
    reason: str
    questions: list[dict]
    partial_intent: ResolvedIntent       # 已解析出的部分字段快照
```

确认回调流程：
1. 前端回传 `{"session_id": "xxx", "confirmed_fields": {"creation_window_days": 20}}`
2. 程序将 `confirmed_fields` 注入 `ParsedIntent.fields`，对应字段加入 `explicit_fields`
3. 从 resolver 重新走（跳过 LLM 理解步骤），产出新的 ResolvedIntent
4. 继续进入规划和执行阶段

### 4.5 ExecutionPlan 与 ExecutionResult

分别表示结构化执行计划和执行结果。

```python
@dataclass
class PlanStep:
    step_id: str
    tool_name: str
    args: dict[str, object]              # 仅存放覆盖/特殊参数
    optional: bool = False


@dataclass
class ExecutionPlan:
    plan_id: str
    skeleton: str
    resolved_intent: ResolvedIntent      # 完整的解析结果，执行时自动透传通用参数
    steps: list[PlanStep]


@dataclass
class ExecutionResult:
    success: bool
    outputs: dict[str, object]
    step_results: dict[str, dict]
    failed_step: str | None = None
    error: str | None = None
```

关于 PlanStep.args 的参数来源约定：
- `args` 中仅存放该步骤需要覆盖的特殊参数（如 `include_all_periods: true`）。
- 通用参数（如 `star_min`、`growth_threshold`、`growth_calc_days`）由执行器从 `resolved_intent` 自动填入。
- 执行器合并逻辑：`final_args = {从 resolved_intent 映射的通用参数} | step.args`，step.args 优先。

说明：
- 第一版采用严格顺序执行，不引入 DAG 依赖（去掉了原 `requires` 字段）。如后续需要 search 和 scan 并行，再引入拓扑排序执行器。
- `intent_snapshot` 改为强类型 `ResolvedIntent`，避免 dict 带来的 typo 和类型安全问题。

---

## 5. 各层如何改

## 5.1 接入层

涉及文件：

- api_server.py
- agent_cli.py
- __main__.py

改造目标：

- 入参统一转成 UserRequest。
- 支持中间状态响应 needs_confirmation。
- 前端确认结果可以回填到 confirmed_fields 后重新进入解析流程。

API 建议增加的中间态返回格式：

```json
{
  "status": "needs_confirmation",
  "session_id": "xxx",
  "reason": "growth_calc_days 与 creation_window_days 存在歧义",
  "questions": [
    {
      "field": "creation_window_days",
      "question": "这里的 20 天指新项目创建时间吗？",
      "options": ["是", "不是"]
    }
  ]
}
```

## 5.2 输入理解层

当前文件：

- parsing/intent_detector.py
- parsing/param_extractor.py
- parsing/tool_arg_normalizer.py

目标改造：

1. 新增 parsing/schema.py
   - 定义所有用户参数的字段类型、默认值、范围、枚举值、是否可继承、是否需要确认。

2. 新增 parsing/llm_interpreter.py
   - 只负责把用户输入转成 ParsedIntent。
   - LLM 输出必须是固定 JSON，不允许直接输出工具调用。
   - LLM 需同时判断 `intent_type`（"new" 还是 "patch"）和 `workflow_mode`。

3. 新增 parsing/resolver.py
   - 负责默认值回填、冲突处理、字段裁决、生成 ResolvedIntent。
   - **处理顺序**：先 resolve 控制字段（`workflow_mode`），再 resolve 数值字段（`top_n`、`growth_threshold` 等依赖 mode 的默认值）。
   - **多轮 patch 逻辑**：当 `intent_type == "patch"` 时，从状态层取上一次 `ResolvedIntent`，仅覆盖本次 `explicit_fields` 中的字段，其余保持不变。

4. 新增 parsing/confirm_rules.py
   - 统一管理哪些情况必须向用户发起确认。

5. 现有 parsing/param_extractor.py 降级为 fallback
   - **触发条件**：LLM 返回非法 JSON、解析超时、或 `confidence < 0.5`。
   - fallback 用正则提取关键字段 → 注入 ParsedIntent → 继续走 resolver 主流程。
   - 不再作为主解析入口。

6. 现有 parsing/intent_detector.py 和 parsing/tool_arg_normalizer.py
   - 在主链路中不再使用。保留代码作为 fallback 参考，最终在阶段5清理。

### 推荐的字段 schema 形式

```python
FIELD_SCHEMA = {
    "workflow_mode": {
        "type": "enum",
        "choices": ["comprehensive", "hot_new", "trending_only", "repo_inspect", "db_query"],
        "default": "comprehensive",
    },
    "top_n": {
        "type": "int",
        "min": 1,
        "max": 200,
        "default_by_mode": {
            "comprehensive": 100,
            "hot_new": 20,
            "trending_only": 25,
        },
    },
    "growth_calc_days": {
        "type": "int",
        "min": 1,
        "default": 7,
    },
    "output_format": {
        "type": "enum",
        "choices": ["chat_summary", "detailed_list", "markdown_report"],
        "default": "chat_summary",
    },
}
```

### 确认规则建议

至少覆盖以下高风险歧义：

1. 用户说“近 20 天新项目榜”，无法区分增长窗口还是创建窗口。
2. 用户说“生成报告”，但未说明基于当前结果还是重新执行一次榜单。
3. 用户说“看热门项目”，但上下文不足，不清楚要综合榜还是新项目榜。
4. 用户说“改成前 50”，但没有上一轮可 patch 的 ResolvedIntent。

## 5.3 工具选择与规划层

建议新增 planning 目录：

- planning/skeletons.py
- planning/planner.py
- planning/validator.py

职责调整：

1. skeletons.py
   - 定义可执行的 workflow skeleton。
   - 限制工具组合边界，避免 agent 自由拼装任意步骤。

2. planner.py
   - 根据 ResolvedIntent.workflow_mode **直接**选择对应 skeleton（纯程序映射，不需要 LLM 参与）。
   - 根据 `report_requested` 等标志决定是否启用 optional_steps。
   - 产出 ExecutionPlan。

3. validator.py
   - 校验步骤是否合法（tool_name 必须在已注册工具列表中）。
   - 校验参数是否完整且在边界内。

### workflow_mode → skeleton 映射

每个 workflow_mode 对应唯一的 skeleton，不再有独立的复合骨架。报告生成、描述补充等附加行为通过 optional_steps 控制。

| workflow_mode | skeleton | 说明 |
|---|---|---|
| comprehensive | comprehensive_rank | 综合榜：search + scan + trending + growth + rank |
| hot_new | hot_new_rank | 新项目榜：同上，但搜索阶段附加 created_at 过滤 |
| trending_only | trending_view | 仅看 Trending |
| repo_inspect | repo_inspect | 单仓库增长与详情 |
| db_query | db_query | 数据库查询 |

示例定义：

```python
WORKFLOW_SKELETONS = {
    "comprehensive_rank": {
        "steps": [
            "search_by_keywords",
            "scan_star_range",
            "fetch_trending",
            "batch_check_growth",
            "rank_candidates",
        ],
        "optional_steps": ["generate_report", "describe_project"],
    },
    "hot_new_rank": {
        "steps": [
            "search_by_keywords",
            "scan_star_range",
            "fetch_trending",
            "batch_check_growth",
            "rank_candidates",
        ],
        "optional_steps": ["generate_report", "describe_project"],
    },
    "trending_view": {
        "steps": ["fetch_trending"],
        "optional_steps": [],
    },
    "repo_inspect": {
        "steps": ["check_repo_growth"],
        "optional_steps": ["describe_project"],
    },
    "db_query": {
        "steps": ["get_db_info"],
        "optional_steps": [],
    },
}
```

说明：
- `comprehensive_rank` 和 `hot_new_rank` 步骤相同，区别在执行器传参时 hot_new 会附加 `days_since_created`。
- 当 `report_requested == true` 时，planner 将 `generate_report` 从 optional 提到 steps 末尾。
- 不再设置 `rank_then_report` 等复合骨架，避免骨架膨胀。

## 5.4 执行层

涉及文件：

- agent.py
- agent_tools.py
- execution/pipeline.py

目标改造：

1. agent.py 从 ReAct 执行器收缩为编排器。
   - 管理会话状态。
   - 串接 interpret -> resolve -> plan -> validate -> execute -> summarize。
   - 删除 ReAct 循环（不再有 `for step in range(MAX_TOOL_CALLS_PER_TURN)` + `_call_llm`）。

2. 新增 execution/engine.py（PlanExecutor）。
   - 逐步执行 ExecutionPlan。
   - 保存 step_results。
   - 遇到工具失败时按策略停止（非 optional 步骤失败 → 终止；optional 步骤失败 → 跳过并记日志）。

3. agent_tools.py 回归能力层。
   - 每个 tool 接收已解析完的明确参数。
   - 不再承担用户语义补全和大量纠偏逻辑。

4. execution/pipeline.py 被 PlanExecutor **替代**。
   - scheduled_update.py 改为构造 ExecutionPlan 后调用 PlanExecutor。
   - pipeline.py 在阶段5清理中移除。

### PlanExecutor 参数合并逻辑

执行器在调用每个 tool 时，自动从 `resolved_intent` 映射通用参数，再用 `step.args` 覆盖：

```python
# 每个 tool 定义自己需要从 ResolvedIntent 取哪些字段
TOOL_PARAM_MAPPING = {
    "search_by_keywords": {
        "project_min_star": "project_min_star",
        "days_since_created": "creation_window_days",
        "categories": "categories",
    },
    "scan_star_range": {
        "min_star": "star_min",
        "max_star": "star_max",
        "days_since_created": "creation_window_days",
    },
    "batch_check_growth": {
        "growth_threshold": "growth_threshold",
        "growth_calc_days": "growth_calc_days",
        "days_since_created": "creation_window_days",
        "force_refresh": "force_refresh",
    },
    "rank_candidates": {
        "mode": "workflow_mode",     # comprehensive → comprehensive, hot_new → hot_new
        "top_n": "top_n",
        "days_since_created": "creation_window_days",
    },
    # ...
}
```

执行器合并代码：

```python
class PlanExecutor:
    def execute(self, plan: ExecutionPlan) -> ExecutionResult:
        step_results = {}
        for step in plan.steps:
            # 从 resolved_intent 映射通用参数，step.args 覆盖
            final_args = self._build_args(step, plan.resolved_intent)
            result = self._run_tool(step.tool_name, final_args, step_results)
            step_results[step.step_id] = result

            if result.get("error") and not step.optional:
                return ExecutionResult(
                    success=False,
                    outputs={},
                    step_results=step_results,
                    failed_step=step.step_id,
                    error=result["error"],
                )

        return ExecutionResult(
            success=True,
            outputs=self._collect_outputs(plan, step_results),
            step_results=step_results,
        )

    def _build_args(self, step: PlanStep, intent: ResolvedIntent) -> dict:
        mapping = TOOL_PARAM_MAPPING.get(step.tool_name, {})
        base = {}
        for tool_param, intent_field in mapping.items():
            val = getattr(intent, intent_field, None)
            if val is not None:
                base[tool_param] = val
        base.update(step.args)  # step.args 优先
        return base
```

## 5.5 状态层

建议保留当前 AgentState 思路，但职责收敛为：

- 最近一次 ResolvedIntent（供多轮 patch 使用）
- 最近一次 ExecutionPlan
- 当前候选仓库缓存（search_repos、seen_repos）
- 最近一次排序结果
- pending confirmation 信息

多轮 patch 机制：
- 当 `ParsedIntent.intent_type == "patch"` 时，resolver 从 `AgentState.last_resolved_intent` 取上一轮结果。
- 仅覆盖 `explicit_fields` 中的字段，其余字段继承上一轮值。
- 如果 `last_resolved_intent` 为 None（首轮对话就说"改成前50"），触发确认规则。

避免把自由语义决策状态继续塞进执行阶段。

## 5.6 总结层

总结层建议单独处理，不再和 agent 主循环混在一起。

职责包括：

- 将 ExecutionResult 转成用户看的简洁结果。
- 根据 output_format 决定输出为短摘要、详细说明或报告说明。
- 当 report_requested 为 true 时，给出报告路径和关键摘要。

### 5.7 SYSTEM_PROMPT 改造

重构后 SYSTEM_PROMPT 应大幅精简。当前 prompt 包含了完整的参数表、两大工作流说明、注意事项等（约 150 行），这些职责已移到 schema 和 skeleton 中。

改造后保留：
- 角色定义（"你是 GitHub 热门项目发现助手"）
- LLM 在输入理解阶段的输出格式约束（必须输出固定 JSON 的 ParsedIntent）
- `intent_type` 和 `workflow_mode` 的判断指引
- 用户参数字段列表（简表，供 LLM 识别字段用）

删除：
- 工具列表和调用说明（LLM 不再直接调用工具）
- 工作流步骤说明（由 skeleton 管理）
- 参数默认值和校验规则（由 schema + resolver 管理）
- "注意事项"中的执行细节（由程序控制）

---

## 6. 当前项目的文件级改造建议

| 文件 | 当前职责 | 改造方向 |
|---|---|---|
| agent.py | ReAct 循环、工具调用、状态管理 | 改为线性编排器（删除 ReAct 循环） |
| parsing/param_extractor.py | 正则提取参数 | 降级为 fallback（LLM 失败时兜底） |
| parsing/intent_detector.py | 正则意图识别 | 主链路不再使用，阶段5清理 |
| parsing/tool_arg_normalizer.py | 执行前参数裁决 | 拆成 resolver 和 confirm_rules |
| agent_tools.py | 能力实现 + 部分语义纠偏 | 保留能力实现，去除过多语义判断 |
| execution/pipeline.py | 定时任务流水线 | 被 PlanExecutor 替代，阶段5清理 |
| api_server.py | API 和聊天入口 | 增加 needs_confirmation 中间态 |
| scheduled_update.py | 定时更新调度 | 改为构造 ExecutionPlan 后调用 PlanExecutor |
| docs/design.md | 当前系统整体设计 | 保留现状说明，不承担重构细节 |

建议新增文件：

- parsing/schema.py
- parsing/llm_interpreter.py
- parsing/resolver.py
- parsing/confirm_rules.py
- planning/skeletons.py
- planning/planner.py
- planning/validator.py
- execution/engine.py

---

## 7. 示例链路

用户输入：

给我近 20 天内新创建项目的榜单前 10 名

### 7.1 输入理解输出 ParsedIntent

```json
{
  "intent_type": "new",
  "workflow_mode": "hot_new",
  "fields": {
    "creation_window_days": 20,
    "top_n": 10
  },
  "explicit_fields": ["workflow_mode", "creation_window_days", "top_n"],
  "ambiguous_fields": [],
  "missing_required_fields": [],
  "confidence": 0.92
}
```

### 7.2 程序产出 ResolvedIntent

```json
{
  "workflow_mode": "hot_new",
  "repo": null,
  "categories": null,
  "top_n": 10,
  "growth_calc_days": 7,
  "creation_window_days": 20,
  "growth_threshold": 800,
  "project_min_star": 1000,
  "star_min": 1300,
  "star_max": 45000,
  "since": null,
  "force_refresh": false,
  "report_requested": false,
  "output_format": "chat_summary"
}
```

### 7.3 规划层产出 ExecutionPlan

```json
{
  "skeleton": "hot_new_rank",
  "steps": [
    {"step_id": "s1", "tool_name": "search_by_keywords", "args": {}},
    {"step_id": "s2", "tool_name": "scan_star_range", "args": {}},
    {"step_id": "s3", "tool_name": "fetch_trending", "args": {"include_all_periods": true}},
    {"step_id": "s4", "tool_name": "batch_check_growth", "args": {}},
    {"step_id": "s5", "tool_name": "rank_candidates", "args": {}}
  ]
}
```

说明：steps 中的 args 只存覆盖参数。`days_since_created`、`top_n`、`growth_calc_days` 等通用参数由执行器从 `resolved_intent` 自动映射（参见 5.4 TOOL_PARAM_MAPPING）。

---

## 8. 落地顺序

建议按 5 个阶段推进，避免一次性推翻全部现有逻辑。

### 阶段 1：统一字段 schema 和 ResolvedIntent

目标：

- 收拢参数定义
- 统一默认值与边界
- **同步改造 agent 参数来源**：agent._execute_tool() 从 ResolvedIntent 取参数，替代散落的 resolve/normalize 方法

交付：

- parsing/schema.py
- parsing/resolver.py
- ResolvedIntent 数据结构
- agent._execute_tool() 参数来源切换（确保阶段1有实际接入）

### 阶段 2：确认流

目标：

- 在执行前拦住歧义输入
- API 支持 needs_confirmation

交付：

- parsing/confirm_rules.py
- api_server.py 的中间态返回
- AgentState 中的 pending confirmation

### 阶段 3：规划层

目标：

- 引入 ExecutionPlan
- 引入 skeleton 约束

交付：

- planning/skeletons.py
- planning/planner.py
- planning/validator.py

### 阶段 4：执行器

目标：

- 将 agent 中串工具的逻辑迁移到 execution/engine.py
- 明确 step_results 和错误处理策略

交付：

- execution/engine.py
- agent.py 编排收缩

### 阶段 5：清理旧逻辑

目标：

- 清理散落在 prompt、normalize 和 tool 内部的重复规则
- 保留 fallback 但让主链路稳定可测

---

## 9. 验收标准

重构完成后，应满足以下标准：

1. 所有用户参数定义、默认值和约束都能在一处查看。
2. agent 不再依赖自由式 tool_calls 执行主流程。
3. 执行前一定能得到 ResolvedIntent 和 ExecutionPlan。
4. 歧义请求不会直接执行，而是返回明确确认问题。
5. 执行器只依赖结构化 plan，不依赖自然语言。
6. 测试可分别覆盖输入理解、确认规则、计划生成、计划校验、执行器和结果总结。

---

## 10. 结论

这次重构的关键不是继续优化某一段提示词，而是建立稳定的数据边界：

- ParsedIntent 是语义理解结果。
- ResolvedIntent 是程序裁决后的正式输入。
- ExecutionPlan 是可校验、可回放、可测试的执行计划。
- ExecutionResult 是统一的执行输出。

一旦这几个边界建立起来，输入解析准度、执行稳定性和后续可维护性都会明显提升。