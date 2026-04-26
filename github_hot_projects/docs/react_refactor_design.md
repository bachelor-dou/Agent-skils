# GitHub 热门项目发现系统 — 以 ReAct 为核心的重构方案

## 1. 问题根因

"近20天内新创建项目的榜单前10名" 创建窗口被解析为默认值 45 天，根因链条：

1. `extract_creation_window_days("近20天内新创建项目的榜单前10名")` → **None**
   - 正则只匹配 "20天内创建"、"近20天创建" 等严格模式
   - "近20天内新创建项目" 不匹配任何模式（"20天内" 和 "创建" 之间插入了 "新"）
2. `has_explicit_creation_window()` → **False**
3. `normalize_tool_args()` 看到"用户未显式指定创建窗口"→ **剥掉 LLM 传入的 `days_since_created=20`**
4. `resolve_new_project_days()` 走到兜底 → 返回 `DAYS_SINCE_CREATED`（默认 45 天）

**核心矛盾**：LLM 已经正确理解了"近20天新创建"并传了 `days_since_created=20`，但程序用正则"二审"否决了 LLM 的判断。正则覆盖不了自然语言的多样性，导致正确输入被误杀。

## 2. 设计理念：受约束的 ReAct

保留 ReAct 循环的灵活性（LLM 可以自主决定调用顺序、处理追问、质疑回答），但用**参数 schema + 确认机制**替代散乱的正则纠偏：

核心原则：
- 信任 LLM 的语义理解能力，不再用正则二次否决
- 程序只做类型校验和边界约束，不做语义猜测
- LLM 不确定时，要求它主动向用户确认（通过 prompt 指引）
- 保留 ReAct 循环处理追问、质疑、多轮补充等开放场景

## 3. 架构对比

| 维度 | 当前 ReAct | 重构后 ReAct |
|------|-----------|------------|
| 参数解析 | LLM 提参 → 正则二审 → 程序纠偏 | LLM 提参 → 程序只做类型/边界校验 |
| 参数纠偏 | 100+ 行正则规则散落在 3 个文件 | 统一 schema 做机械校验，不做语义猜测 |
| 不确定参数 | 程序默默回退默认值 | LLM 主动问用户确认 |
| 工具选择 | LLM 自由选择 + 程序修正 mode | LLM 自由选择，程序不修正语义决策 |
| 灵活性 | 有但被正则压制 | 完整保留，LLM 可处理追问/质疑/多轮 |

## 4. 用户可指定参数

14 个参数不变（同 refactor_design.md 第 3 节），但改由 LLM 直接从用户消息理解并传给 Tool，程序不做语义判断。

## 5. 需要改动的文件

### 5.1 删除/大幅简化

| 文件 | 改动 |
|------|------|
| `parsing/param_extractor.py` | **大幅删减**。去掉所有 `extract_*` 和 `has_explicit_*` 函数中的正则匹配逻辑。仅保留 `latest_user_message()` 工具函数。 |
| `parsing/intent_detector.py` | **删除**。不再用正则判断意图，完全由 LLM 决定。 |
| `parsing/tool_arg_normalizer.py` | **重写为 `parsing/arg_validator.py`**。从"语义纠偏"改为"类型校验 + 边界裁剪"。 |

### 5.2 新增文件

| 文件 | 职责 |
|------|------|
| `parsing/schema.py` | 参数 schema 定义：每个 tool 参数的类型、范围、默认值。**不含语义规则**。 |
| `parsing/arg_validator.py` | 基于 schema 做机械校验：类型转换、边界裁剪、缺失参数补默认值。 |

### 5.3 需要修改的文件

| 文件 | 改动要点 |
|------|---------|
| `agent.py` | 1) SYSTEM_PROMPT 改造（见 5.6）<br>2) `_execute_tool()` 中去掉所有 `_resolve_*` 和 `_normalize_tool_args` 调用，改为调用 `arg_validator.validate(tool_name, args)`<br>3) 去掉 `_is_new_project_workflow_request` 等正则代理方法 |
| `agent_tools.py` | TOOL_SCHEMAS 中每个参数的 description 增强，让 LLM 有足够信息正确填参 |

## 6. 核心代码设计

### 6.1 `parsing/schema.py` — 参数 schema（纯数据）

```python
"""参数 schema 定义：类型、范围、默认值。不含任何语义规则。"""

from ..common.config import (
    MIN_STAR_FILTER, STAR_RANGE_MIN, STAR_RANGE_MAX,
    STAR_GROWTH_THRESHOLD, GROWTH_CALC_DAYS,
    HOT_PROJECT_COUNT, HOT_NEW_PROJECT_COUNT,
)

TOOL_PARAM_SCHEMA = {
    "search_by_keywords": {
        "categories": {"type": "list_str", "default": None},
        "project_min_star": {"type": "int", "min": 0, "default": MIN_STAR_FILTER},
        "max_pages": {"type": "int", "min": 1, "max": 10, "default": 3},
        "days_since_created": {"type": "int", "min": 1, "default": None},
    },
    "scan_star_range": {
        "min_star": {"type": "int", "min": 1, "default": STAR_RANGE_MIN},
        "max_star": {"type": "int", "min": 1, "default": STAR_RANGE_MAX},
        "days_since_created": {"type": "int", "min": 1, "default": None},
    },
    "batch_check_growth": {
        "growth_threshold": {"type": "int", "min": 0, "default": STAR_GROWTH_THRESHOLD},
        "growth_calc_days": {"type": "int", "min": 1, "default": GROWTH_CALC_DAYS},
        "days_since_created": {"type": "int", "min": 1, "default": None},
        "force_refresh": {"type": "bool", "default": False},
    },
    "rank_candidates": {
        "mode": {"type": "enum", "choices": ["comprehensive", "hot_new"], "default": "comprehensive"},
        "top_n": {
            "type": "int", "min": 1, "max": 200,
            "default_by_mode": {
                "comprehensive": HOT_PROJECT_COUNT,
                "hot_new": HOT_NEW_PROJECT_COUNT,
            },
        },
        "days_since_created": {"type": "int", "min": 1, "default": None},
    },
    "check_repo_growth": {
        "repo": {"type": "str", "required": True},
        "growth_calc_days": {"type": "int", "min": 1, "default": GROWTH_CALC_DAYS},
    },
    "fetch_trending": {
        "since": {"type": "enum", "choices": ["daily", "weekly", "monthly"], "default": "weekly"},
        "include_all_periods": {"type": "bool", "default": False},
    },
    "describe_project": {
        "repo": {"type": "str", "required": True},
    },
    "generate_report": {},
    "get_db_info": {
        "repo": {"type": "str", "default": None},
    },
}
```

### 6.2 `parsing/arg_validator.py` — 机械校验

```python
"""基于 schema 做类型校验和边界裁剪。不猜语义，不否决 LLM 的参数。"""

import logging
from .schema import TOOL_PARAM_SCHEMA

logger = logging.getLogger("discover_hot")


def validate_tool_args(tool_name: str, args: dict) -> dict:
    """校验并填充默认值，返回清洁参数。

    规则：
    1. LLM 传了的参数：只做类型转换和边界裁剪，不删除
    2. LLM 没传的参数：填默认值
    3. 不在 schema 中的参数：保留（向前兼容）
    """
    schema = TOOL_PARAM_SCHEMA.get(tool_name, {})
    result = dict(args)

    for param_name, spec in schema.items():
        if param_name in result:
            result[param_name] = _coerce(result[param_name], spec)
        else:
            default = _get_default(spec, result)
            if default is not None:
                result[param_name] = default

    return result


def _get_default(spec: dict, current_args: dict):
    """获取默认值，支持 default_by_mode。"""
    if "default_by_mode" in spec:
        mode = current_args.get("mode", "comprehensive")
        return spec["default_by_mode"].get(mode, list(spec["default_by_mode"].values())[0])
    return spec.get("default")


def _coerce(value, spec):
    """类型转换和边界裁剪。"""
    vtype = spec.get("type", "str")
    if vtype == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            return spec.get("default")
        if "min" in spec:
            value = max(value, spec["min"])
        if "max" in spec:
            value = min(value, spec["max"])
        return value
    elif vtype == "bool":
        return bool(value)
    elif vtype == "enum":
        return value if value in spec.get("choices", []) else spec.get("default")
    elif vtype == "list_str":
        return value if isinstance(value, list) else spec.get("default")
    return value
```

### 6.3 SYSTEM_PROMPT 改造重点

在现有 prompt 基础上，增加以下关键指引：

```
## 参数确认规则（关键）

1. **信任你自己的参数判断**：你从用户消息中理解的参数直接传给 Tool，程序不会否决你的判断。
2. **不确定时必须先确认**：如果你无法确定某个参数值，不要猜默认值，直接问用户。
   典型需要确认的场景：
   a) 用户说"近20天"但不清楚指增长窗口还是创建窗口
   b) 用户说"热门项目"但不清楚要综合榜还是新项目榜
   c) 用户说"生成报告"但不清楚是基于当前结果还是重新执行
3. **参数透传原则**：
   - days_since_created：用户提到"新创建"、"新项目"、"天内创建"时传入具体天数
   - growth_calc_days：用户提到"近N天增长"、"最近N天"时传入
   - 两者可以共存：创建窗口和增长窗口是独立参数
4. **不要遗漏用户提到的参数**：用户说"前10名"必须传 top_n=10，说"近20天新创建"必须传 days_since_created=20。
```

### 6.4 TOOL_SCHEMAS description 增强

在每个 tool 参数的 description 中明确业务语义，例如：

```python
# search_by_keywords 的 days_since_created
{
    "name": "days_since_created",
    "description": "新项目创建时间窗口（天）。指定后只搜索创建时间在该天数以内的仓库。"
                   "例如用户说'近20天内新创建的项目'则传 20。"
                   "与 growth_calc_days（增长统计窗口）是独立参数。"
                   "如果用户意图不涉及新项目过滤，不要传此参数。",
    "type": "integer",
}
```

### 6.5 `agent.py` `_execute_tool()` 简化

```python
def _execute_tool(self, name: str, args: dict) -> dict:
    state = self.state
    # 只做类型/边界校验，不做语义纠偏
    validated_args = validate_tool_args(name, args)
    self._maybe_reset_discovery_state(name, validated_args)

    if name == "search_by_keywords":
        result = tool_search_by_keywords(
            state.token_mgr,
            categories=validated_args.get("categories"),
            project_min_star=validated_args.get("project_min_star"),
            max_pages=validated_args.get("max_pages"),
            days_since_created=validated_args.get("days_since_created"),
        )
        raw_repos = result.pop("_raw_repos", [])
        state.last_search_repos = raw_repos
        state.seen_repos.update(r["full_name"] for r in raw_repos)
        return result
    # ... 其他 tool 类似，直接取 validated_args ...
```

## 7. 改造后数据流

```
用户消息 → SYSTEM_PROMPT（含参数确认指引 + 增强的 Tool description）
         → LLM 理解意图 + 选择 Tool + 填参数
         → agent._execute_tool()
           → validate_tool_args()  // 只做类型/边界校验，不删 LLM 传的参数
           → 调用 tool 函数
           → 返回结果给 LLM
         → LLM 观察结果
           → 继续调 Tool / 向用户确认参数 / 返回回复
```

关键区别：**程序不再用正则二审 LLM 的语义判断**。LLM 传了 `days_since_created=20`，validate 只检查它是正整数，不会因为"正则没匹配到创建窗口"就把它删掉。

## 8. 处理用户追问/质疑

ReAct 循环天然支持：
- 用户说"这个结果不对，创建窗口应该是20天不是45天" → LLM 直接修正参数重新调用
- 用户说"能不能加上AI相关的类别？" → LLM 追加 categories 参数
- 用户说"为什么 vllm 没有在榜单里？" → LLM 调用 check_repo_growth 查询并解释

## 9. 落地顺序

| 阶段 | 改动 | 验收 |
|------|------|------|
| 1 | 新建 `parsing/schema.py` + `parsing/arg_validator.py` | 单元测试通过 |
| 2 | `agent.py`：`_execute_tool()` 改用 `validate_tool_args()`，删除所有 `_resolve_*` 调用 | "近20天新创建项目前10名" 正确传入 `days_since_created=20` |
| 3 | SYSTEM_PROMPT 增加参数确认指引 + TOOL_SCHEMAS description 增强 | LLM 不确定时主动问用户 |
| 4 | 删除 `parsing/intent_detector.py`，清理 `parsing/param_extractor.py` 中的死代码 | 测试全部通过 |
| 5 | `parsing/tool_arg_normalizer.py` 重命名为 `arg_validator.py`（迁移完成后删除旧文件） | 无 import 引用旧模块 |

## 10. 与 Pipeline 方案的关系

| 维度 | Pipeline 方案 (refactor_design.md) | ReAct 方案 (本文档) |
|------|-----------------------------------|-------------------|
| 适用场景 | 明确的批量任务：定时更新、完整榜单生成 | 交互式对话：追问、确认、多轮修改 |
| 稳定性 | 高（确定性流程，可回放） | 中（依赖 LLM 判断质量） |
| 灵活性 | 低（工作流固定） | 高（处理开放式问题） |
| 建议 | 保留用于 scheduled_update.py | 用于 api_server + agent_cli 的用户对话 |

两套方案可以共存：**对话入口走 ReAct，定时任务走 Pipeline**。agent_tools.py 作为共享能力层不变。
