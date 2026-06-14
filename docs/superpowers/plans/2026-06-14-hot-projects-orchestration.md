# hot_projects 编排层重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新建自包含项目 `hot_projects`（原 `github_hot_projects` 去掉 github），重写编排层为"复合工具 + 注册表 + 分阶段缓存 + 单 ReAct + A/B 双 LLM 后端"，下层稳定模块原样复制；功能不变、结构清晰、可扩展。**绝不改动原 `github_hot_projects`。**

**Architecture:** 分层——入口层 / 编排层(agent·tools·pipeline·capabilities) / providers(Provider 接口 + GitHubProvider) / infra(db·llm·并发)。榜单流程内聚为复合工具（唯一 ranking_pipeline），原子工具自由组合；LLM 走 A/B 双方案逐调用回退。

**Tech Stack:** Python 3.10+、requests、httpx、FastAPI、pytest。下层算法（增长估算、并发调度、报告渲染）原样复用。

参考设计：`docs/superpowers/specs/2026-06-14-github-hot-projects-orchestration-redesign.md`

---

## 约定

- 工作目录根：`/root/code/Agent-skils`，原项目 `github_hot_projects/`，新项目 `hot_projects/`。
- 所有 `pytest` 命令在 `/root/code/Agent-skils` 下运行（`python -m pytest hot_projects/tests/...`）。
- 每个 Task 末尾 commit。提交信息用前缀 `feat/refactor/test/chore`。
- **下层复制只改 import 路径与命名，不改算法逻辑。**

---

## Phase 0 — 项目骨架 + 复制下层

### Task 0.1: 创建包骨架

**Files:**
- Create: `hot_projects/__init__.py` 及子包目录

- [ ] **Step 1: 建目录与空 `__init__.py`**

Run:
```bash
cd /root/code/Agent-skils
mkdir -p hot_projects/{agent,tools,pipeline,capabilities,providers/github,infra/concurrency,web,tests,report,logs}
for d in hot_projects hot_projects/agent hot_projects/tools hot_projects/pipeline \
         hot_projects/capabilities hot_projects/providers hot_projects/providers/github \
         hot_projects/infra hot_projects/infra/concurrency hot_projects/tests; do
  touch "$d/__init__.py"
done
echo done
```

- [ ] **Step 2: 验证目录**

Run: `find hot_projects -name __init__.py | sort`
Expected: 列出上述各包的 `__init__.py`

- [ ] **Step 3: Commit**

```bash
cd /root/code/Agent-skils
git add hot_projects
git commit -m "chore: scaffold hot_projects package skeleton"
```

---

### Task 0.2: 复制 infra（db / llm / exceptions / 并发框架）

**Files:**
- Create: `hot_projects/infra/db.py` ← `github_hot_projects/common/db.py`
- Create: `hot_projects/infra/exceptions.py` ← `github_hot_projects/common/exceptions.py`
- Create: `hot_projects/infra/concurrency/{task_base.py,dispatcher.py,tasks.py,task_help.py}` ← `github_hot_projects/tasks/*`
- Create: `hot_projects/infra/llm.py` ← `github_hot_projects/common/llm.py`（下一阶段会改造为 A/B，先原样落地）

- [ ] **Step 1: 复制文件**

Run:
```bash
cd /root/code/Agent-skils
cp github_hot_projects/common/db.py          hot_projects/infra/db.py
cp github_hot_projects/common/exceptions.py  hot_projects/infra/exceptions.py
cp github_hot_projects/common/llm.py         hot_projects/infra/llm.py
cp github_hot_projects/tasks/task_base.py    hot_projects/infra/concurrency/task_base.py
cp github_hot_projects/tasks/async_worker_pool.py hot_projects/infra/concurrency/dispatcher.py
cp github_hot_projects/tasks/task.py         hot_projects/infra/concurrency/tasks.py
cp github_hot_projects/tasks/task_help.py    hot_projects/infra/concurrency/task_help.py
echo copied
```

- [ ] **Step 2: 重写 import 路径**

新旧映射（在复制后的文件内做替换）：
- `from ..common.config` / `from .config` → `from ..config`（infra 下）或 `from ...config`（concurrency 下，见下）
- `from .common.db` / `from ..common.db` → `from .db`（infra 内）或 `from ..db`（concurrency → infra）
- `from .common.async_token_pool` → `from ..providers.github.token_pool`（见 Task 0.3）
- `from ..common.github_api` → `from ...providers.github.api`
- `from ..growth_estimator` → `from ...providers.github.growth_estimator`
- `from ..github_trending` → `from ...providers.github.trending`
- `from .task_base` / `from .task_help` / `from .task` → 保持包内相对（`concurrency` 内部）

执行（逐文件用编辑工具按上表替换；可借助脚本，但**不改任何逻辑行**）。完成后人工核对每个 import。

- [ ] **Step 3: 复制 token_pool（属于 GitHub Provider，提前放好以便 import 成立）**

Run:
```bash
cd /root/code/Agent-skils
cp github_hot_projects/common/async_token_pool.py hot_projects/providers/github/token_pool.py
echo ok
```

- [ ] **Step 4: 暂不验证导入（等 Task 0.3 复制 github_api 等后统一验证）。Commit**

```bash
cd /root/code/Agent-skils
git add hot_projects/infra hot_projects/providers/github/token_pool.py
git commit -m "chore: copy infra (db/llm/exceptions/concurrency) into hot_projects"
```

---

### Task 0.3: 复制 GitHub Provider 下层实现 + 评分/报告

**Files:**
- Create: `hot_projects/providers/github/api.py` ← `common/github_api.py`
- Create: `hot_projects/providers/github/growth_estimator.py` ← `growth_estimator.py`
- Create: `hot_projects/providers/github/trending.py` ← `github_trending.py`
- Create: `hot_projects/ranking.py` ← `ranking.py`
- Create: `hot_projects/report.py` ← `report.py`
- Create: `hot_projects/web/` ← `web/`

- [ ] **Step 1: 复制文件**

Run:
```bash
cd /root/code/Agent-skils
cp github_hot_projects/common/github_api.py   hot_projects/providers/github/api.py
cp github_hot_projects/growth_estimator.py    hot_projects/providers/github/growth_estimator.py
cp github_hot_projects/github_trending.py     hot_projects/providers/github/trending.py
cp github_hot_projects/ranking.py             hot_projects/ranking.py
cp github_hot_projects/report.py              hot_projects/report.py
cp -r github_hot_projects/web/.               hot_projects/web/
echo copied
```

- [ ] **Step 2: 重写 import 路径**

- `api.py`：`from .config` → `from ...config`；`from .async_token_pool` → `from .token_pool`；`from .exceptions` → `from ...infra.exceptions`
- `growth_estimator.py`：`from .common.config` → `from ..config`(改 `from ...config`)；`from .common.github_api` → `from .api`；`from .common.async_token_pool` → `from .token_pool`
- `trending.py`：无项目内 import（仅 requests/re），无需改
- `ranking.py`：`from .common.config` → `from .config`
- `report.py`：`from .common.config` → `from .config`；`from .common.llm` → `from .infra.llm`

逐文件替换，**不改逻辑**。

- [ ] **Step 3: 删除 B 模式死代码（整理原则#3）**

在 `providers/github/api.py`、`infra/concurrency/tasks.py` 中删除成片注释的 "B模式 请求级 token 借还" 注释块与对应的 `token_idx is None: acquire/release` 分支（当前仅用 A 模式，token_idx 由 dispatcher 绑定）。**仅删未启用分支与注释，不改 A 模式行为。** 若不确定边界，本步可跳过（记入后续清理）。

- [ ] **Step 4: Commit**

```bash
cd /root/code/Agent-skils
git add hot_projects/providers hot_projects/ranking.py hot_projects/report.py hot_projects/web
git commit -m "chore: copy github provider impl + ranking/report/web into hot_projects"
```

---

### Task 0.4: config.py（去 github 强绑定 + A/B LLM 环境变量）

**Files:**
- Create: `hot_projects/config.py` ← 基于 `github_hot_projects/common/config.py` 改造

- [ ] **Step 1: 写 config.py**

以原 `common/config.py` 为基础复制，做两处改造：
1. 路径基准 `PACKAGE_DIR = Path(__file__).resolve().parent`（新项目 config 在包根，不再是 parents[1]）。
2. 新增 A/B LLM 配置块（替换原单一 LLM_* + LITE 块）：

```python
# ===== LLM 方案 A（主力）: Azure OpenAI =====
LLM_A_BACKEND = os.environ.get("LLM_A_BACKEND", "azure")
LLM_A_URL = os.environ.get("LLM_A_URL", "https://ceshi-001.openai.azure.com/openai/v1/chat/completions?api-version=preview")
LLM_A_KEY = os.environ.get("LLM_A_KEY", "")
LLM_A_MODEL = os.environ.get("LLM_A_MODEL", "gpt-5.4")
LLM_A_LITE_MODEL = os.environ.get("LLM_A_LITE_MODEL", "gpt-5.4-mini")

# ===== LLM 方案 B（备选）: SiliconFlow =====
LLM_B_BACKEND = os.environ.get("LLM_B_BACKEND", "openai")
LLM_B_URL = os.environ.get("LLM_B_URL", "https://api.siliconflow.cn/v1/chat/completions")
LLM_B_KEY = os.environ.get("LLM_B_KEY", "")
LLM_B_MODEL = os.environ.get("LLM_B_MODEL", "Pro/zai-org/GLM-5")
LLM_B_LITE_MODEL = os.environ.get("LLM_B_LITE_MODEL", "Qwen/Qwen3.5-35B-A3B")
```

保留：`STAR_GROWTH_THRESHOLD`/`MIN_STAR`/`MAX_STAR`/`HOT_PROJECT_COUNT`/`HOT_NEW_PROJECT_COUNT`/`GROWTH_CALC_DAYS`/`DAYS_SINCE_CREATED`/`DATA_EXPIRE_DAYS`/`DEFAULT_SCORE_MODE`/请求控制/`SEARCH_KEYWORDS`/路径(`DATA_DIR`/`DB_FILE_PATH`/`CHECKPOINT_FILE_PATH`/`REPORT_DIR`/`LOG_DIR`)/CORS/IP 黑名单。删除原 `LLM_API_*`/`LLM_LITE_*`（被 A/B 取代）。

- [ ] **Step 2: 验证全包可导入**

Run:
```bash
cd /root/code/Agent-skils
python -c "import hot_projects.config, hot_projects.infra.db, hot_projects.infra.llm, hot_projects.infra.concurrency.tasks, hot_projects.providers.github.api, hot_projects.providers.github.growth_estimator, hot_projects.ranking, hot_projects.report; print('IMPORT_OK')"
```
Expected: `IMPORT_OK`（若报 import 错，回到 0.2/0.3 修路径）

- [ ] **Step 3: Commit**

```bash
cd /root/code/Agent-skils
git add hot_projects/config.py
git commit -m "feat: add hot_projects config with A/B LLM env vars"
```

---

## Phase 1 — LLM 客户端（A/B 双后端 + 逐调用回退）

### Task 1.1: 后端参数适配器

**Files:**
- Create: `hot_projects/infra/llm_client.py`
- Test: `hot_projects/tests/test_llm_client.py`

- [ ] **Step 1: 写失败测试（参数适配）**

```python
# hot_projects/tests/test_llm_client.py
from hot_projects.infra.llm_client import build_payload, build_headers

def test_azure_payload_uses_max_completion_tokens_and_drops_thinking():
    payload = build_payload(
        backend="azure", model="gpt-5.4",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100, temperature=0.3, enable_thinking=True, thinking_budget=512,
        tools=None,
    )
    assert payload["model"] == "gpt-5.4"
    assert payload["max_completion_tokens"] == 100
    assert "max_tokens" not in payload
    assert "enable_thinking" not in payload
    assert "thinking_budget" not in payload
    assert "temperature" not in payload  # azure: 省略温度

def test_openai_payload_keeps_legacy_params():
    payload = build_payload(
        backend="openai", model="GLM-5",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100, temperature=0.3, enable_thinking=True, thinking_budget=512,
        tools=None,
    )
    assert payload["max_tokens"] == 100
    assert payload["temperature"] == 0.3
    assert payload["enable_thinking"] is True
    assert payload["thinking_budget"] == 512

def test_headers_by_backend():
    assert build_headers("azure", "K")["api-key"] == "K"
    assert "Authorization" not in build_headers("azure", "K")
    assert build_headers("openai", "K")["Authorization"] == "Bearer K"
```

- [ ] **Step 2: 运行验证失败**

Run: `cd /root/code/Agent-skils && python -m pytest hot_projects/tests/test_llm_client.py -v`
Expected: FAIL（`build_payload`/`build_headers` 未定义）

- [ ] **Step 3: 实现适配器**

```python
# hot_projects/infra/llm_client.py
"""LLM 客户端：A/B 双后端 + 逐调用回退 + 按后端参数适配。"""
import logging
import time
import requests

logger = logging.getLogger("hot_projects")

LLM_RETRY_BACKOFF_SECONDS = (1.0, 2.0, 4.0)


def build_headers(backend: str, key: str) -> dict:
    if backend == "azure":
        return {"api-key": key, "Content-Type": "application/json"}
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def build_payload(
    backend: str,
    model: str,
    messages: list[dict],
    max_tokens: int | None = None,
    temperature: float | None = None,
    enable_thinking: bool | None = None,
    thinking_budget: int | None = None,
    tools: list[dict] | None = None,
) -> dict:
    payload: dict = {"model": model, "messages": messages}
    if backend == "azure":
        if max_tokens is not None:
            payload["max_completion_tokens"] = max_tokens
        # azure(gpt-5.x): 省略 temperature；不发 enable_thinking/thinking_budget
    else:
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if enable_thinking is not None:
            payload["enable_thinking"] = enable_thinking
        if thinking_budget is not None:
            payload["thinking_budget"] = thinking_budget
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    return payload
```

- [ ] **Step 4: 运行验证通过**

Run: `cd /root/code/Agent-skils && python -m pytest hot_projects/tests/test_llm_client.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /root/code/Agent-skils
git add hot_projects/infra/llm_client.py hot_projects/tests/test_llm_client.py
git commit -m "feat: llm backend param/header adapters with tests"
```

---

### Task 1.2: 逐调用回退的请求函数

**Files:**
- Modify: `hot_projects/infra/llm_client.py`
- Test: `hot_projects/tests/test_llm_client.py`

- [ ] **Step 1: 写失败测试（A 失败回退 B；A 成功不碰 B）**

追加到 `test_llm_client.py`：

```python
from unittest.mock import patch
from hot_projects.infra import llm_client

class _Scheme:
    def __init__(self, backend, url, key, model, lite_model):
        self.backend, self.url, self.key = backend, url, key
        self.model, self.lite_model = model, lite_model

def _ok_response(content="ok"):
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {}}

def test_failover_uses_b_when_a_fails():
    a = _Scheme("azure", "urlA", "kA", "gpt-5.4", "gpt-5.4-mini")
    b = _Scheme("openai", "urlB", "kB", "GLM-5", "Qwen")
    calls = []
    def fake_call(scheme, model, **kw):
        calls.append((scheme.backend, model))
        if scheme.backend == "azure":
            return None  # A 失败
        return _ok_response("from-B")
    with patch.object(llm_client, "_request_once", side_effect=fake_call):
        client = llm_client.LLMClient(a, b)
        resp = client.chat([{"role": "user", "content": "hi"}], lite=False)
    assert resp["choices"][0]["message"]["content"] == "from-B"
    assert calls[0][0] == "azure" and calls[1][0] == "openai"

def test_no_failover_when_a_ok():
    a = _Scheme("azure", "urlA", "kA", "gpt-5.4", "gpt-5.4-mini")
    b = _Scheme("openai", "urlB", "kB", "GLM-5", "Qwen")
    calls = []
    def fake_call(scheme, model, **kw):
        calls.append(scheme.backend)
        return _ok_response("from-A")
    with patch.object(llm_client, "_request_once", side_effect=fake_call):
        client = llm_client.LLMClient(a, b)
        resp = client.chat([{"role": "user", "content": "hi"}], lite=False)
    assert resp["choices"][0]["message"]["content"] == "from-A"
    assert calls == ["azure"]

def test_lite_uses_lite_model():
    a = _Scheme("azure", "urlA", "kA", "gpt-5.4", "gpt-5.4-mini")
    b = _Scheme("openai", "urlB", "kB", "GLM-5", "Qwen")
    seen = []
    def fake_call(scheme, model, **kw):
        seen.append(model)
        return _ok_response()
    with patch.object(llm_client, "_request_once", side_effect=fake_call):
        client = llm_client.LLMClient(a, b)
        client.chat([{"role": "user", "content": "hi"}], lite=True)
    assert seen == ["gpt-5.4-mini"]
```

- [ ] **Step 2: 运行验证失败**

Run: `cd /root/code/Agent-skils && python -m pytest hot_projects/tests/test_llm_client.py -v`
Expected: FAIL（`LLMClient`/`_request_once` 未定义）

- [ ] **Step 3: 实现 LLMClient + _request_once**

追加到 `hot_projects/infra/llm_client.py`：

```python
from dataclasses import dataclass


@dataclass
class LLMScheme:
    backend: str
    url: str
    key: str
    model: str
    lite_model: str


def _request_once(scheme, model, *, messages, tools, max_tokens, temperature,
                  enable_thinking, thinking_budget, timeout=300):
    """单次请求单个后端；成功返回 data dict，失败返回 None。"""
    headers = build_headers(scheme.backend, scheme.key)
    payload = build_payload(
        scheme.backend, model, messages,
        max_tokens=max_tokens, temperature=temperature,
        enable_thinking=enable_thinking, thinking_budget=thinking_budget, tools=tools,
    )
    for attempt in range(len(LLM_RETRY_BACKOFF_SECONDS)):
        try:
            resp = requests.post(scheme.url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                msg = (data.get("choices") or [{}])[0].get("message", {})
                if (msg.get("content") or "").strip() or msg.get("tool_calls"):
                    return data
                logger.warning("[LLM] %s 空响应, attempt=%d", scheme.backend, attempt + 1)
            else:
                logger.warning("[LLM] %s HTTP %s: %s", scheme.backend, resp.status_code, resp.text[:200])
        except requests.RequestException as e:
            logger.warning("[LLM] %s 请求异常: %s", scheme.backend, e)
        if attempt < len(LLM_RETRY_BACKOFF_SECONDS) - 1:
            time.sleep(LLM_RETRY_BACKOFF_SECONDS[attempt])
    return None


class LLMClient:
    """A/B 双后端，逐调用回退：每次先 A，失败用 B；下次仍先 A。"""

    def __init__(self, scheme_a, scheme_b):
        self.a = scheme_a
        self.b = scheme_b

    def chat(self, messages, *, tools=None, lite=False, max_tokens=16384,
             temperature=0.3, enable_thinking=None, thinking_budget=None):
        for scheme in (self.a, self.b):
            model = scheme.lite_model if lite else scheme.model
            data = _request_once(
                scheme, model, messages=messages, tools=tools,
                max_tokens=max_tokens, temperature=temperature,
                enable_thinking=enable_thinking, thinking_budget=thinking_budget,
            )
            if data is not None:
                return data
            logger.warning("[LLM] 方案 %s 失败，尝试回退。", scheme.backend)
        return None
```

- [ ] **Step 4: 运行验证通过**

Run: `cd /root/code/Agent-skils && python -m pytest hot_projects/tests/test_llm_client.py -v`
Expected: PASS（全部）

- [ ] **Step 5: 增加 from_config 工厂 + Commit**

追加：
```python
def client_from_config():
    from .. import config as cfg
    a = LLMScheme(cfg.LLM_A_BACKEND, cfg.LLM_A_URL, cfg.LLM_A_KEY, cfg.LLM_A_MODEL, cfg.LLM_A_LITE_MODEL)
    b = LLMScheme(cfg.LLM_B_BACKEND, cfg.LLM_B_URL, cfg.LLM_B_KEY, cfg.LLM_B_MODEL, cfg.LLM_B_LITE_MODEL)
    return LLMClient(a, b)
```

```bash
cd /root/code/Agent-skils
git add hot_projects/infra/llm_client.py hot_projects/tests/test_llm_client.py
git commit -m "feat: LLMClient with per-call A/B failover and tests"
```

- [ ] **Step 6: 改造 infra/llm.py 的描述/压缩函数走 LLMClient.lite**

把 `infra/llm.py` 中 `call_llm_describe` / `batch_condense_descriptions` 的底层 HTTP 调用替换为 `client_from_config().chat(..., lite=True)`；**保留函数签名与返回结构不变**（report.py / capabilities 依赖它们）。运行下层报告相关测试确保未破坏（见 Phase 8）。

---

## Phase 2 — Provider 接口 + Repo 模型 + GitHubProvider

### Task 2.1: Provider 接口与归一化 Repo 模型

**Files:**
- Create: `hot_projects/providers/base.py`
- Test: `hot_projects/tests/test_provider_base.py`

- [ ] **Step 1: 写失败测试**

```python
# hot_projects/tests/test_provider_base.py
from hot_projects.providers.base import Repo, Provider

def test_repo_from_github_item():
    item = {"full_name": "a/b", "stargazers_count": 1500, "description": "x",
            "language": "Python", "topics": ["ai"], "created_at": "2026-01-01T00:00:00Z"}
    r = Repo.from_github(item)
    assert r.full_name == "a/b"
    assert r.star == 1500
    assert r.language == "Python"
    assert r.created_at.startswith("2026-01-01")

def test_provider_is_abstract():
    import pytest
    with pytest.raises(TypeError):
        Provider()  # 抽象类不可实例化
```

- [ ] **Step 2: 运行验证失败**

Run: `cd /root/code/Agent-skils && python -m pytest hot_projects/tests/test_provider_base.py -v`
Expected: FAIL

- [ ] **Step 3: 实现**

```python
# hot_projects/providers/base.py
"""数据源 Provider 接口 + 归一化 Repo 模型（多平台边界）。"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Repo:
    full_name: str
    star: int
    description: str = ""
    language: str = ""
    topics: list[str] = field(default_factory=list)
    created_at: str = ""
    forks: int = 0
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_github(cls, item: dict) -> "Repo":
        return cls(
            full_name=item.get("full_name", ""),
            star=item.get("stargazers_count", item.get("star", 0)),
            description=(item.get("description") or "")[:500],
            language=item.get("language") or "",
            topics=item.get("topics") or [],
            created_at=item.get("created_at", "") or "",
            forks=item.get("forks_count", item.get("forks", 0)),
            raw=item,
        )


class Provider(ABC):
    """平台数据源接口。编排层只依赖本接口与 Repo。"""

    @abstractmethod
    def search_by_keywords(self, categories, min_star, days_since_created): ...

    @abstractmethod
    def scan_star_range(self, min_star, max_star, seen_repos, days_since_created): ...

    @abstractmethod
    def repo_info(self, repo: str): ...

    @abstractmethod
    def repo_growth(self, repo: str, growth_calc_days: int): ...

    @abstractmethod
    def batch_growth(self, repos, db, **kwargs): ...

    @abstractmethod
    def fetch_trending(self, trending_range: str): ...

    @abstractmethod
    def search_similar(self, name: str, limit: int = 5): ...
```

- [ ] **Step 4: 运行验证通过 + Commit**

Run: `cd /root/code/Agent-skils && python -m pytest hot_projects/tests/test_provider_base.py -v` → PASS
```bash
cd /root/code/Agent-skils
git add hot_projects/providers/base.py hot_projects/tests/test_provider_base.py
git commit -m "feat: Provider interface and normalized Repo model"
```

---

### Task 2.2: GitHubProvider（薄封装现有 capability 函数）

**Files:**
- Create: `hot_projects/providers/github/provider.py`
- Test: `hot_projects/tests/test_github_provider.py`

> GitHubProvider 调用 Phase 3 的 capability 纯函数。**实现顺序**：先做 Phase 3（capabilities）再回填本 Task 的实现；或本 Task 仅建类骨架+`search_similar`（直接用 `providers/github/api.search_github_repos`），其余方法在 Phase 3 后接线。本计划按"先 capabilities 后 provider 接线"执行，故本 Task 仅占位 `search_similar` + 骨架，其余在 Task 5.x 接线时补齐。

- [ ] **Step 1: 写 search_similar 测试**

```python
# hot_projects/tests/test_github_provider.py
from unittest.mock import patch
from hot_projects.providers.github.provider import GitHubProvider

def test_search_similar_returns_repos():
    fake_items = [{"full_name": "vllm-project/vllm", "stargazers_count": 30000},
                  {"full_name": "x/vllm-fork", "stargazers_count": 1300}]
    with patch("hot_projects.providers.github.provider.search_github_repos", return_value=fake_items):
        p = GitHubProvider(token_mgr=object())
        repos = p.search_similar("vllm", limit=5)
    assert repos[0].full_name == "vllm-project/vllm"
    assert len(repos) == 2
```

- [ ] **Step 2: 运行验证失败** → `python -m pytest hot_projects/tests/test_github_provider.py -v` → FAIL

- [ ] **Step 3: 实现骨架 + search_similar**

```python
# hot_projects/providers/github/provider.py
from ..base import Provider, Repo
from .api import search_github_repos, fetch_repo_info


class GitHubProvider(Provider):
    def __init__(self, token_mgr):
        self.token_mgr = token_mgr

    def search_similar(self, name: str, limit: int = 5):
        items = search_github_repos(self.token_mgr, name, token_idx=0, page=1,
                                    per_page=limit, min_star=0) or []
        return [Repo.from_github(it) for it in items[:limit]]

    def repo_info(self, repo: str):
        parts = repo.split("/", 1)
        if len(parts) != 2:
            return None
        item = fetch_repo_info(self.token_mgr, parts[0], parts[1], token_idx=0)
        return Repo.from_github(item) if item else None

    # 以下方法在 Phase 3/5 接线（调用 capabilities 纯函数）
    def search_by_keywords(self, categories, min_star, days_since_created): raise NotImplementedError
    def scan_star_range(self, min_star, max_star, seen_repos, days_since_created): raise NotImplementedError
    def repo_growth(self, repo, growth_calc_days): raise NotImplementedError
    def batch_growth(self, repos, db, **kwargs): raise NotImplementedError
    def fetch_trending(self, trending_range): raise NotImplementedError
```

- [ ] **Step 4: 验证通过 + Commit**

Run: `python -m pytest hot_projects/tests/test_github_provider.py -v` → PASS
```bash
cd /root/code/Agent-skils
git add hot_projects/providers/github/provider.py hot_projects/tests/test_github_provider.py
git commit -m "feat: GitHubProvider skeleton with search_similar/repo_info"
```

---

## Phase 3 — capabilities（基础工具层，纯函数）

### Task 3.1: 迁移 9 个 capability 函数

**Files:**
- Create: `hot_projects/capabilities/__init__.py`（聚合导出）
- Create: `hot_projects/capabilities/collect.py`（search_by_keywords / scan_star_range / fetch_trending / trending_repo_to_search_repo）
- Create: `hot_projects/capabilities/growth.py`（check_repo_growth / batch_check_growth）
- Create: `hot_projects/capabilities/rank.py`（rank_candidates）
- Create: `hot_projects/capabilities/describe.py`（describe_project / get_db_info / generate_report）

- [ ] **Step 1: 复制 agent_tools.py 函数体到对应模块**

把原 `github_hot_projects/agent_tools.py` 中的 9 个 `tool_*` 函数（去掉 `tool_` 前缀，作为纯函数）拆分到上述模块；保留实现逻辑，仅改：
- import 路径（`from .common.*` → 新路径；`from .tasks import ...` → `from ..infra.concurrency.*`；`from .ranking/report/growth_estimator` → 新路径；trending 引用 → `..providers.github.trending`）。
- 去掉 `validate_tool_args(...)` 内联校验？**保留**（这是参数边界裁剪，属算法防御，不动）。

**不改算法**：并发提交、页级补偿、`_raw_repos` 结构、增长三路、checkpoint 全部保持。

- [ ] **Step 2: 验证可导入**

Run:
```bash
cd /root/code/Agent-skils
python -c "from hot_projects.capabilities import collect, growth, rank, describe; print('CAP_OK')"
```
Expected: `CAP_OK`

- [ ] **Step 3: 接线 GitHubProvider 其余方法**

在 `providers/github/provider.py` 中把 `search_by_keywords/scan_star_range/repo_growth/batch_growth/fetch_trending` 实现为调用 capabilities 对应函数（透传 `self.token_mgr`），返回值维持现有 dict 结构（pipeline 直接消费）。

- [ ] **Step 4: 冒烟测试（mock 网络）**

```python
# hot_projects/tests/test_capabilities_smoke.py
def test_get_db_info_overview():
    from hot_projects.capabilities.describe import get_db_info
    db = {"valid": True, "date": "2026-06-10", "projects": {"a/b": {"star": 1}}}
    out = get_db_info(db=db, repo=None)
    assert out["total_projects"] == 1 and out["valid"] is True
```

Run: `python -m pytest hot_projects/tests/test_capabilities_smoke.py -v` → PASS

- [ ] **Step 5: Commit**

```bash
cd /root/code/Agent-skils
git add hot_projects/capabilities hot_projects/providers/github/provider.py hot_projects/tests/test_capabilities_smoke.py
git commit -m "feat: migrate base capabilities as pure functions + wire GitHubProvider"
```

---

## Phase 4 — pipeline（RankingCache + ranking_pipeline）

### Task 4.1: RankingCache（分阶段参数签名缓存）

**Files:**
- Create: `hot_projects/pipeline/cache.py`
- Test: `hot_projects/tests/test_ranking_cache.py`

阶段与依赖（来自 spec 第 5 节）：
```
collect      ← categories, min_star, max_star, days_since_created, sources
growth_calc  ← collect 输出 + growth_calc_days, days_since_created   (昂贵)
threshold    ← growth_calc 输出 + growth_threshold                   (廉价)
rank         ← threshold 输出 + mode, top_n, days_since_created       (廉价)
report       ← rank 输出 + 展示参数
```

- [ ] **Step 1: 写失败测试**

```python
# hot_projects/tests/test_ranking_cache.py
from hot_projects.pipeline.cache import RankingCache

def test_reuse_when_signature_unchanged():
    c = RankingCache()
    c.set("collect", {"min_star": 1200}, payload=["repoX"])
    assert c.get("collect", {"min_star": 1200}) == ["repoX"]

def test_invalidate_on_signature_change():
    c = RankingCache()
    c.set("collect", {"min_star": 1200}, payload=["repoX"])
    assert c.get("collect", {"min_star": 2000}) is None  # 参数变 → miss

def test_downstream_invalidated_when_upstream_changes():
    c = RankingCache()
    c.set("collect", {"min_star": 1200}, payload="C")
    c.set("growth_calc", {"min_star": 1200, "growth_calc_days": 7}, payload="G")
    # 上游 collect 变更后，downstream 取数应 miss
    c.set("collect", {"min_star": 2000}, payload="C2")
    assert c.get("growth_calc", {"min_star": 2000, "growth_calc_days": 7}) is None
```

- [ ] **Step 2: 运行失败** → `python -m pytest hot_projects/tests/test_ranking_cache.py -v` → FAIL

- [ ] **Step 3: 实现**

```python
# hot_projects/pipeline/cache.py
"""榜单分阶段缓存：按 (阶段, 参数签名) 缓存输出；上游变化使下游失效。"""
import json

STAGE_ORDER = ["collect", "growth_calc", "threshold", "rank", "report"]


def _sig(params: dict) -> str:
    return json.dumps(params, ensure_ascii=False, sort_keys=True, default=str)


class RankingCache:
    def __init__(self):
        self._store: dict[str, tuple[str, object]] = {}

    def get(self, stage: str, params: dict):
        entry = self._store.get(stage)
        if entry is None or entry[0] != _sig(params):
            return None
        return entry[1]

    def set(self, stage: str, params: dict, payload) -> None:
        self._store[stage] = (_sig(params), payload)
        # 失效所有下游阶段
        idx = STAGE_ORDER.index(stage)
        for downstream in STAGE_ORDER[idx + 1:]:
            self._store.pop(downstream, None)
```

- [ ] **Step 4: 运行通过 + Commit**

Run: `python -m pytest hot_projects/tests/test_ranking_cache.py -v` → PASS
```bash
cd /root/code/Agent-skils
git add hot_projects/pipeline/cache.py hot_projects/tests/test_ranking_cache.py
git commit -m "feat: RankingCache with stage signature reuse/invalidation"
```

---

### Task 4.2: ranking_pipeline（唯一榜单流水线）

**Files:**
- Create: `hot_projects/pipeline/ranking_pipeline.py`
- Test: `hot_projects/tests/test_ranking_pipeline.py`

承接 spec 10b 隐藏行为：综合榜未指定窗口用 DB 年龄窗口；持久化策略按 `force_refresh` 区分；`prefiltered_days_since_created` 透传。

- [ ] **Step 1: 写失败测试（用 fake provider，校验阶段顺序 + 缓存复用）**

```python
# hot_projects/tests/test_ranking_pipeline.py
from hot_projects.pipeline.ranking_pipeline import run_ranking
from hot_projects.pipeline.cache import RankingCache

class FakeProvider:
    def __init__(self): self.calls = []
    def search_by_keywords(self, **kw): self.calls.append("search"); return {"_raw_repos": [{"full_name": "a/b", "star": 1500, "_raw": {}}]}
    def scan_star_range(self, **kw): self.calls.append("scan"); return {"_raw_repos": []}
    def fetch_trending(self, trending_range): self.calls.append("trending"); return {"_raw_repos": []}
    def batch_growth(self, repos, db, **kw): self.calls.append("growth"); return {"candidates": {"a/b": {"growth": 900, "star": 1500, "created_at": ""}}, "growth_calc_days": 7}

def test_threshold_change_skips_collect_and_growth(monkeypatch):
    import hot_projects.pipeline.ranking_pipeline as P
    monkeypatch.setattr(P, "step2_rank_and_select", lambda cand, **kw: list(cand.items()))
    p = FakeProvider(); cache = RankingCache(); db = {"valid": False, "projects": {}}
    run_ranking(p, mode="comprehensive", params={"min_star": 1200, "growth_calc_days": 7, "growth_threshold": 800, "top_n": 10}, db=db, cache=cache, do_report=False)
    first = list(p.calls)
    run_ranking(p, mode="comprehensive", params={"min_star": 1200, "growth_calc_days": 7, "growth_threshold": 500, "top_n": 10}, db=db, cache=cache, do_report=False)
    # 仅阈值变化：不应再次 search/scan/trending/growth
    assert p.calls == first  # 没有新增昂贵调用
```

- [ ] **Step 2: 运行失败** → FAIL（`run_ranking` 未定义）

- [ ] **Step 3: 实现 run_ranking**

```python
# hot_projects/pipeline/ranking_pipeline.py
"""唯一榜单流水线：collect→growth_calc→threshold→rank→report，分阶段缓存。"""
import logging
from ..ranking import step2_rank_and_select
from ..report import step3_generate_report
from ..infra.db import get_db_age_days
from .cache import RankingCache

logger = logging.getLogger("hot_projects")


def _collect(provider, mode, params):
    repos: list[dict] = []
    seen: set[str] = set()
    sr = provider.search_by_keywords(categories=params.get("categories"),
                                     min_star=params["min_star"],
                                     days_since_created=params.get("days_since_created"))
    raw = sr.get("_raw_repos", [])
    repos.extend(raw); seen.update(r["full_name"] for r in raw)
    if mode != "keyword":
        scan = provider.scan_star_range(min_star=params["min_star"],
                                        max_star=params.get("max_star"),
                                        seen_repos=seen,
                                        days_since_created=params.get("days_since_created"))
        repos.extend(scan.get("_raw_repos", []))
        tr = provider.fetch_trending(trending_range="all")
        for r in tr.get("_raw_repos", []):
            if r["full_name"] not in seen:
                seen.add(r["full_name"]); repos.append(r)
    return repos


def run_ranking(provider, mode, params, db, cache: RankingCache | None = None,
                do_report=True, force_refresh=False):
    cache = cache or RankingCache()

    collect_sig = {"mode": mode, "min_star": params["min_star"], "max_star": params.get("max_star"),
                   "categories": params.get("categories"), "days_since_created": params.get("days_since_created")}
    repos = cache.get("collect", collect_sig)
    if repos is None:
        repos = _collect(provider, mode, params)
        cache.set("collect", collect_sig, repos)

    # 综合榜未指定窗口 → 用 DB 年龄窗口（隐藏行为#1）
    growth_calc_days = params.get("growth_calc_days")
    window_specified = growth_calc_days is not None
    if not window_specified and mode in ("comprehensive", "keyword"):
        age = get_db_age_days(db)
        if db.get("valid") and age and age > 0:
            growth_calc_days = age

    days_since = params.get("days_since_created")
    growth_sig = {**collect_sig, "growth_calc_days": growth_calc_days, "days_since_created": days_since}
    growth = cache.get("growth_calc", growth_sig)
    if growth is None:
        growth = provider.batch_growth(repos, db, growth_threshold=0,
                                       days_since_created=days_since,
                                       growth_calc_days=growth_calc_days or 7,
                                       force_refresh=force_refresh,
                                       window_specified=window_specified)
        cache.set("growth_calc", growth_sig, growth)

    # threshold 过滤（廉价）
    threshold = params.get("growth_threshold", 800)
    thr_sig = {**growth_sig, "growth_threshold": threshold}
    candidates = cache.get("threshold", thr_sig)
    if candidates is None:
        candidates = {k: v for k, v in growth.get("candidates", {}).items() if v["growth"] >= threshold}
        cache.set("threshold", thr_sig, candidates)

    # rank（廉价）
    rank_mode = "hot_new" if mode == "hot_new" else "comprehensive"
    rank_sig = {**thr_sig, "rank_mode": rank_mode, "top_n": params.get("top_n")}
    ranked = cache.get("rank", rank_sig)
    if ranked is None:
        ordered = step2_rank_and_select(candidates, mode=rank_mode, db=db,
                                        days_since_created=days_since,
                                        prefiltered_days_since_created=days_since)
        ranked = ordered[: params.get("top_n") or len(ordered)]
        cache.set("rank", rank_sig, ranked)

    result = {"ranked": ranked, "candidates_count": len(candidates), "mode": rank_mode,
              "growth_calc_days": growth.get("growth_calc_days", growth_calc_days)}

    if do_report:
        report_path = step3_generate_report(
            ranked, db, mode=rank_mode,
            days_since_created=days_since if rank_mode == "hot_new" else None,
            growth_calc_days=result["growth_calc_days"], growth_threshold=threshold,
            min_star=params["min_star"])
        result["report_path"] = report_path
    return result
```

> 持久化（隐藏行为#2）：Agent 路径调用方在 run_ranking 后按需 `save_db_desc_only(db)`；scheduled 用 `force_refresh=True` 并 `save_db(db)`。落点放在调用方（复合工具 / scheduled），保持 pipeline 纯。

- [ ] **Step 4: 运行通过 + Commit**

Run: `python -m pytest hot_projects/tests/test_ranking_pipeline.py -v` → PASS
```bash
cd /root/code/Agent-skils
git add hot_projects/pipeline/ranking_pipeline.py hot_projects/tests/test_ranking_pipeline.py
git commit -m "feat: unified ranking_pipeline with staged cache reuse"
```

---

## Phase 5 — tools（注册表 / schemas / 复合工具 / 原子工具）

### Task 5.1: 工具注册表

**Files:**
- Create: `hot_projects/tools/registry.py`
- Test: `hot_projects/tests/test_registry.py`

- [ ] **Step 1: 失败测试**

```python
# hot_projects/tests/test_registry.py
from hot_projects.tools.registry import ToolSpec, ToolRegistry

def test_register_and_dispatch():
    reg = ToolRegistry()
    reg.register(ToolSpec(name="echo", schema={}, param_schema={}, handler=lambda ctx, args: {"echo": args}, expensive=False))
    assert reg.get("echo").expensive is False
    assert reg.dispatch("echo", ctx=None, args={"x": 1}) == {"echo": {"x": 1}}

def test_unknown_tool():
    reg = ToolRegistry()
    out = reg.dispatch("nope", ctx=None, args={})
    assert "error" in out
```

- [ ] **Step 2: 失败** → FAIL

- [ ] **Step 3: 实现**

```python
# hot_projects/tools/registry.py
from dataclasses import dataclass
from typing import Callable


@dataclass
class ToolSpec:
    name: str
    schema: dict
    param_schema: dict
    handler: Callable
    expensive: bool = False


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict]:
        return [t.schema for t in self._tools.values()]

    def dispatch(self, name: str, ctx, args: dict):
        spec = self._tools.get(name)
        if spec is None:
            return {"error": f"未知 Tool: {name}"}
        return spec.handler(ctx, args)
```

- [ ] **Step 4: 通过 + Commit**

Run: `python -m pytest hot_projects/tests/test_registry.py -v` → PASS
```bash
git add hot_projects/tools/registry.py hot_projects/tests/test_registry.py
git commit -m "feat: tool registry with single dispatch"
```

---

### Task 5.2: schemas（复合工具 + 原子工具的 LLM schema 与参数校验）

**Files:**
- Create: `hot_projects/tools/schemas.py`

- [ ] **Step 1: 写 schemas**

定义暴露给 LLM 的 7 个工具 function-calling schema：`comprehensive_ranking`/`hot_new_ranking`/`keyword_ranking`（参数：categories/min_star/max_star/growth_calc_days/growth_threshold/days_since_created/top_n，按工具裁剪）、`repo_growth`(repo, growth_calc_days)、`describe_project`(repo)、`get_db_info`(repo?)、`fetch_trending`(trending_range)。描述中写明"昂贵榜单工具：执行前先回显参数等用户确认"。同时迁移原 `parsing/schema.py::TOOL_PARAM_SCHEMA` 中对应键的校验规则到这里，供 `validate_tool_args` 复用（复制 `parsing/arg_validator.py` 为 `hot_projects/tools/arg_validator.py`，改 import）。

- [ ] **Step 2: 验证导入**

Run: `python -c "from hot_projects.tools import schemas; print(len(schemas.TOOL_SCHEMAS))"`
Expected: `7`

- [ ] **Step 3: Commit**

```bash
git add hot_projects/tools/schemas.py hot_projects/tools/arg_validator.py
git commit -m "feat: tool schemas + param validation for hot_projects"
```

---

### Task 5.3: 复合榜单工具 + 幂等确认守卫

**Files:**
- Create: `hot_projects/tools/ranking_tools.py`
- Test: `hot_projects/tests/test_ranking_tools.py`

确认守卫（spec 6.1）：首次调用返回"请确认参数"并把参数签名写入 `ctx.state.pending_confirmation_signature`；用户确认后 LLM 同签名再调 → 执行。

- [ ] **Step 1: 失败测试**

```python
# hot_projects/tests/test_ranking_tools.py
from hot_projects.tools.ranking_tools import make_ranking_handler

class _State:
    def __init__(self): self.pending_confirmation_signature=None; self.ranking_cache=None
class _Ctx:
    def __init__(self): self.state=_State(); self.provider=None; self.db={"valid":False,"projects":{}}

def test_first_call_asks_confirmation(monkeypatch):
    ctx=_Ctx()
    handler=make_ranking_handler("comprehensive")
    out=handler(ctx, {"min_star":1200})
    assert out.get("needs_confirmation") is True
    assert ctx.state.pending_confirmation_signature is not None

def test_second_call_same_sig_executes(monkeypatch):
    import hot_projects.tools.ranking_tools as RT
    monkeypatch.setattr(RT, "run_ranking", lambda *a, **k: {"ranked": [], "report_path": "/x.md"})
    ctx=_Ctx()
    handler=make_ranking_handler("comprehensive")
    handler(ctx, {"min_star":1200})            # 第一次 → 确认
    out=handler(ctx, {"min_star":1200})        # 第二次同签名 → 执行
    assert "ranked" in out
```

- [ ] **Step 2: 失败** → FAIL

- [ ] **Step 3: 实现**

```python
# hot_projects/tools/ranking_tools.py
import json
from ..pipeline.ranking_pipeline import run_ranking
from ..pipeline.cache import RankingCache
from ..infra.db import save_db_desc_only


def _sig(mode, params):
    return json.dumps({"mode": mode, **params}, ensure_ascii=False, sort_keys=True, default=str)


def make_ranking_handler(mode: str):
    def handler(ctx, args: dict):
        params = dict(args)
        sig = _sig(mode, params)
        if ctx.state.pending_confirmation_signature != sig:
            ctx.state.pending_confirmation_signature = sig
            return {"needs_confirmation": True, "mode": mode, "params": params,
                    "message": f"将执行[{mode}]榜单，参数={params}。回复『开始』确认执行。"}
        ctx.state.pending_confirmation_signature = None
        if ctx.state.ranking_cache is None:
            ctx.state.ranking_cache = RankingCache()
        result = run_ranking(ctx.provider, mode=mode, params=params, db=ctx.db,
                             cache=ctx.state.ranking_cache, do_report=True, force_refresh=False)
        save_db_desc_only(ctx.db)
        return {"ranked_count": len(result.get("ranked", [])),
                "report_path": result.get("report_path", ""),
                "ranked": [{"rank": i + 1, "repo": n, "growth": v["growth"], "star": v["star"]}
                           for i, (n, v) in enumerate(result.get("ranked", []))]}
    return handler
```

- [ ] **Step 4: 通过 + Commit**

Run: `python -m pytest hot_projects/tests/test_ranking_tools.py -v` → PASS
```bash
git add hot_projects/tools/ranking_tools.py hot_projects/tests/test_ranking_tools.py
git commit -m "feat: composite ranking tools with idempotent confirmation guard"
```

---

### Task 5.4: 原子工具 + 单仓库模糊消歧

**Files:**
- Create: `hot_projects/tools/atomic_tools.py`
- Test: `hot_projects/tests/test_atomic_tools.py`

模糊消歧（spec 6.2）：`repo_growth`/`describe_project` 先精确查；查不到 → `provider.search_similar` 返回候选给 LLM。

- [ ] **Step 1: 失败测试**

```python
# hot_projects/tests/test_atomic_tools.py
from hot_projects.tools.atomic_tools import repo_growth_handler

class _Prov:
    def __init__(self, info=None, similar=None): self._info=info; self._similar=similar or []
    def repo_info(self, repo): return self._info
    def repo_growth(self, repo, growth_calc_days): return {"repo": repo, "growth": 123}
    def search_similar(self, name, limit=5): return self._similar

class _Ctx:
    def __init__(self, prov): self.provider=prov; self.db={"projects":{}}
    class state: active_repo=None

def test_exact_hit_returns_growth():
    from hot_projects.providers.base import Repo
    ctx=_Ctx(_Prov(info=Repo("a/b",1500)))
    out=repo_growth_handler(ctx, {"repo":"a/b"})
    assert out["growth"]==123

def test_miss_returns_candidates():
    from hot_projects.providers.base import Repo
    ctx=_Ctx(_Prov(info=None, similar=[Repo("x/vllm",1300), Repo("y/vllm",1200)]))
    out=repo_growth_handler(ctx, {"repo":"vllm"})
    assert out.get("disambiguation") is True
    assert "x/vllm" in [c["full_name"] for c in out["candidates"]]
```

- [ ] **Step 2: 失败** → FAIL

- [ ] **Step 3: 实现**

```python
# hot_projects/tools/atomic_tools.py
from ..capabilities.describe import get_db_info as _get_db_info


def _disambig(ctx, raw_name):
    cands = ctx.provider.search_similar(raw_name, limit=5)
    if not cands:
        return {"error": f"未找到仓库: {raw_name}，也没有相似项目。"}
    return {"disambiguation": True,
            "message": f"没找到 {raw_name}，你是不是指以下之一？请回复完整 owner/repo。",
            "candidates": [{"full_name": c.full_name, "star": c.star, "desc": c.description} for c in cands]}


def repo_growth_handler(ctx, args: dict):
    repo = (args.get("repo") or "").strip()
    if ctx.provider.repo_info(repo) is None:
        return _disambig(ctx, repo)
    ctx.state.active_repo = repo
    return ctx.provider.repo_growth(repo, growth_calc_days=args.get("growth_calc_days", 7))


def describe_project_handler(ctx, args: dict):
    repo = (args.get("repo") or "").strip()
    if ctx.provider.repo_info(repo) is None:
        return _disambig(ctx, repo)
    ctx.state.active_repo = repo
    from ..capabilities.describe import describe_project
    return describe_project(repo=repo, db=ctx.db, token_mgr=ctx.provider.token_mgr)


def get_db_info_handler(ctx, args: dict):
    return _get_db_info(db=ctx.db, repo=args.get("repo"))


def fetch_trending_handler(ctx, args: dict):
    return ctx.provider.fetch_trending(trending_range=args.get("trending_range", "weekly"))
```

- [ ] **Step 4: 通过 + Commit**

Run: `python -m pytest hot_projects/tests/test_atomic_tools.py -v` → PASS
```bash
git add hot_projects/tools/atomic_tools.py hot_projects/tests/test_atomic_tools.py
git commit -m "feat: atomic tools with single-repo fuzzy disambiguation"
```

---

### Task 5.5: 组装注册表（build_registry）

**Files:**
- Modify: `hot_projects/tools/registry.py`（加 `build_default_registry`）

- [ ] **Step 1: 实现 build_default_registry**

把 5.2 的 schemas + 5.3 复合 handler + 5.4 原子 handler 组装为注册表；复合工具 `expensive=True`，原子工具 `expensive=False`；handler 统一签名 `(ctx, args)`；复合 handler 用 `make_ranking_handler(mode)`。

- [ ] **Step 2: 验证**

Run: `python -c "from hot_projects.tools.registry import build_default_registry; r=build_default_registry(); print(len(r.schemas()))"`
Expected: `7`

- [ ] **Step 3: Commit**

```bash
git add hot_projects/tools/registry.py
git commit -m "feat: assemble default tool registry (composite + atomic)"
```

---

## Phase 6 — agent（精简 ReAct + state + prompts）

### Task 6.1: AgentState（含 RankingCache）

**Files:**
- Create: `hot_projects/agent/state.py`
- Test: `hot_projects/tests/test_agent_state.py`

- [ ] **Step 1: 失败测试**

```python
# hot_projects/tests/test_agent_state.py
from hot_projects.agent.state import AgentState

def test_state_defaults():
    s = AgentState(db={"projects": {}})
    assert s.ranking_cache is not None
    assert s.active_repo is None
    assert s.pending_confirmation_signature is None
    assert isinstance(s.conversation, list)
```

- [ ] **Step 2: 失败** → FAIL

- [ ] **Step 3: 实现**

```python
# hot_projects/agent/state.py
from dataclasses import dataclass, field
from ..pipeline.cache import RankingCache

MAX_CONVERSATION_MESSAGES = 40
KEEP_RECENT_MESSAGES = 10


@dataclass
class AgentState:
    db: dict = field(default_factory=dict)
    conversation: list[dict] = field(default_factory=list)
    conversation_summary: str = ""
    ranking_cache: RankingCache = field(default_factory=RankingCache)
    active_repo: str | None = None
    pending_confirmation_signature: str | None = None
```

- [ ] **Step 4: 通过 + Commit**

Run: `python -m pytest hot_projects/tests/test_agent_state.py -v` → PASS
```bash
git add hot_projects/agent/state.py hot_projects/tests/test_agent_state.py
git commit -m "feat: slim AgentState with RankingCache"
```

---

### Task 6.2: prompts（system prompt）

**Files:**
- Create: `hot_projects/agent/prompts.py`

- [ ] **Step 1: 写 SYSTEM_PROMPT**

内容要点：ReAct 工作方式；事实数据（star/增长/创建时间/Trending）必须调工具核查不得编造；昂贵榜单工具执行前先回显参数并等用户『开始』；growth_calc_days(增长窗口) 与 days_since_created(创建窗口) 独立；单仓库查不到时利用返回的候选询问用户。

- [ ] **Step 2: 验证导入** → `python -c "from hot_projects.agent.prompts import SYSTEM_PROMPT; print(bool(SYSTEM_PROMPT))"` → `True`

- [ ] **Step 3: Commit**

```bash
git add hot_projects/agent/prompts.py
git commit -m "feat: agent system prompt"
```

---

### Task 6.3: agent.py（精简 ReAct 循环）

**Files:**
- Create: `hot_projects/agent/agent.py`
- Test: `hot_projects/tests/test_agent.py`

无路由 LLM、无白名单、无前置校验、无确认状态机；对话压缩 + 工具结果截断沿用（从原 agent 抽取这两个纯函数到 `agent/agent.py` 或 `agent/util.py`）。

- [ ] **Step 1: 失败测试（mock LLMClient + registry）**

```python
# hot_projects/tests/test_agent.py
from hot_projects.agent.agent import HotProjectAgent
from hot_projects.tools.registry import ToolSpec, ToolRegistry

def _llm_toolcall_then_text():
    seq = [
        {"choices": [{"message": {"content": None, "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "get_db_info", "arguments": "{}"}}]}}]},
        {"choices": [{"message": {"content": "数据库共有 1 个项目。"}}]},
    ]
    def chat(messages, **kw): return seq.pop(0)
    class C: pass
    c = C(); c.chat = chat
    return c

def test_react_executes_tool_then_replies():
    reg = ToolRegistry()
    reg.register(ToolSpec("get_db_info", {"type": "function", "function": {"name": "get_db_info", "parameters": {"type": "object", "properties": {}}}}, {}, lambda ctx, args: {"total_projects": 1}, False))
    agent = HotProjectAgent(llm=_llm_toolcall_then_text(), registry=reg, provider=None, db={"projects": {}})
    reply = agent.chat("数据库里有多少项目")
    assert "1" in reply
```

- [ ] **Step 2: 失败** → FAIL

- [ ] **Step 3: 实现核心 ReAct 循环**

```python
# hot_projects/agent/agent.py
import json
import logging
from dataclasses import dataclass
from .state import AgentState, MAX_CONVERSATION_MESSAGES
from .prompts import SYSTEM_PROMPT
from ..tools.arg_validator import validate_tool_args_strict

logger = logging.getLogger("hot_projects")
MAX_TOOL_CALLS_PER_TURN = 15


@dataclass
class ToolContext:
    state: AgentState
    provider: object
    db: dict


class HotProjectAgent:
    def __init__(self, llm, registry, provider, db):
        self.llm = llm
        self.registry = registry
        self.state = AgentState(db=db)
        self.ctx = ToolContext(state=self.state, provider=provider, db=db)
        self.state.conversation.append({"role": "system", "content": SYSTEM_PROMPT})

    def chat(self, user_message: str) -> str:
        if len(user_message) > 2000:
            return "消息过长（超过 2000 字符），请缩短后重试。"
        self.state.conversation.append({"role": "user", "content": user_message})
        for _ in range(MAX_TOOL_CALLS_PER_TURN):
            resp = self.llm.chat(list(self.state.conversation), tools=self.registry.schemas())
            if resp is None:
                msg = "抱歉，LLM 调用失败，请稍后重试。"
                self.state.conversation.append({"role": "assistant", "content": msg})
                return msg
            message = (resp.get("choices") or [{}])[0].get("message", {})
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                content = message.get("content") or "（未生成回复，请重试或换个问法。）"
                self.state.conversation.append({"role": "assistant", "content": content})
                return content
            self.state.conversation.append({"role": "assistant", "content": message.get("content"),
                                            "tool_calls": tool_calls})
            for tc in tool_calls:
                name = tc.get("function", {}).get("name", "")
                raw = tc.get("function", {}).get("arguments", "{}")
                try:
                    args = json.loads(raw) if raw else {}
                    if not isinstance(args, dict):
                        raise ValueError
                except (json.JSONDecodeError, ValueError):
                    result = {"error": "tool arguments 非法 JSON object"}
                else:
                    _, errs = validate_tool_args_strict(name, args)
                    if errs:
                        result = {"error": "参数校验失败", "invalid_arguments": errs, "retryable": True}
                    else:
                        try:
                            result = self.registry.dispatch(name, self.ctx, args)
                        except Exception as e:  # noqa
                            logger.error("[Agent] Tool %s 异常: %s", name, e)
                            result = {"error": f"工具执行异常: {e}"}
                self.state.conversation.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                                                "content": _serialize_result(result)})
        return "已达到单轮最大 Tool 调用次数，请尝试简化请求。"


def _serialize_result(result: dict, max_len: int = 8000) -> str:
    s = json.dumps(result, ensure_ascii=False, default=str)
    return s if len(s) <= max_len else json.dumps({"truncated": True, "preview": s[:max_len]}, ensure_ascii=False)
```

> 对话压缩：从原 `agent._compress_conversation` / `_generate_summary_with_llm` 抽取移植为本模块函数，在 `chat` 开头检查 `len(conversation) > MAX_CONVERSATION_MESSAGES` 时调用（用 `self.llm.chat(..., lite=True)` 生成摘要）。本步可作为 6.4 补充任务。

- [ ] **Step 4: 通过 + Commit**

Run: `python -m pytest hot_projects/tests/test_agent.py -v` → PASS
```bash
git add hot_projects/agent/agent.py hot_projects/tests/test_agent.py
git commit -m "feat: slim ReAct agent loop (no router LLM / no gate / no whitelist)"
```

---

### Task 6.4: 对话压缩移植

**Files:**
- Modify: `hot_projects/agent/agent.py`
- Test: `hot_projects/tests/test_agent_compress.py`

- [ ] **Step 1..5**：移植压缩逻辑（用 lite 模型生成摘要），写测试覆盖"超过阈值触发压缩、保留最近 N 条 + system"，运行通过，commit `feat: conversation compression in agent`。

---

## Phase 7 — 入口层

### Task 7.1: agent 工厂 + agent_cli

**Files:**
- Create: `hot_projects/agent/__init__.py`（`build_agent()` 工厂：组装 LLMClient + registry + GitHubProvider + token_mgr + db）
- Create: `hot_projects/agent_cli.py` ← 改自原 `agent_cli.py`（import 改为 `build_agent`）

- [ ] **Step 1:** 实现 `build_agent()`：
```python
def build_agent():
    from ..infra.llm_client import client_from_config
    from ..infra.db import load_db
    from ..providers.github.token_pool import GitHubTokenPool
    from ..providers.github.provider import GitHubProvider
    from ..tools.registry import build_default_registry
    from .agent import HotProjectAgent
    tm = GitHubTokenPool(); db = load_db()
    return HotProjectAgent(llm=client_from_config(), registry=build_default_registry(),
                           provider=GitHubProvider(tm), db=db)
```
- [ ] **Step 2:** agent_cli 调 `build_agent()`，REPL 不变。
- [ ] **Step 3:** 手动冒烟（需 token/key）：`python -m hot_projects.agent_cli` 输入"数据库里有多少项目"。
- [ ] **Step 4:** Commit `feat: agent factory + CLI entrypoint`.

---

### Task 7.2: scheduled_update（调用统一 pipeline）

**Files:**
- Create: `hot_projects/scheduled_update.py` ← 改自原；`DiscoveryPipeline.run` 改为调用 `run_ranking(provider, mode="comprehensive", params=..., db, cache=RankingCache(), do_report=True, force_refresh=True)` 后 `save_db(db)`；保留 `log_update_summary`。

- [ ] **Step 1..3:** 实现、`python -m hot_projects.scheduled_update --top-n 5`（mock 或小范围）冒烟、commit `feat: scheduled_update via unified pipeline`.

---

### Task 7.3: api_server + __main__

**Files:**
- Create: `hot_projects/api_server.py` ← 改自原（`get_agent` 用 `build_agent()`；其余 REST/WS/安全中间件/报告渲染**不变**）
- Create: `hot_projects/__main__.py` ← `from .api_server import main`

- [ ] **Step 1:** 复制 api_server.py，改 import（`from .agent import build_agent`），`HotProjectAgent()` 替换为 `build_agent()`。
- [ ] **Step 2:** 验证 `python -c "import hot_projects.api_server"` 通过。
- [ ] **Step 3:** Commit `feat: api server + package entrypoint`.

---

## Phase 8 — 测试与端到端验证

### Task 8.1: 复制并适配下层测试

**Files:**
- Create: `hot_projects/tests/` 下迁移下层稳定测试（`test_ranking/test_report/test_growth/test_common(db 部分)/test_async_token_pool/test_async_worker_pool/test_trending`），改 import 路径指向新模块。

- [ ] **Step 1:** 复制相关测试文件，改 import。
- [ ] **Step 2:** Run `cd /root/code/Agent-skils && python -m pytest hot_projects/tests -v`
Expected: 全绿。
- [ ] **Step 3:** Commit `test: port lower-layer tests to hot_projects`.

### Task 8.2: 全量回归 + requirements

- [ ] **Step 1:** `cp github_hot_projects/requirements.txt hot_projects/requirements.txt`
- [ ] **Step 2:** Run `cd /root/code/Agent-skils && python -m pytest hot_projects/tests -v` → 全绿
- [ ] **Step 3:** 设置真实 env（A/B key）后端到端冒烟：
  - `python -m hot_projects.agent_cli` →「查一下 vllm-project/vllm 的增长」（验证原子工具 + provider）
  -「跑个综合榜 top 5」→ 应回显参数等确认 →「开始」→ 产出报告（验证复合工具 + 确认守卫 + pipeline）
  -「增长阈值降到 500 再看看」→ 不重新 search（验证缓存复用）
- [ ] **Step 4:** Commit `chore: add requirements and finalize hot_projects`.

---

## Self-Review（计划完成后自检结论）

- **Spec 覆盖**：分层(Phase0-7)、复合工具(5.3)、注册表(5.1)、分阶段缓存(4.1/4.2)、确认守卫(5.3)、模糊消歧(5.4)、A/B 双 LLM 逐调用回退(1.1/1.2)、Provider 边界(2.x)、单 ReAct(6.3)、scheduled 统一 pipeline(7.2)、下层复制不改逻辑(0.x)、TDD 与下层测试保绿(8.x) 均有对应 Task。
- **隐藏行为**：DB 年龄窗口(4.2)、持久化策略按调用方(4.2 注 + 5.3 + 7.2)、prefiltered 透传(4.2) 已覆盖。
- **不在本次范围**：spec 10c 基础计算缺口(B2/B4/B5/C1/C2) 不在任何 Task —— 符合既定决策（单独立项）。
- **类型一致**：`ToolContext(state/provider/db)`、`run_ranking(provider,mode,params,db,cache,do_report,force_refresh)`、`LLMClient.chat(messages,tools,lite,...)`、`Repo` 字段在各 Task 间保持一致。

---

## Execution Handoff

计划见本文件。两种执行方式：
1. **Subagent-Driven（推荐）**：每个 Task 派新 subagent，任务间评审。
2. **Inline Execution**：本会话内分批执行 + 检查点。
