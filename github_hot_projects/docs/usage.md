# GitHub 热门项目发现 — 使用指南

## 1. 环境准备

```bash
pip install -r requirements.txt
```

通过环境变量配置密钥（必须）：
```bash
export GITHUB_TOKENS="ghp_token1,ghp_token2"   # 多个用逗号分隔
export LLM_API_KEY="sk-xxx"                     # LLM 接口密钥（不配置则跳过描述生成）
```

## 2. 管道模式（一键批处理）

```bash
python -m github_hot_projects
```

自动执行完整流水线，输出 `report/YYYY-MM-DD.md`。适合 cron 定时任务。

## 3. Agent 对话模式

```bash
python -m github_hot_projects.agent_cli
```

### 3.1 一键执行完整流程

```
你> 帮我跑一次完整的热门项目发现
Agent> [full_discovery] 发现 80 个热门项目，报告已保存到 report/2026-04-10.md
```

### 3.2 按类别搜索

支持 25 个类别关键词。可指定一个或多个类别：

```
你> 搜一下 AI Agent 方向的热门项目
Agent> [search_hot_projects(categories=["AI-Agent"])] 找到 47 个仓库

你> 再加上推理和训练方向
Agent> [search_hot_projects(categories=["AI-Inference-Serving", "AI-Training-Finetune"])]
      新增 63 个仓库

你> 搜一下数据库和云原生
Agent> [search_hot_projects(categories=["Database", "Cloud-Native"])] ...
```

可用类别：AI-Agent, AI-MCP, AI-Skill-Prompt-Workflow, AI-CLI-DevTool, AI-LLM-Core, AI-RAG, AI-Inference-Serving, AI-Training-Finetune, AI-Infra, AI-Multimodal, AI-Observability, AI-Data-Synthetic, AI-Edge-OnDevice, Database, Cloud-Native, Frontend, Backend, DevOps, Security, Data-Engineering, System-Tool, Programming-Language 等。

### 3.3 Star 范围扫描

指定 star 区间扫描全量仓库：

```
你> 扫描 5000-20000 star 的仓库
Agent> [scan_star_range(min_star=5000, max_star=20000)] 找到 312 个仓库

你> 只看低 star 区间
Agent> [scan_star_range(min_star=1000, max_star=3000)] ...
```

### 3.4 增长计算

单仓库或批量计算近 10 天（`TIME_WINDOW_DAYS`）的 star 增长：

```
你> 查一下 vllm-project/vllm 最近的 star 增长
Agent> [check_repo_growth(repo="vllm-project/vllm")]
      vllm: 当前 42350 star, 近 10 天增长 1820

你> 计算候选池里所有仓库的增长
Agent> [batch_check_growth] 312 个仓库中 45 个近 10 天增长 >= 800

你> 把增长阈值降到 300 再看看
Agent> [batch_check_growth(growth_threshold=300)] 增长 >= 300 的有 89 个
```

### 3.5 评分排序

两种模式：

**综合排名**（默认）— 综合增长量和增长率，新项目平滑折扣：
```
你> 排个名看看
Agent> [rank_candidates(mode="comprehensive")] Top 10:
      1. xxx/yyy — growth: 3200, score: 8734
      ...
```

**新项目排名** — 仅看创建时间 ≤ 45 天的新项目，按增长量排序：
```
你> 最近有什么新冒出来的爆款？
Agent> [rank_candidates(mode="hot_new")] 筛选到 8 个新项目:
      1. xxx/yyy — 创建于 15 天前, growth: 5200
      ...

你> 只看前 20 个
Agent> [rank_candidates(top_n=20)] ...
```

### 3.6 GitHub Trending

直接查看 Trending 或作为候选补充：

```
你> 看看 GitHub Trending 上有什么
Agent> [fetch_trending(since="daily")] 今日 Trending:
      1. xxx/yyy ⭐ 12345 (+890 today)
      ...

你> 看看本周的 Python 热门
Agent> [fetch_trending(since="weekly", language="python")] ...

你> 看看中文社区的月度趋势
Agent> [fetch_trending(since="monthly", spoken_language="zh")] ...
```

### 3.7 项目描述与报告

```
你> 给 langchain-ai/langchain 生成详细描述
Agent> [describe_project(repo="langchain-ai/langchain")]
      LangChain 是一个用于构建 LLM 应用的开源框架...（200-400 字）

你> 生成完整报告
Agent> [generate_report] 报告已保存到 report/2026-04-10.md
```

### 3.8 数据库查询

```
你> 数据库里有多少仓库？
Agent> [get_db_info] DB 日期: 2026-04-08, 有效: true, 仓库数: 1234

你> 查一下 DB 里 openai/gpt-4 的信息
Agent> [get_db_info(repo="openai/gpt-4")]
      star: 28000, language: Python, topics: [llm, gpt]
```

### 3.9 组合工作流示例

**场景：只看 AI 方向新项目 Top 20**
```
你> 搜一下所有 AI 方向的仓库
Agent> [search_hot_projects(categories=["AI-Agent","AI-LLM-Core","AI-RAG","AI-Inference-Serving"])]

你> 计算增长
Agent> [batch_check_growth]

你> 按新项目排名，取前 20
Agent> [rank_candidates(mode="hot_new", top_n=20)]

你> 生成报告
Agent> [generate_report]
```

**场景：对比某个仓库的增长**
```
你> 查一下 huggingface/transformers 和 vllm-project/vllm 的增长
Agent> [check_repo_growth] transformers: +1200, vllm: +1820
```

## 4. API 服务模式

```bash
python -m github_hot_projects.api_server   # 默认 0.0.0.0:8000
```

### 对话

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user1", "message": "搜一下 AI Agent 方向的热门项目"}'
# → {"session_id": "user1", "reply": "找到 47 个仓库..."}
```

### 报告管理

```bash
curl http://localhost:8000/api/reports                   # 报告列表
curl http://localhost:8000/api/reports/2026-04-10.md     # 报告详情
```

### 会话管理

```bash
curl http://localhost:8000/api/status                          # 服务状态
curl -X DELETE http://localhost:8000/api/sessions/user1        # 清除会话
```

## 5. 输出文件

| 文件 | 位置 | 说明 |
|------|------|------|
| 报告 | `report/YYYY-MM-DD.md` | Markdown 格式热门项目排行 |
| 数据库 | `Github_DB.json` | 仓库历史数据（star、描述等） |
| 日志 | `logs/discover_hot_projects.log` | 管道模式日志 |
| 日志 | `logs/agent.log` | Agent 模式日志 |

建议运行频率：每 7-10 天一次，保持 DB 数据新鲜（过期阈值 11 天）。
