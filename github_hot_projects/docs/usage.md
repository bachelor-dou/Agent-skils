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

## 2. Agent 对话模式

```bash
python -m github_hot_projects.agent_cli
```

通过自然语言触发多步 Tool 工作流，适合交互式探索和按需调整参数。

### 2.1 组合执行完整流程

```
你> 帮我跑一次完整的热门项目发现
Agent> [search_hot_projects] 搜索全部类别...
Agent> [scan_star_range] star 范围扫描...
Agent> [fetch_trending(include_all_periods=true)] Trending 日/周/月去重补充...
Agent> [batch_check_growth] 计算增长...
Agent> [rank_candidates(mode="comprehensive")] 返回热门榜...
Agent> [generate_report] 报告已保存到 report/2026-04-10.md

你> 查一下近期GitHub热门榜前50
Agent> [search_hot_projects] ...
Agent> [scan_star_range] ...
Agent> [fetch_trending(include_all_periods=true)] ...
Agent> [batch_check_growth] ...
Agent> [rank_candidates(top_n=50)] 发现 50 个热门项目...

你> 跑一次完整流程，增长阈值降到300
Agent> [search_hot_projects] ...
Agent> [scan_star_range] ...
Agent> [fetch_trending(include_all_periods=true)] ...
Agent> [batch_check_growth(growth_threshold=300)] ...
Agent> [rank_candidates] ...
```

### 2.2 按类别搜索

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

### 2.3 Star 范围扫描

指定 star 区间扫描全量仓库：

```
你> 扫描 5000-20000 star 的仓库
Agent> [scan_star_range(min_star=5000, max_star=20000)] 找到 312 个仓库

你> 只看低 star 区间
Agent> [scan_star_range(min_star=1000, max_star=3000)] ...
```

### 2.4 增长计算

单仓库查询返回实时详情（当前 star、近期增长、语言、简介、LLM 描述）：

```
你> 查一下 vllm-project/vllm 的情况
Agent> [check_repo_growth(repo="vllm-project/vllm")]
      vllm: 当前 42350 star, 近 10 天增长 1820, Python
      描述: vLLM 是一个高性能 LLM 推理引擎...（200-400字）

你> 计算候选池里所有仓库的增长
Agent> [batch_check_growth] 312 个仓库中 45 个近 10 天增长 >= 800

你> 把增长阈值降到 300 再看看
Agent> [batch_check_growth(growth_threshold=300)] 增长 >= 300 的有 89 个
```

### 2.5 评分排序

两种模式：

**综合排名**（默认）— 综合增长量和增长率，新项目平滑折扣：
```
你> 排个名看看
Agent> [rank_candidates(mode="comprehensive")] Top 10:
      1. xxx/yyy — growth: 3200, score: 8734
      ...
```

**新项目排名** — 筛选创建时间在指定窗口内的新项目，按增长量排序：
```
你> 最近有什么新冒出来的爆款？
Agent> [rank_candidates(mode="hot_new")] 筛选到 8 个新项目:
      1. xxx/yyy — 创建于 15 天前, growth: 5200
      ...

你> 看看近一个月的新项目排名，前20个
Agent> [rank_candidates(mode="hot_new", top_n=20, new_project_days=30)] ...

你> 只看前 20 个
Agent> [rank_candidates(top_n=20)] ...
```

### 2.6 GitHub Trending

直接查看 Trending 或作为候选补充：默认直接查看返回 weekly；完整补源流程会抓取 daily、weekly、monthly 三档并去重。

```
你> 看看 GitHub Trending 上有什么
Agent> [fetch_trending(since="weekly")] 本周 Trending:
      1. xxx/yyy ⭐ 12345 (+890 today)
      ...

你> 看看本周的 Python 热门
Agent> [fetch_trending(since="weekly", language="python")] ...

你> 看看中文社区的月度趋势
Agent> [fetch_trending(since="monthly", spoken_language="zh")] ...

你> 跑完整热门流程时把 Trending 也补全
Agent> [fetch_trending(include_all_periods=true)] 汇总 daily / weekly / monthly 去重 Trending 仓库...
```

### 2.7 项目描述与报告

```
你> 给 langchain-ai/langchain 生成详细描述
Agent> [describe_project(repo="langchain-ai/langchain")]
      LangChain 是一个用于构建 LLM 应用的开源框架...（200-400 字）

你> 生成完整报告
Agent> [generate_report] 报告已保存到 report/2026-04-10.md
```

### 2.8 数据库查询

```
你> 数据库里有多少仓库？
Agent> [get_db_info] DB 日期: 2026-04-08, 有效: true, 仓库数: 1234

你> 查一下 DB 里 openai/gpt-4 的信息
Agent> [get_db_info(repo="openai/gpt-4")]
      star: 28000, language: Python, topics: [llm, gpt]
```

### 2.9 组合工作流示例

**场景：查询近期GitHub项目热门榜前50**
```
你> 查一下近期GitHub项目热门榜前50
Agent> [search_hot_projects] 搜索全部类别...
Agent> [scan_star_range] star 范围扫描...
Agent> [fetch_trending(include_all_periods=true)] Trending 日/周/月去重补充...
Agent> [batch_check_growth] 计算增长...
Agent> [rank_candidates(top_n=50)] Top 50 热门项目:
      1. xxx/yyy — growth: 5200, star: 28000
      ...
```

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

**场景：近一个月的新项目热度排名**
```
你> 近一个月有什么新项目比较火？前20个
Agent> [search_hot_projects] ...
Agent> [scan_star_range] ...
Agent> [fetch_trending(include_all_periods=true)] ...
Agent> [batch_check_growth] ...
Agent> [rank_candidates(mode="hot_new", top_n=20, new_project_days=30)]
      1. xxx/yyy — 创建于 12 天前, growth: 3800
      ...
```

**场景：对比某个仓库的增长**
```
你> 查一下 huggingface/transformers 和 vllm-project/vllm 的增长
Agent> [check_repo_growth] transformers: +1200, vllm: +1820
```

## 3. API 服务模式

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

## 4. 输出文件

| 文件 | 位置 | 说明 |
|------|------|------|
| 报告 | `report/YYYY-MM-DD.md` | Markdown 格式热门项目排行 |
| 数据库 | `Github_DB.json` | 仓库历史数据（star、描述等） |
| 日志 | `logs/agent-YYYY-MM-DD.log` | Agent CLI 按执行日期追加写入的日志 |

建议运行频率：每 7-10 天一次，保持 DB 数据新鲜（过期阈值 11 天）。
