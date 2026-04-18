# GitHub 热门项目发现 — 使用指南

## 快速启动

### 1. 准备环境变量（bashrc 或当前 shell）

如果你已经把变量写在 `~/.bashrc`，先执行一次 `source ~/.bashrc`，或者直接开一个新终端再启动服务。

也可以只在当前终端临时导出：

```bash
export GITHUB_TOKENS="ghp_token1,ghp_token2"
export LLM_API_KEY="sk-xxx"
```

### 2. 安装依赖

```bash
cd /root/code/Agent-skils
pip install -r github_hot_projects/requirements.txt
```

### 3. 后台启动（nohup，推荐）

```bash
cd /root/code/Agent-skils
nohup python -m github_hot_projects >> github_hot_projects/logs/server.log 2>&1 &
```

修改代码重启：

```bash
pkill -f "python -m github_hot_projects" || true; cd /root/code/Agent-skils && nohup python -m github_hot_projects >> github_hot_projects/logs/server.log 2>&1 &
```
### 其它入口

```bash
python -m github_hot_projects.agent_cli  # 终端交互模式
```


### 4. 查看日志 / 停止服务

```bash
tail -f github_hot_projects/logs/server.log
pkill -f "python -m github_hot_projects"
```

### 5. 前台调试

```bash
cd /root/code/Agent-skils
python -m github_hot_projects
```

### 6. 访问

```
手机/浏览器:  http://你的公网IP:8000
本机:        http://127.0.0.1:8000
```

> **安全组**：阿里云 ECS 控制台 → 安全组 → 入方向 → 添加规则：端口 8000/TCP，授权对象 0.0.0.0/0



## 入口脚本职责

| 命令 | 对应入口 | 职责 |
|------|----------|------|
| `python -m github_hot_projects` | `__main__.py` → `api_server.main()` | 默认 Web/API 服务入口 |
| `python -m github_hot_projects.api_server` | `api_server.py` | 直接启动 FastAPI 服务，等价于默认入口 |
| `python -m github_hot_projects.agent_cli` | `agent_cli.py` | 终端 REPL，对话调试 Agent |
| `python -m github_hot_projects.scheduled_update` | `scheduled_update.py` | 定时批处理：搜索、增长计算、排名、生成日报 |
| `python -m github_hot_projects.regenerate_report --log ...` | `regenerate_report.py` | 从历史日志恢复候选并重建指定日期报告 |

Web 页面相关资源已经统一放在 `github_hot_projects/web/`：

- `chat.html` / `chat.css`：聊天页静态资源
- `report.html` / `report.css` / `report.js`：报告页模板与样式脚本

`api_server.py` 现在只负责路由、Markdown 渲染和数据注入，不再内嵌整页 HTML。

## Agent 对话示例

网页端和终端(`python -m github_hot_projects.agent_cli`)均支持自然语言指令。

### 完整流程

```
你> 帮我跑一次完整的热门项目发现
Agent> [search_hot_projects] → [scan_star_range] → [fetch_trending] → [batch_check_growth] → [rank_candidates] → [generate_report]
      报告已保存到 report/2026-04-10.md

你> 查一下近期GitHub热门榜前50
Agent> 同上流程 → [rank_candidates(top_n=50)]
```

### 按类别搜索

支持 25+ 类别关键词：AI-Agent, AI-MCP, AI-LLM-Core, AI-RAG, AI-Inference-Serving, AI-Training-Finetune, Database, Cloud-Native, Frontend, Backend, DevOps, Security 等。

```
你> 搜一下 AI Agent 方向的热门项目        → [search_hot_projects(categories=["AI-Agent"])]
你> 搜一下数据库和云原生                   → [search_hot_projects(categories=["Database", "Cloud-Native"])]
```

### Star 范围扫描 & 增长计算

```
你> 扫描 5000-20000 star 的仓库          → [scan_star_range(min_star=5000, max_star=20000)]
你> 查一下 vllm-project/vllm 的情况      → [check_repo_growth] star: 42350, 近10天 +1820
你> 计算候选池所有仓库增长                → [batch_check_growth]
你> 增长阈值降到 300 再看看              → [batch_check_growth(growth_threshold=300)]
```

### 排名 & Trending

```
你> 排个名看看                            → [rank_candidates(mode="comprehensive")] 综合排名
你> 最近有什么新冒出来的爆款？             → [rank_candidates(mode="hot_new")] 新项目排名
你> 近一个月新项目前20                     → [rank_candidates(mode="hot_new", top_n=20, new_project_days=30)]
你> 看看 GitHub Trending 上有什么          → [fetch_trending(since="weekly")]
你> 看看中文社区月度趋势                   → [fetch_trending(since="monthly", spoken_language="zh")]
```

### 描述 & 报告

```
你> 给 langchain-ai/langchain 生成描述    → [describe_project] 200-400 字 LLM 描述
你> 生成完整报告                           → [generate_report] 保存到 report/YYYY-MM-DD.md
你> 数据库里有多少仓库？                   → [get_db_info] 仓库数、日期等
```

## API 接口

服务启动后可直接调用 REST API：

```bash
# 对话
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user1", "message": "搜一下 AI Agent 方向的热门项目"}'

# 报告
curl http://localhost:8000/api/reports                   # 列表
curl http://localhost:8000/api/reports/2026-04-14.md     # 内容
http://你的IP:8000/api/reports/2026-04-14.md/html       # HTML 渲染（浏览器打开）

# 状态 & 会话
curl http://localhost:8000/api/status
curl -X DELETE http://localhost:8000/api/sessions/user1
```

## 输出文件

| 文件 | 位置 | 说明 |
|------|------|------|
| 报告 | `report/YYYY-MM-DD.md` | Markdown 热门项目排行 |
| 数据库 | `Github_DB.json` | 仓库历史数据 |
| 日志 | `logs/agent-YYYY-MM-DD.log` | Agent 执行日志 |
| 服务日志 | `logs/server.log` | API 服务运行日志 |

建议运行频率：每 7-10 天一次，保持 DB 数据新鲜（过期阈值 11 天）。
