# GitHub 热门项目发现 — 使用指南

## 环境准备

```bash
# 1. 环境变量（写入 ~/.bashrc 或当前终端）
export GITHUB_TOKENS="ghp_token1,ghp_token2"
export LLM_API_KEY="sk-xxx"

# 2. 安装依赖
pip install -r github_hot_projects/requirements.txt
```

## 启动方式

| 命令 | 说明 |
|------|------|
| `python -m github_hot_projects` | **API Server**（默认入口，REST/WS + Web 页面） |
| `python -m github_hot_projects.agent_cli` | **CLI REPL**（终端对话调试） |
| `python -m github_hot_projects.mcp_server` | **MCP Server**（stdio，供 VS Code Copilot / Claude Desktop） |
| `python -m github_hot_projects.scheduled_update` | **定时批处理**（搜索→增长→排名→报告） |
| `python -m github_hot_projects.regenerate_report --log <file>` | 从历史日志恢复报告 |

### 后台运行（推荐）

```bash
cd /root/code/Agent-skils
nohup python -m github_hot_projects >> github_hot_projects/logs/server.log 2>&1 &

# 查看日志
tail -f github_hot_projects/logs/server.log

# 重启
pkill -f "python -m github_hot_projects" || true
nohup python -m github_hot_projects >> github_hot_projects/logs/server.log 2>&1 &
```

### MCP 配置（Claude Desktop / VS Code Copilot）

```json
{
  "mcpServers": {
    "github-hot-projects": {
      "command": "python",
      "args": ["-m", "github_hot_projects.mcp_server"],
      "cwd": "/root/code/Agent-skils",
      "env": {
        "GITHUB_TOKENS": "ghp_token1,ghp_token2",
        "LLM_API_KEY": "sk-xxx"
      }
    }
  }
}
```

### 访问地址

```
浏览器:  http://你的公网IP:8000
本机:    http://127.0.0.1:8000
```

## 对话示例

网页端和 CLI 均支持自然语言，以下为常用指令：

```
# 完整流程
帮我跑一次完整的热门项目发现
查一下近期热门榜前50

# 按类别搜索（25+类别：AI-Agent, AI-MCP, Database, Cloud-Native, Frontend ...）
搜一下 AI Agent 方向的热门项目
搜一下数据库和云原生

# Star 扫描 & 增长
扫描 5000-20000 star 的仓库
查一下 vllm-project/vllm 的情况
增长阈值降到 300 再看看

# 排名 & Trending
排个名看看                          → comprehensive 综合排名
最近有什么新冒出来的爆款？          → hot_new 新项目排名
近一个月新项目前20                  → hot_new + new_project_days=30
看看 GitHub Trending 上有什么       → weekly trending
看看月度趋势                        → monthly trending

# 报告 & 描述
给 langchain-ai/langchain 生成描述
生成完整报告
数据库里有多少仓库？
```

## API 接口

```bash
# 对话
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user1", "message": "搜一下 AI Agent 方向的热门项目"}'

# 报告
curl http://localhost:8000/api/reports                        # 列表
curl http://localhost:8000/api/reports/2026-04-14.md          # Markdown
http://IP:8000/api/reports/2026-04-14.md/html                # HTML 渲染

# 管理
curl http://localhost:8000/api/status
curl -X DELETE http://localhost:8000/api/sessions/user1
```

## 输出文件

| 文件 | 位置 | 说明 |
|------|------|------|
| 报告 | `report/YYYY-MM-DD.md` | Markdown 热门项目排行 |
| 数据库 | `Github_DB.json` | 仓库历史数据（有效期 11 天） |
| 日志 | `logs/agent-YYYY-MM-DD.log` | Agent 执行日志 |
| 服务日志 | `logs/server.log` | API 服务运行日志 |

> 建议每 7-10 天运行一次，保持 DB 数据新鲜。
