# 扩展方案规划

## 当前状态

| 能力 | 状态 | 接入方式 |
|------|------|----------|
| 管道式批处理 | ✅ 可用 | `python -m github_hot_projects` |
| Agent 终端交互 | ✅ 可用 | `python -m github_hot_projects.agent_cli` |
| REST API 服务 | ✅ 可用 | `python -m github_hot_projects.api_server` |
| WebSocket 实时对话 | 🔧 预留 | `/ws/chat/{session_id}` |
| 报告 API 查询 | ✅ 可用 | `/api/reports` |

---

## 扩展方向

### 方向一：微信接入（推荐优先）

**方案 A — 企业微信机器人（最简单）**

```
企业微信群 → Webhook → api_server /api/chat → Agent → 回复
```

- 零审核，直接配置群机器人 Webhook
- 适合团队内部使用
- 实现成本：约 50 行代码

**方案 B — 个人微信（itchat / wechaty）**

```
微信消息 → wechaty SDK → api_server /api/chat → Agent → 微信回复
```

- 需要维护登录态（扫码登录）
- 稳定性受微信封控影响
- 适合个人使用场景

**方案 C — 微信公众号 / 小程序**

```
用户 → 公众号/小程序 → 后端 → api_server /api/chat → Agent → 回复
```

- 需要公众号认证
- 用户体验最好，支持富文本展示报告
- 适合对外发布

**推荐路径**：A（立即可用）→ C（用户体验升级）

### 方向二：Web 前端

在 `github_hot_projects/web/` 下新建前端项目：
- 对话页面：调用 `/api/chat` 或 `/ws/chat`
- 报告列表页面：调用 `/api/reports`
- 报告详情页面：Markdown 渲染
- 仪表盘：项目增长趋势图

技术选型：
| 方案 | 适合场景 |
|------|----------|
| 纯 HTML + JS | 最简单，直接嵌入 |
| Vue 3 + Vite | 轻量 SPA |
| Next.js / Nuxt | SEO + SSR |

### 方向三：定时任务

```bash
# crontab 每天凌晨执行一次完整发现流程
0 3 * * * cd /path/to && python -m github_hot_projects >> cron.log 2>&1
```

或通过 Agent API：
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"cron","message":"执行完整热门项目发现流程"}'
```


### 方向五：持久化与多用户

| 组件 | 当前 | 演进目标 |
|------|------|----------|
| 会话存储 | 内存 dict | Redis / PostgreSQL |
| DB 存储 | JSON 文件 | SQLite / PostgreSQL |
| 报告存储 | 本地文件 | OSS（对象存储） |
| 认证 | 无 | API Key / JWT |
| 配额 | 无 | 每用户每日请求上限 |

### 方向六：Agent 能力增强

- `tool_fetch_github_trending` — 抓取 GitHub Trending 页面
- `tool_compare_repos` — 对比多个仓库增长趋势
- `tool_search_by_topic` — 按 GitHub Topic 搜索
- `tool_export_csv` — 导出 CSV/Excel 报告
- `tool_notify` — 微信/邮件推送提醒

---

## 预留改造点（已在代码中标注）

| # | 改造点 | 涉及文件 | 说明 |
|---|--------|----------|------|
| 1 | AgentState 序列化 | agent.py | `save_session()` / `load_session()` 接口预留 |
| 2 | ReportStore 抽象 | pipeline.py, agent_tools.py | 报告写入改为接口调用 |
| 3 | async Tool 执行 | agent.py | `_execute_tool()` → `async` 版本 |
| 4 | LLM 流式输出 | agent.py | `stream=True` + WebSocket 逐 token 推送 |
| 5 | 用户认证 | api_server.py | `user_id` 参数预留 |
| 6 | DATA_DIR 环境变量 | config.py | ✅ 已实现，支持容器化部署 |
