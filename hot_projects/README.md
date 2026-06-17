# hot_projects — GitHub 热门项目发现 Agent

基于单 ReAct LLM 的 GitHub 热门项目发现系统：按关键词/星段/Trending 收集候选 → 计算近期 star 增长 → 评分排序 → 生成报告。支持终端对话、Web API、定时批处理三种使用方式。

## 1. 安装

```bash
pip install -r hot_projects/requirements.txt
```

## 2. 配置环境变量（仅需 export key）

URL、后端类型、模型名都已写死在 `config.py`，**只需 export 三个 key**：

```bash
# GitHub Token（必填，逗号分隔多个，ghp_ 开头的 PAT）
export GITHUB_TOKENS="ghp_xxx,ghp_yyy"

# 方案 A = Azure OpenAI（主力，必填）—— key 不带 sk- 前缀
export LLM_A_KEY="<azure-key>"

# 方案 B = SiliconFlow（备选，选填）—— key 是 sk- 开头
export LLM_B_KEY="<siliconflow-key>"
```

> ⚠️ 别把 A/B 两个 key 写反：**A 是 Azure（无 `sk-`）、B 是 SiliconFlow（`sk-` 开头）**。
> LLM 调用逐次先用 A，A 失败自动回退 B；B 不填则只用 A。

固定的默认（如需改在 `config.py`）：
- A：`gpt-5.4`（主对话）/ `gpt-5.4-mini`（描述压缩）
- B：`Pro/zai-org/GLM-5` / `Qwen/Qwen3.5-35B-A3B`

## 3. 启动方式

| 命令 | 说明 |
|------|------|
| `python -m hot_projects.agent_cli` | 终端对话（REPL，调试最方便） |
| `python -m hot_projects` | Web/API 服务（默认 8000 端口） |
| `python -m hot_projects.cron_scheduled_update --top-n 100` | 定时批处理（搜索→增长→排名→报告） |

后台运行 Web 服务：

```bash
cd /root/code/Agent-skils
nohup python -m hot_projects >> hot_projects/logs/server.log 2>&1 &
tail -f hot_projects/logs/server.log
```

访问：`http://127.0.0.1:8000`（本机）或 `http://你的IP:8000`。

## 4. 简洁用法（自然语言对话）

```
# 榜单（昂贵，会先回显参数让你确认，回复「开始」才执行）
跑个综合榜 top 20
最近有什么新冒出来的爆款？        → 新项目榜
搜一下 AI Agent 方向的热门项目     → 关键词榜
增长阈值降到 500 再看看           → 复用缓存，仅重排（不重新抓取）

# 单项目（秒级，直接执行）
查一下 vllm-project/vllm 的 star 增长
给 langchain-ai/langchain 生成介绍
vllm 怎么样                       → 名字不全/拼错会返回相似候选让你选

# 其他
看看 GitHub Trending
数据库里有多少项目？
```

## 5. 默认阈值（用户不传参时生效）

| 参数 | 默认 | 含义 |
|------|-----:|------|
| `MIN_STAR` | 1200 | 最低 star 门槛 |
| `MAX_STAR` | 45000 | 星段扫描上限 |
| `STAR_GROWTH_THRESHOLD` | 800 | 增长入选阈值 |
| `GROWTH_CALC_DAYS` | 7 | 增长统计窗口（天）* |
| `DAYS_SINCE_CREATED` | 45 | 新项目判定窗口（天） |
| `HOT_PROJECT_COUNT` | 100 | 综合/关键词榜 Top N |
| `HOT_NEW_PROJECT_COUNT` | 20 | 新项目榜 Top N |

\* 综合榜/关键词榜未指定窗口时：DB 有效则用「DB 距今天数」，否则回退 7 天。

## 6. 输出与日志

| 内容 | 位置 |
|------|------|
| 报告 | `hot_projects/report/YYYY-MM-DD*.md` |
| 数据库 | `hot_projects/Github_DB.json`（运行时生成，已 gitignore） |
| 定时任务主日志 | `hot_projects/logs/cron-YYYY-MM-DD.log` |
| 定时任务调试日志 | `hot_projects/logs/debug/cron-YYYY-MM-DD.debug.log` |
| CLI / Web 日志 | `hot_projects/logs/cli-YYYY-MM-DD.log` / `web.log` |

定时任务主日志默认保留阶段摘要、候选入选、每个项目最终增长结果、报告和 DB 更新统计；
逐关键词搜索、星段细分、逐仓库 stargazers 查询开始等细节写入同日期 debug 日志。

## 7. 项目结构（编排层重构后）

```
hot_projects/
├── agent/        精简 ReAct 循环 + 会话状态 + prompt
├── tools/        工具注册表 + LLM schema + 复合榜单工具(含确认守卫) + 原子工具(含模糊消歧)
├── pipeline/     唯一榜单流水线 ranking_pipeline + 分阶段参数签名缓存
├── capabilities/ 基础工具层（搜索/扫描/增长/排序/报告，纯函数）
├── providers/    Provider 接口 + 归一化 Repo 模型；providers/github/ 为 GitHub 实现
├── infra/        LLM 双后端客户端 + db + 并发调度框架
├── config.py     全局配置（阈值/关键词/路径/安全/LLM）
└── 入口: agent_cli / cron_scheduled_update / api_server / __main__
```

设计要点：榜单的工具调用顺序内聚在「复合工具」内部，顶层 ReAct 只负责选工具，无白名单/无前置硬校验/无状态机。多平台可通过新增 Provider 扩展。
