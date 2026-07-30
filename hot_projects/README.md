# hot_projects — GitHub 热门项目发现 Agent

基于单 ReAct LLM 的 GitHub 热门项目发现系统：按关键词/星段/Trending 收集候选 → 计算近期 star 增长 → 评分排序 → 生成报告。支持终端对话、Web API、定时批处理三种使用方式。

## 1. 安装

```bash
cd /root/code/Agent-skils
source .venv/bin/activate
pip install -r hot_projects/requirements.txt
```

## 2. 配置环境变量（仅需 export key）

平台的 URL、后端类型、模型名都写死在 `config.py`，运行时只需 export 对应的 key：

```bash
# GitHub Token（必填，逗号分隔多个，ghp_ 开头的 PAT）
export GITHUB_TOKENS="ghp_xxx,ghp_yyy"

# LLM key（每个平台一个，未设置的平台自动跳过）
export LLM_A_KEY="..."
```

有哪些平台、各用什么主/子模型，见 `config.py` 的 `LLM_MODELS`。

## 3. 启动方式

建议统一使用项目虚拟环境中的 Python，避免系统 Python 缺少 `fastapi` 等依赖。

| 命令 | 说明 |
|------|------|
| `/root/code/Agent-skils/.venv/bin/python -m hot_projects.agent_cli` | 终端对话（REPL，调试最方便） |
| `/root/code/Agent-skils/.venv/bin/python -m hot_projects` | Web/API 服务（默认 8000 端口） |
| `/root/code/Agent-skils/.venv/bin/python -m hot_projects.cron_scheduled_update --top-n 100` | 定时批处理（搜索→增长→排名→报告） |

前台启动 Web 服务：

```bash
cd /root/code/Agent-skils
/root/code/Agent-skils/.venv/bin/python -m hot_projects
```

后台运行 Web 服务：

```bash
cd /root/code/Agent-skils
nohup /root/code/Agent-skils/.venv/bin/python -m hot_projects >> hot_projects/logs/server.log 2>&1 &
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
| `MIN_STAR` | 500 | 最低 star 门槛 = 入库宽度 ** |
| `MAX_STAR` | 100000 | 星段扫描上限（仅每日发现阶段用）|
| `STAR_GROWTH_THRESHOLD` | 1000 | 增长入选阈值 = 出榜的唯一闸门 |
| `GROWTH_CALC_DAYS` | 7 | 增长统计窗口（天）* |
| `DAYS_SINCE_CREATED` | 45 | 新项目判定窗口（天） |
| `HOT_PROJECT_COUNT` | 100 | 综合/关键词榜 Top N |
| `HOT_NEW_PROJECT_COUNT` | 13 | 新项目榜 Top N |

\* 综合榜/关键词榜未指定窗口时：DB 有效则用「DB 距今天数」，否则回退 7 天。

\*\* 一个数三个身份：每日任务收进 DB 的下沿、榜单候选池的下界（候选池就是 DB）、
Agent 工具 min_star 的默认值。它只决定「哪些仓库被观测」，不决定「哪些仓库出榜」——
后者由 `STAR_GROWTH_THRESHOLD`（窗口内涨幅）独立控制。两个旋钮互不影响：
调低入库宽度不会放水榜单，调高增长阈值也不必回头动它。

## 6. 输出与日志

| 内容 | 位置 |
|------|------|
| 报告 | `hot_projects/report/YYYY-MM-DD*.md` |
| 数据库/收藏 | `hot_projects/data/`（`Github_DB.json`、`favorites.json`，运行时生成，已 gitignore） |
| 定时任务主日志 | `hot_projects/logs/YYYY-MM/cron-YYYY-MM-DD.log`（按月归档） |
| 定时任务调试日志 | `hot_projects/logs/YYYY-MM/debug/cron-YYYY-MM-DD.debug.log` |
| CLI / Web 日志 | `hot_projects/logs/cli-YYYY-MM-DD.log` / `web.log` |

定时任务主日志默认保留阶段摘要、候选入选、每个项目最终增长结果、报告和 DB 更新统计；
逐关键词搜索、星段细分、逐仓库增长定案来源等细节写入同日期 debug 日志。

## 7. 项目结构

**找工具:先看 `tools/registry.py` 的名单 → 打开同名文件看实现。** 每个 LLM 工具一个文件。

```
hot_projects/
├── agent/            精简 ReAct 循环 + 会话状态(通用 tool_state 槽) + prompt
├── tools/            工具层——每个 LLM 工具一个文件，文件名 = 工具名
│   ├── registry.py       唯一注册表(工具名单一览)
│   ├── schemas.py        唯一 LLM function-calling schema + 参数校验规格
│   ├── arg_validator.py  参数严格校验
│   ├── tool/            所有 LLM 工具(一工具一文件，文件名=工具名)
│   │   ├── ranking.py        复合榜单(综合/新项目/关键词)：内部编排 + 缓存 + 确认守卫
│   │   ├── repo_growth.py / describe_project.py / repo_profile.py / search_repos.py
│   │   └── analyze_report.py / get_db_info.py / fetch_trending.py
│   └── basic/           基础能力(被 ≥2 个工具复用的公用实现，不直接暴露给 LLM)
│       ├── core.py       搜索/扫描/增长/排序/描述/DB/Trending 实现
│       ├── report.py     报告生成   scoring.py 评分   report_parse.py 报告解析
│       └── resolve.py    单仓库输入消歧
├── datasource/       数据源适配层：Provider 接口 + 归一化 Repo；datasource/github/ 为 GitHub 实现
├── infra/            LLM 多平台客户端 + llm(描述生成) + db + favorites + 并发调度框架
├── data/             运行时数据(Github_DB.json / favorites.json / 断点，已 gitignore)
├── config.py         全局配置(阈值/关键词/路径写死/安全/LLM)
└── 入口: agent_cli / cron_scheduled_update / api_server(含报告 HTML 渲染+上期对比) / __main__
```

设计要点:工具只有两类——**独立工具**(一文件一工具,内部若需编排就自己 import basic)与 **basic 公用能力**(被 ≥2 个工具复用才准入)。榜单的顺序编排内聚在 `ranking.py`;agent 与定时任务共用同一榜单入口。多平台可通过新增 datasource Provider 扩展。
