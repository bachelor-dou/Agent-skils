# Agent-skils

多 Agent 项目 Monorepo —— 每个子目录为一个独立可运行的 Agent。

## 项目结构

```
Agent-skils/
├── .gitignore                          # 共享忽略规则
├── README.md                           # 本文件
│
├── github_hot_projects/                # GitHub 热门项目发现 Agent
│   ├── docs/                           #   └─ 详细文档（设计/用法/路线图）
│   ├── report/                         #   └─ 生成的每日报告
│   ├── logs/                           #   └─ 运行日志
│   └── requirements.txt                #   └─ 依赖清单
│
├── shared/                             # 公共工具库（骨架，待多 Agent 复用时填充）
│
└── skills/                             # Copilot 自定义 skill
    └── task-runner/                    #   └─ 连续任务执行器
```

## Agent 一览

| 目录 | 说明 | 文档 |
|------|------|------|
| `github_hot_projects/` | 通过多源数据采集 + 智能增长分析，自动发现 GitHub 上近期快速增长的热门开源项目。支持管道批处理、CLI 对话、REST API 三种运行方式。 | [设计文档](github_hot_projects/docs/design.md) · [使用指南](github_hot_projects/docs/usage.md) · [路线图](github_hot_projects/docs/roadmap.md) |
| `shared/` | 多 Agent 复用的公共工具库（LLM 调用、Token 管理等）。当前为骨架目录。 | — |

## 快速开始

每个 Agent 均从**本目录（Agent-skils/）**启动：

```bash
cd /path/to/Agent-skils

# 安装某个 Agent 的依赖
pip install -r github_hot_projects/requirements.txt

# 运行（默认入口由各 Agent 自己定义）
python -m <agent_name>              # 默认入口，例如 github_hot_projects 为 Web/API 服务
python -m <agent_name>.agent_cli    # CLI 对话（如支持）
```

各 Agent 的详细用法、配置说明、模块结构请查阅其 `docs/` 目录。

## 添加新 Agent

1. 在根目录创建新包：`mkdir new_agent && touch new_agent/__init__.py new_agent/__main__.py`
2. 添加 `new_agent/requirements.txt` 和 `new_agent/docs/` 文档目录
3. 确保可通过 `python -m new_agent` 启动
4. 在上方 **Agent 一览** 表格中补充条目

## 设计原则

- **独立可运行**：每个 Agent 从根目录 `python -m <agent_name>` 独立启动
- **数据隔离**：日志、数据、报告存放在各自 Agent 目录下
- **文档自治**：各 Agent 在自己的 `docs/` 下维护详细文档，根 README 仅做索引
- **按需共享**：3+ Agent 复用相同逻辑时，抽取到 `shared/`
