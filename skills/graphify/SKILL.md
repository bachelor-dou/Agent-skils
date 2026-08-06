---
name: graphify
description: >-
  graphify 代码知识图谱工具的安装与用法速查（仓库 Graphify-Labs/graphify，PyPI 包名
  graphifyy）。当用户要安装 graphify、给代码库建知识图谱、用 graphify query/path/explain
  查询代码结构时使用（先读 skills/install-kb）。
---

# graphify 速查

配套：Python 3.10+ 与 uv（推荐）或 pipx

仓库：<https://github.com/Graphify-Labs/graphify>（把代码库/文档/PDF 变成可查询的知识图谱，
代码解析纯本地 AST、零 LLM 费用）

## 安装（上次实测命令）

```bash
# ⚠️ PyPI 包名是 graphifyy（双 y），命令才叫 graphify——别装错包
uv tool install graphifyy        # 或 pipx install graphifyy

graphify install                 # 注册 skill 到 AI 助手（默认 Claude Code）
graphify cursor install          # Cursor：写 .cursor/rules/graphify.mdc（alwaysApply）
# 其他平台：graphify install --platform codex/gemini/... ；跨框架装到 ~/.agents/skills：
# graphify install --platform agents
# 只装进当前仓库（而非用户级）加 --project
```

## 用法

```bash
/graphify .                        # 在 AI 助手里：给当前目录建图（Codex 用 $graphify）
graphify query "问题"              # 按问题取子图，代替 grep
graphify path "A" "B"              # 两个符号间的依赖路径
graphify explain "概念"            # 一个概念的所有关联
graphify update .                  # 改代码后增量更新（纯 AST，零 API 费用）
graphify hook install              # git commit 后自动重建
```

产物在 `graphify-out/`：`graph.html`（可视化）、`GRAPH_REPORT.md`（要点报告）、`graph.json`（全图）。

## 验证

```bash
graphify --version
```

## 踩坑与解决

- **`graphify: command not found`** → uv/pipx 把命令装在 `~/.local/bin`，PATH 里没有
  → `uv tool update-shell`（或 `pipx ensurepath`）后开新终端。
- **`uvx graphify ...` 解析失败（No solution found）** → uvx 把第一个词当包名，而包叫 graphifyy
  → `uvx --from graphifyy graphify install`。
- **重构删文件后图里残留旧节点** → 重建时节点变少会被拒绝覆盖 → 加 `--force`（或 `GRAPHIFY_FORCE=1`）。
- **升级后 IDE 提示 skill 版本不匹配** → `uv tool upgrade graphifyy && graphify install` 覆盖旧 skill 文件。
- **PowerShell 下 `/graphify .` 报路径错误** → Windows 用 `graphify .`（不带斜杠）。

## 更新记录

- 2026-08-05 WSL：本机实测在用 v0.9.21；本仓库已由 `graphify cursor install` 写入
  `.cursor/rules/graphify.mdc`，`graphify-out/` 已建图。安装命令收录自官方 README。
