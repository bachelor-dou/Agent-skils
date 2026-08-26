---
name: ponytail
description: >-
  Ponytail（DietrichGebert/ponytail）的安装与更新速查。当用户要装/更新 ponytail、拷
  ponytail.mdc、或问 Cursor 有没有 /ponytail 命令时使用（先读 skills/install-kb）。
---
# ponytail 速查

配套：无（Cursor 只拷规则文件，不装 npm 包）

仓库：<https://github.com/DietrichGebert/ponytail>（让 agent 按最简能跑的 diff 写代码）

Cursor / Windsurf / Cline 是 **instruction-only**：拷规则，没有 `/ponytail lite|full|ultra` 那些命令（那是 Claude Code / Codex 插件才有的）。

## 安装（上次实测命令）

gpu24、Cursor。规则落到用户级 `~/.cursor/rules/`（也可放到某个仓库的 `.cursor/rules/`）：

```bash
# 从上游拉 Cursor 规则（alwaysApply）
mkdir -p ~/.cursor/rules
curl -fsSL https://raw.githubusercontent.com/DietrichGebert/ponytail/main/.cursor/rules/ponytail.mdc \
  -o ~/.cursor/rules/ponytail.mdc
```

某个仓库单独用：拷到该仓库 `.cursor/rules/ponytail.mdc`。本机 `/root/code/.cursor/rules/ponytail.mdc` 是项目副本，和用户级文件不是同一份。

## 用法

规则 `alwaysApply: true`，新开 Agent 会话即带上。写代码前按梯子：YAGNI → 复用仓库已有 → stdlib → 平台原生 → 已装依赖 → 一行 → 才写最短能跑的。

更新（对照上游，有 diff 再覆盖）：

```bash
curl -fsSL https://raw.githubusercontent.com/DietrichGebert/ponytail/main/.cursor/rules/ponytail.mdc \
  | diff -u ~/.cursor/rules/ponytail.mdc - || true
```

## 验证

```bash
test -f ~/.cursor/rules/ponytail.mdc && echo ok
```

新开一条 Agent 对话后规则才进上下文；当前对话不会自动带上。

## 踩坑与解决

- 现象：Cursor 里打 `/ponytail ultra` 没反应 → 原因：Cursor 只加载 `.mdc`，不装插件命令 → 解决：改规则文件或换 Claude Code/Codex 插件安装。
- 现象：T3 / cursor-agent 没有 ponytail → 原因：CLI 认当前项目 `.cursor/rules` 和（若支持）`~/.cursor/rules`；工作区不在 `$HOME` 下时用户级路径可能扫不到 → 解决：把 `ponytail.mdc` 拷进那个项目的 `.cursor/rules/`。

## 更新记录

- 2026-08-26 gpu24：用户级 `~/.cursor/rules/ponytail.mdc` 与上游 `main` 正文一致，只差文件末换行，不必为这个覆盖。上游 `.cursor/rules` 最近改动 2026-07-10。
