---
name: matt-skills
description: >-
  Matt Pocock 工程技能包的安装速查（仓库 mattpocock/skills，含 grill-me、tdd、code-review
  等）。当用户要安装/更新 matt skills、装 grill/tdd 技能包、用 npx skills 装技能仓库时
  使用（先读 skills/install-kb）。
---

# matt-skills 速查

配套：`skills/nodejs`（npx 需要）；Claude Code 插件方式则需 `skills/claude-code`

仓库：<https://github.com/mattpocock/skills>（工程实践技能集：对齐、TDD、评审、架构等）
方式 B 由发布版 `npx skills` 安装器读取仓库默认分支 `main`，不锁 Release/tag；
`skills@x` 是安装器版本，不是 Matt 技能版本。

## 安装（上次实测命令）

两种方式**二选一**（都装会技能重复）：

```bash
# 方式 A：Claude Code 官方插件（托管只读、自动更新）
claude plugins install mattpocock-skills      # 或会话内 /plugin install mattpocock-skills

# 方式 B：任何平台（Codex/Cursor/...），装成自己可编辑的文件
npx skills@latest add mattpocock/skills
# 交互中选技能和目标平台；⚠️ 务必勾上 setup-matt-pocock-skills
# 用户级装到 ~/.agents/skills/；更新用 npx skills update
```

装完后**每个仓库跑一次**初始化（配置 issue tracker、triage 标签、文档位置）：

```
/setup-matt-pocock-skills
```

## 用法

常用入口（会话内敲）：`/ask-matt`（路由，不知道用哪个就问它）、`/grill-me`（对齐拷问）、
`/grill-with-docs`（拷问 + 建域模型）、`/tdd`、`/code-review`、`/triage`。

## 验证

```bash
ls ~/.agents/skills/    # 方式 B：能看到 ask-matt、tdd 等目录
```

方式 A 在 Claude Code 会话里敲 `/ask-matt` 能响应即可。

## 踩坑与解决

- **两种方式都装了 → 每个技能出现两份** → 只保留一种：插件方式 `claude plugins` 管理，
  文件方式删 `~/.agents/skills/` 下对应目录。
- **`npx skills@latest check` 会直接更新** → 正式更新用 `npx skills@latest update -g`。
- **远端已删除的技能仍留在本机** → 非交互更新只告警、不删除，需人工确认后再清理。

## 更新记录

- 2026-08-05 WSL：方式 B 安装到 `~/.agents/skills/`。
- 2026-08-17 WSL：从官方 `main` 更新并核对一致；记录 `check` 会更新、远端删除不自动清理。
