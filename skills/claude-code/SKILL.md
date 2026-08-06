---
name: claude-code
description: >-
  Claude Code CLI 及 superpowers 插件的安装速查。当用户要安装 claude code、装 superpowers
  技能市场插件时使用（先读 skills/install-kb）。
---

# claude-code 速查

配套：`skills/nodejs`（npm 全局安装）；模型配置常用 `skills/cc-switch` 切换

## 安装（上次实测命令）

```bash
npm install -g @anthropic-ai/claude-code
```

## 安装 superpowers 插件

在 `claude` 会话内执行：

```
/plugin marketplace add obra/superpowers-marketplace   # 添加插件市场
/plugin install superpowers@superpowers-marketplace    # 安装 superpowers 技能
```

## 验证

```bash
claude --version
```

## 踩坑与解决

暂无。

## 更新记录

- 2026-08-05 收录自用户实测笔记。
