---
name: nodejs
description: >-
  Node.js（LTS，NodeSource 源）的安装速查。当用户要在 Linux/WSL 上安装 node/npm、
  配置 NodeSource 源时使用（先读 skills/install-kb）。
---

# nodejs 速查

配套：装前先关代理（装了 clash 的环境用 `clashoff`，见 `skills/clash`）

## 安装（上次实测命令）

```bash
# 1. 先关闭代理（开着代理时配置源易失败）
clashoff

# 2. 配置 NodeSource LTS 源并安装
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash - && sudo apt install -y nodejs
```

## 验证

```bash
node -v
npm -v
```

## 踩坑与解决

暂无。

## 更新记录

- 2026-08-05 收录自用户实测笔记。
