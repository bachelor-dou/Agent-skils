---
name: croc
description: >-
  croc 文件传输工具的安装速查（仓库 schollz/croc）。当用户要在两台电脑间传文件/文件夹、
  安装或使用 croc 时使用（先读 skills/install-kb）。
---

# croc 速查

配套：无

仓库：<https://github.com/schollz/croc>（端到端加密的跨平台文件传输 CLI）

## 安装（上次实测命令）

```bash
# 官方一键安装脚本（Linux/macOS 通用）
curl https://getcroc.com | bash
```

## 用法

```bash
croc send <文件或文件夹>   # 发送端：输出一个 code-phrase
croc <code-phrase>         # 接收端：凭 code-phrase 收文件
```

## 验证

```bash
croc --version
```

## 踩坑与解决

暂无。

## 更新记录

- 2026-08-05 收录自官方 README，暂未实测；实测后更新本节。
