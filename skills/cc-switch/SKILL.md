---
name: cc-switch
description: >-
  CC-Switch（Claude Code 模型配置切换器，仓库 farion1231/cc-switch）的安装与用法速查。
  当用户要在 Linux 上安装/启动 cc-switch、导入导出模型配置时使用（先读 skills/install-kb）。
---

# cc-switch 速查

配套：装前先关代理（`clashoff`，见 `skills/clash`）；常与 `skills/claude-code` 配套使用

仓库：<https://github.com/farion1231/cc-switch>

## 安装（上次实测命令）

```bash
# 下载 deb（x86_64；arm64 换下面注释那个）
wget https://github.com/farion1231/cc-switch/releases/download/v3.15.0/CC-Switch-v3.15.0-Linux-x86_64.deb
# wget https://github.com/farion1231/cc-switch/releases/download/v3.13.0/CC-Switch-v3.13.0-Linux-arm64.deb

# 关代理后安装
apt install ./CC-Switch-v3.15.0-Linux-x86_64.deb
```

## 用法

```bash
cc-switch                                   # 启动，配置模型
nohup cc-switch >/tmp/cc-switch.log 2>&1 &  # 后台启动
```

## 验证

```bash
dpkg -s cc-switch | grep Version
```

## 踩坑与解决

- **换新环境不用重配** → 在旧环境的页面里直接导出配置，新环境导入即可完成环境更替。

## 更新记录

- 2026-08-05 收录自用户实测笔记（v3.15.0 x86_64）。
