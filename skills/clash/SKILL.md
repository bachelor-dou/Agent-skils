---
name: clash
description: >-
  clash-for-linux 代理的安装与用法速查（仓库 wnlen/clash-for-linux）。当用户要在 Linux/WSL
  上安装/使用 clash 代理，涉及 clashctl、机场订阅、代理端口、选节点、webUI secret、验证
  出口地区时使用（先读 skills/install-kb）。
---

# clash-for-linux 速查

配套：unzip（webUI 面板解压需要）

## 安装（上次实测命令）

```bash
# 拉源码。ghfast.top 是 GitHub 加速前缀，能直连 GitHub 时可去掉
git clone --branch master --depth 1 https://ghfast.top/https://github.com/wnlen/clash-for-linux.git
cd clash-for-linux
# 安装。过程中会提示输入机场订阅链接（⚠️ 密钥，别贴进聊天/日志）；也可改 .env 做各项配置
bash install.sh
```

## 用法

```bash
clashctl status        # 看状态：本地代理 127.0.0.1:7890、webUI http://127.0.0.1:9090/ui/
clashctl on            # 开系统代理。最关键的一步：只装好服务不够，开了系统和大多数程序才真正走代理
clashctl off           # 关系统代理
clashctl select        # 切换节点
clashctl secret 新密码  # 改 webUI 登录 secret
# 让当前这个终端立刻拿到代理环境变量（clashctl on 后新变量不会自动进已开的 shell）
source /etc/profile.d/clash-for-linux.sh && clashon
```

## 验证

```bash
curl -I --proxy http://127.0.0.1:7890 https://www.google.com    # 返回 200 / 正常跳转头 = 链路可用
curl --proxy http://127.0.0.1:7890 https://api.ip.sb/geoip      # 看出口 IP 和地区
curl --proxy http://127.0.0.1:7890 https://ifconfig.co/json
```

## 踩坑与解决

- **webUI 显示 `Unauthorized`** → 不是服务没起，是访问了 controller API、或 UI 没带 secret
  → 用 `clashctl secret 新密码` 设置后再带 secret 登录。
- **代理端口连不上（不是 7890）** → install.sh 发现端口被占会随机改端口（Windows 客户端 /
  WSL mirrored 网络时常见，实测改过 7891）
  → 查真实端口：`grep -oE 'mixed-port: *[0-9]+' clash-for-linux/runtime/config.yaml`。
- **`clashctl on` 之后当前终端还是不走代理** → 已开的 shell 拿不到新环境变量
  → `source /etc/profile.d/clash-for-linux.sh && clashon`。
- **出口地区和节点名对不上** → 机场常贴错节点地区标签（实测选「🇦🇷 阿根廷Z01」出口实为香港）
  → 以 geoip 返回的 `country` 为准，不符就 `clashctl select` 换节点。
- **git pull/push 特别慢（几十 KiB/s）** → git 全局 `http.proxy`/`https.proxy` 指向
  clash（`git config --global --get http.proxy` 确认），慢的是当前节点而不是 git
  → 先测节点：`curl --proxy http://127.0.0.1:7890 -o /dev/null -sS -w '%{speed_download} B/s\n' 'https://speed.cloudflare.com/__down?bytes=10000000'`，
  慢就 `clashctl select` 换节点，换完用 geoip 确认出口。
- **开代理后 Python 程序报 socks 错误**（httpx 报 `Using SOCKS proxy, but the 'socksio'
  package is not installed`；requests 报 `Missing dependencies for SOCKS support`）
  → `clashon` 会导出 `all_proxy=socks5://…`（`scripts/core/alias.sh`），而 Python HTTP 库
  默认不带 SOCKS 支持；clash 本身不需要任何 Python 包
  → 给要跑的那个 Python 环境装：`pip install socksio`（httpx 用）或 `pip install pysocks`
  （requests 用），**每个 venv 各自要装**；不想装包就 `unset all_proxy ALL_PROXY`，
  让 Python 走 http_proxy。

## 更新记录

- 2026-07-24 WSL Ubuntu-24.04：装通。端口被占自动改为 7891（从 `runtime/config.yaml`
  的 mixed-port 探测到）；「🇦🇷 阿根廷Z01」节点连通但 geoip 出口实为香港。
- 2026-08-05 WSL（装于 `/root/clash-for-linux`）：确认 `clashon` 导出 socks5 的
  `all_proxy`，Python 侧需 socksio/pysocks；本机项目 `.venv` 已装 socksio 1.0.0，
  系统 Python 未装（未动）。
- 2026-08-09 WSL：git 走 7890 拉 GitHub 只有 66 KiB/s；实测当前节点（出口台湾
  Akari Networks）Cloudflare 测速约 430 KB/s——链路通、节点慢，新增「git 慢」排查条目。
