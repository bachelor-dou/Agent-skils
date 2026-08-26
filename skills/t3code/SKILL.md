---
name: t3code
description: >-
  T3 Code（npx t3）在 gpu24 上的安装与远程用法。当用户要装/开 T3、T3 Connect、手机连
  gpu24、或打不开 Cursor Provider 时使用（先读 skills/install-kb）。
---
# T3 Code 速查

配套：本机需 Node `^22.16 || ^23.11 || >=24.10`（Ubuntu 源的 Node 18 不够）；Cursor CLI 已有则用 `agent login`。不走 Tailscale 时用 T3 Connect。勿动本机已有 Tailscale Serve `443→8888`。

## 安装（上次实测命令）

gpu24、Ubuntu 24.04。用户侧装了 Node v24.19.0 + npm 11.17.0（官方 linux-x64 包，避免 APT 落到 Node 18）。Cursor CLI 已在 `/root/.local/bin/cursor-agent`，`agent status` 已登录。

```bash
# 拉 T3 CLI 并做无桌面 T3 Connect 绑定（终端会打印浏览器 URL，把授权码贴回）
npx t3@latest connect link --headless

# 日常：tmux 会话 t3 里前台跑 server + Cloudflare 隧道。必须监听本机：隧道打到 127.0.0.1:3773。不要 --host 100.x，不要 --tailscale-serve（会抢 443）
tmux new -s t3
# 进会话后：
npx t3@latest serve
# 看到 ready 后 Ctrl-b d 脱离；回来：tmux attach -t t3
```

默认 **不启用 Cursor**。在 server 已运行时写入设置，文件监视会立刻重探：

```bash
# 打开 Cursor Provider，并写死 binary，避免 PATH 探测失败
cat > ~/.t3/userdata/settings.json <<'EOF'
{
  "providers": {
    "cursor": {
      "enabled": true,
      "binaryPath": "/root/.local/bin/cursor-agent"
    }
  }
}
EOF
```

关 tmux 窗口或在会话里 Ctrl-c = 停 server。Cursor 终端关掉不影响已脱离的 `t3` 会话。要 systemd 常驻（本次未实测）：`npx t3@latest service install`。

## 用法

```bash
npx t3@latest connect status   # 看 T3 Connect 是否已登录、已 link
tmux new -s t3                 # 没有 t3 会话时新建并进入
tmux attach -t t3              # 回到已有 t3 会话
npx t3@latest serve            # 在 t3 会话里跑；Ctrl-b d 脱离，勿另开第二份
npx t3@latest service status   # 若已装 systemd 用户服务
```

手机：FClash 开、官方 Tailscale App 关（安卓只能一个 VpnService）。T3 App 或 https://app.t3.codes 用 **T3 Connect 同一账号** 进 gpu24。不要扫 serve 打印的 `localhost` 二维码。

T3 Connect 账号和 Cursor CLI 账号可以不是同一个：前者只管远程连机器，后者才跑模型。

## 验证

- `npx t3 serve` 出现 `T3 Code server is ready` 且有多条 `Registered tunnel connection`（ICMP/ping_group_range 警告可忽略）。
- `~/.t3/caches/cursor.json` 里 `enabled/installed/status` 为 true / true / `ready`。
- 手机能进项目目录，Model 能选 Cursor 模型，发送键可点。

## 踩坑与解决

- 现象：日志只有 Claude/Grok health check failed，从不查 Cursor → 原因：Cursor 默认 `enabled: false`，关着不探测 → 解决：写 `~/.t3/userdata/settings.json` 如上。
- 现象：能看项目、发送键黑、Model 点不了 → 原因：已连 T3 server，但没有可用 Provider（Cursor 关；Codex/Claude 没装）→ 解决：同上打开 Cursor。
- 现象：网页 Settings/Providers 显示 No connected devices → 原因：该浏览器标签没挂上 gpu24 执行环境，不是 server 没起 → 解决：在已连上的手机客户端发；或网页先选中 gpu24 环境再进 Providers。手机 Settings 往往没有 Providers 项。
- 现象：关终端后手机连不上 → 原因：`t3 serve` 随 shell 退出 → 解决：在 `tmux` 会话 `t3` 里跑 `npx t3@latest serve`，Ctrl-b d 脱离；或装 `t3 service`。
- 现象：安卓 Tailscale 和 FClash 不能同时开 → 原因：系统只允许一个 VpnService → 解决：手机长期开 FClash 时用 T3 Connect，不要开官方 Tailscale App。
- 现象：`t3 serve --tailscale-serve` 或默认占 443 → 原因：gpu24 上 443 已映射到 8888 → 解决：T3 Connect 路径不要加 `--tailscale-serve`。
- 现象：T3 Connect 里环境名是一串随机主机名（如阿里云 `iZ…`）→ 原因：0.0.33 没有 `--name`；显示名优先用 Linux Pretty Hostname，否则用 `hostname` → 解决：`hostnamectl set-hostname --pretty 'gpu24'`（只改好看的名字，不改内核 hostname），然后重启 `t3 serve`。手机列表可能要重新进一下。
- 现象：T3 里用不了 IDE 的 matt skills / User Rules → 原因：T3 调的是已安装的 `cursor-agent`，不是没装 CLI；Matt 已在 `~/.cursor/skills/`（IDE/CLI 共用）。T3 没有 Cursor 的 `/技能名` 菜单；`ask-matt` 禁止模型自行调用；IDE User Rules 和 `Ai-workbench/AGENTS.md` 不随 T3 打开的其它项目走 → 解决：消息里写明读 `~/.cursor/skills/<name>/SKILL.md`；项目规则放到该项目的 `.cursor/rules` 或 `AGENTS.md`。

## 更新记录

2026-08-25 gpu24：T3 Connect + `t3 serve` 手机可发；必须在 `settings.json` 打开 Cursor 并指定 `cursor-agent` 路径。
2026-08-26：环境显示名无 CLI 改名；用 `hostnamectl --pretty`。日常用 tmux 会话 `t3` 跑 `npx t3@latest serve`。T3 用不了 IDE slash/User Rules；Matt 文件在 `~/.cursor/skills/`。
