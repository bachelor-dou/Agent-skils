---
name: wsl-full-env
description: >-
  Create a WSL Ubuntu distro, migrate it to E:\env\WSL, make root the default
  user, and install a fixed dev stack (python link, Clash, Node 24, Claude Code,
  CC-Switch). Use when creating, migrating, or provisioning a WSL environment.
---

# WSL 完整环境自动化

新建 Ubuntu → 迁移到 E 盘 → 仅 root → 装固定软件栈。

## 快速路径（推荐，一条命令）

`provision.sh`（同目录）把整套固化成幂等脚本，从任意 WSL 发行版里跑，只在两个不可
自动化的密钥点停（Clash 订阅、CC-Switch 配置）。**优先用它，别再逐条手敲**：

```bash
bash provision.sh <名字>                 # 新建/迁移/仅root + 全栈 + 克隆 clash
# 用户跑 clash install.sh 输入订阅后：
bash provision.sh <名字> --verify-clash  # 探测端口 + 204/geoip 验证
```

幂等：E 盘已有该发行版就跳过新建+迁移；apt/npm 已装即跳过。脚本内已处理 interop 自愈、
stdin 喂脚本（免多层引号）、apt 锁超时、端口探测。**为什么快**：agent 逐条跑时每条
带 `写/proc`+改 WSL 的命令都被安全审查逐条拦；用户自己一次跑脚本则零审批。

下面的分步是脚本的等价说明，供排障或 agent 手动介入时参考。全程 agent 用
`powershell.exe -NoProfile -Command "wsl -d <distro> -u root -- bash -c '...'"`
执行（不要用 `cmd.exe`）。

## 规则

- **只碰目标发行版**：允许操作 `$TargetDistro`、其目录 `E:\env\WSL\<名>\`、其内部文件。
  不读写其他发行版、项目文件、Windows 用户目录。`.wslconfig` 可只读、不可改。
- **只停三处**：环境名、Clash 订阅、CC-Switch 配置导入。其余连续跑，每步跑完即验证下一步。
- 停发行版用 `wsl --terminate <名>`，不要 `wsl --shutdown`（会关全部）。
- 破坏性动作（`wsl --unregister`、改 binfmt）按工具要求走审批。

## 变量

每次 `powershell.exe` 调用是独立会话，需重新声明或内联：

```powershell
$TargetDistro = '<用户给的名字>'   # 校验: 仅 字母数字 . _ -
$Destination  = 'E:\env\WSL'
```

## 1. 新建（先问环境名）

```powershell
wsl --install Ubuntu-24.04 --name $TargetDistro
wsl --list --verbose
```

首启会强制建一个临时用户，记下用户名，步骤 3 删除。

## 2. 迁移到 E 盘

```powershell
$Tar = "$Destination\$TargetDistro\$TargetDistro.tar"
wsl --terminate $TargetDistro
New-Item -ItemType Directory -Force "$Destination\$TargetDistro"
wsl --export $TargetDistro $Tar
wsl --unregister $TargetDistro
wsl --import $TargetDistro "$Destination\$TargetDistro" $Tar --version 2
if (Test-Path -LiteralPath "$Destination\$TargetDistro\ext4.vhdx") {
  Remove-Item -LiteralPath $Tar        # 确认 vhdx 存在后才删 tar
} else { throw 'ext4.vhdx missing; keep tar' }
```

## 3. 仅 root

```powershell
$TemporaryUser = '<步骤1的临时用户>'
wsl -d $TargetDistro -u root -- bash -c "echo -e '[boot]\nsystemd=true\n\n[user]\ndefault=root' > /etc/wsl.conf"
wsl --terminate $TargetDistro
wsl -d $TargetDistro -- whoami                                   # 期望 root
wsl -d $TargetDistro -u root -- bash -c "userdel -r '$TemporaryUser'"
```

`userdel` 报 mail spool 不存在是无害警告。

## 4. 软件栈

固定清单，逐项装完即验证。`ca-certificates`/`curl`/`git`/`gnupg` Ubuntu 已自带，
仅在缺失时装。所有 `apt` 前先等 dpkg 锁（见 Pitfalls）。

### 4.1 python 软链 + unzip

```powershell
wsl -d $TargetDistro -u root -- bash -c "export DEBIAN_FRONTEND=noninteractive; apt-get install -y unzip python-is-python3 && readlink -f \$(command -v python) && python --version"
```

期望 `python` → python3。`unzip` 是 Clash 面板解压所需。

### 4.2 Clash for Linux（订阅检查点）

```powershell
wsl -d $TargetDistro -u root -- bash -c "git clone --branch master --depth 1 https://github.com/wnlen/clash-for-linux.git /opt/clash-for-linux"
```

装到 `/opt`（脚本拒绝在 `/mnt` Windows 挂载盘运行）。订阅链接是机场密钥，
**不要进聊天**——让用户在本机跑安装脚本、按提示输入：

```powershell
wsl -d $TargetDistro -u root -- bash -c "cd /opt/clash-for-linux && bash install.sh"
```

用户确认完成后：开代理 → **探测真实端口** → 选节点（默认阿根廷）→ 测连通 → 验出口地区。
**端口不要写死 7890**：install.sh 若发现端口被占（Windows 客户端 / 其他发行版 + mirrored
网络时常见），会随机改端口（实测过 7891）。唯一可信来源是运行时配置 `mixed-port`：

```powershell
wsl -d $TargetDistro -u root -- clashctl on
wsl -d $TargetDistro -u root -- bash -c 'P=$(grep -oE "mixed-port: *[0-9]+" /opt/clash-for-linux/runtime/config.yaml | grep -oE "[0-9]+" | head -1); echo "PORT=$P"; clashctl select GLOBAL "🇦🇷 阿根廷Z01"; curl -o /dev/null -s -w "code:%{http_code}\n" --max-time 20 --proxy "http://127.0.0.1:$P" https://www.gstatic.com/generate_204; curl -s --max-time 20 --proxy "http://127.0.0.1:$P" https://api.ip.sb/geoip'
```

端口只认 `runtime/config.yaml` 的 `mixed-port`——别 grep 整个目录（模板里还留着 7890 会混淆），
`ss -tlnp | grep mihomo` 可交叉核对（mihomo 同时听 mixed/dns/controller 三个口，按值区分不可靠，
以 config 为准）。两个验证缺一不可：`generate_204` 返回 204 证明代理**通**、且流量确实过了代理
（geoip 返回的是机场服务器 IP 而非本机）；但 geoip 的 `country` **未必等于节点名**——机场常给
节点乱贴地区标签（实测 `🇦🇷 阿根廷Z01` 出口在香港）。谁需要特定出口地区（如某步 Claude 登录），
以 geoip 的真实 `country` 为准，不合适就 `clashctl select` 换节点。
`clashctl select <策略组> <节点>` 非交互；节点名随订阅变，报不存在就换名。命令是 `clashctl`
（无 `clash`）。别用 `google.com -I` 验证（经代理常空返回，假阴性）。`系统流量未自动接管` 正常，
本地 `127.0.0.1:<探测端口>` 仍可显式用。

### 4.3 Node.js 24

```powershell
$NodeSetup = @'
install -d -m 0755 /etc/apt/keyrings
curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor --yes -o /etc/apt/keyrings/nodesource.gpg
echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_24.x nodistro main" > /etc/apt/sources.list.d/nodesource.list
export DEBIAN_FRONTEND=noninteractive
for i in $(seq 1 60); do fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || break; sleep 3; done
apt-get update && apt-get install -y nodejs
node --version && npm --version
'@
wsl -d $TargetDistro -u root -- bash -c $NodeSetup
```

### 4.4 Claude Code（固定版本）

```powershell
wsl -d $TargetDistro -u root -- npm install --global @anthropic-ai/claude-code@2.1.215
wsl -d $TargetDistro -u root -- claude --version      # 期望 2.1.215 (Claude Code)
```

npm 的 allow-scripts 警告无害。无需登录（模型配置由 CC-Switch 提供）。

### 4.5 CC-Switch（配置检查点）

```powershell
$Url = 'https://github.com/farion1231/cc-switch/releases/download/v3.17.0/CC-Switch-v3.17.0-Linux-x86_64.deb'
wsl -d $TargetDistro -u root -- bash -c "curl -fL '$Url' -o /tmp/cc-switch.deb"
wsl -d $TargetDistro -u root -- bash -c "for i in \$(seq 1 60); do fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || break; sleep 3; done; DEBIAN_FRONTEND=noninteractive apt-get install -y /tmp/cc-switch.deb; rm -f /tmp/cc-switch.deb"
wsl -d $TargetDistro -u root -- bash -c "dpkg -s cc-switch | grep Version"   # 期望 3.17.0
```

装完启动 UI，让用户导入之前导出的模型配置（agent 不搬配置）：

```powershell
wsl -d $TargetDistro -u root -- bash -c "nohup cc-switch >/tmp/cc-switch.log 2>&1 &"
```

**不装**：redis、C/C++ 工具链、中文包、全局 Python 包。SSH/凭据/token 一律不复制。

## Pitfalls（都已实测）

- **dpkg 锁**：新系统开机 `unattended-upgrades` 占锁 1–2 分钟。首选给 apt 直接加
  `-o DPkg::Lock::Timeout=300`（24.04 原生支持，无 shell 变量、免踩多层引号）；
  实在要循环再用 `for i in $(seq 1 60); do fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || break; sleep 3; done`
  （注意此 `$i`/`$(...)` 经 bash→powershell→wsl 多层会被吞，只在 powershell 直跑时用）。
- **`Exec format error`**：`wsl.exe`/`powershell.exe` 突然报此错说明 `WSLInterop` binfmt 掉了
  （`ls /proc/sys/fs/binfmt_misc/` 无 `WSLInterop`）。恢复（需审批）：
  `echo ':WSLInterop:M::MZ::/init:PF' > /proc/sys/fs/binfmt_misc/register`。
  **若 agent 的 shell 本身跑在另一个 WSL 发行版里**，每次 `wsl --terminate` 都会把当前
  发行版的 interop 冲掉；给每条 powershell 调用加自愈前缀最省事：
  `[ -e /proc/sys/fs/binfmt_misc/WSLInterop ] || echo ':WSLInterop:M::MZ::/init:PF' > /proc/sys/fs/binfmt_misc/register; powershell.exe ...`。
- **三层引号**：`grep` 别用 `|`（会被当管道，用单一模式）；别用 `${...}` 占位符
  （被吞成空，改 `dpkg -s pkg | grep Version`）。base64 传脚本仅限只读诊断，
  写操作明文（否则审查会拦）。
- **无害噪音**：NAT localhost 代理提示、`Failed to translate '\\wsl.localhost\...'`。

出错先读真实报错、能复用不新装、修根因不堆重试；越界（其他发行版/Windows 文件/
全局配置）就停下问。

## 已验证

2026-07-24 `multi-agent`（首建）：迁移+root、python-is-python3、unzip 6.0、
Clash（出口台湾）、Node 24.18.0/npm 11.16.0、Claude Code 2.1.215、CC-Switch 3.17.0
全部实测通过，NAT 直连无需代理。

2026-07-24 `multi-agent` 重建（agent shell 在 Ubuntu24-04 内）：用 `--no-launch` 建站，
故无临时用户、跳过 userdel；每次 `wsl --terminate` 后 interop 掉，靠自愈前缀恢复；
Clash 端口被占改为 7891（从 `runtime/config.yaml` 的 mixed-port 探测），`🇦🇷 阿根廷Z01`
节点 204 通但 geoip 出口实为香港（机场贴错标签）。剩余：Node、Claude Code、CC-Switch。
