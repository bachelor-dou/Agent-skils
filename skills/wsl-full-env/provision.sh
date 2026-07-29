#!/usr/bin/env bash
# wsl-full-env 一键 provisioner。从任意 WSL 发行版里跑（用 wsl.exe/powershell.exe interop
# 管理目标发行版）。幂等：可重复运行，已完成的步骤会跳过。
#
#   bash provision.sh <名字>                 # 新建/迁移/仅root + 装全栈 + 克隆 clash
#   bash provision.sh <名字> --verify-clash  # 用户跑完 clash install.sh 后，探测端口并验证
#
# 只在两个不可自动化的密钥点停：Clash 订阅链接、CC-Switch 个人配置导入。
set -euo pipefail

DISTRO="${1:?usage: provision.sh <distro-name> [--verify-clash]}"
MODE="${2:-provision}"
IMAGE='Ubuntu-24.04'
DEST_WIN='E:\env\WSL'
CLAUDE_VER='2.1.215'
CCSWITCH_DEB='https://github.com/farion1231/cc-switch/releases/download/v3.17.0/CC-Switch-v3.17.0-Linux-x86_64.deb'

[[ "$DISTRO" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "环境名仅允许 字母数字 . _ -"; exit 1; }

# agent shell 若在别的发行版里，每次 wsl --terminate 会冲掉本发行版的 WSLInterop，
# 导致 wsl.exe/powershell.exe 报 Exec format error。每次 interop 调用前自愈。
heal() { [ -e /proc/sys/fs/binfmt_misc/WSLInterop ] || echo ':WSLInterop:M::MZ::/init:PF' > /proc/sys/fs/binfmt_misc/register 2>/dev/null || true; }
psx()  { heal; powershell.exe -NoProfile -Command "$1"; }
inroot() { heal; wsl.exe -d "$DISTRO" -u root -- bash -s; }   # 脚本走 stdin，免多层引号

registered() { heal; powershell.exe -NoProfile -Command "wsl --list --quiet" | tr -d '\0\r' | grep -qx "$DISTRO"; }
on_e() { [ "$(psx "Test-Path '$DEST_WIN\\$DISTRO\\ext4.vhdx'" | tr -d '\r\n ')" = "True" ]; }

verify_clash() {
  echo "[verify] 开代理 → 探测 mixed-port → 测 204 + geoip"
  inroot <<'VERIFY'
set -e
clashctl on >/dev/null 2>&1 || clashctl on
P=$(grep -oE 'mixed-port: *[0-9]+' /opt/clash-for-linux/runtime/config.yaml | grep -oE '[0-9]+' | head -1)
echo "PORT=$P"
echo -n '204: '; curl -o /dev/null -s -w '%{http_code}\n' --max-time 20 --proxy "http://127.0.0.1:$P" https://www.gstatic.com/generate_204
echo -n 'geoip: '; curl -s --max-time 20 --proxy "http://127.0.0.1:$P" https://api.ip.sb/geoip; echo
VERIFY
  echo "204=代理通；geoip 的 country 是真实出口（节点名可能贴错标签）。"
  echo "要换节点：wsl -d $DISTRO -u root -- clashctl select <策略组> <节点名>"
}

if [ "$MODE" = "--verify-clash" ]; then
  verify_clash
  exit 0
fi

# --- 1+2  新建 + 迁移到 E 盘（幂等：E 盘已有 ext4.vhdx 就跳过）---
if on_e; then
  echo "[skip] $DEST_WIN\\$DISTRO\\ext4.vhdx 已存在，跳过新建+迁移"
else
  if ! registered; then
    echo "[create] wsl --install $IMAGE --name $DISTRO --no-launch"
    psx "wsl --install $IMAGE --name $DISTRO --no-launch"   # --no-launch 免临时用户/免 OOBE
  fi
  echo "[migrate] 导出 → unregister → 导入到 $DEST_WIN\\$DISTRO"
  TAR="$DEST_WIN\\$DISTRO\\$DISTRO.tar"
  psx "wsl --terminate $DISTRO; New-Item -ItemType Directory -Force '$DEST_WIN\\$DISTRO' | Out-Null; wsl --export $DISTRO '$TAR'; wsl --unregister $DISTRO; wsl --import $DISTRO '$DEST_WIN\\$DISTRO' '$TAR' --version 2; if (Test-Path -LiteralPath '$DEST_WIN\\$DISTRO\\ext4.vhdx') { Remove-Item -LiteralPath '$TAR' } else { throw 'ext4.vhdx missing; keep tar' }"
fi

# --- 3+4  仅 root + 全栈（一次 stdin 喂进去，全程 -u root）---
echo "[stack] wsl.conf(root) + python/unzip + node24 + claude-code + cc-switch + clash(clone)"
inroot <<STACK
set -e
export DEBIAN_FRONTEND=noninteractive
cat > /etc/wsl.conf <<'CONF'
[boot]
systemd=true

[user]
default=root
CONF
apt-get -o DPkg::Lock::Timeout=300 update
apt-get -o DPkg::Lock::Timeout=300 install -y unzip python-is-python3
install -d -m 0755 /etc/apt/keyrings
curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor --yes -o /etc/apt/keyrings/nodesource.gpg
echo 'deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_24.x nodistro main' > /etc/apt/sources.list.d/nodesource.list
apt-get -o DPkg::Lock::Timeout=300 update
apt-get -o DPkg::Lock::Timeout=300 install -y nodejs
npm install --global @anthropic-ai/claude-code@${CLAUDE_VER}
curl -fL '${CCSWITCH_DEB}' -o /tmp/cc-switch.deb
apt-get -o DPkg::Lock::Timeout=300 install -y /tmp/cc-switch.deb
rm -f /tmp/cc-switch.deb
[ -d /opt/clash-for-linux/.git ] || git clone --branch master --depth 1 https://github.com/wnlen/clash-for-linux.git /opt/clash-for-linux
echo '=== VERSIONS ==='
readlink -f "\$(command -v python)"; python --version
node --version; npm --version
claude --version
dpkg -s cc-switch | grep Version
STACK

psx "wsl --terminate $DISTRO" >/dev/null 2>&1 || true   # 让 default=root 生效

cat <<EOF

======================================================================
 全栈已装完。剩两个手动密钥点（无法自动化）：

 [1/2] Clash 订阅（机场密钥，别贴聊天/日志）
       wsl -d $DISTRO -u root -- bash -c 'cd /opt/clash-for-linux && bash install.sh'
       跑完验证：bash provision.sh $DISTRO --verify-clash

 [2/2] CC-Switch 配置导入（GUI）
       wsl -d $DISTRO      # 进去后： cc-switch &   导入你的模型配置
======================================================================
EOF
