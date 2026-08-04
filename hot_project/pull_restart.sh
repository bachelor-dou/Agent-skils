#!/usr/bin/env bash
# 更新代码并重启本地 Web 服务。
# 使用方式：bash hot_project/pull_restart.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_PY="$REPO/.venv/bin/python3"
ENV_FILE="/root/.hot_projects.env"
LOG_DIR="$SCRIPT_DIR/logs"
SERVER_OUT="$LOG_DIR/server.out"
# 匹配 Web 服务进程，排除 cron 子模块。
PROC_PAT="-m hot_project$"
PORT=8001

cd "$REPO"
mkdir -p "$LOG_DIR"

# 暂存已跟踪的本地改动。
STASHED=0
if ! git diff --quiet || ! git diff --cached --quiet; then
    git stash push -q -m "pull_restart $(date '+%F %T')"
    STASHED=1
    echo "已暂存本地改动"
fi

# 快进拉取并恢复本地改动。
git pull -q --ff-only || { echo "拉取失败(有未推送提交或已分叉),未重启。"; [ "$STASHED" = 0 ] || git stash pop -q; exit 1; }
echo "已拉取合并"
if [ "$STASHED" = 1 ]; then
    git stash pop -q || { echo "stash pop 冲突,改动仍在 git stash list,未重启。"; exit 1; }
    echo "已恢复暂存改动"
fi

# 加载运行环境并重启服务。
if pkill -f -- "$PROC_PAT" 2>/dev/null; then sleep 1; echo "已停止旧服务"; else echo "无正在运行的服务"; fi
[ -f "$ENV_FILE" ] && { set -a; . "$ENV_FILE"; set +a; } || echo "警告:$ENV_FILE 不存在,缺 token / key"
nohup "$VENV_PY" -m hot_project >>"$SERVER_OUT" 2>&1 &
PID=$!

# 检查服务可用性。
for _ in $(seq 20); do curl -sf -o /dev/null "http://127.0.0.1:$PORT/" && break; sleep 0.5; done
if curl -sf -o /dev/null "http://127.0.0.1:$PORT/"; then
    echo "已重启 pid=$PID 端口 $PORT"
else
    echo "启动失败,$SERVER_OUT 末尾:"; tail -n 15 "$SERVER_OUT"; exit 1
fi
