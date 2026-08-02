#!/usr/bin/env bash
# 手动更新并重启本地 Web 服务:暂存改动 → 快进拉取远端 → 恢复改动 → 重启进程。
# 直接执行即可:bash scripts/pull_restart.sh
set -euo pipefail

REPO="/root/code/Agent-skils"
VENV_PY="$REPO/.venv/bin/python3"
ENV_FILE="/root/.hot_projects.env"          # 运行时机密:GITHUB_TOKENS / LLM_*_KEY
LOG_DIR="$REPO/hot_project/logs"
SERVER_OUT="$LOG_DIR/server.out"
# 精确匹配服务进程:cmdline 以 "-m hot_project" 结尾,避开 "-m hot_project.cron_*"
PROC_PAT="-m hot_project$"
PORT=8001

cd "$REPO"
mkdir -p "$LOG_DIR"

# 1) 暂存已跟踪的未提交改动(未跟踪文件不挡快进,不碰)
STASHED=0
if ! git diff --quiet || ! git diff --cached --quiet; then
    git stash push -q -m "pull_restart $(date '+%F %T')"
    STASHED=1
    echo "已暂存本地改动"
fi

# 2) 快进拉取 + 恢复改动。pop 冲突意味着工作区有冲突标记,别拿它顶掉正在跑的服务
git pull -q --ff-only || { echo "拉取失败(有未推送提交或已分叉),未重启。"; [ "$STASHED" = 0 ] || git stash pop -q; exit 1; }
echo "已拉取合并"
if [ "$STASHED" = 1 ]; then
    git stash pop -q || { echo "stash pop 冲突,改动仍在 git stash list,未重启。"; exit 1; }
    echo "已恢复暂存改动"
fi

# 3) 重启(先 source 机密环境变量)
if pkill -f -- "$PROC_PAT" 2>/dev/null; then sleep 1; echo "已停止旧服务"; else echo "无正在运行的服务"; fi
[ -f "$ENV_FILE" ] && { set -a; . "$ENV_FILE"; set +a; } || echo "警告:$ENV_FILE 不存在,缺 token / key"
nohup "$VENV_PY" -m hot_project >>"$SERVER_OUT" 2>&1 &
PID=$!

# 4) 探首页确认真起来了(进程刚死会变僵尸,kill -0 查不出来)
for _ in $(seq 20); do curl -sf -o /dev/null "http://127.0.0.1:$PORT/" && break; sleep 0.5; done
if curl -sf -o /dev/null "http://127.0.0.1:$PORT/"; then
    echo "已重启 pid=$PID 端口 $PORT"
else
    echo "启动失败,$SERVER_OUT 末尾:"; tail -n 15 "$SERVER_OUT"; exit 1
fi
