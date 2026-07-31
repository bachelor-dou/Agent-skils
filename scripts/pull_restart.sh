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

cd "$REPO"
mkdir -p "$LOG_DIR"

# 1) 暂存已跟踪的未提交改动(不碰未跟踪文件 —— 它们不挡快进,无需暂存;无改动则跳过)
STASHED=0
if ! git diff --quiet || ! git diff --cached --quiet; then
    git stash push -m "pull_restart $(date '+%F %T')"
    STASHED=1
    echo "已暂存本地改动"
fi

# 2) 快进拉取(不做合并提交,避免意外分叉)
if ! git pull --ff-only; then
    echo "拉取失败:本地可能有未推送提交或与远端分叉,已中止,未重启。"
    [ "$STASHED" = 1 ] && git stash pop && echo "已恢复暂存改动"
    exit 1
fi

# 3) 恢复改动
if [ "$STASHED" = 1 ]; then
    if git stash pop; then echo "已恢复暂存改动"
    else echo "恢复暂存改动时发生冲突,请手动解决(git stash list 可查看)。"; fi
fi

# 4) 停掉旧服务
if pkill -f -- "$PROC_PAT" 2>/dev/null; then echo "已停止旧服务"; sleep 1; else echo "无正在运行的服务"; fi

# 5) 重启(先 source 机密环境变量)
if [ -f "$ENV_FILE" ]; then set -a; . "$ENV_FILE"; set +a
else echo "警告:$ENV_FILE 不存在,服务将缺少 token / key"; fi
nohup "$VENV_PY" -m hot_project >>"$SERVER_OUT" 2>&1 &
echo "服务已重启 pid=$! (端口 8001,输出 → $SERVER_OUT)"
