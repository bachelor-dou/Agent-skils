#!/usr/bin/env bash
# 从远端 snapshots 分支拉取 CI 产出的每日 star 快照，供周三出榜使用。
#
# 只 checkout snapshots 这一个目录，不做 git pull / merge —— 本地工作区常有未提交改动，
# 整仓 pull 会失败或触发合并；取单个 path 只覆盖 CI 有的那些天，
# 本地兜底跑出来的、远端还没有的那天不会被删。
#
# checkout <ref> -- <path> 会把文件同时暂存进 index，而该路径在 main 上是 .gitignore 忽略的，
# 若不处理，本地 git add . 时会把快照误提交进 main。故取完立刻 git reset 该路径：
# 文件留在工作区可读，但从 index 撤下 → 重新变回「被忽略的未跟踪文件」，git status 里不可见。
#
# 永远以 0 退出：拉不到（断网、CI 还没跑过、分支还没建）也要让出榜继续，
# 用旧锚点出榜远好过不出榜。
#
# 用法（周三出榜前）：
#   0 13 * * 3 . /root/.hot_projects.env && /root/code/Agent-skils/hot_projects/sync_snapshots.sh
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SNAPSHOT_PATH="hot_projects/data/snapshots"
BRANCH="snapshots"

cd "$REPO_DIR" || exit 0

if ! git fetch --quiet origin "$BRANCH" 2>/dev/null; then
    echo "[sync_snapshots] fetch origin/$BRANCH 失败（断网？或 CI 尚未建分支），沿用本地已有快照。"
    exit 0
fi

if ! git cat-file -e "origin/$BRANCH:$SNAPSHOT_PATH" 2>/dev/null; then
    echo "[sync_snapshots] 远端 $BRANCH 上还没有 $SNAPSHOT_PATH（CI 尚未推过），跳过。"
    exit 0
fi

before=$(ls -1 "$SNAPSHOT_PATH" 2>/dev/null | wc -l)
if git checkout "origin/$BRANCH" -- "$SNAPSHOT_PATH"; then
    # 从 index 撤下（文件保留在工作区），避免污染 main 的提交
    git reset -q -- "$SNAPSHOT_PATH" 2>/dev/null || true
    after=$(ls -1 "$SNAPSHOT_PATH" | wc -l)
    echo "[sync_snapshots] 已同步：本地快照 $before → $after 份，最新 $(ls -1 "$SNAPSHOT_PATH" | tail -1)。"
else
    echo "[sync_snapshots] checkout 失败，沿用本地已有快照。"
fi
exit 0
