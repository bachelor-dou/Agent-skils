#!/usr/bin/env bash
# 把 CI 产出的文件提交并推送到目标分支。两个 workflow（每日快照 / 每周榜单）共用。
#
# 用法: commit_and_push.sh "<提交信息>" <路径>...
# 目标分支取 $TARGET_BRANCH，默认 main —— 数据产物由 CI 直接推 main；人工改动走特性分支 PR。
#
# 冲突处理用「重放」而不是 rebase：
#   拉最新目标分支 → 硬重置 → 把本次产物拷回去 → 重新提交 → 推送，失败就再来一轮。
#   1) checkout 默认浅克隆（depth=1），rebase 常因找不到 merge base 而失败，重放只需要最新一版；
#   2) CI 是这些产物（快照 / DB / 报告）的唯一写入方，直接以本次产物为准即为正确结果。
#   注意：因此不要手改 main 上这些产物文件（快照/DB/报告），否则会被下一次 CI 覆盖。
#
# 目录参数按「以本次产物为准」处理（先删后拷），这样脚本内的过期清理
# （如快照保留 35 天）产生的删除才能真正进 git，否则重置会把旧文件带回来。
set -euo pipefail

branch=${TARGET_BRANCH:-main}
msg=$1
shift
[ $# -gt 0 ] || { echo "用法: commit_and_push.sh <提交信息> <路径>..." >&2; exit 2; }

stash=$(mktemp -d)
for p in "$@"; do
  [ -e "$p" ] || continue
  mkdir -p "$(dirname "$stash/$p")"
  cp -a "$p" "$stash/$p"
done

git config user.name  "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

for attempt in 1 2 3; do
  git fetch --depth=1 -q origin "$branch"
  git reset --hard -q FETCH_HEAD

  for p in "$@"; do
    [ -e "$stash/$p" ] || continue
    if [ -d "$stash/$p" ]; then
      rm -rf "$p"
    fi
    mkdir -p "$(dirname "$p")"
    cp -a "$stash/$p" "$p"
  done

  git add -A -- "$@"
  if git diff --cached --quiet; then
    echo "产物与 $branch 上现有内容一致，无需提交。"
    exit 0
  fi
  git commit -q -m "$msg"

  if git push -q origin "HEAD:$branch"; then
    echo "推送成功（第 $attempt 轮）→ $branch"
    exit 0
  fi
  echo "第 $attempt 轮推送失败（$branch 期间有新提交），拉最新后重放。"
  sleep 5
done

echo "::error::连续 3 轮推送失败，本次产物未入库。"
exit 1
