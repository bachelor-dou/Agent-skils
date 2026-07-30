#!/usr/bin/env bash
# commit_and_push.sh 的自检：在临时 git 仓库里跑真实推送，覆盖三件容易写错的事。
#   1. 常规新增文件能推上去；
#   2. 推送被拒（目标分支期间有别人的提交）时重放能成功，且不吞掉别人的改动；
#   3. 目录参数里的「删除」（快照过期清理）能真正进 git——重置会把旧文件带回来，
#      靠脚本里的 rm -rf 才能保住删除。去掉那行本用例即失败。
# 不设 TARGET_BRANCH，走默认的 snapshots，顺带验证默认分支名没写错。
# 用法: bash .github/scripts/test_commit_and_push.sh
set -euo pipefail

script=$(cd "$(dirname "$0")" && pwd)/commit_and_push.sh
branch=snapshots
root=$(mktemp -d)
trap 'rm -rf "$root"' EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

# -b：裸仓库的 HEAD 也得指向目标分支，否则克隆方会当成空仓库
git init -q --bare -b "$branch" "$root/remote.git"
git -c init.defaultBranch="$branch" init -q "$root/seed"
(
  cd "$root/seed"
  git config user.email t@t; git config user.name t
  mkdir -p snaps
  echo old > snaps/2026-01-01.txt
  echo base > other.txt
  git add -A && git commit -q -m base
  git push -q "$root/remote.git" "HEAD:$branch"
)

clone() {  # 浅克隆，和 actions/checkout 默认行为一致
  git clone -q --depth=1 "file://$root/remote.git" "$root/$1"
  git -C "$root/$1" config user.email t@t
  git -C "$root/$1" config user.name t
}
remote_has() { git -C "$root/remote.git" cat-file -e "$branch:$1" 2>/dev/null; }

# ── 1. 常规新增 ──────────────────────────────────────────────
clone w1
(cd "$root/w1" && echo new > snaps/2026-01-02.txt && "$script" "add snap" snaps >/dev/null)
remote_has snaps/2026-01-02.txt || fail "用例1：新增文件没进 $branch"
echo "ok 1  常规新增已推送"

# ── 2. 推送被拒后重放 ────────────────────────────────────────
clone w2
clone racer
# pre-push 钩子：让 w2 的第一次 push 之前，racer 先抢先推一个提交，制造 non-fast-forward
cat > "$root/w2/.git/hooks/pre-push" <<HOOK
#!/usr/bin/env bash
marker="$root/raced"
[ -e "\$marker" ] && exit 0
touch "\$marker"
cd "$root/racer"
echo racer > racer.txt
git add -A && git commit -q -m racer && git push -q origin "HEAD:$branch"
HOOK
chmod +x "$root/w2/.git/hooks/pre-push"

(cd "$root/w2" && echo mine > snaps/2026-01-03.txt && "$script" "add snap" snaps > "$root/out2")
[ -e "$root/raced" ] || fail "用例2：钩子没触发，没造出冲突"
grep -q "第 2 轮" "$root/out2" || fail "用例2：没走到第 2 轮重放（输出：$(cat "$root/out2")）"
remote_has snaps/2026-01-03.txt || fail "用例2：重放后自己的文件丢了"
remote_has racer.txt           || fail "用例2：重放覆盖了别人的提交"
echo "ok 2  冲突后重放成功，双方改动都在"

# ── 3. 过期清理的删除要能进 git ──────────────────────────────
clone w3
(cd "$root/w3" && rm snaps/2026-01-01.txt && echo x > snaps/2026-01-04.txt \
   && "$script" "prune" snaps >/dev/null)
remote_has snaps/2026-01-04.txt && ! remote_has snaps/2026-01-01.txt \
  || fail "用例3：过期文件的删除没进 $branch（重置把它带回来了）"
echo "ok 3  目录内的删除已同步"

echo "全部通过（目标分支 $branch）"
