#!/usr/bin/env bash
# Create an isolated worktree. Never reset, clean, stash, fetch or overwrite the user's checkout.
set -euo pipefail
[[ $# == 3 ]] || { echo "Usage: $0 EXISTING_REPO NEW_WORKTREE EXTERNAL_RUN_ROOT" >&2; exit 2; }
BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
BASE=04c35a3b76897e6c1569eeede41ed3aecaf7f854
REPO="$(git -C "$1" rev-parse --show-toplevel)"
REPO="$(cd "$REPO" && pwd -P)"
WORKTREE="$(realpath -m -- "$2")"
RUN_ROOT="$(realpath -m -- "$3")"
[[ ! -e "$WORKTREE" ]] || { echo "Refusing existing worktree path: $WORKTREE" >&2; exit 2; }
for ROOT in "$REPO" "$WORKTREE"; do
  case "$RUN_ROOT/" in "$ROOT/"*) echo "RUN_ROOT must be outside source worktrees" >&2; exit 2;; esac
done
git -C "$REPO" cat-file -e "$BASE^{commit}" || {
  echo "Anchor is unavailable locally. Obtain and verify that exact commit before continuing." >&2; exit 2;
}
BRANCH="${H65_DS3_BRANCH:-research/h65-ds3-$(date +%Y%m%d-%H%M%S)}"
if git -C "$REPO" show-ref --verify --quiet "refs/heads/$BRANCH"; then
  echo "Refusing existing branch: $BRANCH" >&2; exit 2
fi
mkdir -p "$RUN_ROOT/audit" "$RUN_ROOT/agents" "$RUN_ROOT/manifests" "$RUN_ROOT/outputs"
git -C "$REPO" status --porcelain > "$RUN_ROOT/audit/original_worktree_status.txt"
git -C "$REPO" rev-parse HEAD > "$RUN_ROOT/audit/original_head.txt"
git -C "$REPO" worktree add -b "$BRANCH" "$WORKTREE" "$BASE"
python "$BUNDLE/bin/source_audit.py" --repo "$WORKTREE" --out "$RUN_ROOT/audit/anchor.json"
printf 'WORKTREE=%q\nRUN_ROOT=%q\nBUNDLE=%q\nBASE=%q\n' "$WORKTREE" "$RUN_ROOT" "$BUNDLE" "$BASE" > "$RUN_ROOT/session.env"
echo "Created $WORKTREE on $BRANCH. No H65 training, evaluation or agent was started."
