#!/usr/bin/env bash
# Local repository preparation only. Does not fetch weights, submit or cancel jobs.
set -euo pipefail
BASE=239d098cd899936c35989fae6243c70259a85adb
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 /absolute/path/to/BMCR-T-AdaTAD /absolute/path/to/new-worktree" >&2
  exit 2
fi
REPO=$1; DEST=$2
[[ "$REPO" = /* && "$DEST" = /* ]] || { echo 'Use absolute paths' >&2; exit 2; }
git -C "$REPO" rev-parse --git-dir >/dev/null
if ! git -C "$REPO" cat-file -e "$BASE^{commit}" 2>/dev/null; then
  echo "Fixed commit missing. Review and run: git -C '$REPO' fetch origin '$BASE'" >&2
  exit 3
fi
[[ ! -e "$DEST" ]] || { echo 'Destination already exists; refusing overwrite' >&2; exit 4; }
# Detached worktree avoids silently reusing a pre-existing scientific branch.
git -C "$REPO" worktree add --detach "$DEST" "$BASE"
test "$(git -C "$DEST" rev-parse HEAD)" = "$BASE"
echo "Prepared fixed worktree: $DEST"
echo 'No experiment was launched. Create a fresh branch here before agent edits.'
