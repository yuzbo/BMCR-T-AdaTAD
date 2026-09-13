#!/usr/bin/env bash
# One coding-agent invocation, no automatic GPU jobs. Run roles sequentially in a shared worktree.
set -euo pipefail
[[ $# == 3 ]] || { echo "Usage: $0 WORKTREE RUN_ROOT ROLE_FILE_NAME" >&2; exit 2; }
BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORKTREE="$(cd "$1" && pwd -P)"
RUN_ROOT="$(realpath -m -- "$2")"
ROLE="$3"
[[ "$ROLE" =~ ^[0-9][0-9]_[a-z_]+\.md$ ]] || { echo 'Invalid role filename' >&2; exit 2; }
[[ -f "$BUNDLE/agents/$ROLE" && -f "$WORKTREE/AGENTS.md" ]] || { echo 'Missing role or repository AGENTS.md' >&2; exit 2; }
command -v codex >/dev/null || { echo 'Codex CLI is not present. Use the Markdown role prompt in your existing coding-agent session.' >&2; exit 2; }
HELP="$(codex exec --help)"
for FLAG in --sandbox --output-last-message; do
  grep -q -- "$FLAG" <<< "$HELP" || { echo "Installed Codex does not advertise $FLAG; review its help manually." >&2; exit 2; }
done
GLOBAL_HELP="$(codex --help)"
grep -q -- --add-dir <<< "$GLOBAL_HELP" || { echo 'Installed Codex lacks advertised --add-dir; use role prompt manually without weakening sandbox.' >&2; exit 2; }
case "$RUN_ROOT/" in "$WORKTREE/"*) echo 'RUN_ROOT must be outside source worktree' >&2; exit 2;; esac
mkdir -p "$RUN_ROOT/agents"
NAME="${ROLE%.md}-$(date +%Y%m%d-%H%M%S)-$$"
PROMPT="$RUN_ROOT/agents/$NAME.prompt.md"
{
  printf '# 本次执行上下文\n\n研究包：%s\n源码工作树：%s\n外部输出目录：%s\n\n' "$BUNDLE" "$WORKTREE" "$RUN_ROOT"
  echo '先阅读研究包的 REPORT_zh.md、contracts/route_v1.json、experiments.json，以及源码 AGENTS.md 和 RTK.md。不要启动 GPU 训练/评估，不要访问或复制任何凭证。只执行当前角色；不要自动连跑后续角色。'
  cat "$BUNDLE/agents/$ROLE"
} > "$PROMPT"
(
  cd "$WORKTREE"
  codex --add-dir "$RUN_ROOT" exec --sandbox workspace-write \
    --output-last-message "$RUN_ROOT/agents/$NAME.result.md" - < "$PROMPT"
) 2>&1 | tee "$RUN_ROOT/agents/$NAME.log"
