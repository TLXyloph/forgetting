#!/usr/bin/env bash
# Fetch Anthropic's original performance take-home (the paper's "VLIW kernel" task).
#
# The repository has no licence and its header forbids publishing solutions, so lethe never
# vendors it: this script clones it (depth 1) into the git-ignored directory
#   research/tasks/vliw/vendor/original_performance_takehome
# which research/tasks/vliw/lethe.toml uses as the task's seed. Judge trees and agent workspaces
# are temporary copies of that clone; nothing from it is ever copied into this repository.
#
# Idempotent: does nothing when the clone is already there.
# Override the vendor directory with VLIW_VENDOR_DIR (used by the tests).
set -euo pipefail

REPO_URL="https://github.com/anthropics/original_performance_takehome"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR="${VLIW_VENDOR_DIR:-$HERE/../tasks/vliw/vendor}"
DEST="$VENDOR/original_performance_takehome"

if [ -f "$DEST/perf_takehome.py" ]; then
  echo "vliw: already present at $DEST"
  exit 0
fi
if [ -e "$DEST" ]; then
  echo "vliw: $DEST exists but has no perf_takehome.py; remove it and run this script again" >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "vliw: git is required to fetch the task" >&2
  exit 1
fi

mkdir -p "$VENDOR"
# git removes the destination directory itself when a clone fails, so a rerun starts clean.
git clone --depth 1 "$REPO_URL" "$DEST"
if [ ! -f "$DEST/perf_takehome.py" ]; then
  echo "vliw: the clone has no perf_takehome.py; has the upstream layout changed?" >&2
  exit 1
fi
echo "vliw: cloned commit $(git -C "$DEST" rev-parse --short HEAD) to $DEST"
