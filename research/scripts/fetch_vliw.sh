#!/usr/bin/env bash
# Fetch Anthropic's original performance take-home (the paper's "VLIW kernel" task).
#
# The repository has no licence and its header forbids publishing solutions, so lethe never
# vendors it: this script clones it into the git-ignored directory
#   research/tasks/vliw/vendor/original_performance_takehome
# which research/tasks/vliw/lethe.toml uses as the task's seed, and checks out the pinned COMMIT
# below. The judge's 147,734-cycle baseline, its pinned sha256 of tests/frozen_problem.py and the
# task text all describe that commit. Judge trees and agent workspaces are temporary copies of the
# clone; nothing from it is ever copied into this repository.
#
# Idempotent: does nothing when the clone is already there at the pinned commit, and refuses a
# clone at any other commit.
#   VLIW_VENDOR_DIR  override the vendor directory (used by the tests)
#   VLIW_REPO_URL    override where to clone from (used by the tests, with a local copy)
set -euo pipefail

COMMIT="5452f74bd977807ac2e74f3d29432b9df6f25197"
REPO_URL="${VLIW_REPO_URL:-https://github.com/anthropics/original_performance_takehome}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR="${VLIW_VENDOR_DIR:-$HERE/../tasks/vliw/vendor}"
DEST="$VENDOR/original_performance_takehome"

# The commit a clone is at; nothing for a directory that is not a clone itself (never the HEAD of
# a repository around it).
head_of() {
  if [ -e "$1/.git" ]; then
    git -C "$1" rev-parse --verify -q HEAD 2>/dev/null || true
  fi
}

if [ -f "$DEST/perf_takehome.py" ]; then
  found="$(head_of "$DEST")"
  if [ "$found" = "$COMMIT" ]; then
    echo "vliw: already present at $DEST (commit ${COMMIT:0:12})"
    exit 0
  fi
  echo "vliw: $DEST is not a clone at the pinned commit $COMMIT (found: ${found:-no git clone});" \
    "remove it and run this script again" >&2
  exit 1
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
# Clone into a private temporary directory; only a checkout verified to be at COMMIT is moved
# into place, so a failed or interrupted run leaves nothing behind.
TMP="$(mktemp -d "$VENDOR/.vliw-clone.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
git clone --quiet "$REPO_URL" "$TMP/repo"
if ! git -C "$TMP/repo" -c advice.detachedHead=false checkout --quiet --detach "$COMMIT"; then
  echo "vliw: $REPO_URL has no commit $COMMIT; has the upstream history changed?" >&2
  exit 1
fi
if [ "$(head_of "$TMP/repo")" != "$COMMIT" ] || [ ! -f "$TMP/repo/perf_takehome.py" ]; then
  echo "vliw: the checkout of $COMMIT has no perf_takehome.py; has the upstream layout changed?" >&2
  exit 1
fi
mv "$TMP/repo" "$DEST"
echo "vliw: cloned commit ${COMMIT:0:12} to $DEST"
