#!/usr/bin/env bash
# Fetch and build AtCoder's official local tester for AHC014 ("RectJoin", the paper's AHC task).
#
# Downloads the contest's tools (Rust sources plus the sample inputs in/0000.txt ...) into the
# git-ignored directory
#   research/tasks/ahc014/vendor/
# unzips them there and builds the official scorer (`vis`) with `cargo build --release`.
# research/tasks/ahc014/judge/score.py runs the candidate on in/0000.txt..0029.txt and scores each
# output with that `vis` binary. Nothing is vendored into the repository.
#
# Needs curl (or wget), unzip, and Rust (install it with rustup) for the build step.
# Idempotent: skips the download when the tools are unpacked and the build when `vis` exists.
#   AHC014_VENDOR_DIR  override the vendor directory (used by the tests)
set -euo pipefail

ZIP_URL="https://img.atcoder.jp/ahc014/a3c240f5b1.zip"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR="${AHC014_VENDOR_DIR:-$HERE/../tasks/ahc014/vendor}"
ZIP="$VENDOR/tools.zip"

# The Cargo.toml of the unpacked tester (the archive layout is not assumed).
find_manifest() {
  find "$VENDOR" -name Cargo.toml -not -path '*/target/*' 2>/dev/null | head -n 1 || true
}

mkdir -p "$VENDOR"
manifest="$(find_manifest)"
if [ -z "$manifest" ]; then
  if [ ! -f "$ZIP" ]; then
    echo "ahc014: downloading $ZIP_URL"
    if command -v curl >/dev/null 2>&1; then
      curl -fsSL -o "$ZIP.part" "$ZIP_URL"
    elif command -v wget >/dev/null 2>&1; then
      wget -q -O "$ZIP.part" "$ZIP_URL"
    else
      echo "ahc014: curl or wget is required to download the tools" >&2
      exit 1
    fi
    mv "$ZIP.part" "$ZIP"
  fi
  if ! command -v unzip >/dev/null 2>&1; then
    echo "ahc014: unzip is required" >&2
    exit 1
  fi
  unzip -q -o "$ZIP" -d "$VENDOR"
  manifest="$(find_manifest)"
  if [ -z "$manifest" ]; then
    echo "ahc014: no Cargo.toml in the downloaded archive; has its layout changed?" >&2
    exit 1
  fi
fi
TOOLS="$(dirname "$manifest")"

if [ -x "$TOOLS/target/release/vis" ]; then
  echo "ahc014: the tester is already built at $TOOLS/target/release/vis"
  exit 0
fi
if ! command -v cargo >/dev/null 2>&1; then
  echo "ahc014: cargo was not found. Install Rust (rustup) to build the official tester:" >&2
  echo "          curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh" >&2
  echo "        then run this script again. The unpacked sources and inputs are in $TOOLS" >&2
  exit 1
fi

(cd "$TOOLS" && cargo build --release)
if [ ! -x "$TOOLS/target/release/vis" ]; then
  echo "ahc014: the build finished but $TOOLS/target/release/vis is missing" >&2
  exit 1
fi
echo "ahc014: built $TOOLS/target/release/vis"
