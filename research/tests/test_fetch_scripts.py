"""The fetch scripts, offline: pinning, verification and idempotence. Nothing here touches the
network: the take-home is cloned from the local vendor copy when there is one."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
VLIW_VENDOR = Path(__file__).resolve().parents[1] / "tasks" / "vliw" / "vendor"
VLIW_VENDOR = VLIW_VENDOR / "original_performance_takehome"
PINNED = "5452f74bd977807ac2e74f3d29432b9df6f25197"
needs_vliw = pytest.mark.skipif(not VLIW_VENDOR.is_dir(), reason="fetch_vliw.sh not run")


def run_script(
    script: str, env_var: str, vendor: Path, path: str | None = None, **extra: str
) -> Any:
    env = {**os.environ, env_var: str(vendor), **({"PATH": path} if path else {}), **extra}
    return subprocess.run(
        ["bash", str(SCRIPTS / script)], env=env, capture_output=True, text=True, timeout=120
    )


def git(cwd: Path, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1")
    env.update(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@x", GIT_COMMITTER_NAME="t")
    env.update(GIT_COMMITTER_EMAIL="t@x")
    done = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def check_script(script: str) -> None:
    path = SCRIPTS / script
    assert os.access(path, os.X_OK) and path.read_text().startswith("#!/usr/bin/env bash")
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0


def test_fetch_vliw_refuses_an_incomplete_or_unpinned_clone(tmp_path: Path) -> None:
    check_script("fetch_vliw.sh")
    dest = tmp_path / "original_performance_takehome"
    dest.mkdir()
    done = run_script("fetch_vliw.sh", "VLIW_VENDOR_DIR", tmp_path)
    assert done.returncode == 1 and "perf_takehome.py" in done.stderr
    (dest / "perf_takehome.py").write_text("")
    done = run_script("fetch_vliw.sh", "VLIW_VENDOR_DIR", tmp_path)
    assert done.returncode == 1 and PINNED in done.stderr and "no git clone" in done.stderr
    git(dest, "init", "-q")
    git(dest, "add", "perf_takehome.py")
    git(dest, "commit", "-q", "-m", "some other commit")
    done = run_script("fetch_vliw.sh", "VLIW_VENDOR_DIR", tmp_path)
    assert (
        done.returncode == 1
        and PINNED in done.stderr
        and git(dest, "rev-parse", "HEAD") in (done.stderr)
    )


@needs_vliw
def test_fetch_vliw_checks_out_the_pinned_commit_and_is_then_idempotent(tmp_path: Path) -> None:
    assert git(VLIW_VENDOR, "rev-parse", "HEAD") == PINNED  # the clone the judges were pinned to
    assert f'COMMIT="{PINNED}"' in (SCRIPTS / "fetch_vliw.sh").read_text()
    local = {"VLIW_REPO_URL": str(VLIW_VENDOR)}

    done = run_script("fetch_vliw.sh", "VLIW_VENDOR_DIR", tmp_path, **local)

    assert done.returncode == 0, done.stderr
    assert git(tmp_path / "original_performance_takehome", "rev-parse", "HEAD") == PINNED
    again = run_script("fetch_vliw.sh", "VLIW_VENDOR_DIR", tmp_path, **local)
    assert again.returncode == 0 and "already present" in again.stdout
    assert sorted(p.name for p in tmp_path.iterdir()) == ["original_performance_takehome"]


def test_fetch_ahc014_explains_how_to_get_rust_and_is_idempotent(tmp_path: Path) -> None:
    check_script("fetch_ahc014.sh")
    if shutil.which("cargo", path="/usr/bin:/bin"):
        pytest.skip("cargo lives in /usr/bin on this machine")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "Cargo.toml").write_text("[package]\n")
    no_cargo = run_script("fetch_ahc014.sh", "AHC014_VENDOR_DIR", tmp_path, "/usr/bin:/bin")
    assert no_cargo.returncode == 1 and "Install Rust (rustup)" in no_cargo.stderr
    vis = tmp_path / "tools" / "target" / "release" / "vis"
    vis.parent.mkdir(parents=True)
    vis.write_text("#!/bin/sh\n")
    vis.chmod(0o755)
    built = run_script("fetch_ahc014.sh", "AHC014_VENDOR_DIR", tmp_path, "/usr/bin:/bin")
    assert built.returncode == 0 and "already built" in built.stdout


def test_fetch_ahc014_refuses_an_archive_that_is_not_the_pinned_one(tmp_path: Path) -> None:
    (tmp_path / "tools.zip").write_bytes(b"PK\x03\x04 not the contest's archive")

    done = run_script("fetch_ahc014.sh", "AHC014_VENDOR_DIR", tmp_path, "/usr/bin:/bin")

    assert done.returncode == 1 and "sha256" in done.stderr and "not the pinned" in done.stderr
    assert not list(tmp_path.rglob("Cargo.toml"))  # nothing unverified was unpacked
