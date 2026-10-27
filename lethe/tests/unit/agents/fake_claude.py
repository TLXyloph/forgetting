"""Stand-in for the `claude` CLI, driven by FAKE_CLAUDE_* environment variables.

conftest.py copies this file into tmp_path behind a `#!<sys.executable>` shebang and points
`ClaudeCodeOptions.executable` at it, so the tests exercise a real subprocess without ever
starting the real CLI (no network, no credentials). It is not a test module.

  FAKE_CLAUDE_RECORD         directory to record into: argv.json, cwd.txt, env.json, stdin.bin,
                             child.pid
  FAKE_CLAUDE_STREAM         file whose bytes are written to stdout (a recorded stream)
  FAKE_CLAUDE_STRIP_NEWLINE  "1": drop the stream's final newline (a line cut off mid-write)
  FAKE_CLAUDE_BIG_LINE       N: write one extra JSON line of about N bytes before the stream
  FAKE_CLAUDE_STDERR         text written to stderr
  FAKE_CLAUDE_STDERR_FILE    file whose bytes are written to stderr (for large output)
  FAKE_CLAUDE_EXIT           exit status (default 0)
  FAKE_CLAUDE_SKIP_STDIN     "1": never read stdin (exit while the caller is still writing)
  FAKE_CLAUDE_CHILD          "group": spawn a sleeping grandchild in the same process group
  FAKE_CLAUDE_HANG           seconds to sleep after the output has been written and flushed
  FAKE_CLAUDE_SELF_KILL      "1": SIGKILL itself at the very end instead of exiting normally
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _record(directory: Path | None, name: str, data: bytes) -> None:
    """Write atomically so a test polling for the file never sees a half-written one."""
    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / f"{name}.{os.getpid()}.tmp"  # per-process: concurrent fakes share a dir
    tmp.write_bytes(data)
    os.replace(tmp, directory / name)


def main() -> int:
    env = os.environ
    record = Path(env["FAKE_CLAUDE_RECORD"]) if env.get("FAKE_CLAUDE_RECORD") else None
    _record(record, "argv.json", json.dumps(sys.argv[1:]).encode("utf-8"))
    _record(record, "cwd.txt", os.getcwd().encode("utf-8"))
    _record(record, "env.json", json.dumps(sorted(os.environ)).encode("utf-8"))  # names only

    if env.get("FAKE_CLAUDE_SKIP_STDIN") != "1":
        _record(record, "stdin.bin", sys.stdin.buffer.read())

    if env.get("FAKE_CLAUDE_CHILD") == "group":
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _record(record, "child.pid", str(child.pid).encode("ascii"))

    out = sys.stdout.buffer
    big = int(env.get("FAKE_CLAUDE_BIG_LINE", "0"))
    if big:
        out.write(json.dumps({"type": "user", "blob": "x" * big}).encode("utf-8") + b"\n")
    stream = env.get("FAKE_CLAUDE_STREAM")
    if stream:
        payload = Path(stream).read_bytes()
        if env.get("FAKE_CLAUDE_STRIP_NEWLINE") == "1":
            payload = payload.rstrip(b"\n")
        out.write(payload)
    out.flush()

    if env.get("FAKE_CLAUDE_STDERR"):
        sys.stderr.write(env["FAKE_CLAUDE_STDERR"])
        sys.stderr.flush()
    if env.get("FAKE_CLAUDE_STDERR_FILE"):
        sys.stderr.buffer.write(Path(env["FAKE_CLAUDE_STDERR_FILE"]).read_bytes())
        sys.stderr.buffer.flush()

    hang = float(env.get("FAKE_CLAUDE_HANG", "0"))
    if hang:
        time.sleep(hang)
    if env.get("FAKE_CLAUDE_SELF_KILL") == "1":
        os.kill(os.getpid(), signal.SIGKILL)  # like the OOM killer: no exit status, a signal
    return int(env.get("FAKE_CLAUDE_EXIT", "0"))


if __name__ == "__main__":
    sys.exit(main())
