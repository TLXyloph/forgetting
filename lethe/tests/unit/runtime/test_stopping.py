"""`interrupt_on_termination`: a background or killed run stops like Ctrl-C, cancelling turns."""

from __future__ import annotations

import signal
import subprocess
import sys
import time

from lethe.api import interrupt_on_termination

CHILD = """
import asyncio, sys
from lethe.api import interrupt_on_termination

async def run():
    try:
        print("running", flush=True)
        await asyncio.sleep(60)
    except asyncio.CancelledError:
        print("cancelled cleanly", flush=True)  # where the orchestrator ends "interrupted"
        raise

interrupt_on_termination()
try:
    asyncio.run(run())
except KeyboardInterrupt:
    sys.exit(130)
"""


def test_it_restores_the_ctrl_c_handler_and_maps_sigterm_onto_it() -> None:
    saved = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # as a shell starts a background command
        interrupt_on_termination()
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
        assert callable(signal.getsignal(signal.SIGTERM))
    finally:
        signal.signal(signal.SIGINT, saved[0])
        signal.signal(signal.SIGTERM, saved[1])


def run_child_and_signal(sig: int) -> tuple[int, str]:
    def ignore_sigint() -> None:  # what `cmd &` in a non-interactive shell does to the child
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    proc = subprocess.Popen(
        [sys.executable, "-c", CHILD],
        stdout=subprocess.PIPE,
        text=True,
        preexec_fn=ignore_sigint,  # noqa: PLW1509 (test-only, single-threaded spawn)
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "running"
    time.sleep(0.2)
    proc.send_signal(sig)
    out, _ = proc.communicate(timeout=20)
    return proc.returncode, out


def test_sigterm_cancels_a_background_run_cleanly() -> None:
    code, out = run_child_and_signal(signal.SIGTERM)
    assert "cancelled cleanly" in out and code == 130


def test_ctrl_c_works_even_when_the_shell_started_the_run_with_sigint_ignored() -> None:
    code, out = run_child_and_signal(signal.SIGINT)
    assert "cancelled cleanly" in out and code == 130
