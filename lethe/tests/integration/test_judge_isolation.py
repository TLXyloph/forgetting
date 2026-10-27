"""Judge trees live outside the run: code the judge executes cannot reach the run directory
through a relative path (a verified forgetting bypass), and lethe keeps no judge directory there.

The candidate below is the reviewer's exploit: after a clean forget it reads its pre-forget
notes through `../../../ws/a1/e1/notes.txt` (relative to the judge tree) and crashes, so the judge
puts the text in its feedback. Candidate code still runs with the user's permissions, so an
absolute path would work: that is the trust model `lethe check` and `lethe run` print.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

from lethe.agents.scripted import ScriptCall, ScriptedBackend, ScriptReply
from lethe.api import load_run, run_task

JUDGE = textwrap.dedent(
    """
    import json, subprocess, sys
    done = subprocess.run([sys.executable, "solver.py"], capture_output=True, text=True)
    if done.returncode != 0:
        print(json.dumps({"valid": False, "feedback": done.stderr[-600:]}))
    else:
        print(json.dumps({"score": int(done.stdout)}))
    """
)
SECRET = "PRE-FORGET-SECRET: idea 7 works"
LEAK = textwrap.dedent(
    """
    import sys
    sys.stderr.write(open("../../../ws/a1/e1/notes.txt").read())
    sys.exit(1)
    """
)


def make_task(tmp_path: Path) -> Path:
    task = tmp_path / "task"
    (task / "seed").mkdir(parents=True)
    (task / "judge").mkdir()
    (task / "seed" / "solver.py").write_text("print(1)\n")
    (task / "judge" / "score.py").write_text(JUDGE)
    (task / "lethe.toml").write_text(
        textwrap.dedent(f"""
        [task]
        name = "leak"
        description = "Print a big number."
        seed = "seed"
        editable = ["solver.py"]

        [judge]
        run = '"{sys.executable}" "$LETHE_TASK_DIR/judge/score.py"'
        direction = "max"
        timeout = 30
        """)
    )
    return task / "lethe.toml"


def agent(call: ScriptCall) -> ScriptReply:
    if call.workspace.name == "e1":
        (call.workspace / "notes.txt").write_text(SECRET + "\n")
        (call.workspace / "solver.py").write_text(f"print({call.turn_in_session + 1})\n")
    else:
        (call.workspace / "solver.py").write_text(
            LEAK if call.turn_in_session == 1 else "print(9)\n"
        )
    return ScriptReply(final_text="ok")


def test_a_candidate_cannot_reach_the_run_directory_through_the_judge_tree(tmp_path):
    run_dir = run_task(
        make_task(tmp_path),
        overrides={
            "team": {"agents": 1, "peers": "none"},
            "forget": {
                "carry_scratch": False,
                "when": [{"type": "budget_fractions", "fractions": [0.5]}],
            },
            "budget": {"usd": 0, "proposals": 4, "primary": "proposals"},
        },
        runs_dir=tmp_path / "runs",
        run_name="r",
        backend=ScriptedBackend(agent),
    )

    state = load_run(run_dir)
    assert [r.scope for r in state.epoch_history] == ["start", "team"]
    leak = next(c for c in state.candidates.values() if c.epoch == 2)
    assert leak.valid is False and "No such file" in leak.feedback  # the path went nowhere
    assert SECRET not in (run_dir / "events.jsonl").read_text()
    assert not (run_dir / "judge").exists()  # judge trees are made outside the run
