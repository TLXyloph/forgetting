"""Candidate code runs with a fixed string-hash seed, so identical candidates score identically."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from example_judges_support import EXAMPLES, assert_valid, load_config, run_judge  # noqa: E402

# Refuses to run when str hashing is randomized: set iteration order (and so tie-breaking in
# schedulers, packers, counters) would then differ between two evaluations of the same code.
GUARD = (
    "import sys\nif sys.flags.hash_randomization:\n    raise SystemExit('hash seed not fixed')\n"
)


@pytest.mark.parametrize(
    ("name", "editable"), [("bin-packing", "solver.py"), ("word-freq", "wordfreq.py")]
)
def test_candidates_run_with_a_fixed_hash_seed(name, editable, tmp_path):
    seed_dir = EXAMPLES / name / load_config(name)["task"]["seed"]
    source = (seed_dir / editable).read_text(encoding="utf-8")
    assert_valid(run_judge(name, tmp_path, overlay={editable: source + "\n" + GUARD}))
