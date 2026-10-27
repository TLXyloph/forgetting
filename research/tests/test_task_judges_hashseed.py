"""The VLIW judge builds and runs kernels with a fixed string-hash seed (deterministic scores)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from task_judges_support import VLIW_VENDOR, assert_valid, needs_vliw, run_judge  # noqa: E402

GUARD = (
    "import sys\nif sys.flags.hash_randomization:\n    raise SystemExit('hash seed not fixed')\n"
)


@needs_vliw
def test_vliw_builds_with_a_fixed_hash_seed(tmp_path):
    source = (VLIW_VENDOR / "perf_takehome.py").read_text(encoding="utf-8")
    result = assert_valid(
        run_judge("vliw", tmp_path, overlay={"perf_takehome.py": source + "\n" + GUARD})
    )
    assert result["score"] == 147734
