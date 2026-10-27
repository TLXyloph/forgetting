"""What candidate code can see and influence inside the example judges.

Candidates run as `python -I -B -c <inline runner>` with a minimal environment: they never get
the hidden judge's path, a data file's path, LETHE_* variables or credentials. word-freq times
each call in the judge's own process (a verified exploit replaced the clock inside the
candidate's process and scored 0.0 ms), on corpora drawn afresh for every evaluation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from example_judges_support import assert_valid, judge_module, run_judge  # noqa: E402

FAST_WORDFREQ = """
import re
from collections import Counter
_W = re.compile(r"[a-z']+")

def top_words(text, k):
    c = Counter(_W.findall(text.lower()))
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
"""
# The reviewer's exploit: replace the clock the old judge used to time the call in-process.
FAKE_CLOCK = """
import __main__ as _m
_ticks = iter([0.0, 1e-9] * 100000)
_m._perf_counter = lambda: next(_ticks)
"""
# Raises (so the candidate fails to load) if the judge leaks anything it should not.
NOSY = """
import os, sys
_seen = " ".join(sys.argv) + " " + " ".join(os.environ)
for _leak in ("score.py", "judge", ".json", "LETHE_", "ANTHROPIC_API_KEY", "AWS_SECRET"):
    assert _leak not in _seen, _leak
"""
SECRETS = {"ANTHROPIC_API_KEY": "sk-test-secret", "AWS_SECRET_ACCESS_KEY": "aws-test-secret"}
FFD = """
def solve(sizes, capacity):
    bins, loads = [], []
    for i in sorted(range(len(sizes)), key=lambda i: (-sizes[i], i)):
        for b, load in enumerate(loads):
            if load + sizes[i] <= capacity:
                bins[b].append(i)
                loads[b] += sizes[i]
                break
        else:
            bins.append([i])
            loads.append(sizes[i])
    return bins
"""


def test_word_freq_candidate_cannot_replace_the_clock_that_times_it(tmp_path: Path) -> None:
    honest = assert_valid(run_judge("word-freq", tmp_path / "a", {"wordfreq.py": FAST_WORDFREQ}))
    hacked = run_judge("word-freq", tmp_path / "b", {"wordfreq.py": FAST_WORDFREQ + FAKE_CLOCK})
    assert assert_valid(hacked)["score"] > 0.5 * honest["score"] > 1.0


def test_word_freq_candidate_sees_no_judge_paths_variables_or_secrets(tmp_path: Path) -> None:
    code = FAST_WORDFREQ + NOSY
    assert_valid(run_judge("word-freq", tmp_path, {"wordfreq.py": code}, extra_env=SECRETS))


def test_word_freq_corpora_are_drawn_afresh_for_every_evaluation(monkeypatch) -> None:
    wf = judge_module("word-freq")
    monkeypatch.setenv("LETHE_SEED", "1")
    first = wf.timing_corpora()
    assert first == wf.timing_corpora()  # a pure function of LETHE_SEED
    assert len(first) == wf.TIMING_RUNS and len(set(first)) == wf.TIMING_RUNS
    monkeypatch.setenv("LETHE_SEED", "2")
    assert set(wf.timing_corpora()).isdisjoint(first)
    assert wf.build_corpus() not in first  # never the pinned reference corpus
    assert all(280_000 < len(wf.WORD_RE.findall(c.lower())) < 320_000 for c in first)


def test_bin_packing_candidate_sees_no_judge_paths_variables_or_secrets(tmp_path: Path) -> None:
    code = FFD + NOSY + "\nprint('stray output on stdout')\n"
    assert_valid(run_judge("bin-packing", tmp_path, {"solver.py": code}, extra_env=SECRETS))


@pytest.mark.parametrize("name", ["bin-packing", "word-freq"])
def test_the_runner_is_inline_and_the_judge_never_runs_itself(name: str) -> None:
    source = judge_module(name).__file__
    text = Path(source).read_text(encoding="utf-8")
    assert '"-s", "-B", "-c"' in text and "__file__).resolve()" not in text
    assert 'env["PYTHONHASHSEED"] = "0"' in text  # deterministic scores (no -I, which drops it)
