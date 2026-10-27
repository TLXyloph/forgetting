"""Run every product example's judge on its seed exactly as lethe would, plus broken candidates.

lethe materializes a judge tree (the seed with the candidate's editable files overlaid), runs
`[judge].run` through a shell with cwd = that tree and the LETHE_* variables set, and reads the last
non-empty stdout line. These tests do the same thing without importing lethe's runner (see
example_judges_support.py).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from example_judges_support import (  # noqa: E402
    ALLOWED_KEYS,
    EXAMPLES,
    NAMES,
    assert_invalid,
    assert_valid,
    judge_module,
    load_config,
    run_judge,
    seed_run,
)

from lethe.workspace.paths import PathFilter  # noqa: E402

# --------------------------------------------------------------------------------------
# config + layout
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_config_follows_schema(name: str) -> None:
    cfg = load_config(name)
    assert set(cfg) == {"task", "judge", "budget"}, "examples set only [task], [judge], [budget]"
    for section, keys in cfg.items():
        assert set(keys) <= ALLOWED_KEYS[section], f"unknown keys in [{section}]"
    task, judge, budget = cfg["task"], cfg["judge"], cfg["budget"]
    assert re.fullmatch(r"[A-Za-z0-9_.-]+", task["name"]) and task["name"] == name
    assert task["description"].strip() and task["editable"]
    seed = EXAMPLES / name / task["seed"]
    files = [p.relative_to(seed).as_posix() for p in seed.rglob("*") if p.is_file()]
    for pattern in task["editable"]:
        assert any(PathFilter([pattern]).matches(f) for f in files), f"{pattern} matches nothing"
    assert judge["direction"] in ("max", "min") and "$LETHE_TASK_DIR/judge/" in judge["run"]
    assert (EXAMPLES / name / "judge" / "score.py").is_file()
    assert not list(seed.rglob("score.py")), "judge code must not live in the seed"
    assert 0 < budget["usd"] <= 10.0, "product examples use a modest budget"


@pytest.mark.parametrize("name", NAMES)
def test_description_states_rules_without_strategy(name: str) -> None:
    text = load_config(name)["task"]["description"].lower()
    for hint in ("local search", "annealing", "heuristic", "hill", "genetic", "try to ", "tip:"):
        assert hint not in text, f"description must not suggest a strategy ({hint!r})"
    assert "maximize" in text or "minimize" in text


# --------------------------------------------------------------------------------------
# every seed is a valid candidate; every broken candidate is rejected
# --------------------------------------------------------------------------------------

BROKEN = {  # example -> (overlay, words the feedback must contain)
    "bin-packing": (
        {"solver.py": "def solve(s, c):\n    return [list(range(len(s)))]\n"},
        "capacity",
    ),
    "word-freq": (
        {"wordfreq.py": "def top_words(t, k):\n    return [('nope', 1)][:k]\n"},
        "incorrect",
    ),
    "regex-dates": ({"pattern.txt": "(\n"}, "compile"),
}


@pytest.mark.parametrize("name", NAMES)
def test_seed_is_a_valid_candidate(name: str, tmp_path_factory: pytest.TempPathFactory) -> None:
    assert assert_valid(seed_run(name, tmp_path_factory))["metrics"]


@pytest.mark.parametrize("name", NAMES)
def test_broken_candidate_is_invalid(name: str, tmp_path: Path) -> None:
    overlay, needle = BROKEN[name]
    assert_invalid(run_judge(name, tmp_path, overlay=overlay), needle)


# --------------------------------------------------------------------------------------
# bin-packing
# --------------------------------------------------------------------------------------

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


def bp_overlay(body: str) -> dict[str, str]:
    return {"solver.py": f"def solve(sizes, capacity):\n    {body}\n"}


def test_bin_packing_reports_24_instances_in_three_distributions(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    res = assert_valid(seed_run("bin-packing", tmp_path_factory))
    cases = res["cases"]
    assert len(cases) == 24 and all(0 < v <= 100 for v in cases.values())
    for dist in ("uniform", "wide", "triplet"):
        assert sum(k.startswith(dist + "-") for k in cases) == 8
    assert res["score"] == pytest.approx(sum(cases.values()) / 24)
    assert "worst" in res["feedback"].lower() and "uniform" in res["feedback"]


def test_bin_packing_better_solver_scores_higher_and_is_deterministic(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    seed = assert_valid(seed_run("bin-packing", tmp_path_factory))
    ffd = assert_valid(run_judge("bin-packing", tmp_path / "a", overlay={"solver.py": FFD}))
    again = assert_valid(run_judge("bin-packing", tmp_path / "b", overlay={"solver.py": FFD}))
    assert seed["score"] + 1.0 < ffd["score"] <= 100.0 and ffd["score"] == again["score"]


@pytest.mark.parametrize(
    ("body", "needle"),
    [
        ("return [[i] for i in range(len(sizes) - 1)]", "missing"),
        ("return [[i] for i in range(len(sizes))] + [[0]]", "twice"),
        ("return [[len(sizes)]] + [[i] for i in range(len(sizes))]", "range"),
        ("return [[float(i)] for i in range(len(sizes))]", "int"),
        ("return 'bins'", "list"),
        ("raise ValueError('boom')", "boom"),
        ("import sys; sys.exit(3)", "exited with code 3"),
    ],
)
def test_bin_packing_rejects_bad_packings(body: str, needle: str, tmp_path: Path) -> None:
    assert_invalid(run_judge("bin-packing", tmp_path, overlay=bp_overlay(body)), needle)


def test_bin_packing_total_timeout_makes_candidate_invalid(tmp_path: Path) -> None:
    code = "import time\n\ndef solve(sizes, capacity):\n    time.sleep(30)\n    return []\n"
    env = {"BIN_PACKING_TIMEOUT_S": "1.5"}
    assert_invalid(run_judge("bin-packing", tmp_path, {"solver.py": code}, env), "timed out")


def test_bin_packing_candidate_noise_and_environment(tmp_path: Path) -> None:
    """Stray prints cannot corrupt the result channel; LETHE_* variables are not passed on."""
    code = FFD + (
        "\nimport os\nprint('debug noise')\n"
        "assert not any(k.startswith('LETHE_') for k in os.environ)\n"
    )
    assert_valid(run_judge("bin-packing", tmp_path, overlay={"solver.py": code}))


# --------------------------------------------------------------------------------------
# word-freq
# --------------------------------------------------------------------------------------

FAST_WORDFREQ = """
import heapq
import re
from collections import Counter

def top_words(text, k):
    counts = Counter(re.findall(r"[a-z']+", text.lower()))
    return heapq.nsmallest(k, counts.items(), key=lambda kv: (-kv[1], kv[0]))
"""
WRONG_TIES = """
import re
from collections import Counter

def top_words(text, k):
    return Counter(re.findall(r"[a-z']+", text.lower())).most_common(k)
"""


def test_word_freq_seed_reports_milliseconds_and_a_fast_solution_beats_it(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    res = assert_valid(seed_run("word-freq", tmp_path_factory))
    metrics = res["metrics"]
    assert metrics["min_ms"] <= res["score"] <= metrics["max_ms"]
    assert res["score"] == pytest.approx(metrics["median_ms"]) and metrics["corpus_words"] > 250_000
    assert load_config("word-freq")["judge"]["direction"] == "min"
    fast = assert_valid(run_judge("word-freq", tmp_path, {"wordfreq.py": FAST_WORDFREQ}))
    assert fast["score"] < res["score"] / 3, (fast["score"], res["score"])


def test_word_freq_candidate_cannot_fake_its_timing(tmp_path: Path) -> None:
    code = FAST_WORDFREQ + "\nimport time\ntime.perf_counter = lambda: 0.0\n"
    res = assert_valid(run_judge("word-freq", tmp_path, {"wordfreq.py": code}))
    assert res["score"] > 5.0, "the call is timed with a clock bound before the import"


def test_word_freq_wrong_tie_break_is_named_by_case_kind_without_contents(tmp_path: Path) -> None:
    res = assert_invalid(run_judge("word-freq", tmp_path, {"wordfreq.py": WRONG_TIES}), "ties")
    for _kind, text, _k in judge_module("word-freq").gate_cases():
        assert len(text) < 12 or text not in res["feedback"]
    assert len(res["feedback"]) < 800


def test_word_freq_reference_and_hidden_cases_follow_the_stated_rules() -> None:
    wf = judge_module("word-freq")
    text = "Don't stop, DON'T! it's 'tis ''' x2y state-of-the-art b a b a c"
    ref = wf.reference(text, 100)
    assert ("don't", 2) in ref and ("'tis", 1) in ref and ("'''", 1) in ref
    split_words = ["x", "y", "state", "of", "the", "art"]
    assert [w for w, _ in ref if w in split_words] == ["art", "of", "state", "the", "x", "y"]
    assert ref[:2] == [("a", 2), ("b", 2)] and wf.reference(text, 0) == wf.reference("", 5) == []
    kinds = {kind for kind, _text, _k in wf.gate_cases()}
    needed = {"empty", "no-words", "ties", "apostrophes", "case", "unicode", "k-zero", "k-large"}
    assert needed | {"separators", "long-word"} <= kinds


# --------------------------------------------------------------------------------------
# regex-dates
# --------------------------------------------------------------------------------------


def exact_date_regex() -> str:
    """An independent, exactly correct pattern for the stated rules (used to test the labels)."""
    year = "(?:19|20)[0-9]{2}"
    leap = "(?:19(?:0[48]|[2468][048]|[13579][26])|20(?:0[048]|[2468][048]|[13579][26]))"
    d31, d30, feb = (
        "(?:0[1-9]|[12][0-9]|3[01])",
        "(?:0[1-9]|[12][0-9]|30)",
        "(?:0[1-9]|1[0-9]|2[0-8])",
    )
    m31, m30 = "(?:0[13578]|1[02])", "(?:0[469]|11)"
    n31, n30 = "(?:Jan|Mar|May|Jul|Aug|Oct|Dec)", "(?:Apr|Jun|Sep|Nov)"
    iso = f"(?:{year}-(?:{m31}-{d31}|{m30}-{d30}|02-{feb})|{leap}-02-29)"
    dmy = f"(?:(?:{d31}/{m31}|{d30}/{m30}|{feb}/02)/{year}|29/02/{leap})"
    txt = f"(?:(?:{d31} {n31}|{d30} {n30}|{feb} Feb) {year}|29 Feb {leap})"
    return f"{iso}|{dmy}|{txt}"


def independent_truth(s: str) -> bool:
    """Date validity via datetime, independent of the judge's own labeling code."""
    import datetime

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    if m := re.fullmatch(r"([0-9]{4})-([0-9]{2})-([0-9]{2})", s):
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
    elif m := re.fullmatch(r"([0-9]{2})/([0-9]{2})/([0-9]{4})", s):
        d, mo, y = int(m[1]), int(m[2]), int(m[3])
    elif m := re.fullmatch(r"([0-9]{2}) (" + "|".join(months) + r") ([0-9]{4})", s):
        d, mo, y = int(m[1]), months.index(m[2]) + 1, int(m[3])
    else:
        return False
    try:
        datetime.date(y, mo, d)
    except ValueError:
        return False
    return 1900 <= y <= 2099


def test_regex_dates_exact_pattern_gets_full_accuracy(tmp_path: Path) -> None:
    pattern = exact_date_regex()
    res = assert_valid(run_judge("regex-dates", tmp_path, {"pattern.txt": pattern + "\n"}))
    assert res["metrics"]["accuracy"] == 1.0
    assert res["score"] == pytest.approx(100.0 - 0.02 * len(pattern))
    assert all(v == 100.0 for v in res["cases"].values())


def test_regex_dates_seed_is_weak_and_feedback_leaks_no_hidden_strings(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    res = assert_valid(seed_run("regex-dates", tmp_path_factory))
    seed_pattern = (EXAMPLES / "regex-dates" / "seed" / "pattern.txt").read_text().strip()
    metrics = res["metrics"]
    assert 0.3 < metrics["accuracy"] < 0.8 and metrics["pattern_length"] == len(seed_pattern)
    assert res["score"] == pytest.approx(100 * metrics["accuracy"] - 0.02 * len(seed_pattern))
    assert res["score"] < 100.0 - 0.02 * len(exact_date_regex()) - 20
    hidden = judge_module("regex-dates").hidden_set()
    assert all(len(s) < 10 or s not in res["feedback"] for s, _label, _group in hidden)
    assert "neg-" in res["feedback"] and set(res["cases"]) == {g for _s, _l, g in hidden}


def test_regex_dates_catastrophic_backtracking_is_invalid(tmp_path: Path) -> None:
    env = {"REGEX_DATES_TIMEOUT_S": "2"}
    run = run_judge("regex-dates", tmp_path, {"pattern.txt": "(?:[0-9]+)+Z\n"}, env)
    assert_invalid(run, "too slow")


def test_regex_dates_hidden_labels_agree_with_independent_truth() -> None:
    hidden = judge_module("regex-dates").hidden_set()
    assert len(hidden) >= 2000 and len({s for s, _l, _g in hidden}) == len(hidden)
    assert 0.4 < sum(label for _s, label, _g in hidden) / len(hidden) < 0.6
    for string, label, group in hidden:
        assert label == independent_truth(string) == group.startswith("pos-"), (group, string)


def test_regex_dates_visible_examples_are_correctly_labeled_and_not_hidden() -> None:
    rd = judge_module("regex-dates")
    text = (EXAMPLES / "regex-dates" / "seed" / "examples.txt").read_text(encoding="utf-8")
    lines = text.removesuffix("\n").split("\n")  # not splitlines(): exotic characters inside
    assert text.endswith("\n") and {"+ 2024-01-31", "- 2024-13-01"} <= set(lines)
    assert all(ln[0] in "+-" and ln[1] == " " for ln in lines)
    visible = {ln[2:]: ln[0] == "+" for ln in lines}
    assert 40 <= len(visible) <= 100 and rd.VISIBLE_SEED != rd.HIDDEN_SEED
    assert all(label == independent_truth(s) for s, label in visible.items())
    assert not {s for s, _l, _g in rd.hidden_set()} & set(visible), "visible examples stay visible"
    assert rd.visible_examples() == list(visible.items())


# --------------------------------------------------------------------------------------
# the fixed benchmarks must never drift (scores are compared across runs and machines)
# --------------------------------------------------------------------------------------

# Digests of the generated benchmark data (identical on Python 3.9-3.14). If a generator is changed
# on purpose, update these: every score ever recorded is relative to this data.
PINNED = {
    "bin-packing": "1031d6dcb4402ab4",
    "word-freq": "b0198ded40fe9977",
    "regex-dates": "00b8a2d57dc85f15",
}


def digest(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def test_benchmarks_are_pinned() -> None:
    bp, wf, rd = (judge_module(n) for n in NAMES)
    actual = {
        "bin-packing": digest([[i.name, i.sizes, i.capacity] for i in bp.build_instances()]),
        "word-freq": digest([wf.build_corpus(), wf.gate_cases()]),
        "regex-dates": digest([rd.hidden_set(), rd.visible_examples()]),
    }
    assert actual == PINNED
