"""Hidden judge for the regex-dates example (a non-program artifact). Python 3.11+, stdlib only.

It never runs inside an agent workspace; lethe reaches it through $LETHE_TASK_DIR and runs it with
cwd = the judge tree (the seed with the candidate's editable files overlaid).

The candidate is a single regular expression in `pattern.txt`. It is compiled and applied with
`re.fullmatch` to a hidden labeled set of about 3000 strings inside a SUBPROCESS with a timeout
(catastrophic backtracking makes the candidate invalid; REGEX_DATES_TIMEOUT_S overrides the 10 s
limit, which the tests use). The set is generated from a fixed seed that differs from the seed of
the visible `examples.txt`; both come from the same generator, so the labels agree with `truth`.

    score = 100 * accuracy - 0.02 * len(pattern)          (maximize)

`python3 score.py --emit-visible` prints the visible examples (how `seed/examples.txt` was made).
Prints exactly one JSON line; exit 75 means the judge itself broke, never the candidate.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import random
import re
import signal
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX
    resource = None  # type: ignore[assignment]

VISIBLE_SEED = 7
HIDDEN_SEED = 20261026
DEFAULT_TIMEOUT_S = 10.0
MAX_PATTERN_CHARS = 100_000
LENGTH_PENALTY = 0.02
MAX_FILE_BYTES = 64 * 1024 * 1024

MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
FULL_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]  # fmt: skip
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
STYLES = ("iso", "dmy", "mon")
POS_GROUP = {"iso": "pos-iso", "dmy": "pos-dmy", "mon": "pos-mon"}
HIDDEN_COUNTS = {
    "pos-iso": 500, "pos-dmy": 500, "pos-mon": 500,
    "neg-month": 200, "neg-day": 250, "neg-leap": 150, "neg-year": 200,
    "neg-format": 300, "neg-extra": 200, "neg-unicode": 100, "neg-long": 100,
}  # fmt: skip
VISIBLE_COUNTS = {
    "pos-iso": 10, "pos-dmy": 10, "pos-mon": 10,
    "neg-month": 4, "neg-day": 5, "neg-leap": 4, "neg-year": 4,
    "neg-format": 6, "neg-extra": 4, "neg-unicode": 2, "neg-long": 1,
}  # fmt: skip
ANCHORS = [("2024-01-31", True), ("2024-13-01", False)]  # always the first visible lines


# ---------------------------------------------------------------------------------------------
# ground truth
# ---------------------------------------------------------------------------------------------


def is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def month_len(year: int, month: int) -> int:
    return 29 if month == 2 and is_leap(year) else MONTH_DAYS[month - 1]


_ISO = re.compile(r"([0-9]{4})-([0-9]{2})-([0-9]{2})")
_DMY = re.compile(r"([0-9]{2})/([0-9]{2})/([0-9]{4})")
_MON = re.compile(r"([0-9]{2}) (" + "|".join(MON) + r") ([0-9]{4})")


def truth(s: str) -> bool:
    """Is `s` a valid date in one of the three formats (ASCII digits, year 1900-2099)?"""
    if m := _ISO.fullmatch(s):
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
    elif m := _DMY.fullmatch(s):
        d, mo, y = int(m[1]), int(m[2]), int(m[3])
    elif m := _MON.fullmatch(s):
        d, mo, y = int(m[1]), MON.index(m[2]) + 1, int(m[3])
    else:
        return False
    return 1900 <= y <= 2099 and 1 <= mo <= 12 and 1 <= d <= month_len(y, mo)


def fmt(style: str, y: int, m: int | str, d: int | str) -> str:
    """Write a date; `m` may be a string to build malformed look-alikes (numeric styles)."""
    mm = f"{m:02d}" if isinstance(m, int) else m
    dd = f"{d:02d}" if isinstance(d, int) else d
    if style == "iso":
        return f"{y:04d}-{mm}-{dd}"
    if style == "dmy":
        return f"{dd}/{mm}/{y:04d}"
    month = MON[m - 1] if isinstance(m, int) else m
    return f"{dd} {month} {y:04d}"


# ---------------------------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------------------------


class Rng:
    """Deterministic PRNG facade over `random.Random(seed).random()` (a guaranteed stream)."""

    def __init__(self, seed: int) -> None:
        self.random = random.Random(seed).random

    def randint(self, lo: int, hi: int) -> int:
        return lo + min(hi - lo, int(self.random() * (hi - lo + 1)))

    def pick(self, seq: list[Any] | tuple[Any, ...]) -> Any:
        return seq[self.randint(0, len(seq) - 1)]

    def chance(self, p: float) -> bool:
        return self.random() < p

    def shuffle(self, items: list[Any]) -> None:
        for i in range(len(items) - 1, 0, -1):
            j = self.randint(0, i)
            items[i], items[j] = items[j], items[i]


LEAP_YEARS = [y for y in range(1900, 2100) if is_leap(y)]
COMMON_YEARS = [y for y in range(1900, 2100) if not is_leap(y) and y != 1900]


def valid_date(rng: Rng) -> tuple[int, int, int]:
    y, m = rng.randint(1900, 2099), rng.randint(1, 12)
    return y, m, rng.randint(1, month_len(y, m))


def positive(rng: Rng, style: str, i: int) -> str:
    if i % 12 == 0:  # a leap day
        return fmt(style, rng.pick(LEAP_YEARS), 2, 29)
    y, m = rng.randint(1900, 2099), rng.randint(1, 12)
    if i % 5 == 1:  # a first or last day, sometimes at the edge of the year range
        y = rng.pick([1900, 2099, y])
        return fmt(style, y, m, rng.pick([1, month_len(y, m)]))
    return fmt(style, y, m, rng.randint(1, month_len(y, m)))


def neg_month(rng: Rng, style: str) -> str:
    y, d = rng.randint(1900, 2099), rng.randint(1, 28)
    if style == "mon":
        bad = (
            "Jnu Fbe Mrc Aip Mya Jly Agu Spe Otc Nvo Dce Sept Janu Marc Mai Okt Dez Abr Ene Foo Xyz"
        )
        return fmt(style, y, rng.pick(bad.split()), d)
    month = rng.randint(13, 31) if rng.chance(0.6) else rng.pick([0, *range(32, 100)])
    return fmt(style, y, f"{month:02d}", d)


def neg_day(rng: Rng, style: str) -> str:
    y, kind = rng.randint(1900, 2099), rng.random()
    if kind < 0.1:
        return fmt(style, y, rng.randint(1, 12), "00")
    if kind < 0.4:
        return fmt(style, y, rng.randint(1, 12), rng.randint(32, 99))
    if kind < 0.7:
        return fmt(style, y, rng.pick([4, 6, 9, 11]), 31)
    return fmt(style, y, 2, rng.pick([30, 31]))


def neg_leap(rng: Rng, style: str) -> str:
    return fmt(style, 1900 if rng.chance(0.2) else rng.pick(COMMON_YEARS), 2, 29)


def neg_year(rng: Rng, style: str) -> str:
    m = rng.randint(1, 12)
    d = rng.randint(1, 28)
    if rng.chance(0.3):
        return fmt(style, rng.pick([1899, 2100]), m, d)
    y = rng.randint(0, 9999 - 200)
    return fmt(style, y if y < 1900 else y + 200, m, d)


def neg_format(rng: Rng, _style: str) -> str:
    y, m, d = valid_date(rng)
    small_m, small_d = rng.randint(1, 9), rng.randint(1, 9)
    big_d = rng.randint(13, 28)
    month = MON[m - 1]
    forms = [
        f"{y:04d}/{m:02d}/{d:02d}", f"{y:04d}.{m:02d}.{d:02d}", f"{y:04d} {m:02d} {d:02d}",
        f"{d:02d}-{m:02d}-{y:04d}", f"{d:02d}.{m:02d}.{y:04d}", f"{m:02d}/{big_d:02d}/{y:04d}",
        f"{y:04d}-{small_m}-{d:02d}", f"{y:04d}-{m:02d}-{small_d}", f"{small_d}/{m:02d}/{y:04d}",
        f"{d:02d}/{small_m}/{y:04d}", f"{small_d} {month} {y:04d}",
        f"{d:02d} {month.lower()} {y:04d}", f"{d:02d} {month.upper()} {y:04d}",
        f"{d:02d} {FULL_MONTHS[(m + 5) % 12]} {y:04d}", f"{month} {d:02d}, {y:04d}",
        f"{d:02d}{'st' if d % 10 == 1 else 'th'} {month} {y:04d}",
        f"{d:02d}  {month} {y:04d}", f"{d:02d} {month}  {y:04d}", f"{d:02d}-{month}-{y:04d}",
        f"{y:04d}{m:02d}{d:02d}", f"{d:02d}{m:02d}{y:04d}", f"{y % 100:02d}-{m:02d}-{d:02d}",
        f"{d:02d}/{m:02d}/{y % 100:02d}", f"{d:02d} {month} {y % 100:02d}",
        f"{y:04d}-{month}-{d:02d}", f"{y:04d}-{m:02d}/{d:02d}", f"{d:02d}/{m:02d}-{y:04d}",
        f"{d:02d} {m:02d} {y:04d}", f"{y:04d}-{m:02d}-{d:02d}-", f"{d:02d}/{m:02d}/",
    ]  # fmt: skip
    return rng.pick(forms)


def neg_extra(rng: Rng, style: str) -> str:
    s = fmt(style, *valid_date(rng))
    t = fmt(rng.pick(STYLES), *valid_date(rng))
    decorated = [
        "x" + s, s + "x", f"({s})", s + ".", s + ",", s + "T00:00", "1" + s, s + "1", s + "-",
        "-" + s, s + " UTC", "on " + s, s + "!", s + s, f"{s}/{t}", f"{s} - {t}", f"{s};{t}",
        f"[{s}]", f'"{s}"', s[:-1], s[1:], "#" + s, s + "th", "~" + s,
    ]  # fmt: skip
    return rng.pick(decorated)


def neg_unicode(rng: Rng, style: str) -> str:
    s = fmt(style, *valid_date(rng))
    kind = rng.random()
    if kind < 0.6:  # digits from other scripts (matched by \d, not by [0-9])
        base = rng.pick([0x0660, 0x06F0, 0x0966, 0xFF10, 0x09E6, 0x0E50])
        swapped = "".join(chr(base + int(c)) if c.isdigit() and rng.chance(0.7) else c for c in s)
        return swapped if swapped != s else chr(base + int(s[0])) + s[1:]
    if kind < 0.85:  # look-alike separators
        table = {"-": "\u2010\u2011\u2212", "/": "\uff0f\u2215", " ": "\u00a0\u2003\u3000"}
        for i in range(len(s)):
            if s[i] in table and rng.chance(0.8):
                return s[:i] + rng.pick(table[s[i]]) + s[i + 1 :]
        return s.replace("0", "\u0660", 1)
    homoglyphs = {"a": "\u0430", "o": "\u043e", "e": "\u0435", "c": "\u0441"}  # Cyrillic
    out = "".join(homoglyphs.get(c, c) if c.isalpha() else c for c in s)
    return out if out != s else s.replace("0", "\uff10", 1)


def neg_long(rng: Rng, _style: str) -> str:
    n = rng.randint(10, 40)
    kind = rng.randint(0, 5)
    if kind == 0:
        return rng.pick(["1", "9"]) * n
    if kind == 1:
        return "2024-" * (n // 5 + 2) + "01"
    if kind == 2:
        return "0" * n + "-"
    if kind == 3:
        return "31/01/2024/" * (n // 8 + 2)
    if kind == 4:
        return "9" * n + "x"
    return "31 Jan 2024 " * (n // 8 + 2)


NEGATIVES = {
    "neg-month": neg_month, "neg-day": neg_day, "neg-leap": neg_leap, "neg-year": neg_year,
    "neg-format": neg_format, "neg-extra": neg_extra, "neg-unicode": neg_unicode,
    "neg-long": neg_long,
}  # fmt: skip


def build_set(
    seed: int, counts: dict[str, int], avoid: frozenset[str] | set[str] = frozenset()
) -> list[tuple[str, bool, str]]:
    """Unique (string, label, group) triples; labels are checked against `truth`."""
    rng = Rng(seed)
    seen = set(avoid)
    out: list[tuple[str, bool, str]] = []
    for group, count in counts.items():
        for i in range(count):
            for _ in range(10_000):
                if group.startswith("pos-"):
                    style = next(s for s, g in POS_GROUP.items() if g == group)
                    s = positive(rng, style, i)
                else:
                    s = NEGATIVES[group](rng, rng.pick(STYLES))
                label = group.startswith("pos-")
                if s not in seen and truth(s) == label:
                    break
            else:
                raise RuntimeError(f"could not generate a fresh {group} string")
            seen.add(s)
            out.append((s, label, group))
    rng.shuffle(out)
    return out


def visible_examples() -> list[tuple[str, bool]]:
    anchors = {s for s, _ in ANCHORS}
    extra = build_set(VISIBLE_SEED, VISIBLE_COUNTS, avoid=anchors)
    return [*ANCHORS, *((s, label) for s, label, _group in extra)]


def hidden_set() -> list[tuple[str, bool, str]]:
    visible = {s for s, _ in visible_examples()}
    return build_set(HIDDEN_SEED, HIDDEN_COUNTS, avoid=visible)


# ---------------------------------------------------------------------------------------------
# judging
# ---------------------------------------------------------------------------------------------


def _limits(cpu_seconds: int) -> None:  # runs in the child between fork and exec
    if resource is not None:
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_BYTES, MAX_FILE_BYTES))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 5))


def run_pattern(pattern: str, strings: list[str], timeout: float) -> dict[str, Any]:
    """Compile and apply the pattern in a subprocess; returns its report or {'timeout': True}."""
    with tempfile.TemporaryDirectory(prefix="regexdates-judge-") as tmp:
        scratch = Path(tmp)
        (scratch / "pattern.txt").write_text(pattern, encoding="utf-8")
        (scratch / "strings.json").write_text(json.dumps(strings), encoding="utf-8")
        argv = [sys.executable, "-P", str(Path(__file__).resolve()), "--child"]
        argv += [
            str(scratch / "pattern.txt"),
            str(scratch / "strings.json"),
            str(scratch / "out.json"),
        ]
        env = {k: v for k, v in os.environ.items() if not k.startswith("LETHE_")}
        proc = subprocess.Popen(
            argv,
            cwd=scratch,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            preexec_fn=functools.partial(_limits, max(60, int(timeout * 8))),
        )
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"timeout": True}
        finally:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        try:
            return json.loads((scratch / "out.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"crash": f"the regex engine process exited with code {proc.returncode}"}


def invalid(feedback: str, **metrics: Any) -> dict[str, Any]:
    return {"valid": False, "feedback": feedback, "metrics": metrics}


def read_pattern(path: Path) -> tuple[str | None, str]:
    """(pattern, error). Trailing line breaks are not part of the pattern."""
    try:
        text = path.read_bytes().decode("utf-8")
    except OSError:
        return None, "pattern.txt is missing"
    except UnicodeDecodeError:
        return None, "pattern.txt is not valid UTF-8"
    pattern = text.rstrip("\r\n")
    if len(pattern) > MAX_PATTERN_CHARS:
        return None, f"pattern.txt is too long ({len(pattern)} > {MAX_PATTERN_CHARS} characters)"
    return pattern, ""


def judge(cwd: Path, timeout: float) -> dict[str, Any]:
    pattern, problem = read_pattern(cwd / "pattern.txt")
    if pattern is None:
        return invalid(problem)
    hidden = hidden_set()
    report = run_pattern(pattern, [s for s, _label, _group in hidden], timeout)
    if report.get("timeout"):
        return invalid(
            f"The pattern is too slow: matching the hidden strings did not finish within "
            f"{timeout:g} s (catastrophic backtracking?).",
            pattern_length=len(pattern),
        )
    if "compile_error" in report:
        return invalid(
            f"The pattern does not compile: {report['compile_error']}", pattern_length=len(pattern)
        )
    flags = report.get("flags")
    if not isinstance(flags, list) or len(flags) != len(hidden):
        return invalid(report.get("crash", "The regex run produced no usable result."))
    right: dict[str, int] = {}
    total: dict[str, int] = {}
    false_pos: dict[str, int] = {}
    false_neg: dict[str, int] = {}
    for (_s, label, group), flag in zip(hidden, flags, strict=True):
        total[group] = total.get(group, 0) + 1
        if bool(flag) == label:
            right[group] = right.get(group, 0) + 1
        elif flag:
            false_pos[group] = false_pos.get(group, 0) + 1
        else:
            false_neg[group] = false_neg.get(group, 0) + 1
    correct = sum(right.values())
    accuracy = correct / len(hidden)
    score = 100.0 * accuracy - LENGTH_PENALTY * len(pattern)
    lines = [
        f"Accuracy {100 * accuracy:.2f}% ({correct}/{len(hidden)}), pattern length {len(pattern)} "
        f"-> score {score:.3f} = 100*accuracy - {LENGTH_PENALTY}*length."
    ]
    if false_pos:
        lines.append(
            "Accepted but not valid dates: "
            + ", ".join(f"{g} {n}/{total[g]}" for g, n in sorted(false_pos.items()))
            + "."
        )
    if false_neg:
        lines.append(
            "Valid dates rejected: "
            + ", ".join(f"{g} {n}/{total[g]}" for g, n in sorted(false_neg.items()))
            + "."
        )
    if not false_pos and not false_neg:
        lines.append("Every hidden string is classified correctly.")
    return {
        "valid": True,
        "score": round(score, 6),
        "feedback": " ".join(lines),
        "metrics": {
            "accuracy": round(accuracy, 6),
            "pattern_length": len(pattern),
            "strings": len(hidden),
            "false_positives": sum(false_pos.values()),
            "false_negatives": sum(false_neg.values()),
        },
        "cases": {g: round(100.0 * right.get(g, 0) / total[g], 4) for g in HIDDEN_COUNTS},
    }


def child_main(pattern_path: str, strings_path: str, out_path: str) -> int:
    """Runs in the subprocess: compile the pattern and classify every string."""
    pattern = Path(pattern_path).read_text(encoding="utf-8")
    strings = json.loads(Path(strings_path).read_text(encoding="utf-8"))
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        report: dict[str, Any] = {"compile_error": f"{exc.msg} (position {exc.pos})"}
    except Exception as exc:  # RecursionError, OverflowError, ... from absurd patterns
        report = {"compile_error": f"{type(exc).__name__}"}
    else:
        report = {"flags": [1 if compiled.fullmatch(s) else 0 for s in strings]}
    Path(out_path).write_text(json.dumps(report), encoding="utf-8")
    return 0


def main() -> None:
    timeout = float(os.environ.get("REGEX_DATES_TIMEOUT_S", DEFAULT_TIMEOUT_S))
    sys.stdout.write(json.dumps(judge(Path.cwd(), timeout), allow_nan=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--child":
        sys.exit(child_main(sys.argv[2], sys.argv[3], sys.argv[4]))
    if sys.argv[1:] == ["--emit-visible"]:
        sys.stdout.write("".join(f"{'+' if ok else '-'} {s}\n" for s, ok in visible_examples()))
        sys.exit(0)
    try:
        main()
    except Exception:  # the judge itself broke: lethe retries and does not charge the candidate
        traceback.print_exc()
        sys.exit(75)
