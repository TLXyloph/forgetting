"""Small statistics toolkit, standard library only.

Provides what the experiment analysis needs for comparing arms with a handful of replicates:
mean and sample standard deviation, Welch's unequal-variance t-test, a percentile bootstrap
confidence interval for a difference of means, and an exact one-sided Mann-Whitney test.

The Student-t distribution is evaluated through the regularized incomplete beta function,
computed with the continued fraction of Numerical Recipes (section 6.4, modified Lentz method),
so no special-function library is needed.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "MANN_WHITNEY_EXACT_MAX_N",
    "WelchResult",
    "betainc",
    "bootstrap_diff_ci",
    "mann_whitney_exact_p",
    "mann_whitney_u",
    "mean",
    "sample_sd",
    "student_t_cdf",
    "student_t_two_sided_p",
    "welch_t_test",
]

_CF_EPS = 1e-15
_CF_TINY = 1e-300
_CF_MAX_ITER = 20_000


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean, summed with `math.fsum` so rounding error does not accumulate."""
    if len(values) == 0:
        raise ValueError("mean of an empty sequence is undefined")
    return math.fsum(values) / len(values)


def _sample_variance(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        raise ValueError("sample variance needs at least two values")
    m = mean(values)
    return math.fsum((v - m) ** 2 for v in values) / (n - 1)


def sample_sd(values: Sequence[float]) -> float:
    """Sample standard deviation (n - 1 in the denominator)."""
    return math.sqrt(_sample_variance(values))


# ---------------------------------------------------------------------------------------------
# regularized incomplete beta function and the Student-t distribution
# ---------------------------------------------------------------------------------------------


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Continued fraction for I_x(a, b), evaluated with the modified Lentz method."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _CF_TINY:
        d = _CF_TINY
    d = 1.0 / d
    h = d
    for m in range(1, _CF_MAX_ITER + 1):
        m2 = 2 * m
        # even step of the recurrence
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _CF_TINY:
            d = _CF_TINY
        c = 1.0 + aa / c
        if abs(c) < _CF_TINY:
            c = _CF_TINY
        d = 1.0 / d
        h *= d * c
        # odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _CF_TINY:
            d = _CF_TINY
        c = 1.0 + aa / c
        if abs(c) < _CF_TINY:
            c = _CF_TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _CF_EPS:
            return h
    raise ArithmeticError(f"incomplete beta continued fraction did not converge (a={a}, b={b})")


def _betainc(a: float, b: float, x: float, y: float) -> float:
    """I_x(a, b) where `y` is 1 - x supplied by the caller (it can be known more precisely)."""
    if not (a > 0.0 and b > 0.0):
        raise ValueError(f"betainc needs a > 0 and b > 0, got a={a}, b={b}")
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"betainc needs 0 <= x <= 1, got x={x}")
    if x == 0.0:
        return 0.0
    if y == 0.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(y)
    )
    front = math.exp(log_front)
    # the fraction converges quickly only on one side of the mean; reflect when needed
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, y) / b


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b) for a, b > 0 and 0 <= x <= 1."""
    return _betainc(a, b, x, 1.0 - x)


def student_t_two_sided_p(t: float, df: float) -> float:
    """P(|T| >= |t|) for a Student-t variable with `df` degrees of freedom (df may be fractional).

    Uses P = I_x(df / 2, 1 / 2) with x = df / (df + t^2), which stays accurate in the far tail.
    """
    if not 0.0 < df < math.inf:
        raise ValueError(f"degrees of freedom must be positive and finite, got {df}")
    if math.isnan(t):
        return math.nan
    if math.isinf(t):
        return 0.0
    if t == 0.0:
        return 1.0
    t_sq = t * t
    return _betainc(df / 2.0, 0.5, df / (df + t_sq), t_sq / (df + t_sq))


def student_t_cdf(t: float, df: float) -> float:
    """P(T <= t) for a Student-t variable with `df` degrees of freedom."""
    if t == 0.0:
        return 0.5
    upper_tail = 0.5 * student_t_two_sided_p(t, df)  # P(T >= |t|)
    return 1.0 - upper_tail if t > 0 else upper_tail


# ---------------------------------------------------------------------------------------------
# Welch's t-test
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WelchResult:
    """Welch's t statistic for mean(a) - mean(b), its degrees of freedom and two-sided p-value."""

    t: float
    df: float
    p: float


def welch_t_test(a: Sequence[float], b: Sequence[float]) -> WelchResult:
    """Welch's unequal-variance t-test of mean(a) against mean(b).

    Degrees of freedom follow Welch-Satterthwaite. If both samples are constant the test
    statistic is degenerate: identical constants give t = 0 and p = 1, different constants give
    t = +/-inf and p = 0 (the degrees of freedom are then undefined and reported as NaN).
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        raise ValueError("Welch's t-test needs at least two observations in each sample")
    va, vb = _sample_variance(a) / na, _sample_variance(b) / nb
    diff = mean(a) - mean(b)
    se_sq = va + vb
    if se_sq == 0.0:
        t = 0.0 if diff == 0.0 else math.copysign(math.inf, diff)
        return WelchResult(t=t, df=math.nan, p=1.0 if diff == 0.0 else 0.0)
    t = diff / math.sqrt(se_sq)
    df = se_sq**2 / (va**2 / (na - 1) + vb**2 / (nb - 1))
    return WelchResult(t=t, df=df, p=student_t_two_sided_p(t, df))


# ---------------------------------------------------------------------------------------------
# percentile bootstrap
# ---------------------------------------------------------------------------------------------


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Quantile with linear interpolation between order statistics (numpy's default method)."""
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def bootstrap_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for mean(a) - mean(b).

    Each bootstrap replicate resamples `a` and `b` independently with replacement. The stream
    of resamples comes from a private `random.Random(seed)`, so results are reproducible and
    the global random state is untouched.
    """
    if len(a) == 0 or len(b) == 0:
        raise ValueError("bootstrap needs at least one observation in each sample")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")
    rng = random.Random(seed)
    na, nb = len(a), len(b)
    diffs = sorted(
        math.fsum(rng.choices(a, k=na)) / na - math.fsum(rng.choices(b, k=nb)) / nb
        for _ in range(n_boot)
    )
    tail = (1.0 - confidence) / 2.0
    return _quantile(diffs, tail), _quantile(diffs, 1.0 - tail)


# ---------------------------------------------------------------------------------------------
# exact Mann-Whitney test
# ---------------------------------------------------------------------------------------------

MANN_WHITNEY_EXACT_MAX_N = 20  # C(20, 10) = 184,756 splits at most


def mann_whitney_u(a: Sequence[float], b: Sequence[float]) -> float:
    """Mann-Whitney U of `a`: the pairs (x from a, y from b) with x > y, a tie counting 1/2."""
    return math.fsum((x > y) + 0.5 * (x == y) for x in a for y in b)


def _doubled_midranks(values: Sequence[float]) -> list[int]:
    """Twice the 1-based rank of each value, tied values sharing the mean of their ranks."""
    order = sorted(range(len(values)), key=values.__getitem__)
    doubled = [0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for position in range(start, end + 1):
            doubled[order[position]] = start + end + 2  # (start + 1) + (end + 1)
        start = end + 1
    return doubled


def mann_whitney_exact_p(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Exact one-sided p-value of the Mann-Whitney test that `a` tends to be larger than `b`.

    The p-value is P(U >= observed U of `a`) when every split of the pooled values into groups
    of len(a) and len(b) is equally likely, enumerated in full. U is a rank sum in disguise
    (U = R_a - n_a (n_a + 1) / 2 with midranks), so ties are handled by conditioning on them.
    None when a sample is empty or the pooled size exceeds `MANN_WHITNEY_EXACT_MAX_N`.
    """
    na, n = len(a), len(a) + len(b)
    if na == 0 or len(b) == 0 or n > MANN_WHITNEY_EXACT_MAX_N:
        return None
    ranks = _doubled_midranks([*a, *b])  # integers, so the rank sums compare exactly
    observed = sum(ranks[:na])
    at_least = sum(
        sum(ranks[i] for i in split) >= observed for split in itertools.combinations(range(n), na)
    )
    return at_least / math.comb(n, na)
