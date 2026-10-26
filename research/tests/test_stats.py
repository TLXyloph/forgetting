"""Tests for research.stats against known values.

Reference values were computed independently of the implementation: closed forms for the
Student-t tail with 1, 2 and 3 degrees of freedom, standard t-table critical values, and
direct numerical integration of the t density (no incomplete beta function involved).
"""

from __future__ import annotations

import math

import pytest

from research import stats

# ---------------------------------------------------------------------------------------------
# mean / sample sd
# ---------------------------------------------------------------------------------------------


def test_mean_of_values() -> None:
    assert stats.mean([1, 2, 3, 4]) == pytest.approx(2.5)
    assert stats.mean([5.0]) == 5.0


def test_mean_is_exact_for_floats_that_naive_summation_gets_wrong() -> None:
    assert stats.mean([0.1] * 10) == pytest.approx(0.1, abs=1e-18)


def test_mean_of_nothing_is_an_error() -> None:
    with pytest.raises(ValueError):
        stats.mean([])


def test_sample_sd_uses_n_minus_one() -> None:
    # classic data set: population sd is 2, sample sd is sqrt(32 / 7)
    data = [2, 4, 4, 4, 5, 5, 7, 9]
    assert stats.sample_sd(data) == pytest.approx(math.sqrt(32 / 7))


def test_sample_sd_of_constant_values_is_zero() -> None:
    assert stats.sample_sd([3.0, 3.0, 3.0]) == 0.0


def test_sample_sd_needs_two_values() -> None:
    with pytest.raises(ValueError):
        stats.sample_sd([1.0])


# ---------------------------------------------------------------------------------------------
# regularized incomplete beta
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("x", [0.0, 0.2, 0.5, 0.9, 1.0])
def test_betainc_uniform_case_is_identity(x: float) -> None:
    assert stats.betainc(1.0, 1.0, x) == pytest.approx(x, abs=1e-12)


@pytest.mark.parametrize("x", [0.1, 0.35, 0.8])
def test_betainc_closed_forms_for_unit_parameters(x: float) -> None:
    # I_x(a, 1) = x^a and I_x(1, b) = 1 - (1 - x)^b
    assert stats.betainc(3.5, 1.0, x) == pytest.approx(x**3.5, rel=1e-10)
    assert stats.betainc(1.0, 2.5, x) == pytest.approx(1 - (1 - x) ** 2.5, rel=1e-10)


@pytest.mark.parametrize(("a", "b", "x"), [(2.0, 3.0, 0.3), (0.5, 0.5, 0.7), (10.0, 0.5, 0.95)])
def test_betainc_reflection_symmetry(a: float, b: float, x: float) -> None:
    assert stats.betainc(a, b, x) == pytest.approx(1.0 - stats.betainc(b, a, 1.0 - x), abs=1e-12)


def test_betainc_arcsine_distribution_median() -> None:
    # Beta(1/2, 1/2) is symmetric about 1/2
    assert stats.betainc(0.5, 0.5, 0.5) == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize(("a", "b", "x"), [(0.0, 1.0, 0.5), (1.0, -1.0, 0.5), (1.0, 1.0, 1.5)])
def test_betainc_rejects_invalid_arguments(a: float, b: float, x: float) -> None:
    with pytest.raises(ValueError):
        stats.betainc(a, b, x)


# ---------------------------------------------------------------------------------------------
# Student-t CDF and two-sided p-values
# ---------------------------------------------------------------------------------------------


def test_two_sided_p_matches_the_brief_value_t2_df10() -> None:
    p = stats.student_t_two_sided_p(2.0, 10)
    assert round(p, 4) == 0.0734
    assert p == pytest.approx(0.0733880348, abs=1e-8)  # numerical integration of the density


@pytest.mark.parametrize("t", [0.3, 1.0, 2.0, 3.5, 12.0])
def test_two_sided_p_df1_is_the_cauchy_closed_form(t: float) -> None:
    expected = 1 - (2 / math.pi) * math.atan(t)
    assert stats.student_t_two_sided_p(t, 1) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("t", [0.3, 1.0, 2.0, 3.5, 12.0])
def test_two_sided_p_df2_closed_form(t: float) -> None:
    expected = 1 - t / math.sqrt(2 + t * t)
    assert stats.student_t_two_sided_p(t, 2) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("t", [0.3, 1.0, 2.0, 3.5])
def test_two_sided_p_df3_closed_form(t: float) -> None:
    s = t / math.sqrt(3)
    expected = 1 - (2 / math.pi) * (s / (1 + t * t / 3) + math.atan(s))
    assert stats.student_t_two_sided_p(t, 3) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    ("t", "df", "alpha"),
    [
        (12.7062, 1, 0.05),
        (4.30265, 2, 0.05),
        (3.18245, 3, 0.05),
        (2.77645, 4, 0.05),
        (2.57058, 5, 0.05),
        (4.03214, 5, 0.01),
        (1.812461, 10, 0.10),
        (2.228139, 10, 0.05),
        (3.169273, 10, 0.01),
        (2.04227, 30, 0.05),
    ],
)
def test_two_sided_p_reproduces_t_table_critical_values(t: float, df: int, alpha: float) -> None:
    assert stats.student_t_two_sided_p(t, df) == pytest.approx(alpha, abs=2e-6)


@pytest.mark.parametrize(
    ("t", "df", "expected"),
    [(5.0, 3.0, 0.0153924381), (0.5, 3.7, 0.6453356333), (1.0, 8.0, 0.3465935071)],
)
def test_two_sided_p_fractional_degrees_of_freedom(t: float, df: float, expected: float) -> None:
    # Welch-Satterthwaite degrees of freedom are not integers
    assert stats.student_t_two_sided_p(t, df) == pytest.approx(expected, abs=1e-8)


def test_two_sided_p_at_zero_is_one() -> None:
    assert stats.student_t_two_sided_p(0.0, 7) == 1.0


def test_two_sided_p_is_symmetric_in_t() -> None:
    assert stats.student_t_two_sided_p(-2.3, 6.5) == stats.student_t_two_sided_p(2.3, 6.5)


def test_two_sided_p_decreases_with_abs_t() -> None:
    ps = [stats.student_t_two_sided_p(t, 9) for t in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)]
    assert ps == sorted(ps, reverse=True)
    assert 0.0 < ps[-1] < 1e-4


def test_two_sided_p_of_infinite_t_is_zero() -> None:
    assert stats.student_t_two_sided_p(math.inf, 5) == 0.0
    assert stats.student_t_two_sided_p(-math.inf, 5) == 0.0


def test_two_sided_p_tiny_t_stays_close_to_one() -> None:
    p = stats.student_t_two_sided_p(1e-8, 5)
    assert p == pytest.approx(1.0, abs=1e-7)
    assert p < 1.0


def test_two_sided_p_for_large_df_approaches_the_normal_value() -> None:
    assert stats.student_t_two_sided_p(1.959964, 1e5) == pytest.approx(0.05, abs=2e-4)


def test_two_sided_p_rejects_non_positive_df() -> None:
    with pytest.raises(ValueError):
        stats.student_t_two_sided_p(1.0, 0.0)


def test_cdf_is_one_half_at_zero_and_antisymmetric() -> None:
    assert stats.student_t_cdf(0.0, 4) == 0.5
    assert stats.student_t_cdf(1.7, 4) + stats.student_t_cdf(-1.7, 4) == pytest.approx(1.0)


def test_cdf_known_value_t2_df10() -> None:
    assert stats.student_t_cdf(2.0, 10) == pytest.approx(1 - 0.0733880348 / 2, abs=1e-8)


def test_cdf_df1_is_the_arctangent() -> None:
    for t in (-3.0, -0.4, 0.9, 5.0):
        assert stats.student_t_cdf(t, 1) == pytest.approx(0.5 + math.atan(t) / math.pi, abs=1e-12)


def test_cdf_is_consistent_with_two_sided_p() -> None:
    for t, df in ((0.8, 3), (2.5, 12), (6.0, 2.5)):
        two_sided = 2 * (1 - stats.student_t_cdf(abs(t), df))
        assert stats.student_t_two_sided_p(t, df) == pytest.approx(two_sided, abs=1e-12)


# ---------------------------------------------------------------------------------------------
# Welch's t-test
# ---------------------------------------------------------------------------------------------


def test_welch_equal_variance_equal_size_matches_pooled_t_test() -> None:
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [2.0, 3.0, 4.0, 5.0, 6.0]
    result = stats.welch_t_test(a, b)
    assert result.t == pytest.approx(-1.0)
    assert result.df == pytest.approx(8.0)  # Welch-Satterthwaite reduces to 2(n - 1)
    assert result.p == pytest.approx(0.3465935071, abs=1e-8)


def test_welch_unequal_sizes_and_variances() -> None:
    a = [12.0, 15.0, 14.0, 10.0, 13.0, 16.0]
    b = [9.0, 11.0, 10.0, 12.0]
    result = stats.welch_t_test(a, b)
    assert result.t == pytest.approx(2.59247569565, abs=1e-9)
    assert result.df == pytest.approx(7.97670405522, abs=1e-9)
    assert result.p == pytest.approx(0.0320668355, abs=1e-8)


def test_welch_three_versus_four_replicates() -> None:
    # the sizes this project actually has: 3 solo runs against 4 runs of another arm
    result = stats.welch_t_test([100.0, 104.0, 98.0], [120.0, 140.0, 90.0, 125.0])
    assert result.t == pytest.approx(-1.70108463416, abs=1e-9)
    assert result.df == pytest.approx(3.16845316009, abs=1e-9)
    assert result.p == pytest.approx(0.1826066149, abs=1e-8)


def test_welch_swapping_samples_flips_t_only() -> None:
    a = [1.0, 4.0, 2.0, 8.0]
    b = [3.0, 9.0, 7.0]
    forward, backward = stats.welch_t_test(a, b), stats.welch_t_test(b, a)
    assert forward.t == pytest.approx(-backward.t)
    assert forward.df == pytest.approx(backward.df)
    assert forward.p == pytest.approx(backward.p)


def test_welch_identical_samples_give_p_one() -> None:
    result = stats.welch_t_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert result.t == 0.0
    assert result.p == 1.0


def test_welch_identical_constant_samples_give_p_one() -> None:
    result = stats.welch_t_test([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
    assert result.t == 0.0
    assert result.p == 1.0
    assert math.isnan(result.df)  # 0 / 0 in Welch-Satterthwaite


def test_welch_different_constant_samples_are_infinitely_significant() -> None:
    result = stats.welch_t_test([5.0, 5.0, 5.0], [6.0, 6.0, 6.0])
    assert result.t == -math.inf
    assert result.p == 0.0


def test_welch_requires_two_observations_per_sample() -> None:
    with pytest.raises(ValueError):
        stats.welch_t_test([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        stats.welch_t_test([1.0, 2.0], [])


# ---------------------------------------------------------------------------------------------
# percentile bootstrap CI for a difference of means
# ---------------------------------------------------------------------------------------------

SEPARATED_A = [10.0, 12.0, 11.0, 13.0, 14.0]
SEPARATED_B = [1.0, 2.0, 3.0, 2.0, 1.0]


def test_bootstrap_is_deterministic_for_a_seed() -> None:
    first = stats.bootstrap_diff_ci(SEPARATED_A, SEPARATED_B, n_boot=2000, seed=7)
    second = stats.bootstrap_diff_ci(SEPARATED_A, SEPARATED_B, n_boot=2000, seed=7)
    assert first == second


def test_bootstrap_differs_between_seeds() -> None:
    one = stats.bootstrap_diff_ci(SEPARATED_A, SEPARATED_B, n_boot=500, seed=1)
    two = stats.bootstrap_diff_ci(SEPARATED_A, SEPARATED_B, n_boot=500, seed=2)
    assert one != two


def test_bootstrap_does_not_disturb_the_global_random_state() -> None:
    import random

    random.seed(123)
    expected = random.random()
    random.seed(123)
    stats.bootstrap_diff_ci(SEPARATED_A, SEPARATED_B, n_boot=200, seed=5)
    assert random.random() == expected


def test_bootstrap_interval_brackets_the_observed_difference() -> None:
    low, high = stats.bootstrap_diff_ci(SEPARATED_A, SEPARATED_B, n_boot=4000, seed=0)
    observed = stats.mean(SEPARATED_A) - stats.mean(SEPARATED_B)
    assert low < observed < high


def test_bootstrap_clearly_separated_samples_exclude_zero() -> None:
    low, high = stats.bootstrap_diff_ci(SEPARATED_A, SEPARATED_B, n_boot=4000, seed=0)
    assert low > 5.0
    assert high < 14.0


def test_bootstrap_identical_samples_include_zero() -> None:
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    low, high = stats.bootstrap_diff_ci(data, data, n_boot=4000, seed=0)
    assert low < 0.0 < high


def test_bootstrap_is_antisymmetric_in_the_arguments() -> None:
    low, high = stats.bootstrap_diff_ci(SEPARATED_A, SEPARATED_B, n_boot=3000, seed=3)
    rlow, rhigh = stats.bootstrap_diff_ci(SEPARATED_B, SEPARATED_A, n_boot=3000, seed=3)
    # same resampling stream with the roles swapped would mirror exactly; with independent
    # streams per sample they mirror only statistically
    assert rlow == pytest.approx(-high, abs=1.0)
    assert rhigh == pytest.approx(-low, abs=1.0)


def test_bootstrap_higher_confidence_gives_a_wider_interval() -> None:
    narrow = stats.bootstrap_diff_ci(SEPARATED_A, SEPARATED_B, n_boot=4000, confidence=0.80, seed=0)
    wide = stats.bootstrap_diff_ci(SEPARATED_A, SEPARATED_B, n_boot=4000, confidence=0.99, seed=0)
    assert wide[0] <= narrow[0]
    assert wide[1] >= narrow[1]
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_bootstrap_width_matches_normal_theory_for_moderate_samples() -> None:
    # a deterministic, well-spread pair of samples with n = 20 each
    a = [50.0 + ((i * 37) % 21) - 10 for i in range(20)]
    b = [45.0 + ((i * 53) % 25) - 12 for i in range(20)]
    low, high = stats.bootstrap_diff_ci(a, b, n_boot=6000, seed=11)
    plug_in_var_a = stats.sample_sd(a) ** 2 * (len(a) - 1) / len(a)
    plug_in_var_b = stats.sample_sd(b) ** 2 * (len(b) - 1) / len(b)
    se = math.sqrt(plug_in_var_a / len(a) + plug_in_var_b / len(b))
    assert (high - low) == pytest.approx(2 * 1.96 * se, rel=0.15)
    assert (low + high) / 2 == pytest.approx(stats.mean(a) - stats.mean(b), abs=0.5 * se)


def test_bootstrap_of_constant_samples_is_degenerate() -> None:
    low, high = stats.bootstrap_diff_ci([5.0, 5.0, 5.0], [2.0, 2.0], n_boot=300, seed=0)
    assert (low, high) == (3.0, 3.0)


def test_bootstrap_works_with_single_observations() -> None:
    assert stats.bootstrap_diff_ci([4.0], [1.0], n_boot=100, seed=0) == (3.0, 3.0)


@pytest.mark.parametrize(
    "kwargs",
    [{"confidence": 0.0}, {"confidence": 1.0}, {"confidence": 1.5}, {"n_boot": 0}],
)
def test_bootstrap_rejects_invalid_parameters(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        stats.bootstrap_diff_ci([1.0, 2.0], [3.0, 4.0], **kwargs)  # type: ignore[arg-type]


def test_bootstrap_rejects_empty_samples() -> None:
    with pytest.raises(ValueError):
        stats.bootstrap_diff_ci([], [1.0])
    with pytest.raises(ValueError):
        stats.bootstrap_diff_ci([1.0], [])
