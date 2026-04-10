"""Tests for replicant.analysis (cost + stats)."""

import math
import pytest

from replicant.analysis import (
    estimate_cost, get_pricing,
    mann_whitney, chi_square, cohen_d, sig_marker,
    compare_means, compare_proportions,
)


# ── Cost ────────────────────────────────────────────────────────────

def test_get_pricing_known():
    in_, out_ = get_pricing("stepfun/step-3.5-flash")
    assert in_ == 0.10
    assert out_ == 0.30


def test_get_pricing_unknown():
    in_, out_ = get_pricing("not/a-real-model")
    assert in_ == 0.0 and out_ == 0.0


def test_estimate_cost_basic():
    est = estimate_cost("stepfun/step-3.5-flash", n_calls=100)
    assert est["n_calls"] == 100
    assert est["total_cost"] > 0
    assert est["is_reasoning"] is True


def test_estimate_cost_non_reasoning():
    est = estimate_cost("meta-llama/llama-3.1-8b-instruct", n_calls=100)
    assert est["is_reasoning"] is False
    # Non-reasoning should use fewer output tokens by default
    reasoning = estimate_cost("stepfun/step-3.5-flash", n_calls=100)
    assert est["output_tokens"] < reasoning["output_tokens"]


def test_estimate_cost_zero_for_free():
    est = estimate_cost("qwen/qwen3.6-plus:free", n_calls=100)
    assert est["total_cost"] == 0.0


# ── Stats: Mann-Whitney ─────────────────────────────────────────────

def test_mann_whitney_significant_difference():
    a = [1, 2, 3, 4, 5]
    b = [10, 11, 12, 13, 14]
    u, p = mann_whitney(a, b)
    assert not math.isnan(p)
    assert p < 0.05


def test_mann_whitney_no_difference():
    a = [1, 2, 3, 4, 5]
    b = [1, 2, 3, 4, 5]
    u, p = mann_whitney(a, b)
    # Identical samples should have p > 0.05
    assert p > 0.05 or math.isnan(p)


def test_mann_whitney_empty():
    u, p = mann_whitney([], [1, 2, 3])
    assert math.isnan(u) and math.isnan(p)


def test_mann_whitney_zero_variance():
    """Both samples with zero variance should not crash."""
    u, p = mann_whitney([5, 5, 5], [5, 5, 5])
    assert math.isnan(u) and math.isnan(p)


# ── Stats: Chi-square ───────────────────────────────────────────────

def test_chi_square_significant():
    chi2, p = chi_square(a_yes=80, a_total=100, b_yes=20, b_total=100)
    assert p < 0.05


def test_chi_square_no_difference():
    chi2, p = chi_square(a_yes=50, a_total=100, b_yes=50, b_total=100)
    assert p > 0.05


def test_chi_square_zero_cell():
    """Zero cells should return NaN, not crash."""
    chi2, p = chi_square(a_yes=0, a_total=10, b_yes=5, b_total=10)
    assert math.isnan(chi2) and math.isnan(p)


# ── Stats: Cohen's d ────────────────────────────────────────────────

def test_cohen_d_large_effect():
    a = [1, 2, 3, 4, 5]
    b = [10, 11, 12, 13, 14]
    d = cohen_d(a, b)
    assert abs(d) > 1.0  # large effect


def test_cohen_d_no_effect():
    a = [1, 2, 3, 4, 5]
    b = [1, 2, 3, 4, 5]
    d = cohen_d(a, b)
    assert d == 0.0


def test_cohen_d_too_small():
    d = cohen_d([1], [2])
    assert math.isnan(d)


# ── Significance markers ────────────────────────────────────────────

def test_sig_marker():
    assert sig_marker(0.0001) == "***"
    assert sig_marker(0.005) == "**"
    assert sig_marker(0.03) == "*"
    assert sig_marker(0.1) == ""
    assert sig_marker(float("nan")) == ""


# ── Comparison helpers ──────────────────────────────────────────────

def test_compare_means_returns_dict(capsys):
    result = compare_means([1, 2, 3], [10, 11, 12], label="test")
    assert "a_mean" in result
    assert "b_mean" in result
    assert "p" in result
    assert result["a_mean"] == 2.0
    assert result["b_mean"] == 11.0


def test_compare_proportions_returns_dict(capsys):
    result = compare_proportions(80, 100, 20, 100, label="vote")
    assert result["a_pct"] == 0.8
    assert result["b_pct"] == 0.2
    assert result["p"] < 0.05
