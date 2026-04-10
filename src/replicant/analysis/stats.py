"""
Statistical comparison helpers.

Generic functions for comparing two experimental conditions and printing
formatted summary tables. Used by paper-specific compare scripts.
"""

import math
import pandas as pd
from scipy import stats


# ── Statistical tests ───────────────────────────────────────────────

def mann_whitney(a, b) -> tuple[float, float]:
    """
    Two-sided Mann-Whitney U test for two independent samples.
    Returns (U_statistic, p_value). Returns (nan, nan) if either sample is empty
    or both have zero variance.
    """
    a = pd.Series(a).dropna()
    b = pd.Series(b).dropna()
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    if a.std() == 0 and b.std() == 0:
        return float("nan"), float("nan")
    try:
        return stats.mannwhitneyu(a, b, alternative="two-sided")
    except Exception:
        return float("nan"), float("nan")


def chi_square(a_yes: int, a_total: int, b_yes: int, b_total: int) -> tuple[float, float]:
    """
    Chi-square test of independence for two proportions.
    Returns (chi2_statistic, p_value).
    """
    a_no = a_total - a_yes
    b_no = b_total - b_yes
    table = [[a_yes, a_no], [b_yes, b_no]]
    if min(a_yes, a_no, b_yes, b_no) <= 0:
        return float("nan"), float("nan")
    try:
        chi2, p, _, _ = stats.chi2_contingency(table)
        return chi2, p
    except Exception:
        return float("nan"), float("nan")


def cohen_d(a, b) -> float:
    """Cohen's d effect size for two independent samples."""
    a = pd.Series(a).dropna()
    b = pd.Series(b).dropna()
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_sd = math.sqrt(((len(a) - 1) * a.std() ** 2 + (len(b) - 1) * b.std() ** 2) /
                          (len(a) + len(b) - 2))
    if pooled_sd == 0:
        return float("nan")
    return (a.mean() - b.mean()) / pooled_sd


# ── Formatting helpers ──────────────────────────────────────────────

def sig_marker(p: float) -> str:
    """Significance marker: * (p<.05), ** (p<.01), *** (p<.001)."""
    if math.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def compare_means(a, b, label: str = "", width: int = 35) -> dict:
    """
    Compare two groups on a numeric variable. Prints a formatted line and
    returns the comparison stats.
    """
    a = pd.Series(a).astype(float).dropna()
    b = pd.Series(b).astype(float).dropna()
    u, p = mann_whitney(a, b)
    d = cohen_d(a, b)
    sig = sig_marker(p)

    if label:
        print(
            f"{label:<{width}s} {a.mean():>8.2f}   {b.mean():>10.2f}   "
            f"{p:>8.3f} {sig}",
            flush=True,
        )

    return {
        "label": label,
        "a_mean": a.mean(), "a_std": a.std(), "a_n": len(a),
        "b_mean": b.mean(), "b_std": b.std(), "b_n": len(b),
        "u": u, "p": p, "cohens_d": d,
    }


def compare_proportions(a_yes: int, a_total: int, b_yes: int, b_total: int,
                         label: str = "", width: int = 35) -> dict:
    """
    Compare two proportions with chi-square. Prints a formatted line.
    """
    chi2, p = chi_square(a_yes, a_total, b_yes, b_total)
    a_pct = a_yes / a_total if a_total > 0 else 0
    b_pct = b_yes / b_total if b_total > 0 else 0
    sig = sig_marker(p)

    if label:
        print(
            f"{label:<{width}s} {a_pct:>8.0%}   {b_pct:>10.0%}   "
            f"{p:>8.3f} {sig}",
            flush=True,
        )

    return {
        "label": label,
        "a_yes": a_yes, "a_total": a_total, "a_pct": a_pct,
        "b_yes": b_yes, "b_total": b_total, "b_pct": b_pct,
        "chi2": chi2, "p": p,
    }


def print_comparison_header(col_a: str = "A", col_b: str = "B",
                            metric_width: int = 35):
    """Print a header for a comparison table."""
    print(
        f"\n{'Metric':<{metric_width}s} {col_a:>10s} {col_b:>12s} "
        f"{'p-value':>10s}",
        flush=True,
    )
    print("-" * (metric_width + 35), flush=True)
