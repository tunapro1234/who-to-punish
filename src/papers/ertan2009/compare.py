"""
Compare default vs personality conditions and original paper.

Reads saved CSVs from results/ and produces:
  - Condition comparison table
  - Paper comparison table
  - Statistical tests (Mann-Whitney U for non-normal data)
  - Personality breakdown (if personality data available)
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

RESULTS_DIR = "results"

REGIMES = ["no_punishment", "punish_low_only", "punish_high_only", "unrestricted"]
PROFILES = ["all_cooperate", "mixed", "one_freerider"]

# Ertan et al. (2009) known results (approximate from paper)
PAPER = {
    "baseline_contribution": 7.5,  # approximate early-round average
    "vote_punish_low_pct": 0.85,   # late rounds
    "vote_punish_high_pct": 0.00,
    "n_subjects": 160,
    "group_size": 4,  # paper used groups of 4
}


def load_condition(prefix):
    """Load all 4 parts for a condition."""
    data = {}
    for part in ["baseline", "voting", "regimes", "punishment"]:
        path = os.path.join(RESULTS_DIR, f"{prefix}_{part}.csv")
        if os.path.exists(path):
            data[part] = pd.read_csv(path)
    return data


def summarize_condition(data, label):
    """Extract key metrics from a condition's data."""
    s = {"label": label}

    # Baseline
    df = data["baseline"]
    c = df["answer.contribution"].astype(float)
    s["baseline_mean"] = c.mean()
    s["baseline_std"] = c.std()
    s["baseline_median"] = c.median()
    s["n"] = len(df)

    # Voting
    df_v = data["voting"]
    n = len(df_v)
    s["vote_low_pct"] = (df_v["answer.vote_punish_low"] == "Yes").sum() / n
    s["vote_high_pct"] = (df_v["answer.vote_punish_high"] == "Yes").sum() / n

    # Regimes
    df_r = data["regimes"]
    for regime in REGIMES:
        vals = df_r[df_r["scenario.regime_name"] == regime]["answer.contribution"].astype(float)
        s[f"regime_{regime}_mean"] = vals.mean()
        s[f"regime_{regime}_std"] = vals.std()

    # Punishment
    df_p = data["punishment"]
    for profile in PROFILES:
        vals = df_p[df_p["scenario.profile_name"] == profile]["answer.punish_amount"].astype(float)
        s[f"punish_{profile}_mean"] = vals.mean()
        nonzero = (vals > 0).sum()
        s[f"punish_{profile}_rate"] = nonzero / len(vals) if len(vals) > 0 else 0

    # Targets
    targets = df_p["answer.punish_target"].value_counts(normalize=True)
    s["targets"] = targets.to_dict()

    return s


def compare_conditions(default_data, personality_data):
    """Statistical comparison between conditions."""
    print("\n" + "=" * 70, flush=True)
    print("CONDITION COMPARISON: Default vs Personality", flush=True)
    print("=" * 70, flush=True)

    d = summarize_condition(default_data, "Default")
    p = summarize_condition(personality_data, "Personality")

    # Header
    print(f"\n{'Metric':<35s} {'Default':>10s} {'Personality':>12s} {'p-value':>10s}", flush=True)
    print("-" * 70, flush=True)

    # Baseline contribution
    d_vals = default_data["baseline"]["answer.contribution"].astype(float)
    p_vals = personality_data["baseline"]["answer.contribution"].astype(float)
    u_stat, p_val = stats.mannwhitneyu(d_vals, p_vals, alternative="two-sided")
    sig = "*" if p_val < 0.05 else ("**" if p_val < 0.01 else "")
    print(f"{'Baseline contribution':<35s} {d['baseline_mean']:>8.1f}   {p['baseline_mean']:>10.1f}   {p_val:>8.3f} {sig}", flush=True)

    # Voting
    # Chi-square for proportions
    d_low = int(d["vote_low_pct"] * d["n"])
    p_low = int(p["vote_low_pct"] * p["n"])
    table = [[d_low, d["n"] - d_low], [p_low, p["n"] - p_low]]
    if min(d_low, d["n"] - d_low, p_low, p["n"] - p_low) > 0:
        chi2, p_val = stats.chi2_contingency(table)[:2]
    else:
        p_val = float("nan")
    print(f"{'Vote punish low (%)':<35s} {d['vote_low_pct']:>8.0%}   {p['vote_low_pct']:>10.0%}   {p_val:>8.3f}", flush=True)
    print(f"{'Vote punish high (%)':<35s} {d['vote_high_pct']:>8.0%}   {p['vote_high_pct']:>10.0%}", flush=True)

    # Regimes
    print(f"\n{'Contribution by regime:':<35s}", flush=True)
    for regime in REGIMES:
        d_v = default_data["regimes"][default_data["regimes"]["scenario.regime_name"] == regime]["answer.contribution"].astype(float)
        p_v = personality_data["regimes"][personality_data["regimes"]["scenario.regime_name"] == regime]["answer.contribution"].astype(float)
        if len(d_v) > 0 and len(p_v) > 0 and d_v.std() + p_v.std() > 0:
            _, p_val = stats.mannwhitneyu(d_v, p_v, alternative="two-sided")
        else:
            p_val = float("nan")
        sig = "*" if p_val < 0.05 else ""
        print(f"  {regime:<33s} {d_v.mean():>8.1f}   {p_v.mean():>10.1f}   {p_val:>8.3f} {sig}", flush=True)

    # Punishment
    print(f"\n{'Punishment amount by profile:':<35s}", flush=True)
    for profile in PROFILES:
        d_v = default_data["punishment"][default_data["punishment"]["scenario.profile_name"] == profile]["answer.punish_amount"].astype(float)
        p_v = personality_data["punishment"][personality_data["punishment"]["scenario.profile_name"] == profile]["answer.punish_amount"].astype(float)
        if len(d_v) > 0 and len(p_v) > 0 and d_v.std() + p_v.std() > 0:
            _, p_val = stats.mannwhitneyu(d_v, p_v, alternative="two-sided")
        else:
            p_val = float("nan")
        print(f"  {profile:<33s} {d_v.mean():>8.2f}   {p_v.mean():>10.2f}   {p_val:>8.3f}", flush=True)

    # Target distribution
    print(f"\n{'Target distribution:':<35s}", flush=True)
    print(f"  {'Target':<33s} {'Default':>10s} {'Personality':>12s}", flush=True)
    all_targets = set(d["targets"].keys()) | set(p["targets"].keys())
    for t in sorted(all_targets):
        d_pct = d["targets"].get(t, 0)
        p_pct = p["targets"].get(t, 0)
        print(f"  {t:<33s} {d_pct:>9.0%}   {p_pct:>11.0%}", flush=True)

    return d, p


def compare_to_paper(summary, label):
    """Compare a condition's results to the original paper."""
    print(f"\n{'='*70}", flush=True)
    print(f"COMPARISON TO PAPER: {label}", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"\n{'Metric':<35s} {'Paper':>10s} {'Simulation':>12s} {'Delta':>10s}", flush=True)
    print("-" * 70, flush=True)

    rows = [
        ("Baseline contribution", PAPER["baseline_contribution"], summary["baseline_mean"]),
        ("Vote punish low (%)", PAPER["vote_punish_low_pct"] * 100, summary["vote_low_pct"] * 100),
        ("Vote punish high (%)", PAPER["vote_punish_high_pct"] * 100, summary["vote_high_pct"] * 100),
    ]
    for name, paper_val, sim_val in rows:
        delta = sim_val - paper_val
        sign = "+" if delta > 0 else ""
        print(f"{name:<35s} {paper_val:>10.1f} {sim_val:>12.1f} {sign}{delta:>9.1f}", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare experiment conditions")
    parser.add_argument("--default-prefix", default="default")
    parser.add_argument("--personality-prefix", default="big5_random")
    args = parser.parse_args()

    print("Loading results...", flush=True)
    default_data = load_condition(args.default_prefix)
    personality_data = load_condition(args.personality_prefix)

    if not default_data:
        print(f"ERROR: No data found for prefix '{args.default_prefix}'")
        return
    if not personality_data:
        print(f"ERROR: No data found for prefix '{args.personality_prefix}'")
        return

    print(f"Default: {len(default_data['baseline'])} agents", flush=True)
    print(f"Personality: {len(personality_data['baseline'])} agents", flush=True)

    d_summary, p_summary = compare_conditions(default_data, personality_data)
    compare_to_paper(d_summary, "Default")
    compare_to_paper(p_summary, "Personality")


if __name__ == "__main__":
    main()
