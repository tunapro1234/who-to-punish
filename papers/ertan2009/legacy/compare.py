"""
Compare default vs personality conditions for Ertan 2009.

Reads saved CSVs from results/ and uses replicant.analysis helpers
for the actual statistical tests.
"""

import os
import pandas as pd

from replicant.analysis import (
    compare_means, compare_proportions, print_comparison_header,
)

from .config import REGIMES, PROFILES, PAPER_FINDINGS

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "results")


def load_condition(prefix: str) -> dict:
    """Load all 4 parts for a condition."""
    data = {}
    for part in ["baseline", "voting", "regimes", "punishment"]:
        path = os.path.join(RESULTS_DIR, f"{prefix}_{part}.csv")
        if os.path.exists(path):
            data[part] = pd.read_csv(path)
    return data


def compare_conditions(default_data: dict, personality_data: dict):
    """Statistical comparison between default and personality conditions."""
    print("\n" + "=" * 70, flush=True)
    print("CONDITION COMPARISON: Default vs Personality", flush=True)
    print("=" * 70, flush=True)

    print_comparison_header("Default", "Personality")

    # Baseline contribution
    compare_means(
        default_data["baseline"]["answer.contribution"],
        personality_data["baseline"]["answer.contribution"],
        label="Baseline contribution",
    )

    # Voting (proportions)
    n_d = len(default_data["voting"])
    n_p = len(personality_data["voting"])
    d_low = (default_data["voting"]["answer.vote_punish_low"] == "Yes").sum()
    p_low = (personality_data["voting"]["answer.vote_punish_low"] == "Yes").sum()
    compare_proportions(d_low, n_d, p_low, n_p, label="Vote punish low (%)")

    d_high = (default_data["voting"]["answer.vote_punish_high"] == "Yes").sum()
    p_high = (personality_data["voting"]["answer.vote_punish_high"] == "Yes").sum()
    compare_proportions(d_high, n_d, p_high, n_p, label="Vote punish high (%)")

    # Contribution by regime
    print(f"\nContribution by regime:", flush=True)
    for regime in REGIMES:
        d_v = default_data["regimes"][
            default_data["regimes"]["scenario.regime_name"] == regime
        ]["answer.contribution"]
        p_v = personality_data["regimes"][
            personality_data["regimes"]["scenario.regime_name"] == regime
        ]["answer.contribution"]
        compare_means(d_v, p_v, label=f"  {regime}")

    # Punishment by profile
    print(f"\nPunishment amount by profile:", flush=True)
    for p in PROFILES:
        d_v = default_data["punishment"][
            default_data["punishment"]["scenario.profile_name"] == p["name"]
        ]["answer.punish_amount"]
        p_v = personality_data["punishment"][
            personality_data["punishment"]["scenario.profile_name"] == p["name"]
        ]["answer.punish_amount"]
        compare_means(d_v, p_v, label=f"  {p['name']}")


def compare_to_paper(data: dict, label: str):
    """Compare a single condition's results to the original paper."""
    print(f"\n{'='*70}", flush=True)
    print(f"COMPARISON TO PAPER: {label}", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"\n{'Metric':<35s} {'Paper':>10s} {'Simulation':>12s} {'Delta':>10s}",
          flush=True)
    print("-" * 70, flush=True)

    baseline_mean = data["baseline"]["answer.contribution"].astype(float).mean()
    n_v = len(data["voting"])
    vote_low_pct = (data["voting"]["answer.vote_punish_low"] == "Yes").sum() / n_v
    vote_high_pct = (data["voting"]["answer.vote_punish_high"] == "Yes").sum() / n_v

    rows = [
        ("Vote punish low (%)",
         PAPER_FINDINGS["vote_punish_low_pct"][0] * 100, vote_low_pct * 100),
        ("Vote punish high (%)",
         PAPER_FINDINGS["vote_punish_high_pct"][0] * 100, vote_high_pct * 100),
    ]
    for name, paper_val, sim_val in rows:
        delta = sim_val - paper_val
        sign = "+" if delta > 0 else ""
        print(
            f"{name:<35s} {paper_val:>10.1f} {sim_val:>12.1f} {sign}{delta:>9.1f}",
            flush=True,
        )


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
        print(f"ERROR: No data for prefix '{args.default_prefix}'")
        return
    if not personality_data:
        print(f"ERROR: No data for prefix '{args.personality_prefix}'")
        return

    print(f"Default: {len(default_data['baseline'])} agents", flush=True)
    print(f"Personality: {len(personality_data['baseline'])} agents", flush=True)

    compare_conditions(default_data, personality_data)
    compare_to_paper(default_data, "Default")
    compare_to_paper(personality_data, "Personality")


if __name__ == "__main__":
    main()
