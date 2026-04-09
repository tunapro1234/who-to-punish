"""
Ertan, Page & Putterman (2009)
"Who to Punish? Individual Decisions and Majority Rule in Mitigating the Free Rider Problem"
European Economic Review, 53(5), 495-511.

This module wires the public goods template to the specific parameters from the paper.
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from src.framework import BehavioralExperiment, PersonalityFactory, PaperComparison
from src.framework.templates.public_goods import (
    contribution_survey,
    contribution_regime_survey,
    voting_survey,
    punishment_survey,
)

# ── Paper-specific parameters ────────────────────────────────────────

ENDOWMENT = 20
GROUP_SIZE = 5
MPCR = 0.4
PUNISHMENT_RATIO = 3

REGIMES = {
    "no_punishment": "NO punishment allowed.",
    "punish_low_only": (
        "Members CAN punish below-average contributors only. "
        f"Cost: 1 token, removes {PUNISHMENT_RATIO} from target."
    ),
    "punish_high_only": (
        "Members CAN punish above-average contributors only. "
        f"Cost: 1 token, removes {PUNISHMENT_RATIO} from target."
    ),
    "unrestricted": (
        "Members CAN punish ANY member. "
        f"Cost: 1 token, removes {PUNISHMENT_RATIO} from target."
    ),
}

ACTIVE_REGIMES = {k: v for k, v in REGIMES.items() if k != "no_punishment"}

PROFILES = [
    {"name": "all_cooperate",  "others": "18, 17, 19, 16", "avg": 17.5},
    {"name": "mixed",          "others": "15, 5, 18, 2",   "avg": 10.0},
    {"name": "one_freerider",  "others": "17, 18, 16, 2",  "avg": 13.25},
]

# ── Known findings from the paper ────────────────────────────────────

PAPER_FINDINGS = {
    "vote_punish_low_pct":  (0.85, "% voting to allow punishing low contributors (late rounds)"),
    "vote_punish_high_pct": (0.00, "% voting to allow punishing high contributors"),
}


# ── Build experiment ─────────────────────────────────────────────────

def build(model="stepfun/step-3.5-flash"):
    exp = BehavioralExperiment("ertan2009", model=model)

    exp.add_part(
        "baseline",
        contribution_survey(ENDOWMENT, GROUP_SIZE, MPCR),
        description="Contribution without punishment",
    )

    exp.add_part(
        "voting",
        voting_survey(PUNISHMENT_RATIO),
        description="Vote on punishment rules",
    )

    survey_r, scenarios_r = contribution_regime_survey(ENDOWMENT, GROUP_SIZE, MPCR, REGIMES)
    exp.add_part(
        "regimes",
        survey_r,
        scenarios=scenarios_r,
        description="Contribution under punishment regimes",
    )

    survey_p, scenarios_p = punishment_survey(
        ENDOWMENT, GROUP_SIZE, MPCR, PUNISHMENT_RATIO,
        ACTIVE_REGIMES, PROFILES,
    )
    exp.add_part(
        "punishment",
        survey_p,
        scenarios=scenarios_p,
        description="Punishment decisions",
    )

    return exp


# ── Analysis ─────────────────────────────────────────────────────────

def analyze(results, label=""):
    """Analyze and return summary dict."""
    prefix = f"[{label}] " if label else ""

    print(f"\n{'='*55}", flush=True)
    print(f"{prefix}RESULTS SUMMARY", flush=True)
    print(f"{'='*55}", flush=True)

    summary = {}

    # Baseline
    df_b = results["baseline"]
    contribs = df_b["answer.contribution"].astype(float)
    summary["baseline_mean"] = contribs.mean()
    summary["baseline_std"] = contribs.std()
    print(f"\nBaseline contribution: {contribs.mean():.1f} (SD={contribs.std():.1f})", flush=True)

    # Voting
    df_v = results["voting"]
    n = len(df_v)
    low_pct = (df_v["answer.vote_punish_low"] == "Yes").sum() / n
    high_pct = (df_v["answer.vote_punish_high"] == "Yes").sum() / n
    summary["vote_punish_low"] = low_pct
    summary["vote_punish_high"] = high_pct
    print(f"\nVoting:", flush=True)
    print(f"  Punish low:  {low_pct:.0%} ({int(low_pct*n)}/{n})", flush=True)
    print(f"  Punish high: {high_pct:.0%} ({int(high_pct*n)}/{n})", flush=True)

    # Regimes
    df_r = results["regimes"]
    print(f"\nContribution by regime:", flush=True)
    summary["regimes"] = {}
    for regime in REGIMES:
        vals = df_r[df_r["scenario.regime_name"] == regime]["answer.contribution"].astype(float)
        summary["regimes"][regime] = {"mean": vals.mean(), "std": vals.std()}
        print(f"  {regime:<20s}: {vals.mean():5.1f} (SD={vals.std():.1f})", flush=True)

    # Punishment
    df_p = results["punishment"]
    print(f"\nPunishment amount by profile:", flush=True)
    summary["punishment"] = {}
    for p in PROFILES:
        vals = df_p[df_p["scenario.profile_name"] == p["name"]]["answer.punish_amount"].astype(float)
        nonzero = (vals > 0).sum()
        summary["punishment"][p["name"]] = {"mean": vals.mean(), "std": vals.std(), "nonzero": int(nonzero), "n": len(vals)}
        print(f"  {p['name']:<20s}: {vals.mean():.2f} (SD={vals.std():.2f}, punished={nonzero}/{len(vals)})", flush=True)

    print(f"\nTarget distribution:", flush=True)
    targets = df_p["answer.punish_target"].value_counts()
    summary["targets"] = targets.to_dict()
    for target, count in targets.items():
        print(f"  {target:<30s}: {count:>3d} ({count/len(df_p)*100:.0f}%)", flush=True)

    # Paper comparison
    comp = PaperComparison("Ertan, Page & Putterman (2009)")
    for key, (val, desc) in PAPER_FINDINGS.items():
        comp.add_finding(key, val, desc)
    comp.compare("vote_punish_low_pct", low_pct)
    comp.compare("vote_punish_high_pct", high_pct)
    comp.report()

    return summary


def save_results(results, prefix, output_dir="results"):
    """Save raw DataFrames to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    for part_name, df in results.items():
        path = os.path.join(output_dir, f"{prefix}_{part_name}.csv")
        df.to_csv(path, index=False)
    print(f"Saved to {output_dir}/{prefix}_*.csv", flush=True)


# ── CLI entry point ──────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ertan et al. 2009 replication")
    parser.add_argument("-n", type=int, default=50, help="Number of agents")
    parser.add_argument("--model", default="stepfun/step-3.5-flash")
    parser.add_argument("--big5", action="store_true", help="Use BFI-2 personality agents (balanced profiles)")
    parser.add_argument("--random-personalities", action="store_true", help="Use random BFI-2 personality agents")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for personality sampling")
    parser.add_argument("--output-prefix", default=None, help="Prefix for output files")
    args = parser.parse_args()

    exp = build(model=args.model)
    factory = PersonalityFactory()

    if args.big5:
        agents = factory.create_population(per_profile=max(1, args.n // 5))
        condition = "big5_profiles"
        print(f"BFI-2 profile agents: {len(agents)}", flush=True)
    elif args.random_personalities:
        agents = factory.create_random_population(n=args.n, seed=args.seed)
        condition = "big5_random"
        print(f"Random personality agents: {len(agents)} (seed={args.seed})", flush=True)
    else:
        agents = factory.create_default(n=args.n)
        condition = "default"

    prefix = args.output_prefix or condition

    results = exp.run(agents)
    summary = analyze(results, label=condition)
    save_results(results, prefix)

    # Save summary as JSON
    summary_path = os.path.join("results", f"{prefix}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
