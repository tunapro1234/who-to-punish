"""
Ertan, Page & Putterman (2009)
"Who to Punish? Individual Decisions and Majority Rule in Mitigating the Free Rider Problem"
European Economic Review, 53(5), 495-511.

This module wires the public goods template to the specific parameters from the paper.
"""

import sys
import os

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

# ── Known findings from the paper (for comparison) ───────────────────

PAPER_FINDINGS = {
    "vote_punish_low_pct":  (0.85, "% voting to allow punishing low contributors (late rounds)"),
    "vote_punish_high_pct": (0.00, "% voting to allow punishing high contributors"),
}


# ── Build experiment ─────────────────────────────────────────────────

def build(model="qwen/qwen3.6-plus:free"):
    """Build the full Ertan 2009 experiment."""
    exp = BehavioralExperiment("ertan2009", model=model)

    # Part 1: Baseline (no punishment)
    exp.add_part(
        "baseline",
        contribution_survey(ENDOWMENT, GROUP_SIZE, MPCR),
        description="Contribution without punishment",
    )

    # Part 2: Voting on punishment rules
    exp.add_part(
        "voting",
        voting_survey(PUNISHMENT_RATIO),
        description="Vote on punishment rules",
    )

    # Part 3: Contribution under each regime
    survey_r, scenarios_r = contribution_regime_survey(ENDOWMENT, GROUP_SIZE, MPCR, REGIMES)
    exp.add_part(
        "regimes",
        survey_r,
        scenarios=scenarios_r,
        description="Contribution under punishment regimes",
    )

    # Part 4: Punishment decisions
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


def analyze(results):
    """Print summary and compare to paper."""
    print("\n" + "=" * 55, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 55, flush=True)

    # Baseline (shares DataFrame with voting via combined survey)
    df = results["baseline"]
    avg_b = df["answer.contribution"].astype(float).mean()
    print(f"\nBaseline avg contribution: {avg_b:.1f} / {ENDOWMENT}", flush=True)

    # Voting (same DataFrame as baseline from combined survey)
    df_v = results["voting"]
    n = len(df_v)
    low_pct = (df_v["answer.vote_punish_low"] == "Yes").sum() / n
    high_pct = (df_v["answer.vote_punish_high"] == "Yes").sum() / n
    print(f"\nVoting:", flush=True)
    print(f"  Punish low:  {low_pct:.0%}", flush=True)
    print(f"  Punish high: {high_pct:.0%}", flush=True)

    # Regimes
    df_r = results["regimes"]
    print(f"\nContribution by regime:", flush=True)
    for regime in REGIMES:
        vals = df_r[df_r["scenario.regime_name"] == regime]["answer.contribution"].astype(float)
        print(f"  {regime:<20s}: {vals.mean():.1f}", flush=True)

    # Punishment
    df_p = results["punishment"]
    print(f"\nPunishment by profile:", flush=True)
    for p in PROFILES:
        vals = df_p[df_p["scenario.profile_name"] == p["name"]]["answer.punish_amount"].astype(float)
        print(f"  {p['name']:<20s}: {vals.mean():.2f}", flush=True)
    print(f"\nTarget distribution:", flush=True)
    print(df_p["answer.punish_target"].value_counts().to_string(), flush=True)

    # Paper comparison
    comp = PaperComparison("Ertan, Page & Putterman (2009)")
    for key, (val, desc) in PAPER_FINDINGS.items():
        comp.add_finding(key, val, desc)
    comp.compare("vote_punish_low_pct", low_pct)
    comp.compare("vote_punish_high_pct", high_pct)
    comp.report()


# ── CLI entry point ──────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ertan et al. 2009 replication")
    parser.add_argument("-n", type=int, default=10, help="Number of agents")
    parser.add_argument("--model", default="qwen/qwen3.6-plus:free")
    parser.add_argument("--big5", action="store_true", help="Use BFI-2 personality agents (balanced profiles)")
    parser.add_argument("--random-personalities", action="store_true", help="Use random BFI-2 personality agents")
    args = parser.parse_args()

    exp = build(model=args.model)

    factory = PersonalityFactory()
    if args.big5:
        agents = factory.create_population(per_profile=max(1, args.n // 5))
        print(f"BFI-2 agents: {len(agents)}", flush=True)
    elif args.random_personalities:
        agents = factory.create_random_population(n=args.n)
        print(f"Random personality agents: {len(agents)}", flush=True)
    else:
        agents = factory.create_default(n=args.n)

    results = exp.run(agents)
    analyze(results)


if __name__ == "__main__":
    main()
