"""
Ertan 2009 — 3-condition comparison.

Reads the flat CSVs and summary JSONs for the three N=50 oTree runs:
  - otree_default       (no personality)
  - otree_random        (random population personalities)
  - otree_cooperative   (skewed high-agreeableness, low-neuroticism)

Produces a unified comparison table, Mann-Whitney tests between conditions,
chi-square tests on voting, and writes everything to a machine-readable
summary at results/comparison.json for the paper rebuild step.
"""

import json
import os

import pandas as pd

from replicant.analysis.stats import mann_whitney, chi_square

PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PAPER_DIR, "results")

CONDITIONS = [
    ("default", "otree_default", "Default (no personality)"),
    ("random", "otree_random", "Random personalities"),
    ("cooperative", "otree_cooperative", "Hyper-cooperative (agr=4.5, neu=1.5)"),
]

REGIME_FIELDS = {
    "no_punishment": "contrib_no_punishment",
    "punish_low_only": "contrib_punish_low",
    "punish_high_only": "contrib_punish_high",
    "unrestricted": "contrib_unrestricted",
}

PROFILE_FIELDS = {
    "all_cooperate": "punish_amount_cooperate",
    "mixed": "punish_amount_mixed",
    "one_freerider": "punish_amount_freerider",
}


def load_condition(prefix: str):
    csv_path = os.path.join(RESULTS_DIR, f"{prefix}_flat.csv")
    json_path = os.path.join(RESULTS_DIR, f"{prefix}_summary.json")
    if not os.path.exists(csv_path) or not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing {csv_path} or {json_path}")
    df = pd.read_csv(csv_path)
    with open(json_path) as f:
        summary = json.load(f)
    return df, summary


def pct_yes(series: pd.Series) -> float:
    """Fraction of 'Yes' values in a voting column."""
    clean = series.dropna().astype(str)
    if len(clean) == 0:
        return float("nan")
    return (clean == "Yes").mean()


def describe_regime(df: pd.DataFrame, field: str) -> dict:
    vals = pd.to_numeric(df[field], errors="coerce").dropna()
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "n": int(len(vals)),
        "values": vals.tolist(),
    }


def main():
    loaded = {}
    for key, prefix, label in CONDITIONS:
        try:
            df, summary = load_condition(prefix)
            loaded[key] = {"df": df, "summary": summary, "label": label, "prefix": prefix}
            print(f"[{key:<12s}] loaded N={len(df)} from {prefix}_flat.csv")
        except FileNotFoundError as e:
            print(f"[{key:<12s}] MISSING: {e}")

    if len(loaded) < 2:
        print("\nNot enough conditions to compare. Exiting.")
        return

    report = {"conditions": {}, "regime_stats": {}, "voting_stats": {}, "punishment_stats": {}}

    print(f"\n{'='*75}")
    print("CONDITION SUMMARY")
    print(f"{'='*75}")
    header = f"{'Metric':<28s}" + "".join(f"{c['label'][:18]:>20s}" for c in loaded.values())
    print(header)
    print("-" * len(header))

    # Baseline
    row = f"{'Baseline contribution':<28s}"
    for key, c in loaded.items():
        m = c["summary"]["baseline_mean"]
        s = c["summary"]["baseline_std"]
        row += f"{m:>11.2f} ({s:4.1f})"
        report["conditions"].setdefault(key, {})["baseline"] = {"mean": m, "std": s}
    print(row)

    # Regimes
    for regime, field in REGIME_FIELDS.items():
        row = f"  {regime:<26s}"
        for key, c in loaded.items():
            stats_d = describe_regime(c["df"], field)
            row += f"{stats_d['mean']:>11.2f} ({stats_d['std']:4.1f})"
            report["conditions"].setdefault(key, {}).setdefault("regimes", {})[regime] = {
                "mean": stats_d["mean"], "std": stats_d["std"], "n": stats_d["n"],
            }
        print(row)

    # Voting
    print()
    for vote_field, vote_label in [
        ("vote_punish_low", "Vote punish low (%)"),
        ("vote_punish_high", "Vote punish high (%)"),
    ]:
        row = f"{vote_label:<28s}"
        for key, c in loaded.items():
            p = pct_yes(c["df"][vote_field])
            row += f"{p*100:>18.0f}%"
            report["conditions"].setdefault(key, {}).setdefault("voting", {})[vote_field] = p
        print(row)

    # Punishment
    print()
    for profile, field in PROFILE_FIELDS.items():
        row = f"  punish_{profile:<21s}"
        for key, c in loaded.items():
            vals = pd.to_numeric(c["df"][field], errors="coerce").dropna()
            m = float(vals.mean())
            nz = int((vals > 0).sum())
            row += f"{m:>11.2f} ({nz:>3d}/{len(vals)})"
            report["conditions"].setdefault(key, {}).setdefault("punishment", {})[profile] = {
                "mean": m, "nonzero": nz, "n": int(len(vals)),
            }
        print(row)

    # ── Pairwise stats tests ────────────────────────────────────────
    print(f"\n{'='*75}")
    print("PAIRWISE STATISTICAL TESTS")
    print(f"{'='*75}")
    keys = list(loaded.keys())
    pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]]

    for a, b in pairs:
        print(f"\n--- {loaded[a]['label']}  vs  {loaded[b]['label']} ---")
        pair_key = f"{a}_vs_{b}"
        report["regime_stats"][pair_key] = {}
        report["voting_stats"][pair_key] = {}
        report["punishment_stats"][pair_key] = {}

        # Contribution: baseline + regimes
        for metric, field in [("baseline", "contribution")] + list(REGIME_FIELDS.items()):
            va = pd.to_numeric(loaded[a]["df"][field], errors="coerce")
            vb = pd.to_numeric(loaded[b]["df"][field], errors="coerce")
            u, p = mann_whitney(va, vb)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {metric:<20s}  U={u:.1f}  p={p:.4f} {sig}")
            report["regime_stats"][pair_key][metric] = {"U": float(u), "p": float(p)}

        # Voting: chi-square
        for vote_field in ["vote_punish_low", "vote_punish_high"]:
            ay = int((loaded[a]["df"][vote_field] == "Yes").sum())
            at = int(loaded[a]["df"][vote_field].notna().sum())
            by = int((loaded[b]["df"][vote_field] == "Yes").sum())
            bt = int(loaded[b]["df"][vote_field].notna().sum())
            chi2, p = chi_square(ay, at, by, bt)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {vote_field:<20s}  chi2={chi2:.2f}  p={p:.4f} {sig}  ({ay}/{at} vs {by}/{bt})")
            report["voting_stats"][pair_key][vote_field] = {
                "chi2": float(chi2), "p": float(p),
                "a_yes": ay, "a_total": at, "b_yes": by, "b_total": bt,
            }

        # Punishment: Mann-Whitney on punishment amounts
        for profile, field in PROFILE_FIELDS.items():
            va = pd.to_numeric(loaded[a]["df"][field], errors="coerce")
            vb = pd.to_numeric(loaded[b]["df"][field], errors="coerce")
            u, p = mann_whitney(va, vb)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  punish_{profile:<14s}  U={u:.1f}  p={p:.4f} {sig}")
            report["punishment_stats"][pair_key][profile] = {"U": float(u), "p": float(p)}

    # Save
    out_path = os.path.join(RESULTS_DIR, "comparison.json")
    # Strip raw value lists before saving to keep file small
    save_report = json.loads(json.dumps(report, default=float))
    with open(out_path, "w") as f:
        json.dump(save_report, f, indent=2)
    print(f"\nSaved comparison to {out_path}")


if __name__ == "__main__":
    main()
