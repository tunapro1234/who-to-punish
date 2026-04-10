"""
Ertan 2009 — analysis of oTree bot results.

The oTree pipeline returns a list of bot result dicts, each containing
the agent's decisions across all pages. This module flattens that into
summary metrics and compares to the original paper.
"""

import os
import json
import csv

from replicant import PaperComparison
from .config import REGIMES, PROFILES, PAPER_FINDINGS


# ── Field name mapping (what the oTree app uses) ────────────────────

# Page name → expected fields
PAGE_FIELDS = {
    "Baseline": ["contribution"],
    "Voting": ["vote_punish_low", "vote_punish_high"],
    "Regimes": [
        "contrib_no_punishment",
        "contrib_punish_low",
        "contrib_punish_high",
        "contrib_unrestricted",
    ],
    "Punishment": [
        "punish_amount_mixed", "punish_target_mixed",
        "punish_amount_freerider", "punish_target_freerider",
        "punish_amount_cooperate", "punish_target_cooperate",
    ],
}


# ── Helpers ─────────────────────────────────────────────────────────

def _get_field(decisions: list[dict], field_name: str):
    """Find a field's answer across the bot's pages."""
    for d in decisions:
        if "answers" in d and field_name in d["answers"]:
            return d["answers"][field_name]
    return None


def _safe_mean(values):
    """Mean of non-None values, or None if empty."""
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    try:
        return sum(float(v) for v in clean) / len(clean)
    except (TypeError, ValueError):
        return None


def _safe_std(values):
    """Standard deviation of non-None numeric values."""
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return 0.0
    try:
        nums = [float(v) for v in clean]
        m = sum(nums) / len(nums)
        return (sum((x - m) ** 2 for x in nums) / (len(nums) - 1)) ** 0.5
    except (TypeError, ValueError):
        return 0.0


# ── Analysis ────────────────────────────────────────────────────────

def analyze(results: list[dict], label: str = "") -> dict:
    """
    Analyze a list of bot results from an oTree session.

    Args:
        results: list of dicts, one per bot, each with a 'decisions' list
        label: optional condition label for printing

    Returns:
        Summary dict.
    """
    prefix = f"[{label}] " if label else ""
    print(f"\n{'='*55}", flush=True)
    print(f"{prefix}RESULTS SUMMARY", flush=True)
    print(f"{'='*55}", flush=True)

    summary = {"label": label, "n": len(results)}

    # Filter out bots that errored
    valid = [r for r in results if "error" not in r and r.get("decisions")]
    n_valid = len(valid)
    print(f"\n{n_valid}/{len(results)} bots completed", flush=True)
    if n_valid == 0:
        print("No valid results.", flush=True)
        return summary

    # ── Baseline ─────────────────────────────────────────────────
    contribs = [_get_field(r["decisions"], "contribution") for r in valid]
    summary["baseline_mean"] = _safe_mean(contribs)
    summary["baseline_std"] = _safe_std(contribs)
    if summary["baseline_mean"] is not None:
        print(
            f"\nBaseline contribution: {summary['baseline_mean']:.1f} "
            f"(SD={summary['baseline_std']:.1f})",
            flush=True,
        )

    # ── Voting ───────────────────────────────────────────────────
    low_votes = [_get_field(r["decisions"], "vote_punish_low") for r in valid]
    high_votes = [_get_field(r["decisions"], "vote_punish_high") for r in valid]
    n = len(valid)
    low_yes = sum(1 for v in low_votes if v == "Yes")
    high_yes = sum(1 for v in high_votes if v == "Yes")
    summary["vote_punish_low"] = low_yes / n if n > 0 else 0
    summary["vote_punish_high"] = high_yes / n if n > 0 else 0
    print(f"\nVoting:", flush=True)
    print(f"  Punish low:  {summary['vote_punish_low']:.0%} ({low_yes}/{n})", flush=True)
    print(f"  Punish high: {summary['vote_punish_high']:.0%} ({high_yes}/{n})", flush=True)

    # ── Regimes ──────────────────────────────────────────────────
    print(f"\nContribution by regime:", flush=True)
    summary["regimes"] = {}
    regime_field_map = {
        "no_punishment": "contrib_no_punishment",
        "punish_low_only": "contrib_punish_low",
        "punish_high_only": "contrib_punish_high",
        "unrestricted": "contrib_unrestricted",
    }
    for regime, field in regime_field_map.items():
        vals = [_get_field(r["decisions"], field) for r in valid]
        m = _safe_mean(vals)
        s = _safe_std(vals)
        summary["regimes"][regime] = {"mean": m, "std": s}
        if m is not None:
            print(f"  {regime:<20s}: {m:5.1f} (SD={s:.1f})", flush=True)

    # ── Punishment ───────────────────────────────────────────────
    print(f"\nPunishment amount by profile:", flush=True)
    summary["punishment"] = {}
    profile_field_map = {
        "all_cooperate": ("punish_amount_cooperate", "punish_target_cooperate"),
        "mixed":         ("punish_amount_mixed", "punish_target_mixed"),
        "one_freerider": ("punish_amount_freerider", "punish_target_freerider"),
    }
    for profile_name, (amount_field, target_field) in profile_field_map.items():
        amounts = [_get_field(r["decisions"], amount_field) for r in valid]
        m = _safe_mean(amounts)
        nonzero = sum(1 for a in amounts if a is not None and float(a) > 0)
        summary["punishment"][profile_name] = {
            "mean": m, "nonzero": nonzero, "n": len(amounts),
        }
        if m is not None:
            print(
                f"  {profile_name:<20s}: {m:.2f} "
                f"(punished={nonzero}/{len(amounts)})",
                flush=True,
            )

    # Target distribution (across all profiles)
    all_targets = []
    for r in valid:
        for _, target_field in profile_field_map.values():
            t = _get_field(r["decisions"], target_field)
            if t is not None:
                all_targets.append(t)
    target_counts = {}
    for t in all_targets:
        target_counts[t] = target_counts.get(t, 0) + 1
    summary["targets"] = target_counts

    print(f"\nTarget distribution:", flush=True)
    for target in sorted(target_counts, key=target_counts.get, reverse=True):
        count = target_counts[target]
        pct = count / len(all_targets) if all_targets else 0
        print(f"  {target:<30s}: {count:>3d} ({pct:.0%})", flush=True)

    # ── Paper comparison ─────────────────────────────────────────
    comp = PaperComparison("Ertan, Page & Putterman (2009)")
    for key, (val, desc) in PAPER_FINDINGS.items():
        comp.add_finding(key, val, desc)
    comp.compare("vote_punish_low_pct", summary["vote_punish_low"])
    comp.compare("vote_punish_high_pct", summary["vote_punish_high"])
    comp.report()

    return summary


# ── Save raw data ───────────────────────────────────────────────────

def save_results(results: list[dict], prefix: str, output_dir: str = "results"):
    """
    Save raw bot results to a JSON file plus a flattened CSV.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Raw JSON
    json_path = os.path.join(output_dir, f"{prefix}_raw.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Flattened CSV (one row per bot, all decisions in columns)
    csv_path = os.path.join(output_dir, f"{prefix}_flat.csv")
    valid = [r for r in results if "error" not in r and r.get("decisions")]
    if valid:
        # Collect all field names that appear in any bot's decisions
        all_fields = set()
        for r in valid:
            for d in r["decisions"]:
                if "answers" in d:
                    all_fields.update(d["answers"].keys())
        all_fields = sorted(all_fields)

        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["agent"] + all_fields)
            for r in valid:
                row = [r["agent"]]
                for field in all_fields:
                    row.append(_get_field(r["decisions"], field))
                w.writerow(row)

    print(f"Saved to {output_dir}/{prefix}_*.{{json,csv}}", flush=True)
