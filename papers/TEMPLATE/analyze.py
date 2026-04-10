"""
TEMPLATE — analysis of oTree bot results.

The oTree pipeline returns a list of bot result dicts. Use _get_field()
to extract specific form values across pages.
"""

import os
import json
import csv

from replicant import PaperComparison
from .config import PAPER_FINDINGS


def _get_field(decisions: list[dict], field_name: str):
    """Find a field's answer across the bot's pages."""
    for d in decisions:
        if "answers" in d and field_name in d["answers"]:
            return d["answers"][field_name]
    return None


def _safe_mean(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    try:
        return sum(float(v) for v in clean) / len(clean)
    except (TypeError, ValueError):
        return None


def analyze(results: list[dict], label: str = "") -> dict:
    """
    Analyze a list of bot results from an oTree session.
    Customize the metrics extracted to match your paper.
    """
    prefix = f"[{label}] " if label else ""
    print(f"\n{'='*55}", flush=True)
    print(f"{prefix}RESULTS SUMMARY", flush=True)
    print(f"{'='*55}", flush=True)

    summary = {"label": label, "n": len(results)}

    valid = [r for r in results if "error" not in r and r.get("decisions")]
    print(f"\n{len(valid)}/{len(results)} bots completed", flush=True)
    if not valid:
        return summary

    # Example: extract a metric from your form fields
    # offers = [_get_field(r["decisions"], "offer") for r in valid]
    # summary["mean_offer"] = _safe_mean(offers)
    # if summary["mean_offer"] is not None:
    #     print(f"Mean offer: {summary['mean_offer']:.2f}", flush=True)

    # Compare to paper findings
    if PAPER_FINDINGS:
        comp = PaperComparison("Your Paper Citation")
        for key, (val, desc) in PAPER_FINDINGS.items():
            comp.add_finding(key, val, desc)
            if key in summary:
                comp.compare(key, summary[key])
        comp.report()

    return summary


def save_results(results: list[dict], prefix: str, output_dir: str = "results"):
    """Save raw results to JSON + flattened CSV."""
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, f"{prefix}_raw.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    valid = [r for r in results if "error" not in r and r.get("decisions")]
    if valid:
        all_fields = set()
        for r in valid:
            for d in r["decisions"]:
                if "answers" in d:
                    all_fields.update(d["answers"].keys())
        all_fields = sorted(all_fields)

        csv_path = os.path.join(output_dir, f"{prefix}_flat.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["agent"] + all_fields)
            for r in valid:
                row = [r["agent"]]
                for field in all_fields:
                    row.append(_get_field(r["decisions"], field))
                w.writerow(row)

    print(f"Saved to {output_dir}/{prefix}_*", flush=True)
