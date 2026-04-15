"""
Ertan 2009 — regenerate paper figures from the new oTree pipeline results.

Reads flat CSVs for the three conditions and writes PDFs to
replicated/figures/. Figures match the structure of the existing paper.tex.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PAPER_DIR, "results")
FIGURES_DIR = os.path.join(PAPER_DIR, "replicated", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

CONDITIONS = [
    ("default", "otree_default", "Default", "#888888"),
    ("random", "otree_random", "Random personalities", "#1f77b4"),
    ("cooperative", "otree_cooperative", "High-agr, low-neu", "#2ca02c"),
]

REGIMES = ["no_punishment", "punish_low_only", "punish_high_only", "unrestricted"]
REGIME_LABELS = ["No\npunishment", "Punish low\nonly", "Punish high\nonly", "Unrestricted"]
REGIME_FIELDS = {
    "no_punishment": "contrib_no_punishment",
    "punish_low_only": "contrib_punish_low",
    "punish_high_only": "contrib_punish_high",
    "unrestricted": "contrib_unrestricted",
}

PROFILES = ["all_cooperate", "mixed", "one_freerider"]
PROFILE_LABELS = ["All cooperate", "Mixed", "One free-rider"]
PROFILE_FIELDS = {
    "all_cooperate": "punish_amount_cooperate",
    "mixed": "punish_amount_mixed",
    "one_freerider": "punish_amount_freerider",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

HUMAN_BASELINE = 7.5  # approximate, from Ertan 2009


def load_all():
    loaded = {}
    for key, prefix, label, color in CONDITIONS:
        csv_path = os.path.join(RESULTS_DIR, f"{prefix}_flat.csv")
        if not os.path.exists(csv_path):
            print(f"[skip] {prefix} — missing")
            continue
        df = pd.read_csv(csv_path)
        loaded[key] = {"df": df, "label": label, "color": color}
    return loaded


def fig1_regime_contributions(loaded):
    """Grouped bar chart: contribution by regime for each condition."""
    fig, ax = plt.subplots(figsize=(9, 5))
    keys = list(loaded.keys())
    x = np.arange(len(REGIMES))
    width = 0.8 / max(len(keys), 1)

    for i, key in enumerate(keys):
        df = loaded[key]["df"]
        means, stds = [], []
        for regime in REGIMES:
            vals = pd.to_numeric(df[REGIME_FIELDS[regime]], errors="coerce").dropna()
            means.append(vals.mean())
            stds.append(vals.std())
        offset = (i - (len(keys) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds,
               label=loaded[key]["label"], color=loaded[key]["color"],
               capsize=3, error_kw={"elinewidth": 0.8})

    ax.axhline(HUMAN_BASELINE, linestyle="--", color="red", linewidth=1, alpha=0.7,
               label=f"Human baseline ({HUMAN_BASELINE})")
    ax.set_xticks(x)
    ax.set_xticklabels(REGIME_LABELS)
    ax.set_ylabel("Contribution (tokens)")
    ax.set_title("Contribution by Punishment Regime")
    ax.set_ylim(0, 22)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.savefig(os.path.join(FIGURES_DIR, "fig1_regime_contributions.pdf"))
    plt.close(fig)
    print("wrote fig1_regime_contributions.pdf")


def fig2_baseline_distribution(loaded):
    """Histogram of baseline contributions per condition."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(-0.5, 21.5, 1)
    for key in loaded:
        df = loaded[key]["df"]
        vals = pd.to_numeric(df["contribution"], errors="coerce").dropna()
        ax.hist(vals, bins=bins, alpha=0.55,
                label=f"{loaded[key]['label']} (M={vals.mean():.1f})",
                color=loaded[key]["color"], edgecolor="black", linewidth=0.4)

    ax.axvline(HUMAN_BASELINE, linestyle="--", color="red", linewidth=1.2,
               label=f"Human baseline ({HUMAN_BASELINE})")
    ax.set_xlabel("Baseline contribution (tokens)")
    ax.set_ylabel("Number of agents")
    ax.set_title("Distribution of Baseline Contributions")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.savefig(os.path.join(FIGURES_DIR, "fig2_baseline_distribution.pdf"))
    plt.close(fig)
    print("wrote fig2_baseline_distribution.pdf")


def fig3_punishment_targets(loaded):
    """Stacked bar: punishment target distribution per condition."""
    target_order = ["Nobody", "Lowest contributor", "Below-average contributors"]
    target_colors = ["#cccccc", "#d62728", "#ff7f0e"]

    data = {}
    for key in loaded:
        df = loaded[key]["df"]
        all_targets = []
        for field in ["punish_target_mixed", "punish_target_freerider", "punish_target_cooperate"]:
            all_targets += df[field].dropna().astype(str).tolist()
        counts = {t: 0 for t in target_order}
        for t in all_targets:
            if t in counts:
                counts[t] += 1
        total = sum(counts.values()) or 1
        data[key] = {t: counts[t] / total * 100 for t in target_order}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    keys = list(loaded.keys())
    x = np.arange(len(keys))
    bottom = np.zeros(len(keys))
    for t, color in zip(target_order, target_colors):
        vals = np.array([data[k][t] for k in keys])
        ax.bar(x, vals, bottom=bottom, label=t, color=color, edgecolor="white")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([loaded[k]["label"] for k in keys])
    ax.set_ylabel("Share of punishment choices (%)")
    ax.set_title("Punishment Target Distribution")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    ax.set_ylim(0, 100)
    fig.savefig(os.path.join(FIGURES_DIR, "fig3_punishment_targets.pdf"))
    plt.close(fig)
    print("wrote fig3_punishment_targets.pdf")


def fig4_punishment_by_profile(loaded):
    """Grouped bar: mean punishment amount per profile, per condition."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    keys = list(loaded.keys())
    x = np.arange(len(PROFILES))
    width = 0.8 / max(len(keys), 1)

    for i, key in enumerate(keys):
        df = loaded[key]["df"]
        means, stds = [], []
        for profile in PROFILES:
            vals = pd.to_numeric(df[PROFILE_FIELDS[profile]], errors="coerce").dropna()
            means.append(vals.mean())
            stds.append(vals.std())
        offset = (i - (len(keys) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds,
               label=loaded[key]["label"], color=loaded[key]["color"],
               capsize=3, error_kw={"elinewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(PROFILE_LABELS)
    ax.set_ylabel("Mean punishment spending (tokens)")
    ax.set_title("Costly Punishment by Group Profile")
    ax.set_ylim(0, 5.2)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.savefig(os.path.join(FIGURES_DIR, "fig4_punishment_by_profile.pdf"))
    plt.close(fig)
    print("wrote fig4_punishment_by_profile.pdf")


def main():
    loaded = load_all()
    if not loaded:
        print("No conditions loaded. Run the experiments first.")
        return
    fig1_regime_contributions(loaded)
    fig2_baseline_distribution(loaded)
    fig3_punishment_targets(loaded)
    fig4_punishment_by_profile(loaded)
    print(f"\nAll figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
