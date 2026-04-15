"""
Ertan 2009 — figures for the replication paper.

Reads papers/ertan2009/results/*_raw.json and writes PDFs to
papers/ertan2009/replicated/figures/.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PAPER_DIR, "results")
FIGURES_DIR = os.path.join(PAPER_DIR, "replicated", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

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

CONDITIONS = [
    ("default", "Default (no personality)", "#888888"),
    ("random", "Random personalities", "#1f77b4"),
]

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


def load_condition(prefix: str):
    path = os.path.join(RESULTS_DIR, f"{prefix}_raw.json")
    with open(path) as f:
        data = json.load(f)

    def get_field(decisions, field):
        for d in decisions:
            if "answers" in d and field in d["answers"]:
                return d["answers"][field]
        return None

    def nums(field):
        return [float(get_field(a["decisions"], field))
                for a in data
                if get_field(a["decisions"], field) is not None]

    def pct_yes(field):
        yes = sum(1 for a in data if get_field(a["decisions"], field) == "Yes")
        return yes / len(data)

    return {"raw": data, "nums": nums, "pct_yes": pct_yes}


def fig1_regime_contributions(loaded):
    """Grouped bar chart: contribution by regime for each condition."""
    fig, ax = plt.subplots(figsize=(9, 5))
    keys = list(loaded.keys())
    x = np.arange(len(REGIMES))
    width = 0.8 / len(keys)

    for i, key in enumerate(keys):
        means, stds = [], []
        for regime in REGIMES:
            vals = loaded[key]["nums"](REGIME_FIELDS[regime])
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
        offset = (i - (len(keys) - 1) / 2) * width
        label = next((c[1] for c in CONDITIONS if c[0] == key), key)
        color = next((c[2] for c in CONDITIONS if c[0] == key), "#888")
        ax.bar(x + offset, means, width, yerr=stds,
               label=label, color=color,
               capsize=3, error_kw={"elinewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(REGIME_LABELS)
    ax.set_ylabel("Contribution (tokens, max = 10)")
    ax.set_title("Contribution by Punishment Regime (v2 parameters)")
    ax.set_ylim(0, 11)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.savefig(os.path.join(FIGURES_DIR, "fig1_regime_contributions.pdf"))
    plt.close(fig)
    print("wrote fig1_regime_contributions.pdf")


def fig2_baseline_distribution(loaded):
    """Histogram of baseline contributions per condition."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(-0.5, 11.5, 1)
    for key in loaded:
        vals = loaded[key]["nums"]("contribution")
        label = next((c[1] for c in CONDITIONS if c[0] == key), key)
        color = next((c[2] for c in CONDITIONS if c[0] == key), "#888")
        ax.hist(vals, bins=bins, alpha=0.6,
                label=f"{label} (M={np.mean(vals):.2f})",
                color=color, edgecolor="black", linewidth=0.4)

    ax.set_xlabel("Baseline contribution (tokens, max = 10)")
    ax.set_ylabel("Number of agents")
    ax.set_title("Distribution of Baseline Contributions (v2 parameters)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_xticks(range(0, 11))
    fig.savefig(os.path.join(FIGURES_DIR, "fig2_baseline_distribution.pdf"))
    plt.close(fig)
    print("wrote fig2_baseline_distribution.pdf")


def fig3_voting(loaded):
    """Bar chart of voting percentages."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    keys = list(loaded.keys())
    x = np.arange(2)  # two votes
    width = 0.35

    for i, key in enumerate(keys):
        low = loaded[key]["pct_yes"]("vote_punish_low") * 100
        high = loaded[key]["pct_yes"]("vote_punish_high") * 100
        label = next((c[1] for c in CONDITIONS if c[0] == key), key)
        color = next((c[2] for c in CONDITIONS if c[0] == key), "#888")
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, [low, high], width,
                      label=label, color=color, edgecolor="black",
                      linewidth=0.4)
        for b, v in zip(bars, [low, high]):
            ax.text(b.get_x() + b.get_width() / 2, v + 1,
                    f"{v:.0f}%", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(["Allow punish\nlow contributors",
                        "Allow punish\nhigh contributors"])
    ax.set_ylabel("% voting Yes")
    ax.set_title("Voting on Punishment Institutions (v2 parameters)")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.savefig(os.path.join(FIGURES_DIR, "fig3_voting.pdf"))
    plt.close(fig)
    print("wrote fig3_voting.pdf")


def fig4_punishment_by_profile(loaded):
    """Grouped bar: mean punishment amount per profile, per condition."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    keys = list(loaded.keys())
    x = np.arange(len(PROFILES))
    width = 0.8 / len(keys)

    for i, key in enumerate(keys):
        means, stds = [], []
        for profile in PROFILES:
            vals = loaded[key]["nums"](PROFILE_FIELDS[profile])
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
        offset = (i - (len(keys) - 1) / 2) * width
        label = next((c[1] for c in CONDITIONS if c[0] == key), key)
        color = next((c[2] for c in CONDITIONS if c[0] == key), "#888")
        ax.bar(x + offset, means, width, yerr=stds,
               label=label, color=color,
               capsize=3, error_kw={"elinewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(PROFILE_LABELS)
    ax.set_ylabel("Mean punishment spending (tokens, max = 10)")
    ax.set_title("Costly Punishment by Group Profile (v2 parameters)")
    ax.set_ylim(0, 5)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.savefig(os.path.join(FIGURES_DIR, "fig4_punishment_by_profile.pdf"))
    plt.close(fig)
    print("wrote fig4_punishment_by_profile.pdf")


def main():
    loaded = {}
    for prefix, _, _ in CONDITIONS:
        try:
            loaded[prefix] = load_condition(prefix)
        except FileNotFoundError:
            print(f"[skip] {prefix} — missing")

    if not loaded:
        print("No conditions loaded.")
        return

    fig1_regime_contributions(loaded)
    fig2_baseline_distribution(loaded)
    fig3_voting(loaded)
    fig4_punishment_by_profile(loaded)
    print(f"\nAll figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
