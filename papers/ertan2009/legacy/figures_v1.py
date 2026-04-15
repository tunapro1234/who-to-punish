"""Generate paper figures from experiment results."""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "results")
FIGURES_DIR = os.path.join(_HERE, "replicated", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

REGIMES = ["no_punishment", "punish_low_only", "punish_high_only", "unrestricted"]
REGIME_LABELS = ["No\npunishment", "Punish low\nonly", "Punish high\nonly", "Unrestricted"]
PROFILES = ["all_cooperate", "mixed", "one_freerider"]
PROFILE_LABELS = ["All cooperate", "Mixed", "One free-rider"]

# Style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


def load(prefix):
    data = {}
    for part in ["baseline", "voting", "regimes", "punishment"]:
        path = os.path.join(RESULTS_DIR, f"{prefix}_{part}.csv")
        if os.path.exists(path):
            data[part] = pd.read_csv(path)
    return data


def fig1_regime_contributions(default, personality):
    """Bar chart: contribution by regime, default vs personality vs paper."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(REGIMES))
    width = 0.25

    # Default
    d_means = [default["regimes"][default["regimes"]["scenario.regime_name"] == r]["answer.contribution"].astype(float).mean() for r in REGIMES]
    d_stds = [default["regimes"][default["regimes"]["scenario.regime_name"] == r]["answer.contribution"].astype(float).std() for r in REGIMES]

    # Personality
    p_means = [personality["regimes"][personality["regimes"]["scenario.regime_name"] == r]["answer.contribution"].astype(float).mean() for r in REGIMES]
    p_stds = [personality["regimes"][personality["regimes"]["scenario.regime_name"] == r]["answer.contribution"].astype(float).std() for r in REGIMES]

    ax.bar(x - width, d_means, width, yerr=d_stds, label="Default", color="#4C72B0", capsize=3, alpha=0.85)
    ax.bar(x, p_means, width, yerr=p_stds, label="Personality", color="#DD8452", capsize=3, alpha=0.85)

    # Paper baseline reference line
    ax.axhline(y=7.5, color="gray", linestyle="--", linewidth=1, label="Paper baseline (~7.5)")

    ax.set_ylabel("Contribution (tokens)")
    ax.set_xlabel("Punishment regime")
    ax.set_xticks(x - width/2)
    ax.set_xticklabels(REGIME_LABELS)
    ax.set_ylim(0, 22)
    ax.legend(loc="upper right")
    ax.set_title("Mean Contribution by Punishment Regime")

    path = os.path.join(FIGURES_DIR, "fig1_regime_contributions.pdf")
    fig.savefig(path)
    print(f"Saved {path}", flush=True)
    plt.close(fig)


def fig2_baseline_distribution(default, personality):
    """Histogram: baseline contribution distribution, default vs personality."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

    d_vals = default["baseline"]["answer.contribution"].astype(float)
    p_vals = personality["baseline"]["answer.contribution"].astype(float)

    bins = np.arange(-0.5, 21.5, 1)

    ax1.hist(d_vals, bins=bins, color="#4C72B0", alpha=0.85, edgecolor="white")
    ax1.set_title("Default agents")
    ax1.set_xlabel("Contribution (tokens)")
    ax1.set_ylabel("Count")
    ax1.set_xlim(-1, 21)
    ax1.axvline(d_vals.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean={d_vals.mean():.1f}")
    ax1.legend()

    ax2.hist(p_vals, bins=bins, color="#DD8452", alpha=0.85, edgecolor="white")
    ax2.set_title("Personality agents")
    ax2.set_xlabel("Contribution (tokens)")
    ax2.set_xlim(-1, 21)
    ax2.axvline(p_vals.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean={p_vals.mean():.1f}")
    ax2.axvline(7.5, color="gray", linestyle=":", linewidth=1.5, label="Paper (~7.5)")
    ax2.legend()

    fig.suptitle("Baseline Contribution Distribution", fontsize=13)
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "fig2_baseline_distribution.pdf")
    fig.savefig(path)
    print(f"Saved {path}", flush=True)
    plt.close(fig)


def fig3_punishment_targets(default, personality):
    """Stacked bar: punishment target distribution by condition."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    targets = ["Lowest contributor", "Below-average contributors", "Nobody"]
    colors = ["#C44E52", "#DD8452", "#8C8C8C"]

    for label, data, x_pos in [("Default", default, 0), ("Personality", personality, 1)]:
        df_p = data["punishment"]
        total = len(df_p)
        bottom = 0
        for target, color in zip(targets, colors):
            count = (df_p["answer.punish_target"] == target).sum()
            pct = count / total
            ax.bar(x_pos, pct, bottom=bottom, color=color, width=0.5,
                   label=target if x_pos == 0 else "")
            if pct > 0.05:
                ax.text(x_pos, bottom + pct / 2, f"{pct:.0%}", ha="center", va="center", fontsize=10, color="white", fontweight="bold")
            bottom += pct

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Default", "Personality"])
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.set_title("Punishment Target Distribution")

    path = os.path.join(FIGURES_DIR, "fig3_punishment_targets.pdf")
    fig.savefig(path)
    print(f"Saved {path}", flush=True)
    plt.close(fig)


def fig4_punishment_by_profile(personality):
    """Bar chart: punishment amount by contribution profile."""
    fig, ax = plt.subplots(figsize=(6, 4))

    means = []
    stds = []
    rates = []
    df_p = personality["punishment"]
    for profile in PROFILES:
        vals = df_p[df_p["scenario.profile_name"] == profile]["answer.punish_amount"].astype(float)
        means.append(vals.mean())
        stds.append(vals.std())
        rates.append((vals > 0).sum() / len(vals))

    x = np.arange(len(PROFILES))
    bars = ax.bar(x, means, yerr=stds, color="#DD8452", capsize=4, alpha=0.85, width=0.5)

    # Add punishment rate labels
    for i, (bar, rate) in enumerate(zip(bars, rates)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[i] + 0.05,
                f"{rate:.0%}", ha="center", va="bottom", fontsize=10, color="#666")

    ax.set_xticks(x)
    ax.set_xticklabels(PROFILE_LABELS)
    ax.set_ylabel("Punishment tokens spent")
    ax.set_xlabel("Other members' contribution profile")
    ax.set_title("Punishment Amount by Group Composition\n(Personality condition)")
    ax.set_ylim(0, max(means) + max(stds) + 0.5)

    path = os.path.join(FIGURES_DIR, "fig4_punishment_by_profile.pdf")
    fig.savefig(path)
    print(f"Saved {path}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    print("Loading data...", flush=True)
    default = load("default")
    personality = load("big5_random")

    fig1_regime_contributions(default, personality)
    fig2_baseline_distribution(default, personality)
    fig3_punishment_targets(default, personality)
    fig4_punishment_by_profile(personality)
    print("All figures done.", flush=True)
