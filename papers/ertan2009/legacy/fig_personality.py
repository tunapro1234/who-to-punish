"""Generate personality breakdown figure."""
import os, random
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from replicant.personalities import POPULATION_NORMS, DOMAINS

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "results")
FIGURES_DIR = os.path.join(_HERE, "replicated", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Recreate agent traits (seed=42)
rng = random.Random(42)
agent_traits = []
for i in range(50):
    weights = {}
    for domain, norms in POPULATION_NORMS.items():
        score = rng.gauss(norms["mean"], norms["sd"])
        score = max(1.0, min(5.0, score))
        weights[domain] = score
    agent_traits.append({"name": f"random_{i+1}", **weights})
traits_df = pd.DataFrame(agent_traits)

baseline = pd.read_csv(os.path.join(RESULTS_DIR, "big5_random_baseline.csv"))
merged = baseline.merge(traits_df, left_on="agent.agent_name", right_on="name")

# Figure 5: Personality correlations with baseline contribution
fig, axes = plt.subplots(1, 5, figsize=(14, 3.2), sharey=True)

domain_labels = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

for ax, domain, label, color in zip(axes, DOMAINS, domain_labels, colors):
    x = merged[domain]
    y = merged["answer.contribution"].astype(float)
    ax.scatter(x, y, alpha=0.5, s=30, color=color)

    # Regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 50)
    ax.plot(x_line, p(x_line), color=color, linewidth=2)

    r = x.corr(y)
    ax.set_title(f"{label}\nr = {r:+.2f}", fontsize=10)
    ax.set_xlabel("Score (1-5)", fontsize=9)
    if ax == axes[0]:
        ax.set_ylabel("Contribution", fontsize=10)
    ax.set_xlim(1, 5)
    ax.set_ylim(-1, 22)

fig.suptitle("Personality Trait Scores vs. Baseline Contribution", fontsize=12, y=1.02)
fig.tight_layout()

path = os.path.join(FIGURES_DIR, "fig5_personality_correlations.pdf")
fig.savefig(path)
print(f"Saved {path}", flush=True)
plt.close(fig)
