"""
Example 2: Replicate with random Big Five personalities sampled from norms.

Run from project root:
    python examples/02_random_personalities.py

Each agent gets a unique personality sampled from US adult population
distributions (Soto & John, 2017). This is the most realistic baseline
for comparing LLM agents to human samples.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # for `papers`

from replicant import PersonalityFactory
from papers.ertan2009 import build, analyze


def main():
    exp = build(model="stepfun/step-3.5-flash")

    # Sample 20 agents from US adult Big Five population norms
    factory = PersonalityFactory()
    agents = factory.create_random_population(n=20, seed=42)

    print(f"Created {len(agents)} agents from population norms")

    results = exp.run(agents)
    analyze(results, label="random_population")


if __name__ == "__main__":
    main()
