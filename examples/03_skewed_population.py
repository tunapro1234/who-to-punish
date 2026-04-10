"""
Example 3: Skewed population (e.g., all disagreeable agents).

Run from project root:
    python examples/03_skewed_population.py

Tests how a non-representative population behaves in the same game.
What happens when everyone is selfish? Or anxious? Or very conscientious?

Expected: low-agreeableness agents contribute less and punish more
(including antisocial punishment of cooperators).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # for `papers`

from replicant import PersonalityFactory
from papers.ertan2009 import build, analyze


def main():
    exp = build(model="stepfun/step-3.5-flash")

    # All-disagreeable population
    # Override the agreeableness mean from 3.8 (default) to 1.5
    factory = PersonalityFactory()
    agents = factory.create_random_population(
        n=20,
        seed=42,
        mean_overrides={"agreeableness": 1.5},
    )

    print(f"Created {len(agents)} disagreeable agents")
    print("(agreeableness mean shifted from 3.8 to 1.5)")

    results = exp.run(agents)
    analyze(results, label="disagreeable")


if __name__ == "__main__":
    main()
