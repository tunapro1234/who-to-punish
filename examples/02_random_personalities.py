"""
Example 2: Replicate with random Big Five personalities sampled from norms.

Run from project root:
    docker compose up -d
    python examples/02_random_personalities.py

Each agent gets a unique personality sampled from US adult population
distributions (Soto & John, 2017).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # for `papers`

from replicant import sample_personalities
from replicant.otree import OTreeSession
from papers.ertan2009 import OTREE_SESSION_CONFIG, DEFAULT_SERVER, analyze


def main():
    n = 10

    # Sample N personalities from US adult Big Five norms
    personalities = sample_personalities(n=n, seed=42)
    print(f"Created {n} personalities from population norms")

    session = OTreeSession(server_url=DEFAULT_SERVER, model="stepfun/step-3.5-flash")
    results = session.run(
        OTREE_SESSION_CONFIG,
        n_bots=n,
        personalities=personalities,
    )
    analyze(results, label="random_population")


if __name__ == "__main__":
    main()
