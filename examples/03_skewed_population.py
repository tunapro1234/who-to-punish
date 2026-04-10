"""
Example 3: Skewed population (e.g., all disagreeable agents).

Run from project root:
    docker compose up -d
    python examples/03_skewed_population.py

Tests how a non-representative population behaves in the same game.
What happens when everyone is selfish? Or anxious?

Expected: low-agreeableness agents contribute less and punish more
(including antisocial punishment of cooperators).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # for `papers`

from replicant import sample_personalities
from replicant.otree import OTreeSession
from papers.ertan2009 import OTREE_SESSION_CONFIG, DEFAULT_SERVER, analyze


def main():
    n = 10

    # All-disagreeable population
    # Override the agreeableness mean from 3.8 (default) to 1.5
    personalities = sample_personalities(n=n, seed=42, agreeableness=1.5)
    print(f"Created {n} disagreeable agents (agreeableness mean = 1.5)")

    session = OTreeSession(server_url=DEFAULT_SERVER, model="stepfun/step-3.5-flash")
    results = session.run(
        OTREE_SESSION_CONFIG,
        n_bots=n,
        personalities=personalities,
    )
    analyze(results, label="disagreeable")


if __name__ == "__main__":
    main()
