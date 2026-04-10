"""
Example 1: Basic paper replication with default LLM agents.

Run from project root:
    docker compose up -d                  # start oTree server first
    python examples/01_basic_replication.py

This is the simplest possible use case. No personalities, no skews,
just LLM agents playing the Ertan 2009 public goods game on oTree.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # for `papers`

from replicant.otree import OTreeSession
from papers.ertan2009 import OTREE_SESSION_CONFIG, DEFAULT_SERVER, analyze


def main():
    session = OTreeSession(server_url=DEFAULT_SERVER, model="stepfun/step-3.5-flash")
    print(f"Running 10 default bots on {DEFAULT_SERVER}")
    results = session.run(OTREE_SESSION_CONFIG, n_bots=10)
    analyze(results, label="default")


if __name__ == "__main__":
    main()
