"""
Example 1: Basic paper replication with default LLM agents.

Run from project root:
    python examples/01_basic_replication.py

This is the simplest possible use case. No personalities, no skews,
just LLM agents playing the Ertan 2009 public goods game.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # for `papers`

from replicant import PersonalityFactory
from papers.ertan2009 import build, analyze


def main():
    # Build the experiment from paper config
    exp = build(model="stepfun/step-3.5-flash")

    # Create plain agents (no personality)
    factory = PersonalityFactory()
    agents = factory.create_default(n=10)

    # Run all 4 parts
    results = exp.run(agents)

    # Analyze and compare to the original paper
    analyze(results, label="default")


if __name__ == "__main__":
    main()
