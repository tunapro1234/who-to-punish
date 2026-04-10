"""
TEMPLATE — CLI entry point.

Usage (from project root):
    docker compose up -d
    python -m papers.your_paper.run -n 10
    python -m papers.your_paper.run -n 10 --random-personalities
"""

import os
from replicant.cli import run_paper

from .experiment import OTREE_SESSION_CONFIG
from .analyze import analyze, save_results

PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PAPER_DIR, "results")


if __name__ == "__main__":
    run_paper(
        session_config=OTREE_SESSION_CONFIG,
        analyze_fn=analyze,
        save_fn=save_results,
        results_dir=RESULTS_DIR,
        description="My paper replication",
    )
