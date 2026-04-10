"""
Ertan 2009 — CLI entry point.

Usage (from project root):
    docker compose up -d                                  # start oTree server
    python -m papers.ertan2009.run -n 10
    python -m papers.ertan2009.run -n 10 --random-personalities
    python -m papers.ertan2009.run -n 10 --skew-agreeableness 1.5
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
        description="Ertan, Page & Putterman (2009) — Who to Punish",
    )
