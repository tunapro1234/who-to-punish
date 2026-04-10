"""
TEMPLATE — paper replication scaffold.

Copy this directory and rename to start a new paper replication.
See README.md in this directory for instructions.
"""

from .config import PAPER_FINDINGS
from .experiment import OTREE_SESSION_CONFIG, OTREE_APP_PATH, DEFAULT_SERVER
from .analyze import analyze, save_results

__all__ = [
    "OTREE_SESSION_CONFIG", "OTREE_APP_PATH", "DEFAULT_SERVER",
    "analyze", "save_results", "PAPER_FINDINGS",
]
