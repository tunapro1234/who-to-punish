"""
Ertan, Page & Putterman (2009) — replication.

"Who to Punish? Individual Decisions and Majority Rule in Mitigating
the Free Rider Problem". European Economic Review, 53(5), 495-511.

Uses the oTree app at tests/otree_server/ertan2009/.
"""

from .config import (
    ENDOWMENT, GROUP_SIZE, MPCR, PUNISHMENT_RATIO,
    REGIMES, ACTIVE_REGIMES, PROFILES, PAPER_FINDINGS,
)
from .experiment import OTREE_SESSION_CONFIG, OTREE_APP_PATH, DEFAULT_SERVER
from .analyze import analyze, save_results

__all__ = [
    "OTREE_SESSION_CONFIG", "OTREE_APP_PATH", "DEFAULT_SERVER",
    "analyze", "save_results",
    "ENDOWMENT", "GROUP_SIZE", "MPCR", "PUNISHMENT_RATIO",
    "REGIMES", "ACTIVE_REGIMES", "PROFILES", "PAPER_FINDINGS",
]
