"""
Ertan, Page & Putterman (2009) — experiment definition.

Points to the oTree app at tests/otree_server/ertan2009_v2/ and provides
helpers for running the experiment with LLM bots.

The "v2" app is the canonical replication: it matches the original paper's
parameters exactly (group=4, endowment=10, MPCR=0.4, punishment ratio 1:4).
The older ertan2009 oTree app (group=5, endowment=20, ratio 1:3) is kept
on the server for backwards compatibility but is not used by this paper.
"""

import os

# The oTree app session config name (defined in tests/otree_server/settings.py)
OTREE_SESSION_CONFIG = "ertan2009_v2"

# Default oTree server URL
DEFAULT_SERVER = "http://localhost:8000"

# Path to the oTree app source (for documentation / parser use)
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
OTREE_APP_PATH = os.path.join(_PROJECT_ROOT, "tests", "otree_server", "ertan2009_v2")
