"""
Ertan, Page & Putterman (2009) — experiment definition.

Points to the oTree app at tests/otree_server/ertan2009/ and provides
helpers for running the experiment with LLM bots.
"""

import os

# The oTree app session config name (defined in tests/otree_server/settings.py)
OTREE_SESSION_CONFIG = "ertan2009"

# Default oTree server URL
DEFAULT_SERVER = "http://localhost:8000"

# Path to the oTree app source (for documentation / parser use)
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
OTREE_APP_PATH = os.path.join(_PROJECT_ROOT, "tests", "otree_server", "ertan2009")
