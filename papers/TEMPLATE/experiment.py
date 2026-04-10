"""
TEMPLATE — experiment definition.

A paper replication points to an oTree app and provides metadata.
The actual experiment lives at tests/otree_server/<your_paper_name>/.
"""

import os

# The oTree session config name (defined in tests/otree_server/settings.py)
OTREE_SESSION_CONFIG = "your_paper_name"  # CHANGE ME

# Default oTree server URL
DEFAULT_SERVER = "http://localhost:8000"

# Path to the oTree app source (for documentation)
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
OTREE_APP_PATH = os.path.join(_PROJECT_ROOT, "tests", "otree_server", OTREE_SESSION_CONFIG)
