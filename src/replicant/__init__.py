"""
replicant — connect LLMs to oTree experiments with Big Five personalities.

Usage:
    from replicant.otree import HybridSession
    from replicant import sample_personalities, build_personality

    session = HybridSession("http://localhost:8000")
    urls = session.create("ertan2009", n_participants=5)

    personalities = sample_personalities(n=5, seed=42)
    results = session.run_bots(urls, personalities=personalities)

The personality system uses BFI-2-Expanded format (Soto & John, 2017),
validated at r=0.91 against the Mini-IPIP cross-instrument benchmark.
"""

from .personalities.factory import (
    PersonalityFactory,
    build_personality,
    sample_personalities,
)
from .analysis.comparison import PaperComparison

__version__ = "0.3.0"
__all__ = [
    "PersonalityFactory",
    "build_personality",
    "sample_personalities",
    "PaperComparison",
]
