"""
Ertan, Page & Putterman (2009) — experiment definition.

Wires the public goods game template to the paper-specific parameters.
"""

from replicant import BehavioralExperiment
from replicant.experiments.templates.public_goods import (
    contribution_survey,
    contribution_regime_survey,
    voting_survey,
    punishment_survey,
)

from .config import (
    ENDOWMENT, GROUP_SIZE, MPCR, PUNISHMENT_RATIO,
    REGIMES, ACTIVE_REGIMES, PROFILES,
)


def build(model: str = "stepfun/step-3.5-flash") -> BehavioralExperiment:
    """Build the full Ertan 2009 experiment."""
    exp = BehavioralExperiment("ertan2009", model=model)

    exp.add_part(
        "baseline",
        contribution_survey(ENDOWMENT, GROUP_SIZE, MPCR),
        description="Contribution without punishment",
    )

    exp.add_part(
        "voting",
        voting_survey(PUNISHMENT_RATIO),
        description="Vote on punishment rules",
    )

    survey_r, scenarios_r = contribution_regime_survey(
        ENDOWMENT, GROUP_SIZE, MPCR, REGIMES,
    )
    exp.add_part(
        "regimes", survey_r, scenarios=scenarios_r,
        description="Contribution under punishment regimes",
    )

    survey_p, scenarios_p = punishment_survey(
        ENDOWMENT, GROUP_SIZE, MPCR, PUNISHMENT_RATIO,
        ACTIVE_REGIMES, PROFILES,
    )
    exp.add_part(
        "punishment", survey_p, scenarios=scenarios_p,
        description="Punishment decisions",
    )

    return exp
