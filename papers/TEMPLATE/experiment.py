"""
TEMPLATE — experiment definition.

Define the build() function which returns a BehavioralExperiment.
You can either:
  (a) Use a reusable template from replicant.experiments.templates
  (b) Build your own EDSL Survey from scratch
"""

from replicant import BehavioralExperiment

# Option A: use a reusable template
# from replicant.experiments.templates.public_goods import contribution_survey

# Option B: build your own from scratch
# from edsl import QuestionNumerical, QuestionMultipleChoice, Survey, Scenario

from .config import ENDOWMENT, GROUP_SIZE  # add other params as needed


def build(model: str = "stepfun/step-3.5-flash") -> BehavioralExperiment:
    """Build the experiment from paper parameters."""
    exp = BehavioralExperiment("template", model=model)

    # Example: a single-question survey
    # from edsl import QuestionNumerical, Survey
    # q = QuestionNumerical(
    #     question_name="offer",
    #     question_text=f"You have {ENDOWMENT} tokens. How many do you offer?",
    #     min_value=0,
    #     max_value=ENDOWMENT,
    # )
    # exp.add_part("offer", Survey(questions=[q]), description="Initial offer")

    return exp
