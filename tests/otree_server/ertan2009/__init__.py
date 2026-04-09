from otree.api import *

doc = """
Ertan, Page & Putterman (2009) — "Who to Punish?"
Public goods game with voting on punishment rules.

4 parts:
  1. Baseline contribution (no punishment)
  2. Vote on punishment rules
  3. Contribution under 4 punishment regimes
  4. Punishment decisions under 3 contribution profiles
"""


class C(BaseConstants):
    NAME_IN_URL = 'ertan2009'
    PLAYERS_PER_GROUP = None  # individual decisions (no group interaction needed for LLM replication)
    NUM_ROUNDS = 1
    ENDOWMENT = 20
    GROUP_SIZE = 5
    MPCR = 0.4
    PUNISHMENT_RATIO = 3


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    # Part 1: Baseline contribution
    contribution = models.IntegerField(
        min=0, max=C.ENDOWMENT,
        label=(
            f"You are in a group of {C.GROUP_SIZE} people. "
            f"Each person receives {C.ENDOWMENT} tokens. "
            f"Each person independently decides how many tokens to contribute to a group project. "
            f"For every token contributed, each group member (including you) receives {C.MPCR} tokens. "
            f"Tokens you keep remain yours. "
            f"There is NO punishment mechanism in this round. "
            f"How many tokens (0 to {C.ENDOWMENT}) do you contribute?"
        ),
    )

    # Part 2: Voting
    vote_punish_low = models.StringField(
        choices=['Yes', 'No'],
        widget=widgets.RadioSelect,
        label=(
            "Your group will vote on punishment rules. "
            f"Punishment costs 1 token to the punisher and removes {C.PUNISHMENT_RATIO} "
            "tokens from the target. "
            "Should members be ALLOWED to punish those who contribute "
            "BELOW the group average?"
        ),
    )
    vote_punish_high = models.StringField(
        choices=['Yes', 'No'],
        widget=widgets.RadioSelect,
        label=(
            "Should members be ALLOWED to punish those who contribute "
            "ABOVE the group average?"
        ),
    )

    # Part 3: Contribution under regimes
    contrib_no_punishment = models.IntegerField(
        min=0, max=C.ENDOWMENT,
        label=(
            f"You are in a group of {C.GROUP_SIZE} people. "
            f"Each person receives {C.ENDOWMENT} tokens. "
            f"For every token contributed, each member receives {C.MPCR} tokens. "
            "Punishment rule: NO punishment allowed. "
            f"How many tokens (0 to {C.ENDOWMENT}) do you contribute?"
        ),
    )
    contrib_punish_low = models.IntegerField(
        min=0, max=C.ENDOWMENT,
        label=(
            f"You are in a group of {C.GROUP_SIZE} people. "
            f"Each person receives {C.ENDOWMENT} tokens. "
            f"For every token contributed, each member receives {C.MPCR} tokens. "
            f"Punishment rule: Members CAN punish below-average contributors only. "
            f"Cost: 1 token, removes {C.PUNISHMENT_RATIO} from target. "
            f"How many tokens (0 to {C.ENDOWMENT}) do you contribute?"
        ),
    )
    contrib_punish_high = models.IntegerField(
        min=0, max=C.ENDOWMENT,
        label=(
            f"You are in a group of {C.GROUP_SIZE} people. "
            f"Each person receives {C.ENDOWMENT} tokens. "
            f"For every token contributed, each member receives {C.MPCR} tokens. "
            f"Punishment rule: Members CAN punish above-average contributors only. "
            f"Cost: 1 token, removes {C.PUNISHMENT_RATIO} from target. "
            f"How many tokens (0 to {C.ENDOWMENT}) do you contribute?"
        ),
    )
    contrib_unrestricted = models.IntegerField(
        min=0, max=C.ENDOWMENT,
        label=(
            f"You are in a group of {C.GROUP_SIZE} people. "
            f"Each person receives {C.ENDOWMENT} tokens. "
            f"For every token contributed, each member receives {C.MPCR} tokens. "
            f"Punishment rule: Members CAN punish ANY member. "
            f"Cost: 1 token, removes {C.PUNISHMENT_RATIO} from target. "
            f"How many tokens (0 to {C.ENDOWMENT}) do you contribute?"
        ),
    )

    # Part 4: Punishment decisions (3 profiles × 3 regimes = 9, but keep key ones)
    # Profile: mixed (others contributed 15, 5, 18, 2 — avg 10.0)
    punish_amount_mixed = models.IntegerField(
        min=0, max=5,
        label=(
            f"You are in a group of {C.GROUP_SIZE} with {C.ENDOWMENT} tokens each. "
            f"For every token contributed, each member receives {C.MPCR} tokens. "
            f"Punishment rule: Members CAN punish below-average contributors only. "
            f"Cost: 1 token, removes {C.PUNISHMENT_RATIO} from target. "
            "The other members contributed: 15, 5, 18, 2 tokens. "
            "Group average: 10.0. You contributed 15 tokens. "
            f"Each token you spend on punishment removes {C.PUNISHMENT_RATIO} "
            "tokens from the target. "
            "How many tokens (0 to 5) do you spend on punishing?"
        ),
    )
    punish_target_mixed = models.StringField(
        choices=[
            'Lowest contributor',
            'Below-average contributors',
            'Above-average contributors',
            'Highest contributor',
            'Nobody',
        ],
        widget=widgets.RadioSelect,
        label=(
            "Who do you primarily target with your punishment? "
            "Others contributed: 15, 5, 18, 2 (avg: 10.0)."
        ),
    )

    # Profile: one_freerider (others contributed 17, 18, 16, 2 — avg 13.25)
    punish_amount_freerider = models.IntegerField(
        min=0, max=5,
        label=(
            f"You are in a group of {C.GROUP_SIZE} with {C.ENDOWMENT} tokens each. "
            f"For every token contributed, each member receives {C.MPCR} tokens. "
            f"Punishment rule: Members CAN punish below-average contributors only. "
            f"Cost: 1 token, removes {C.PUNISHMENT_RATIO} from target. "
            "The other members contributed: 17, 18, 16, 2 tokens. "
            "Group average: 13.25. You contributed 15 tokens. "
            f"Each token you spend on punishment removes {C.PUNISHMENT_RATIO} "
            "tokens from the target. "
            "How many tokens (0 to 5) do you spend on punishing?"
        ),
    )
    punish_target_freerider = models.StringField(
        choices=[
            'Lowest contributor',
            'Below-average contributors',
            'Above-average contributors',
            'Highest contributor',
            'Nobody',
        ],
        widget=widgets.RadioSelect,
        label=(
            "Who do you primarily target with your punishment? "
            "Others contributed: 17, 18, 16, 2 (avg: 13.25)."
        ),
    )

    # Profile: all_cooperate (others contributed 18, 17, 19, 16 — avg 17.5)
    punish_amount_cooperate = models.IntegerField(
        min=0, max=5,
        label=(
            f"You are in a group of {C.GROUP_SIZE} with {C.ENDOWMENT} tokens each. "
            f"For every token contributed, each member receives {C.MPCR} tokens. "
            f"Punishment rule: Members CAN punish below-average contributors only. "
            f"Cost: 1 token, removes {C.PUNISHMENT_RATIO} from target. "
            "The other members contributed: 18, 17, 19, 16 tokens. "
            "Group average: 17.5. You contributed 15 tokens. "
            f"Each token you spend on punishment removes {C.PUNISHMENT_RATIO} "
            "tokens from the target. "
            "How many tokens (0 to 5) do you spend on punishing?"
        ),
    )
    punish_target_cooperate = models.StringField(
        choices=[
            'Lowest contributor',
            'Below-average contributors',
            'Above-average contributors',
            'Highest contributor',
            'Nobody',
        ],
        widget=widgets.RadioSelect,
        label=(
            "Who do you primarily target with your punishment? "
            "Others contributed: 18, 17, 19, 16 (avg: 17.5)."
        ),
    )


# ── Pages ───────────────────────────────────────────────────────────

class Baseline(Page):
    form_model = 'player'
    form_fields = ['contribution']


class Voting(Page):
    form_model = 'player'
    form_fields = ['vote_punish_low', 'vote_punish_high']


class Regimes(Page):
    form_model = 'player'
    form_fields = [
        'contrib_no_punishment',
        'contrib_punish_low',
        'contrib_punish_high',
        'contrib_unrestricted',
    ]


class Punishment(Page):
    form_model = 'player'
    form_fields = [
        'punish_amount_mixed', 'punish_target_mixed',
        'punish_amount_freerider', 'punish_target_freerider',
        'punish_amount_cooperate', 'punish_target_cooperate',
    ]


class Done(Page):
    pass


page_sequence = [Baseline, Voting, Regimes, Punishment, Done]
