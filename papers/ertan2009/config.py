"""
Ertan, Page & Putterman (2009) — paper-specific parameters and findings.

"Who to Punish? Individual Decisions and Majority Rule in Mitigating the
Free Rider Problem". European Economic Review, 53(5), 495-511.

Parameters match the original paper exactly.
"""

# ── Game parameters (match original paper) ──────────────────────────

ENDOWMENT = 10
GROUP_SIZE = 4
MPCR = 0.4               # marginal per capita return
PUNISHMENT_RATIO = 4     # 1 token spent → 4 removed from target

# ── Punishment regimes ──────────────────────────────────────────────

REGIMES = {
    "no_punishment": "NO punishment allowed.",
    "punish_low_only": (
        "Members CAN punish below-average contributors only. "
        f"Cost: 1 token, removes {PUNISHMENT_RATIO} from target."
    ),
    "punish_high_only": (
        "Members CAN punish above-average contributors only. "
        f"Cost: 1 token, removes {PUNISHMENT_RATIO} from target."
    ),
    "unrestricted": (
        "Members CAN punish ANY member. "
        f"Cost: 1 token, removes {PUNISHMENT_RATIO} from target."
    ),
}

ACTIVE_REGIMES = {k: v for k, v in REGIMES.items() if k != "no_punishment"}

# ── Contribution profiles for punishment decisions ──────────────────
# 3 other members (group of 4), endowment 10

PROFILES = [
    {"name": "all_cooperate",  "others": "9, 10, 8", "avg": 9.0},
    {"name": "mixed",          "others": "8, 3, 9",  "avg": 6.67},
    {"name": "one_freerider",  "others": "9, 8, 1",  "avg": 6.0},
]

# ── Known findings from the paper (for comparison) ──────────────────

PAPER_FINDINGS = {
    "vote_punish_low_pct":  (0.85, "% voting to allow punishing low contributors (late rounds)"),
    "vote_punish_high_pct": (0.00, "% voting to allow punishing high contributors"),
}
