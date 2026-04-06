"""
PersonalityFactory — Generate EDSL Agents with BFI-2 personality descriptions.

Uses the Big Five Inventory 2 (Soto & John, 2017) approach: instead of giving
LLMs numeric trait scores (which don't reliably influence behavior), we use
expanded descriptive statements from the BFI-2 item pool to build rich
personality profiles.

Each agent gets a narrative personality description built from BFI-2-S items,
which the LLM can naturally embody.
"""

import random
from edsl import Agent, AgentList


# ── BFI-2-S items (Soto & John, 2017) ───────────────────────────────
# 30 items, 2 per facet, 6 per domain
# Each item starts with "I am someone who..."
# R = reverse-scored (describes the opposite pole)

BFI2_ITEMS = {
    "extraversion": {
        "sociability": [
            ("is outgoing, sociable", False),
            ("tends to be quiet", True),
        ],
        "assertiveness": [
            ("is dominant, acts as a leader", False),
            ("prefers to have others take charge", True),
        ],
        "energy_level": [
            ("is full of energy", False),
            ("is less active than other people", True),
        ],
    },
    "agreeableness": {
        "compassion": [
            ("is compassionate, has a soft heart", False),
            ("can be cold and uncaring", True),
        ],
        "respectfulness": [
            ("is respectful, treats others with respect", False),
            ("is sometimes rude to others", True),
        ],
        "trust": [
            ("assumes the best about people", False),
            ("tends to find fault with others", True),
        ],
    },
    "conscientiousness": {
        "organization": [
            ("keeps things neat and tidy", False),
            ("tends to be disorganized", True),
        ],
        "productiveness": [
            ("is persistent, works until the task is finished", False),
            ("has difficulty getting started on tasks", True),
        ],
        "responsibility": [
            ("is reliable, can always be counted on", False),
            ("can be somewhat careless", True),
        ],
    },
    "negative_emotionality": {
        "anxiety": [
            ("worries a lot", False),
            ("is relaxed, handles stress well", True),
        ],
        "depression": [
            ("tends to feel depressed, blue", False),
            ("feels secure, comfortable with self", True),
        ],
        "emotional_volatility": [
            ("is temperamental, gets emotional easily", False),
            ("is emotionally stable, not easily upset", True),
        ],
    },
    "open_mindedness": {
        "aesthetic_sensitivity": [
            ("is fascinated by art, music, or literature", False),
            ("has few artistic interests", True),
        ],
        "intellectual_curiosity": [
            ("is complex, a deep thinker", False),
            ("has little interest in abstract ideas", True),
        ],
        "creative_imagination": [
            ("is original, comes up with new ideas", False),
            ("has little creativity", True),
        ],
    },
}

# ── Population norms (approximate, US adults, 1-5 scale) ────────────
# From Soto & John (2017) and Srivastava et al. (2003)
POPULATION_NORMS = {
    "extraversion":          {"mean": 3.3, "sd": 0.75},
    "agreeableness":         {"mean": 3.8, "sd": 0.60},
    "conscientiousness":     {"mean": 3.7, "sd": 0.65},
    "negative_emotionality": {"mean": 2.8, "sd": 0.75},
    "open_mindedness":       {"mean": 3.6, "sd": 0.65},
}

DEFAULT_INSTRUCTION = (
    "You are a participant in an experiment. "
    "Your personality description below reflects who you are. "
    "Let it naturally shape your decisions and reasoning — "
    "act as a real person with this personality would."
)


def _build_description(levels):
    """
    Build a natural personality description from domain levels.

    Args:
        levels: dict of {domain: "high"|"low"|"average"}
    Returns:
        str: narrative personality description
    """
    lines = ["I am someone who:"]

    for domain, level in levels.items():
        items = BFI2_ITEMS[domain]
        for facet, item_pair in items.items():
            positive_item, _ = item_pair[0]
            negative_item, _ = item_pair[1]

            if level == "high":
                lines.append(f"- {positive_item}")
            elif level == "low":
                lines.append(f"- {negative_item}")
            # "average" — skip, don't include extreme descriptors

    return "\n".join(lines)


def _score_to_level(score):
    """Convert 1-5 score to high/average/low."""
    if score >= 4.0:
        return "high"
    elif score <= 2.0:
        return "low"
    else:
        return "average"


# ── Predefined profiles ─────────────────────────────────────────────

PROFILES = {
    "cooperative": {
        "extraversion": "average",
        "agreeableness": "high",
        "conscientiousness": "high",
        "negative_emotionality": "low",
        "open_mindedness": "average",
    },
    "selfish": {
        "extraversion": "average",
        "agreeableness": "low",
        "conscientiousness": "low",
        "negative_emotionality": "high",
        "open_mindedness": "average",
    },
    "leader": {
        "extraversion": "high",
        "agreeableness": "average",
        "conscientiousness": "high",
        "negative_emotionality": "low",
        "open_mindedness": "high",
    },
    "anxious": {
        "extraversion": "low",
        "agreeableness": "average",
        "conscientiousness": "average",
        "negative_emotionality": "high",
        "open_mindedness": "low",
    },
    "average": {
        "extraversion": "average",
        "agreeableness": "average",
        "conscientiousness": "average",
        "negative_emotionality": "average",
        "open_mindedness": "average",
    },
}


class PersonalityFactory:
    """
    Generate EDSL Agents with BFI-2 based personality descriptions.

    Usage:
        factory = PersonalityFactory()

        # Named profiles
        agents = factory.create_profile("cooperative", n=5)

        # All profiles, balanced
        agents = factory.create_population(per_profile=4)

        # Random from population distribution
        agents = factory.create_random_population(n=20)

        # Plain agents (no personality)
        agents = factory.create_default(n=10)
    """

    def __init__(self, instruction=None):
        self.instruction = instruction or DEFAULT_INSTRUCTION

    def _make_agent(self, name, levels):
        """Create an EDSL Agent from domain levels."""
        description = _build_description(levels)
        return Agent(
            name=name,
            traits={"personality": description},
            instruction=self.instruction,
        )

    def create_profile(self, profile_name, n=1):
        """Create n agents from a named profile."""
        if profile_name not in PROFILES:
            raise ValueError(
                f"Unknown profile: {profile_name}. "
                f"Choose from: {list(PROFILES.keys())}"
            )
        levels = PROFILES[profile_name]
        return AgentList([
            self._make_agent(f"{profile_name}_{i+1}", levels)
            for i in range(n)
        ])

    def create_population(self, per_profile=4):
        """Create a balanced population with all named profiles."""
        agents = []
        for name in PROFILES:
            agents.extend(self.create_profile(name, per_profile))
        return AgentList(agents)

    def create_random_population(self, n=20, seed=None):
        """
        Create agents with random personality scores sampled from
        population norms (Soto & John, 2017).
        """
        rng = random.Random(seed)
        agents = []

        for i in range(n):
            levels = {}
            for domain, norms in POPULATION_NORMS.items():
                score = rng.gauss(norms["mean"], norms["sd"])
                score = max(1.0, min(5.0, score))  # clamp to 1-5
                levels[domain] = _score_to_level(score)
            agents.append(self._make_agent(f"random_{i+1}", levels))

        return AgentList(agents)

    def create_custom(self, levels, n=1, name_prefix="custom"):
        """Create agents from custom domain levels dict."""
        return AgentList([
            self._make_agent(f"{name_prefix}_{i+1}", levels)
            for i in range(n)
        ])

    def create_default(self, n=10):
        """Create plain agents with no personality traits."""
        return AgentList([Agent(name=f"subject_{i+1}") for i in range(n)])

    @staticmethod
    def list_profiles():
        """Print available profiles."""
        for name, levels in PROFILES.items():
            summary = ", ".join(
                f"{d}={l}" for d, l in levels.items() if l != "average"
            )
            print(f"  {name:<15s}: {summary or 'all average'}")

    @staticmethod
    def show_description(profile_name):
        """Print the full personality description for a profile."""
        if profile_name not in PROFILES:
            raise ValueError(f"Unknown profile: {profile_name}")
        print(_build_description(PROFILES[profile_name]))
