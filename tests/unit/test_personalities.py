"""Tests for personality generation (no API calls)."""

import pytest

from replicant import build_personality, sample_personalities, PersonalityFactory
from replicant.personalities import POPULATION_NORMS, DOMAINS, PROFILES


# ── build_personality ───────────────────────────────────────────────

def test_build_personality_default():
    desc = build_personality()
    assert isinstance(desc, str)
    assert len(desc) > 100
    assert "I am someone who" in desc


def test_build_personality_extreme():
    high = build_personality(agreeableness=5.0)
    low = build_personality(agreeableness=1.0)
    assert high != low
    # High should mention positive traits, low should mention negative
    assert "compassionate" in high or "kind" in high
    assert "cold" in low or "rude" in low or "fault" in low


def test_build_personality_has_15_lines():
    """Each personality has 3 sentences per domain × 5 domains = 15 lines."""
    desc = build_personality(extraversion=3.0)
    lines = [l for l in desc.split("\n") if l.strip()]
    assert len(lines) == 15


# ── sample_personalities ────────────────────────────────────────────

def test_sample_personalities_count():
    descs = sample_personalities(n=5, seed=42)
    assert len(descs) == 5
    assert all(isinstance(d, str) for d in descs)


def test_sample_personalities_reproducible():
    a = sample_personalities(n=5, seed=42)
    b = sample_personalities(n=5, seed=42)
    assert a == b


def test_sample_personalities_different_seeds():
    a = sample_personalities(n=5, seed=42)
    b = sample_personalities(n=5, seed=43)
    assert a != b


def test_sample_personalities_skewed():
    """Setting a trait mean should bias the population."""
    # All identical agents (sd_scale=0)
    descs = sample_personalities(n=3, seed=42, agreeableness=5.0, sd_scale=0.0)
    assert len(set(descs)) == 1  # all identical
    # With max agreeableness, all should mention positive traits
    assert "compassionate" in descs[0] or "kind" in descs[0]


def test_sample_personalities_zero_variance():
    """sd_scale=0 produces identical agents."""
    descs = sample_personalities(n=10, sd_scale=0.0)
    assert len(set(descs)) == 1


# ── PersonalityFactory ──────────────────────────────────────────────

def test_factory_default_agents():
    factory = PersonalityFactory()
    agents = factory.create_default(n=3)
    assert len(agents) == 3


def test_factory_named_profile():
    factory = PersonalityFactory()
    agents = factory.create_profile("cooperative", n=2)
    assert len(agents) == 2
    for agent in agents:
        assert "personality" in agent.traits


def test_factory_invalid_profile():
    factory = PersonalityFactory()
    with pytest.raises(ValueError):
        factory.create_profile("nonexistent_profile")


def test_factory_random_population():
    factory = PersonalityFactory()
    agents = factory.create_random_population(n=5, seed=42)
    assert len(agents) == 5
    for agent in agents:
        assert "personality" in agent.traits


def test_factory_random_with_overrides():
    factory = PersonalityFactory()
    agents = factory.create_random_population(
        n=3, seed=42, mean_overrides={"agreeableness": 1.5}
    )
    assert len(agents) == 3


def test_all_profiles_exist():
    expected = {"cooperative", "selfish", "leader", "anxious", "average"}
    assert expected == set(PROFILES.keys())


def test_population_norms_complete():
    for domain in DOMAINS:
        assert domain in POPULATION_NORMS
        assert "mean" in POPULATION_NORMS[domain]
        assert "sd" in POPULATION_NORMS[domain]
