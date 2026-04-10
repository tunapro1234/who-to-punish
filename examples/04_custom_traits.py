"""
Example 4: Direct OCEAN trait input.

Run from project root:
    python examples/04_custom_traits.py

Sometimes you want exact control over each agent's personality. Use
build_personality() to specify all five Big Five scores directly.

Each trait is on a 1-5 scale (3.0 = average, 5.0 = high, 1.0 = low).
"""

from replicant import build_personality
from replicant.personalities import sample_personalities


def main():
    # ── Single personality with exact OCEAN scores ──
    print("=" * 60)
    print("Example A: One agent with explicit OCEAN scores")
    print("=" * 60)

    desc = build_personality(
        openness=4.5,           # high - curious, creative
        conscientiousness=2.0,  # low - careless, impulsive
        extraversion=4.0,       # high - outgoing
        agreeableness=1.5,      # very low - cold, hostile
        neuroticism=4.5,        # high - anxious, volatile
    )
    print(desc)

    # ── Sample N agents with one trait fixed ──
    print("\n" + "=" * 60)
    print("Example B: 5 agents, all with very low agreeableness (1.5)")
    print("Other traits sampled from population norms")
    print("=" * 60)

    personalities = sample_personalities(
        n=5,
        seed=42,
        agreeableness=1.5,
    )
    for i, p in enumerate(personalities, 1):
        print(f"\n--- Agent {i} ---")
        print(p[:300] + "...")

    # ── Sample with no variance (every agent identical) ──
    print("\n" + "=" * 60)
    print("Example C: 3 identical agents (sd_scale=0.0)")
    print("=" * 60)

    personalities = sample_personalities(
        n=3,
        agreeableness=2.0,
        neuroticism=4.5,
        sd_scale=0.0,
    )
    print("All 3 agents have identical descriptions (sd_scale=0)")
    print(f"First 200 chars: {personalities[0][:200]}")


if __name__ == "__main__":
    main()
