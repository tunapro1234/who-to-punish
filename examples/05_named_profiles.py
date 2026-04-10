"""
Example 5: Named personality profiles (cooperative, selfish, leader, etc.).

Run from project root:
    python examples/05_named_profiles.py

Instead of sampling from a distribution, use one of the 5 predefined
personality archetypes:
  - cooperative: high agreeableness + conscientiousness, low neuroticism
  - selfish: low agreeableness + conscientiousness, high neuroticism
  - leader: high extraversion + conscientiousness + openness
  - anxious: low extraversion, high neuroticism, low openness
  - average: all traits at the population mean

Useful for comparing how distinct personality types affect behavior.
"""

from replicant import PersonalityFactory
from replicant.personalities import PROFILES


def main():
    print("Available profiles:")
    PersonalityFactory.list_profiles()

    factory = PersonalityFactory()

    # Show what each profile looks like
    print("\n" + "=" * 60)
    print("Profile descriptions")
    print("=" * 60)
    for profile_name in PROFILES:
        agent = factory.create_profile(profile_name, n=1)[0]
        print(f"\n--- {profile_name.upper()} ---")
        print(agent.traits["personality"])


if __name__ == "__main__":
    main()
