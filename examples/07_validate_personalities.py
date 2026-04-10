"""
Example 7: Validate personality induction with cross-instrument testing.

Run from project root:
    python examples/07_validate_personalities.py

Before trusting your personality-conditioned experiment results, verify
that the personalities actually stick. We assign each agent a personality
via BFI-2-Expanded sentences and measure it back via the Mini-IPIP
20-item Likert questionnaire — different wording so the model can't
just parrot the assignment.

Two validation modes:
  - run_level_validation: % match at high/average/low classification
  - run_continuous_validation: Pearson r, R², MAE per Big Five domain
"""

from replicant.personalities import (
    run_level_validation,
    run_continuous_validation,
)


def main():
    # Mode 1: Continuous validation (more rigorous)
    # Sample 20 agents from population norms, measure each via Mini-IPIP,
    # compute Pearson r and MAE per domain.
    print("Running continuous validation (N=20)...")
    print("Expected: ~5-10 minutes with Step 3.5 Flash\n")

    results = run_continuous_validation(
        model="stepfun/step-3.5-flash",
        n=20,
        seed=42,
    )

    # Pooled correlation across all domains
    print(f"\nPooled Pearson r = {results['pooled']['r']:.3f}")
    print(f"Pooled R² = {results['pooled']['r_squared']:.3f}")

    # Mode 2: Level validation (faster, simpler metric)
    # Tests 5 named profiles, reports % match at high/average/low classification
    # Uncomment to also run:
    #
    # print("\nRunning level validation (5 profiles)...")
    # results = run_level_validation(model="stepfun/step-3.5-flash")
    # print(f"Overall match: {results['overall_match_pct']:.0%}")


if __name__ == "__main__":
    main()
