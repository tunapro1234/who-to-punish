"""
TEMPLATE — CLI entry point.

Usage (from project root):
    python -m papers.template.run -n 50
    python -m papers.template.run -n 50 --random-personalities
    python -m papers.template.run -n 20 --big5
"""

import os
import json
import argparse

from replicant import PersonalityFactory

from .experiment import build
from .analyze import analyze, save_results


PAPER_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(PAPER_DIR, "results")


def main():
    parser = argparse.ArgumentParser(description="Paper replication")
    parser.add_argument("-n", type=int, default=50, help="Number of agents")
    parser.add_argument("--model", default="stepfun/step-3.5-flash")
    parser.add_argument("--big5", action="store_true",
                        help="Use BFI-2 named profiles (balanced)")
    parser.add_argument("--random-personalities", action="store_true",
                        help="Use random BFI-2 personalities sampled from population norms")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    exp = build(model=args.model)
    factory = PersonalityFactory()

    if args.big5:
        agents = factory.create_population(per_profile=max(1, args.n // 5))
        condition = "big5_profiles"
        print(f"BFI-2 profile agents: {len(agents)}", flush=True)
    elif args.random_personalities:
        agents = factory.create_random_population(n=args.n, seed=args.seed)
        condition = "big5_random"
        print(f"Random personality agents: {len(agents)} (seed={args.seed})", flush=True)
    else:
        agents = factory.create_default(n=args.n)
        condition = "default"

    prefix = args.output_prefix or condition

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = exp.run(agents)
    summary = analyze(results, label=condition)
    save_results(results, prefix, output_dir=RESULTS_DIR)

    summary_path = os.path.join(RESULTS_DIR, f"{prefix}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
