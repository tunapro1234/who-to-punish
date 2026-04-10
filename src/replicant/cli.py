"""
Generic CLI runner for paper replications.

Each paper's run.py just calls run_paper() with its session config and
analyze function. All the argparse boilerplate, personality sampling,
and result saving is handled here.
"""

import os
import json
import argparse
from typing import Callable

from .otree import OTreeSession
from .personalities.factory import sample_personalities


def build_parser(description: str = "Paper replication") -> argparse.ArgumentParser:
    """Build the standard CLI argument parser."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-n", type=int, default=10, help="Number of LLM bots")
    parser.add_argument("--server", default="http://localhost:8000",
                        help="oTree server URL")
    parser.add_argument("--model", default="stepfun/step-3.5-flash")
    parser.add_argument("--random-personalities", action="store_true",
                        help="Sample personalities from US adult population norms")
    parser.add_argument("--skew-extraversion", type=float, default=None)
    parser.add_argument("--skew-agreeableness", type=float, default=None)
    parser.add_argument("--skew-conscientiousness", type=float, default=None)
    parser.add_argument("--skew-neuroticism", type=float, default=None)
    parser.add_argument("--skew-openness", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-prefix", default=None)
    return parser


def determine_condition(args: argparse.Namespace) -> tuple[list[str] | None, str]:
    """
    From parsed args, determine personalities to use and a condition label.

    Returns:
        (personalities or None, condition_label)
    """
    skews = {
        "extraversion": args.skew_extraversion,
        "agreeableness": args.skew_agreeableness,
        "conscientiousness": args.skew_conscientiousness,
        "neuroticism": args.skew_neuroticism,
        "openness": args.skew_openness,
    }
    has_skew = any(v is not None for v in skews.values())

    if not args.random_personalities and not has_skew:
        return None, "default"

    personalities = sample_personalities(
        n=args.n,
        seed=args.seed,
        **{k: v for k, v in skews.items() if v is not None},
    )

    if has_skew:
        skewed_label = "_".join(
            f"{k[:3]}{v}" for k, v in skews.items() if v is not None
        )
        return personalities, f"skewed_{skewed_label}"

    return personalities, "random_population"


def run_paper(
    session_config: str,
    analyze_fn: Callable,
    save_fn: Callable,
    results_dir: str,
    description: str = "Paper replication",
    args: argparse.Namespace = None,
):
    """
    Generic main() for paper run.py scripts.

    Args:
        session_config: oTree session config name (e.g. "ertan2009")
        analyze_fn: function(results, label=...) -> summary dict
        save_fn: function(results, prefix, output_dir=...)
        results_dir: directory to save results to
        description: argparse description
        args: optional pre-parsed args (for testing); if None, parses sys.argv

    Usage in a paper's run.py:
        from replicant.cli import run_paper
        from .analyze import analyze, save_results

        if __name__ == "__main__":
            run_paper(
                session_config="ertan2009",
                analyze_fn=analyze,
                save_fn=save_results,
                results_dir=os.path.join(os.path.dirname(__file__), "results"),
            )
    """
    if args is None:
        parser = build_parser(description)
        args = parser.parse_args()

    personalities, condition = determine_condition(args)
    prefix = args.output_prefix or condition

    print(f"Running {args.n} bots on {args.server}", flush=True)
    print(f"Condition: {condition}", flush=True)

    session = OTreeSession(server_url=args.server, model=args.model)
    results = session.run(
        session_config,
        n_bots=args.n,
        personalities=personalities,
        rest_key=os.environ.get("OTREE_REST_KEY", "test-rest-key"),
    )

    os.makedirs(results_dir, exist_ok=True)
    summary = analyze_fn(results, label=condition)
    save_fn(results, prefix, output_dir=results_dir)

    summary_path = os.path.join(results_dir, f"{prefix}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved to {summary_path}", flush=True)

    return results, summary
