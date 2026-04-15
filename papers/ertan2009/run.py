"""
Ertan 2009 — exact parameter replication.

Runs N=50 default + N=50 random personalities on the ertan2009_v2 oTree
app (group=4, endowment=10, punishment ratio 1:4 — original paper
parameters) using step-3.5-flash in chat mode.

Run from project root after `docker compose up -d`:
    python papers/ertan2009/run.py
"""

import os
import sys
import json
import time

# Allow running from project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from replicant.otree.client import OTreeClient
from replicant.otree.bot import run_bots
from replicant.personalities.factory import sample_personalities
from replicant.otree.export import OTreeExporter

SERVER = "http://localhost:8000"
SESSION_CONFIG = "ertan2009_v2"
MODEL = "stepfun/step-3.5-flash"
MODE = "chat"
N = 50
REST_KEY = os.environ.get("OTREE_REST_KEY", "test-rest-key")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "papers", "ertan2009", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_condition(label: str, personalities: list[str]) -> dict:
    """Run N bots in one condition, save raw JSON, return summary."""
    start = time.time()
    print(f"\n{'='*65}", flush=True)
    print(f"CONDITION: {label} (N={len(personalities)})", flush=True)
    print(f"{'='*65}\n", flush=True)

    urls = OTreeClient.create_session(SERVER, SESSION_CONFIG, N, REST_KEY)
    session_code = urls[0].rsplit("/", 1)[-1][:8]  # for logging only
    print(f"Created session with {len(urls)} participant URLs\n", flush=True)

    names = [f"{label}_{i+1}" for i in range(N)]
    results = run_bots(
        server_url=SERVER,
        participant_urls=urls,
        personalities=personalities,
        model=MODEL,
        api_key=os.environ["OPEN_ROUTER_API_KEY"],
        names=names,
        verbose=True,
        mode=MODE,
    )

    elapsed = time.time() - start
    print(f"\nCondition '{label}' took {elapsed/60:.1f} min", flush=True)

    # Save raw JSON
    raw_path = os.path.join(RESULTS_DIR, f"{label}_raw.json")
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved raw data to {raw_path}", flush=True)

    # Quick summary
    n_complete = 0
    n_field_failures = 0
    for r in results:
        decisions = r.get("decisions", [])
        pages_ok = [d for d in decisions if "answers" in d]
        errs = [d for d in decisions if "error" in d]
        if len(pages_ok) == 4:  # Baseline, Voting, Regimes, Punishment
            n_complete += 1
        n_field_failures += len([d for d in errs if "field" in d.get("error", "")])

    print(f"Completion: {n_complete}/{N} fully completed, "
          f"{n_field_failures} field failures", flush=True)

    return {
        "label": label,
        "n": N,
        "n_complete": n_complete,
        "n_field_failures": n_field_failures,
        "elapsed_minutes": round(elapsed / 60, 2),
    }


def main():
    api_key = os.environ.get("OPEN_ROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPEN_ROUTER_API_KEY not set. Source .env first.")
        sys.exit(1)

    print(f"\n{'='*65}")
    print("ERTAN 2009 V2 — EXACT REPLICATION")
    print(f"{'='*65}")
    print(f"  App:     {SESSION_CONFIG} (group=4, endowment=10, ratio=1:4)")
    print(f"  Model:   {MODEL}")
    print(f"  Mode:    {MODE}")
    print(f"  N:       {N} per condition")
    print(f"  Output:  {RESULTS_DIR}/")

    summaries = []

    # Condition 1: Default (no personality)
    default_personalities = [""] * N
    summaries.append(run_condition("default", default_personalities))

    # Condition 2: Random Big Five from population norms
    random_personalities = sample_personalities(n=N, seed=42)
    summaries.append(run_condition("random", random_personalities))

    # Final summary
    print(f"\n{'='*65}")
    print("FINAL SUMMARY")
    print(f"{'='*65}")
    for s in summaries:
        print(f"  {s['label']:<20s} complete={s['n_complete']}/{s['n']}  "
              f"failures={s['n_field_failures']}  time={s['elapsed_minutes']}min")

    with open(os.path.join(RESULTS_DIR, "run_summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    # Also export oTree CSV for the full session data
    print(f"\nExporting oTree CSV...")
    exporter = OTreeExporter(SERVER, rest_key=REST_KEY)
    csv_path = exporter.export_app(SESSION_CONFIG, output_dir=RESULTS_DIR)
    print(f"Saved to {csv_path}")


if __name__ == "__main__":
    main()
