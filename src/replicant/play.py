"""
Interactive runner — connect an LLM agent to an oTree experiment.

Usage:
    python -m replicant.play

Paste a participant URL, pick a personality mode, and watch it play.
One terminal = one participant. Open multiple terminals for multiple agents.
"""

import os
import sys
import json


def _get_api_key():
    key = os.environ.get("OPEN_ROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if key:
        masked = key[:8] + "..." + key[-4:]
        print(f"  API key: {masked} (from environment)")
        return key

    print()
    print("  No OPEN_ROUTER_API_KEY found in environment.")
    print("  Get one at: https://openrouter.ai/settings/keys")
    print()
    key = input("  Paste your API key: ").strip()
    if not key:
        print("\n  Cannot run without an API key.")
        sys.exit(1)
    return key


def _get_urls():
    print()
    print("  Paste participant URL(s), one per line.")
    print("  These look like: http://host:8000/InitializeParticipant/abc123")
    print("  Empty line when done.")
    print()

    urls = []
    while True:
        prompt = f"  url [{len(urls) + 1}]: " if not urls else f"  url [{len(urls) + 1}]: "
        try:
            line = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        if "/InitializeParticipant/" not in line and "/join/" not in line:
            if line.startswith("http"):
                print("    (warning: doesn't look like an oTree participant URL, using anyway)")
            else:
                print("    (skipping — not a URL)")
                continue
        urls.append(line)

    if not urls:
        print("  No URLs provided.")
        sys.exit(1)

    return urls


def _get_personality_mode(n):
    print()
    print("  Personality mode:")
    print("    [1] None — default LLM, no personality prompt")
    print("    [2] Random — sampled from US adult Big Five norms")
    print("    [3] Manual — enter OCEAN scores (1-5 scale)")
    print("    [4] Skewed — random population with overridden trait means")
    print()

    choice = input("  Choice [1]: ").strip() or "1"

    if choice == "1":
        return [""] * n, "default"

    if choice == "2":
        seed_str = input("  Random seed [42]: ").strip() or "42"
        seed = int(seed_str)
        from .personalities.factory import sample_personalities
        personalities = sample_personalities(n=n, seed=seed)
        return personalities, f"random (seed={seed})"

    if choice == "3":
        print()
        print("  Enter OCEAN scores (1.0 = very low, 3.0 = average, 5.0 = very high)")
        print("  Press Enter for average (3.0)")
        print()
        e = input("    Extraversion [3.0]: ").strip() or "3.0"
        a = input("    Agreeableness [3.0]: ").strip() or "3.0"
        c = input("    Conscientiousness [3.0]: ").strip() or "3.0"
        ne = input("    Neuroticism [3.0]: ").strip() or "3.0"
        o = input("    Openness [3.0]: ").strip() or "3.0"
        from .personalities.factory import build_personality
        desc = build_personality(
            extraversion=float(e),
            agreeableness=float(a),
            conscientiousness=float(c),
            neuroticism=float(ne),
            openness=float(o),
        )
        label = f"E={e} A={a} C={c} N={ne} O={o}"
        return [desc] * n, f"manual ({label})"

    if choice == "4":
        print()
        print("  Override trait means (1-5 scale). Enter to keep population default.")
        print()
        from .personalities.factory import sample_personalities, POPULATION_NORMS
        kwargs = {}
        for trait, norms in POPULATION_NORMS.items():
            val = input(f"    {trait} [pop. mean {norms['mean']}]: ").strip()
            if val:
                kwargs[trait] = float(val)
        seed_str = input("  Random seed [42]: ").strip() or "42"
        seed = int(seed_str)
        personalities = sample_personalities(n=n, seed=seed, **kwargs)
        skew_str = ", ".join(f"{k}={v}" for k, v in kwargs.items()) or "none"
        return personalities, f"skewed ({skew_str}, seed={seed})"

    print(f"  Unknown choice '{choice}', using default.")
    return [""] * n, "default"


def _get_model():
    print()
    print("  Model options:")
    print("    stepfun/step-3.5-flash      — reasoning, $0.10/$0.30 per M tokens")
    print("    deepseek/deepseek-v3.2      — fast, $0.26/$0.38 per M tokens")
    print("    qwen/qwen3.6-plus:free      — free (rate-limited)")
    print()
    model = input("  Model [stepfun/step-3.5-flash]: ").strip()
    return model or "stepfun/step-3.5-flash"


def _get_mode():
    print()
    print("  Bot mode:")
    print("    [1] Stateless     — fresh context each page (cheapest, no memory)")
    print("    [2] Conversation  — persistent chat (remembers previous rounds)")
    print("    [3] Agent         — OpenClaw agent (self-learning, requires openclaw)")
    print()
    choice = input("  Mode [2]: ").strip() or "2"
    modes = {"1": "stateless", "2": "conversation", "3": "agent"}
    mode = modes.get(choice, "conversation")
    return mode


def _print_results(results):
    print()
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)

    for r in results:
        name = r.get("agent", "?")
        if "error" in r:
            print(f"\n  {name}: ERROR — {r['error']}")
            continue

        decisions = r.get("decisions", [])
        errors = [d for d in decisions if "error" in d]
        pages = [d for d in decisions if "answers" in d]
        print(f"\n  {name}: {len(pages)} pages completed", end="")
        if errors:
            print(f", {len(errors)} errors", end="")
        print()

        for d in pages:
            page = d.get("page", "?")
            answers = d.get("answers", {})
            answers_str = ", ".join(f"{k}={v}" for k, v in answers.items())
            print(f"    {page}: {answers_str}")

        for d in errors:
            print(f"    ERROR: {d.get('error', '?')[:80]}")


def main():
    print()
    print("=" * 60)
    print("  pyreplicant — LLM agent for oTree experiments")
    print("=" * 60)

    # API key
    api_key = _get_api_key()

    # URLs
    urls = _get_urls()
    n = len(urls)
    print(f"\n  {n} participant(s) registered.")

    # Server URL (extract from first participant URL)
    from urllib.parse import urlparse
    parsed = urlparse(urls[0])
    server_url = f"{parsed.scheme}://{parsed.netloc}"

    # Personality
    personalities, personality_label = _get_personality_mode(n)

    # Model
    model = _get_model()

    # Mode
    mode = _get_mode()

    # Confirm
    print()
    print("-" * 60)
    print(f"  Participants:  {n}")
    print(f"  Server:        {server_url}")
    print(f"  Model:         {model}")
    print(f"  Personality:   {personality_label}")
    print(f"  Mode:          {mode}")
    print("-" * 60)
    print()

    go = input("  Start? [Y/n]: ").strip().lower()
    if go and go not in ("y", "yes"):
        print("  Aborted.")
        sys.exit(0)

    # Show personality for manual mode
    if personality_label.startswith("manual") and n == 1:
        print()
        print("  Personality description:")
        for line in personalities[0].split("\n"):
            print(f"    {line}")

    # Run
    print()
    print("=" * 60)
    print("  RUNNING")
    print("=" * 60)
    print()

    from .otree.bot import run_bots

    names = [f"bot_{i+1}" for i in range(n)]
    results = run_bots(
        server_url=server_url,
        participant_urls=urls,
        personalities=personalities,
        model=model,
        api_key=api_key,
        names=names,
        verbose=True,
        mode=mode,
    )

    _print_results(results)

    # Save
    print()
    save = input("  Save results to JSON? [y/N]: ").strip().lower()
    if save in ("y", "yes"):
        path = input("  Filename [results.json]: ").strip() or "results.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Saved to {path}")

    print()


if __name__ == "__main__":
    main()
