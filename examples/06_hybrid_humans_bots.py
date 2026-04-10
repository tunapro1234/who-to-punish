"""
Example 6: Hybrid oTree session — mix LLM bots with human participants.

Run from project root:
    python examples/06_hybrid_humans_bots.py

It will:
  1. Create an oTree session with 6 participants for "ertan2009"
  2. Print the first 3 URLs (give these to humans)
  3. Run the last 3 URLs as LLM bots
  4. Wait for everyone (humans + bots) to finish

The bots automatically wait at WaitPages until human participants
have submitted their decisions, just like real human participants would.
"""

from replicant.otree import HybridSession


def main():
    # Configuration
    SERVER = "http://localhost:8000"
    SESSION_CONFIG = "ertan2009"
    N_TOTAL = 6
    N_HUMANS = 3
    MODEL = "stepfun/step-3.5-flash"

    session = HybridSession(server_url=SERVER, model=MODEL)

    # 1. Create session
    print(f"Creating oTree session: {SESSION_CONFIG} (N={N_TOTAL})...")
    urls = session.create(SESSION_CONFIG, n_participants=N_TOTAL)
    print(f"Session code: {session.session_code}\n")

    # 2. Split into human and bot URLs
    human_urls, bot_urls = session.split(n_humans=N_HUMANS)

    # 3. Show human links
    session.print_human_links(human_urls)

    # 4. Confirm before starting bots
    input("Press ENTER to start the bots (humans should already have their links)...")

    # 5. Run bots with random Big Five personalities
    results = session.run_bots(
        bot_urls,
        big5=True,
        seed=42,
    )

    # 6. Print bot results
    print("\n" + "=" * 60)
    print("BOT RESULTS")
    print("=" * 60)
    for r in results:
        print(f"\n=== {r['agent']} ===")
        if 'error' in r:
            print(f"  ERROR: {r['error'][:200]}")
        else:
            for d in r['decisions']:
                if 'error' in d:
                    print(f"  {d['page']}: ERROR - {d['error'][:100]}")
                else:
                    print(f"  {d['page']}: {d['answers']}")

    # 7. Optional: wait for humans to finish
    print("\nWaiting for humans to complete the experiment...")
    try:
        session.wait_for_humans(human_urls, poll_interval=10)
        print("All humans finished!")
    except TimeoutError:
        print("Timeout waiting for humans (still running?)")

    print(f"\nSession {session.session_code} complete.")
    print(f"View results at: {SERVER}/SessionData/{session.session_code}")


if __name__ == "__main__":
    main()
