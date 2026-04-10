"""
Hybrid sessions: mix LLM bots and human participants in the same oTree experiment.

The oTree server creates N participant URLs. You assign some to humans
(send them via email/link/QR) and some to LLM bots (run via this module).
oTree handles synchronization automatically — WaitPages ensure all group
members (human and bot) advance together before the next round.

Usage:
    from replicant.otree.hybrid import HybridSession

    session = HybridSession(
        server_url="http://localhost:8000",
        model="stepfun/step-3.5-flash",
    )

    # Create session with 6 participants
    urls = session.create("ertan2009", n_participants=6)

    # First 3 URLs go to humans (print or save them)
    print("Send these to humans:")
    for url in urls[:3]:
        print(f"  {url}")

    # Last 3 URLs run as LLM bots
    results = session.run_bots(urls[3:], big5=True)

The bots will wait at WaitPages until humans submit their decisions,
just like additional human participants would.
"""

import os
import time
import random
import requests
import httpx

from .client import OTreeClient
from .bot import run_bots
from ..personalities import POPULATION_NORMS, build_description
from ..personalities.factory import _score_to_weight


class HybridSession:
    """
    Run an oTree experiment with a mix of LLM bots and human participants.
    """

    def __init__(self, server_url: str, model: str = "stepfun/step-3.5-flash",
                 api_key: str = None, rest_key: str = "test-rest-key"):
        self.server_url = server_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("OPEN_ROUTER_API_KEY", "")
        self.rest_key = rest_key
        self.session_code = None
        self._urls = []

    def create(self, session_config: str, n_participants: int) -> list[str]:
        """
        Create an oTree session and return participant start URLs.

        Args:
            session_config: oTree session config name (e.g. "ertan2009")
            n_participants: total number of participants (humans + bots)

        Returns:
            list of participant start URLs
        """
        # 1. Create the session via REST API
        r = requests.post(
            f"{self.server_url}/api/sessions/",
            json={
                "session_config_name": session_config,
                "num_participants": n_participants,
            },
            headers={"otree-rest-key": self.rest_key},
        )
        r.raise_for_status()
        session = r.json()
        self.session_code = session["code"]
        join_url = session["session_wide_url"]

        # 2. Get participant URLs by joining
        urls = []
        for _ in range(n_participants):
            resp = httpx.get(join_url, follow_redirects=True)
            urls.append(str(resp.url))

        self._urls = urls
        return urls

    def split(self, n_humans: int) -> tuple[list[str], list[str]]:
        """
        Split participant URLs into human and bot groups.

        Returns:
            (human_urls, bot_urls)
        """
        if not self._urls:
            raise ValueError("No session created yet. Call create() first.")
        return self._urls[:n_humans], self._urls[n_humans:]

    def print_human_links(self, urls: list[str]):
        """Print human participant URLs in a readable format."""
        print("\n" + "=" * 60, flush=True)
        print(f"HUMAN PARTICIPANT LINKS ({len(urls)})", flush=True)
        print("=" * 60, flush=True)
        for i, url in enumerate(urls, 1):
            print(f"  Participant {i}: {url}", flush=True)
        print("=" * 60 + "\n", flush=True)
        print(f"Send these links to your human participants.", flush=True)
        print(f"Bots will wait at WaitPages until all humans submit.", flush=True)

    def run_bots(self, urls: list[str], personalities: list[str] = None,
                 big5: bool = False, names: list[str] = None,
                 seed: int = None) -> list[dict]:
        """
        Run LLM bots on the given participant URLs.

        Args:
            urls: list of participant start URLs to run as bots
            personalities: optional list of personality descriptions (one per bot)
            big5: if True and personalities is None, sample Big Five from population norms
            names: optional list of bot names
            seed: random seed for personality sampling

        Returns:
            list of result dicts, one per bot
        """
        n = len(urls)

        if personalities is None:
            if big5:
                rng = random.Random(seed)
                personalities = []
                for i in range(n):
                    weights = {}
                    for domain, norms in POPULATION_NORMS.items():
                        score = rng.gauss(norms["mean"], norms["sd"])
                        score = max(1.0, min(5.0, score))
                        weights[domain] = _score_to_weight(score)
                    personalities.append(build_description(weights))
            else:
                personalities = [""] * n

        names = names or [f"bot_{i+1}" for i in range(n)]

        print(f"\nStarting {n} bots on {self.server_url}...", flush=True)
        print(f"Model: {self.model}", flush=True)
        print(f"(Bots will wait at WaitPages for human participants)\n", flush=True)

        return run_bots(self.server_url, urls, personalities, self.model,
                        self.api_key, names)

    def wait_for_humans(self, urls: list[str], poll_interval: int = 5,
                        timeout: int = 1800):
        """
        Block until all human participants have completed the experiment.
        Useful when bots finish first and you want to wait for humans
        before collecting results.
        """
        deadline = time.time() + timeout
        client = OTreeClient(self.server_url)

        while time.time() < deadline:
            done = 0
            for url in urls:
                page = client.get_page(url)
                if page.is_finished:
                    done += 1
            print(f"  Humans completed: {done}/{len(urls)}", flush=True)
            if done == len(urls):
                return
            time.sleep(poll_interval)

        raise TimeoutError(f"Humans did not finish within {timeout}s")
