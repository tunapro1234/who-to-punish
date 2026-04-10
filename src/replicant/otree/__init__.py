"""
oTree integration — run LLM agents as participants on a real oTree server.

Two modes:

  1. Server mode (primary): LLM bots connect to a running oTree server
     as participants. oTree handles rounds, groups, payoffs. Supports
     human-AI mixing — some participants are humans, some are LLMs.

  2. Standalone mode: Parse oTree source code and translate to EDSL
     surveys. No server needed, good for quick prototyping.

Server mode usage:
    from replicant.otree import OTreeSession

    session = OTreeSession("http://localhost:8000", model="stepfun/step-3.5-flash")
    results = session.run("public_goods", n_bots=6)

Standalone mode usage:
    from replicant.otree import OTreeExperiment

    exp = OTreeExperiment("path/to/public_goods_simple")
    exp.describe()
    results = exp.run(n_agents=20)
"""

import os

from .parser import parse, OTreeApp, FieldDef, PageDef
from .translator import translate
from .client import OTreeClient, PageData, FormField
from .bot import LLMBot, FormController, run_bots
from .hybrid import HybridSession

from ..experiments import BehavioralExperiment
from ..personalities import PersonalityFactory, build_description, POPULATION_NORMS
from ..personalities.factory import _score_to_weight


# ── Server mode ─────────────────────────────────────────────────────

class OTreeSession:
    """
    Connect LLM agents to a running oTree server.

    The oTree server runs normally — same experiment, same web interface.
    LLM bots join as participants, read each page, make decisions via LLM,
    and submit. Humans can participate alongside bots in the same session.
    """

    def __init__(self, server_url: str, model: str = "stepfun/step-3.5-flash",
                 api_key: str = None):
        self.server_url = server_url.rstrip('/')
        self.model = model
        self.api_key = api_key or os.environ.get("OPEN_ROUTER_API_KEY", "")

    def run(self, session_config: str, n_bots: int = 6,
            big5: bool = False, rest_key: str | None = None) -> list[dict]:
        """
        Create an oTree session and run LLM bots through it.

        Args:
            session_config: Name of the oTree session config (e.g. "public_goods").
            n_bots: Number of LLM participants.
            big5: Use Big Five personality agents.
            rest_key: oTree REST API key (from settings.py OTREE_REST_KEY).

        Returns:
            List of result dicts, one per bot.
        """
        urls = OTreeClient.create_session(
            self.server_url, session_config, n_bots, rest_key,
        )
        return self.run_bots(urls, big5=big5)

    def run_bots(self, participant_urls: list[str],
                 big5: bool = False,
                 personalities: list[str] | None = None,
                 names: list[str] | None = None) -> list[dict]:
        """
        Run LLM bots for existing participant URLs.

        Use this for human-AI mixing: create a session with N participants,
        give some URLs to humans, and pass the rest here.
        """
        n = len(participant_urls)

        if personalities is None:
            factory = PersonalityFactory()
            if big5:
                # Sample from population norms
                import random
                rng = random.Random()
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

        print(f"\n{'='*55}", flush=True)
        print(f"oTree Session | {self.model} | {n} bots", flush=True)
        print(f"Server: {self.server_url}", flush=True)
        print(f"{'='*55}\n", flush=True)

        return run_bots(self.server_url, participant_urls, personalities,
                        self.model, self.api_key, names)


# ── Standalone mode ─────────────────────────────────────────────────

class OTreeExperiment:
    """
    Parse oTree source code and run with EDSL (no server needed).

    Good for quick prototyping or when you don't want to run an oTree server.
    Parses the oTree app's __init__.py, translates fields to EDSL questions,
    and runs them through the BehavioralExperiment runner.
    """

    def __init__(self, app_path: str, model: str = "stepfun/step-3.5-flash"):
        self.app = parse(app_path)
        self.survey = translate(self.app)
        self.model = model

    def run(self, n_agents: int = 10, big5: bool = False, agents=None):
        exp = BehavioralExperiment(self.app.name, model=self.model)
        exp.add_part("main", self.survey, description=self.app.doc or self.app.name)

        if agents is None:
            factory = PersonalityFactory()
            if big5:
                agents = factory.create_population(per_profile=max(1, n_agents // 5))
            else:
                agents = factory.create_default(n=n_agents)

        return exp.run(agents)

    def describe(self):
        app = self.app
        print(f"oTree App: {app.name}")
        if app.doc:
            print(f"  {app.doc.strip()[:80]}")
        print(f"  Players per group: {app.players_per_group or 'individual'}")
        print(f"  Rounds: {app.num_rounds}")

        if app.constants:
            print(f"\n  Constants:")
            for k, v in app.constants.items():
                print(f"    {k} = {v}")

        if app.player_fields:
            print(f"\n  Player fields:")
            for f in app.player_fields:
                label = f" — {f.label}" if f.label else ""
                print(f"    {f.name}: {f.field_type}{label}")

        if app.group_fields:
            print(f"\n  Group fields:")
            for f in app.group_fields:
                label = f" — {f.label}" if f.label else ""
                print(f"    {f.name}: {f.field_type}{label}")

        pages = app.decision_pages()
        if pages:
            print(f"\n  Decision pages ({len(pages)}):")
            for p in pages:
                fields = ', '.join(p.form_fields)
                print(f"    {p.name} → [{fields}]")

        print(f"\n  Page sequence: {' → '.join(app.page_sequence)}")
        print(f"\n  EDSL survey: {len(self.survey.questions)} question(s)")


__all__ = [
    "OTreeSession", "OTreeExperiment", "HybridSession",
    "parse", "translate",
    "OTreeApp", "FieldDef", "PageDef",
    "OTreeClient", "PageData", "FormField",
    "LLMBot", "FormController", "run_bots",
]
