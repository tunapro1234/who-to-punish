"""
oTree integration — run oTree experiments with LLM agents.

Two modes:

  1. Server mode (primary): LLM bots connect to a running oTree server
     as participants. oTree handles rounds, groups, payoffs. Supports
     human-AI mixing — some participants are humans, some are LLMs.

  2. Standalone mode: Parse oTree source code and translate to EDSL
     surveys. No server needed, good for quick prototyping.

Server mode usage:
    from src.framework.otree import OTreeSession

    session = OTreeSession("http://localhost:8000", model="gpt-4o")
    results = session.run("public_goods", n_bots=6)

    # Or with human-AI mixing:
    urls = OTreeClient.create_session("http://localhost:8000", "public_goods", 6)
    human_urls = urls[:3]   # give these to humans
    ai_urls = urls[3:]      # run bots on these
    results = session.run_bots(ai_urls)

Standalone mode usage:
    from src.framework.otree import OTreeExperiment

    exp = OTreeExperiment("path/to/public_goods_simple")
    exp.describe()
    results = exp.run(n_agents=20, model="qwen/qwen3.6-plus:free")
"""

from .parser import parse, OTreeApp, FieldDef, PageDef
from .translator import translate
from .client import OTreeClient, PageData, FormField
from .bot import LLMBot, run_bots

from ..experiment import BehavioralExperiment
from ..personalities import PersonalityFactory

from edsl import Agent, AgentList, Model


# ── Server mode ─────────────────────────────────────────────────────

class OTreeSession:
    """
    Connect LLM agents to a running oTree server.

    The oTree server runs normally — same experiment, same web interface.
    LLM bots join as participants, read each page, make decisions via LLM,
    and submit. Humans can participate alongside bots in the same session.
    """

    def __init__(self, server_url: str, model: str = "qwen/qwen3.6-plus:free"):
        self.server_url = server_url.rstrip('/')
        self.model = _resolve_model(model)

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
                 big5: bool = False, agents: AgentList | None = None) -> list[dict]:
        """
        Run LLM bots for existing participant URLs.

        Use this for human-AI mixing: create a session with N participants,
        give some URLs to humans, and pass the rest here.
        """
        n = len(participant_urls)

        if agents is None:
            factory = PersonalityFactory()
            if big5:
                agents = factory.create_population(per_profile=max(1, n // 5))
                # Trim or pad to match URL count
                agents = AgentList(list(agents)[:n])
            else:
                agents = factory.create_default(n=n)

        print(f"\n{'='*55}", flush=True)
        print(f"oTree Session | {self.model.model} | {n} bots", flush=True)
        print(f"Server: {self.server_url}", flush=True)
        print(f"{'='*55}\n", flush=True)

        return run_bots(self.server_url, participant_urls, agents, self.model)


# ── Standalone mode ─────────────────────────────────────────────────

class OTreeExperiment:
    """
    Parse oTree source code and run with EDSL (no server needed).

    Good for quick prototyping or when you don't want to run an oTree server.
    Parses the oTree app's __init__.py, translates fields to EDSL questions,
    and runs them through the BehavioralExperiment runner.
    """

    def __init__(self, app_path: str, model: str = "qwen/qwen3.6-plus:free"):
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


# ── Helpers ─────────────────────────────────────────────────────────

def _resolve_model(model):
    if isinstance(model, str):
        return Model(model, service_name="open_router", max_tokens=100000)
    return model
