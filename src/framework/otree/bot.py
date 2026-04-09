"""
LLM-powered bot that plays through oTree experiments.

Connects to a running oTree server as a participant, reads each page,
uses an LLM (via EDSL) to make decisions, and submits responses. From
oTree's perspective, the bot is indistinguishable from a human in a browser.

Multiple bots run concurrently via threads, so group games with WaitPages
work naturally — all group members advance together.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from edsl import Agent, AgentList, Model, Survey
from edsl import QuestionFreeText, QuestionMultipleChoice, QuestionNumerical

from .client import OTreeClient, PageData, FormField


class LLMBot:
    """
    An LLM agent that plays through an oTree experiment.

    Each bot has its own HTTP client (for thread safety) and EDSL Agent
    (for personality). It fetches pages, translates form fields into EDSL
    questions, gets LLM decisions, and submits them.
    """

    def __init__(self, server_url: str, participant_url: str,
                 agent: Agent, model: Model):
        self.client = OTreeClient(server_url)
        self.participant_url = participant_url
        self.agent = agent
        self.model = model
        self.log: list[dict] = []

    def play(self) -> list[dict]:
        """
        Play through the entire experiment.

        Returns a list of decision records:
            [{'page': str, 'answers': dict}, ...]
        """
        page = self.client.get_page(self.participant_url)

        while not page.is_finished:
            if page.is_wait_page:
                page = self.client.wait_for_page(page)
                continue

            if page.form_fields:
                answers = self._decide(page)
                self.log.append({
                    'page': page.title or page.url,
                    'answers': answers,
                })
                page = self.client.submit(page, answers)
            else:
                # Info-only page — advance
                page = self.client.submit(page, {})

        return self.log

    def _decide(self, page: PageData) -> dict:
        """Ask the LLM to fill in the page's form fields."""
        questions = []
        for field in page.form_fields:
            q = _field_to_question(field, page.body_text)
            if q:
                questions.append(q)

        if not questions:
            return {}

        survey = Survey(questions=questions)
        result = survey.by(AgentList([self.agent])).by(self.model).run()
        df = result.to_pandas()

        answers = {}
        for field in page.form_fields:
            col = f'answer.{field.name}'
            if col not in df.columns:
                continue
            val = df[col].iloc[0]
            # Map display text back to form values for choice fields
            if field.choices:
                val = _display_to_value(val, field.choices)
            answers[field.name] = val

        return answers


# ── Batch runner ────────────────────────────────────────────────────

def run_bots(server_url: str, participant_urls: list[str],
             agents: AgentList, model: Model) -> list[dict]:
    """
    Run LLM bots concurrently for a list of participant URLs.

    Each bot gets its own thread so WaitPages resolve naturally
    (all group members advance together).

    Returns a list of result dicts, one per bot:
        [{'agent': str, 'url': str, 'decisions': [...]}, ...]
    """
    if len(participant_urls) != len(agents):
        raise ValueError(
            f"Got {len(participant_urls)} URLs but {len(agents)} agents"
        )

    bots = [
        LLMBot(server_url, url, agent, model)
        for url, agent in zip(participant_urls, agents)
    ]

    results = [None] * len(bots)

    with ThreadPoolExecutor(max_workers=len(bots)) as pool:
        futures = {pool.submit(bot.play): i for i, bot in enumerate(bots)}
        for future in as_completed(futures):
            idx = futures[future]
            bot = bots[idx]
            try:
                decisions = future.result()
                results[idx] = {
                    'agent': bot.agent.name,
                    'url': bot.participant_url,
                    'decisions': decisions,
                }
            except Exception as e:
                results[idx] = {
                    'agent': bot.agent.name,
                    'url': bot.participant_url,
                    'error': str(e),
                }

    return results


# ── Field → Question mapping ───────────────────────────────────────

def _field_to_question(field: FormField, context: str):
    """Map an HTML form field to an EDSL Question."""
    label = field.label or field.name.replace('_', ' ').capitalize()
    text = f"{context}\n\n{label}" if context else label

    if field.choices:
        options = [display for _, display in field.choices]
        return QuestionMultipleChoice(
            question_name=field.name,
            question_text=text,
            question_options=options,
        )

    if field.input_type == 'number':
        kwargs = dict(question_name=field.name, question_text=text)
        try:
            if field.min_value is not None:
                kwargs['min_value'] = float(field.min_value)
        except ValueError:
            pass
        try:
            if field.max_value is not None:
                kwargs['max_value'] = float(field.max_value)
        except ValueError:
            pass
        return QuestionNumerical(**kwargs)

    return QuestionFreeText(question_name=field.name, question_text=text)


def _display_to_value(answer: str, choices: list[tuple[str, str]]) -> str:
    """Map EDSL answer (display text) back to the oTree form value."""
    answer_str = str(answer)
    for value, display in choices:
        if display == answer_str:
            return value
    # Fallback: case-insensitive
    for value, display in choices:
        if display.lower() == answer_str.lower():
            return value
    return answer_str
