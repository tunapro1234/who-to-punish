"""
LLM-powered bot that plays through oTree experiments.

Connects to a running oTree server as a participant, reads each page,
uses an LLM to make decisions, and submits responses. From oTree's
perspective, the bot is indistinguishable from a human in a browser.

Uses direct OpenRouter API calls (not EDSL) to avoid asyncio event loop
conflicts when running multiple bots concurrently in threads.
"""

import os
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests as http_requests

from .client import OTreeClient, PageData, FormField


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMBot:
    """
    An LLM agent that plays through an oTree experiment.

    Each bot has its own HTTP client and calls the LLM API directly
    (no EDSL) to avoid async event loop issues in threaded execution.
    """

    def __init__(self, server_url: str, participant_url: str,
                 personality: str, model: str, api_key: str):
        self.client = OTreeClient(server_url)
        self.participant_url = participant_url
        self.personality = personality
        self.model = model
        self.api_key = api_key
        self.name = ""
        self.log: list[dict] = []

    def play(self) -> list[dict]:
        """Play through the entire experiment."""
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
                page = self.client.submit(page, {})

        return self.log

    def _decide(self, page: PageData) -> dict:
        """Ask the LLM to fill in the page's form fields."""
        context = _escape_jinja(page.body_text)

        if not page.form_fields:
            return {}

        # Build prompt describing the form
        field_descriptions = []
        for field in page.form_fields:
            label = field.label or field.name.replace('_', ' ')
            if field.choices:
                options = ", ".join(f'"{d}"' for _, d in field.choices)
                field_descriptions.append(
                    f'- {field.name}: {label} (choose one: {options})'
                )
            elif field.input_type == 'number':
                bounds = ""
                if field.min_value is not None and field.max_value is not None:
                    bounds = f" (range: {field.min_value} to {field.max_value})"
                elif field.min_value is not None:
                    bounds = f" (min: {field.min_value})"
                elif field.max_value is not None:
                    bounds = f" (max: {field.max_value})"
                field_descriptions.append(
                    f'- {field.name}: {label}{bounds} [number]'
                )
            else:
                field_descriptions.append(f'- {field.name}: {label} [text]')

        fields_text = "\n".join(field_descriptions)

        system_prompt = (
            "You are a participant in an economics experiment. "
            "Your personality description below reflects who you are. "
            "Let it naturally shape your decisions.\n\n"
            f"Your personality:\n{self.personality}\n\n"
            "Respond ONLY with a JSON object containing your answers. "
            "No explanation, no markdown, just JSON."
        )

        user_prompt = (
            f"Page context:\n{context}\n\n"
            f"Fields to fill:\n{fields_text}\n\n"
            f"Respond with a JSON object like: "
            "{{" + ", ".join(f'"{f.name}": ...' for f in page.form_fields) + "}}"
        )

        response = self._call_llm(system_prompt, user_prompt)
        answers = _parse_json_response(response, page.form_fields)
        return answers

    def _call_llm(self, system_prompt: str, user_prompt: str, retries: int = 3) -> str:
        """Direct OpenRouter API call."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": 4000,
            "temperature": 0.5,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        for attempt in range(retries):
            try:
                r = http_requests.post(OPENROUTER_URL, headers=headers,
                                       json=payload, timeout=120)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return "{}"
        return "{}"


# ── Batch runner ────────────────────────────────────────────────────

def run_bots(server_url: str, participant_urls: list[str],
             personalities: list[str], model: str,
             api_key: str = None, names: list[str] = None) -> list[dict]:
    """
    Run LLM bots concurrently for a list of participant URLs.

    Args:
        server_url: oTree server URL
        participant_urls: list of participant start URLs
        personalities: list of personality description strings
        model: OpenRouter model name
        api_key: OpenRouter API key (or from env)
        names: optional list of agent names

    Returns list of result dicts.
    """
    api_key = api_key or os.environ.get("OPEN_ROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("No API key. Set OPEN_ROUTER_API_KEY.")

    n = len(participant_urls)
    if len(personalities) != n:
        raise ValueError(f"Got {n} URLs but {len(personalities)} personalities")

    names = names or [f"bot_{i+1}" for i in range(n)]

    bots = []
    for i in range(n):
        bot = LLMBot(server_url, participant_urls[i], personalities[i], model, api_key)
        bot.name = names[i]
        bots.append(bot)

    results = [None] * n

    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = {pool.submit(bot.play): i for i, bot in enumerate(bots)}
        for future in as_completed(futures):
            idx = futures[future]
            bot = bots[idx]
            try:
                decisions = future.result()
                results[idx] = {
                    'agent': bot.name,
                    'url': bot.participant_url,
                    'decisions': decisions,
                }
            except Exception as e:
                results[idx] = {
                    'agent': bot.name,
                    'url': bot.participant_url,
                    'error': str(e),
                }

    return results


# ── Helpers ─────────────────────────────────────────────────────────

def _parse_json_response(text: str, fields: list[FormField]) -> dict:
    """Extract JSON answers from LLM response."""
    if not text:
        return {}
    # Try direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # Find JSON in text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def _escape_jinja(text: str) -> str:
    """Escape {{ }} and {% %} so EDSL doesn't interpret them as templates."""
    import re
    text = re.sub(r'\{\{', '{ {', text)
    text = re.sub(r'\}\}', '} }', text)
    text = re.sub(r'\{%', '{ %', text)
    text = re.sub(r'%\}', '% }', text)
    return text


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
