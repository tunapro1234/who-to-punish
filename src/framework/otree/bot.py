"""
LLM-powered bot that plays through oTree experiments.

Architecture:
  oTree HTML → Parser (client.py) → FormController → LLM → Validator → oTree POST

The FormController sits between the LLM and oTree, ensuring:
  1. Every required field has a value before submission
  2. Values are within declared bounds (min/max, valid choices)
  3. Failed LLM responses trigger re-prompting (not empty submission)
  4. Max retry limit prevents infinite loops
  5. Validation errors from oTree are detected and re-prompted with feedback
"""

import os
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests as http_requests

from .client import OTreeClient, PageData, FormField


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_LLM_RETRIES = 8
MAX_PAGE_RETRIES = 3


# ═══════════════════════════════════════════════════════════════════
# Form Controller — validates LLM output before sending to oTree
# ═══════════════════════════════════════════════════════════════════

class FormController:
    """
    Validates and sanitizes LLM responses before oTree submission.
    Retries the LLM with feedback when validation fails.
    """

    @staticmethod
    def validate(answers: dict, fields: list[FormField]) -> tuple[dict, list[str]]:
        """
        Validate answers against field definitions.
        Returns (cleaned_answers, list_of_errors).
        """
        cleaned = {}
        errors = []

        for field in fields:
            value = answers.get(field.name)

            if value is None or value == "" or str(value) == "nan":
                errors.append(f"Missing required field: {field.name}")
                continue

            # Validate choice fields
            if field.choices:
                valid_values = [v for v, _ in field.choices]
                valid_displays = [d for _, d in field.choices]
                str_val = str(value)

                if str_val in valid_values:
                    cleaned[field.name] = str_val
                elif str_val in valid_displays:
                    # Map display text to value
                    for v, d in field.choices:
                        if d == str_val:
                            cleaned[field.name] = v
                            break
                else:
                    # Case-insensitive fallback
                    matched = False
                    for v, d in field.choices:
                        if d.lower() == str_val.lower() or v.lower() == str_val.lower():
                            cleaned[field.name] = v
                            matched = True
                            break
                    if not matched:
                        errors.append(
                            f"{field.name}: '{value}' is not a valid choice. "
                            f"Options: {valid_displays}"
                        )
                continue

            # Validate number fields
            if field.input_type == 'number':
                try:
                    num = float(value)
                    if num == int(num):
                        num = int(num)
                except (ValueError, TypeError):
                    errors.append(f"{field.name}: '{value}' is not a number")
                    continue

                if field.min_value is not None and num < float(field.min_value):
                    errors.append(
                        f"{field.name}: {num} is below minimum {field.min_value}"
                    )
                    continue
                if field.max_value is not None and num > float(field.max_value):
                    errors.append(
                        f"{field.name}: {num} is above maximum {field.max_value}"
                    )
                    continue
                cleaned[field.name] = num
                continue

            # Text fields — accept as-is
            cleaned[field.name] = str(value)

        return cleaned, errors

    @staticmethod
    def is_complete(answers: dict, fields: list[FormField]) -> bool:
        """Check if all required fields have values."""
        return all(field.name in answers for field in fields)


# ═══════════════════════════════════════════════════════════════════
# LLM Bot
# ═══════════════════════════════════════════════════════════════════

class LLMBot:
    """
    An LLM agent that plays through an oTree experiment.
    Uses FormController to validate responses before submission.
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
        self.controller = FormController()

    def play(self) -> list[dict]:
        """Play through the entire experiment."""
        page = self.client.get_page(self.participant_url)
        prev_url = None
        page_retry_count = 0

        while not page.is_finished:
            if page.is_wait_page:
                page = self.client.wait_for_page(page)
                prev_url = None
                page_retry_count = 0
                continue

            # Detect oTree validation error (same page returned after submit)
            if page.url == prev_url:
                page_retry_count += 1
                if page_retry_count > MAX_PAGE_RETRIES:
                    self._log_error(page, "Max page retries exceeded, skipping")
                    break
            else:
                page_retry_count = 0

            if page.form_fields:
                answers = self._decide_with_validation(page)
                if answers:
                    self.log.append({
                        'page': self._page_label(page),
                        'answers': answers,
                    })
                    prev_url = page.url
                    page = self.client.submit(page, answers)
                else:
                    self._log_error(page, "Could not produce valid answers")
                    break
            else:
                prev_url = page.url
                page = self.client.submit(page, {})

        return self.log

    def _decide_with_validation(self, page: PageData) -> dict:
        """
        Get LLM decision, validate it, retry with feedback if invalid.

        Strategy:
        - 3 or fewer fields: ask all at once (single LLM call)
        - 4+ fields: ask one at a time to avoid overwhelming the LLM
        """
        context = _escape_jinja(page.body_text)
        fields = page.form_fields

        if len(fields) <= 3:
            return self._decide_batch(context, fields)
        else:
            return self._decide_sequential(context, fields)

    def _decide_batch(self, context: str, fields: list[FormField]) -> dict:
        """Ask all fields at once. Used for simple pages (<=3 fields)."""
        feedback = ""
        for attempt in range(MAX_LLM_RETRIES):
            prompt = self._build_prompt(context, fields, feedback)
            response = self._call_llm(prompt["system"], prompt["user"])
            raw_answers = _parse_json_response(response)

            cleaned, errors = self.controller.validate(raw_answers, fields)
            if not errors and self.controller.is_complete(cleaned, fields):
                return cleaned

            feedback = self._build_feedback(errors, cleaned, fields)

        return {}

    def _decide_sequential(self, context: str, fields: list[FormField]) -> dict:
        """Ask one field at a time. Used for complex pages (4+ fields)."""
        all_answers = {}

        for field in fields:
            feedback = ""
            for attempt in range(MAX_LLM_RETRIES):
                prompt = self._build_prompt(
                    context, [field], feedback,
                    prior_answers=all_answers,
                )
                response = self._call_llm(prompt["system"], prompt["user"])
                raw = _parse_json_response(response)

                cleaned, errors = self.controller.validate(raw, [field])
                if not errors and field.name in cleaned:
                    all_answers[field.name] = cleaned[field.name]
                    break

                feedback = self._build_feedback(errors, cleaned, [field])

            if field.name not in all_answers:
                return {}  # failed on this field, abort

        return all_answers

    def _build_feedback(self, errors, cleaned, fields):
        """Build error feedback for LLM retry."""
        if errors:
            return (
                "Your previous response had errors:\n"
                + "\n".join(f"  - {e}" for e in errors)
                + "\nPlease fix these and try again."
            )
        if not cleaned:
            return (
                "Your response could not be parsed as JSON. "
                "Respond with ONLY a JSON object, no markdown or explanation."
            )
        missing = [f.name for f in fields if f.name not in cleaned]
        return f"Missing fields: {missing}. Include ALL fields."

    def _build_prompt(self, context: str, fields: list[FormField],
                      feedback: str = "", prior_answers: dict = None) -> dict:
        """Build system and user prompts for the LLM."""
        field_descriptions = []
        for field in fields:
            label = field.label or field.name.replace('_', ' ')
            if field.choices:
                options = ", ".join(f'"{d}"' for _, d in field.choices)
                field_descriptions.append(
                    f'- "{field.name}": {label} (choose exactly one of: {options})'
                )
            elif field.input_type == 'number':
                bounds = ""
                if field.min_value is not None and field.max_value is not None:
                    bounds = f" (integer from {field.min_value} to {field.max_value})"
                elif field.min_value is not None:
                    bounds = f" (min: {field.min_value})"
                elif field.max_value is not None:
                    bounds = f" (max: {field.max_value})"
                field_descriptions.append(
                    f'- "{field.name}": {label}{bounds}'
                )
            else:
                field_descriptions.append(f'- "{field.name}": {label}')

        fields_text = "\n".join(field_descriptions)
        example = ", ".join(f'"{f.name}": ...' for f in fields)

        system_prompt = (
            "You are a participant in an economics experiment. "
            "Your personality:\n" + self.personality + "\n\n"
            "CRITICAL INSTRUCTION: Respond with ONLY a valid JSON object. "
            "No explanation, no markdown code blocks, no text before or after. "
            "Just the raw JSON object."
        )

        user_prompt = f"Page context:\n{context}\n\n"
        if prior_answers:
            user_prompt += "Your previous answers on this page:\n"
            user_prompt += json.dumps(prior_answers, indent=2) + "\n\n"
        user_prompt += f"Field to answer:\n{fields_text}"
        if feedback:
            user_prompt += f"\n\n{feedback}"
        user_prompt += f"\n\nRespond: {{{example}}}"

        return {"system": system_prompt, "user": user_prompt}

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Direct OpenRouter API call with retry."""
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
        for attempt in range(3):
            try:
                r = http_requests.post(OPENROUTER_URL, headers=headers,
                                       json=payload, timeout=120)
                if r.status_code == 429:
                    time.sleep(3 * (attempt + 1))
                    continue
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception:
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return ""

    def _page_label(self, page: PageData) -> str:
        """Extract a readable page name from the URL."""
        parts = page.url.rstrip('/').split('/')
        for part in reversed(parts):
            if part and not part.isdigit():
                return part
        return page.title or page.url

    def _log_error(self, page: PageData, message: str):
        """Log an error for a page."""
        self.log.append({
            'page': self._page_label(page),
            'error': message,
        })


# ═══════════════════════════════════════════════════════════════════
# Batch runner
# ═══════════════════════════════════════════════════════════════════

def run_bots(server_url: str, participant_urls: list[str],
             personalities: list[str], model: str,
             api_key: str = None, names: list[str] = None) -> list[dict]:
    """
    Run LLM bots concurrently for a list of participant URLs.
    Each bot gets its own thread so WaitPages resolve naturally.
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


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown and noise."""
    if not text:
        return {}

    # Strip markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    # Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Find JSON object in text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except (json.JSONDecodeError, TypeError):
            pass

    return {}


def _escape_jinja(text: str) -> str:
    """Escape {{ }} and {% %} so they don't get interpreted as templates."""
    text = re.sub(r'\{\{', '{ {', text)
    text = re.sub(r'\}\}', '} }', text)
    text = re.sub(r'\{%', '{ %', text)
    text = re.sub(r'%\}', '% }', text)
    return text
