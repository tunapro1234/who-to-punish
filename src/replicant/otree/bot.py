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

    Maintains a persistent conversation with the LLM across all pages,
    so the agent remembers its previous decisions and outcomes (e.g.
    results pages between rounds). This enables multi-round games where
    the agent can learn and adapt over time.
    """

    def __init__(self, server_url: str, participant_url: str,
                 personality: str, model: str, api_key: str,
                 verbose: bool = False):
        self.client = OTreeClient(server_url)
        self.participant_url = participant_url
        self.personality = personality
        self.model = model
        self.api_key = api_key
        self.name = ""
        self.log: list[dict] = []
        self.controller = FormController()
        self.verbose = verbose
        self.page_count = 0

        # Persistent conversation — the LLM sees its full history
        system_parts = [
            "You are a participant in an economics experiment.",
        ]
        if personality:
            system_parts.append(f"Your personality:\n{personality}")
        system_parts.append(
            "\nCRITICAL INSTRUCTION: When asked to make a decision, "
            "respond with ONLY a valid JSON object. "
            "No explanation, no markdown code blocks, no text before or after. "
            "Just the raw JSON object."
        )
        self.messages = [
            {"role": "system", "content": "\n".join(system_parts)},
        ]

    def _print(self, msg: str):
        if self.verbose:
            prefix = f"  [{self.name}]" if self.name else "  [bot]"
            print(f"{prefix} {msg}", flush=True)

    def play(self) -> list[dict]:
        """Play through the entire experiment."""
        self._print("connecting...")
        page = self.client.get_page(self.participant_url)
        prev_url = None
        page_retry_count = 0

        while not page.is_finished:
            if page.is_wait_page:
                self._print("waiting for other participants...")
                page = self.client.wait_for_page(page)
                prev_url = None
                page_retry_count = 0
                continue

            self.page_count += 1
            page_name = self._page_label(page)

            # Detect oTree validation error (same page returned after submit)
            if page.url == prev_url:
                page_retry_count += 1
                if page_retry_count > MAX_PAGE_RETRIES:
                    self._print(f"page {page_name}: FAILED (max retries)")
                    self._log_error(page, "Max page retries exceeded, skipping")
                    break
            else:
                page_retry_count = 0

            if page.form_fields:
                self._print(f"page {self.page_count}: {page_name} ({len(page.form_fields)} fields)")
                answers = self._decide_with_validation(page)
                if answers:
                    self.log.append({
                        'page': page_name,
                        'answers': answers,
                    })
                    for k, v in answers.items():
                        self._print(f"  {k} = {v}")
                    prev_url = page.url
                    page = self.client.submit(page, answers)
                else:
                    self._print(f"page {self.page_count}: {page_name}: FAILED (no valid answers)")
                    self._log_error(page, "Could not produce valid answers")
                    break
            else:
                # Info/results page — add to conversation as context
                context = _escape_jinja(page.body_text)
                if context.strip():
                    self._print(f"page {self.page_count}: {page_name} (info)")
                    self.messages.append({
                        "role": "user",
                        "content": f"[Experiment info — {page_name}]\n{context}",
                    })
                    self.messages.append({
                        "role": "assistant",
                        "content": "(noted)",
                    })
                else:
                    self._print(f"page {self.page_count}: {page_name} (no fields, advancing)")
                prev_url = page.url
                page = self.client.submit(page, {})

        if page.is_finished:
            self._print(f"done ({self.page_count} pages)")

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
        user_msg = self._build_user_prompt(context, fields)

        for attempt in range(MAX_LLM_RETRIES):
            if attempt > 0:
                self._print(f"  retry {attempt}/{MAX_LLM_RETRIES}...")
            response = self._call_llm(user_msg)
            raw_answers = _parse_json_response(response)

            cleaned, errors = self.controller.validate(raw_answers, fields)
            if not errors and self.controller.is_complete(cleaned, fields):
                return cleaned

            if errors and self.verbose:
                for e in errors:
                    self._print(f"  validation: {e}")

            # Retry: add feedback as next message in the conversation
            user_msg = self._build_feedback(errors, cleaned, fields)

        return {}

    def _decide_sequential(self, context: str, fields: list[FormField]) -> dict:
        """Ask one field at a time. Used for complex pages (4+ fields)."""
        all_answers = {}

        for fi, field in enumerate(fields):
            self._print(f"  field {fi+1}/{len(fields)}: {field.name}...")
            user_msg = self._build_user_prompt(
                context, [field], prior_answers=all_answers,
            )

            for attempt in range(MAX_LLM_RETRIES):
                if attempt > 0:
                    self._print(f"    retry {attempt}/{MAX_LLM_RETRIES}...")
                response = self._call_llm(user_msg)
                raw = _parse_json_response(response)

                # If the LLM returned a value but under a different key,
                # try to salvage it
                if field.name not in raw and len(raw) == 1:
                    only_key = list(raw.keys())[0]
                    raw[field.name] = raw.pop(only_key)

                # If the LLM returned the value directly (not in JSON),
                # try to parse it as the field value
                if not raw and response.strip():
                    raw = _try_direct_value(response.strip(), field)

                cleaned, errors = self.controller.validate(raw, [field])
                if not errors and field.name in cleaned:
                    all_answers[field.name] = cleaned[field.name]
                    break

                # Retry: feedback becomes the next message
                user_msg = self._build_feedback(errors, cleaned, [field])

            if field.name not in all_answers:
                self.log.append({
                    'page': 'FIELD_FAILURE',
                    'error': f"Failed on field '{field.name}' after {MAX_LLM_RETRIES} attempts. "
                             f"Last response: {response[:200] if response else 'empty'}",
                })
                return {}

        return all_answers

    def _build_feedback(self, errors, cleaned, fields):
        """Build error feedback for LLM retry."""
        if errors:
            return (
                "Your previous response had errors:\n"
                + "\n".join(f"  - {e}" for e in errors)
                + "\nPlease fix these and try again. "
                "Respond with ONLY a JSON object."
            )
        if not cleaned:
            return (
                "Your response could not be parsed as JSON. "
                "Respond with ONLY a JSON object, no markdown or explanation."
            )
        missing = [f.name for f in fields if f.name not in cleaned]
        return f"Missing fields: {missing}. Include ALL fields. JSON only."

    def _build_user_prompt(self, context: str, fields: list[FormField],
                           prior_answers: dict = None) -> str:
        """Build the user message for a decision."""
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

        prompt = f"Page context:\n{context}\n\n"
        if prior_answers:
            prompt += "Your previous answers on this page:\n"
            prompt += json.dumps(prior_answers, indent=2) + "\n\n"
        prompt += f"Decision:\n{fields_text}"
        prompt += f"\n\nRespond: {{{example}}}"

        return prompt

    def _call_llm(self, user_prompt: str) -> str:
        """
        Send a message in the persistent conversation and get a response.
        The full conversation history is sent each time, giving the LLM
        memory of all previous pages, decisions, and outcomes.
        """
        self.messages.append({"role": "user", "content": user_prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": 16000,
            "temperature": 0.5,
            "messages": self.messages,
        }

        content = ""
        for attempt in range(3):
            try:
                r = http_requests.post(OPENROUTER_URL, headers=headers,
                                       json=payload, timeout=120)
                if r.status_code == 429:
                    time.sleep(3 * (attempt + 1))
                    continue
                r.raise_for_status()
                content = r.json()["choices"][0]["message"].get("content", "")
                if content:
                    break
                # Reasoning model might put answer in reasoning field
                # with empty content — retry
                time.sleep(2)
                continue
            except Exception:
                if attempt < 2:
                    time.sleep(2 ** attempt)

        # Store the response in conversation history
        self.messages.append({"role": "assistant", "content": content or "(no response)"})
        return content

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
             api_key: str = None, names: list[str] = None,
             verbose: bool = False) -> list[dict]:
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
        bot = LLMBot(server_url, participant_urls[i], personalities[i],
                     model, api_key, verbose=verbose)
        bot.name = names[i]
        bots.append(bot)

    results = [None] * n
    completed = 0

    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = {pool.submit(bot.play): i for i, bot in enumerate(bots)}
        for future in as_completed(futures):
            idx = futures[future]
            bot = bots[idx]
            completed += 1
            try:
                decisions = future.result()
                results[idx] = {
                    'agent': bot.name,
                    'url': bot.participant_url,
                    'decisions': decisions,
                }
                if verbose:
                    n_ok = len([d for d in decisions if 'error' not in d])
                    print(f"  [{bot.name}] finished: {n_ok} pages answered "
                          f"({completed}/{n} bots done)", flush=True)
            except Exception as e:
                results[idx] = {
                    'agent': bot.name,
                    'url': bot.participant_url,
                    'error': str(e),
                }
                if verbose:
                    print(f"  [{bot.name}] ERROR: {e} "
                          f"({completed}/{n} bots done)", flush=True)

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


def _try_direct_value(text: str, field: FormField) -> dict:
    """Try to interpret raw text as a direct value for the field."""
    text = text.strip().strip('"').strip("'")

    if field.input_type == 'number':
        try:
            val = float(text.split()[0])  # take first number-like token
            return {field.name: val}
        except (ValueError, IndexError):
            # Try to find any number in the text
            import re
            m = re.search(r'\d+', text)
            if m:
                return {field.name: int(m.group())}

    if field.choices:
        # Check if the text matches a choice
        for value, display in field.choices:
            if display.lower() in text.lower() or value.lower() in text.lower():
                return {field.name: value}

    return {}


def _escape_jinja(text: str) -> str:
    """Escape {{ }} and {% %} so they don't get interpreted as templates."""
    text = re.sub(r'\{\{', '{ {', text)
    text = re.sub(r'\}\}', '} }', text)
    text = re.sub(r'\{%', '{ %', text)
    text = re.sub(r'%\}', '% }', text)
    return text
