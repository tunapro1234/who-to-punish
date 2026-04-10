"""
HTTP client for connecting to a running oTree server as a participant.

Handles the participant flow: fetch pages, parse HTML to extract instructions
and form fields, submit decisions, and poll WaitPages. This is the core of
the "LLMs as oTree clients" approach — the server runs normally, and LLM
agents connect exactly like human participants would via a browser.
"""

import re
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse, urljoin

import requests


# ── Page data structures ────────────────────────────────────────────

@dataclass
class FormField:
    """A form field extracted from an oTree page."""
    name: str
    input_type: str  # 'number', 'text', 'radio', 'select', 'checkbox'
    label: str = ""
    choices: list[tuple[str, str]] | None = None  # [(value, display_text)]
    min_value: str | None = None
    max_value: str | None = None


@dataclass
class PageData:
    """Parsed content of an oTree page."""
    url: str
    title: str = ""
    body_text: str = ""
    form_fields: list[FormField] = field(default_factory=list)
    form_action: str = ""
    csrf_token: str = ""
    is_wait_page: bool = False
    is_finished: bool = False


# ── Client ──────────────────────────────────────────────────────────

class OTreeClient:
    """
    HTTP client for a single oTree participant.

    Each instance maintains its own session (cookies, etc.),
    so multiple clients can run concurrently for different participants.
    """

    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip('/')
        self.http = requests.Session()

    def get_page(self, url: str) -> PageData:
        """Fetch a participant page and parse it."""
        resp = self.http.get(url, allow_redirects=True)
        resp.raise_for_status()
        return _parse_page(resp.text, resp.url)

    def submit(self, page: PageData, data: dict) -> PageData:
        """Submit form data for a page. Returns the next page."""
        form_data = {}
        if page.csrf_token:
            form_data['csrfmiddlewaretoken'] = page.csrf_token
        form_data.update(data)

        # oTree forms POST to the current page URL (action is often empty)
        url = page.form_action or page.url
        resp = self.http.post(url, data=form_data, allow_redirects=True)
        resp.raise_for_status()
        return _parse_page(resp.text, resp.url)

    def wait_for_page(self, page: PageData,
                      timeout: float = 1800, poll_interval: float = 2.0) -> PageData:
        """Poll a WaitPage until oTree advances it. Default 30 min timeout (allows human participants)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(poll_interval)
            next_page = self.get_page(page.url)
            if not next_page.is_wait_page:
                return next_page
        raise TimeoutError(f"WaitPage at {page.url} did not advance within {timeout}s")

    @staticmethod
    def create_session(server_url: str, session_config: str,
                       n_participants: int, rest_key: str | None = None) -> list[str]:
        """
        Create an oTree session. Returns participant start URLs.

        Tries in order:
          1. REST API (if rest_key provided or OTREE_REST_KEY is set)
          2. Admin interface (works in devserver mode, no config needed)
        """
        server_url = server_url.rstrip('/')

        if rest_key:
            return _create_via_rest_api(server_url, session_config,
                                        n_participants, rest_key)

        # Try REST API without key (some setups allow it)
        try:
            return _create_via_rest_api(server_url, session_config,
                                        n_participants, None)
        except Exception:
            pass

        # Fall back to admin interface
        return _create_via_admin(server_url, session_config, n_participants)

    @staticmethod
    def get_session_urls(server_url: str, session_code: str) -> list[str]:
        """Fetch participant URLs for an existing session."""
        server_url = server_url.rstrip('/')
        resp = requests.get(
            f'{server_url}/SessionStartLinks/{session_code}',
            allow_redirects=True,
        )
        resp.raise_for_status()
        return _extract_participant_urls(resp.text, server_url)


# ── Session creation helpers ─────────────────────────────────────────

def _create_via_rest_api(server_url: str, session_config: str,
                         n_participants: int, rest_key: str | None) -> list[str]:
    headers = {'Content-Type': 'application/json'}
    if rest_key:
        headers['otree-rest-key'] = rest_key

    # Step 1: Create session
    resp = requests.post(
        f'{server_url}/api/sessions',
        json={
            'session_config_name': session_config,
            'num_participants': n_participants,
        },
        headers=headers,
    )
    resp.raise_for_status()
    session_code = resp.json()['code']

    # Step 2: Fetch participant codes
    resp = requests.post(
        f'{server_url}/api/get_session/{session_code}',
        json={},
        headers=headers,
    )
    resp.raise_for_status()

    return [
        f"{server_url}/InitializeParticipant/{p['code']}"
        for p in resp.json()['participants']
    ]


def _create_via_admin(server_url: str, session_config: str,
                      n_participants: int) -> list[str]:
    """Create a session via oTree's admin web interface (no REST key needed)."""
    s = requests.Session()

    # Get CSRF token from admin page
    resp = s.get(f'{server_url}/create_session/', allow_redirects=True)
    resp.raise_for_status()

    csrf = re.search(
        r'name=["\']csrfmiddlewaretoken["\'][\s]+value=["\']([^"\']+)', resp.text
    )
    if not csrf:
        csrf = re.search(r'value=["\']([^"\']+)["\'][\s]+name=["\']csrfmiddlewaretoken', resp.text)
    if not csrf:
        raise RuntimeError(
            "Could not access oTree admin. If your server uses AUTH_LEVEL, "
            "provide a REST key with --rest-key instead."
        )

    # Create the session
    resp = s.post(
        f'{server_url}/create_session/',
        data={
            'csrfmiddlewaretoken': csrf.group(1),
            'session_config_name': session_config,
            'num_participants': str(n_participants),
        },
        allow_redirects=True,
    )
    resp.raise_for_status()

    # Extract participant URLs from the redirect page
    urls = _extract_participant_urls(resp.text, server_url)
    if not urls:
        raise RuntimeError(
            "Session created but could not extract participant URLs. "
            "Try using --session-code or --urls instead."
        )
    return urls


def _extract_participant_urls(html: str, server_url: str) -> list[str]:
    """Pull participant start links from an oTree admin page."""
    urls = []
    for m in re.finditer(r'/InitializeParticipant/(\w+)', html):
        urls.append(f"{server_url}/InitializeParticipant/{m.group(1)}")
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


# ── HTML parsing ────────────────────────────────────────────────────

def _parse_page(html: str, url: str) -> PageData:
    """Parse oTree HTML into a PageData structure."""
    page = PageData(url=url)

    if _is_wait_page(html):
        page.is_wait_page = True
        return page

    if _is_finished(html, url):
        page.is_finished = True
        return page

    page.csrf_token = _find_csrf(html)
    page.form_action = _find_form_action(html, url)
    page.title = _find_title(html)
    page.body_text = _extract_body_text(html)
    page.form_fields = _extract_form_fields(html)
    return page


def _is_wait_page(html: str) -> bool:
    return 'otree-wait-page' in html or 'wait-page-body' in html


def _is_finished(html: str, url: str) -> bool:
    return ('OutOfRangeNotification' in url
            or 'This page is not available' in html)


def _find_csrf(html: str) -> str:
    m = re.search(r'name=["\']csrfmiddlewaretoken["\'][\s]+value=["\']([^"\']+)', html)
    if not m:
        m = re.search(r'value=["\']([^"\']+)["\'][\s]+name=["\']csrfmiddlewaretoken', html)
    return m.group(1) if m else ""


def _find_form_action(html: str, page_url: str) -> str:
    m = re.search(r'<form[^>]+action=["\']([^"\']+)', html)
    if m:
        action = m.group(1)
        if action.startswith('/'):
            parsed = urlparse(page_url)
            return f"{parsed.scheme}://{parsed.netloc}{action}"
        return action
    return ""


def _find_title(html: str) -> str:
    m = re.search(r'<title>([^<]+)</title>', html)
    return m.group(1).strip() if m else ""


def _extract_body_text(html: str) -> str:
    """Extract readable text from the page, focusing on the content area."""
    content = html

    # Try to narrow to the form / content area
    for pattern in [
        r'class=["\'][^"\']*otree-body[^"\']*["\'][^>]*>(.*)',
        r'<form[^>]*>(.*?)</form>',
    ]:
        m = re.search(pattern, content, re.DOTALL)
        if m:
            content = m.group(1)
            break

    # Strip non-content elements
    content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL)
    content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL)
    content = re.sub(r'<input[^>]+type=["\']hidden["\'][^>]*>', '', content)
    content = re.sub(r'<button[^>]*>.*?</button>', '', content, flags=re.DOTALL)

    # Tags → spaces, then clean up
    content = re.sub(r'<[^>]+>', ' ', content)
    for entity, char in [('&amp;','&'), ('&lt;','<'), ('&gt;','>'),
                         ('&nbsp;',' '), ('&#39;',"'"), ('&quot;','"')]:
        content = content.replace(entity, char)
    return re.sub(r'\s+', ' ', content).strip()


def _extract_form_fields(html: str) -> list[FormField]:
    """Extract all visible form fields from the page."""
    fields = []
    labels = _extract_labels(html)

    # <input type="number">
    for m in re.finditer(r'<input\b[^>]+type=["\']number["\'][^>]*>', html):
        tag = m.group(0)
        name = _attr(tag, 'name')
        if name:
            fields.append(FormField(
                name=name, input_type='number',
                label=labels.get(name, ''),
                min_value=_attr(tag, 'min'),
                max_value=_attr(tag, 'max'),
            ))

    # <input type="text"> (excluding hidden/csrf)
    for m in re.finditer(r'<input\b[^>]+type=["\']text["\'][^>]*>', html):
        tag = m.group(0)
        name = _attr(tag, 'name')
        if name and name != 'csrfmiddlewaretoken':
            fields.append(FormField(
                name=name, input_type='text',
                label=labels.get(name, ''),
            ))

    # <input type="radio"> — group by name
    radio_groups: dict[str, list[tuple[str, str]]] = {}
    for m in re.finditer(r'<input\b[^>]+type=["\']radio["\'][^>]*>', html):
        tag = m.group(0)
        name = _attr(tag, 'name')
        value = _attr(tag, 'value') or ''
        if name:
            if name not in radio_groups:
                radio_groups[name] = []
            display = _find_radio_label(html, name, value) or value
            radio_groups[name].append((value, display))
    for name, choices in radio_groups.items():
        fields.append(FormField(
            name=name, input_type='radio',
            label=labels.get(name, ''),
            choices=choices,
        ))

    # <button name="x" value="y"> — oTree uses these as choice buttons
    button_groups: dict[str, list[tuple[str, str]]] = {}
    for m in re.finditer(
        r'<button\b[^>]+name=["\'](\w+)["\'][^>]+value=["\']([^"\']*)["\'][^>]*>(.*?)</button>',
        html, re.DOTALL,
    ):
        name, value, inner = m.group(1), m.group(2), m.group(3)
        display = re.sub(r'<[^>]+>', '', inner).strip()
        if name not in button_groups:
            button_groups[name] = []
        button_groups[name].append((value, display or value))
    for name, choices in button_groups.items():
        fields.append(FormField(
            name=name, input_type='button',
            label=labels.get(name, ''),
            choices=choices,
        ))

    # <select>
    for m in re.finditer(
        r'<select\b[^>]+name=["\'](\w+)["\'][^>]*>(.*?)</select>',
        html, re.DOTALL,
    ):
        name, options_html = m.group(1), m.group(2)
        choices = []
        for opt in re.finditer(
            r'<option\b[^>]+value=["\']([^"\']*)["\'][^>]*>([^<]*)', options_html
        ):
            val, text = opt.group(1), opt.group(2).strip()
            if val:  # skip empty placeholder options
                choices.append((val, text))
        if choices:
            fields.append(FormField(
                name=name, input_type='select',
                label=labels.get(name, ''),
                choices=choices,
            ))

    # <textarea>
    for m in re.finditer(r'<textarea\b[^>]+name=["\'](\w+)["\']', html):
        name = m.group(1)
        fields.append(FormField(
            name=name, input_type='text',
            label=labels.get(name, ''),
        ))

    return fields


def _extract_labels(html: str) -> dict[str, str]:
    """Map field names to their <label> text."""
    labels = {}
    for m in re.finditer(
        r'<label\b[^>]+for=["\']id_(\w+)["\'][^>]*>(.*?)</label>',
        html, re.DOTALL,
    ):
        name = m.group(1)
        text = re.sub(r'<[^>]+>', '', m.group(2)).strip()
        if text:
            labels[name] = text
    return labels


def _find_radio_label(html: str, name: str, value: str) -> str | None:
    """Find display text for a specific radio option."""
    escaped_name = re.escape(name)
    escaped_val = re.escape(value)
    # Label wrapping the radio input
    pattern = (
        rf'<label[^>]*>\s*<input[^>]+name=["\']{ escaped_name }["\']'
        rf'[^>]+value=["\']{ escaped_val }["\'][^>]*>\s*([^<]+)</label>'
    )
    m = re.search(pattern, html, re.DOTALL)
    return m.group(1).strip() if m else None


def _attr(tag: str, name: str) -> str | None:
    """Extract an attribute value from a single HTML tag."""
    m = re.search(rf'\b{name}=["\']([^"\']*)["\']', tag)
    return m.group(1) if m else None
