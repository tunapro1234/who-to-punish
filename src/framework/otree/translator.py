"""
Translate a parsed OTreeApp into EDSL surveys.

Maps oTree concepts to EDSL:
- Page with form_fields  →  EDSL Questions
- IntegerField / CurrencyField  →  QuestionNumerical
- BooleanField / choices  →  QuestionMultipleChoice
- StringField (no choices)  →  QuestionFreeText
- Constants + template text  →  question context
"""

import re

from edsl import QuestionFreeText, QuestionMultipleChoice, QuestionNumerical, Survey

from .parser import OTreeApp, FieldDef, PageDef


def translate(app: OTreeApp) -> Survey:
    """
    Convert a parsed OTreeApp into an EDSL Survey.

    Iterates through the page sequence, skipping WaitPages and pages without
    form fields. Each form field becomes an EDSL question with game context
    built from constants and template text.
    """
    questions = []

    for page in app.decision_pages():
        fields = _resolve_fields(app, page)
        context = _build_context(app, page)

        for fd in fields:
            fd = _resolve_constant_refs(fd, app.constants)
            q = _to_question(fd, context)
            if q:
                questions.append(q)

    if not questions:
        raise ValueError(
            f"No translatable questions found in oTree app '{app.name}'. "
            "Check that pages have form_model and form_fields defined."
        )

    return Survey(questions=questions)


# ── Field resolution ────────────────────────────────────────────────

def _resolve_fields(app: OTreeApp, page: PageDef) -> list[FieldDef]:
    """Look up FieldDefs for a page's form_fields."""
    if page.form_model == 'group':
        source = app.group_fields
    else:
        source = app.player_fields
    field_map = {f.name: f for f in source}
    return [field_map[name] for name in page.form_fields if name in field_map]


def _resolve_constant_refs(fd: FieldDef, constants: dict) -> FieldDef:
    """Replace constant references (e.g. 'C.ENDOWMENT') with actual values."""
    fd.min_value = _resolve_ref(fd.min_value, constants)
    fd.max_value = _resolve_ref(fd.max_value, constants)
    fd.label = _resolve_label_refs(fd.label, constants)
    return fd


def _resolve_ref(value, constants: dict):
    """Resolve a single value that might be a constant reference."""
    if isinstance(value, str) and value.startswith('C.'):
        key = value[2:]
        return constants.get(key, value)
    return value


def _resolve_label_refs(label: str | None, constants: dict) -> str | None:
    """Substitute {{ C.X }} references in label text."""
    if not label:
        return label
    def replace(m):
        key = m.group(1)
        return str(constants.get(key, f'C.{key}'))
    return re.sub(r'\{\{\s*C\.(\w+)\s*\}\}', replace, label)


# ── Context building ────────────────────────────────────────────────

def _build_context(app: OTreeApp, page: PageDef) -> str:
    """
    Build game context from template text and constants.

    Priority:
    1. Template text (the actual instructions participants see)
    2. App doc string (game description)
    3. Auto-generated summary from constants
    """
    parts = []

    if page.template_text:
        text = _substitute_constants(page.template_text, app.constants)
        parts.append(text)
    elif app.doc:
        parts.append(app.doc.strip())

    # Add group context if multiplayer
    if app.players_per_group and app.players_per_group > 1:
        parts.append(f"You are in a group of {app.players_per_group} players.")

    # If no template text, include constants as context
    if not page.template_text and app.constants:
        lines = []
        for name, value in app.constants.items():
            readable = name.lower().replace('_', ' ')
            lines.append(f"- {readable}: {value}")
        if lines:
            parts.append("Game parameters:\n" + "\n".join(lines))

    return "\n\n".join(parts)


def _substitute_constants(text: str, constants: dict) -> str:
    """Replace {{ C.X }} and {{ Constants.X }} in template text."""
    def replace(m):
        key = m.group(1)
        return str(constants.get(key, m.group(0)))
    text = re.sub(r'\{\{\s*C\.(\w+)\s*\}\}', replace, text)
    text = re.sub(r'\{\{\s*Constants\.(\w+)\s*\}\}', replace, text)
    return text


# ── Question mapping ────────────────────────────────────────────────

def _to_question(fd: FieldDef, context: str):
    """Convert a FieldDef to an EDSL Question."""
    label = fd.label or _auto_label(fd.name)
    question_text = f"{context}\n\n{label}" if context else label

    # Any field with explicit choices → MultipleChoice
    if fd.choices:
        return QuestionMultipleChoice(
            question_name=fd.name,
            question_text=question_text,
            question_options=_normalize_choices(fd.choices),
        )

    # Boolean without choices → Yes/No
    if fd.field_type == 'Boolean':
        return QuestionMultipleChoice(
            question_name=fd.name,
            question_text=question_text,
            question_options=['Yes', 'No'],
        )

    # Numeric types → Numerical
    if fd.field_type in ('Integer', 'Float', 'Currency'):
        kwargs = dict(question_name=fd.name, question_text=question_text)
        if isinstance(fd.min_value, (int, float)):
            kwargs['min_value'] = fd.min_value
        if isinstance(fd.max_value, (int, float)):
            kwargs['max_value'] = fd.max_value
        return QuestionNumerical(**kwargs)

    # String / LongString → FreeText
    return QuestionFreeText(
        question_name=fd.name,
        question_text=question_text,
    )


def _normalize_choices(choices: list) -> list[str]:
    """Convert oTree choices (may be [value, label] pairs) to string list."""
    result = []
    for c in choices:
        if isinstance(c, (list, tuple)) and len(c) >= 2:
            result.append(str(c[1]))
        else:
            result.append(str(c))
    return result


def _auto_label(field_name: str) -> str:
    """Generate a readable label from a field name."""
    return field_name.replace('_', ' ').capitalize() + '?'
