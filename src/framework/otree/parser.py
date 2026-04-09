"""
Parse oTree app source into a structured representation.

Reads an oTree __init__.py (modern "no-self" format) and extracts:
- Constants (game parameters)
- Player and Group fields (participant decisions)
- Page definitions (screens, form fields)
- Page sequence (execution order)
- Template text (participant-facing instructions)

Also supports legacy format (models.py + pages.py).
"""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Data structures ─────────────────────────────────────────────────

@dataclass
class FieldDef:
    """A Player or Group field (participant decision or computed value)."""
    name: str
    field_type: str  # Integer, Boolean, String, Currency, Float, LongString
    label: str | None = None
    min_value: Any = None
    max_value: Any = None
    choices: list | None = None
    initial: Any = None
    doc: str | None = None
    widget: str | None = None


@dataclass
class PageDef:
    """A Page or WaitPage in the experiment flow."""
    name: str
    page_type: str  # 'Page' or 'WaitPage'
    form_model: str | None = None
    form_fields: list[str] = field(default_factory=list)
    template_text: str | None = None


@dataclass
class OTreeApp:
    """Complete parsed representation of an oTree app."""
    name: str
    doc: str | None = None
    constants: dict[str, Any] = field(default_factory=dict)
    players_per_group: int | None = None
    num_rounds: int = 1
    player_fields: list[FieldDef] = field(default_factory=list)
    group_fields: list[FieldDef] = field(default_factory=list)
    pages: list[PageDef] = field(default_factory=list)
    page_sequence: list[str] = field(default_factory=list)

    def decision_pages(self) -> list[PageDef]:
        """Pages where participants make decisions (have form fields)."""
        seq = {name: i for i, name in enumerate(self.page_sequence)}
        pages = [p for p in self.pages if p.form_fields and p.page_type == 'Page']
        pages.sort(key=lambda p: seq.get(p.name, 999))
        return pages

    def get_field(self, name: str) -> FieldDef | None:
        """Look up a field by name across Player and Group."""
        for f in self.player_fields + self.group_fields:
            if f.name == name:
                return f
        return None


# ── Field type mapping ──────────────────────────────────────────────

FIELD_TYPES = {
    'IntegerField': 'Integer',
    'PositiveIntegerField': 'Integer',
    'FloatField': 'Float',
    'BooleanField': 'Boolean',
    'StringField': 'String',
    'CharField': 'String',
    'LongStringField': 'LongString',
    'TextField': 'LongString',
    'CurrencyField': 'Currency',
    'DecimalField': 'Float',
}


# ── Public API ──────────────────────────────────────────────────────

def parse(app_path: str | Path) -> OTreeApp:
    """
    Parse an oTree app directory into an OTreeApp.

    Supports both modern format (single __init__.py) and legacy format
    (models.py + pages.py).
    """
    app_path = Path(app_path)

    if not app_path.is_dir():
        raise FileNotFoundError(f"Not a directory: {app_path}")

    app = OTreeApp(name=app_path.name)

    init_file = app_path / '__init__.py'
    models_file = app_path / 'models.py'
    pages_file = app_path / 'pages.py'

    if init_file.exists() and _is_modern_format(init_file):
        _parse_source(init_file.read_text(), app)
    elif models_file.exists():
        _parse_source(models_file.read_text(), app)
        if pages_file.exists():
            _parse_source(pages_file.read_text(), app)
    else:
        raise FileNotFoundError(
            f"No oTree app found in {app_path}. "
            "Expected __init__.py (modern) or models.py (legacy)."
        )

    # Load template text for decision pages
    for page in app.pages:
        if page.page_type == 'Page':
            _load_template(app_path, page)

    return app


# ── Source parsing ──────────────────────────────────────────────────

def _is_modern_format(init_file: Path) -> bool:
    """Check if __init__.py uses modern oTree format (has imports)."""
    text = init_file.read_text()
    return 'import' in text and len(text) > 50


def _parse_source(source: str, app: OTreeApp):
    """Parse a single Python source file into the app."""
    tree = ast.parse(source)

    if app.doc is None:
        app.doc = _extract_doc(tree)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            _parse_class(node, app)
        elif isinstance(node, ast.Assign):
            _parse_module_assign(node, app)


def _extract_doc(tree: ast.Module) -> str | None:
    """Extract module docstring or `doc = '...'` variable."""
    for node in tree.body:
        # Module docstring
        if (isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            return node.value.value
        # doc = "..." assignment
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Name) and target.id == 'doc'
                        and isinstance(node.value, ast.Constant)):
                    return node.value.value
    return None


# ── Class parsing ───────────────────────────────────────────────────

def _parse_class(node: ast.ClassDef, app: OTreeApp):
    bases = [_base_name(b) for b in node.bases]

    if node.name == 'C' or 'BaseConstants' in bases:
        _parse_constants(node, app)
    elif 'BasePlayer' in bases:
        _parse_model_fields(node, app.player_fields)
    elif 'BaseGroup' in bases:
        _parse_model_fields(node, app.group_fields)
    elif 'Page' in bases:
        _parse_page(node, 'Page', app)
    elif 'WaitPage' in bases:
        _parse_page(node, 'WaitPage', app)


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ''


def _parse_constants(node: ast.ClassDef, app: OTreeApp):
    for item in node.body:
        if not isinstance(item, ast.Assign):
            continue
        for target in item.targets:
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            value = _eval_node(item.value)

            if name == 'PLAYERS_PER_GROUP':
                app.players_per_group = value
            elif name == 'NUM_ROUNDS':
                app.num_rounds = value
            elif name == 'NAME_IN_URL':
                pass
            else:
                app.constants[name] = value


def _parse_model_fields(node: ast.ClassDef, fields: list[FieldDef]):
    for item in node.body:
        if not isinstance(item, ast.Assign):
            continue
        for target in item.targets:
            if isinstance(target, ast.Name) and isinstance(item.value, ast.Call):
                fd = _parse_field(target.id, item.value)
                if fd:
                    fields.append(fd)


def _parse_field(name: str, call: ast.Call) -> FieldDef | None:
    func_name = _call_name(call)
    if func_name not in FIELD_TYPES:
        return None

    fd = FieldDef(name=name, field_type=FIELD_TYPES[func_name])

    for kw in call.keywords:
        val = _eval_node(kw.value)
        match kw.arg:
            case 'label': fd.label = val
            case 'min': fd.min_value = val
            case 'max': fd.max_value = val
            case 'choices': fd.choices = val
            case 'initial': fd.initial = val
            case 'doc': fd.doc = val
            case 'widget': fd.widget = _widget_name(kw.value)

    return fd


def _parse_page(node: ast.ClassDef, page_type: str, app: OTreeApp):
    page = PageDef(name=node.name, page_type=page_type)

    for item in node.body:
        if not isinstance(item, ast.Assign):
            continue
        for target in item.targets:
            if not isinstance(target, ast.Name):
                continue
            val = _eval_node(item.value)
            if target.id == 'form_model':
                page.form_model = val
            elif target.id == 'form_fields':
                page.form_fields = val if isinstance(val, list) else []

    app.pages.append(page)


def _parse_module_assign(node: ast.Assign, app: OTreeApp):
    for target in node.targets:
        if isinstance(target, ast.Name) and target.id == 'page_sequence':
            if isinstance(node.value, ast.List):
                app.page_sequence = [
                    el.id for el in node.value.elts
                    if isinstance(el, ast.Name)
                ]


# ── AST evaluation helpers ──────────────────────────────────────────

def _eval_node(node: ast.expr) -> Any:
    """Evaluate a constant expression. Returns the value or a string placeholder."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        val = _eval_node(node.operand)
        return -val if isinstance(val, (int, float)) else val
    if isinstance(node, ast.Call):
        name = _call_name(node)
        # cu(100) / Currency(100) → extract numeric value
        if name in ('cu', 'Currency') and node.args:
            return _eval_node(node.args[0])
        return f'{name}(...)'
    if isinstance(node, ast.List):
        return [_eval_node(el) for el in node.elts]
    if isinstance(node, ast.Tuple):
        return [_eval_node(el) for el in node.elts]
    if isinstance(node, ast.Attribute):
        obj = _eval_node(node.value)
        return f'{obj}.{node.attr}'
    if isinstance(node, ast.Name):
        return node.id
    return None


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ''


def _widget_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


# ── Template loading ────────────────────────────────────────────────

def _load_template(app_path: Path, page: PageDef):
    """Extract participant-facing text from an oTree HTML template."""
    template_file = app_path / f'{page.name}.html'

    # Legacy format: templates/<app_name>/PageName.html
    if not template_file.exists():
        legacy = app_path / 'templates' / app_path.name / f'{page.name}.html'
        if legacy.exists():
            template_file = legacy
        else:
            return

    html = template_file.read_text()

    # Extract {{ block content }} ... {{ endblock }}
    match = re.search(
        r'\{\{\s*block\s+content\s*\}\}(.*?)\{\{\s*endblock\s*\}\}',
        html, re.DOTALL,
    )
    if not match:
        return

    content = match.group(1)
    # Strip HTML tags
    content = re.sub(r'<[^>]+>', ' ', content)
    # Remove oTree widget tags but keep variable references
    content = re.sub(r'\{\{\s*(formfields?|next_button|include_sibling\s+\S+)\s*\}\}', '', content)
    # Normalize whitespace
    content = re.sub(r'\s+', ' ', content).strip()

    if content:
        page.template_text = content
