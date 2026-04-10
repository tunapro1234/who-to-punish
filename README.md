# replicant

[![PyPI](https://img.shields.io/pypi/v/pyreplicant.svg)](https://pypi.org/project/pyreplicant/)
[![Python](https://img.shields.io/pypi/pyversions/pyreplicant.svg)](https://pypi.org/project/pyreplicant/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Connect LLM agents to oTree experiments with Big Five personalities.

> Note: install as `pyreplicant`, import as `replicant` (like `pillow`/`PIL`).

`replicant` lets you run real [oTree](https://www.otree.org/) experiments using LLM agents as participants — alone, or mixed with humans. Each agent gets a personality drawn from validated psychometric distributions (BFI-2-Expanded, Soto & John 2017), and the framework handles all the HTTP plumbing: form parsing, response validation, retries, and synchronization.

## What is this?

A Python toolkit with three main pieces:

- **BFI-2-Expanded personality induction** with cross-instrument validation (r = 0.91 vs Mini-IPIP)
- **oTree LLM bots** that connect to a real oTree server as HTTP clients
- **Paper replication tools** — copy a template, point to your oTree app, run

The first paper replicated is **Ertan, Page & Putterman (2009)** — *"Who to Punish? Individual Decisions and Majority Rule in Mitigating the Free Rider Problem"* (European Economic Review, 53(5), 495-511).

## Why "replicant"?

Two reasons. First, replication: the goal is to reproduce human behavioral findings with LLM agents. Second, the Blade Runner sense: synthetic agents that approximate but don't perfectly equal humans. Both meanings apply.

## Project Structure

```
src/
└── replicant/                            # The framework (installable package)
    ├── personalities/
    │   ├── factory.py                    # PersonalityFactory + sentence bank
    │   └── validation.py                 # Mini-IPIP cross-instrument testing
    ├── otree/
    │   ├── client.py                     # HTTP client for oTree server
    │   ├── bot.py                        # LLMBot + FormController
    │   ├── hybrid.py                     # HybridSession (humans + bots)
    │   ├── parser.py                     # Parse oTree app source
    │   └── translator.py                 # oTree app introspection
    ├── analysis/
    │   ├── cost.py                       # Cost estimation per model
    │   ├── stats.py                      # Mann-Whitney, chi-square, etc.
    │   └── comparison.py                 # PaperComparison helper
    ├── cli.py                            # Generic run_paper() helper
    └── preflight.py                      # Pre-flight checks (API key, model)

papers/                                   # Paper replications (each self-contained)
├── TEMPLATE/                             # Copy this to start a new paper
└── ertan2009/
    ├── config.py                         # Game parameters + paper findings
    ├── experiment.py                     # Points to oTree app
    ├── analyze.py                        # Paper-specific analysis
    ├── run.py                            # CLI entry point (~20 lines)
    ├── results/                          # JSON/CSV data
    ├── replicated/                       # The research paper output
    │   ├── paper.tex
    │   ├── paper.pdf
    │   └── figures/
    └── legacy/                           # Pre-oTree EDSL pipeline scripts

tests/
└── otree_server/                         # Real oTree server (Docker)
    ├── ertan2009/                        # Our experiment as an oTree app
    ├── dictator/, prisoner/, trust/, ...  # 12+ classic games
    └── Dockerfile
```

## Quick Start

### Install

```bash
pip install pyreplicant
```

For analysis features (figures, statistics):

```bash
pip install "pyreplicant[analysis]"
```

### Prerequisites

- Python 3.10+
- An [OpenRouter](https://openrouter.ai) API key (for LLM access)
- (Optional) Docker for the oTree server

### Setup

```bash
export OPEN_ROUTER_API_KEY="your-key-here"
```

### Run a paper replication

```bash
# 1. Start the oTree server
docker compose up -d

# 2. Run with default agents
python -m papers.ertan2009.run -n 10

# With Big Five personalities sampled from population norms
python -m papers.ertan2009.run -n 10 --random-personalities --seed 42

# With a skewed population (all disagreeable)
python -m papers.ertan2009.run -n 10 --skew-agreeableness 1.5

# Different model
python -m papers.ertan2009.run -n 10 --model deepseek/deepseek-v3.2
```

### Quick example: build a personality

```python
from replicant import build_personality, sample_personalities

# A single agent with explicit OCEAN scores
desc = build_personality(extraversion=4.5, agreeableness=1.5, neuroticism=4.0)
print(desc)

# A population sampled from US adult norms
personalities = sample_personalities(n=20, seed=42)

# A skewed population (all disagreeable)
disagreeable = sample_personalities(n=20, agreeableness=1.5)
```

### Validate personality induction

```python
from replicant.personalities import run_continuous_validation

results = run_continuous_validation(n=20, model="stepfun/step-3.5-flash")
# Reports Pearson r, R², MAE, bias per Big Five domain
# Pooled r ≈ 0.91 for Step 3.5 Flash
```

## How it works

```
oTree server          replicant LLM bot
─────────────         ──────────────────
HTML page  ────────►  HTML parser
                      ▼
                      FormController
                      (validates fields)
                      ▼
                      LLM (with personality)
                      ▼
                      JSON answer
HTML page  ◄────────  oTree form POST
```

The bot reads each oTree page, extracts form fields, builds a prompt that includes the agent's personality and the page context, calls the LLM, validates the response against the field constraints, and submits. From oTree's perspective, the bot is indistinguishable from a human participant in a browser.

### Human-LLM Hybrid Sessions

You can mix LLM bots with real human participants in the same session. oTree handles synchronization — bots wait at WaitPages until humans submit their decisions.

```python
from replicant.otree import HybridSession

session = HybridSession("http://localhost:8000", model="stepfun/step-3.5-flash")

# Create session with 6 participants
urls = session.create("ertan2009", n_participants=6)

# Split: 3 humans, 3 bots
human_urls, bot_urls = session.split(n_humans=3)

# Show human links (give these to your participants)
session.print_human_links(human_urls)

# Run bots — they'll wait at WaitPages for humans
results = session.run_bots(bot_urls, big5=True)
```

See `examples/06_hybrid_humans_bots.py` for a complete example.

After the session finishes, fetch the data (humans + bots together) from oTree:

```python
csv_path = session.export_results(app_name="ertan2009", output_dir="results")
```

This calls oTree's REST API export endpoint and saves the participant data
as a CSV — one row per participant, all form responses included.

## BFI-2 Personality System

`replicant` uses the **BFI-2-Expanded** format (Soto & John, 2017) — instead of giving LLMs numeric trait scores (which don't reliably influence behavior), it builds rich personality descriptions from a 7-level sentence bank.

```python
from replicant import build_personality, sample_personalities

# Direct OCEAN scores (1-5 scale)
desc = build_personality(
    extraversion=4.5,
    agreeableness=1.5,
    conscientiousness=3.0,
    neuroticism=4.0,
    openness=3.5,
)

# Sample from US adult population norms
personalities = sample_personalities(n=50, seed=42)

# Skewed population (e.g., all disagreeable)
personalities = sample_personalities(n=50, agreeableness=1.5)

# Multiple skews
personalities = sample_personalities(
    n=50,
    agreeableness=1.5,
    neuroticism=4.5,
)
```

### Validation

We test that personality induction actually works using **cross-instrument validation**: assign with BFI-2-Expanded sentences, measure with the Mini-IPIP 20-item Likert scale (different wording so the model can't parrot the assignment). For Step 3.5 Flash:

| Domain | Pearson r | R² | MAE | Bias |
|--------|-----------|-----|-----|------|
| Extraversion | 0.95 | 0.91 | 0.55 | -0.03 |
| Agreeableness | 0.80 | 0.64 | 0.74 | +0.74 |
| Conscientiousness | 0.91 | 0.83 | 0.55 | +0.44 |
| Neuroticism | 0.94 | 0.88 | 0.83 | -0.69 |
| Openness | 0.93 | 0.86 | 0.67 | +0.66 |
| **Pooled** | **0.91** | **0.83** | 0.67 | — |

The agreeableness and neuroticism biases reflect RLHF training (models drift toward warmth and away from negative affect) — see the paper for details.

## Adding a New Paper

1. Copy `papers/TEMPLATE/` to `papers/<your_paper>/`
2. Create your oTree app at `tests/otree_server/<your_paper>/`
3. Register the session config in `tests/otree_server/settings.py`
4. Edit `papers/<your_paper>/config.py` with the paper's known findings
5. Edit `analyze.py` to extract the metrics that matter
6. Run: `python -m papers.<your_paper>.run -n 10 --random-personalities`

## Known Models

| Model | Cost | Notes |
|-------|------|-------|
| `stepfun/step-3.5-flash` | $0.10 / $0.30 per M | Reasoning model, slow but thoughtful |
| `deepseek/deepseek-v3.2` | $0.26 / $0.38 per M | Fast non-reasoning model |
| `qwen/qwen3.6-plus:free` | Free | Often rate-limited |
| `meta-llama/llama-3.1-8b-instruct` | $0.02 / $0.05 per M | Cheapest option |

## Citation

If you use `replicant` in research, please cite the underlying tools:

```
EDSL: Horton, J. J., et al. (2024). Expected Parrot Domain-Specific Language.
oTree: Chen, D. L., Schonger, M., & Wickens, C. (2016). oTree—An open-source
       platform for laboratory, online, and field experiments.
```

## License

MIT
