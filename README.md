# behave-lab

LLM-based replication of behavioral economics experiments using [EDSL](https://github.com/expectedparrot/edsl).

## What is this?

This project uses large language models as simulated participants to replicate classic behavioral economics experiments. LLM agents answer the same questions real human subjects answered, and we compare the results.

The first experiment replicated is **Ertan, Page & Putterman (2009)** — *"Who to Punish? Individual Decisions and Majority Rule in Mitigating the Free Rider Problem"* (European Economic Review, 53(5), 495-511).

## Project Structure

```
src/
  framework/                        # Our layer on top of EDSL
    experiment.py                    # BehavioralExperiment runner
    personalities.py                 # PersonalityFactory (Big Five agents)
    comparison.py                    # PaperComparison (simulation vs paper)
    templates/
      public_goods.py                # Reusable public goods game surveys
  papers/
    ertan2009/
      experiment.py                  # Paper-specific parameters and analysis
results/                             # Output CSVs and checkpoints (gitignored)
```

## Architecture

```
EDSL (Survey, Agent, Model, Scenario)        ← LLM infrastructure
  |
framework/templates/                          ← Reusable game templates
framework/experiment.py                       ← Multi-part experiment runner
framework/personalities.py                    ← Big Five personality factory
framework/comparison.py                       ← Paper vs simulation comparison
  |
papers/ertan2009/experiment.py                ← Paper-specific implementation
```

## Quick Start

### Prerequisites

- Python 3.10+
- [EDSL](https://github.com/expectedparrot/edsl) installed
- An [OpenRouter](https://openrouter.ai) API key

### Setup

```bash
export OPEN_ROUTER_API_KEY="your-key-here"
```

### Run the Ertan et al. (2009) experiment

```bash
# Default: 10 agents, MiniMax M2.7
python src/papers/ertan2009/experiment.py -n 10

# Specify a different model
python src/papers/ertan2009/experiment.py -n 10 --model meta-llama/llama-3.3-70b-instruct:free

# With Big Five personality agents
python src/papers/ertan2009/experiment.py -n 20 --big5
```

## The Ertan et al. (2009) Experiment

### Design

A **public goods game** (Voluntary Contributions Mechanism) where:
- Groups of 5, each with 20 tokens
- Each token contributed returns 0.4 to every member (MPCR = 0.4)
- Groups vote on whether to allow punishment of low/high contributors
- Punishment costs 1 token and removes 3 from the target

### 4 Parts

1. **Baseline** — contribution decision without punishment
2. **Voting** — should punishment of below/above-average contributors be allowed?
3. **Regimes** — contribution under 4 punishment rules (none, low-only, high-only, unrestricted)
4. **Punishment** — given others' contributions, how much to punish and who to target?

### Key Findings (Paper vs Simulation)

| Finding | Paper | LLM Agents |
|---------|-------|------------|
| No group votes to punish high contributors | 0% | 0% |
| Groups vote to punish low contributors | ~85% | ~100% |
| Punishment targets lowest contributor | Yes | Yes |

## Adding a New Experiment

1. Create a template in `src/framework/templates/` if the game type is new
2. Create a paper directory under `src/papers/`
3. Wire the template to paper-specific parameters
4. Define known findings from the paper for comparison

```python
from src.framework import BehavioralExperiment, PaperComparison
from src.framework.templates.public_goods import contribution_survey

exp = BehavioralExperiment("my_paper", model="minimax/minimax-m2.7")
exp.add_part("baseline", contribution_survey(20, 5, 0.4))
results = exp.run(agents)
```

## EDSL Bug Fix

This project includes a fix for an EDSL bug where sequential `Survey.run()` calls in the same process return NaN. The root cause is stale `aiohttp` sessions cached across event loops. The fix clears async client instances before each `asyncio.run()` call (patched in `edsl/runner/runner.py`).
