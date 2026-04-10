# Changelog

All notable changes to `replicant` will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `replicant.analysis.cost` — pre-flight cost estimation for experiments
- `replicant.analysis.stats` — generic statistical helpers (Mann-Whitney U,
  chi-square, Cohen's d, formatted comparison tables)
- `BehavioralExperiment.estimate_calls()` — predict total API calls
- `BehavioralExperiment.run()` now prints a cost estimate before starting
- `CHANGELOG.md`

### Changed
- `replicant` is now pip-installable (`pip install -e .`) — no more
  `sys.path.insert` boilerplate in paper files or examples
- `papers/ertan2009/compare.py` refactored to use `replicant.analysis.stats`
  helpers instead of duplicating Mann-Whitney / chi-square code
- Removed empty `src/__init__.py` (was making `src` an unintended package)

## [0.2.0] — 2026-04-09

### Added
- **`replicant` package** — renamed from `framework/` with cleaner submodule layout
- **`papers/` outside `src/`** — each paper self-contained with its own
  `results/` and `replicated/` (LaTeX paper) subdirectories
- **`papers/TEMPLATE/`** — copy-paste scaffold for new paper replications
- **Top-level personality helpers**: `build_personality(extraversion=...)`,
  `sample_personalities(n, agreeableness=1.5)` — direct OCEAN trait input
- **`HybridSession`** for mixing LLM bots with human participants in oTree
- **7 example scripts** in `examples/` covering all common use cases
- **Continuous personality validation** with Pearson r, R², MAE per Big Five
  domain (replaces the old 88% binary classification metric)
- **Skewed population support** via `mean_overrides` argument

### Changed
- Validation now reports r=0.911 pooled correlation (was 88% binary match)
- Paper updated to show continuous validation metrics

## [0.1.0] — Initial development

### Added
- **EDSL-based pipeline**: `BehavioralExperiment` with multi-part survey runner
- **BFI-2-Expanded personality system** with 7-level sentence bank
- **Cross-instrument validation** (assign BFI-2, measure Mini-IPIP)
- **5 named personality profiles** (cooperative, selfish, leader, anxious, average)
- **Random population sampling** from US adult Big Five norms
- **Automatic NaN retry** in experiment runner
- **oTree integration** — LLM bots connect to a real oTree server as HTTP clients
- **`FormController`** validates LLM responses before oTree submission
- **Sequential field mode** for complex pages (4+ form fields)
- **Ertan, Page & Putterman (2009) replication** as the first paper
- **16-page LaTeX research paper** with N=50 results and 5 figures
- **Docker oTree server** with 12+ classic experimental economics games
- **EDSL bug fix** — patched async client cache to prevent stale-loop NaN
  responses on sequential `Survey.run()` calls (later fixed upstream in EDSL #2429)

### Notable findings
- Default LLM agents (no personality) act as hyper-rational free-riders:
  zero contribution across all conditions
- Personality-endowed agents (N=50) show baseline contribution of 13.0
  (vs ~7.5 in the original paper) — exhibits an RLHF prosociality bias
- Personality agents reproduce the human pattern: agreeableness predicts
  cooperation (r=+0.54), neuroticism predicts free-riding (r=-0.43)
- Disagreeable populations (mean_overrides agreeableness=1.5) show
  antisocial punishment — punishing even cooperative agents (1.2 tokens)
