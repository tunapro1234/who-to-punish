# Legacy: EDSL-based scripts

These scripts were used to generate the original N=50 results in
`papers/ertan2009/results/big5_random_*.csv` and `default_*.csv`,
which were collected via the (now removed) EDSL pipeline.

They are kept here for reproducibility of the published paper but
are not used by the current oTree-based pipeline.

| File | Purpose |
|------|---------|
| `compare.py` | Statistical comparison default vs personality (EDSL CSVs) |
| `figures.py` | Generate paper figures 1-4 from EDSL CSVs |
| `fig_personality.py` | Generate paper figure 5 from EDSL CSVs |

For new experiments, use `papers.ertan2009.run` (the oTree-based CLI)
and `papers.ertan2009.analyze` (the oTree-based analysis).
