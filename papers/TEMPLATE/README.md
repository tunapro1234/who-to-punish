# Paper Template

Copy this directory to create a new paper replication:

```bash
cp -r papers/TEMPLATE papers/your_paper_name
cd papers/your_paper_name
```

Then:

1. **Create the oTree app** at `tests/otree_server/your_paper_name/`. Define
   the game logic, pages, and form fields. See `tests/otree_server/ertan2009/`
   as an example.

2. **Register the session config** in `tests/otree_server/settings.py` so
   the oTree server knows about your app.

3. **Edit `config.py`** to record the paper's known findings.

4. **Edit `experiment.py`** to point to your oTree app name.

5. **Edit `analyze.py`** to extract the metrics that matter for your paper.

6. **`run.py`** usually doesn't need changes — it's a generic CLI.

## File overview

```
your_paper/
├── __init__.py              # Package exports
├── config.py                # Paper findings (for comparison)
├── experiment.py            # Points to the oTree app
├── analyze.py               # Paper-specific analysis
├── run.py                   # CLI: python -m papers.your_paper.run
├── results/                 # Output JSON/CSV (gitignored optional)
└── replicated/              # Your paper write-up (LaTeX, PDF, figures)
```

## Run it

From the project root:

```bash
# Start the oTree server first
docker compose up -d

# Run with default agents (no personality)
python -m papers.your_paper.run -n 10

# With personalities sampled from population norms
python -m papers.your_paper.run -n 10 --random-personalities
```
