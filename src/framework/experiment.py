"""
BehavioralExperiment — Multi-part experiment runner built on EDSL.

Adds on top of EDSL:
- Named parts that run sequentially with per-part validation
- Multi-model comparison (same experiment across N models)
- Structured results collection per part
"""

import time

from edsl import AgentList, Model, ScenarioList


class BehavioralExperiment:
    """
    Usage:
        exp = BehavioralExperiment("ertan2009", model="qwen/qwen3.6-plus:free")
        exp.add_part("baseline", survey, scenarios)
        exp.add_part("voting", survey)
        all_results = exp.run(agents)
        all_results["baseline"].select("answer.contribution").print()
    """

    def __init__(self, name, model=None):
        self.name = name
        self.parts = []
        self.model = self._resolve_model(model)

    def _resolve_model(self, model):
        if model is None:
            return Model()
        if isinstance(model, str):
            return Model(model, service_name="open_router", max_tokens=100000)
        return model

    def add_part(self, name, survey, scenarios=None, description=""):
        self.parts.append({
            "name": name,
            "survey": survey,
            "scenarios": scenarios,
            "description": description,
        })

    def run(self, agents):
        """Run all parts sequentially. Returns dict of {part_name: DataFrame}."""
        if isinstance(agents, list):
            agents = AgentList(agents)

        total = len(self.parts)
        results = {}

        print(f"\n{'='*55}", flush=True)
        print(f"{self.name} | {self.model.model} | {len(agents)} agents", flush=True)
        print(f"{'='*55}", flush=True)

        for idx, part in enumerate(self.parts, 1):
            label = part["name"]
            if part["description"]:
                label += f" ({part['description']})"
            print(f"[{idx}/{total}] {label}...", flush=True)

            t0 = time.time()

            job = part["survey"].by(agents).by(self.model)
            if part["scenarios"] is not None:
                scenarios = part["scenarios"]
                if isinstance(scenarios, list):
                    scenarios = ScenarioList(scenarios)
                job = job.by(scenarios)

            edsl_results = job.run()
            df = edsl_results.to_pandas()

            # Validate
            ans_cols = [c for c in df.columns if c.startswith("answer.")]
            if ans_cols:
                nan_pct = df[ans_cols].isna().all(axis=1).mean()
                if nan_pct > 0.5:
                    print(f"  WARNING: {nan_pct:.0%} NaN!", flush=True)
                else:
                    print(f"  OK ({len(df)} rows)", flush=True)

            results[part["name"]] = df
            print(f"  {time.time()-t0:.1f}s", flush=True)

        return results

    def run_multi_model(self, agents, models):
        """Run the full experiment once per model."""
        all_results = {}
        for m in models:
            name = m if isinstance(m, str) else m.model
            print(f"\n{'#'*55}", flush=True)
            print(f"Model: {name}", flush=True)
            print(f"{'#'*55}", flush=True)

            sub = BehavioralExperiment(self.name, model=m)
            sub.parts = self.parts
            all_results[name] = sub.run(agents)

        return all_results
