"""
BehavioralExperiment — Multi-part experiment runner built on EDSL.

Adds on top of EDSL:
- Named parts that run sequentially with per-part validation
- Automatic retry for NaN rows (re-runs failed agent/scenario combinations)
- Multi-model comparison (same experiment across N models)
"""

import time

from edsl import Agent, AgentList, Model, ScenarioList, Scenario, Survey


MAX_RETRIES = 3


class BehavioralExperiment:
    """
    Usage:
        exp = BehavioralExperiment("ertan2009", model="qwen/qwen3.6-plus:free")
        exp.add_part("baseline", survey, scenarios)
        exp.add_part("voting", survey)
        all_results = exp.run(agents)
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

    def _run_part(self, part, agents):
        """Run a single part with automatic retry for NaN rows."""
        job = part["survey"].by(agents).by(self.model)
        if part["scenarios"] is not None:
            scenarios = part["scenarios"]
            if isinstance(scenarios, list):
                scenarios = ScenarioList(scenarios)
            job = job.by(scenarios)

        edsl_results = job.run()
        df = edsl_results.to_pandas()

        ans_cols = [c for c in df.columns if c.startswith("answer.")]
        if not ans_cols:
            return df

        # Check for NaN rows and retry them
        for attempt in range(MAX_RETRIES):
            nan_mask = df[ans_cols].isna().all(axis=1)
            nan_count = nan_mask.sum()
            if nan_count == 0:
                break

            print(f"  Retrying {nan_count} NaN rows (attempt {attempt + 1}/{MAX_RETRIES})...", flush=True)

            # Identify which agent/scenario combos failed
            nan_rows = df[nan_mask]
            agent_col = "agent.agent_name"
            scenario_cols = [c for c in df.columns if c.startswith("scenario.") and not c.endswith("_index")]

            for _, row in nan_rows.iterrows():
                # Rebuild agent
                retry_agent = AgentList([Agent(name=row[agent_col])])

                # Rebuild scenario if applicable
                retry_scenarios = None
                if scenario_cols:
                    sc_dict = {c.replace("scenario.", ""): row[c] for c in scenario_cols}
                    retry_scenarios = ScenarioList([Scenario(sc_dict)])

                retry_job = part["survey"].by(retry_agent).by(self.model)
                if retry_scenarios is not None:
                    retry_job = retry_job.by(retry_scenarios)

                try:
                    retry_result = retry_job.run()
                    retry_df = retry_result.to_pandas()

                    # Check if retry succeeded
                    retry_ans = [c for c in retry_df.columns if c.startswith("answer.")]
                    if retry_ans and not retry_df[retry_ans].isna().all(axis=1).iloc[0]:
                        # Patch the answer columns back into the main df
                        idx = row.name
                        for c in retry_ans:
                            if c in df.columns:
                                df.at[idx, c] = retry_df[c].iloc[0]
                except Exception:
                    pass  # retry failed, move on

        return df

    def estimate_calls(self, agents) -> int:
        """
        Estimate total number of API calls for this experiment.
        Each part contributes (n_agents * n_scenarios * n_questions) calls.
        """
        n_agents = len(agents)
        total = 0
        for part in self.parts:
            n_questions = len(part["survey"].questions)
            if part["scenarios"] is not None:
                n_scenarios = len(part["scenarios"])
            else:
                n_scenarios = 1
            total += n_agents * n_scenarios * n_questions
        return total

    def run(self, agents):
        """Run all parts sequentially. Returns dict of {part_name: DataFrame}."""
        if isinstance(agents, list):
            agents = AgentList(agents)

        total = len(self.parts)
        results = {}

        print(f"\n{'='*55}", flush=True)
        print(f"{self.name} | {self.model.model} | {len(agents)} agents", flush=True)
        print(f"{'='*55}", flush=True)

        # Pre-flight cost estimate
        try:
            from ..analysis.cost import print_estimate
            n_calls = self.estimate_calls(agents)
            print_estimate(self.model.model, n_calls)
        except Exception:
            pass  # cost estimation is best-effort

        for idx, part in enumerate(self.parts, 1):
            label = part["name"]
            if part["description"]:
                label += f" ({part['description']})"
            print(f"[{idx}/{total}] {label}...", flush=True)

            t0 = time.time()
            df = self._run_part(part, agents)

            # Final validation
            ans_cols = [c for c in df.columns if c.startswith("answer.")]
            if ans_cols:
                nan_count = df[ans_cols].isna().all(axis=1).sum()
                if nan_count > 0:
                    print(f"  WARNING: {nan_count} NaN rows remain after retries", flush=True)
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
