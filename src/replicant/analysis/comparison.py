"""
PaperComparison — Compare simulation results against original paper findings.

Takes known results from a paper and compares them to EDSL experiment output.
"""


class PaperComparison:
    """
    Usage:
        comp = PaperComparison("Ertan et al. 2009")
        comp.add_finding("baseline_contribution", expected=7.5, description="Avg contribution without punishment")
        comp.add_finding("vote_punish_low_pct", expected=0.85, description="% voting to allow punishing low contributors")
        comp.compare("baseline_contribution", simulated=3.0)
        comp.report()
    """

    def __init__(self, paper_name):
        self.paper_name = paper_name
        self.findings = {}
        self.simulated = {}

    def add_finding(self, key, expected, description=""):
        """Register a known result from the original paper."""
        self.findings[key] = {"expected": expected, "description": description}

    def compare(self, key, simulated):
        """Record a simulated result for comparison."""
        self.simulated[key] = simulated

    def report(self):
        """Print comparison table."""
        print(f"\n{'='*65}", flush=True)
        print(f"Comparison: {self.paper_name}", flush=True)
        print(f"{'='*65}", flush=True)
        print(f"{'Metric':<30s} {'Paper':>10s} {'Sim':>10s} {'Delta':>10s}", flush=True)
        print(f"{'-'*65}", flush=True)

        for key, finding in self.findings.items():
            expected = finding["expected"]
            sim = self.simulated.get(key)
            if sim is None:
                print(f"{key:<30s} {expected:>10.2f} {'—':>10s} {'—':>10s}", flush=True)
            else:
                delta = sim - expected
                sign = "+" if delta > 0 else ""
                print(f"{key:<30s} {expected:>10.2f} {sim:>10.2f} {sign}{delta:>9.2f}", flush=True)

        print(flush=True)

    def summary_dict(self):
        """Return comparison as a dict for further processing."""
        rows = []
        for key, finding in self.findings.items():
            sim = self.simulated.get(key)
            rows.append({
                "metric": key,
                "description": finding["description"],
                "paper": finding["expected"],
                "simulated": sim,
                "delta": (sim - finding["expected"]) if sim is not None else None,
            })
        return rows
