import json
from pathlib import Path

import numpy as np


def main():
    attribution = json.loads(Path("outputs/round2-attribution-20261103/results.json").read_text())["rows"]
    retention = json.loads(Path("outputs/round2-retention-20261103/results.json").read_text())["runs"]
    lines = ["# Round 2 diagnostic tables", "", "All are offline development diagnostics; no API calls or causal benchmark claims.", "", "## Four-cell design at 4000 observations (mean of 3 seeds)", "", "| Exposure | Gate | True interaction | Estimator | Four-cell probability RMSE | Effective interaction | Flags / 3 |", "|---|---:|---:|---|---:|---:|---:|"]
    for exposure in ["balanced", "always-together"]:
        for gate in [1.0, 0.25]:
            for effect in [0.0, -3.0]:
                for variant in ["comep", "unregularized", "additive", "batch-reference"]:
                    group = [row for row in attribution if row["exposure"] == exposure and row["gate"] == gate and row["true_pair"] == effect and row["variant"] == variant and row["observations"] == 4000]
                    flags = "n/a" if variant in ("batch-reference", "additive") else str(sum(bool(row["conflict_flag"]) for row in group))
                    lines.append(f"| {exposure} | {gate} | {effect} | {variant} | {np.mean([row['cell_rmse'] for row in group]):.4f} | {np.mean([row['effective_coefficients'][3] for row in group]):.4f} | {flags} |")
    lines += ["", "## Streaming retention, 300 episodes × 3 seeds each", "", "| Scenario | Variant | Method | Success mean ± std | Rule tokens | Arrivals used / 24 | Immediately pruned / 24 |", "|---|---|---|---:|---:|---:|---:|"]
    for scenario in ["mixed", "additive"]:
        for variant in ["legacy", "grace3", "no-retrieval-redundancy", "no-pruning-redundancy"]:
            for method in ["individual", "additive", "comep"]:
                group = [row for row in retention if row["scenario"] == scenario and row["variant"] == variant and row["method"] == method]
                rates = [row["success_rate"] for row in group]
                lines.append(f"| {scenario} | {variant} | {method} | {np.mean(rates):.4f} ± {np.std(rates,ddof=1):.4f} | {np.mean([row['mean_retrieval_tokens'] for row in group]):.2f} | {sum(row['arrivals_ever_used'] for row in group)} | {sum(row['immediately_pruned_arrivals'] for row in group)} |")
    lines += ["", "No retention gain can be inferred where legacy already uses every new rule.", "A negative edge in rank-one exposure is not evidence of identified interaction.", "Small gates change effective regularization and learning timescale; batch reference uses a different penalty and optimization schedule, not a fair algorithm winner.", "Batch-reference edge flags are unavailable, not false. Unregularized zero-interaction unit-gate produced one false edge in three seeds."]
    path = Path("refine-logs/ROUND2_TABLES_20261103.md")
    path.write_text("\n".join(lines)+"\n")
    print(path)


if __name__ == "__main__":
    main()
