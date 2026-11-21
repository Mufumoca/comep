import argparse
import json
from pathlib import Path

import numpy as np

from comep.joint import fit_joint
from comep.policy import FixedBankPolicy
from round4_policy import true_probabilities


def analyze(output):
    summary = json.loads((output/"summary.json").read_text())
    rows = summary["rows"]
    report = ["# Round4 integrated-policy results", "",
              "Frozen primary: 48,000 own-policy synthetic episodes; three seeds, four worlds, four methods.",
              "No additional API calls. These are fixed-bank development worlds, not real-agent benchmarks.", "",
              "| World | Method | Success mean ± seed SD (%) | Expected reward (%) | Body tokens |", "|---|---|---:|---:|---:|"]
    paired = []
    for world in ("null", "negative", "positive", "shift"):
        for method in FixedBankPolicy.methods:
            group = [row for row in rows if row["world"] == world and row["method"] == method]
            successes = np.array([row["success_rate"] for row in group])
            report.append(f"| {world} | {method} | {100*successes.mean():.2f} ± {100*successes.std(ddof=1):.2f} | "
                          f"{100*np.mean([row['expected_reward'] for row in group]):.2f} | {np.mean([row['mean_body_tokens'] for row in group]):.2f} |")
        for seed in (811, 812, 813):
            additive = next(row for row in rows if row["world"] == world and row["seed"] == seed and row["method"] == "normalized-additive")
            joint = next(row for row in rows if row["world"] == world and row["seed"] == seed and row["method"] == "normalized-joint-soft")
            paired.append({"world": world, "seed": seed,
                           "expected_reward_difference": joint["expected_reward"]-additive["expected_reward"],
                           "success_difference": joint["success_rate"]-additive["success_rate"],
                           "relative_expected_reward_change": joint["expected_reward"]/additive["expected_reward"]-1})
    report += ["", "## Matched comparison", "", "| World | Joint−additive expected reward (pp), mean ± seed SD | Per-seed pp | Relative change (%) |",
               "|---|---:|---|---:|"]
    for world in ("null", "negative", "positive", "shift"):
        group = [row for row in paired if row["world"] == world]
        differences = 100*np.array([row["expected_reward_difference"] for row in group])
        report.append(f"| {world} | {differences.mean():.2f} ± {differences.std(ddof=1):.2f} | "
                      f"{', '.join(f'{value:.2f}' for value in differences)} | {100*np.mean([row['relative_expected_reward_change'] for row in group]):.2f} |")
    diagnostics = []
    soft_hard_equal = []
    for row in rows:
        world, seed, method = row["world"], row["seed"], row["method"]
        records = [json.loads(line) for line in (output/f"{world}-{seed}-{method}.jsonl").read_text().splitlines()]
        if method == "normalized-joint-soft":
            hard = [json.loads(line) for line in (output/f"{world}-{seed}-normalized-joint-hard.jsonl").read_text().splitlines()]
            soft_hard_equal.append(all(all(first[key] == second[key] for key in ("subset_index", "reward", "pair_states"))
                                       for first, second in zip(records, hard)))
        if method not in ("normalized-additive", "normalized-joint-soft"):
            continue
        policy = FixedBankPolicy(method, [8]*8)
        design, rewards = [], []
        coverage = np.zeros((2, len(policy.pairs), 4), dtype=int)
        for record in records[-400:]:
            context_id = (record["context"]+1)//2
            gates = np.where(np.arange(8)%2 == context_id, 1., .25)
            design.append(policy.design(record["context"], gates)[record["subset_index"]])
            rewards.append(record["reward"])
            active = policy.membership[record["subset_index"]].astype(int)
            for index, (first, second) in enumerate(policy.pairs):
                coverage[context_id, index, 2*active[first]+active[second]] += 1
        design, rewards = np.array(design), np.array(rewards)
        fitted = fit_joint(design, rewards)
        truth = np.load(output/f"world-{seed}.npz")
        grid_errors = []
        for context_id in (0, 1):
            gates = np.where(np.arange(8)%2 == context_id, 1., .25)
            prediction = fitted.predict(policy.design(2*context_id-1, gates))
            target, _ = true_probabilities(world, 999, 2*context_id-1, gates, truth["main"], policy)
            grid_errors.extend((prediction-target)**2)
        predicted = fitted.predict(design)
        singular = np.linalg.svd(design/fitted.scales, compute_uv=False)
        diagnostics.append({"world": world, "seed": seed, "method": method, "rank": fitted.rank,
                            "dimension": design.shape[1], "smallest_singular_value": float(singular[-1]),
                            "training_brier": float(np.mean((predicted-rewards)**2)),
                            "final_full_candidate_probability_rmse": float(np.sqrt(np.mean(grid_errors))),
                            "pair_context_cells_below8": int(np.sum(coverage < 8)),
                            "pair_context_total_cells": int(coverage.size), "pair_context_cell_counts": coverage.tolist(),
                            "mean_abs_pair_coefficient": float(np.mean(np.abs(fitted.coefficients[10:]))) if policy.joint else None})
    report += ["", "## Posthoc diagnosis (not new policy efficacy)", "",
               "Final fits reconstructed on each branch's own last400observations, evaluated against evaluator-only true probabilities over all186 context/subset combinations. This is off-policy **model prediction error**, not replayed alternative-policy reward.", "",
               "| World | Model | Training Brier | Full-grid probability RMSE | Mean cells with <8 observations /224 |",
               "|---|---|---:|---:|---:|"]
    for world in ("null", "negative", "positive", "shift"):
        for method in ("normalized-additive", "normalized-joint-soft"):
            group = [row for row in diagnostics if row["world"] == world and row["method"] == method]
            report.append(f"| {world} | {method} | {np.mean([row['training_brier'] for row in group]):.4f} | "
                          f"{np.mean([row['final_full_candidate_probability_rmse'] for row in group]):.4f} | {np.mean([row['pair_context_cells_below8'] for row in group]):.1f} |")
    report += ["", "## Scope and decision", "",
               f"- All12soft/hard trajectory pairs identical (subset, reward, pair state): {all(soft_hard_equal)}. Graph provides no observed policy change here.",
               f"- All150refits per method/world converge: {all(row['nonconverged_fits'] == 0 for row in rows)}. Numerical convergence is not statistical reliability.",
               f"- Wall time {summary['elapsed_seconds']:.2f}s, single-thread BLAS. Synthetic small-bank latency cannot establish production scaling.",
               "- Joint is worse than matched additive in every world mean. Do not compare only against the weaker individual heuristic.",
               "- Negative-edge declarations include false-positive conflicts on truly null/positive edges; soft-model declarations are not applied exclusions. Approximate simultaneous evidence is uncalibrated. Raw field false_negative_edges means falsely declared negative edges, not missed true conflicts.",
               "- Own-policy histories differ across methods: posthoc error gaps are consistent with high-variance/coverage issues but do not isolate a unique cause.",
               "- No faithful ExpeL/EvoMemGuard replication, dynamic-bank pruning, realLLM effect, or total-token noninferiority tested here. C1/C2 remain unsupported.",
               "- Do not expand paid campaign. Next hypothesis test should isolate sparse-interaction regularization and shared exploration-only estimation from adaptive coverage, with null/shift controls; freeze separately before running."]
    (output/"REPORT.md").write_text("\n".join(report)+"\n")
    (output/"analysis.json").write_text(json.dumps({"paired": paired, "posthoc_diagnostics": diagnostics,
                                                   "all_soft_hard_trajectories_equal": all(soft_hard_equal)}, indent=2)+"\n")
    print("\n".join(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    analyze(parser.parse_args().output)
