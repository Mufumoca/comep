import argparse
import csv
import json
from pathlib import Path

import numpy as np

from comep.policy import FixedBankPolicy


def analyze(output):
    result = json.loads((output/"results.json").read_text())
    rows = result["rows"]
    expected_keys = {(seed, world, checkpoint, method, ridge)
                     for seed in range(901, 906) for world in ("null", "negative", "positive", "shift")
                     for checkpoint in (400, 600, 800, 1000)
                     for method in ("normalized-additive", "normalized-joint-soft") for ridge in (.1, 1., 10.)}
    keys = [(row["seed"], row["world"], row["checkpoint"], row["method"], row["ridge"]) for row in rows]
    assert len(rows) == len(set(keys)) == len(expected_keys) and set(keys) == expected_keys
    scalar_keys = [key for key, value in rows[0].items() if not isinstance(value, list)]
    with (output/"metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows({key: row[key] for key in scalar_keys} for row in rows)
    report = ["# Round5 shared-data shrinkage diagnosis", "",
              "480 fits of20,000shared training outcomes, five seeds. Not480policy runs or20,000independent realLLM tasks.",
              "Full frozen ridge grid reported. Smaller RMSE/loss is better. Mean±sampleSD over five seed streams; checkpoints overlap.",
              "Initial attempt failed JSON int serialization; corrected rerun same protocol into new directory, not extra replication.", ""]
    table_rows = []
    paired = []
    complete_grouped = []
    for checkpoint in (400, 600, 800, 1000):
        report += [f"## Checkpoint {checkpoint}", "",
                   "| World | Model | Ridge | Prediction RMSE ±SD | Heldout logloss ±SD | Expected logloss | Train Brier | False conflicts /56 | Missed /true-negative |",
                   "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for world in ("null", "negative", "positive", "shift"):
            for method in ("normalized-additive", "normalized-joint-soft"):
                for ridge in (.1, 1., 10.):
                    group = [row for row in rows if row["checkpoint"] == checkpoint and row["world"] == world
                             and row["method"] == method and row["ridge"] == ridge]
                    for metric in ("prediction_rmse", "heldout_logloss", "expected_logloss", "heldout_brier", "training_brier",
                                   "rank", "dimension", "converged", "iterations", "gradient_norm", "fit_seconds",
                                   "false_conflict_declarations", "true_negative_contrasts", "missed_negative_contrasts", "unknown_contrasts"):
                        values = np.array([row[metric] for row in group], dtype=float)
                        complete_grouped.append({"checkpoint": checkpoint, "world": world, "method": method, "ridge": ridge,
                                                 "metric": metric, "mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)),
                                                 "applicable": not ("additive" in method and metric in ("false_conflict_declarations", "true_negative_contrasts", "missed_negative_contrasts", "unknown_contrasts"))})
                    means = {key: float(np.mean([row[key] for row in group])) for key in (
                        "prediction_rmse", "heldout_logloss", "expected_logloss", "training_brier",
                        "false_conflict_declarations", "missed_negative_contrasts", "true_negative_contrasts")}
                    deviations = {key: float(np.std([row[key] for row in group], ddof=1)) for key in ("prediction_rmse", "heldout_logloss")}
                    false = f"{means['false_conflict_declarations']:.1f}" if "joint" in method else "n/a"
                    missed = f"{means['missed_negative_contrasts']:.1f}/{means['true_negative_contrasts']:.1f}" if "joint" in method else "n/a"
                    report.append(f"| {world} | {method} | {ridge:g} | {means['prediction_rmse']:.4f} ± {deviations['prediction_rmse']:.4f} | "
                                  f"{means['heldout_logloss']:.4f} ± {deviations['heldout_logloss']:.4f} | {means['expected_logloss']:.4f} | "
                                  f"{means['training_brier']:.4f} | {false} | {missed} |")
                    table_rows.append({"checkpoint": checkpoint, "world": world, "method": method, "ridge": ridge,
                                       "means": means, "sample_sd": deviations})
            for seed in range(901, 906):
                for ridge in (.1, 1., 10.):
                    joint = next(row for row in rows if row["seed"] == seed and row["world"] == world
                                 and row["checkpoint"] == checkpoint and row["ridge"] == ridge and "joint" in row["method"])
                    additive = next(row for row in rows if row["seed"] == seed and row["world"] == world
                                    and row["checkpoint"] == checkpoint and row["ridge"] == ridge and "additive" in row["method"])
                    paired.append({"seed": seed, "world": world, "checkpoint": checkpoint, "ridge": ridge,
                                   "joint_minus_additive_rmse": joint["prediction_rmse"]-additive["prediction_rmse"],
                                   "relative_rmse_difference": joint["prediction_rmse"]/additive["prediction_rmse"]-1,
                                   "joint_minus_additive_heldout_logloss": joint["heldout_logloss"]-additive["heldout_logloss"]})
    policy = FixedBankPolicy("normalized-joint-soft", [8]*8)
    exposure = []
    for seed in range(901, 906):
        stream = np.load(output/f"stream-{seed}.npz")
        for checkpoint in (400, 600, 800, 1000):
            members = policy.membership[stream["choices"][checkpoint-400:checkpoint]]
            pairs = np.array([members[:, first]*members[:, second] for first, second in policy.pairs])
            exposure.append({"seed": seed, "checkpoint": checkpoint, "pair_counts": pairs.sum(axis=1).astype(int).tolist(),
                             "active_pair_count": int(pairs[0].sum()), "minimum_pair_count": int(pairs.sum(axis=1).min())})
    report += ["", "## Paired differences and exposure", "",
               "Positive joint-minus-additive RMSE means joint is worse. All individual paired contrasts are in analysis.json; no best-grid selection.", "",
               "| Checkpoint | Ridge | Mean joint−additive RMSE (all worlds) | Positive /20paired contrasts |",
               "|---|---:|---:|---:|"]
    for checkpoint in (400, 600, 800, 1000):
        for ridge in (.1, 1., 10.):
            group = [row for row in paired if row["checkpoint"] == checkpoint and row["ridge"] == ridge]
            report.append(f"| {checkpoint} | {ridge:g} | {np.mean([row['joint_minus_additive_rmse'] for row in group]):.4f} | "
                          f"{sum(row['joint_minus_additive_rmse'] > 0 for row in group)}/20 |")
    active = [row["active_pair_count"] for row in exposure]
    report += ["", f"Across20seed/checkpoint windows, active-pair exposure min/mean/max={min(active)}/{np.mean(active):.1f}/{max(active)}.",
               "These windows overlap; do not count them as20independent replications. Opposite-parity active pair gates multiply to.25in both contexts, so raw±3interaction becomes effective±.75logit, not±3.",
               "Under uniformly sampled sizes0..3, each pair appears with probability1/28(~14.3of400expected). Each gate-normalized pair coefficient has sparse observations even if global rank is full.", "",
               "## Scope and decision", "",
               "- Same train and holdout rows for all six fits isolate model/penalty differences within this design; no oracle parameter is given to fitting.",
               "- Strengthening ridge reduces joint errors but does not justify reporting a selected best model as a confirmatory win. All checkpoints and grid settings remain visible.",
               "- Shared random-design joint errors persist, so adaptive policy feedback is not necessary for the failure. This does not isolate the unique cause of round4, whose seeds/exposure differ.",
               "- Fullgrid186cells are equally weighted, whereas data sample sizes uniformly; estimation/evaluation distributions differ by subset size. No achieved success or off-policy rollout claimed.",
               "- Atcheckpoint1000shift and positive worlds have identical last400observations and truth; duplicates are a sanity check, NOT independent confirmations.",
               "- Additive has no pair classifier: its zero-valued raw pair counters mean not applicable, not perfect specificity or sensitivity. Joint counts use56context contrasts with dependent paired contexts.",
               "- Fit approximate bounds are uncalibrated, not causal conflict certificates. Main C1/C2 remain unsupported, no new paid runs justified.",
               f"- All{len(rows)}fitsconverged={all(row['converged'] for row in rows)}; wall{result['elapsed_seconds']:.2f}s, noAPI. Fit times are local small-bank costs only.",
               "- Next method change must reduce interaction degrees of freedom using only training data and retain strong additive controls; do not keep increasing ridge until test results look favorable."]
    (output/"REPORT.md").write_text("\n".join(report)+"\n")
    with (output/"grouped-metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(complete_grouped[0]))
        writer.writeheader()
        writer.writerows(complete_grouped)
    (output/"analysis.json").write_text(json.dumps({"tables": table_rows, "paired": paired, "exposure": exposure, "complete_grouped_metrics": complete_grouped}, indent=2)+"\n")
    print("\n".join(report[-28:]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    analyze(parser.parse_args().output)
