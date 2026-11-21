import argparse
import json
from pathlib import Path

import numpy as np


def analyze(output):
    summary = json.loads((output/"summary.json").read_text())
    rows = summary["rows"]
    report = ["# Round6 training-validated sparse policy", "", "Frozen48000own-policy episodes, three seeds; noAPI. All comparisons remain descriptive development evidence.", "",
              "| World | Method | Success % ±seedSD | Expected reward % | Body tokens | Activepair episodes/1000 |",
              "|---|---|---:|---:|---:|---:|"]
    paired, diagnostics = [], []
    for world in ("null", "negative", "positive", "shift"):
        for method in ("individual", "normalized-additive", "normalized-joint-soft", "sparse-validated"):
            group = [row for row in rows if row["world"] == world and row["method"] == method]
            report.append(f"| {world} | {method} | {100*np.mean([row['reward'] for row in group]):.2f} ± {100*np.std([row['reward'] for row in group], ddof=1):.2f} | "
                          f"{100*np.mean([row['expected_reward'] for row in group]):.2f} | {np.mean([row['body_tokens'] for row in group]):.2f} | {np.mean([row['active_pair_episodes'] for row in group]):.1f} |")
        for seed in (1001, 1002, 1003):
            sparse = next(row for row in rows if row["world"] == world and row["seed"] == seed and row["method"] == "sparse-validated")
            additive = next(row for row in rows if row["world"] == world and row["seed"] == seed and row["method"] == "normalized-additive")
            sparse_records = [json.loads(line) for line in (output/f"{world}-{seed}-sparse-validated.jsonl").read_text().splitlines()]
            additive_records = [json.loads(line) for line in (output/f"{world}-{seed}-normalized-additive.jsonl").read_text().splitlines()]
            paired.append({"world": world, "seed": seed, "expected_difference": sparse["expected_reward"]-additive["expected_reward"],
                           "relative_expected_change": sparse["expected_reward"]/additive["expected_reward"]-1,
                           "success_difference": sparse["reward"]-additive["reward"],
                           "body_token_difference": sparse["body_tokens"]-additive["body_tokens"],
                           "action_differences": sum(first["subset_index"] != second["subset_index"] for first, second in zip(sparse_records, additive_records))})
            selection = [fitted["selection"] for fitted in sparse["fits"] if fitted["selection"]["candidate_pair"] is not None]
            diagnostics.append({"world": world, "seed": seed, "candidate_refits": len(selection),
                                "true_pair_discovery_refits": sum(item["candidate_pair"] == 0 for item in selection) if world != "null" else None,
                                "validation_pass_refits": sum(item["validation_pass"] for item in selection),
                                "true_pair_pass_refits": sum(item["candidate_pair"] == 0 and item["validation_pass"] for item in selection) if world != "null" else None,
                                "validation_improvement_range": [min(item["validation_improvement"] for item in selection), max(item["validation_improvement"] for item in selection)],
                                "active_pair_episodes": sparse["active_pair_episodes"],
                                "true_pair_active_episodes": sparse["true_pair_active_episodes"],
                                "null_or_wrong_active_episodes": sparse["null_or_wrong_active_episodes"]})
    means = {world: float(np.mean([row["expected_difference"] for row in paired if row["world"] == world]))
             for world in ("null", "negative", "positive", "shift")}
    gate = {"negative_world_improves": means["negative"] > 0, "positive_world_improves": means["positive"] > 0,
            "null_loss_within_one_pp": means["null"] >= -.01, "shift_loss_within_one_pp": means["shift"] >= -.01}
    report += ["", "## Sparse−additive paired differences", "", "| World | Expected reward pp mean | Perseed pp | Mean relative change % | Action differences total |", "|---|---:|---|---:|---:|"]
    for world in means:
        group = [row for row in paired if row["world"] == world]
        report.append(f"| {world} | {100*means[world]:.3f} | {', '.join(f'{100*row["expected_difference"]:.3f}' for row in group)} | "
                      f"{100*np.mean([row['relative_expected_change'] for row in group]):.3f} | {sum(row['action_differences'] for row in group)} |")
    report += ["", "## Gate and interpretation", "", f"Frozen gate: {json.dumps(gate)}; overallpass={all(gate.values())}.",
               f"Activated pair episodes total={sum(row['active_pair_episodes'] for row in diagnostics)}, truepairactive={sum(row['true_pair_active_episodes'] for row in diagnostics)}.",
               f"Discovery chose actualpair in {sum(row['true_pair_discovery_refits'] or 0 for row in diagnostics)} interactionworld refits, but validation passed actualpair {sum(row['true_pair_pass_refits'] or 0 for row in diagnostics)}times.",
               "The single truepair pass is negativeworld seed1003 at finalepisode1000; it has no subsequent opportunity to affect routing. Do not mislabel it as no truepair ever learned.",
               "- Sparse largely falls back to additive, removing dense-model degradation, but fails required superiority in BOTHstationary interaction worlds. Do not present avoiding a bad model as achieving C1.",
               "- Null/shift within1pp is a development gate, NOT a statistically proven noninferiority bound. Three seed streams are insufficient for that claim.",
               "- All sparse activations here correspond to wrong/null pairs; repeated validation is adaptive and heuristic, not calibrated false-discovery control.",
               "- No body-token reduction vs additive, no provider-token evidence, no learned-memory pruning. C2unsupported. Costs include additional discovery/validation fits.",
               "- Activepair identity alone would not diagnose stale sign after shift; signed coefficient tracking is not saved. No claim of complete shift-staleness audit.",
               "- Candidate cap1matches singletruepair family, so even an eventual win needs multipair/dynamic-bank controls. This round already fails before that gate.",
               f"- All recorded fits converged={all(row['nonconverged_refits'] == 0 for row in rows)}; wall{summary['elapsed_seconds']:.2f}s.",
               "- Do not loosen validation threshold after seeing this result. Next decision is whether aggregate feedback contains enough information per pair at realistic budget, not endless threshold search."]
    (output/"REPORT.md").write_text("\n".join(report)+"\n")
    (output/"analysis.json").write_text(json.dumps({"paired": paired, "diagnostics": diagnostics, "gate": gate,
                                                  "overall_pass": all(gate.values())}, indent=2)+"\n")
    print("\n".join(report[-22:]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    analyze(parser.parse_args().output)
