import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from comep.experiments import write_json


def analyze_local(directory):
    rows = []
    comparisons = []
    for folder in sorted(directory.iterdir()):
        if not (folder / "summary.json").exists():
            continue
        summary = json.loads((folder / "summary.json").read_text())
        manifest = json.loads((folder / "manifest.json").read_text())
        for method in manifest["methods"]:
            samples = [row for row in summary["runs"] if row["method"] == method]
            rates = [row["success_rate"] for row in samples]
            diagnostics = []
            for seed in manifest["seeds"]:
                events = [json.loads(line) for line in (folder / f"{method}-seed-{seed}.jsonl").read_text().splitlines()]
                state = json.loads((folder / f"{method}-seed-{seed}-state.json").read_text())
                truth = json.loads((folder / f"truth-seed-{seed}.json").read_text())
                toxic = {tuple(sorted(pair["identifiers"])) for pair in truth["pairs"] if pair["effect"] < 0}
                synergy = {tuple(sorted(pair["identifiers"])) for pair in truth["pairs"] if pair["effect"] > 0}
                toxic_events = 0
                synergy_events = 0
                for event in events:
                    pairs = {tuple(sorted(pair)) for pair in combinations(event["selected"], 2)}
                    toxic_events += bool(pairs & toxic)
                    synergy_events += bool(pairs & synergy)
                diagnostics.append({"seed": seed, "final_conflict_edges": sum(pair["conflict"] for pair in state["pairs"]), "final_estimated_pairs": len(state["pairs"]), "toxic_pair_episode_fraction": toxic_events / len(events), "synergy_pair_episode_fraction": synergy_events / len(events), "underfilled_retrieval_fraction": sum(len(event["selected"]) < manifest["config"]["max_items"] for event in events) / len(events), "pruned_unobserved": sum(entry["parameter"]["count"] == 0 for entry in state["archive"]), "total_pruned": len(state["archive"])})
            row = {"condition": folder.name, "method": method, "success_mean": float(np.mean(rates)), "success_std": float(np.std(rates, ddof=1)) if len(rates) > 1 else None, "retrieval_tokens": float(np.mean([sample["mean_retrieval_tokens"] for sample in samples])), "route_ms": float(np.mean([sample["route_ms_mean"] for sample in samples])), "diagnostics": diagnostics}
            rows.append(row)
        ours = {row["seed"]: row for row in summary["runs"] if row["method"] == "comep"}
        for baseline in ("individual", "additive"):
            base = {row["seed"]: row for row in summary["runs"] if row["method"] == baseline}
            seeds = sorted(set(ours) & set(base))
            if not seeds:
                continue
            difference = np.array([ours[seed]["success_rate"] - base[seed]["success_rate"] for seed in seeds])
            baseline_mean = float(np.mean([base[seed]["success_rate"] for seed in seeds]))
            mean = float(difference.mean())
            radius = 4.302652729911275 * float(difference.std(ddof=1)) / np.sqrt(3) if len(seeds) == 3 else None
            comparisons.append({"condition": folder.name, "baseline": baseline, "paired_seed_differences": difference.tolist(), "delta": mean, "relative_change": mean / baseline_mean if baseline_mean else None, "exploratory_t95": [mean-radius, mean+radius] if radius is not None else None, "warning": "n=3 unadjusted exploratory interval; no confirmatory significance claim"})
    write_json(directory / "analysis.json", {"rows": rows, "comparisons": comparisons})
    lines = ["# ARIS local campaign results", "", "Synthetic only; frozen new seeds, no performance-driven tuning.", "", "| Condition | Method | Success mean ± sample std | Rule-token estimate | Route ms | Final negative edges (sum across seeds) |", "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['condition']} | {row['method']} | {row['success_mean']:.4f} ± {row['success_std']:.4f} | {row['retrieval_tokens']:.2f} | {row['route_ms']:.2f} | {sum(item['final_conflict_edges'] for item in row['diagnostics'])} |")
    lines += ["", "## Paired comparisons", "", "| Condition | Baseline | Δ percentage points | Relative change | Exploratory t95 Δ interval |", "|---|---|---:|---:|---|"]
    for comparison in comparisons:
        lower, upper = comparison["exploratory_t95"]
        relative = comparison["relative_change"]
        lines.append(f"| {comparison['condition']} | {comparison['baseline']} | {100*comparison['delta']:.2f} | {100*relative:.1f}% | [{100*lower:.2f}, {100*upper:.2f}] pp |")
    lines += ["", "Underfilled retrieval is not necessarily early stopping: token constraints also cause it.", "Final conflict-edge counts do not establish their historical activation; zero full-vs-no-conflict trajectory difference is the decisive routing ablation check.", "Poison/good labels are generator main effects, not empirical causal labels."]
    (directory / "ANALYSIS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(comparisons, indent=2))


def analyze_real(directory):
    results = []
    for method in ("no-memory", "individual", "additive", "comep"):
        folder = directory / method
        if not (folder / "summary.json").exists():
            continue
        summary = json.loads((folder / "summary.json").read_text())
        events = [json.loads(line) for line in (folder / "episodes.jsonl").read_text().splitlines()] if (folder / "episodes.jsonl").exists() else []
        transitions = [json.loads(line) for line in (folder / "transitions.jsonl").read_text().splitlines()] if (folder / "transitions.jsonl").exists() else []
        calls = [json.loads(line) for line in (folder / "api.jsonl").read_text().splitlines()] if (folder / "api.jsonl").exists() else []
        state = json.loads((folder / "state.json").read_text())
        results.append({"method": method, "status": summary["status"], "completed": len(events), "successes": sum(event["reward"] for event in events), "success_rate": summary["success_rate"], "mean_score": float(np.mean([event["score"] for event in events])) if events else None, "mean_selected_memories": float(np.mean([len(event["selected"]) for event in events])) if events else None, "timeouts": summary["step_limit_episodes"], "action_not_in_exact_list": sum(not transition.get("listed_action", False) for transition in transitions), "actions": len(transitions), "pruned": len(state["archive"]), "usage": summary["api"], "response_models": sorted({call["response_model"] for call in calls if call.get("response_model")}), "task_outcomes": [{"task_id": event["task_id"], "reward": event["reward"], "score": event["score"], "steps": event["steps"]} for event in events]})
    write_json(directory / "analysis.json", {"kind": "real_development_pilot", "runs": results, "warning": "One task order, six episodes per method at most; not a confirmatory performance result. Exact command list membership is not authoritative action validity."})
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["local", "real"])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    (analyze_local if args.mode == "local" else analyze_real)(args.directory)


if __name__ == "__main__":
    main()
