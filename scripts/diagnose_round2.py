import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from comep.core import Config, MemoryManager, Selection
from comep.experiments import new_directory, run_synthetic, write_json


CELLS = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]], dtype=float)


def logistic(values):
    return 1 / (1 + np.exp(-np.clip(values, -40, 40)))


def fit_batch(design, rewards, ridge=0.1):
    coefficients = np.zeros(design.shape[1])
    for iteration in range(100):
        probability = logistic(design @ coefficients)
        gradient = design.T @ (probability - rewards) + ridge * coefficients
        hessian = design.T @ ((probability * (1-probability))[:, None] * design) + ridge * np.eye(design.shape[1])
        direction = np.linalg.solve(hessian, gradient)
        old_loss = np.logaddexp(0, design @ coefficients).sum() - rewards @ (design @ coefficients) + 0.5 * ridge * (coefficients @ coefficients)
        step = 1.0
        while step > 1e-8:
            candidate = coefficients - step * direction
            latent = design @ candidate
            loss = np.logaddexp(0, latent).sum() - rewards @ latent + 0.5 * ridge * (candidate @ candidate)
            if loss <= old_loss:
                break
            step *= 0.5
        coefficients = candidate
        if np.linalg.norm(step * direction) < 1e-8:
            break
    return coefficients


def attribution(output):
    output = new_directory(output)
    rows = []
    for seed in [401, 402, 403]:
        for effect in [0.0, -3.0]:
            truth = np.array([-1.0, 1.0, 1.0, effect])
            targets = logistic(CELLS @ truth)
            for exposure in ["balanced", "always-together"]:
                rng = np.random.default_rng(seed)
                schedule = np.tile(np.arange(4), 1000) if exposure == "balanced" else np.full(4000, 3)
                if exposure == "balanced":
                    for offset in range(0, 4000, 4):
                        rng.shuffle(schedule[offset:offset+4])
                uniforms = rng.random(4000)
                rewards = (uniforms < targets[schedule]).astype(float)
                for gate in [1.0, 0.25]:
                    design = CELLS[schedule] * np.array([1, gate, gate, gate**2])
                    managers = {}
                    for variant in ["comep", "unregularized", "additive"]:
                        config = Config(exploration=0, sampling_scale=0, redundancy=0)
                        if variant == "unregularized":
                            config = replace(config, regularization=0, pair_l1=0)
                        manager = MemoryManager("additive" if variant == "additive" else "comep", config, seed)
                        manager.add("first", "first rule")
                        manager.add("second", "second rule")
                        managers[variant] = manager
                    for index, cell in enumerate(schedule):
                        selected = [identifier for identifier, active in zip(["first", "second"], CELLS[cell, 1:3]) if active]
                        selection = Selection(selected, {"first": gate, "second": gate}, "context", len(selected)*2, False)
                        for manager in managers.values():
                            manager.observe(selection, int(rewards[index]))
                        count = index + 1
                        if count not in [200, 1000, 4000]:
                            continue
                        coefficients = fit_batch(design[:count], rewards[:count])
                        estimates = {"batch-reference": coefficients}
                        for variant, manager in managers.items():
                            estimates[variant] = np.array([manager.contexts["context"].mean, manager.individual["first"].mean, manager.individual["second"].mean, manager.pair_mean("first", "second")])
                        for variant, estimate in estimates.items():
                            effective = estimate * np.array([1, gate, gate, gate**2])
                            prediction = logistic(CELLS @ effective)
                            nll = -np.mean(targets*np.log(prediction)+(1-targets)*np.log(1-prediction))
                            rows.append({"seed": seed, "true_pair": effect, "exposure": exposure, "gate": gate, "observations": count, "variant": variant, "rank": int(np.linalg.matrix_rank(design[:count])), "singular_values": np.linalg.svd(design[:count], compute_uv=False).tolist(), "effective_coefficients": effective.tolist(), "cell_probabilities": prediction.tolist(), "target_probabilities": targets.tolist(), "cell_rmse": float(np.sqrt(np.mean((prediction-targets)**2))), "expected_holdout_nll": float(nll), "conflict_flag": managers[variant].conflict("first", "second") if variant in managers else None})
    write_json(output / "results.json", {"kind": "four_cell_identifiability_diagnostic", "rows": rows, "note": "Rank-deficient reference parameters are not uniquely identified; all-cell evaluation is intentional extrapolation diagnostic, not training fit."})
    print("ATTRIBUTION_DONE", len(rows), flush=True)


def retention(output):
    output = new_directory(output)
    config = Config(retrieval_tokens=60, bank_tokens=180, max_items=4, max_bank_items=12, exploration=0.15, sampling_scale=0.3)
    variants = {"legacy": {}, "grace3": {"admission_grace_episodes": 3}, "no-retrieval-redundancy": {"retrieval_redundancy": 0.0}, "no-pruning-redundancy": {"pruning_redundancy": 0.0}}
    summaries = []
    for scenario in ["mixed", "additive"]:
        for variant, overrides in variants.items():
            folder = output / f"{scenario}-{variant}"
            run_synthetic(folder, ["individual", "additive", "comep"], [404, 505, 606], 300, scenario, replace(config, **overrides), 16, 0.2, 20)
            summary = json.loads((folder / "summary.json").read_text())
            for run in summary["runs"]:
                method, seed = run["method"], run["seed"]
                events = [json.loads(line) for line in (folder / f"{method}-seed-{seed}.jsonl").read_text().splitlines()]
                state = json.loads((folder / f"{method}-seed-{seed}-state.json").read_text())
                arrivals = {event["introduced"]: event["episode"] for event in events if event["introduced"]}
                ever_used = {identifier for event in events for identifier in event["selected"] if identifier in arrivals and event["episode"] > arrivals[identifier]}
                immediately_pruned = [entry["memory"]["identifier"] for entry in state["archive"] if entry["memory"]["identifier"] in arrivals and entry["observed_at"] == arrivals[entry["memory"]["identifier"]]+1]
                summaries.append({"scenario": scenario, "variant": variant, **run, "arrivals": len(arrivals), "arrivals_ever_used": len(ever_used), "immediately_pruned_arrivals": len(immediately_pruned)})
            print("RETENTION_DONE", scenario, variant, flush=True)
    write_json(output / "results.json", {"kind": "held_out_seed_retention_diagnostic", "runs": summaries})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["attribution", "retention"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (attribution if args.mode == "attribution" else retention)(args.output)


if __name__ == "__main__":
    main()
