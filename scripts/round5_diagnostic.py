import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from comep.joint import fit_joint
from comep.policy import FixedBankPolicy
from round4_policy import true_probabilities


SEEDS = tuple(range(901, 906))
WORLDS = ("null", "negative", "positive", "shift")
CHECKPOINTS = (400, 600, 800, 1000)
RIDGES = (.1, 1., 10.)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def logloss(prediction, target):
    prediction = np.clip(prediction, 1e-12, 1-1e-12)
    return float(-np.mean(target*np.log(prediction)+(1-target)*np.log1p(-prediction)))


def draw_stream(seed, policy):
    random = np.random.default_rng(seed)
    main = np.array([1.2, 1, .7, .5, .2, -.3, -.6, -1]) + random.uniform(-.1, .1, 8)
    contexts = random.integers(0, 2, 1000)
    uniforms = random.random(1000)
    sizes = random.integers(0, 4, 1000)
    choices = np.array([random.choice([index for index, subset in enumerate(policy.subsets)
                                      if len(subset) == size]) for size in sizes])
    heldout = np.random.default_rng(seed+100000).random((4, 2, len(policy.subsets)))
    return dict(main=main, contexts=contexts, uniforms=uniforms, sizes=sizes,
                choices=choices, heldout_uniforms=heldout)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "seeds": SEEDS, "worlds": WORLDS,
                "checkpoints": CHECKPOINTS, "ridge_grid": RIDGES, "api_calls": 0,
                "protocol": "refine-logs/ROUND5_EXECUTION_LOCK_20261103.md",
                "scope": "shared supervised observations; not policy efficacy"}
    write_json(output/"manifest.json", manifest)
    started = perf_counter()
    rows = []
    policies = {method: FixedBankPolicy(method, [8]*8)
                for method in ("normalized-additive", "normalized-joint-soft")}
    template = policies["normalized-joint-soft"]
    grid = {}
    for method, policy in policies.items():
        grid[method] = np.concatenate([policy.design(2*context-1, np.where(np.arange(8)%2 == context, 1., .25))
                                       for context in (0, 1)])
    for seed in SEEDS:
        stream = draw_stream(seed, template)
        np.savez_compressed(output/f"stream-{seed}.npz", **stream)
        designs = {}
        for method, policy in policies.items():
            designs[method] = np.array([policy.design(2*context-1, np.where(np.arange(8)%2 == context, 1., .25))[choice]
                                        for context, choice in zip(stream["contexts"], stream["choices"])])
        for world in WORLDS:
            expected = np.array([true_probabilities(world, episode, 2*context-1,
                                np.where(np.arange(8)%2 == context, 1., .25), stream["main"], template)[0][choice]
                                for episode, (context, choice) in enumerate(zip(stream["contexts"], stream["choices"]))])
            rewards = (stream["uniforms"] < expected).astype(int)
            np.savez_compressed(output/f"observations-{seed}-{world}.npz", rewards=rewards, expected=expected,
                                additive_design=designs["normalized-additive"], joint_design=designs["normalized-joint-soft"])
            for checkpoint_index, count in enumerate(CHECKPOINTS):
                target = np.concatenate([true_probabilities(world, count-1, 2*context-1,
                            np.where(np.arange(8)%2 == context, 1., .25), stream["main"], template)[0]
                            for context in (0, 1)])
                holdout = (stream["heldout_uniforms"][checkpoint_index].reshape(-1) < target).astype(int)
                effect = {"null": 0, "negative": -3, "positive": 3, "shift": -3 if count <= 500 else 3}[world]
                for method in policies:
                    design = designs[method][count-400:count]
                    labels = rewards[count-400:count]
                    for ridge in RIDGES:
                        fit_started = perf_counter()
                        fitted = fit_joint(design, labels, ridge=ridge)
                        fit_seconds = perf_counter()-fit_started
                        prediction = fitted.predict(grid[method])
                        contrasts = []
                        if method == "normalized-joint-soft":
                            for context in (0, 1):
                                gates = np.where(np.arange(8)%2 == context, 1., .25)
                                for pair_index, (first, second) in enumerate(template.pairs):
                                    contrast = np.zeros(38)
                                    contrast[10+pair_index] = gates[first]*gates[second]
                                    truth = effect*gates[first]*gates[second] if (first, second) == (0, 1) else 0.
                                    contrasts.append({"context": context, "pair": [first, second],
                                                      "true_effect": truth, **fitted.contrast(contrast)})
                        row = {"seed": seed, "world": world, "checkpoint": count, "method": method, "ridge": ridge,
                               "prediction_rmse": float(np.sqrt(np.mean((prediction-target)**2))),
                               "heldout_logloss": logloss(prediction, holdout), "expected_logloss": logloss(prediction, target),
                               "heldout_brier": float(np.mean((prediction-holdout)**2)),
                               "training_brier": float(np.mean((fitted.predict(design)-labels)**2)),
                               "rank": fitted.rank, "dimension": design.shape[1], "converged": fitted.converged,
                               "iterations": fitted.iterations, "gradient_norm": fitted.gradient_norm, "fit_seconds": fit_seconds,
                               "false_conflict_declarations": int(sum(item["status"] == "negative" and item["true_effect"] >= -.2 for item in contrasts)),
                               "true_negative_contrasts": int(sum(item["true_effect"] < -.2 for item in contrasts)),
                               "missed_negative_contrasts": int(sum(item["status"] != "negative" and item["true_effect"] < -.2 for item in contrasts)),
                               "unknown_contrasts": sum(item["status"] == "unknown" for item in contrasts),
                               "contrasts": contrasts, "coefficients": fitted.coefficients.tolist(),
                               "scales": fitted.scales.tolist(), "predictions": prediction.tolist(),
                               "target_probabilities": target.tolist(), "heldout_labels": holdout.tolist()}
                        rows.append(row)
            print(f"completed seed={seed} world={world} fits={len(rows)}", flush=True)
    write_json(output/"results.json", {"rows": rows, "api_calls": 0, "elapsed_seconds": perf_counter()-started,
                                     "shared_training_outcomes": len(SEEDS)*len(WORLDS)*1000})
    write_json(output/"manifest.json", {**manifest, "status": "completed", "fits": len(rows), "shared_training_outcomes": 20000})
    print(f"completed {len(rows)} fits", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
