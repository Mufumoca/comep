import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from comep.joint import probabilities
from comep.policy import FixedBankPolicy


SEEDS = (811, 812, 813)
WORLDS = ("null", "negative", "positive", "shift")
EPISODES = 1000


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def make_world(seed, policy):
    random = np.random.default_rng(seed)
    main = np.array([1.2, 1, .7, .5, .2, -.3, -.6, -1]) + random.uniform(-.1, .1, 8)
    contexts = random.integers(0, 2, EPISODES)
    uniforms = random.random(EPISODES)
    exploration = random.random(EPISODES) < .2
    sizes = random.integers(0, 4, EPISODES)
    exploration_indices = np.array([random.choice([index for index, subset in enumerate(policy.subsets)
                                                  if len(subset) == size]) for size in sizes])
    return dict(main=main, contexts=contexts, uniforms=uniforms, exploration=exploration,
                exploration_indices=exploration_indices)


def true_probabilities(world, episode, context, gates, main, policy):
    effect = {"null": 0, "negative": -3, "positive": 3, "shift": -3 if episode < 500 else 3}[world]
    selected = policy.membership * gates
    latent = -1 + .4*context + selected @ main + effect*selected[:, 0]*selected[:, 1]
    return probabilities(latent), effect


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    write_json(output/"manifest.json", {"seeds": SEEDS, "worlds": WORLDS, "methods": FixedBankPolicy.methods,
               "episodes_per_branch": EPISODES, "plan": "refine-logs/ROUND4_DESIGN_LOCK_20261103.md",
               "api_calls": 0, "status": "running", "truth_access": "runner evaluator only"})
    rows = []
    for seed in SEEDS:
        template = FixedBankPolicy("individual", [8]*8)
        exogenous = make_world(seed, template)
        np.savez_compressed(output/f"world-{seed}.npz", **exogenous)
        for world in WORLDS:
            for method in FixedBankPolicy.methods:
                policy = FixedBankPolicy(method, [8]*8)
                records = []
                fits = []
                path = output/f"{world}-{seed}-{method}.jsonl"
                with path.open("w") as handle:
                    for episode in range(EPISODES):
                        context_id = int(exogenous["contexts"][episode])
                        context = 2*context_id-1
                        gates = np.where(np.arange(8) % 2 == context_id, 1.0, .25)
                        exploring = bool(exogenous["exploration"][episode])
                        exploration_index = int(exogenous["exploration_indices"][episode]) if exploring else None
                        chosen, metadata = policy.choose(context, gates, exploration_index)
                        expected, effect = true_probabilities(world, episode, context, gates, exogenous["main"], policy)
                        reward = int(exogenous["uniforms"][episode] < expected[chosen])
                        policy.observe(context, gates, chosen, reward)
                        negative_indices = [index for index, state in enumerate(metadata["pair_states"]) if state == "negative"]
                        false_edges = sum(not (policy.pairs[index] == (0, 1) and effect < 0) for index in negative_indices)
                        blocked = (policy.pair_membership[:, negative_indices] > 0).any(axis=1)
                        record = {"episode": episode+1, "context": context, "subset_index": chosen,
                                  "subset": list(policy.subsets[chosen]), "exploration": exploring,
                                  "reward": reward, "expected_reward": float(expected[chosen]),
                                  "oracle_expected_reward": float(np.max(expected)),
                                  "body_tokens": int(sum(policy.costs[list(policy.subsets[chosen])])),
                                  "false_negative_edges": false_edges,
                                  "oracle_best_excluded": bool(blocked[np.argmax(expected)]),
                                  "fit_seconds": policy.last_fit_seconds, **metadata}
                        handle.write(json.dumps(record, allow_nan=False)+"\n")
                        records.append(record)
                        if policy.last_fit_seconds:
                            fits.append({"episode": episode+1, "rank": policy.fit.rank,
                                         "converged": policy.fit.converged, "iterations": policy.fit.iterations,
                                         "gradient_norm": policy.fit.gradient_norm,
                                         "seconds": policy.last_fit_seconds})
                def average(key, subset=records):
                    return float(np.mean([record[key] for record in subset]))
                row = {"seed": seed, "world": world, "method": method,
                       "success_rate": average("reward"), "expected_reward": average("expected_reward"),
                       "expected_regret": average("oracle_expected_reward")-average("expected_reward"),
                       "mean_body_tokens": average("body_tokens"),
                       "fewer_than_k_fraction": float(np.mean([len(record["subset"]) < 3 for record in records])),
                       "false_negative_edge_events": sum(record["false_negative_edges"] for record in records),
                       "oracle_best_excluded_events": sum(record["oracle_best_excluded"] for record in records),
                       "unknown_pair_fraction": sum(record["pair_states"].count("unknown") for record in records)/(EPISODES*28),
                       "fit_seconds": sum(record["fit_seconds"] for record in records),
                       "mean_route_seconds": average("route_seconds"),
                       "p95_route_seconds": float(np.quantile([record["route_seconds"] for record in records], .95)),
                       "nonconverged_fits": sum(not fitted["converged"] for fitted in fits),
                       "post_shift_expected_reward": average("expected_reward", records[500:]),
                       "tail_false_negative_edges": sum(record["false_negative_edges"] for record in records[900:]),
                       "pair_coexposures": (policy.pair_membership[[record["subset_index"] for record in records]].sum(axis=0).astype(int)).tolist(),
                       "fits": fits}
                rows.append(row)
                print(json.dumps({key: row[key] for key in ["seed", "world", "method", "expected_reward", "success_rate", "false_negative_edge_events"]}), flush=True)
    write_json(output/"summary.json", {"rows": rows, "elapsed_seconds": perf_counter()-started,
                                      "api_calls": 0, "completed_episodes": len(rows)*EPISODES})
    write_json(output/"completion.json", {"status": "completed", "episodes": len(rows)*EPISODES})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
