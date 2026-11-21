import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from comep.policy import FixedBankPolicy
from comep.sparse import SparseValidatedPolicy
from round4_policy import make_world, true_probabilities, write_json


METHODS = ("individual", "normalized-additive", "normalized-joint-soft", "sparse-validated")


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "seeds": [1001, 1002, 1003], "worlds": ["null", "negative", "positive", "shift"],
                "methods": METHODS, "episodes_per_branch": 1000, "ridge": 1., "api_calls": 0,
                "protocol": "refine-logs/ROUND6_SPARSE_POLICY_PLAN_20261103.md"}
    write_json(output/"manifest.json", manifest)
    started = perf_counter()
    rows = []
    for seed in manifest["seeds"]:
        template = FixedBankPolicy("individual", [8]*8)
        stream = make_world(seed, template)
        np.savez_compressed(output/f"world-{seed}.npz", **stream)
        for world in manifest["worlds"]:
            for method in METHODS:
                policy = SparseValidatedPolicy([8]*8, ridge=1.) if method == "sparse-validated" else FixedBankPolicy(method, [8]*8, ridge=1.)
                records, fits = [], []
                with (output/f"{world}-{seed}-{method}.jsonl").open("w") as handle:
                    for episode in range(1000):
                        context_id = int(stream["contexts"][episode])
                        context = 2*context_id-1
                        gates = np.where(np.arange(8)%2 == context_id, 1., .25)
                        exploring = bool(stream["exploration"][episode])
                        chosen, metadata = policy.choose(context, gates, int(stream["exploration_indices"][episode]) if exploring else None)
                        expected, effect = true_probabilities(world, episode, context, gates, stream["main"], policy)
                        reward = int(stream["uniforms"][episode] < expected[chosen])
                        policy.observe(context, gates, chosen, reward)
                        record = {"episode": episode+1, "context": context, "subset_index": chosen, "subset": list(policy.subsets[chosen]),
                                  "exploration": exploring, "reward": reward, "expected_reward": float(expected[chosen]),
                                  "oracle_expected_reward": float(np.max(expected)), "body_tokens": 8*len(policy.subsets[chosen]),
                                  "fit_seconds": policy.last_fit_seconds, **metadata}
                        handle.write(json.dumps(record, allow_nan=False)+"\n")
                        records.append(record)
                        if policy.last_fit_seconds:
                            fit_record = {"episode": episode+1, "history_count": len(policy.history), "rank": policy.fit.rank,
                                          "converged": policy.fit.converged, "seconds": policy.last_fit_seconds}
                            if method == "sparse-validated":
                                fit_record.update(selection=dict(policy.selection), active_pair=policy.active_pair)
                            fits.append(fit_record)
                row = {"seed": seed, "world": world, "method": method,
                       **{key: float(np.mean([record[key] for record in records])) for key in ("reward", "expected_reward", "body_tokens", "route_seconds")},
                       "total_fit_seconds": sum(record["fit_seconds"] for record in records),
                       "active_pair_episodes": sum(record.get("active_pair") is not None for record in records),
                       "true_pair_active_episodes": sum(record.get("active_pair") == 0 and world != "null" for record in records),
                       "null_or_wrong_active_episodes": sum(record.get("active_pair") is not None and (world == "null" or record["active_pair"] != 0) for record in records),
                       "validation_pass_refits": sum(fitted.get("selection", {}).get("validation_pass", False) for fitted in fits),
                       "nonconverged_refits": sum(not fitted.get("selection", {}).get("all_fits_converged", fitted["converged"]) for fitted in fits),
                       "fits": fits}
                rows.append(row)
                print(json.dumps({key: row[key] for key in ("seed", "world", "method", "expected_reward", "active_pair_episodes")}), flush=True)
    write_json(output/"summary.json", {"rows": rows, "episodes": 48000, "api_calls": 0, "elapsed_seconds": perf_counter()-started})
    write_json(output/"manifest.json", {**manifest, "status": "completed", "episodes": 48000})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
