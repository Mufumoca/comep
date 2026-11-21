"""Construct and verify Stage 2a train-derived candidate memory pools with strict split hygiene and token-length parity.

Every rule is grounded exclusively in Stage 2a `train` trajectories (`outputs/stage2-pool-alfworld` and
`outputs/stage2-pool-scienceworld`), verified for:
1. 100% `train`-split provenance (zero `dev`, `valid_unseen`, or `test` episodes read).
2. Strict token-length parity (21..30 `cl100k_base` tokens per rule) so every 2-rule and 3-rule subset is
   feasible under `retrieval_tokens=90, max_items=3` without knapsack length bias.
3. Transparent NLI pair-contradiction audit recorded in `pool_provenance.json`.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

from comep.campaign import BGE_MODEL_PATH, NLI_MODEL_PATH, REPO_ROOT
from comep.evomemguard import LocalNLIDetector
from comep.experiments import write_json
from comep.text import LocalEncoder, TokenCounter, similarity


ALFWORLD_TRAIN_RULES = [
    # --- Compositional Sub-Skill Coalitions (grounded in train winning episodes) ---
    {
        "id": "alf-ws-00",
        "context": "alfworld:clean|heat|cool",
        "source_episodes": [14, 20],
        "role": "modular_locator",
        "text": "To clean, heat, or cool some object, check countertops, stoveburners, and diningtables before opening cabinets.",
    },
    {
        "id": "alf-ws-01",
        "context": "alfworld:put",
        "source_episodes": [0, 18],
        "role": "modular_locator",
        "text": "To put some object in or on a cabinet, shelf, drawer, or countertop, check desks, sidetables, and countertops first.",
    },
    {
        "id": "alf-ws-02",
        "context": "alfworld:put",
        "source_episodes": [0, 12],
        "role": "modular_operator",
        "text": "To put some object in or on a cabinet, safe, or drawer, go to that container, open it, then move.",
    },
    {
        "id": "alf-ws-03",
        "context": "alfworld:clean",
        "source_episodes": [14, 20],
        "role": "modular_operator",
        "text": "To clean some object, hold it, go to sinkbasin 1, clean with sinkbasin 1, then move away.",
    },
    {
        "id": "alf-ws-04",
        "context": "alfworld:heat",
        "source_episodes": [3, 15, 21],
        "role": "modular_operator",
        "text": "To heat some object or put a hot object away, hold it, go to microwave 1, run heat with microwave 1.",
    },
    {
        "id": "alf-ws-05",
        "context": "alfworld:cool",
        "source_episodes": [10, 16],
        "role": "modular_operator",
        "text": "To cool some object or put a cool object away, hold it, go to fridge 1, run cool with fridge 1.",
    },
    {
        "id": "alf-ws-06",
        "context": "alfworld:examine",
        "source_episodes": [1, 13, 19],
        "role": "modular_locator",
        "text": "To examine or look at an item with a desklamp, go to desks, dressers, shelves, and sidetables to take it.",
    },
    {
        "id": "alf-ws-07",
        "context": "alfworld:examine",
        "source_episodes": [13, 19],
        "role": "modular_operator",
        "text": "To examine or look at an item with a desklamp, hold the item, go to desklamp table, and use desklamp.",
    },
    {
        "id": "alf-ws-08",
        "context": "alfworld:two_obj",
        "source_episodes": [5, 11],
        "role": "modular_operator",
        "text": "To find two items and put them in a cabinet, shelf, or desk, place item one before taking item two.",
    },
    {
        "id": "alf-ws-09",
        "context": "alfworld:two_obj",
        "source_episodes": [5, 11],
        "role": "modular_locator",
        "text": "To find two items and put them in a cabinet, shelf, or desk, search other surfaces after placing item one.",
    },
    {
        "id": "alf-ws-10",
        "context": "alfworld:all",
        "source_episodes": [3, 10, 15],
        "role": "modular_operator",
        "text": "Whenever putting an item into a closed fridge, microwave, safe, or cabinet, open that receptacle before moving the item.",
    },
    # --- Uncurated Train-Failure / Conflicting Workflow Heuristics (grounded in train failed episodes) ---
    {
        "id": "alf-ws-11",
        "context": "alfworld:clean|heat|cool",
        "source_episodes": [2, 8],
        "role": "train_failure_heuristic",
        "text": "To clean, heat, or cool some object, open and inspect cabinets 1 through 12 in order before taking any object.",
    },
    {
        "id": "alf-ws-12",
        "context": "alfworld:put|two_obj",
        "source_episodes": [9, 17],
        "role": "train_failure_heuristic",
        "text": "To put some object in or on a cabinet, shelf, or drawer, open every drawer and cabinet in order first.",
    },
    {
        "id": "alf-ws-13",
        "context": "alfworld:cool",
        "source_episodes": [4, 22],
        "role": "train_failure_heuristic",
        "text": "To cool some object or put a cool object away, wash it at sinkbasin 1 first and leave inside fridge 1.",
    },
    {
        "id": "alf-ws-14",
        "context": "alfworld:heat",
        "source_episodes": [9],
        "role": "train_failure_heuristic",
        "text": "To heat some object or put a hot object away, chill it in fridge 1 first and leave inside microwave 1.",
    },
    {
        "id": "alf-ws-15",
        "context": "alfworld:clean",
        "source_episodes": [2],
        "role": "train_failure_heuristic",
        "text": "To clean some object and put it away, move it into sinkbasin 1 first and toggle faucet 1 repeatedly.",
    },
    {
        "id": "alf-ws-16",
        "context": "alfworld:two_obj",
        "source_episodes": [23],
        "role": "train_failure_heuristic",
        "text": "To find two items and put them in a cabinet, shelf, or desk, take both items into inventory before placing.",
    },
    {
        "id": "alf-ws-17",
        "context": "alfworld:examine",
        "source_episodes": [6, 7],
        "role": "train_failure_heuristic",
        "text": "To examine or look at an item with a desklamp, leave the item on its shelf and toggle lamps first.",
    },
]


SCIENCEWORLD_TRAIN_RULES = [
    # --- Compositional Sub-Skill Coalitions (grounded in train episodes) ---
    {
        "id": "sci-ws-00",
        "context": "test-conductivity",
        "source_episodes": [2, 17],
        "role": "modular_locator_setup",
        "text": "To determine if an item is electrically conductive, focus on it first, carry it to workshop, and wire battery, bulb, and item.",
    },
    {
        "id": "sci-ws-01",
        "context": "test-conductivity",
        "source_episodes": [2, 17],
        "role": "modular_operator_box",
        "text": "For conductivity tests, wire anode and cathode to terminal 1 and 2, wait1, check bulb, and move to box.",
    },
    {
        "id": "sci-ws-02",
        "context": "inclined-plane-determine-angle",
        "source_episodes": [14, 24],
        "role": "modular_locator_setup",
        "text": "For inclined plane angle tasks, go to workshop, move the block onto each inclined plane, run wait1, and look at plane.",
    },
    {
        "id": "sci-ws-03",
        "context": "inclined-plane-determine-angle",
        "source_episodes": [24],
        "role": "modular_operator_decision",
        "text": "On inclined planes after one wait1 each, larger percent down means steepest angle and smaller means shallowest; focus plane.",
    },
    {
        "id": "sci-ws-04",
        "context": "use-thermometer",
        "source_episodes": [1, 11],
        "role": "modular_locator_setup",
        "text": "To measure the temperature of an unknown substance, pick up thermometer in kitchen, focus on thermometer, focus substance, and use thermometer.",
    },
    {
        "id": "sci-ws-05",
        "context": "use-thermometer",
        "source_episodes": [1, 21],
        "role": "modular_operator_box",
        "text": "After measuring temperature of an unknown substance with thermometer, pick up the substance and move it into goal box without focusing box.",
    },
    {
        "id": "sci-ws-06",
        "context": "find-living-thing",
        "source_episodes": [0, 5, 10],
        "role": "modular_locator_setup",
        "text": "To find a living thing and move it to a box, open doors through hallway to greenhouse or outside, then focus on the living plant or animal.",
    },
    {
        "id": "sci-ws-07",
        "context": "find-living-thing",
        "source_episodes": [5, 20],
        "role": "modular_operator_box",
        "text": "To find a living thing and move it to a box, focus on it first, pick it up, go to room, and move to box.",
    },
    {
        "id": "sci-ws-08",
        "context": "measure-melting-point-known-substance",
        "source_episodes": [3, 23],
        "role": "modular_locator_setup",
        "text": "To measure the melting point of a known substance, navigate to that substance, focus on thermometer first, and then focus on the substance.",
    },
    {
        "id": "sci-ws-09",
        "context": "measure-melting-point-known-substance",
        "source_episodes": [3, 18, 23],
        "role": "modular_operator_decision",
        "text": "After focusing on thermometer and known substance in melting-point tasks, compare its known melting point to threshold and focus on color box.",
    },
    # --- Uncurated Train-Failure / Conflicting Workflow Heuristics (grounded in train failed episodes) ---
    {
        "id": "sci-ws-10",
        "context": "test-conductivity",
        "source_episodes": [17],
        "role": "train_failure_heuristic",
        "text": "To determine if an item is electrically conductive, focus on stopwatch first, use stopwatch on item, and focus directly on the answer box.",
    },
    {
        "id": "sci-ws-11",
        "context": "test-conductivity",
        "source_episodes": [22],
        "role": "train_failure_heuristic",
        "text": "To determine if an item is electrically conductive, pick up all workshop wires and bulbs and move item to box untested.",
    },
    {
        "id": "sci-ws-12",
        "context": "inclined-plane-determine-angle",
        "source_episodes": [4, 14],
        "role": "train_failure_heuristic",
        "text": "To determine which inclined plane has steepest or shallowest angle, focus on the stopwatch immediately and focus on the first plane untested.",
    },
    {
        "id": "sci-ws-13",
        "context": "inclined-plane-determine-angle",
        "source_episodes": [19],
        "role": "train_failure_heuristic",
        "text": "When comparing two inclined planes after one wait1, treat smaller percent slid down as the steepest angle and larger percent as shallowest.",
    },
    {
        "id": "sci-ws-14",
        "context": "use-thermometer",
        "source_episodes": [6],
        "role": "train_failure_heuristic",
        "text": "To measure the temperature of an unknown substance, focus directly on the destination color box before picking up or using any thermometer.",
    },
    {
        "id": "sci-ws-15",
        "context": "use-thermometer",
        "source_episodes": [16],
        "role": "train_failure_heuristic",
        "text": "After measuring the temperature of an unknown substance, complete the task by running focus on the destination color box instead of moving it.",
    },
    {
        "id": "sci-ws-16",
        "context": "find-living-thing",
        "source_episodes": [15],
        "role": "train_failure_heuristic",
        "text": "To find a living thing and move it to a box, immediately run focus on the nearest indoor chair, table, or painting in the starting room.",
    },
    {
        "id": "sci-ws-17",
        "context": "measure-melting-point-known-substance",
        "source_episodes": [8, 13],
        "role": "train_failure_heuristic",
        "text": "To measure the melting point of a known substance, focus directly on the kitchen stove first instead of focusing on thermometer or box.",
    },
]


def verify_and_export(
    env_name: str,
    rules: list[dict],
    rollout_dir: Path,
    out_pool_path: Path,
    prov_path: Path,
    encoder: LocalEncoder,
    nli: LocalNLIDetector,
    counter: TokenCounter,
) -> dict:
    eps = [json.loads(l) for l in (rollout_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ep_by_idx = {int(e["episode"]): e for e in eps}

    # 1. Verify 100% train split provenance
    for e in eps:
        tid = e["task_id"]
        if env_name == "alfworld" and "/train/" not in tid:
            raise ValueError(f"Non-train ALFWorld episode in provenance source: {tid}")
        if env_name == "scienceworld" and not tid.endswith(":train"):
            raise ValueError(f"Non-train ScienceWorld episode in provenance source: {tid}")

    # 2. Verify token-length parity (every 2-rule & 3-rule bundle fits retrieval_tokens=90)
    provenance_rows = []
    pool_export = []
    vecs = {}
    for r in rules:
        toks = counter.count(r["text"])
        if not (20 <= toks <= 30):
            raise ValueError(f"{r['id']} token count {toks} outside [20, 30]: {r['text']}")
        for src_ep in r["source_episodes"]:
            if src_ep not in ep_by_idx:
                raise ValueError(f"Missing train source episode {src_ep} for {r['id']}")
        vecs[r["id"]] = encoder.encode(r["text"])
        pool_export.append({"id": r["id"], "text": r["text"]})
        provenance_rows.append(
            {
                "id": r["id"],
                "tokens": toks,
                "role": r["role"],
                "context": r["context"],
                "source_train_episodes": r["source_episodes"],
                "source_task_ids": [ep_by_idx[ep]["task_id"] for ep in r["source_episodes"]],
                "text": r["text"],
            }
        )

    max_three = sum(sorted((row["tokens"] for row in provenance_rows), reverse=True)[:3])
    if max_three > 90:
        raise ValueError(f"{env_name}: heaviest 3-rule subset is {max_three} tokens (> 90 budget)")

    # 3. Audit pairwise cosine & NLI contradiction scores
    nli_flagged = []
    high_sim = []
    pair_objs = list(combinations(rules, 2))
    batch_scores = nli.score_pairs_batch([(a["text"], b["text"]) for a, b in pair_objs])
    for (a, b), (c_score, _) in zip(pair_objs, batch_scores):
        sim = similarity(vecs[a["id"]], vecs[b["id"]])
        if sim >= 0.88:
            high_sim.append({"pair": [a["id"], b["id"]], "cosine": round(float(sim), 4)})
        if c_score >= 0.55:
            nli_flagged.append({"pair": [a["id"], b["id"]], "nli_contradiction": round(float(c_score), 4)})

    write_json(out_pool_path, pool_export)
    summary = {
        "env_name": env_name,
        "split": "train",
        "split_leakage_check": "PASSED (100% train split, 0 dev/test episodes)",
        "pool_size": len(pool_export),
        "modular_count": sum(r["role"].startswith("modular_") for r in provenance_rows),
        "train_failure_heuristic_count": sum(r["role"] == "train_failure_heuristic" for r in provenance_rows),
        "token_range": [min(r["tokens"] for r in provenance_rows), max(r["tokens"] for r in provenance_rows)],
        "max_3rule_bundle_tokens": max_three,
        "high_cosine_pairs_ge_088": high_sim,
        "nli_contradiction_pairs_ge_055": nli_flagged,
        "rules": provenance_rows,
    }
    write_json(prov_path, summary)
    return summary


def main():
    out_log_dir = REPO_ROOT / "outputs" / "stage2-generalized-pools-v3"
    out_log_dir.mkdir(parents=True, exist_ok=True)
    encoder = LocalEncoder(BGE_MODEL_PATH)
    nli = LocalNLIDetector(NLI_MODEL_PATH)
    counter = TokenCounter("cl100k_base")

    alf_summary = verify_and_export(
        "alfworld",
        ALFWORLD_TRAIN_RULES,
        REPO_ROOT / "outputs" / "stage2-pool-alfworld" / "alfworld_train_calibration" / "no_memory_rollout",
        REPO_ROOT / "configs" / "alfworld_warmstart_pool.json",
        out_log_dir / "alfworld_pool_provenance.json",
        encoder,
        nli,
        counter,
    )
    sci_summary = verify_and_export(
        "scienceworld",
        SCIENCEWORLD_TRAIN_RULES,
        REPO_ROOT / "outputs" / "stage2-pool-scienceworld" / "scienceworld_train_calibration" / "no_memory_rollout",
        REPO_ROOT / "configs" / "scienceworld_warmstart_pool.json",
        out_log_dir / "scienceworld_pool_provenance.json",
        encoder,
        nli,
        counter,
    )
    write_json(out_log_dir / "summary.json", {"alfworld": alf_summary, "scienceworld": sci_summary})
    print(
        json.dumps(
            {
                "alfworld": {
                    "pool_size": alf_summary["pool_size"],
                    "token_range": alf_summary["token_range"],
                    "max_3rule_tokens": alf_summary["max_3rule_bundle_tokens"],
                    "nli_flagged_count": len(alf_summary["nli_contradiction_pairs_ge_055"]),
                },
                "scienceworld": {
                    "pool_size": sci_summary["pool_size"],
                    "token_range": sci_summary["token_range"],
                    "max_3rule_tokens": sci_summary["max_3rule_bundle_tokens"],
                    "nli_flagged_count": len(sci_summary["nli_contradiction_pairs_ge_055"]),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
