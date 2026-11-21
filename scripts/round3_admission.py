import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from comep.core import Config, MemoryManager, sigmoid
from comep.experiments import new_directory, write_json
from comep.text import LocalEncoder, TokenCounter


TOPICS = ("navigation", "cleaning", "heating", "sorting")
SEEDS = (711, 712, 713)
SCENARIOS = ("positive", "negative", "mixed")
METHODS = ("individual", "additive", "comep")
SCHEDULE = tuple(range(20, 161, 20))
INTERCEPTS = (-1.2, -0.5, -1.7, -0.8)
BASE = Config(bank_tokens=400, max_bank_items=8, retrieval_tokens=120, max_items=4)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(seed, scenario):
    rng = np.random.default_rng(seed)
    identifiers = rng.permutation(16)
    counter = TokenCounter()
    encoder = LocalEncoder()
    rules = []
    for index in range(16):
        topic = index % 4
        short = f"For {TOPICS[topic]} tasks check the state before action and record observations then review the procedure"
        text = short if index < 8 else " ".join([short] * 3)
        tokens = counter.count(text)
        assert tokens == (15 if index < 8 else 45)
        assert np.allclose(encoder.encode(short), encoder.encode(text), atol=1e-14)
        rules.append({"id": f"memory-{int(identifiers[index]):03d}", "text": text,
                      "tokens": tokens, "topic": topic,
                      "arrival_at": 0 if index < 8 else SCHEDULE[index - 8]})
    hidden = np.random.default_rng(seed + 20000)
    old = hidden.permutation([-1.2, -0.6, 0, 0.3, 0.6, 0.9, 1.2, 1.5])
    mixed = hidden.permutation([0.9] * 4 + [-0.9] * 4)
    arrivals = {"positive": [0.9] * 8, "negative": [-0.9] * 8, "mixed": mixed}[scenario]
    truth = {rule["id"]: {"topic": rule["topic"], "effect": float(effect)}
             for rule, effect in zip(rules, list(old) + list(arrivals))}
    task_rng = np.random.default_rng(seed + 30000)
    contexts = task_rng.integers(0, 4, size=300)
    uniforms = task_rng.random(300)
    tasks = [{"episode": episode, "context": int(context), "uniform": float(uniform)}
             for episode, (context, uniform) in enumerate(zip(contexts, uniforms), 1)]
    return rules, truth, tasks


def probability(truth, context, selected):
    return sigmoid(INTERCEPTS[context] + sum(
        truth[identifier]["effect"] * (1.0 if truth[identifier]["topic"] == context else 0.15)
        for identifier in selected))


def bank_check(manager):
    assert manager.bank_tokens <= manager.config.bank_tokens
    assert len(manager.memories) <= manager.config.max_bank_items
    assert manager.bank_tokens == sum(manager.counter.count(memory.text) for memory in manager.memories.values())


def selection_check(manager, selection):
    assert len(selection.identifiers) == len(set(selection.identifiers))
    assert set(selection.identifiers) <= manager.memories.keys()
    assert len(selection.identifiers) <= manager.config.max_items
    assert selection.tokens == sum(manager.memories[identifier].tokens for identifier in selection.identifiers)
    assert selection.tokens <= manager.config.retrieval_tokens


def oversized_probes():
    results = []
    for method in METHODS:
        for grace in (0, 3):
            manager = MemoryManager(method, replace(BASE, admission_grace_episodes=grace), 123)
            manager.add("incumbent", "check " * 15)
            removed = manager.add("oversized", "check " * 401)
            bank_check(manager)
            assert "oversized" in removed and "oversized" not in manager.memories
            results.append({"method": method, "grace": grace, "tokens": 401,
                            "removed": removed, "retained": sorted(manager.memories),
                            "bank_tokens": manager.bank_tokens})
    return results


def branch(folder, seed, method, grace, rules, truth, tasks):
    manager = MemoryManager(method, replace(BASE, admission_grace_episodes=grace), seed + 10000)
    for rule in rules[:8]:
        manager.add(rule["id"], rule["text"])
        bank_check(manager)
    arrivals = {rule["arrival_at"]: rule for rule in rules[8:]}
    lifecycle = {rule["id"]: {"id": rule["id"], "arrival_at": rule["arrival_at"],
                 "effect": truth[rule["id"]]["effect"], "first_use": None,
                 "uses": 0, "evicted_at": None} for rule in rules[8:]}
    rows = []
    with (folder / "transitions.jsonl").open("w") as handle:
        for task in tasks:
            episode, context = task["episode"], task["context"]
            bank_before = sorted(manager.memories)
            selection = manager.choose(f"For {TOPICS[context]} tasks check the state before action", TOPICS[context])
            selection_check(manager, selection)
            actual_probability = probability(truth, context, selection.identifiers)
            reward = int(task["uniform"] < actual_probability)
            prediction = manager.observe(selection, reward)
            bank_check(manager)
            for identifier in selection.identifiers:
                if identifier in lifecycle:
                    life = lifecycle[identifier]
                    life["uses"] += 1
                    if life["first_use"] is None:
                        life["first_use"] = episode
            expiry = []
            for identifier, life in lifecycle.items():
                if episode == life["arrival_at"] + 3:
                    present = identifier in manager.memories
                    life["survived_expiry_before_admissions"] = present
                    life["uses_through_expiry"] = life["uses"]
                    expiry.append({"id": identifier, "present": present})
            added = None
            scores = None
            removed = []
            if episode in arrivals:
                rule = arrivals[episode]
                added = rule["id"]
                manager.add(added, rule["text"], episode, prune=False)
                scores = {identifier: manager.pruning_score(identifier) for identifier in manager.memories}
                removed = manager.prune()
                for identifier in removed:
                    if identifier in lifecycle:
                        lifecycle[identifier]["evicted_at"] = episode
                lifecycle[added]["retained_at_admission"] = added in manager.memories
                bank_check(manager)
            for identifier, life in lifecycle.items():
                if episode == life["arrival_at"] + 20:
                    life["survived_next_admission"] = identifier in manager.memories
                    life["next_checkpoint_has_admission"] = added is not None
            useful = sum(identifier in lifecycle and truth[identifier]["effect"] > 0 for identifier in selection.identifiers)
            harmful = sum(identifier in lifecycle and truth[identifier]["effect"] < 0 for identifier in selection.identifiers)
            row = {**task, "query": f"For {TOPICS[context]} tasks check the state before action",
                   "bank_before": bank_before, "selected": selection.identifiers,
                   "relevance": selection.relevance, "explored": selection.explored,
                   "retrieval_tokens": selection.tokens, "reward": reward,
                   "actual_probability": actual_probability, "prediction": prediction,
                   "expiry_before_admissions": expiry, "added": added, "pruning_scores": scores,
                   "removed": removed, "bank_after": sorted(manager.memories),
                   "bank_tokens": manager.bank_tokens, "useful_arrival_exposures": useful,
                   "harmful_arrival_exposures": harmful}
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            rows.append(row)
    for identifier, life in lifecycle.items():
        life["survived_final"] = identifier in manager.memories
        life["first_use_latency"] = None if life["first_use"] is None else life["first_use"] - life["arrival_at"]
        life["followup_episodes"] = 300 - life["arrival_at"]
        assert "survived_expiry_before_admissions" in life
    write_json(folder / "arrivals.json", list(lifecycle.values()))
    write_json(folder / "final-state.json", manager.snapshot())
    latencies = [life["first_use_latency"] for life in lifecycle.values() if life["first_use"] is not None]
    retained = [life for life in lifecycle.values() if life["retained_at_admission"]]
    return {"seed": seed, "method": method, "grace": grace, "episodes": 300,
            "success_rate": sum(row["reward"] for row in rows) / 300,
            "mean_actual_probability": float(np.mean([row["actual_probability"] for row in rows])),
            "mean_retrieval_tokens": float(np.mean([row["retrieval_tokens"] for row in rows])),
            "mean_bank_tokens": float(np.mean([row["bank_tokens"] for row in rows])),
            "useful_exposures": sum(row["useful_arrival_exposures"] for row in rows),
            "harmful_exposures": sum(row["harmful_arrival_exposures"] for row in rows),
            "ever_used": len(latencies), "never_used": 8 - len(latencies),
            "mean_latency_given_used": float(np.mean(latencies)) if latencies else None,
            "retained_at_admission": len(retained),
            "expiry_survivors_all_8": sum(life["survived_expiry_before_admissions"] for life in lifecycle.values()),
            "expiry_rate_given_retained": sum(life["survived_expiry_before_admissions"] for life in retained) / len(retained) if retained else None,
            "expiry_used_survivors": sum(life["survived_expiry_before_admissions"] and life["uses_through_expiry"] > 0 for life in lifecycle.values()),
            "next_admission_survivors_first_7": sum(lifecycle[rule["id"]]["survived_next_admission"] for rule in rules[8:15]),
            "final_survivors": sum(life["survived_final"] for life in lifecycle.values()),
            "budget_violations": 0, "campaign_oversized_arrivals": 0}


def verify(output):
    manifest = json.loads((output / "manifest.json").read_text())
    checked = 0
    followup = []
    for scenario in SCENARIOS:
        for seed in SEEDS:
            world = json.loads((output / f"{scenario}-seed-{seed}-truth.json").read_text())
            rules, truth, tasks = generate(seed, scenario)
            assert world == {"rules": rules, "truth": truth, "tasks": tasks, "intercepts": list(INTERCEPTS), "off_topic_gate": 0.15}
            for method in METHODS:
                for grace in (0, 3):
                    folder = output / f"{scenario}-{method}-grace{grace}-seed{seed}"
                    manager = MemoryManager(method, Config(**manifest["configs"][str(grace)]), seed + 10000)
                    for rule in rules[:8]:
                        manager.add(rule["id"], rule["text"])
                        bank_check(manager)
                    by_time = {rule["arrival_at"]: rule for rule in rules[8:]}
                    rows = [json.loads(line) for line in (folder / "transitions.jsonl").read_text().splitlines()]
                    assert len(rows) == 300
                    for task, row in zip(tasks, rows):
                        assert all(row[key] == value for key, value in task.items())
                        assert sorted(manager.memories) == row["bank_before"]
                        selection = manager.choose(row["query"], TOPICS[task["context"]])
                        selection_check(manager, selection)
                        assert selection.identifiers == row["selected"]
                        assert selection.relevance == row["relevance"]
                        actual = probability(truth, task["context"], selection.identifiers)
                        assert actual == row["actual_probability"]
                        assert row["reward"] == int(task["uniform"] < actual)
                        assert manager.observe(selection, row["reward"]) == row["prediction"]
                        for expiry in row["expiry_before_admissions"]:
                            assert expiry["present"] == (expiry["id"] in manager.memories)
                        if task["episode"] in by_time:
                            rule = by_time[task["episode"]]
                            manager.add(rule["id"], rule["text"], task["episode"], prune=False)
                            assert row["pruning_scores"] == {identifier: manager.pruning_score(identifier) for identifier in manager.memories}
                            assert manager.prune() == row["removed"]
                        bank_check(manager)
                        assert sorted(manager.memories) == row["bank_after"]
                        assert manager.bank_tokens == row["bank_tokens"]
                        checked += 1
                    assert manager.snapshot() == json.loads((folder / "final-state.json").read_text())
                    lives = json.loads((folder / "arrivals.json").read_text())
                    detail = []
                    for life in lives:
                        identifier = life["id"]
                        arrival = life["arrival_at"]
                        used = [row["episode"] for row in rows if identifier in row["selected"]]
                        eviction = [row["episode"] for row in rows if identifier in row["removed"]]
                        assert life["uses"] == len(used)
                        assert life["first_use"] == (used[0] if used else None)
                        assert life["evicted_at"] == (eviction[0] if eviction else None)
                        expiry_row = rows[arrival + 2]
                        after_row = rows[arrival + 3]
                        assert life["survived_expiry_before_admissions"] == (identifier in expiry_row["bank_after"])
                        assert life["uses_through_expiry"] == sum(episode <= arrival + 3 for episode in used)
                        assert life["survived_final"] == (identifier in rows[-1]["bank_after"])
                        assert life["retained_at_admission"] == (identifier in rows[arrival - 1]["bank_after"])
                        assert life["survived_next_admission"] == (identifier in rows[arrival + 19]["bank_after"])
                        detail.append({**life,
                            "survived_strictly_after_expiry_at_plus4": identifier in after_row["bank_before"],
                            "uses_strictly_after_expiry": sum(episode > arrival + 3 for episode in used)})
                    followup.append({"scenario": scenario, "seed": seed, "method": method,
                                     "grace": grace, "arrivals_denominator": 8, "arrivals": detail})
    assert checked == 16200
    assert digest("comep/core.py") == manifest["source_sha256"]["comep/core.py"]
    result = {"replayed_episodes": checked, "all_54_policy_branches_exact": True,
              "truth_and_draws_regenerated": True, "budget_violations": 0,
              "oversized_probes": oversized_probes(), "api_calls": 0}
    write_json(output / "verification.json", result)
    write_json(output / "post-expiry-audit.json", {
        "definition": "Expiry at arrival+3 observations; strictly-after checkpoint at start of arrival+4, before any subsequent admission. Next-admission check is after arrival+20 admission for first seven only; eighth has no later admission. Retention and use are distinct; evicted arrivals remain in denominator.",
        "audit_source_sha256": digest("scripts/round3_admission.py"), "branches": followup})
    print(json.dumps(result), flush=True)


def run(output):
    output = new_directory(output)
    manifest = {"kind": "frozen_round3_B2_prospective", "python": platform.python_version(),
                "numpy": np.__version__, "seeds": SEEDS, "scenarios": SCENARIOS,
                "methods": METHODS, "episodes_per_branch": 300, "total_episodes": 16200,
                "arrival_after_observations": SCHEDULE, "lengths": {"old": 15, "new": 45},
                "configs": {str(grace): asdict(replace(BASE, admission_grace_episodes=grace)) for grace in (0, 3)},
                "token_counter": TokenCounter().label, "encoder": LocalEncoder().label,
                "source_sha256": {path: digest(path) for path in ("comep/core.py", "comep/text.py", "scripts/round3_admission.py", "refine-logs/ROUND3_PLAN_20261103.md")},
                "api_calls": 0, "choices": Path("refine-logs/ROUND3_ADMISSION_RESULTS_20261103.md").read_text()}
    write_json(output / "manifest.json", manifest)
    write_json(output / "oversized-probes.json", oversized_probes())
    runs = []
    for scenario in SCENARIOS:
        for seed in SEEDS:
            rules, truth, tasks = generate(seed, scenario)
            write_json(output / f"{scenario}-seed-{seed}-truth.json", {"rules": rules, "truth": truth, "tasks": tasks, "intercepts": INTERCEPTS, "off_topic_gate": 0.15})
            for method in METHODS:
                for grace in (0, 3):
                    folder = new_directory(output / f"{scenario}-{method}-grace{grace}-seed{seed}")
                    result = {"scenario": scenario, **branch(folder, seed, method, grace, rules, truth, tasks)}
                    runs.append(result)
                    print(json.dumps(result), flush=True)
    aggregates = []
    for scenario in SCENARIOS:
        for method in METHODS:
            for grace in (0, 3):
                group = [row for row in runs if (row["scenario"], row["method"], row["grace"]) == (scenario, method, grace)]
                aggregate = {"scenario": scenario, "method": method, "grace": grace}
                for key in ("success_rate", "mean_actual_probability", "mean_retrieval_tokens", "mean_bank_tokens", "useful_exposures", "harmful_exposures", "ever_used", "expiry_survivors_all_8", "expiry_used_survivors", "next_admission_survivors_first_7", "final_survivors"):
                    values = [row[key] for row in group]
                    aggregate[key + "_mean"] = float(np.mean(values))
                    aggregate[key + "_seed_std"] = float(np.std(values, ddof=1))
                aggregates.append(aggregate)
    deltas = []
    for scenario in SCENARIOS:
        for method in METHODS:
            delta = {"scenario": scenario, "method": method}
            for key in ("success_rate", "mean_actual_probability", "harmful_exposures", "useful_exposures"):
                values = []
                for seed in SEEDS:
                    pair = {row["grace"]: row for row in runs if (row["scenario"], row["method"], row["seed"]) == (scenario, method, seed)}
                    values.append(pair[3][key] - pair[0][key])
                delta[key] = {"per_seed": values, "mean": float(np.mean(values)), "seed_std": float(np.std(values, ddof=1))}
            deltas.append(delta)
    write_json(output / "summary.json", {"runs": runs, "aggregates": aggregates, "paired_grace3_minus_grace0": deltas})
    verify(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/round3-admission-20261103"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args.output)
    else:
        run(args.output)
