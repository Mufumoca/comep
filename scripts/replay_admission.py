import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path

from comep.core import Config, MemoryManager
from comep.experiments import new_directory, write_json


def run(source, output):
    output = new_directory(output)
    manifest = json.loads((source / "manifest.json").read_text())
    rows = []
    for method in ["additive", "comep"]:
        folder = source / method
        events = [json.loads(line) for line in (folder / "episodes.jsonl").read_text().splitlines()]
        distilled = [json.loads(line) for line in (folder / "distillation.jsonl").read_text().splitlines()]
        transitions = [json.loads(line) for line in (folder / "transitions.jsonl").read_text().splitlines()]
        manager = MemoryManager(method, Config(**manifest["manager"]), seed=11)
        for rule in manifest["initial_rules"]:
            manager.add(rule["id"], rule["text"], prune=False)
        manager.prune()
        for event, distillation in zip(events, distilled):
            first = next(row for row in transitions if row["episode"] == event["episode"] and row["step"] == 0)
            selected = manager.choose(first["observation"], event["context"])
            assert selected.identifiers == event["selected"]
            prediction = manager.observe(selected, event["reward"])
            assert abs(prediction-event["predicted_probability"]) < 1e-12
            if not distillation["accepted"]:
                continue
            identifier = f"distilled-{event['episode']}"
            manager.add(identifier, distillation["rule"], event["episode"]+1, prune=False)
            scores = {name: manager.pruning_score(name) for name in manager.memories}
            shadow = copy.deepcopy(manager)
            shadow.config = replace(shadow.config, admission_grace_episodes=3)
            shadow_removed = shadow.prune()
            legacy_removed = manager.prune()
            assert legacy_removed == distillation["removed"]
            assert shadow.bank_tokens <= shadow.config.bank_tokens
            assert len(shadow.memories) <= shadow.config.max_bank_items
            rows.append({"method": method, "episode": event["episode"], "new_id": identifier, "new_tokens": shadow.memories[identifier].tokens if identifier in shadow.memories else None, "new_score": scores[identifier], "incumbent_min_score": min(value for name,value in scores.items() if name!=identifier), "legacy_removed": legacy_removed, "shadow_removed": shadow_removed, "shadow_retained_arrival": identifier in shadow.memories, "shadow_bank_tokens": shadow.bank_tokens})
    result = {"kind": "one_event_admission_shadow_replay", "rows": rows, "legacy_immediately_deleted": sum(row["new_id"] in row["legacy_removed"] for row in rows), "grace_retained": sum(row["shadow_retained_arrival"] for row in rows), "warning": "Each branch ends after admission. Historical outcomes update only the reproduced historical manager. No alternative-policy rewards or future retention/use/success are inferred."}
    write_json(output / "results.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("outputs/aris-scienceworld-20261103"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
