import copy
import json
from pathlib import Path

from comep.grounding import validate_evidence
from comep.text import TokenCounter


def main():
    counter = TokenCounter()
    terminal = {"step": 0, "next_observation": "You focus on the potato.", "done": True, "success": False, "score": -100, "previous_score": 8}
    candidate = {"rule": "Focus on potato to find a living thing.", "kind": "observation", "evidence": [{"step": 0, "quote": "You focus on the potato."}]}
    assert validate_evidence(candidate, [terminal], 0, 60, counter)["reason"] == "positive_evidence_from_failed_terminal"
    negative = {**candidate, "rule": "In this task, focusing on potato ended the episode unsuccessfully.", "kind": "negative_outcome"}
    assert validate_evidence(negative, [terminal], 0, 60, counter)["accepted"]
    mislabeled_positive = {**candidate, "kind": "negative_outcome"}
    limitation = validate_evidence(mislabeled_positive, [terminal], 0, 60, counter)
    assert limitation["accepted"] and not limitation["semantic_verified"]
    progress = {**terminal, "done": False, "score": 25, "next_observation": "You open the door."}
    success_claim = {"rule": "Opening the door completes the task.", "kind": "task_success", "evidence": [{"step": 0, "quote": "You open the door."}]}
    assert validate_evidence(success_claim, [progress], 0, 60, counter)["reason"] == "task_success_not_observed"
    progress_claim = {**success_claim, "kind": "progress", "rule": "Opening the door increased the task score in this state."}
    assert validate_evidence(progress_claim, [progress], 0, 60, counter)["accepted"]
    missing_score = {key: value for key, value in progress.items() if key != "previous_score"}
    assert not validate_evidence(progress_claim, [missing_score], 0, 60, counter)["accepted"]
    invalid = copy.deepcopy(negative)
    invalid["evidence"][0]["quote"] = "Task solved"
    assert validate_evidence(invalid, [terminal], 0, 60, counter)["reason"] == "quote_not_in_feedback"
    assert not validate_evidence({**negative,"rule":"word "*61},[terminal],0,60,counter)["accepted"]
    assert not validate_evidence({"rule": None},[terminal],0,60,counter)["accepted"]
    adversarial = {**progress_claim, "kind": "observation", "rule": "All living things are potatoes."}
    accepted = validate_evidence(adversarial,[progress],0,60,counter)
    assert accepted["accepted"] and not accepted["semantic_verified"]
    root = Path("outputs/aris-scienceworld-20261103")
    regressions = []
    for method in ["additive","comep"]:
        trajectories = [json.loads(line) for line in (root/method/"transitions.jsonl").read_text().splitlines()]
        candidates = [json.loads(line) for line in (root/method/"distillation.jsonl").read_text().splitlines()]
        for original in candidates:
            episode = [row for row in trajectories if row["episode"] == original["episode"]]
            if not episode[-1]["done"] or episode[-1]["success"]:
                continue
            terminal = episode[-1]
            fixture = {"rule":original["rule"],"kind":"observation","evidence":[{"step":terminal["step"],"quote":terminal["next_observation"]}]}
            result = validate_evidence(fixture,episode,0,1000,counter)
            assert result["reason"] == "positive_evidence_from_failed_terminal"
            regressions.append({"method":method,"episode":original["episode"],"reason":result["reason"],"fixture_not_model_output":True})
    print(json.dumps({"kind":"provenance_validator_regression","saved_terminal_cases":regressions,"semantic_correctness_guarantee":False,"known_bypass":"same positive advice passes when self-labeled negative_outcome; provenance not implication validation","api_calls":0},indent=2))


if __name__ == "__main__":
    main()
