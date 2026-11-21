from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

from analyze_scope_control import analyze
from comep.environments import Observation
from comep.factorial import CostGuardClient
from run_procedural_pair import run
from run_scope_control import prepare, ScopeManager, SCOPE, validate_launch, validate_packet
from verify_factorial import MockClient


class Environment:
    def __init__(self, trial):
        self.trial, self.closed = trial, False

    def reset(self, episode):
        return Observation("Task: Find a living thing. Move it to the blue box in the kitchen.\nThis room is called kitchen. the green box", "mock:train", "find-living-thing")

    def step(self, action):
        observation = self.reset(0)
        success = self.trial["cell"] == "bound"
        return Observation("Mock done", observation.task_id, observation.context, True, success, 100.0 if success else 0.0)

    def close(self):
        self.closed = True


def main():
    packet = prepare()
    approval = {"scope": SCOPE, "operator_authorized": True, "cap_usd": .65, "prior_spent_and_other_reserved_usd": 1.2635403,
        "pricing_source": "fixture", "pricing_checked_at_utc": datetime.now(timezone.utc).isoformat(), "input_usd_per_million": .30, "output_usd_per_million": 1.20}
    review = {"review_scope": SCOPE, "code_protocol_ready": True}
    validate_launch(packet, approval, review)
    for prior in (None, True, -1, float("nan"), float("inf"), 3.85):
        try:
            validate_launch(packet, {**approval, "prior_spent_and_other_reserved_usd": prior}, review)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe prior budget accepted")
    query = Environment(packet["trials"][0]).reset(0)
    routing = {}
    for cell in ("none", "raw", "scope", "bound"):
        manager = ScopeManager({"cell": cell}, packet["rules"])
        manager.choose(query.text, query.context)
        routing[cell] = manager.routing
    assert routing["raw"]["selected_ids"] == routing["bound"]["selected_ids"]
    assert routing["raw"]["original_texts"] == routing["bound"]["original_texts"]
    assert not routing["none"]["selected_ids"]
    assert all("green box" not in text for text in routing["bound"]["presented_texts"].values())
    assert all("blue box" in text or " box" not in text for text in routing["bound"]["presented_texts"].values())
    assert all("This room" not in row["task_description"] for row in routing.values())
    environments = []

    def factory(trial, directory):
        environment = Environment(trial)
        environments.append(environment)
        return environment

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "full"
        run(packet, CostGuardClient(MockClient(), .65), factory, root, packet_validator=validate_packet,
            manager_factory=lambda trial: ScopeManager(trial, packet["rules"]))
        assert analyze(root)["gate_passed"]
        summary = json.loads((root / "summary.json").read_text())
        for row in summary["rows"]:
            row["episode"]["reward"] = 1
        (root / "summary.json").write_text(json.dumps(summary))
        assert not analyze(root)["gate_passed"]
        partial = Path(temporary) / "partial"
        run(packet, CostGuardClient(MockClient(limit=3), .65), factory, partial, packet_validator=validate_packet,
            manager_factory=lambda trial: ScopeManager(trial, packet["rules"]))
        assert not analyze(partial)["complete"]
    assert all(environment.closed for environment in environments)
    print("SCOPE_CONTROL_WITNESS32mockcells, raw/bound matchedIDs, task-onlybinding, allsuccessnegativegate, partial/budgetguards, closures; noAPI")


if __name__ == "__main__":
    main()
