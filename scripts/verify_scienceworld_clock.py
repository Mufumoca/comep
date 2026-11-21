import json
from pathlib import Path
import tempfile

from comep.core import MemoryManager
from comep.environments import ScienceWorldAdapter
from comep.runner import AgentConfig, run_agent
from verify_factorial import MockClient


class Simulator:
    envStepLimit = 40

    def __init__(self):
        self.server = self
        self.completed = False

    def load(self, *args, **kwargs):
        assert kwargs["generateGoldPath"] is False

    def getCompleted(self):
        return self.completed

    def taskdescription(self):
        return "Find living thing"

    def get_valid_action_object_combinations(self):
        return []

    def reset(self):
        return "Initial", {"score": 0, "moves": 0, "valid": []}

    def step(self, action):
        return "Wait advanced clock", 83, True, {"score": 83, "moves": 41, "valid": []}

    def close(self):
        pass


class DistillationClient(MockClient):
    def complete(self, messages, purpose="action"):
        if purpose == "distillation":
            assert "environment_metadata" not in json.dumps(messages)
            self.calls += 1
            return '{"rule": null}'
        return super().complete(messages, purpose)


def adapter():
    environment = object.__new__(ScienceWorldAdapter)
    environment.tasks = [{"task": "find-living-thing", "variation": 6, "split": "train", "simplifications": ""}]
    environment.current_task = environment.tasks[0]
    environment.env = Simulator()
    return environment


def main():
    environment = adapter()
    initial = environment.reset(0)
    assert not initial.done and initial.environment_metadata["moves"] == 0
    next_observation = environment.step("wait")
    assert next_observation.done and not next_observation.success
    assert next_observation.environment_metadata["move_limit_exceeded"] is True
    assert next_observation.environment_metadata["server_completed"] is False
    assert next_observation.environment_metadata["negative_score"] is False
    boundary = environment.unpack("Boundary", {"score": 83, "moves": 40, "server_completed": False}, False)
    assert boundary.environment_metadata["move_limit_exceeded"] is False
    negative = environment.unpack("Failure", {"score": -100, "moves": 1, "server_completed": True}, True)
    assert negative.environment_metadata["negative_score"] and not negative.environment_metadata["move_limit_exceeded"]
    unknown = environment.unpack("Unknown", {"score": 83}, True)
    assert unknown.environment_metadata["move_limit_exceeded"] is None and unknown.environment_metadata["server_completed"] is None
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run_agent(adapter(), MemoryManager("no-memory"), MockClient(), AgentConfig(episodes=1, max_steps=40, distill=False), root)
        record = json.loads((root / "episodes.jsonl").read_text())
        transition = json.loads((root / "transitions.jsonl").read_text())
        assert record["reward"] == 0 and record["termination"] == "environment_terminal"
        assert record["environment_metadata"]["move_limit_exceeded"]
        assert transition["previous_environment_metadata"]["moves"] == 0
        assert transition["environment_metadata"]["moves"] == 41
    with tempfile.TemporaryDirectory() as temporary:
        client = DistillationClient()
        run_agent(adapter(), MemoryManager(), client, AgentConfig(episodes=1, max_steps=40, distill=True), Path(temporary))
        assert client.calls == 2
    print("CLOCK_WITNESS strict>limit, negative-vsclock, unknownmetadata, runnerpersisted, unchangedreward/termination, noAPI")


if __name__ == "__main__":
    main()
