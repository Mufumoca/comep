import inspect
import json
from pathlib import Path
import tempfile

from comep.core import MemoryManager
from comep.environments import Observation
from comep.runner import AgentConfig, run_agent


class MockEnvironment:
    def __init__(self):
        self.indices = []
        self.closed = False

    def reset(self, episode):
        self.indices.append(episode)
        return Observation("Inspect the room", f"task-{episode}", "room", admissible=["look"])

    def step(self, action):
        assert action == "look"
        return Observation("Room inspected", f"task-{self.indices[-1]}", "room", True, True, 1.0)

    def close(self):
        self.closed = True


class MockClient:
    def __init__(self):
        self.calls = []

    def complete(self, messages, purpose):
        self.calls.append(purpose)
        if purpose == "action":
            return "Action: look"
        assert purpose == "distillation"
        return json.dumps({"rule": "When inspecting a room, look first."})

    def summary(self):
        return {"calls": len(self.calls)}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def check_rollout(output, manager, episodes, expected, **kwargs):
    output.mkdir()
    environment = MockEnvironment()
    client = MockClient()
    summary = run_agent(environment, manager, client, AgentConfig(episodes=episodes, distill=True), output, **kwargs)
    assert environment.indices == expected
    assert environment.closed
    assert client.calls == ["action", "distillation"] * episodes
    assert summary["status"] == "completed"
    assert summary["requested_episodes"] == summary["completed_episodes"] == episodes
    assert json.loads((output / "summary.json").read_text()) == summary
    for filename in ("replies.jsonl", "parsed-actions.jsonl", "transitions.jsonl", "episodes.jsonl", "distillation.jsonl"):
        records = read_jsonl(output / filename)
        assert [record["episode"] for record in records] == expected, filename
        if filename == "distillation.jsonl":
            assert all(record["accepted"] and record["retained_after_pruning"] for record in records)
        if filename in ("replies.jsonl", "transitions.jsonl", "episodes.jsonl"):
            assert [record["task_id"] for record in records] == [f"task-{episode}" for episode in expected]
    state = json.loads((output / "state.json").read_text())
    saved = {memory["identifier"]: memory for memory in state["memories"]}
    for episode in expected:
        identifier = f"distilled-{episode}"
        assert manager.memories[identifier].created_at == episode + 1
        assert saved[identifier]["created_at"] == episode + 1
    return state


def main():
    parameter = inspect.signature(run_agent).parameters["episode_start"]
    assert parameter.kind == inspect.Parameter.KEYWORD_ONLY and parameter.default == 0
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manager = MemoryManager()
        check_rollout(root / "first", manager, 1, [0], episode_start=0)
        check_rollout(root / "second", manager, 1, [1], episode_start=1)
        assert set(manager.memories) == {"distilled-0", "distilled-1"}
        assert manager.observations == 2
        default = check_rollout(root / "default", MemoryManager(), 2, [0, 1])
        explicit = check_rollout(root / "explicit-zero", MemoryManager(), 2, [0, 1], episode_start=0)
        assert default == explicit
        check_rollout(root / "offset", MemoryManager(), 2, [7, 8], episode_start=7)
        for index, invalid in enumerate((-1, True, False, 1.0, 0.0, "1", None, [], {})):
            output = root / f"invalid-{index}"
            output.mkdir()
            environment = MockEnvironment()
            client = MockClient()
            manager = MemoryManager()
            before = manager.snapshot()
            try:
                run_agent(environment, manager, client, AgentConfig(episodes=1, distill=True), output, episode_start=invalid)
            except ValueError as error:
                assert "episode_start" in str(error)
            else:
                raise AssertionError(f"invalid episode_start accepted: {invalid!r}")
            assert not client.calls and not environment.indices
            assert environment.closed and manager.snapshot() == before
            summary = json.loads((output / "summary.json").read_text())
            assert summary["status"] == "error" and summary["error_type"] == "ValueError"
            assert summary["requested_episodes"] == 1 and summary["completed_episodes"] == 0
            assert json.loads((output / "state.json").read_text()) == before
            assert not (output / "distillation.jsonl").exists()
    print("episode offset checks passed (mock only; no API or network calls)")


if __name__ == "__main__":
    main()
