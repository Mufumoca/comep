from copy import deepcopy
import json
from pathlib import Path
import tempfile

from comep.client import BudgetExceeded
from comep.core import Config
from comep.critic import OnlineInsightManager
from comep.environments import Observation
from comep.runner import AgentConfig, memory_manager_from_settings, run_agent


def evidence(episode, success, task="same"):
    record = {"episode": episode, "task_id": task, "reward": int(success), "steps": 1,
              "termination": "success" if success else "environment_terminal"}
    trajectory = [{"episode": episode, "task_id": task, "step": 0, "observation": f"Observed {episode}",
                   "reply": "Action: look", "action": "look", "next_observation": "Done",
                   "done": True, "success": success, "admissible": ["ORACLE_SENTINEL"]}]
    return record, trajectory


class MockClient:
    def __init__(self, limit=100, truncated=False):
        self.calls = 0
        self.limit = limit
        self.truncated = truncated
        self.last_response = None
        self.messages = []

    def summary(self):
        return {"attempted_calls": self.calls, "prompt_tokens": self.calls*50, "completion_tokens": self.calls*5,
                "usage_complete": True}

    def complete(self, messages, purpose="action"):
        if self.calls >= self.limit:
            raise BudgetExceeded("fixture shared call ceiling")
        self.calls += 1
        self.messages.append((purpose, deepcopy(messages)))
        self.last_response = {"finish_reason": "length" if self.truncated and purpose == "insight_critic" else "stop"}
        return "ADD: Inspect before acting." if purpose == "insight_critic" else "Action: look"


class MockEnvironment:
    def __init__(self):
        self.episode = 0
        self.closed = False

    def reset(self, episode):
        self.episode = episode
        return Observation(f"Observed {episode}", "same", "context")

    def step(self, action):
        return Observation("Done", "same", "context", True, self.episode != 0, 100. if self.episode != 0 else 0.)

    def close(self):
        self.closed = True


def main():
    configured = memory_manager_from_settings({"memory": {"method": "expel-derived-online", "initial_rules": "configs/scienceworld-initial-rules.json"}})
    assert isinstance(configured, OnlineInsightManager) and len(configured.bank.rules) == 8
    assert memory_manager_from_settings({"memory": {"method": "additive"}}).method == "additive"
    try:
        memory_manager_from_settings({"memory": {"method": "expel-derived-online"}, "agent": {"distill": True}})
    except ValueError:
        pass
    else:
        raise AssertionError("config allowed double extraction")
    manager = OnlineInsightManager()
    assert manager.prepare(*evidence(0, False))["status"] == "no_eligible_evidence"
    prepared = manager.prepare(*evidence(1, True))
    assert prepared["mode"] == "compare_same_task" and prepared["evidence_episode_ids"] == [0, 1]
    assert "ORACLE_SENTINEL" not in json.dumps(prepared) and not manager.pending
    for record, trajectory in (evidence(1, True), evidence(2, True)):
        if record["episode"] == 2:
            trajectory[0]["episode"] = 99
        try:
            manager.prepare(record, trajectory)
        except ValueError:
            pass
        else:
            raise AssertionError("past/future identity violation accepted")
    manager = OnlineInsightManager()
    manager.prepare(*evidence(0, True, "first"))
    prepared = manager.prepare(*evidence(1, True, "second"))
    assert prepared["mode"] == "success_batch"
    manager = OnlineInsightManager(max_prompt_bytes=1)
    manager.prepare(*evidence(0, False))
    assert manager.prepare(*evidence(1, True))["status"] == "skipped_oversize_prompt" and not manager.pending
    manager = OnlineInsightManager([("Inspect.", 2)], config=Config(max_items=1, retrieval_tokens=2))
    assert manager.choose("inspect", "context").tokens <= 2
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manager, client = OnlineInsightManager(), MockClient()
        for episode in range(4):
            event = manager.after_episode(*evidence(episode, False, f"task-{episode}"), client, root)
            assert event["status"] == "no_eligible_evidence"
        assert client.calls == manager.manager_llm_calls == 0
        manager = OnlineInsightManager(max_prompt_bytes=1)
        manager.after_episode(*evidence(0, False), client, root)
        event = manager.after_episode(*evidence(1, True), client, root)
        assert event["status"] == "skipped_oversize_prompt" and client.calls == 0
        for label, limit, truncated in (("full", 100, False), ("budget", 2, False), ("truncated", 100, True)):
            output = root/label
            output.mkdir()
            environment, client, manager = MockEnvironment(), MockClient(limit, truncated), OnlineInsightManager()
            try:
                result = run_agent(environment, manager, client, AgentConfig(episodes=3, max_steps=1), output)
            except ValueError:
                assert truncated
                result = json.loads((output/"summary.json").read_text())
            assert environment.closed
            if label == "full":
                assert client.calls == 4 and result["manager_llm_calls"] == 1
                assert result["completed_episodes"] == 3 and len(manager.bank.rules) == 1
                assert "Inspect before acting." in client.messages[-1][1][0]["content"]
                assert "Inspect before acting." not in client.messages[1][1][0]["content"]
                assert result["api"]["prompt_tokens"] == 200
            elif label == "budget":
                assert client.calls == 2 and result["status"] == "budget_exhausted"
                assert result["completed_episodes"] == 2 and result["manager_llm_calls"] == 0 and not manager.bank.rules
            else:
                assert result["failure_stage"] == "insight_critic" and result["manager_llm_calls"] == 1
                assert not manager.bank.rules
                event = json.loads((output/"critic.jsonl").read_text().splitlines()[-1])
                assert event["raw_response"] == "ADD: Inspect before acting." and event["usage"]["attempted_calls"] == 1
        output = root/"double"
        output.mkdir()
        environment, client = MockEnvironment(), MockClient()
        try:
            run_agent(environment, OnlineInsightManager(), client, AgentConfig(distill=True), output)
        except ValueError:
            pass
        else:
            raise AssertionError("double extraction accepted")
        assert environment.closed and client.calls == 0
        class InvalidClient(MockClient):
            def __init__(self, response):
                super().__init__()
                self.response = response

            def complete(self, messages, purpose="action"):
                reply = super().complete(messages, purpose)
                return self.response if purpose == "insight_critic" else reply

        for index, reply in enumerate(("not an operation", "ADD: Useful.\nEDIT 999: Bad target.", "ADD: Useful.\ncut off")):
            output = root/f"invalid-{index}"
            output.mkdir()
            environment, client, manager = MockEnvironment(), InvalidClient(reply), OnlineInsightManager()
            try:
                run_agent(environment, manager, client, AgentConfig(episodes=3, max_steps=1), output)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid critic continued")
            summary = json.loads((output/"summary.json").read_text())
            event = json.loads((output/"critic.jsonl").read_text().splitlines()[-1])
            assert summary["failure_stage"] == "insight_critic" and summary["completed_episodes"] == 2
            assert environment.closed and client.calls == 3 and manager.manager_llm_calls == 1 and not manager.bank.rules
            assert event["raw_response"] == reply and event["usage"]["attempted_calls"] == 1
        manager, client = OnlineInsightManager(), InvalidClient("NO_CHANGE")
        manager.after_episode(*evidence(0, False), client, root)
        event = manager.after_episode(*evidence(1, True), client, root)
        assert event["status"] == "no_change" and client.calls == 1 and not manager.bank.rules
    print("CRITIC_WITNESS chronology, same-task/success-batch, no oracle, oversize, runner shared-budget, truncation and double-extraction checks passed; no real API")


if __name__ == "__main__":
    main()
