import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from comep.core import Config, MemoryManager, Parameter, Selection


def check_core():
    for bad in ({"bank_tokens": 0}, {"exploration": 2}, {"sampling_scale": float("nan")}):
        try:
            Config(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid config accepted: {bad}")
    config = Config(retrieval_tokens=8, bank_tokens=12, max_items=2, max_bank_items=3, sampling_scale=0, exploration=0, redundancy=0, pair_l1=0)
    manager = MemoryManager(config=config)
    for identifier in ("first", "second"):
        manager.add(identifier, "navigate room safely")
        manager.individual[identifier].mean = -0.2
    manager.pairs[("first", "second")] = Parameter(mean=1)
    selection = manager.choose("navigate room safely")
    assert len(selection.identifiers) == 2, "positive synergy must survive singleton early stop"
    before = manager.pairs[("first", "second")].mean
    manager.observe(selection, 0)
    assert manager.pairs[("first", "second")].mean < before
    manager.pairs[("first", "second")] = Parameter(mean=-10, precision=100, count=20)
    assert manager.conflict("first", "second")
    assert len(manager.choose("navigate room safely").identifiers) <= 1
    for identifier in range(10):
        manager.add(f"new-{identifier}", "navigate room safely", identifier)
        assert manager.bank_tokens <= config.bank_tokens
        assert len(manager.memories) <= config.max_bank_items
        assert all(set(pair) <= set(manager.memories) for pair in manager.pairs)
    manager.add("oversized", "word " * 30)
    assert "oversized" not in manager.memories
    for method in ("no-memory", "full-memory", "individual", "mmr", "additive", "proposal"):
        alternative = MemoryManager(method, config)
        alternative.add("rule", "navigate room safely")
        chosen = alternative.choose("navigate room safely")
        alternative.observe(chosen, 1)
        assert chosen.tokens <= config.retrieval_tokens
        json.dumps(alternative.snapshot(), allow_nan=False)
    twins = [MemoryManager(seed=42), MemoryManager(seed=42)]
    for twin in twins:
        twin.add("rule", "navigate room safely")
    assert twins[0].choose("navigate").identifiers == twins[1].choose("navigate").identifiers
    stable = MemoryManager(config=Config(exploration=0, sampling_scale=0, redundancy=0))
    stable.add("rule", "navigate")
    observation = Selection(["rule"], {"rule": 1.0}, "easy", 1, False)
    initial = stable.predict(observation)
    for _ in range(40):
        stable.observe(observation, 1)
    assert stable.predict(observation) > initial
    assert np.isfinite(stable.individual["rule"].mean)
    grace = MemoryManager(config=Config(bank_tokens=20, max_bank_items=1, admission_grace_episodes=3))
    grace.add("incumbent", "navigate safely", step=900)
    for _ in range(3):
        grace.observe(Selection([], {}, "default", 0, False), 1)
    grace.individual["incumbent"].mean = 10
    grace.add("arrival", "navigate carefully and inspect the room first", step=1)
    assert "arrival" in grace.memories and "incumbent" not in grace.memories
    grace.add("oversize", "word " * 21)
    assert "arrival" in grace.memories and "oversize" not in grace.memories
    for _ in range(3):
        grace.observe(Selection([], {}, "default", 0, False), 0)
    grace.add("next", "navigate safely", step=0)
    assert "next" in grace.memories
    assert grace.bank_tokens <= 20 and len(grace.memories) == 1
    all_young = MemoryManager(config=Config(max_bank_items=1, admission_grace_episodes=3))
    all_young.add("first", "first rule", step=0)
    all_young.add("second", "second rule", step=1)
    assert list(all_young.memories) == ["second"]
    for bad in ({"admission_grace_episodes": -1}, {"pruning_redundancy": float("nan")}):
        try:
            Config(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid admission or redundancy configuration accepted")
    # Verify EvoMemGuard explicit NLI conflict guard, duplicate compression, and budget compliance
    evo = MemoryManager("evomemguard", Config(bank_tokens=60, max_bank_items=4, retrieval_tokens=30, max_items=2))
    evo.add("r1", "Always heat the mug in the microwave before placing it on the counter.")
    evo.add("r2", "Never heat the mug in the microwave; skip microwave heating completely.")
    assert evo.conflict("r1", "r2"), "evomemguard must detect explicit directive contradiction"
    sel_evo = evo.choose("heat the mug in the microwave")
    assert not ("r1" in sel_evo.identifiers and "r2" in sel_evo.identifiers), "evomemguard must not co-select conflicting rules"
    evo.observe(sel_evo, 1)
    evo.add("r1_dup", "Always heat the mug in the microwave before placing it on the counter today.")
    assert "r1_dup" not in evo.memories, "evomemguard compressor must reject verbose near-duplicate"
    json.dumps(evo.snapshot(), allow_nan=False)
    # Verify Laplace retroactive-blame protection when a toxic partner is evicted
    blame = MemoryManager("comep", Config(bank_tokens=100, max_bank_items=2, exploration=0, sampling_scale=0))
    blame.add("good", "clean mug at sinkbasin directly")
    blame.add("toxic", "dump dirty mug in trashcan")
    for _ in range(4):
        blame.observe(Selection(["good"], {"good": 1.0}, "clean", 5, False), 1)
    for _ in range(4):
        blame.observe(Selection(["good", "toxic"], {"good": 1.0, "toxic": 1.0}, "clean", 10, False), 0)
    good_before = blame.individual["good"].mean
    assert good_before > 0.3 and blame.individual["toxic"].mean < -0.3
    blame.add("neutral", "inspect countertop area")
    assert "toxic" not in blame.memories and "good" in blame.memories
    blame.observe(Selection(["good"], {"good": 1.0}, "clean", 5, False), 1)
    assert blame.individual["good"].mean >= good_before - 0.1, "evicting toxic rule must not dump retroactive blame onto survivor"
    print("core checks passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["core", "synthetic", "integration", "all"])
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "smoke.json"
        subprocess.run([sys.executable, "-m", "comep", "smoke", "--output", str(output)], check=True)
        assert json.loads(output.read_text())["kind"] == "spine_check"
    check_core()
    if args.stage in ("synthetic", "integration", "all"):
        check_synthetic()
    if args.stage in ("integration", "all"):
        check_integration()


def check_synthetic():
    from comep.experiments import run_synthetic

    with tempfile.TemporaryDirectory() as directory:
        settings = dict(methods=["no-memory", "full-memory", "individual", "mmr", "additive", "proposal", "evomemguard", "comep"], seeds=[1, 2], episodes=12, size=10, arrival_every=3, config=Config(bank_tokens=75, max_bank_items=6, retrieval_tokens=35, max_items=3))
        first = run_synthetic(Path(directory) / "first", **settings)
        second = run_synthetic(Path(directory) / "second", **settings)
        assert first == second
        agent_run = run_synthetic(Path(directory) / "agent", scenario="agent-composite", **settings)
        assert len(agent_run) == 8
        for method in settings["methods"]:
            events = [json.loads(line) for line in (Path(directory) / "first" / f"{method}-seed-1.jsonl").read_text().splitlines()]
            assert len(events) == 12
            assert all(event["retrieval_tokens"] <= 35 for event in events)
            if method != "full-memory":
                assert all(event["bank_tokens"] <= 75 for event in events)
            if method == "no-memory":
                assert all(not event["selected"] for event in events)
        try:
            run_synthetic(Path(directory) / "first", **settings)
        except ValueError:
            pass
        else:
            raise AssertionError("experiment overwrote existing artifacts")
    print("synthetic checks passed")


def check_integration():
    from comep.client import APIConfig, BudgetExceeded, ChatClient
    from comep.environments import ALFWorldAdapter, Observation, ScienceWorldAdapter
    from comep.runner import AgentConfig, parse_action, run_agent

    class Handler(BaseHTTPRequestHandler):
        responses = []
        requests = []

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            assert self.headers["Authorization"] == "Bearer local-test-only"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.requests.append(body)
            response = self.responses.pop(0)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def log_message(self, *args):
            pass

    class Environment:
        closed = False

        def reset(self, episode):
            return Observation("Move north to finish", f"task-{episode}", "navigation", admissible=["north"])

        def step(self, action):
            assert action == "north"
            return Observation("Finished", "task-0", "navigation", True, True, 1)

        def close(self):
            self.closed = True

    def response(content, usage=True):
        body = {"choices": [{"message": {"content": content}}]}
        if usage:
            body["usage"] = {"prompt_tokens": 10, "completion_tokens": 5}
        return body

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    old_key = os.environ.get("COMEP_TEST_KEY")
    os.environ["COMEP_TEST_KEY"] = "local-test-only"
    try:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            api = APIConfig(f"http://127.0.0.1:{server.server_port}/v1", "mock-model", key_env="COMEP_TEST_KEY", max_calls=2, input_price_per_million=1, output_price_per_million=2, thinking="disabled")
            client = ChatClient(api, output / "api.jsonl")
            Handler.responses = [response("Thought: move\nAction: north"), response('{"rule":"Move north when the goal is north."}')]
            environment = Environment()
            result = run_agent(environment, MemoryManager(), client, AgentConfig(episodes=1, distill=True), output)
            assert result["status"] == "completed" and result["success_rate"] == 1
            assert result["api"]["attempted_calls"] == 2
            assert abs(result["api"]["estimated_cost_usd"] - 0.00004) < 1e-10
            assert environment.closed
            assert len(json.loads((output / "state.json").read_text())["memories"]) == 1
            assert "local-test-only" not in (output / "api.jsonl").read_text()
            try:
                client.complete([{"role": "user", "content": "more"}])
            except BudgetExceeded:
                pass
            else:
                raise AssertionError("call cap did not stop request")
            assert len(Handler.requests) == 2
            assert Handler.requests[0]["thinking"] == {"type": "disabled"}
            try:
                APIConfig(api.base_url, "mock-model", thinking="invalid")
            except ValueError:
                pass
            else:
                raise AssertionError("invalid thinking mode accepted")
            check_output = output / "cli-check"
            check_config = output / "api.toml"
            check_config.write_text(f'[api]\nbase_url = "{api.base_url}"\nmodel = "mock-model"\nkey_env = "COMEP_TEST_KEY"\nthinking = "disabled"\nmax_calls = 1\n')
            Handler.responses = [response("Action: look")]
            subprocess.run([sys.executable, "-m", "comep", "api-check", "--config", str(check_config), "--output", str(check_output)], check=True, capture_output=True)
            assert json.loads((check_output / "summary.json").read_text())["status"] == "ok"
            missing_usage = ChatClient(api, output / "missing.jsonl")
            Handler.responses = [response("Action: north", usage=False)]
            try:
                missing_usage.complete([{"role": "user", "content": "hello"}])
            except ValueError:
                assert not missing_usage.summary()["usage_complete"]
            else:
                raise AssertionError("missing provider usage accepted")
            partial_output = output / "partial"
            partial_output.mkdir()
            limited = ChatClient(APIConfig(api.base_url, "mock-model", key_env=api.key_env, max_total_tokens=1), partial_output / "api.jsonl")
            partial = run_agent(Environment(), MemoryManager(), limited, AgentConfig(episodes=1), partial_output)
            assert partial["status"] == "budget_exhausted" and partial["success_rate"] is None
            assert limited.calls == 0
            timeout_output = output / "timeout"
            timeout_output.mkdir()

            class TimeoutEnvironment(Environment):
                def step(self, action):
                    return Observation("Still in room", "task-0", "navigation")

            Handler.responses = [response("Action: north")]
            timeout_client = ChatClient(api, timeout_output / "api.jsonl")
            timeout_result = run_agent(TimeoutEnvironment(), MemoryManager(), timeout_client, AgentConfig(episodes=1, max_steps=1), timeout_output)
            assert timeout_result["status"] == "completed"
            assert timeout_result["step_limit_episodes"] == 1 and timeout_result["success_rate"] == 0
            grounded_output = output / "grounded"
            grounded_output.mkdir()
            Handler.responses = [response("Action: north"), response(json.dumps({"rule": "Moving north completed this task.", "kind": "task_success", "evidence": [{"step": 0, "quote": "Finished"}]}))]
            grounded = run_agent(Environment(), MemoryManager(), ChatClient(api, grounded_output / "api.jsonl"), AgentConfig(episodes=1, distill=True, evidence_distillation=True), grounded_output)
            assert grounded["status"] == "completed"
            record = json.loads((grounded_output / "distillation.jsonl").read_text())
            assert record["accepted"] and record["retained_after_pruning"] and not record["evidence_validation"]["semantic_verified"]

            class FailedEnvironment(Environment):
                def step(self, action):
                    return Observation("Wrong target", "task-0", "navigation", True, False, -100)

            refused_output = output / "grounding-refused"
            refused_output.mkdir()
            Handler.responses = [response("Action: north"), response(json.dumps({"rule": "Always move north to win.", "kind": "observation", "evidence": [{"step": 0, "quote": "Wrong target"}]}))]
            run_agent(FailedEnvironment(), MemoryManager(), ChatClient(api, refused_output / "api.jsonl"), AgentConfig(episodes=1, distill=True, evidence_distillation=True), refused_output)
            record = json.loads((refused_output / "distillation.jsonl").read_text())
            assert not record["accepted"] and record["evidence_validation"]["reason"] == "positive_evidence_from_failed_terminal"
        alf = object.__new__(ALFWorldAdapter)
        observed = alf.unpack(["done"], {"extra.gamefile": ["game"], "won": [True]}, True)
        assert observed.success and observed.task_id == "game"

        class AlfIdentityEnvironment:
            def __init__(self, info):
                self.info = info

            def step(self, actions):
                return ["room"], [0], [False], self.info

        for info in ({}, {"extra.gamefile": [None]}, {"extra.gamefile": ["game"]}):
            alf.env = AlfIdentityEnvironment(info)
            assert alf.step("look").task_id == "game"
        alf.env = AlfIdentityEnvironment({"extra.gamefile": ["different"]})
        try:
            alf.step("look")
        except ValueError:
            pass
        else:
            raise AssertionError("changed game identity accepted")
        try:
            alf.unpack(["room"], {"extra.gamefile": [None]})
        except ValueError:
            pass
        else:
            raise AssertionError("missing reset identity accepted")
        science = object.__new__(ScienceWorldAdapter)
        science.current_task = {"task": "test", "variation": 1, "split": "validation"}

        class ScienceMetadata:
            def taskdescription(self):
                return "Measure temperature"

            def get_valid_action_object_combinations(self):
                return ["look around"]

        science.env = ScienceMetadata()
        assert not science.unpack("failed", {"score": -1}, True).success
        assert science.unpack("won", {"score": 100}, True).success
        assert "Measure temperature" in science.unpack("room", {"score": 0}).text
        assert science.unpack("room", {"score": 0}).admissible == ["look around"]

        class SciencePartitions(ScienceMetadata):
            def load(self, task, variation, simplificationStr="", generateGoldPath=False):
                assert generateGoldPath is False

            def get_variations_dev(self):
                return [7, 8]

        science.env = SciencePartitions()
        science.tasks = [{"task": "test", "variation": 7, "split": "dev"}]
        science.verify_splits()
        assert science.split_verified
        science.tasks = [{"task": "test", "variation": 2, "split": "dev"}]
        try:
            science.verify_splits()
        except ValueError:
            pass
        else:
            raise AssertionError("wrong official partition accepted")
        assert parse_action("Thought: plan\nAction: go north") == "go north"
        assert parse_action("**Action:** `clean mug 1 with sinkbasin 1`.") == "clean mug 1 with sinkbasin 1"
        assert parse_action('- **Action**: "go north"') == "go north"
        from comep.environments import _infer_alfworld_context
        assert _infer_alfworld_context("You are in a room.\nYour task is to: clean some spoon and put it in diningtable.") == "alfworld:clean"
        assert _infer_alfworld_context("You are in a room.\nYour task is to: put a hot mug in coffeemachine.") == "alfworld:heat"
        assert _infer_alfworld_context("You are in a room.\nYour task is to: cool some apple and put it in garbagecan.") == "alfworld:cool"
        assert _infer_alfworld_context("You are in a room.\nYour task is to: look at alarmclock under the desklamp.") == "alfworld:examine"
        assert _infer_alfworld_context("You are in a room.\nYour task is to: put two peppershaker in cabinet.") == "alfworld:two_obj"
        assert _infer_alfworld_context("You are in a room.\nYour task is to: put some pillow on ottoman.") == "alfworld:put"
        assert _infer_alfworld_context("done") == "alfworld"
        try:
            parse_action("no command")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid action response accepted")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        if old_key is None:
            os.environ.pop("COMEP_TEST_KEY", None)
        else:
            os.environ["COMEP_TEST_KEY"] = old_key
    print("integration checks passed (local HTTP and adapter contracts, not installed benchmarks)")


if __name__ == "__main__":
    main()
