from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from comep.client import APIConfig, BudgetExceeded, ChatClient
from comep.environments import Observation
from comep.factorial import CostGuardClient, run_factorial, validate_schedule, validate_frozen_sources, validate_launch_readiness
from analyze_real_factorial import analyze


class MockClient:
    def __init__(self, limit=1000):
        self.config = APIConfig("http://localhost", "mock-contract", input_price_per_million=1., output_price_per_million=2.)
        self.calls = 0
        self.limit = limit
        self.halted = False
        self.messages = []

    def complete(self, messages, purpose="action"):
        if self.calls >= self.limit:
            raise BudgetExceeded("mock call ceiling")
        self.messages.append(deepcopy(messages))
        self.calls += 1
        return "Action: look around"

    def summary(self):
        return {"attempted_calls": self.calls, "prompt_tokens": self.calls*100, "completion_tokens": self.calls*10,
                "usage_complete": True, "estimated_cost_usd": self.calls*.00012}


class MockEnvironment:
    def __init__(self, trial):
        self.trial = trial
        self.closed = False

    def reset(self, episode):
        task = self.trial["task"]
        return Observation("Fresh task, not historical preflight.", f"{task['task']}:{task['variation']}:dev", task["task"])

    def step(self, action):
        assert action == "look around"
        initial = self.reset(0)
        success = self.trial["cell"] == "AB"
        return Observation("Mock terminal", initial.task_id, initial.context, True, success, 100. if success else 0.)

    def close(self):
        self.closed = True


def main():
    packet = json.loads(Path("outputs/real-factorial-preflight-20261103/schedule.json").read_text())
    validate_schedule(packet)
    pilot = json.loads(Path("outputs/aris-scienceworld-20261103/no-memory/manifest.json").read_text())
    rules = {row["id"]: row["text"] for row in json.loads(Path("configs/scienceworld-initial-rules.json").read_text())}
    validate_frozen_sources(packet, pilot["tasks"], rules, pilot["agent"]["action_instructions"])
    for mutation in ("rules", "tasks", "order", "prompt"):
        changed = deepcopy(packet)
        if mutation == "rules":
            for trial in changed["trials"]:
                trial["rules"] = ["Changed rule." for _ in trial["rules"]]
        elif mutation == "tasks":
            for trial in changed["trials"]:
                trial["task"] = pilot["tasks"][0]
        elif mutation == "order":
            changed["trials"][0], changed["trials"][1] = changed["trials"][1], changed["trials"][0]
            for index, trial in enumerate(changed["trials"]):
                trial["trial"] = index
        else:
            changed["trials"][0]["preflight_first_messages"][0]["content"] = "Changed prompt"
        try:
            validate_frozen_sources(changed, pilot["tasks"], rules, pilot["agent"]["action_instructions"])
        except ValueError:
            pass
        else:
            raise AssertionError(f"{mutation} drift accepted")
    approved_packet = {**packet, "priced_budget_approved": True, "model_provider_pricing_verified": True, "not_launchable_until": []}
    approval = {"status": "approved", "operator_authorized": True, "credential_rotated_confirmed": True, "model": packet["model"],
                "input_usd_per_million": .30, "output_usd_per_million": 1.20, "max_estimated_usd": .66,
                "pricing_source": "mock://verified-fixture-only", "pricing_checked_at_utc": datetime.now(timezone.utc).isoformat()}
    passed_review = {"code_protocol_ready": True, "review_scope": "bounded_descriptive_factorial_code_protocol", "paid_launch_ready": False}
    validate_launch_readiness(approved_packet, approval, passed_review, .30, 1.20, .66, True)
    reuse = {**approval, "credential_rotated_confirmed": False, "credential_reuse_authorized": True,
             "credential_reuse_authorization_source": "explicit_user_instruction"}
    validate_launch_readiness(approved_packet, reuse, passed_review, .30, 1.20, .66, True)
    for invalid in ({**reuse, "credential_reuse_authorized": False}, {**reuse, "credential_reuse_authorization_source": "inferred"}):
        try:
            validate_launch_readiness(approved_packet, invalid, passed_review, .30, 1.20, .66, True)
        except ValueError:
            pass
        else:
            raise AssertionError("unauthorized reuse accepted")
    for changed_packet, changed_approval, review, rate, credential in (
        (packet, approval, passed_review, .30, True), (approved_packet, {}, passed_review, .30, True),
        (approved_packet, approval, {"paid_launch_ready": False}, .30, True),
        (approved_packet, approval, {"code_protocol_ready": True}, .30, True),
        (approved_packet, approval, {"code_protocol_ready": False, "review_scope": "bounded_descriptive_factorial_code_protocol"}, .30, True),
        (approved_packet, approval, passed_review, 0., True), (approved_packet, approval, passed_review, .30, False)):
        try:
            validate_launch_readiness(changed_packet, changed_approval, review, rate, 1.20, .66, credential)
        except ValueError:
            pass
        else:
            raise AssertionError("unready launch accepted")
    bad = deepcopy(packet)
    bad["trials"][1] = deepcopy(bad["trials"][0])
    try:
        validate_schedule(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate accepted")
    with tempfile.TemporaryDirectory() as temp:
        class FakeResponse:
            def __init__(self, body):
                self.body = body

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, limit):
                return json.dumps(self.body).encode()

        for index, choices in enumerate(([], [{"message": None}], [{"message": {"content": None}}])):
            log_path = Path(temp)/f"malformed-envelope-{index}.jsonl"
            config = APIConfig("http://localhost", "mock-contract", key_env="COMEP_TEST_DUMMY_KEY",
                               input_price_per_million=.30, output_price_per_million=1.20)
            with patch.dict("os.environ", {"COMEP_TEST_DUMMY_KEY": "dummy-not-a-credential"}):
                raw_client = ChatClient(config, log_path)
            body = {"choices": choices, "usage": {"prompt_tokens": 123, "completion_tokens": 7}}
            with patch.object(raw_client.opener, "open", return_value=FakeResponse(body)):
                try:
                    raw_client.complete([{"role": "user", "content": "test"}])
                except ValueError:
                    pass
                else:
                    raise AssertionError("malformed envelope accepted")
            usage = raw_client.summary()
            assert raw_client.halted and usage["usage_complete"] and usage["attempted_calls"] == 1
            assert usage["prompt_tokens"] == 123 and usage["completion_tokens"] == 7
            assert abs(usage["estimated_cost_usd"]-.0000453) < 1e-12
            assert json.loads(log_path.read_text())["usage"] == body["usage"]
        output = Path(temp)/"complete"
        environments = []

        def factory(trial):
            environment = MockEnvironment(trial)
            environments.append(environment)
            return environment

        raw = MockClient()
        client = CostGuardClient(raw, 1.)
        records = run_factorial(packet, client, factory, output, "mock commands", kind="mock_contract_only")
        assert len(records) == raw.calls == len(environments) == 48
        assert all(environment.closed for environment in environments)
        for trial, messages in zip(packet["trials"], raw.messages):
            assert messages[-1]["content"] == "Fresh task, not historical preflight."
            assert all(rule in messages[0]["content"] for rule in trial["rules"])
            if trial["cell"] == "none":
                assert "Memories:" not in messages[0]["content"]
        result = analyze(output)
        assert result["complete_balanced"] and result["overall"]["reward"]["mean"] == 1.
        assert all(row["usage"]["attempted_calls"] == 1 for row in records)
        raw = MockClient(limit=3)
        partial = Path(temp)/"partial"
        run_factorial(packet, CostGuardClient(raw, 1.), factory, partial, "mock", kind="mock_contract_only")
        assert raw.calls == 3
        result = analyze(partial)
        assert not result["complete_balanced"] and result["overall"] is None and result["completed_cells"] == 3
        raw = MockClient()
        guard = CostGuardClient(raw, .000001)
        try:
            guard.complete([{"role": "user", "content": "hello"}])
        except BudgetExceeded:
            pass
        else:
            raise AssertionError("unfunded reservation allowed")
        assert raw.calls == 0
        raw = MockClient()
        def fail_factory(trial):
            raise RuntimeError("fixture initialization failure")
        failed = Path(temp)/"failed"
        try:
            run_factorial(packet, CostGuardClient(raw, 1.), fail_factory, failed, "mock", kind="mock_contract_only")
        except RuntimeError:
            pass
        else:
            raise AssertionError("initialization failure concealed")
        summary = json.loads((failed/"summary.json").read_text())
        assert summary["status"] == "error" and len(summary["rows"]) == 1 and summary["rows"][0]["episode"] is None
        assert summary["api"]["attempted_calls"] == 0
        assert summary["rows"][0]["failure_stage"] == "initialization"
        class ResetErrorEnvironment(MockEnvironment):
            def reset(self, episode):
                raise RuntimeError("fixture reset exception")
        reset_environments = []
        def reset_factory(trial):
            environment = ResetErrorEnvironment(trial)
            reset_environments.append(environment)
            return environment
        reset_error = Path(temp)/"reset_error"
        try:
            run_factorial(packet, CostGuardClient(MockClient(), 1.), reset_factory, reset_error, "mock", kind="mock_contract_only")
        except RuntimeError:
            pass
        else:
            raise AssertionError("reset error concealed")
        reset_summary = json.loads((reset_error/"summary.json").read_text())
        assert reset_summary["rows"][0]["failure_stage"] == "environment_reset"
        assert reset_summary["api"]["attempted_calls"] == 0 and all(environment.closed for environment in reset_environments)
        class MalformedClient(MockClient):
            def complete(self, messages, purpose="action"):
                super().complete(messages, purpose)
                return "No action available"
        malformed = Path(temp)/"malformed"
        try:
            run_factorial(packet, CostGuardClient(MalformedClient(), 1.), factory, malformed, "mock", kind="mock_contract_only")
        except ValueError:
            pass
        else:
            raise AssertionError("malformed reply accepted")
        assert "No action available" in (malformed/"trial-000"/"replies.jsonl").read_text()
        assert json.loads((malformed/"summary.json").read_text())["rows"][0]["failure_stage"] == "parse"
        class TruncatedClient(MockClient):
            def complete(self, messages, purpose="action"):
                reply = super().complete(messages, purpose)
                self.last_response = {"response_model": "mock", "system_fingerprint": None, "finish_reason": "length", "reply": reply}
                return reply
        truncated = Path(temp)/"truncated"
        try:
            run_factorial(packet, CostGuardClient(TruncatedClient(), 1.), factory, truncated, "mock", kind="mock_contract_only")
        except RuntimeError:
            pass
        else:
            raise AssertionError("truncated generation accepted")
        assert '"finish_reason": "length"' in (truncated/"trial-000"/"responses.jsonl").read_text()
        class StepErrorEnvironment(MockEnvironment):
            def step(self, action):
                raise RuntimeError("fixture step exception")
        step_error = Path(temp)/"step_error"
        try:
            run_factorial(packet, CostGuardClient(MockClient(), 1.), StepErrorEnvironment, step_error, "mock", kind="mock_contract_only")
        except RuntimeError:
            pass
        else:
            raise AssertionError("step error concealed")
        assert (step_error/"trial-000"/"replies.jsonl").exists() and (step_error/"trial-000"/"parsed-actions.jsonl").exists()
        assert json.loads((step_error/"summary.json").read_text())["rows"][0]["failure_stage"] == "environment"
        class ChangedModelClient(MockClient):
            def complete(self, messages, purpose="action"):
                reply = super().complete(messages, purpose)
                self.last_response = {"response_model": "first" if self.calls == 1 else "second", "system_fingerprint": None,
                                      "finish_reason": "stop", "reply": reply}
                return reply
        raw = ChangedModelClient()
        guard = CostGuardClient(raw, 1.)
        guard.complete([{"role": "user", "content": "test"}])
        try:
            guard.complete([{"role": "user", "content": "test"}])
        except RuntimeError:
            pass
        else:
            raise AssertionError("backend drift ignored")
        import time
        raw = MockClient()
        guard = CostGuardClient(raw, 1.)
        guard.last_completed_at = time.monotonic()-301
        try:
            guard.complete([{"role": "user", "content": "test"}])
        except RuntimeError:
            pass
        else:
            raise AssertionError("interrupted campaign silently continued")
        assert raw.calls == 0
    print("FACTORIAL_WITNESS 48mockcells, freshobservations/exactmemories, sharedbudget, nofillpartial, closures and pre-requestcostguard passed; no real API")


if __name__ == "__main__":
    main()
