from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from comep.client import APIConfig, BudgetExceeded
from comep.collection import run_collection, validate_collection_launch
from comep.environments import Observation
from comep.factorial import CostGuardClient
from prepare_experience_calibration import prepare


class MockClient:
    def __init__(self, mode="valid", limit=246):
        self.config = APIConfig("http://localhost", "deepseek-flash", max_calls=limit,
            input_price_per_million=.30, output_price_per_million=1.20)
        self.mode, self.calls, self.halted = mode, 0, False
        self.last_response = None
        self.requests = []

    def summary(self):
        return {"attempted_calls": self.calls, "prompt_tokens": self.calls*100, "completion_tokens": self.calls*20,
                "usage_complete": True, "estimated_cost_usd": self.calls*.000054}

    def complete(self, messages, purpose="action"):
        if self.calls >= self.config.max_calls:
            raise BudgetExceeded("fixture shared limit")
        self.calls += 1
        self.requests.append((purpose, deepcopy(messages)))
        reply = "Action: inspect"
        reason = "stop"
        if purpose == "experience_extraction":
            if self.mode == "provider_error":
                self.last_response = None
                raise RuntimeError("fixture charged failure")
            reply = json.dumps({"rule": "After inspecting, use observed names.", "kind": "task_success", "evidence": [{"step": 0, "quote": "Observed success."}]})
            if self.mode == "invalid":
                reply = "not JSON"
            elif self.mode == "abstain":
                reply = '{"rule": null, "evidence": []}'
            elif self.mode == "bad_quote":
                reply = reply.replace("Observed success.", "Invented success.")
            elif self.mode == "truncated":
                reason = "length"
        self.last_response = {"response_model": "deepseek-flash", "system_fingerprint": "fixture", "reply": reply, "finish_reason": reason}
        return reply


class MockEnvironment:
    def __init__(self, tasks):
        self.tasks, self.closed = tasks, False

    def reset(self, episode):
        task = self.tasks[episode]
        self.identifier = f"{task['task']}:{task['variation']}:train"
        return Observation("Observe visible objects.", self.identifier, task["task"], admissible=["ORACLE_SENTINEL"])

    def step(self, action):
        assert action == "inspect"
        return Observation("Observed success.", self.identifier, "context", True, True, 100.)

    def close(self):
        self.closed = True


def main():
    packet = json.loads(json.dumps(prepare()))
    approval = {"status": "approved", "operator_authorized": True, "scope": "six_train_experience_collection_v2",
        "credential_reuse_authorized": True, "credential_reuse_authorization_source": "explicit_user_instruction",
        "model": packet["model"], "pricing_source": "fixture", "pricing_checked_at_utc": datetime.now(timezone.utc).isoformat(), **packet["budget_proposal"]}
    review = {"code_protocol_ready": True, "review_scope": "six_train_experience_collection_v2"}
    validate_collection_launch(packet, approval, review, True)
    for changed, changed_review, credential in (({**approval, "scope": "factorial"}, review, True),
            ({**approval, "operator_authorized": False}, review, True), (approval, {"verdict": "go"}, True),
            (approval, review, False), ({**approval, "max_calls": 999}, review, True),
            ({**approval, "pricing_checked_at_utc": "2020-01-01T00:00:00+00:00"}, review, True)):
        try:
            validate_collection_launch(packet, changed, changed_review, credential)
        except ValueError:
            pass
        else:
            raise AssertionError("unready collection accepted")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for mode in ("valid", "invalid", "abstain", "bad_quote", "truncated", "provider_error", "budget", "initialization"):
            raw = MockClient(mode, limit=3 if mode == "budget" else 246)
            environments = []
            def factory(tasks):
                assert tasks == packet["stages"]["experience_train"] and all(task["split"] == "train" and task["variation"] in (1, 2) for task in tasks)
                if mode == "initialization":
                    raise RuntimeError("fixture initialization")
                environment = MockEnvironment(tasks)
                environments.append(environment)
                return environment
            output = root/mode
            try:
                run_collection(packet, CostGuardClient(raw, .25), factory, output, "mock_collection_contract")
            except RuntimeError:
                assert mode in ("truncated", "provider_error", "initialization")
            summary = json.loads((output/"collection-summary.json").read_text())
            assert all(environment.closed for environment in environments)
            assert all("Memories:" not in json.dumps(messages) and "ORACLE_SENTINEL" not in json.dumps(messages) for _, messages in raw.requests)
            if mode == "initialization":
                assert raw.calls == 0 and summary["failure_stage"] == "initialization"
                continue
            events = [json.loads(line) for line in (output/"candidates.jsonl").read_text().splitlines()]
            final_events = [row for row in events if row["status"] != "prepared"]
            if mode in ("valid", "invalid", "abstain", "bad_quote"):
                assert raw.calls == 12 and summary["completed_episodes"] == 6 and len(final_events) == 6
                assert summary["provenance_accepted"] == (6 if mode == "valid" else 0)
                assert [purpose for purpose, _ in raw.requests] == [purpose for _ in range(6) for purpose in ("action", "experience_extraction")]
                assert all("After inspecting" not in json.dumps(messages) for purpose, messages in raw.requests if purpose == "action")
            elif mode == "budget":
                assert raw.calls == 3 and summary["completed_episodes"] == 2 and summary["status"] == "budget_exhausted"
            else:
                assert raw.calls == 2 and summary["completed_episodes"] == 1 and summary["status"] == "error"
                assert final_events[-1]["usage"]["attempted_calls"] == 1 and final_events[-1]["status"] == "error"
                if mode == "truncated":
                    assert final_events[-1]["response_metadata"]["finish_reason"] == "length"
        output = root/"not_launched"
        result = subprocess.run([sys.executable, "scripts/run_experience_collection.py", "--packet", "outputs/experience-calibration-plan-20261103/proposal-v2.json",
            "--approval", "nonexistent", "--preflight-review", "nonexistent", "--output", str(output)], capture_output=True, text=True)
        assert result.returncode != 0 and "explicit --execute-paid required" in result.stderr and not output.exists()
        changed_packet = deepcopy(packet)
        changed_packet["stages"]["experience_train"][0]["variation"] = 0
        tampered = root/"changed.json"
        tampered.write_text(json.dumps(changed_packet))
        result = subprocess.run([sys.executable, "scripts/run_experience_collection.py", "--packet", str(tampered),
            "--approval", "nonexistent", "--preflight-review", "nonexistent", "--output", str(output), "--execute-paid"], capture_output=True, text=True)
        assert result.returncode != 0 and "frozen collection packet/source mismatch" in result.stderr and not output.exists()
        result = subprocess.run([sys.executable, "scripts/run_experience_collection.py", "--packet", "outputs/experience-calibration-plan-20261103/proposal-v2.json",
            "--approval", "configs/experience-collection-approval.template.json", "--preflight-review", "review-stage/EXPERIENCE_COLLECTION_PLAN_FOLLOWUP_20261103.json",
            "--output", str(output), "--execute-paid"], capture_output=True, text=True)
        assert result.returncode != 0 and "collection-specific operator approval" in result.stderr and not output.exists()
    print("COLLECTION_WITNESS six no-memory train episodes, no candidate/oracle prompt leakage, shared12calls, rejection/abstention, truncation/provider/budget stops, closure and no-dispatch gates passed; no real API")


if __name__ == "__main__":
    main()
