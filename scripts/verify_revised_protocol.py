"""Offline zero-API regression suite for the v2.2 revised ablation evaluation protocol."""
from __future__ import annotations

import http.server
import json
import pickle
import socketserver
import sys
import tempfile
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from comep.campaign import ABLATION_SPECS, build_manager, compute_spec_fingerprint, execute_worker, invoke_worker_subprocess
from comep.client import APIConfig, ChatClient, complete_and_validate
from comep.runner import (
    ACTION_FORMAT_CORRECTION_PROMPT,
    DISTILLATION_FORMAT_CORRECTION_PROMPT,
    parse_action,
    validate_distillation_json,
)


class _MockServer:
    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.call_count = 0
        self._lock = threading.Lock()
        parent = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                _ = self.rfile.read(length)
                with parent._lock:
                    idx = min(parent.call_count, len(parent.responses) - 1)
                    body = parent.responses[idx]
                    parent.call_count += 1
                raw = json.dumps(body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, format, *args):
                return

        self.httpd = socketserver.TCPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def _ok_body(content: str, pt: int = 100, ct: int = 20, fr: str = "stop") -> dict:
    return {
        "model": "mock-deepseek",
        "system_fingerprint": "fp_1",
        "usage": {"prompt_tokens": pt, "completion_tokens": ct},
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": fr}],
    }


def test_1_shared_cache_and_replicate_isolation():
    server = _MockServer([_ok_body("Thought: ok\nAction: look", 120, 15), _ok_body("Thought: s1\nAction: inventory", 120, 15)])
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            cache_dir = str(tmp_p / "cache")
            cfg0_a = APIConfig(
                base_url=server.base_url,
                model="mock-deepseek",
                key_env="TEST_API_KEY",
                input_price_per_million=1.0,
                output_price_per_million=2.0,
                cache_dir=cache_dir,
                cache_namespace="alfworld:seed_0",
            )
            cfg0_b = APIConfig(
                base_url=server.base_url,
                model="mock-deepseek",
                key_env="TEST_API_KEY",
                input_price_per_million=1.0,
                output_price_per_million=2.0,
                cache_dir=cache_dir,
                cache_namespace="alfworld:seed_0",
            )
            cfg1 = APIConfig(
                base_url=server.base_url,
                model="mock-deepseek",
                key_env="TEST_API_KEY",
                input_price_per_million=1.0,
                output_price_per_million=2.0,
                cache_dir=cache_dir,
                cache_namespace="alfworld:seed_1",
            )
            import os

            os.environ["TEST_API_KEY"] = "dummy-key"
            c0_a = ChatClient(cfg0_a, tmp_p / "c0a.jsonl")
            c0_b = ChatClient(cfg0_b, tmp_p / "c0b.jsonl")
            c1 = ChatClient(cfg1, tmp_p / "c1.jsonl")

            msgs = [{"role": "user", "content": "Task 0"}]
            r_a = c0_a.complete(msgs, "action", request_context={"episode": 0, "step": 0})
            r_b = c0_b.complete(msgs, "action", request_context={"episode": 0, "step": 0})
            assert r_a == r_b == "Thought: ok\nAction: look"
            assert server.call_count == 1, f"Expected 1 HTTP call for shared replicate, got {server.call_count}"
            assert c0_a.summary()["http_requests"] == 1 and c0_a.summary()["cache_hits"] == 0
            assert c0_b.summary()["http_requests"] == 0 and c0_b.summary()["cache_hits"] == 1
            # Test 3 part A: full-cache client has provider_reported_cost_usd == 0.0 (not estimated_cost_usd!)
            assert c0_b.summary()["provider_reported_cost_usd"] == 0.0
            assert c0_b.summary()["estimated_cost_usd"] > 0.0

            # Different replicate namespace must NOT share cache
            r_1 = c1.complete(msgs, "action", request_context={"episode": 0, "step": 0})
            assert r_1 == "Thought: s1\nAction: inventory"
            assert server.call_count == 2

            # Concurrent lock contention test across simultaneous threads/clients on a brand-new key
            barrier = threading.Barrier(2)
            results_conc = []

            def _worker(idx: int):
                cc = ChatClient(cfg0_a, tmp_p / f"conc_{idx}.jsonl")
                barrier.wait()
                res = cc.complete([{"role": "user", "content": "Concurrent task"}], "action", request_context={"episode": 9, "step": 0})
                results_conc.append((res, cc.summary()))

            t1 = threading.Thread(target=_worker, args=(0,))
            t2 = threading.Thread(target=_worker, args=(1,))
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            assert len(results_conc) == 2
            assert results_conc[0][0] == results_conc[1][0]
            assert server.call_count == 3, f"Expected exactly 1 new HTTP call under concurrent lock contention, total={server.call_count}"
    finally:
        server.close()


def test_2_format_retries_and_identical_malformed_outputs():
    # Two identical malformed replies followed by a valid Action on 3rd attempt
    server = _MockServer(
        [
            _ok_body("I am thinking without an action line", 80, 10),
            _ok_body("I am thinking without an action line", 100, 10),
            _ok_body("Thought: fixed\nAction: open cabinet 1", 120, 12),
            _ok_body("{\"rule\": null}", 90, 8),
        ]
    )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            cache_dir = str(tmp_p / "cache")
            import os

            os.environ["TEST_API_KEY"] = "dummy-key"
            cfg = APIConfig(
                base_url=server.base_url,
                model="mock-deepseek",
                key_env="TEST_API_KEY",
                max_format_retries=2,
                cache_dir=cache_dir,
                cache_namespace="alfworld:seed_0",
            )
            c1 = ChatClient(cfg, tmp_p / "c1.jsonl")
            msgs = [{"role": "user", "content": "Step 0"}]
            reply, parsed, meta = complete_and_validate(
                c1,
                msgs,
                "action",
                parse_action,
                ACTION_FORMAT_CORRECTION_PROMPT,
                request_context={"episode": 0, "step": 0},
            )
            assert parsed == "open cabinet 1"
            assert meta["format_attempts"] == 3 and meta["format_retries"] == 2 and not meta["format_fallback"]
            assert server.call_count == 3, f"Expected 3 distinct HTTP calls despite identical malformed text, got {server.call_count}"

            # Second client on same replicate hits all 3 cached attempts without any new HTTP call
            c2 = ChatClient(cfg, tmp_p / "c2.jsonl")
            reply2, parsed2, meta2 = complete_and_validate(
                c2,
                msgs,
                "action",
                parse_action,
                ACTION_FORMAT_CORRECTION_PROMPT,
                request_context={"episode": 0, "step": 0},
            )
            assert parsed2 == "open cabinet 1" and meta2["format_attempts"] == 3
            assert server.call_count == 3 and c2.summary()["cache_hits"] == 3 and c2.summary()["http_requests"] == 0

            # Valid {"rule": null} must NOT retry
            _, dist_parsed, dist_meta = complete_and_validate(
                c1,
                [{"role": "user", "content": "Distill"}],
                "distillation",
                validate_distillation_json,
                DISTILLATION_FORMAT_CORRECTION_PROMPT,
                request_context={"episode": 0},
            )
            assert dist_parsed == {"rule": None}
            assert dist_meta["format_attempts"] == 1 and dist_meta["format_retries"] == 0
            assert server.call_count == 4
    finally:
        server.close()


def test_3_http_ledger_survives_checkpoint_rollback():
    server = _MockServer([_ok_body("Action: look", 100, 20), _ok_body("Action: look", 150, 30)])
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            import os

            os.environ["TEST_API_KEY"] = "dummy-key"
            cfg = APIConfig(
                base_url=server.base_url,
                model="mock-deepseek",
                key_env="TEST_API_KEY",
                input_price_per_million=1.0,
                output_price_per_million=2.0,
            )
            client = ChatClient(cfg, tmp_p / "api.jsonl")
            client.complete([{"role": "user", "content": "Ep 0"}], "action")
            # Save checkpoint after Ep 0
            ckpt = {
                "calls": client.calls,
                "prompt_tokens": client.prompt_tokens,
                "completion_tokens": client.completion_tokens,
            }
            # Now make an HTTP call in Ep 1 before a simulated crash
            client.complete([{"role": "user", "content": "Ep 1 before crash"}], "action")
            assert client.http_requests == 2

            # Simulate worker restart rolling back logical counters to Ep 0 checkpoint while syncing HTTP ledger
            resumed = ChatClient(cfg, tmp_p / "api.jsonl")
            resumed.calls = ckpt["calls"]
            resumed.prompt_tokens = ckpt["prompt_tokens"]
            resumed.completion_tokens = ckpt["completion_tokens"]
            resumed.sync_http_ledger()
            s = resumed.summary()
            assert s["attempted_calls"] == 1
            assert s["http_requests"] == 2
            assert s["provider_reported_prompt_tokens"] == 250
            assert s["provider_reported_completion_tokens"] == 50
            assert s["http_usage_complete"] is True

            # Simulate an interrupted mid-flight HTTP request (phase="started" with no matching "finished")
            with resumed.http_ledger_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"phase": "started", "attempt_id": "c99_interrupted", "http_request": 3}) + "\n")
            resumed.sync_http_ledger()
            s2 = resumed.summary()
            assert s2["http_requests"] == 3
            assert s2["http_usage_complete"] is False
            assert s2["provider_reported_cost_usd"] is None
    finally:
        server.close()


def test_4_protocol_fingerprint_rejects_mismatched_reuse():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        man_file = tmp_p / "tasks.json"
        man_file.write_text(json.dumps([{"task_id": "t0"}]), encoding="utf-8")
        base_spec = {
            "env_name": "scienceworld",
            "split": "test",
            "manifest_path": str(man_file),
            "seed": 0,
            "episodes": 5,
            "max_steps": 10,
            "method": "comep",
            "initial_rules": [{"id": "r0", "text": "Rule zero"}, {"id": "r1", "text": "Rule one"}],
            "distill": True,
            "admissible_limit": 0,
            "action_instructions": "Act.",
            "manager": {},
            "output_dir": str(tmp_p / "run"),
            "api": {"base_url": "https://api.deepseek.com", "model": "deepseek-flash", "protocol_version": "v2.2"},
        }
        fp_base = compute_spec_fingerprint(base_spec)
        # Changing rule order, rule ID, manager override, model, or manifest content must change fingerprint
        reordered = {**base_spec, "initial_rules": list(reversed(base_spec["initial_rules"]))}
        assert compute_spec_fingerprint(reordered) != fp_base
        diff_mgr = {**base_spec, "manager": {"warmup_gate": True}}
        assert compute_spec_fingerprint(diff_mgr) != fp_base
        diff_model = {**base_spec, "api": {**base_spec["api"], "model": "other-model"}}
        assert compute_spec_fingerprint(diff_model) != fp_base

        # Write a completed summary with fp_base, then verify execute_worker & invoke_worker_subprocess reject diff_mgr
        run_dir = Path(base_spec["output_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "worker_spec.json").write_text(json.dumps({**base_spec, "spec_fingerprint": fp_base}), encoding="utf-8")
        (run_dir / "summary.json").write_text(
            json.dumps({"status": "completed", "completed_episodes": 5, "spec_fingerprint": fp_base}),
            encoding="utf-8",
        )
        # Matching spec reuses summary cleanly
        res = execute_worker(run_dir / "worker_spec.json")
        assert res["spec_fingerprint"] == fp_base

        # Mismatched spec must raise RuntimeError in invoke_worker_subprocess
        raised = False
        try:
            invoke_worker_subprocess(diff_mgr)
        except RuntimeError:
            raised = True
        assert raised, "Expected RuntimeError when attempting to overwrite/reuse run with mismatched spec_fingerprint"

        # Mismatched checkpoint must raise RuntimeError in execute_worker
        (run_dir / "summary.json").unlink()
        with (run_dir / "checkpoint.pkl").open("wb") as fh:
            pickle.dump({"spec_fingerprint": "wrong_fp", "next_episode": 1}, fh)
        raised_ckpt = False
        try:
            import os

            os.environ["OPENAI_API_KEY"] = "dummy"
            execute_worker(run_dir / "worker_spec.json")
        except RuntimeError:
            raised_ckpt = True
        assert raised_ckpt, "Expected RuntimeError when resuming checkpoint with mismatched spec_fingerprint"


def test_5_legacy_sgd_v1_and_ablation_specs():
    rules = [{"id": f"r{i}", "text": f"For task {i}, open cabinet {i} and clean mug {i}."} for i in range(6)]
    for name in ABLATION_SPECS:
        mgr = build_manager(name, {}, seed=0, initial_rules=rules)
        sel = mgr.choose("Your task is to: clean mug.", "alfworld:clean")
        mgr.observe(sel, 1)
        snap = mgr.snapshot()
        assert snap["trigger_counts"]["episodes_routed"] == 1


if __name__ == "__main__":
    test_1_shared_cache_and_replicate_isolation()
    test_2_format_retries_and_identical_malformed_outputs()
    test_3_http_ledger_survives_checkpoint_rollback()
    test_4_protocol_fingerprint_rejects_mismatched_reuse()
    test_5_legacy_sgd_v1_and_ablation_specs()
    print("ALL 5 REVISED PROTOCOL REGRESSION TESTS PASSED!")
