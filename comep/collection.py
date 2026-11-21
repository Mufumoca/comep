from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from .core import MemoryManager
from .experiments import new_directory, write_json
from .grounding import evidence_prompt, validate_evidence
from .runner import AgentConfig, run_agent


def validate_collection_launch(packet, approval, review, credential_available, scope="six_train_experience_collection_v2"):
    budget = packet["budget_proposal"]
    if approval.get("status") != "approved" or approval.get("operator_authorized") is not True or approval.get("scope") != scope:
        raise ValueError("explicit collection-specific operator approval required")
    reuse = approval.get("credential_reuse_authorized") is True and approval.get("credential_reuse_authorization_source") == "explicit_user_instruction"
    if approval.get("credential_rotated_confirmed") is not True and not reuse:
        raise ValueError("credential authorization required")
    if review.get("code_protocol_ready") is not True or review.get("review_scope") != scope:
        raise ValueError("collection executable preflight required; planning review is insufficient")
    if approval.get("model") != packet["model"]:
        raise ValueError("model differs from approval")
    for key in ("max_calls", "max_total_tokens", "max_output_tokens", "input_usd_per_million", "output_usd_per_million", "local_estimate_allowance_usd"):
        if approval.get(key) != budget[key]:
            raise ValueError(f"approved budget mismatch: {key}")
        if isinstance(budget[key], bool) or not math.isfinite(budget[key]) or budget[key] <= 0:
            raise ValueError("positive finite budget required")
    if not approval.get("pricing_source"):
        raise ValueError("pricing source required")
    checked = datetime.fromisoformat(approval.get("pricing_checked_at_utc", ""))
    if checked.tzinfo is None or not 0 <= (datetime.now(timezone.utc)-checked).total_seconds() <= 86400:
        raise ValueError("pricing check must be within24hours")
    if not credential_available:
        raise ValueError("credential unavailable")


class CollectionManager(MemoryManager):
    def __init__(self):
        super().__init__("no-memory")
        self.manager_llm_calls = 0
        self.candidates = []

    def after_episode(self, record, trajectory, client, output):
        visible = [{key: row[key] for key in ("episode", "step", "task_id", "observation", "reply", "action", "next_observation", "done", "success", "score", "previous_score")} for row in trajectory]
        event = {"episode": record["episode"], "task_id": record["task_id"], "status": "prepared",
                 "source_steps": len(visible), "semantic_verified": False}
        messages = [{"role": "system", "content": "Extract a concise grounded memory rule; return JSON only."},
                    {"role": "user", "content": evidence_prompt(visible, record["reward"], record["termination"], 60)+"\n"+json.dumps(visible, ensure_ascii=False)}]
        path = Path(output)/"candidates.jsonl"
        with path.open("a") as handle:
            handle.write(json.dumps({**event, "messages": messages}, allow_nan=False)+"\n")
        before = client.summary()
        try:
            response = client.complete(messages, "experience_extraction")
            event["raw_response"] = response
            try:
                candidate = json.loads(response)
            except json.JSONDecodeError:
                event["status"] = "rejected"
                event["validation"] = {"accepted": False, "reason": "invalid_json", "semantic_verified": False}
            else:
                event["candidate"] = candidate
                event["validation"] = validate_evidence(candidate, visible, record["reward"], 60, self.counter)
                event["status"] = "provenance_accepted" if event["validation"]["accepted"] else "rejected"
        except Exception as error:
            event["status"] = "error"
            event["error_type"] = type(error).__name__
            metadata = getattr(getattr(client, "client", client), "last_response", None)
            if metadata is not None:
                event["response_metadata"] = metadata
            raise
        finally:
            after = client.summary()
            event["usage"] = {key: after[key]-before[key] for key in ("attempted_calls", "prompt_tokens", "completion_tokens")}
            event["usage_complete"] = after["usage_complete"]
            self.manager_llm_calls += event["usage"]["attempted_calls"]
            self.candidates.append(event)
            with path.open("a") as handle:
                handle.write(json.dumps(event, allow_nan=False)+"\n")

    def snapshot(self):
        return {**super().snapshot(), "collection_candidates": self.candidates,
                "manager_llm_calls": self.manager_llm_calls,
                "candidate_policy": "Frozen artifacts only; never admitted to action memory; semantics unverified"}


def run_collection(packet, client, environment_factory, output, kind="real_training_experience_collection"):
    output = new_directory(output)
    manager = CollectionManager()
    agent = AgentConfig(episodes=6, max_steps=40, history_turns=6, admissible_limit=0,
                        distill=False, action_instructions=packet["action_instructions"])
    write_json(output/"campaign.json", {"kind": kind, "packet": packet, "agent": asdict(agent),
        "api_config": asdict(client.client.config), "estimated_dollar_cap": client.cap_usd,
        "scope": "Candidate collection only; no dev rollout, learned policy comparison or semantic acceptance"})
    client.request_log = output/"requests.jsonl"
    stage, status, error_type = "initialization", "started", None
    try:
        environment = environment_factory(packet["stages"]["experience_train"])
        stage = "runner"
        result = run_agent(environment, manager, client, agent, output)
        status = result["status"]
        return result
    except Exception as error:
        status, error_type = "error", type(error).__name__
        raise
    finally:
        result = json.loads((output/"summary.json").read_text()) if (output/"summary.json").exists() else None
        write_json(output/"collection-summary.json", {"kind": kind, "status": status, "error_type": error_type,
            "failure_stage": result.get("failure_stage") if result is not None else stage,
            "completed_episodes": result["completed_episodes"] if result is not None else 0,
            "extraction_records": len(manager.candidates), "provenance_accepted": sum(row["status"] == "provenance_accepted" for row in manager.candidates),
            "semantic_review_complete": False, "new_dev_episodes": 0, "api": client.summary()})
