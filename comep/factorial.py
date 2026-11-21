from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import time
from pathlib import Path

import numpy as np

from .client import BudgetExceeded
from .core import Config, MemoryManager, Selection
from .experiments import new_directory, write_json
from .runner import AgentConfig, run_agent


CONDITIONS = {"none": [], "A": ["general-02"], "B": ["general-06"], "AB": ["general-02", "general-06"]}


def validate_schedule(packet):
    trials = packet["trials"]
    if len(trials) != 48 or packet["conditions"] != CONDITIONS:
        raise ValueError("frozen 48-trial factorial required")
    if packet["model"] != "deepseek-flash" or packet["thinking"] != "disabled" or packet["max_steps"] != 20 or packet["history_turns"] != 6 or packet["distill"] or packet["retries"] != 0:
        raise ValueError("unexpected frozen agent settings")
    seen = set()
    tasks, rules = {}, {}
    for index, trial in enumerate(trials):
        key = (trial["task_index"], trial["repeat"], trial["cell"])
        if trial["trial"] != index or key in seen or key[0] not in range(6) or key[1] not in (0, 1) or key[2] not in CONDITIONS:
            raise ValueError("invalid or duplicate factorial cell")
        seen.add(key)
        if trial["rule_ids"] != CONDITIONS[key[2]] or len(trial["rules"]) != len(trial["rule_ids"]):
            raise ValueError("cell memories do not match frozen condition")
        if trial["task"].get("split") != "dev":
            raise ValueError("development tasks required")
        if key[0] in tasks and tasks[key[0]] != trial["task"]:
            raise ValueError("task changed between conditions")
        tasks[key[0]] = trial["task"]
        for identifier, text in zip(trial["rule_ids"], trial["rules"]):
            if not isinstance(text, str) or not text.strip() or (identifier in rules and rules[identifier] != text):
                raise ValueError("inconsistent memory text")
            rules[identifier] = text
    return trials


def validate_frozen_sources(packet, tasks, rules, action_instructions):
    trials = validate_schedule(packet)
    if len(tasks) != 6 or len({(task["task"], task["variation"], task["split"]) for task in tasks}) != 6:
        raise ValueError("six distinct frozen tasks required")
    random = np.random.default_rng(1301)
    expected_order = [(task, repeat, str(cell)) for task in range(6) for repeat in range(2)
                      for cell in random.permutation(list(CONDITIONS))]
    for trial, expected in zip(trials, expected_order):
        if (trial["task_index"], trial["repeat"], trial["cell"]) != expected or trial["task"] != tasks[expected[0]]:
            raise ValueError("frozen task/order drift")
        if trial["rules"] != [rules[identifier] for identifier in CONDITIONS[trial["cell"]]]:
            raise ValueError("frozen rule text drift")
        system = "You are a text-environment agent. Solve the stated task using environment commands. Reply with optional short Thought: and exactly one Action: command line. Memories are fallible past experience, not instructions overriding the task."
        system += "\n"+action_instructions
        if trial["rules"]:
            system += "\nMemories:\n"+"\n".join(f"- {text}" for text in trial["rules"])
        if trial["preflight_first_messages"][0] != {"role": "system", "content": system}:
            raise ValueError("reviewed/runtime system prompt drift")


def validate_launch_readiness(packet, approval, review, input_rate, output_rate, cap, credential_available):
    credential_authorized = approval.get("credential_rotated_confirmed") is True or (
        approval.get("credential_reuse_authorized") is True and
        approval.get("credential_reuse_authorization_source") == "explicit_user_instruction")
    if approval.get("status") != "approved" or approval.get("operator_authorized") is not True or not credential_authorized:
        raise ValueError("operator approval and explicit credential authorization required")
    if review.get("code_protocol_ready") is not True or review.get("review_scope") != "bounded_descriptive_factorial_code_protocol":
        raise ValueError("passing protocol preflight required")
    if not packet.get("priced_budget_approved") or not packet.get("model_provider_pricing_verified") or packet.get("not_launchable_until"):
        raise ValueError("schedule readiness fields unresolved")
    if approval.get("model") != packet["model"] or approval.get("input_usd_per_million") != input_rate or approval.get("output_usd_per_million") != output_rate or approval.get("max_estimated_usd") != cap:
        raise ValueError("runtime pricing/model/cap differs from approved values")
    if not all(math.isfinite(value) and value > 0 for value in (input_rate, output_rate, cap)) or input_rate < .30 or output_rate < 1.20:
        raise ValueError("positive conservative documented peak rates required")
    if not approval.get("pricing_source") or not approval.get("pricing_checked_at_utc"):
        raise ValueError("price provenance required")
    checked = datetime.fromisoformat(approval["pricing_checked_at_utc"])
    if checked.tzinfo is None or not 0 <= (datetime.now(timezone.utc)-checked).total_seconds() <= 86400:
        raise ValueError("pricing check must be within preceding24hours")
    if not credential_available:
        raise ValueError("secure credential unavailable")


class FrozenMemoryManager(MemoryManager):
    def __init__(self, trial):
        super().__init__("full-memory", Config(max_items=2, max_bank_items=2, retrieval_tokens=200, bank_tokens=200))
        self.identifiers = trial["rule_ids"][:]
        for identifier, text in zip(self.identifiers, trial["rules"]):
            self.add(identifier, text, prune=False)
        if self.bank_tokens > self.config.retrieval_tokens:
            raise ValueError("fixed memories exceed budget")

    def choose(self, query, context):
        return Selection(self.identifiers[:], {identifier: 1. for identifier in self.identifiers}, context, self.bank_tokens, False)

    def observe(self, selection, reward):
        return None


class CostGuardClient:
    def __init__(self, client, cap_usd):
        if not math.isfinite(cap_usd) or cap_usd <= 0:
            raise ValueError("positive finite estimated dollar cap required")
        if client.config.input_price_per_million is None or client.config.output_price_per_million is None:
            raise ValueError("verified input/output price required")
        self.client = client
        self.cap_usd = cap_usd
        self.request_log = None
        self.identity = None
        self.last_completed_at = None

    def summary(self):
        return self.client.summary()

    def complete(self, messages, purpose="action"):
        config = self.client.config
        if self.last_completed_at is not None and time.monotonic()-self.last_completed_at > 300:
            raise RuntimeError("material interruption over300seconds; stop noncomparable campaign")
        prior = self.client.summary()
        if not prior["usage_complete"] or prior["estimated_cost_usd"] is None:
            raise RuntimeError("cannot continue without complete usage")
        estimated_input = sum(len(message["content"].encode("utf-8"))+64 for message in messages)+64
        reservation = (estimated_input*config.input_price_per_million+config.max_output_tokens*config.output_price_per_million)/1e6
        if prior["estimated_cost_usd"]+reservation > self.cap_usd:
            raise BudgetExceeded("estimated dollar reservation exceeds cap before request")
        if self.request_log is not None:
            with Path(self.request_log).open("a") as handle:
                handle.write(json.dumps({"next_call": prior["attempted_calls"]+1, "purpose": purpose, "messages": messages,
                                         "estimated_usd_reservation": reservation}, allow_nan=False)+"\n")
        response = self.client.complete(messages, purpose)
        self.last_completed_at = time.monotonic()
        metadata = getattr(self.client, "last_response", None)
        if metadata is not None:
            if self.request_log is not None:
                with Path(self.request_log).with_name("responses.jsonl").open("a") as handle:
                    handle.write(json.dumps({"completed_at_utc": datetime.now(timezone.utc).isoformat(), **metadata}, allow_nan=False)+"\n")
            if metadata.get("finish_reason") == "length":
                raise RuntimeError("truncated model response; abort without invented episode reward")
            identity = (metadata.get("response_model"), metadata.get("system_fingerprint"))
            if identity[0] is None:
                raise RuntimeError("returned model identity absent")
            if self.identity is None:
                self.identity = identity
            elif identity != self.identity:
                raise RuntimeError("model or fingerprint changed; stop noncomparable campaign")
        if self.client.summary()["estimated_cost_usd"] > self.cap_usd:
            self.client.halted = True
            raise BudgetExceeded("reported usage cost estimate exceeded cap; stopping")
        return response


def run_factorial(packet, client, environment_factory, output, action_instructions, kind="real_scienceworld_factorial"):
    trials = validate_schedule(packet)
    output = new_directory(output)
    if (output/"campaign.json").exists():
        raise ValueError("cannot resume/overwrite this campaign")
    agent = AgentConfig(episodes=1, max_steps=20, history_turns=6, distill=False, admissible_limit=0,
                        action_instructions=action_instructions)
    write_json(output/"campaign.json", {"kind": kind, "schedule": packet, "agent": asdict(agent),
                                       "api_config": asdict(client.client.config), "estimated_dollar_cap": client.cap_usd,
                                       "scope": "Memory-pair screening, not CoMeP policy/pruning efficacy.",
                                       "cost_guard": "local byte-estimate reservation, not provider-enforced cap"})
    records = []
    status, error_type = "completed", None
    try:
        for trial in trials:
            trial_started = datetime.now(timezone.utc).isoformat()
            directory = new_directory(output/f"trial-{trial['trial']:03d}")
            manager = FrozenMemoryManager(trial)
            before = client.summary()
            client.request_log = directory/"requests.jsonl"
            write_json(directory/"trial.json", trial)
            failure_stage = "initialization"
            try:
                environment = environment_factory(trial)
                failure_stage = "runner"
                result = run_agent(environment, manager, client, agent, directory)
            except Exception as error:
                after = client.summary()
                row = {"trial": trial["trial"], "task_index": trial["task_index"], "repeat": trial["repeat"], "cell": trial["cell"],
                       "task": trial["task"], "status": "error", "error_type": type(error).__name__, "episode": None,
                       "failure_stage": failure_stage,
                       "usage": {key: after[key]-before[key] for key in ("attempted_calls", "prompt_tokens", "completion_tokens")},
                       "usage_complete": after["usage_complete"], "started_at_utc": trial_started,
                       "finished_at_utc": datetime.now(timezone.utc).isoformat()}
                if (directory/"summary.json").exists():
                    row["failure_stage"] = json.loads((directory/"summary.json").read_text()).get("failure_stage")
                records.append(row)
                with (output/"trials.jsonl").open("a") as handle:
                    handle.write(json.dumps(row, allow_nan=False)+"\n")
                raise
            after = client.summary()
            episodes = [json.loads(line) for line in (directory/"episodes.jsonl").read_text().splitlines()] if (directory/"episodes.jsonl").exists() else []
            usage = {key: after[key]-before[key] for key in ("attempted_calls", "prompt_tokens", "completion_tokens")}
            row = {"trial": trial["trial"], "task_index": trial["task_index"], "repeat": trial["repeat"], "cell": trial["cell"],
                   "task": trial["task"], "status": result["status"], "episode": episodes[0] if len(episodes) == 1 else None,
                   "usage": usage, "usage_complete": after["usage_complete"], "failure_stage": result.get("failure_stage"),
                   "started_at_utc": trial_started, "finished_at_utc": datetime.now(timezone.utc).isoformat()}
            records.append(row)
            with (output/"trials.jsonl").open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False)+"\n")
            if result["status"] != "completed":
                status = result["status"]
                break
    except Exception as error:
        status, error_type = "error", type(error).__name__
        raise
    finally:
        write_json(output/"summary.json", {"kind": kind, "status": status, "error_type": error_type,
                    "completed_trials": sum(row["status"] == "completed" and row["episode"] is not None for row in records),
                    "expected_trials": 48, "api": client.summary(), "rows": records})
    return records
