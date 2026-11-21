import argparse
from dataclasses import asdict, replace
import getpass
import json
import os
from pathlib import Path
import sys

import numpy as np

from comep.client import APIConfig, ChatClient
from comep.core import Config
from comep.experiments import SCENARIOS, new_directory, run_synthetic, write_json


SEEDS = [101, 202, 303]
METHODS = ["no-memory", "full-memory", "individual", "mmr", "additive", "proposal", "comep"]


def local_campaign(output):
    output = new_directory(output)
    config = Config(retrieval_tokens=60, bank_tokens=180, max_items=4, max_bank_items=12, exploration=0.15, sampling_scale=0.3, pair_lookahead=True)
    for scenario in SCENARIOS:
        run_synthetic(output / scenario, METHODS, SEEDS, episodes=300, scenario=scenario, config=config, size=16, poison_fraction=0.2, arrival_every=20)
        print("SCENARIO_DONE", scenario, flush=True)
    ablations = {"no-conflict": {"hard_conflicts": False}, "no-lookahead": {"pair_lookahead": False}, "no-mmr": {"redundancy": 0.0}, "no-exploration": {"exploration": 0.0}}
    for name, overrides in ablations.items():
        run_synthetic(output / name, ["comep"], SEEDS, episodes=300, scenario="mixed", config=replace(config, **overrides), size=16, poison_fraction=0.2, arrival_every=20)
        print("ABLATION_DONE", name, flush=True)


def real_campaign(output, tasks_path):
    from comep.environments import ScienceWorldAdapter
    from comep.core import MemoryManager
    from comep.runner import AgentConfig, run_agent
    from scienceworld_setup import java_environment

    java_environment()
    output = new_directory(output)
    if not os.environ.get("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("DeepSeek credential (hidden, process-only): ")
    methods = ["no-memory", "individual", "additive", "comep"]
    config = Config(retrieval_tokens=120, bank_tokens=400, max_items=4, max_bank_items=8)
    agent = AgentConfig(episodes=6, max_steps=20, history_turns=6, distill=True, seed=11, admissible_limit=0, action_instructions="ScienceWorld command examples (substitute observed object names): look around; inventory; look at OBJECT; open DOOR; go to ROOM; pick up OBJECT; move OBJECT to CONTAINER; focus on OBJECT; use INSTRUMENT on OBJECT; connect TERMINAL to TERMINAL; activate DEVICE; wait. Follow the task's requested focus and ordering. Do not use task solutions or hidden state.")
    rules = json.loads(Path("configs/scienceworld-initial-rules.json").read_text())
    tasks = json.loads(tasks_path.read_text())
    if len(tasks) != 6:
        raise ValueError("frozen real pilot requires exactly six tasks")
    write_json(output / "manifest.json", {"kind": "real_scienceworld_dev_pilot", "methods": methods, "tasks": tasks, "agent": asdict(agent), "manager": asdict(config), "initial_rules": rules, "max_calls": 504, "max_total_tokens": 500000, "protocol_amendment": "refine-logs/PROTOCOL_AMENDMENT_20261103_001000.md", "official_split_verification_required": True, "api_model": "deepseek-flash", "api_thinking": "disabled", "seed_reproducibility": "memory RNG only; provider alias is not bitwise deterministic"})
    used_calls = 0
    used_tokens = 0
    results = []
    try:
        for method in methods:
            if used_calls >= 504 or used_tokens >= 500000:
                break
            directory = new_directory(output / method)
            environment = ScienceWorldAdapter(tasks_path, max_steps=20)
            try:
                environment.verify_splits()
            except Exception:
                environment.close()
                raise
            manager = MemoryManager(method, config, seed=11)
            for rule in rules:
                manager.add(rule["id"], rule["text"], prune=False)
            manager.prune()
            api = APIConfig("https://api.deepseek.com", "deepseek-flash", key_env="DEEPSEEK_API_KEY", max_calls=504-used_calls, max_total_tokens=500000-used_tokens, max_output_tokens=256, timeout_seconds=90, thinking="disabled")
            client = ChatClient(api, directory / "api.jsonl")
            write_json(directory / "manifest.json", {"method": method, "agent": asdict(agent), "api": asdict(api), "split_verified": environment.split_verified, "tasks": tasks, "manager": asdict(config)})
            print("REAL_START", method, flush=True)
            try:
                result = run_agent(environment, manager, client, agent, directory)
            finally:
                usage = client.summary()
                used_calls += usage["attempted_calls"]
                used_tokens += usage["prompt_tokens"] + usage["completion_tokens"]
            results.append({"method": method, **result})
            write_json(output / "summary.json", {"runs": results, "attempted_calls": used_calls, "reported_tokens": used_tokens, "status": "running"})
            print("REAL_DONE", method, result["status"], result["success_rate"], used_calls, used_tokens, flush=True)
            if result["status"] != "completed":
                break
    finally:
        write_json(output / "summary.json", {"runs": results, "attempted_calls": used_calls, "reported_tokens": used_tokens, "status": "completed" if len(results) == len(methods) and all(row["status"] == "completed" for row in results) else "incomplete"})
        os.environ.pop("DEEPSEEK_API_KEY", None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["local", "real"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, default=Path("outputs/scienceworld-setup/tasks.json"))
    args = parser.parse_args()
    if args.mode == "local":
        local_campaign(args.output)
    else:
        real_campaign(args.output, args.tasks)


if __name__ == "__main__":
    main()
