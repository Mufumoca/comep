import argparse
import json
from pathlib import Path
import tomllib

from .core import Config, METHODS
from .experiments import SCENARIOS, run_synthetic


def main():
    parser = argparse.ArgumentParser(prog="comep")
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="Check entry point; not an experiment")
    smoke.add_argument("--output", type=Path, required=True)
    synthetic = commands.add_parser("synthetic", help="Run explicitly labeled local simulations")
    synthetic.add_argument("--output", type=Path, required=True)
    synthetic.add_argument("--config", type=Path)
    synthetic.add_argument("--episodes", type=int)
    synthetic.add_argument("--seeds", type=int, nargs="+")
    synthetic.add_argument("--methods", choices=METHODS, nargs="+")
    synthetic.add_argument("--scenario", choices=SCENARIOS)
    run = commands.add_parser("run", help="Opt-in real environment/API rollout")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    report = commands.add_parser("report", help="Plot synthetic diagnostics; requires plots extra")
    report.add_argument("directory", type=Path)
    api_check = commands.add_parser("api-check", help="One paid action-format connectivity request; not an experiment")
    api_check.add_argument("--config", type=Path, required=True)
    api_check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "smoke":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        from .core import MemoryManager

        manager = MemoryManager(seed=0)
        manager.add("example", "Check the room before navigating.")
        selected = manager.choose("navigate the room")
        manager.observe(selected, 1)
        args.output.write_text(json.dumps({"kind": "spine_check", "stub": False, "selected": selected.identifiers, "state": manager.snapshot()}) + "\n")
        print(args.output)
    elif args.command == "synthetic":
        settings = tomllib.loads(args.config.read_text()) if args.config else {}
        experiment = settings.get("experiment", {}).copy()
        memory_cfg = settings.get("memory", {})
        if "embedding_model" in memory_cfg and "embedding_model" not in experiment:
            experiment["embedding_model"] = memory_cfg["embedding_model"]
        if "token_encoding" in memory_cfg and "token_encoding" not in experiment:
            experiment["token_encoding"] = memory_cfg["token_encoding"]
        experiment.setdefault("methods", list(METHODS))
        experiment.setdefault("seeds", [0])
        for name in ("episodes", "seeds", "methods", "scenario"):
            if getattr(args, name) is not None:
                experiment[name] = getattr(args, name)
        result = run_synthetic(args.output, config=Config(**settings.get("manager", {})), **experiment)
        print(json.dumps(result, indent=2))
    elif args.command == "run":
        from .runner import run_from_config

        print(json.dumps(run_from_config(args.config, args.output), indent=2))
    elif args.command == "report":
        from .report import render_report

        print(render_report(args.directory))
    elif args.command == "api-check":
        from dataclasses import asdict
        from .client import APIConfig, ChatClient
        from .experiments import new_directory, write_json
        from .runner import parse_action

        settings = tomllib.loads(args.config.read_text())
        config = APIConfig(**settings["api"])
        directory = new_directory(args.output)
        client = ChatClient(config, directory / "api.jsonl")
        result = {"kind": "api_connectivity_check", "benchmark": False, "api_config": asdict(config), "status": "started"}
        try:
            reply = client.complete([{"role": "system", "content": "Follow the requested output format exactly."}, {"role": "user", "content": "Connectivity check only, not a real environment. The only admissible action is look. Reply with exactly: Action: look"}], "connectivity_check")
            result["reply"] = reply
            result["action"] = parse_action(reply)
            if result["action"] != "look":
                raise ValueError("model did not follow requested action format")
            result["status"] = "ok"
        except Exception as error:
            result["status"] = "error"
            result["error_type"] = type(error).__name__
            raise
        finally:
            result["usage"] = client.summary()
            write_json(directory / "summary.json", result)
        print(json.dumps(result, indent=2))
