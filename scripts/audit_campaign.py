import argparse
import json
from pathlib import Path

from comep.experiments import write_json


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def audit_real(directory):
    campaign = json.loads((directory / "manifest.json").read_text())
    expected = [f"{task['task']}:{task['variation']}:{task['split']}" for task in campaign["tasks"]]
    total_calls = 0
    total_tokens = 0
    warnings = []
    initial = {}
    checked = []
    for method in campaign["methods"]:
        folder = directory / method
        if not (folder / "summary.json").exists():
            warnings.append(f"{method}: no completed method summary")
            continue
        manifest = json.loads((folder / "manifest.json").read_text())
        assert manifest["split_verified"] is True
        summary = json.loads((folder / "summary.json").read_text())
        episodes = records(folder / "episodes.jsonl")
        transitions = records(folder / "transitions.jsonl")
        calls = records(folder / "api.jsonl")
        state = json.loads((folder / "state.json").read_text())
        assert summary["completed_episodes"] == len(episodes)
        assert [row["task_id"] for row in episodes] == expected[:len(episodes)]
        computed_success = sum(row["reward"] for row in episodes) / len(episodes) if episodes else None
        assert summary["success_rate"] == computed_success
        for episode in episodes:
            trajectory = [row for row in transitions if row["episode"] == episode["episode"]]
            assert episode["steps"] == len(trajectory) <= 20
            assert episode["reward"] == int(trajectory[-1]["success"])
            assert episode["score"] == trajectory[-1]["score"]
            assert episode["reward"] == int(episode["score"] >= 100)
            if episode["termination"] == "step_limit":
                assert episode["steps"] == 20 and not trajectory[-1]["done"]
            key = episode["task_id"]
            if key in initial:
                if initial[key] != trajectory[0]["observation"]:
                    same_lines = sorted(initial[key].splitlines()) == sorted(trajectory[0]["observation"].splitlines())
                    warnings.append(f"{method}: {key} initial observation differs; same line multiset={same_lines}. Environment text order is not bitwise controlled.")
            else:
                initial[key] = trajectory[0]["observation"]
            assert episode["retrieval_tokens"] <= campaign["manager"]["retrieval_tokens"]
        assert state["bank_tokens"] <= campaign["manager"]["bank_tokens"]
        assert len(state["memories"]) <= campaign["manager"]["max_bank_items"]
        assert summary["api"]["attempted_calls"] == len(calls)
        assert len(calls) == len({row["call"] for row in calls})
        prompt = sum(row.get("usage", {}).get("prompt_tokens", 0) for row in calls)
        completion = sum(row.get("usage", {}).get("completion_tokens", 0) for row in calls)
        assert prompt == summary["api"]["prompt_tokens"]
        assert completion == summary["api"]["completion_tokens"]
        assert all(row["status"] == "ok" for row in calls)
        total_calls += len(calls)
        total_tokens += prompt + completion
        if summary["status"] != "completed" or len(episodes) != 6:
            warnings.append(f"{method}: incomplete coverage; no unqualified comparison")
        checked.append({"method": method, "episodes": len(episodes), "successes": sum(row["reward"] for row in episodes), "calls": len(calls), "tokens": prompt + completion, "status": summary["status"]})
    assert total_calls <= campaign["max_calls"]
    assert total_tokens <= campaign["max_total_tokens"]
    result = {"kind": "deterministic_log_consistency_audit", "status": "warn" if warnings else "pass", "checked": checked, "warnings": warnings, "total_calls": total_calls, "total_tokens": total_tokens, "checks": ["success recomputed from raw terminal scores", "same ordered task IDs", "initial observation differences explicitly reported", "usage sums match provider event logs", "bank and retrieval budgets", "horizon accounting", "official split verification recorded"], "not_verified": ["provider invoice", "scientific generalization", "causal attribution", "bitwise model and environment reproducibility"]}
    write_json(directory / "consistency-audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    audit_real(parser.parse_args().directory)
