from copy import deepcopy
import json
from pathlib import Path
import tempfile

from analyze_procedural_replication import analyze
from comep.factorial import CostGuardClient
from run_procedural_pair import prepare, run
from verify_factorial import MockClient, MockEnvironment


def main():
    old = json.loads(Path("outputs/procedural-pair-preflight-20261103/packet.json").read_text())
    assert prepare() == old
    packet = prepare("replication")
    assert len(packet["trials"]) == 48
    assert {row["task"]["variation"] for row in packet["trials"]} == {5, 6, 7, 8}
    environments = []

    def factory(trial, directory):
        environment = MockEnvironment(trial)
        environments.append(environment)
        return environment

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "complete"
        run(packet, CostGuardClient(MockClient(), 1.2), factory, root)
        result = analyze(root)
        assert result["complete"] and result["followup_justified"]
        original = json.loads((root / "summary.json").read_text())
        for condition in ("all_zero", "merged_equal", "singleton_equal", "reverse_equal"):
            summary = deepcopy(original)
            for row in summary["rows"]:
                cell = row["cell"]
                row["episode"]["reward"] = int(cell == "AB")
                if condition == "all_zero":
                    row["episode"]["reward"] = 0
                elif condition == "merged_equal" and cell == "merged":
                    row["episode"]["reward"] = 1
                elif condition == "singleton_equal" and cell == "A":
                    row["episode"]["reward"] = 1
                elif condition == "reverse_equal" and cell == "BA":
                    row["episode"]["reward"] = 1
            (root / "summary.json").write_text(json.dumps(summary))
            result = analyze(root)
            if condition == "reverse_equal":
                assert not result["gates"]["order_signal"]
            else:
                assert not result["followup_justified"]
        summary = deepcopy(original)
        summary["rows"].append(summary["rows"][0])
        (root / "summary.json").write_text(json.dumps(summary))
        try:
            analyze(root)
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate accepted")
        partial = Path(temporary) / "partial"
        run(packet, CostGuardClient(MockClient(limit=3), 1.2), factory, partial)
        assert not analyze(partial)["complete"]
    assert all(environment.closed for environment in environments)
    print("PROCEDURAL_REPLICATION_WITNESS 48 mock cells, frozen pilot, negative gates, duplicate/partial rejection and closures; no paid API")


if __name__ == "__main__":
    main()
