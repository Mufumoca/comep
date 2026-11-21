import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
from typing import List, Tuple

from comep.insights import InsightBank, parse_operations


def reference_functions():
    directory = Path("outputs/baseline-source-audit-20261103")
    path = directory/"expel_agent__expel.py"
    manifest = json.loads((directory/"source_manifest.json").read_text())
    expected = next(row["artifact_sha256"] for row in manifest if row["file"] == path.name)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    names = {"parse_rules", "retrieve_rule_index", "is_existing_rule", "update_rules"}
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == 4
    namespace = {"re": re, "List": List, "Tuple": Tuple}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def main():
    reference = reference_functions()
    generator = random.Random(1301)
    comparisons = 0
    for size in (0, 1, 4, 7):
        initial = [(f"Rule {index}.", 1+index % 3) for index in range(size)]
        for _ in range(100):
            lines = []
            for step in range(generator.randrange(1, 8)):
                kind = generator.choice(("ADD", "EDIT", "REMOVE", "AGREE"))
                number = generator.randrange(1, size+3)
                text = generator.choice([f"New {step}.", "Cut off", "ADD is banned."] + [text for text, _ in initial])
                lines.append(f"{kind} {number}: {text}")
            response = "\n".join(lines)
            parsed = parse_operations(response)
            assert parsed == reference["parse_rules"](response)
            expected = reference["update_rules"](deepcopy(initial), deepcopy(parsed), list_full=size >= 7)
            saved = deepcopy(initial)
            bank = InsightBank(initial, max_num_rules=2)
            bank.apply(response)
            assert bank.pairs() == expected and initial == saved
            comparisons += 1
    bank = InsightBank([("Keep.", 2), ("Other.", 2)])
    trace = bank.apply("REMOVE 99: Keep.\nAGREE 1: Keep.\nEDIT 2: Changed.\nADD: Fresh.\nADD: Fresh.")
    assert bank.pairs() == [("Changed.", 3), ("Keep.", 2), ("Fresh.", 2), ("Fresh.", 2)]
    assert not trace["rejected_operations"]
    bank = InsightBank([("Keep.", 2)])
    trace = bank.apply("EDIT 0: Changed.")
    assert trace["rejected_operations"][0]["reason"] == "invalid_edit_index" and bank.pairs() == [("Keep.", 2)]
    trace = bank.apply("EDIT 0: Prefix Keep.")
    assert trace["normalized_operations"] and bank.pairs() == [("Keep.", 3)]
    bank = InsightBank([("First.", 2), ("Second.", 2)], max_items=2)
    trace = bank.apply("EDIT 1: Revised.\nADD: New.\nADD: Extra.")
    assert [row["identifier"] for row in trace["capacity_removals"]] == [1, 2]
    assert bank.pairs() == [("Revised.", 3), ("Extra.", 2)]
    bank = InsightBank(max_tokens=3)
    trace = bank.apply("ADD: Far too many separate words here.")
    assert not bank.rules and len(trace["capacity_removals"]) == 1
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary)/"input.json"
        output = Path(temporary)/"output.json"
        source.write_text(json.dumps({"initial_rules": [["Keep.", 2]], "responses": ["AGREE: Keep."]}))
        subprocess.run([sys.executable, "scripts/replay_insights.py", "--input", str(source), "--output", str(output)], check=True)
        result = json.loads(output.read_text())
        assert result["llm_calls"] == 0 and result["final"][0]["importance"] == 3
    print(f"INSIGHT_WITNESS {comparisons} pinned-source differential cases; operation/capacity/deviation/CLI tests passed; zero API, not efficacy")


if __name__ == "__main__":
    main()
