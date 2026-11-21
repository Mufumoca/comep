import argparse
import json
from pathlib import Path

import numpy as np

from comep.joint import probabilities
from comep.policy import FixedBankPolicy


def run(output):
    policy = FixedBankPolicy("individual", [8]*8)
    rows = []
    for seed in (1001, 1002, 1003):
        main = np.load(output/f"world-{seed}.npz")["main"]
        for context in (0, 1):
            selected = policy.membership*np.where(np.arange(8)%2 == context, 1., .25)
            additive = -1+.4*(2*context-1)+selected@main
            additive_choice = int(np.argmax(additive))
            for effect in (-3, 3):
                truth = probabilities(additive+effect*selected[:, 0]*selected[:, 1])
                oracle_choice = int(np.argmax(truth))
                rows.append({"seed": seed, "context": context, "effect": effect,
                             "true_main_only_choice": list(policy.subsets[additive_choice]),
                             "full_oracle_choice": list(policy.subsets[oracle_choice]),
                             "true_main_only_probability": float(truth[additive_choice]),
                             "full_oracle_probability": float(truth[oracle_choice]),
                             "interaction_aware_headroom": float(truth[oracle_choice]-truth[additive_choice])})
    result = {"scope": "Posthoc generator decision-headroom audit. Uses true parameters only in evaluator. Not an achieved learned-policy performance, not bound against fitted misspecified additive.", "rows": rows}
    (output/"posthoc-headroom.json").write_text(json.dumps(result, indent=2)+"\n")
    for effect in (-3, 3):
        group = [row for row in rows if row["effect"] == effect]
        print(effect, "zero_headroom", sum(row["interaction_aware_headroom"] == 0 for row in group),
              "of", len(group), "mean", np.mean([row["interaction_aware_headroom"] for row in group]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
