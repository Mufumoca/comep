import argparse
import hashlib
import json
import os
from pathlib import Path

import jdk4py
import numpy as np

from comep.experiments import new_directory, write_json


def java_environment():
    os.environ["JAVA_HOME"] = str(jdk4py.JAVA_HOME)
    java_bin = str(jdk4py.JAVA_HOME / "bin")
    if java_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = java_bin + os.pathsep + os.environ.get("PATH", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    java_environment()
    from scienceworld import ScienceWorldEnv

    output = new_directory(args.output)
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(8, 8))
    witness = matrix @ matrix.T
    assert witness.shape == (8, 8) and np.isfinite(witness).all()
    environment = ScienceWorldEnv(envStepLimit=20)
    rows = []
    tasks = []
    partitions = {}
    names = ["find-living-thing", "use-thermometer", "power-component"]
    try:
        available = environment.get_task_names()
        for name in names:
            if name not in available:
                raise ValueError(f"predeclared task not available: {name}")
            environment.load(name, 0)
            partitions[name] = {"train": environment.get_variations_train(), "dev": environment.get_variations_dev(), "test": environment.get_variations_test()}
            if len(partitions[name]["dev"]) < 2:
                raise ValueError(f"insufficient official dev variations for {name}")
        for index in range(2):
            for name in names:
                variation = partitions[name]["dev"][index]
                task = {"task": name, "variation": variation, "split": "dev", "simplifications": ""}
                tasks.append(task)
                environment.load(name, variation, simplificationStr="", generateGoldPath=False)
                observation, info = environment.reset()
                after, reward, done, next_info = environment.step("look around")
                assert info["taskDesc"].strip() and isinstance(info["score"], (int, float))
                assert isinstance(after, str) and isinstance(next_info["valid"], list)
                rows.append({"task": task, "objective": info["taskDesc"], "initial_score": info["score"], "look_score": next_info["score"], "observation_chars": len(observation), "admissible_count": len(info["valid"]), "admissible_chars": sum(map(len, info["valid"])), "split_verified": variation in partitions[name]["dev"]})
    finally:
        environment.close()
    spec = json.loads(Path("configs/compute-scienceworld.json").read_text())
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:8]
    write_json(output / "tasks.json", tasks)
    write_json(output / "partitions.json", partitions)
    write_json(output / "witness.json", {"kind": "real_environment_witness", "api_calls": 0, "cpu_checksum": float(witness.sum()), "spec_hash": digest, "tasks": rows})
    print("ENV_WITNESS", json.dumps({"tasks": len(tasks), "shape": list(witness.shape), "spec_hash": digest, "all_splits_verified": all(row["split_verified"] for row in rows)}))


if __name__ == "__main__":
    main()
