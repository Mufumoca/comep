import json
from pathlib import Path

import numpy as np


def render_report(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    if summary["kind"] != "synthetic_experiment":
        raise ValueError("this report expects a synthetic experiment directory")
    manifest = json.loads((directory / "manifest.json").read_text())
    figure_dir = directory / "figures"
    figure_dir.mkdir(exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for method in manifest["methods"]:
        curves = []
        pruning = []
        for seed in manifest["seeds"]:
            records = [json.loads(line) for line in (directory / f"{method}-seed-{seed}.jsonl").read_text().splitlines()]
            rewards = np.array([record["reward"] for record in records])
            curves.append(np.cumsum(rewards) / np.arange(1, len(records) + 1))
            pruning.append([np.nan if record["pruned_good_fraction"] is None else record["pruned_good_fraction"] for record in records])
        axes[0].plot(np.arange(1, len(curves[0]) + 1), np.mean(curves, axis=0), label=method)
        axes[1].plot(np.arange(1, len(curves[0]) + 1), np.mean(pruning, axis=0), label=method)
    axes[0].set(xlabel="Episode", ylabel="Cumulative success rate", ylim=(0, 1))
    axes[1].set(xlabel="Episode", ylabel="Pruned positive-main-effect fraction", ylim=(0, 1))
    axes[0].legend(fontsize=7)
    figure.suptitle("Synthetic diagnostic only — mean across seeds")
    figure.tight_layout()
    figure.savefig(figure_dir / "learning-and-pruning.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 5))
    for aggregate in summary["aggregate"]:
        axis.errorbar(aggregate["mean_retrieval_tokens"], aggregate["success_rate_mean"], yerr=aggregate["success_rate_seed_se"] or 0, fmt="o", label=aggregate["method"])
    axis.legend(fontsize=8)
    axis.set(xlabel="Mean retrieved lexical-estimate tokens (not total API tokens)", ylabel="Success rate (error bars: seed standard error)", title="Synthetic token / success trade-off")
    figure.tight_layout()
    figure.savefig(figure_dir / "token-tradeoff.png", dpi=160)
    plt.close(figure)
    if "comep" in manifest["methods"]:
        seed = manifest["seeds"][0]
        state = json.loads((directory / f"comep-seed-{seed}-state.json").read_text())
        identifiers = [memory["identifier"] for memory in state["memories"]]
        if identifiers:
            positions = {identifier: index for index, identifier in enumerate(identifiers)}
            matrix = np.full((len(identifiers), len(identifiers)), np.nan)
            for pair in state["pairs"]:
                first, second = (positions[identifier] for identifier in pair["identifiers"])
                matrix[first, second] = matrix[second, first] = pair["mean"]
            figure, axis = plt.subplots(figsize=(8, 7))
            extent = max(0.01, max((abs(pair["mean"]) for pair in state["pairs"]), default=0))
            image = axis.imshow(np.ma.masked_invalid(matrix), cmap="RdBu", vmin=-extent, vmax=extent)
            axis.set_xticks(range(len(identifiers)), identifiers, rotation=90, fontsize=7)
            axis.set_yticks(range(len(identifiers)), identifiers, fontsize=7)
            axis.set_title(f"Estimated interactions, synthetic seed {seed}\nBlank = unobserved; not causal evidence")
            figure.colorbar(image, ax=axis)
            figure.tight_layout()
            figure.savefig(figure_dir / "interaction-heatmap.png", dpi=160)
            plt.close(figure)
    lines = ["# Synthetic experiment report", "", "This report is not evidence of benchmark performance or causal conflict detection.", "", "| Method | Success mean | Seed SE | Mean retrieved tokens |", "|---|---:|---:|---:|"]
    for row in summary["aggregate"]:
        uncertainty = "n/a" if row["success_rate_seed_se"] is None else f"{row['success_rate_seed_se']:.4f}"
        lines.append(f"| {row['method']} | {row['success_rate_mean']:.4f} | {uncertainty} | {row['mean_retrieval_tokens']:.2f} |")
    lines.extend(["", "Token counts are lexical estimates of selected rule bodies, not actual prompt or total API usage.", "", "![Learning and pruning](figures/learning-and-pruning.png)", "", "![Token trade-off](figures/token-tradeoff.png)"])
    if (figure_dir / "interaction-heatmap.png").exists():
        lines.extend(["", "![Estimated interactions](figures/interaction-heatmap.png)"])
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory / "REPORT.md"
