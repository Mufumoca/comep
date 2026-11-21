"""Export the 4-panel Stage 3 backbone comparison figure (fig4_four_backbone_comparison.pdf/.png)."""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["mathtext.fontset"] = "stix"

REPO_ROOT = Path(__file__).resolve().parent.parent

METHODS = [
    ("no-memory", "No-\nMemory", "#95a5a6"),
    ("full-memory", "Full-Memory\n(Top-k)", "#7f8c8d"),
    ("individual", "Individual\nCredit", "#8c564b"),
    ("mmr", "MMR-style\nDiversity", "#e67e22"),
    ("additive", "Additive\nLaplace UCB", "#2980b9"),
    ("expel-derived-online", "ExpeL-\nderived\nOnline", "#8e44ad"),
    ("evomemguard", "EvoMem-\nGuard-\nstyle", "#27ae60"),
    ("comep", "CoMeP\n(Ours)", "#c0392b"),
]

BACKBONES = [
    (
        "outputs/stage3-main",
        "(a) DeepSeek-V4.1-Flash (495 eps/method)",
        90,
        75,
        96,
    ),
    (
        "outputs/stage3-qwen3-14b-v22",
        "(b) Qwen3-14B-AWQ (330 eps/method; 4-bit AWQ)",
        60,
        50,
        55,
    ),
    (
        "outputs/stage3-gemma4-12b",
        "(c) Gemma-4-12B-it (330 eps/method; W4A16 QAT + Proxy Bias)",
        60,
        50,
        38,
    ),
    (
        "outputs/stage3-qwen3-8b",
        "(d) Qwen3-8B (330 eps/method; BF16)",
        60,
        50,
        46,
    ),
]


def load_backbone_stats(rel_dir: str, n_alf: int, n_sci: int) -> dict[str, dict]:
    base = REPO_ROOT / rel_dir
    out: dict[str, dict] = {}
    n_tot_seed = n_alf + n_sci
    for m_key, _, _ in METHODS:
        seed_srs = []
        tok_alf = []
        tok_sci = []
        for s in (0, 1, 2):
            sa = json.loads((base / "alfworld" / f"{m_key}_seed{s}" / "summary.json").read_text(encoding="utf-8"))
            ss = json.loads((base / "scienceworld" / f"{m_key}_seed{s}" / "summary.json").read_text(encoding="utf-8"))
            wa = round(float(sa["success_rate"]) * n_alf)
            ws = round(float(ss["success_rate"]) * n_sci)
            seed_srs.append(((wa + ws) / n_tot_seed) * 100.0)
            tok_alf.append(float(sa["mean_retrieval_tokens"]))
            tok_sci.append(float(ss["mean_retrieval_tokens"]))
        out[m_key] = {
            "mean": float(np.mean(seed_srs)),
            "sd": float(np.std(seed_srs, ddof=1)),
            "seeds": seed_srs,
            "macro_tok": 0.5 * (float(np.mean(tok_alf)) + float(np.mean(tok_sci))),
        }
    return out


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 8.0))
    axes_flat = axes.flatten()
    for ax, (rel_dir, title, n_alf, n_sci, ymax) in zip(axes_flat, BACKBONES):
        stats = load_backbone_stats(rel_dir, n_alf, n_sci)
        xs = np.arange(len(METHODS))
        means = [stats[m]["mean"] for m, _, _ in METHODS]
        sds = [stats[m]["sd"] for m, _, _ in METHODS]
        colors = [c for _, _, c in METHODS]
        bars = ax.bar(
            xs,
            means,
            yerr=sds,
            capsize=3.5,
            color=colors,
            edgecolor="#222222",
            linewidth=0.7,
            alpha=0.88,
            width=0.58,
            error_kw={"elinewidth": 1.1, "ecolor": "#222222"},
        )
        bars[-1].set_alpha(1.0)
        bars[-1].set_linewidth(1.3)
        for idx, (m_key, _, _) in enumerate(METHODS):
            s_vals = stats[m_key]["seeds"]
            jit = [-0.10, 0.0, 0.10]
            ax.scatter(
                [idx + j for j in jit],
                s_vals,
                color="#111111",
                s=14,
                zorder=5,
                alpha=0.75,
            )
            top_y = max(means[idx] + sds[idx], max(s_vals))
            lbl = f"{means[idx]:.1f}±{sds[idx]:.1f}%\n({stats[m_key]['macro_tok']:.1f} tok)"
            ax.text(
                idx,
                top_y + ymax * 0.020,
                lbl,
                ha="center",
                va="bottom",
                fontsize=7.3,
                fontweight="bold" if m_key == "comep" else "normal",
                color="#900000" if m_key == "comep" else "#222222",
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [lbl for _, lbl, _ in METHODS],
            fontsize=8.2,
        )
        for tick_lbl in ax.get_xticklabels():
            if "CoMeP" in tick_lbl.get_text():
                tick_lbl.set_fontweight("bold")
        ax.set_ylim(0, ymax)
        ax.set_ylabel("Combined Pooled Success Rate (%)", fontsize=9.5)
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=6)
        ax.grid(axis="y", linestyle="--", alpha=0.32)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=1.3, w_pad=1.8, h_pad=1.6)
    for out_dir in [REPO_ROOT / "paper" / "figures", REPO_ROOT / "paper_artifacts" / "figures"]:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "fig4_four_backbone_comparison.pdf")
        fig.savefig(out_dir / "fig4_four_backbone_comparison.png", dpi=220)
    plt.close(fig)
    print("Exported fig4_four_backbone_comparison.pdf/.png with STIXGeneral TrueType (pdf.fonttype=42)")


if __name__ == "__main__":
    main()
