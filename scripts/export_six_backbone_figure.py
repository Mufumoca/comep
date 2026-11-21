"""Export the 6-panel Stage 3 backbone comparison figure (fig4_six_backbone_comparison.pdf/.png)
and optionally export standalone single-Axes bar charts for the two newly added backbones."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["mathtext.fontset"] = "stix"

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from campaign_source_resolver import (
    STAGE3_CAMPAIGN_SPECS,
    verify_and_load_stage3_campaign,
    verify_source_hashes,
)

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

PANEL_METADATA = {
    "deepseek_v41_flash": (
        "(a) DeepSeek-V4.1-Flash (495 eps/method)",
        "DeepSeek-V4.1-Flash (495 eps/method)",
        96,
    ),
    "glm53_flash_v22": (
        "(b) GLM-5.3-Flash (495 eps/method; OpenCode Go API)",
        "GLM-5.3-Flash (495 eps/method; OpenCode Go API)",
        84,
    ),
    "mimo_v26_flash_v22": (
        "(c) MiMo-V2.6-Flash (495 eps/method; OpenCode Go API)",
        "MiMo-V2.6-Flash (495 eps/method; OpenCode Go API)",
        56,
    ),
    "qwen3_14b_awq_v22": (
        "(d) Qwen3-14B-AWQ (330 eps/method; 4-bit AWQ)",
        "Qwen3-14B-AWQ (330 eps/method; 4-bit AWQ)",
        55,
    ),
    "gemma4_12b_it_v22": (
        "(e) Gemma-4-12B-it (330 eps/method; W4A16 QAT + Proxy Bias)",
        "Gemma-4-12B-it (330 eps/method; W4A16 QAT + Proxy Bias)",
        38,
    ),
    "qwen3_8b": (
        "(f) Qwen3-8B (330 eps/method; BF16)",
        "Qwen3-8B (330 eps/method; BF16)",
        46,
    ),
}


def draw_backbone_bar_ax(ax, methods_stats: dict[str, dict], title: str, ymax: float, label_fs: float = 7.3, title_fs: float = 10.5) -> None:
    xs = np.arange(len(METHODS))
    means = [methods_stats[m]["pooled_sr_pct"] for m, _, _ in METHODS]
    sds = [methods_stats[m]["pooled_sd_pct"] for m, _, _ in METHODS]
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
        s_vals = methods_stats[m_key]["seed_pooled_srs_pct"]
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
        lbl = f"{means[idx]:.1f}±{sds[idx]:.1f}%\n({methods_stats[m_key]['environment_macro_retrieval_tokens']:.1f} tok)"
        ax.text(
            idx,
            top_y + ymax * 0.020,
            lbl,
            ha="center",
            va="bottom",
            fontsize=label_fs,
            fontweight="bold" if m_key == "comep" else "normal",
            color="#900000" if m_key == "comep" else "#222222",
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for _, lbl, _ in METHODS], fontsize=8.2)
    for tick_lbl in ax.get_xticklabels():
        if "CoMeP" in tick_lbl.get_text():
            tick_lbl.set_fontweight("bold")
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Combined Pooled Success Rate (%)", fontsize=9.5)
    ax.set_title(title, fontsize=title_fs, fontweight="bold", pad=6)
    ax.grid(axis="y", linestyle="--", alpha=0.32)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dirs",
        nargs="*",
        default=[str(REPO_ROOT / "paper" / "figures"), str(REPO_ROOT / "paper_artifacts" / "figures")],
        help="Directories to receive fig4_six_backbone_comparison.pdf/.png",
    )
    parser.add_argument(
        "--journal-new-panels-dir",
        default=None,
        help="Optional directory to write standalone single-Axes Figure_7.pdf (GLM-5.3-Flash) and Figure_8.pdf (MiMo-V2.6-Flash)",
    )
    args = parser.parse_args()

    verify_source_hashes()
    loaded = [verify_and_load_stage3_campaign(spec) for spec in STAGE3_CAMPAIGN_SPECS]

    fig, axes = plt.subplots(3, 2, figsize=(14.4, 11.7))
    axes_flat = axes.flatten()
    for ax, camp in zip(axes_flat, loaded):
        ckey = camp["spec"]["key"]
        panel_title, _, ymax = PANEL_METADATA[ckey]
        draw_backbone_bar_ax(ax, camp["methods"], panel_title, ymax, label_fs=7.3, title_fs=10.5)

    fig.tight_layout(pad=1.3, w_pad=1.8, h_pad=1.6)
    for d_str in args.out_dirs:
        out_dir = Path(d_str)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "fig4_six_backbone_comparison.pdf")
        fig.savefig(out_dir / "fig4_six_backbone_comparison.png", dpi=220)
    plt.close(fig)

    if args.journal_new_panels_dir:
        j_dir = Path(args.journal_new_panels_dir)
        j_dir.mkdir(parents=True, exist_ok=True)
        by_key = {c["spec"]["key"]: c for c in loaded}
        for dst_fn, ckey in (("Figure_7.pdf", "glm53_flash_v22"), ("Figure_8.pdf", "mimo_v26_flash_v22")):
            _, standalone_title, ymax = PANEL_METADATA[ckey]
            fig_s, ax_s = plt.subplots(figsize=(6.6, 3.8))
            draw_backbone_bar_ax(ax_s, by_key[ckey]["methods"], standalone_title, ymax, label_fs=7.4, title_fs=10.2)
            fig_s.tight_layout(pad=0.8)
            fig_s.savefig(j_dir / dst_fn)
            plt.close(fig_s)

    print("Exported fig4_six_backbone_comparison.pdf/.png with STIXGeneral TrueType (pdf.fonttype=42)")


if __name__ == "__main__":
    main()
