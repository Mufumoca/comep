"""Export channel-free backbone comparison figures for the GPT-6-Luna token-efficiency edition.

Changes ONLY panel titles that previously named model access channels / APIs / proxy serving
while preserving 100% identical verified Stage-3 data, method sets (all 8 full-campaign methods),
bar values, error bars, seed dots, token annotations, y-axis limits, fonts, and geometry.
"""
from __future__ import annotations

from pathlib import Path
import shutil
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

from campaign_source_resolver import (  # noqa: E402
    verify_and_load_stage3_campaign,
    verify_source_hashes,
)
from export_gpt6_luna_manuscript_artifacts import (  # noqa: E402
    JOURNAL_METHODS_BAR,
    JOURNAL_PALETTE,
    STAGE3_GPT6_CAMPAIGN_SPECS,
    draw_arxiv_panel_ax,
)

CLEAN_ARXIV_PANEL_METADATA: dict[str, tuple[str, float]] = {
    "deepseek_v41_flash": ("(a) DeepSeek-V4.1-Flash (495 eps/method)", 96),
    "glm53_flash_v22": ("(b) GLM-5.3-Flash (495 eps/method; Gated)", 84),
    "gpt6_luna_v22": ("(c) GPT-6-Luna (495 eps/method; Gated)", 84),
    "qwen3_14b_awq_v22": ("(d) Qwen3-14B-AWQ (330 eps/method; 4-bit AWQ)", 55),
    "gemma4_12b_it_v22": ("(e) Gemma-4-12B-it (330 eps/method; W4A16 QAT + Logit Bias)", 38),
    "qwen3_8b": ("(f) Qwen3-8B (330 eps/method; BF16)", 46),
}

CLEAN_JOURNAL_SINGLE_SPECS: tuple[tuple[str, str, str, float], ...] = (
    ("glm53_flash_v22", "Figure_7_clean.pdf", "GLM-5.3-Flash (495 eps/method; Gated)", 90.0),
    ("gpt6_luna_v22", "Figure_8_gpt6_luna_clean.pdf", "GPT-6-Luna (495 eps/method; Gated)", 88.0),
    ("gemma4_12b_it_v22", "Figure_10_clean.pdf", "Gemma-4-12B-it (330 eps/method; W4A16 QAT + Logit Bias)", 39.0),
)


def export_clean_arxiv_six_backbone(loaded_stage3: list[dict], art_dir: Path, paper_dir: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14.4, 11.7))
    for ax, camp in zip(axes.flatten(), loaded_stage3):
        ckey = camp["spec"]["key"]
        panel_title, ymax = CLEAN_ARXIV_PANEL_METADATA[ckey]
        draw_arxiv_panel_ax(ax, camp["methods"], panel_title, ymax)

    fig.tight_layout(pad=1.3, w_pad=1.8, h_pad=1.6)
    art_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)
    art_pdf = art_dir / "fig4_six_backbone_comparison_gpt6_luna_clean.pdf"
    art_png = art_dir / "fig4_six_backbone_comparison_gpt6_luna_clean.png"
    paper_pdf = paper_dir / "fig4_six_backbone_comparison_gpt6_luna_clean.pdf"
    fig.savefig(art_pdf, metadata={"CreationDate": None})
    fig.savefig(art_png, dpi=220)
    plt.close(fig)
    shutil.copy2(art_pdf, paper_pdf)


def export_clean_journal_single(methods_stats: dict[str, dict], title: str, ymax: float, out_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.02, 2.82))
    xs = np.arange(len(JOURNAL_METHODS_BAR))
    means = [methods_stats[m]["pooled_sr_pct"] for m, _ in JOURNAL_METHODS_BAR]
    sds = [methods_stats[m]["pooled_sd_pct"] for m, _ in JOURNAL_METHODS_BAR]
    colors = [JOURNAL_PALETTE[m] for m, _ in JOURNAL_METHODS_BAR]

    ax.grid(axis="y", linestyle="--", color="#CBD5E1", alpha=0.55, zorder=0)

    no_mem_y = methods_stats["no-memory"]["pooled_sr_pct"]
    ax.axhline(no_mem_y, color="#475569", linestyle=(0, (4.5, 3.0)), linewidth=1.0, alpha=0.82, zorder=2)

    bars = ax.bar(
        xs,
        means,
        yerr=sds,
        capsize=3.2,
        color=colors,
        edgecolor="#334155",
        linewidth=0.65,
        alpha=0.88,
        width=0.55,
        error_kw={"elinewidth": 1.0, "ecolor": "#1E293B", "zorder": 4},
        zorder=3,
    )
    bars[-1].set_alpha(0.97)
    bars[-1].set_edgecolor("#7F1D1D")
    bars[-1].set_linewidth(1.3)

    for idx, (m_key, _) in enumerate(JOURNAL_METHODS_BAR):
        s_vals = methods_stats[m_key]["seed_pooled_srs_pct"]
        ax.scatter(
            [idx + j for j in (-0.10, 0.0, 0.10)],
            s_vals,
            color="#0F172A",
            s=12,
            zorder=5,
            alpha=0.78,
        )
        top_y = max(means[idx] + sds[idx], max(s_vals))
        tok = methods_stats[m_key]["environment_macro_retrieval_tokens"]
        lbl = f"{means[idx]:.1f}±{sds[idx]:.1f}%\n({tok:.1f} tok)"
        ax.text(
            idx,
            top_y + ymax * 0.024,
            lbl,
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold" if m_key == "comep" else "normal",
            color="#991B1B" if m_key == "comep" else "#1E293B",
            bbox=dict(boxstyle="round,pad=0.08", facecolor="#FFFFFF", edgecolor="none", alpha=0.86),
            zorder=6,
        )

    ax.plot([], [], color="#475569", linestyle=(0, (4.5, 3.0)), linewidth=1.0, label=f"No-Memory ({no_mem_y:.1f}%)")
    ax.legend(loc="upper left", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=7.3, handlelength=2.1)

    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for _, lbl in JOURNAL_METHODS_BAR], fontsize=7.5)
    ax.tick_params(axis="y", labelsize=7.5)
    for tick_lbl in ax.get_xticklabels():
        if "CoMeP" in tick_lbl.get_text():
            tick_lbl.set_fontweight("bold")
            tick_lbl.set_color("#991B1B")

    ax.set_ylim(0, ymax * 1.06)
    ax.set_ylabel("Combined Pooled Success Rate (%)", fontsize=8.2, color="#0F172A")
    ax.set_title(title, fontsize=8.9, fontweight="bold", pad=6, color="#0F172A")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#475569")
    ax.spines["bottom"].set_color("#475569")

    fig.tight_layout(pad=0.5)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, metadata={"CreationDate": None})
    plt.close(fig)


def main() -> None:
    verify_source_hashes()
    loaded_stage3 = [verify_and_load_stage3_campaign(spec) for spec in STAGE3_GPT6_CAMPAIGN_SPECS]
    by_key = {c["spec"]["key"]: c for c in loaded_stage3}

    export_clean_arxiv_six_backbone(
        loaded_stage3,
        REPO_ROOT / "paper_artifacts" / "figures",
        REPO_ROOT / "paper" / "figures",
    )

    nc_fig_dir = REPO_ROOT / "paper" / "neurocomputing" / "figures"
    for ckey, fname, title, ymax in CLEAN_JOURNAL_SINGLE_SPECS:
        export_clean_journal_single(by_key[ckey]["methods"], title, ymax, nc_fig_dir / fname)

    print("SUCCESS: Exported all 6 channel-free backbone comparison figures.")


if __name__ == "__main__":
    main()
