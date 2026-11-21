#!/usr/bin/env python3
"""Deterministic Round-16 figure exporter (non-destructive new filenames):
  1. paper/neurocomputing/figures/Figure_1_workflow_closed_loop.pdf
     (verified byte-identical copy of Round-15 approved fig0_workflow_closed_loop.pdf,
      allowing Figure 5 to move forward as Figure 1 without touching Figure_5_closed_loop.pdf).
  2. paper/neurocomputing/figures/Figure_2_credit_initial.pdf
     (restores the initial minimal Section 3.1 credit-assignment schematic at native
      3.30 in x 2.38 in single-column geometry, STIXGeneral >= 8.4pt, zero Type-3 fonts).
  3. paper/neurocomputing/figures/Figure_6_seven_methods_bar.pdf
     paper/figures/fig4_six_backbone_seven_methods_bar.{pdf,png}
     paper_artifacts/figures/fig4_six_backbone_seven_methods_bar.{pdf,png}
     (Neurocomputing PHM-LLM-style 3-row stacked clustered bar chart across ALFWorld,
      ScienceWorld, and Combined Pooled SR for all 6 backbones and the 7 main-table
      methods ONLY — excluding Additive Laplace UCB).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

DEFAULT_REPO_ROOT = Path("/root/.paseo/worktrees/1s5ku8sn/lucid-bumblebee")
sys.path.insert(0, str(DEFAULT_REPO_ROOT))
sys.path.insert(0, str(DEFAULT_REPO_ROOT / "scripts"))

from campaign_source_resolver import (
    resolve_and_verify_campaign_dir,
    sha256_file,
    verify_source_hashes,
)
from export_gpt6_luna_manuscript_artifacts import (
    NO_ADDITIVE_METHODS,
    STAGE3_GPT6_CAMPAIGN_SPECS,
)

# Set STIXGeneral serif AFTER imports so imported modules cannot override rcParams
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["mathtext.fontset"] = "stix"

EXPECTED_WORKFLOW_PDF_SHA256 = "e49491960886ad9fc19e4ba7932af6d0f8b9a87c0a32315d679c56091c7339b8"

# Exact 7-method progression (cool slate/blue -> cyan -> sage -> warm sand -> ochre -> coral red for Ours)
# Modeled directly after Neurocomputing 712 PHM-LLM Fig. 5 (excluding Additive Laplace UCB)
SEVEN_METHOD_STYLE: tuple[tuple[str, str, str], ...] = (
    ("no-memory", "No-Memory", "#3F6088"),
    ("full-memory", "Full-Memory (Top-k)", "#6C9AC3"),
    ("individual", "Individual Credit", "#A8D5DE"),
    ("mmr", "MMR-style Diversity", "#6BA57B"),
    ("expel-derived-online", "ExpeL-derived Online", "#F0CE77"),
    ("evomemguard", "EvoMemGuard-style", "#E39652"),
    ("comep", "CoMeP (Ours)", "#D65F4D"),
)


def load_verified_seven_method_metrics() -> list[dict]:
    """Load exact 3-seed means and sample SDs (ddof=1) from pinned raw episodes.jsonl & summary.json."""
    assert tuple(m for m, _, _ in SEVEN_METHOD_STYLE) == NO_ADDITIVE_METHODS
    verify_source_hashes()

    records: list[dict] = []
    seeds = (0, 1, 2)
    for spec in STAGE3_GPT6_CAMPAIGN_SPECS:
        rel_dir = spec["rel_dir"]
        n_alf = int(spec["n_alf"])
        n_sci = int(spec["n_sci"])
        n_tot = n_alf + n_sci

        req_files = ["stage3_main_summary.json", *spec.get("pinned_hashes", {}).keys()]
        for env_name in ("alfworld", "scienceworld"):
            for m_key in NO_ADDITIVE_METHODS:
                for s in seeds:
                    req_files.append(f"{env_name}/{m_key}_seed{s}/summary.json")
                    req_files.append(f"{env_name}/{m_key}_seed{s}/episodes.jsonl")
        req_files = sorted(set(req_files))

        _, base = resolve_and_verify_campaign_dir(
            rel_dir=rel_dir,
            required_rel_files=req_files,
            pinned_hashes=spec.get("pinned_hashes"),
        )

        m_out: dict[str, dict] = {}
        for m_key in NO_ADDITIVE_METHODS:
            alf_wins: list[int] = []
            sci_wins: list[int] = []
            pool_wins: list[int] = []
            for s in seeds:
                for env_name, n_exp, target_list in (
                    ("alfworld", n_alf, alf_wins),
                    ("scienceworld", n_sci, sci_wins),
                ):
                    run_dir = base / env_name / f"{m_key}_seed{s}"
                    summ = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
                    if summ.get("status") != "completed":
                        raise RuntimeError(f"Incomplete run: {run_dir}")
                    ep_recs = [
                        json.loads(ln)
                        for ln in (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
                        if ln.strip()
                    ]
                    if len(ep_recs) != n_exp:
                        raise RuntimeError(f"Episode count mismatch in {run_dir}")
                    w = sum(int(round(float(r["reward"]))) for r in ep_recs)
                    if w != int(round(float(summ["success_rate"]) * n_exp)):
                        raise RuntimeError(f"Win mismatch in {run_dir}")
                    target_list.append(w)
                pool_wins.append(alf_wins[-1] + sci_wins[-1])

            alf_srs = [100.0 * w / n_alf for w in alf_wins]
            sci_srs = [100.0 * w / n_sci for w in sci_wins]
            pool_srs = [100.0 * w / n_tot for w in pool_wins]

            m_out[m_key] = {
                "alf_mean": float(np.mean(alf_srs)),
                "alf_sd": float(np.std(alf_srs, ddof=1)),
                "sci_mean": float(np.mean(sci_srs)),
                "sci_sd": float(np.std(sci_srs, ddof=1)),
                "pool_mean": float(np.mean(pool_srs)),
                "pool_sd": float(np.std(pool_srs, ddof=1)),
                "alf_wins": alf_wins,
                "sci_wins": sci_wins,
                "pool_wins": pool_wins,
            }
        records.append(
            {
                "key": spec["key"],
                "backbone_name": spec["backbone_name"],
                "n_alf": n_alf,
                "n_sci": n_sci,
                "n_tot_per_method": 3 * n_tot,
                "warmup_gate": bool(spec["warmup_gate"]),
                "methods": m_out,
            }
        )
    return records


def export_initial_credit_schematic(out_pdf: Path, out_png: Path | None = None) -> None:
    """Restore the initial minimal Section 3.1 schematic (matching approved_figs_item7/Figure_1.pdf
    and panel (a) of fig0_credit_coupling.pdf) at native 3.30 in x 2.38 in single-column size."""
    fig, ax = plt.subplots(figsize=(3.30, 2.38))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Top half: Independent Credit Update Paradigm
    ax.text(
        5.0,
        9.52,
        "Independent Credit Update",
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color="#222222",
    )
    r_ind = mpatches.Circle(
        (1.95, 7.25), 0.76, facecolor="#ffffff", edgecolor="#222222", linewidth=1.15, zorder=3
    )
    ax.add_patch(r_ind)
    ax.text(1.95, 7.25, "$r_\\tau$", ha="center", va="center", fontsize=9.2)

    upd_i = FancyBboxPatch(
        (4.35, 7.85),
        3.15,
        0.95,
        boxstyle="round,pad=0.10",
        facecolor="#f8f9fa",
        edgecolor="#222222",
        linewidth=1.05,
        zorder=3,
    )
    upd_j = FancyBboxPatch(
        (4.35, 5.72),
        3.15,
        0.95,
        boxstyle="round,pad=0.10",
        facecolor="#f8f9fa",
        edgecolor="#222222",
        linewidth=1.05,
        zorder=3,
    )
    ax.add_patch(upd_i)
    ax.add_patch(upd_j)
    ax.text(5.92, 8.32, "Update $m_i$", ha="center", va="center", fontsize=8.6, color="#111111")
    ax.text(5.92, 6.20, "Update $m_j$", ha="center", va="center", fontsize=8.6, color="#111111")

    ax.add_patch(
        FancyArrowPatch(
            (2.71, 7.50),
            (4.30, 8.28),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.05,
            color="#222222",
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (2.71, 7.00),
            (4.30, 6.22),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.05,
            color="#222222",
        )
    )
    ax.text(
        8.82,
        7.25,
        "Ignores\n$c_\\tau,\\; s_{ij}$",
        ha="center",
        va="center",
        fontsize=8.4,
        color="#555555",
    )

    ax.plot([0.25, 9.75], [5.05, 5.05], color="#b5b5b5", linewidth=0.9, linestyle=":")

    # Bottom half: CoMeP Predictive Model
    ax.text(
        5.0,
        4.42,
        "CoMeP Predictive Model",
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color="#1f4e78",
    )
    nodes_com = [(1.55, 2.95, "$m_i$"), (5.0, 2.95, "$m_j$"), (8.45, 2.95, "$c_\\tau$")]
    for nx, ny, lbl in nodes_com:
        ec = "#1f4e78" if "c_" in lbl else "#222222"
        fc = "#ebf3f9" if "c_" in lbl else "#ffffff"
        circ = mpatches.Circle((nx, ny), 0.66, facecolor=fc, edgecolor=ec, linewidth=1.1, zorder=3)
        ax.add_patch(circ)
        ax.text(nx, ny, lbl, ha="center", va="center", fontsize=9.0)

    syn_box = FancyBboxPatch(
        (2.72, 1.54),
        1.16,
        0.65,
        boxstyle="round,pad=0.08",
        facecolor="#fdecea",
        edgecolor="#c0392b",
        linewidth=1.1,
        zorder=3,
    )
    ax.add_patch(syn_box)
    ax.text(
        3.30,
        1.86,
        "$s_{ij}$",
        ha="center",
        va="center",
        fontsize=8.8,
        color="#c0392b",
        fontweight="bold",
    )
    ax.plot([1.55 + 0.46, 2.72], [2.95 - 0.46, 1.86], color="#c0392b", linewidth=1.05)
    ax.plot([5.0 - 0.46, 3.88], [2.95 - 0.46, 1.86], color="#c0392b", linewidth=1.05)

    r_com = mpatches.Ellipse(
        (5.0, 0.56), 2.65, 0.98, facecolor="#ffffff", edgecolor="#222222", linewidth=1.15, zorder=3
    )
    ax.add_patch(r_com)
    ax.text(5.0, 0.56, "$\\eta_\\tau \\to r_\\tau$", ha="center", va="center", fontsize=8.8)

    ax.add_patch(
        FancyArrowPatch(
            (1.55, 2.29),
            (3.72, 0.66),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.05,
            color="#222222",
            connectionstyle="arc3,rad=0.18",
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (5.0, 2.29),
            (5.0, 1.07),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.05,
            color="#222222",
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (3.30, 1.54),
            (4.22, 0.89),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.1,
            color="#c0392b",
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (8.45, 2.29),
            (6.28, 0.66),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.1,
            color="#1f77b4",
            connectionstyle="arc3,rad=-0.18",
        )
    )
    ax.text(8.02, 1.15, "$\\alpha_{c_\\tau}$", fontsize=8.8, color="#1f77b4", fontweight="bold")
    ax.text(1.22, 1.25, "$\\theta_i$", fontsize=8.8, color="#222222")
    ax.text(5.42, 1.60, "$\\theta_j$", fontsize=8.8, color="#222222")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, metadata={"CreationDate": None})
    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, metadata={"CreationTime": None})
    plt.close(fig)


def export_seven_method_consolidated_bar(
    records: list[dict],
    out_pdfs: list[Path],
    out_pngs: list[Path],
    audit_json: Path | None = None,
) -> None:
    """Export Neurocomputing PHM-LLM-style 3-row stacked clustered bar chart
    across the 6 backbones and 7 main-table methods (excluding Additive Laplace UCB).

    Native size: 7.20 in x 5.05 in (518.4 pt x 363.6 pt).
      - In Neurocomputing 0.98\\textwidth = 484.6 pt (scale 0.935x): 9.6pt -> 8.97pt >= 7pt.
      - In arXiv 0.99\\textwidth = 393.5 pt (scale 0.759x): 9.6pt -> 7.29pt >= 7pt.
    """
    fig, axes = plt.subplots(3, 1, figsize=(7.20, 5.05), sharex=True)

    n_bb = len(records)
    n_m = len(SEVEN_METHOD_STYLE)
    x_centers = np.arange(n_bb, dtype=float)
    bar_width = 0.112
    offsets = (np.arange(n_m, dtype=float) - 0.5 * (n_m - 1)) * bar_width

    row_specs = (
        ("alf_mean", "alf_sd", "ALFWorld SR (%)"),
        ("sci_mean", "sci_sd", "ScienceWorld SR (%)"),
        ("pool_mean", "pool_sd", "Combined SR (%)"),
    )

    clipped_whiskers: list[dict] = []
    legend_handles: list[mpatches.Patch] = []
    legend_labels: list[str] = []

    for row_idx, (ax, (mean_key, sd_key, ylabel)) in enumerate(zip(axes, row_specs)):
        ax.set_facecolor("#FFFFFF")
        ax.grid(axis="y", linestyle="--", color="#D1D5DB", linewidth=0.65, alpha=0.85, zorder=0)

        for m_idx, (m_key, m_label, m_color) in enumerate(SEVEN_METHOD_STYLE):
            means = np.array([rec["methods"][m_key][mean_key] for rec in records], dtype=float)
            sds = np.array([rec["methods"][m_key][sd_key] for rec in records], dtype=float)
            yerr_low = np.minimum(sds, means)
            yerr_high = np.minimum(sds, 100.0 - means)

            for bb_idx, rec in enumerate(records):
                if sds[bb_idx] > means[bb_idx] + 1e-9:
                    clipped_whiskers.append(
                        {
                            "panel": ylabel,
                            "backbone": rec["backbone_name"],
                            "method": m_key,
                            "mean_pct": round(float(means[bb_idx]), 4),
                            "sd_pct": round(float(sds[bb_idx]), 4),
                        }
                    )

            ax.bar(
                x_centers + offsets[m_idx],
                means,
                width=bar_width,
                yerr=np.vstack([yerr_low, yerr_high]),
                capsize=1.9,
                color=m_color,
                edgecolor="#FFFFFF",
                linewidth=0.50,
                error_kw={
                    "ecolor": "#1E293B",
                    "elinewidth": 0.80,
                    "capthick": 0.80,
                    "zorder": 4,
                },
                zorder=3,
            )

            if row_idx == 0:
                legend_handles.append(
                    mpatches.Patch(facecolor=m_color, edgecolor="#FFFFFF", linewidth=0.4)
                )
                legend_labels.append(m_label)

        ax.set_ylim(0.0, 100.0)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(axis="y", labelsize=9.6, direction="out", length=3.4, width=0.95, colors="#111111")
        ax.tick_params(axis="x", labelsize=9.6, direction="out", length=3.4, width=0.95, colors="#111111")
        ax.set_ylabel(ylabel, fontsize=10.0, fontweight="bold", color="#111111", labelpad=5)
        ax.set_xlim(-0.54, n_bb - 0.46)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#111111")
            spine.set_linewidth(1.05)

    bb_labels = [rec["backbone_name"] for rec in records]
    axes[-1].set_xticks(x_centers)
    axes[-1].set_xticklabels(bb_labels, fontsize=9.6, color="#111111")
    axes[-1].set_xlabel("LLM Backbone", fontsize=10.2, fontweight="bold", style="italic", labelpad=5)

    # Compact two-row top rounded legend box matching Neurocomputing PHM-LLM aesthetic
    # while keeping full table method names at >= 9.6pt native (>= 7.3pt in arXiv!)
    leg = fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.515, 0.995),
        ncol=4,
        frameon=True,
        fancybox=True,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        framealpha=1.0,
        fontsize=9.6,
        handlelength=1.25,
        handleheight=0.78,
        handletextpad=0.45,
        columnspacing=1.15,
        borderpad=0.42,
    )
    leg.get_frame().set_linewidth(0.85)
    for txt in leg.get_texts():
        if "CoMeP" in txt.get_text():
            txt.set_fontweight("bold")

    fig.subplots_adjust(left=0.088, right=0.992, top=0.875, bottom=0.095, hspace=0.14)

    for p in out_pdfs:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, metadata={"CreationDate": None})
    for p in out_pngs:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, metadata={"CreationTime": None})
    plt.close(fig)

    if audit_json is not None:
        audit_payload = {
            "displayed_methods": [
                {"key": k, "label": lbl, "hex_color": col}
                for k, lbl, col in SEVEN_METHOD_STYLE
            ],
            "excluded_ablation_methods": ["additive"],
            "clipped_lower_whiskers_at_zero": clipped_whiskers,
            "backbones": records,
        }
        audit_json.parent.mkdir(parents=True, exist_ok=True)
        audit_json.write_text(json.dumps(audit_payload, indent=2) + "\n", encoding="utf-8")


def verify_pdf_fonts_and_min_size(
    pdf_path: Path,
    target_widths_pt: dict[str, float],
    configured_min_native_pt: float,
    min_rendered_ordinary_pt: float = 7.0,
) -> dict:
    """Verify 0 Type-3 / unembedded fonts and check ordinary text sizes via exact scaling
    and integer-rounded pdftohtml -xml -zoom 1.0 extraction."""
    fonts_out = subprocess.check_output(["pdffonts", str(pdf_path)], text=True)
    lines = [ln for ln in fonts_out.splitlines()[2:] if ln.strip()]
    if not lines:
        raise RuntimeError(f"No embedded fonts found in {pdf_path}")
    for ln in lines:
        if "Type 3" in ln:
            raise RuntimeError(f"Type 3 font detected in {pdf_path}: {ln}")
        toks = ln.split()
        if len(toks) >= 5 and toks[-5] == "no":
            raise RuntimeError(f"Unembedded font detected in {pdf_path}: {ln}")

    xml_bytes = subprocess.check_output(
        ["pdftohtml", "-xml", "-zoom", "1.0", "-stdout", str(pdf_path)]
    )
    root = ET.fromstring(xml_bytes)
    page = root.find("page")
    assert page is not None
    native_w = float(page.attrib["width"])
    native_h = float(page.attrib["height"])

    font_sizes = {
        fs.attrib["id"]: float(fs.attrib["size"])
        for fs in page.findall("fontspec")
    }
    ordinary_sizes: list[tuple[float, str]] = []
    for t in page.findall("text"):
        txt = "".join(t.itertext()).strip()
        if not txt:
            continue
        fid = t.attrib["font"]
        sz = font_sizes[fid]
        # Exclude single/two-char pure mathematical subscripts (tau, i, j, ij, c_tau)
        if txt in {"τ", "i", "j", "ij", "c"} and sz < 8.0:
            continue
        ordinary_sizes.append((sz, txt))

    pdftohtml_rounded_native_pt = min(sz for sz, _ in ordinary_sizes)
    exact_scaled_mins: dict[str, float] = {}
    pdftohtml_rounded_scaled_mins: dict[str, float] = {}
    for label, tw in target_widths_pt.items():
        scale = tw / native_w
        exact_min = configured_min_native_pt * scale
        rounded_min = pdftohtml_rounded_native_pt * scale
        exact_scaled_mins[label] = round(exact_min, 3)
        pdftohtml_rounded_scaled_mins[label] = round(rounded_min, 3)
        if min(exact_min, rounded_min) + 1e-6 < min_rendered_ordinary_pt:
            raise RuntimeError(
                f"{pdf_path.name} rendered in {label} ({tw}pt, scale={scale:.3f}) has min ordinary font "
                f"exact={exact_min:.2f}pt / rounded={rounded_min:.2f}pt < {min_rendered_ordinary_pt}pt!"
            )
    return {
        "pdf": str(pdf_path),
        "sha256": sha256_file(pdf_path),
        "native_width_pt": native_w,
        "native_height_pt": native_h,
        "configured_min_native_ordinary_pt": configured_min_native_pt,
        "pdftohtml_integer_rounded_native_pt": pdftohtml_rounded_native_pt,
        "exact_scaled_min_ordinary_pt": exact_scaled_mins,
        "pdftohtml_rounded_scaled_min_ordinary_pt": pdftohtml_rounded_scaled_mins,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        required=True,
        help="Staging directory (during pre-review) or repo root (upon promotion).",
    )
    args = parser.parse_args()
    out_root = Path(args.out_root).resolve()

    # 1. Verify and copy closed-loop workflow diagram to Figure_1_workflow_closed_loop.pdf
    src_workflow = DEFAULT_REPO_ROOT / "paper" / "figures" / "fig0_workflow_closed_loop.pdf"
    got_sha = sha256_file(src_workflow)
    if got_sha != EXPECTED_WORKFLOW_PDF_SHA256:
        raise RuntimeError(
            f"Unexpected fig0_workflow_closed_loop.pdf SHA256: {got_sha} != {EXPECTED_WORKFLOW_PDF_SHA256}"
        )
    dst_fig1 = out_root / "paper" / "neurocomputing" / "figures" / "Figure_1_workflow_closed_loop.pdf"
    dst_fig1.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_workflow, dst_fig1)

    # 2. Export restored initial Section 3.1 schematic as Figure_2_credit_initial.pdf
    dst_fig2 = out_root / "paper" / "neurocomputing" / "figures" / "Figure_2_credit_initial.pdf"
    dst_fig2_png = out_root / "paper_artifacts" / "figures" / "Figure_2_credit_initial.png"
    export_initial_credit_schematic(dst_fig2, dst_fig2_png)

    # 3. Export consolidated 7-method 3-row stacked clustered bar chart
    records = load_verified_seven_method_metrics()
    dst_bar_neuro = out_root / "paper" / "neurocomputing" / "figures" / "Figure_6_seven_methods_bar.pdf"
    dst_bar_arxiv = out_root / "paper" / "figures" / "fig4_six_backbone_seven_methods_bar.pdf"
    dst_bar_art_pdf = out_root / "paper_artifacts" / "figures" / "fig4_six_backbone_seven_methods_bar.pdf"
    dst_bar_arxiv_png = out_root / "paper" / "figures" / "fig4_six_backbone_seven_methods_bar.png"
    dst_bar_art_png = out_root / "paper_artifacts" / "figures" / "fig4_six_backbone_seven_methods_bar.png"
    dst_audit_json = out_root / "paper_artifacts" / "seven_method_bar_audit_round16.json"

    export_seven_method_consolidated_bar(
        records=records,
        out_pdfs=[dst_bar_neuro, dst_bar_arxiv, dst_bar_art_pdf],
        out_pngs=[dst_bar_arxiv_png, dst_bar_art_png],
        audit_json=dst_audit_json,
    )

    # Verify byte-identical PDFs across bar chart targets if deterministic or copy from primary
    shutil.copy2(dst_bar_neuro, dst_bar_arxiv)
    shutil.copy2(dst_bar_neuro, dst_bar_art_pdf)

    # 4. Run automated font & legibility checks
    check_fig2 = verify_pdf_fonts_and_min_size(
        dst_fig2,
        target_widths_pt={
            "neuro_en_0.95col": 0.95 * 238.2,
            "neuro_zh_0.95col": 0.95 * 238.2,
        },
        configured_min_native_pt=8.4,
        min_rendered_ordinary_pt=7.0,
    )
    check_bar = verify_pdf_fonts_and_min_size(
        dst_bar_neuro,
        target_widths_pt={
            "neuro_0.98textwidth": 0.98 * 494.5,
            "arxiv_0.99textwidth": 0.99 * 397.5,
        },
        configured_min_native_pt=9.6,
        min_rendered_ordinary_pt=7.0,
    )
    print(json.dumps({"fig2_initial_check": check_fig2, "seven_method_bar_check": check_bar}, indent=2))


if __name__ == "__main__":
    main()
