"""Export FreDF-style conceptual and workflow diagrams for Section 3 (fig0_credit_coupling.pdf and fig0_workflow_diagram.pdf) at native 5.5-inch textwidth so all embedded labels render at 7-10pt."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import TwoSlopeNorm
import numpy as np

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["mathtext.fontset"] = "stix"

REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_fig0_credit_coupling(out_dirs: list[Path], manifest_out: Path) -> None:
    alf_ep_path = REPO_ROOT / "outputs/stage3-main/alfworld/comep_seed0/episodes.jsonl"
    syn_truth_path = REPO_ROOT / "outputs/stage1-gate1-benchmark/agent-composite/truth-seed-0.json"
    syn_state_path = REPO_ROOT / "outputs/stage1-gate1-benchmark/agent-composite/comep-seed-0-state.json"
    syn_manifest_path = REPO_ROOT / "outputs/stage1-gate1-benchmark/agent-composite/manifest.json"

    # --- Load Panel (b): ALFWorld seed 0 co-retrieval counts across all 90 episodes ---
    alf_episodes = [json.loads(line) for line in alf_ep_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    item_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    for rec in alf_episodes:
        sel = sorted(set(rec["selected"]))
        for item_id in sel:
            item_counts[item_id] += 1
        for a, b in combinations(sel, 2):
            pair_counts[(a, b)] += 1

    # Deterministic top-12 rules by total exposure count descending, tie-broken by rule ID ascending
    top12_rules = [
        r_id
        for r_id, _ in sorted(item_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
    ]
    n_top = len(top12_rules)
    co_matrix = np.full((n_top, n_top), np.nan, dtype=float)
    for i in range(n_top):
        for j in range(n_top):
            if i != j:
                key = tuple(sorted((top12_rules[i], top12_rules[j])))
                co_matrix[i, j] = float(pair_counts.get(key, 0))

    def _short_alf_label(rid: str) -> str:
        if rid.startswith("alf-ws-"):
            return "ws-" + rid.split("-")[-1]
        if rid.startswith("distilled-s0-"):
            return "d-" + rid.split("-")[-1]
        return rid

    top12_short = [_short_alf_label(r) for r in top12_rules]

    # --- Load Panels (c) & (d): Synthetic agent-composite seed 0 (24 rules) ---
    truth_data = json.loads(syn_truth_path.read_text(encoding="utf-8"))
    state_data = json.loads(syn_state_path.read_text(encoding="utf-8"))

    all_24_rules = sorted(
        truth_data["rules"].keys(),
        key=lambda rid: int(rid.split("-")[-1]),
    )
    n_syn = len(all_24_rules)
    syn_idx = {rid: idx for idx, rid in enumerate(all_24_rules)}
    syn_short = [f"m{int(rid.split('-')[-1]):02d}" for rid in all_24_rules]

    # Panel (c): Planted oracle interaction matrix (0 where unplanted, NaN on diagonal)
    planted_mat = np.zeros((n_syn, n_syn), dtype=float)
    np.fill_diagonal(planted_mat, np.nan)
    for p in truth_data["pairs"]:
        a, b = p["identifiers"]
        ia, ib = syn_idx[a], syn_idx[b]
        planted_mat[ia, ib] = planted_mat[ib, ia] = float(p["effect"])

    # Panel (d): Stored fitted interaction coefficients in comep-seed-0-state.json (NaN where absent from final snapshot)
    fitted_mat = np.full((n_syn, n_syn), np.nan, dtype=float)
    stored_pair_count = 0
    planted_pair_set = {tuple(sorted(p["identifiers"])) for p in truth_data["pairs"]}
    stored_matching_planted = 0
    for p in state_data["pairs"]:
        a, b = p["identifiers"]
        if a in syn_idx and b in syn_idx:
            ia, ib = syn_idx[a], syn_idx[b]
            fitted_mat[ia, ib] = fitted_mat[ib, ia] = float(p["mean"])
            stored_pair_count += 1
            if tuple(sorted((a, b))) in planted_pair_set:
                stored_matching_planted += 1

    # Save audit manifest
    audit_payload = {
        "script": "scripts/export_method_diagrams.py",
        "source_hashes": {
            "outputs/stage3-main/alfworld/comep_seed0/episodes.jsonl": sha256_file(alf_ep_path),
            "outputs/stage1-gate1-benchmark/agent-composite/truth-seed-0.json": sha256_file(syn_truth_path),
            "outputs/stage1-gate1-benchmark/agent-composite/comep-seed-0-state.json": sha256_file(syn_state_path),
            "outputs/stage1-gate1-benchmark/agent-composite/manifest.json": sha256_file(syn_manifest_path),
        },
        "panel_b_alfworld_co_retrieval": {
            "total_episodes": len(alf_episodes),
            "selection_rule": "Top 12 rules by total episode exposure count n_i across all 90 episodes (including rules later pruned), tie-broken by rule ID ascending.",
            "top12_rules": top12_rules,
            "top12_exposures": [int(item_counts[r]) for r in top12_rules],
            "co_retrieval_matrix": [[None if np.isnan(v) else int(v) for v in row] for row in co_matrix],
        },
        "panel_cd_synthetic_agent_composite_seed0": {
            "rule_order": all_24_rules,
            "planted_nonzero_pair_count": len(truth_data["pairs"]),
            "stored_pair_count_in_state": stored_pair_count,
            "stored_pairs_matching_planted": stored_matching_planted,
            "shared_color_scale_vmin_vmax": [-4.0, 4.0],
            "planted_matrix": [[None if np.isnan(v) else round(float(v), 4) for v in row] for row in planted_mat],
            "fitted_matrix_null_means_absent_from_final_snapshot": [[None if np.isnan(v) else round(float(v), 4) for v in row] for row in fitted_mat],
        },
    }
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(audit_payload, indent=2) + "\n", encoding="utf-8")

    # --- Plot 2x2 grid at native 5.5-inch textwidth so fonts are 7-9.5pt in PDF ---
    fig = plt.figure(figsize=(5.5, 4.65))
    gs = fig.add_gridspec(2, 2, hspace=0.44, wspace=0.36)

    # Panel (a): Schematic of credit assignment / predictive dependencies
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_xlim(0, 10)
    ax0.set_ylim(0, 10)
    ax0.axis("off")
    ax0.set_title("(a) Credit Assignment Schematic", fontsize=8.8, fontweight="bold", pad=4)

    # Top half: Independent Credit Update Paradigm (r_tau updates each co-retrieved rule separately)
    ax0.text(5.0, 9.55, "Independent Credit Update", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#222222")
    r_ind = mpatches.Circle((2.0, 7.25), 0.72, facecolor="#ffffff", edgecolor="#222222", linewidth=1.1, zorder=3)
    ax0.add_patch(r_ind)
    ax0.text(2.0, 7.25, "$r_\\tau$", ha="center", va="center", fontsize=8.2)

    upd_i = FancyBboxPatch((4.5, 7.85), 3.1, 0.92, boxstyle="round,pad=0.10", facecolor="#f8f9fa", edgecolor="#222222", linewidth=1.0, zorder=3)
    upd_j = FancyBboxPatch((4.5, 5.75), 3.1, 0.92, boxstyle="round,pad=0.10", facecolor="#f8f9fa", edgecolor="#222222", linewidth=1.0, zorder=3)
    ax0.add_patch(upd_i)
    ax0.add_patch(upd_j)
    ax0.text(6.05, 8.31, "Update $m_i$", ha="center", va="center", fontsize=7.5, color="#111111")
    ax0.text(6.05, 6.21, "Update $m_j$", ha="center", va="center", fontsize=7.5, color="#111111")

    ax0.add_patch(FancyArrowPatch((2.72, 7.50), (4.45, 8.28), arrowstyle="-|>", mutation_scale=8, linewidth=1.0, color="#222222"))
    ax0.add_patch(FancyArrowPatch((2.72, 7.00), (4.45, 6.24), arrowstyle="-|>", mutation_scale=8, linewidth=1.0, color="#222222"))
    ax0.text(8.85, 7.25, "Ignores\n$c_\\tau, s_{ij}$", ha="center", va="center", fontsize=7.0, color="#666666")

    ax0.plot([0.3, 9.7], [5.05, 5.05], color="#bbbbbb", linewidth=0.8, linestyle=":")

    # Bottom half: CoMeP Joint Predictive Model
    ax0.text(5.0, 4.45, "CoMeP Predictive Model", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#1f4e78")
    nodes_com = [(1.6, 2.95, "$m_i$"), (5.0, 2.95, "$m_j$"), (8.4, 2.95, "$c_\\tau$")]
    for nx, ny, lbl in nodes_com:
        ec = "#1f4e78" if "c_" in lbl else "#222222"
        fc = "#ebf3f9" if "c_" in lbl else "#ffffff"
        circ = mpatches.Circle((nx, ny), 0.62, facecolor=fc, edgecolor=ec, linewidth=1.05, zorder=3)
        ax0.add_patch(circ)
        ax0.text(nx, ny, lbl, ha="center", va="center", fontsize=8.0)

    # Joint interaction node s_ij between m_i and m_j
    syn_box = FancyBboxPatch((2.75, 1.55), 1.1, 0.62, boxstyle="round,pad=0.08", facecolor="#fdecea", edgecolor="#c0392b", linewidth=1.05, zorder=3)
    ax0.add_patch(syn_box)
    ax0.text(3.3, 1.86, "$s_{ij}$", ha="center", va="center", fontsize=7.8, color="#c0392b", fontweight="bold")
    ax0.plot([1.6 + 0.44, 2.75], [2.95 - 0.44, 1.86], color="#c0392b", linewidth=1.0)
    ax0.plot([5.0 - 0.44, 3.85], [2.95 - 0.44, 1.86], color="#c0392b", linewidth=1.0)

    r_com = mpatches.Ellipse((5.0, 0.55), 2.5, 0.95, facecolor="#ffffff", edgecolor="#222222", linewidth=1.1, zorder=3)
    ax0.add_patch(r_com)
    ax0.text(5.0, 0.55, "$\\eta_\\tau \\to r_\\tau$", ha="center", va="center", fontsize=7.6)

    ax0.add_patch(FancyArrowPatch((1.6, 2.33), (3.80, 0.65), arrowstyle="-|>", mutation_scale=8, linewidth=1.0, color="#222222", connectionstyle="arc3,rad=0.18"))
    ax0.add_patch(FancyArrowPatch((5.0, 2.33), (5.0, 1.05), arrowstyle="-|>", mutation_scale=8, linewidth=1.0, color="#222222"))
    ax0.add_patch(FancyArrowPatch((3.3, 1.55), (4.25, 0.88), arrowstyle="-|>", mutation_scale=8, linewidth=1.05, color="#c0392b"))
    ax0.add_patch(FancyArrowPatch((8.4, 2.33), (6.20, 0.65), arrowstyle="-|>", mutation_scale=8, linewidth=1.05, color="#1f77b4", connectionstyle="arc3,rad=-0.18"))
    ax0.text(7.65, 1.35, "$\\alpha_{c_\\tau}$", fontsize=7.6, color="#1f77b4", fontweight="bold")
    ax0.text(1.30, 1.25, "$\\theta_i$", fontsize=7.6, color="#222222")
    ax0.text(5.45, 1.62, "$\\theta_j$", fontsize=7.6, color="#222222")

    # Panel (b): ALFWorld Co-Retrieval Counts
    ax1 = fig.add_subplot(gs[0, 1])
    cmap_b = plt.cm.Blues.copy()
    cmap_b.set_bad(color="#e6e6e6")
    im1 = ax1.imshow(np.ma.masked_invalid(co_matrix), cmap=cmap_b, vmin=0, vmax=max(1.0, float(np.nanmax(co_matrix))))
    ax1.set_xticks(range(n_top))
    ax1.set_xticklabels(top12_short, rotation=90, fontsize=6.6)
    ax1.set_yticks(range(n_top))
    ax1.set_yticklabels([f"{s} ({item_counts[r]})" for s, r in zip(top12_short, top12_rules)], fontsize=6.5)
    ax1.set_title("(b) Co-Retrieval Counts $n_{ij}$ (ALFWorld)", fontsize=8.6, fontweight="bold", pad=4)
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)
    cbar1.ax.tick_params(labelsize=6.8)

    # Shared diverging colormap for Panels (c) and (d)
    cmap_cd = plt.cm.RdBu_r.copy()
    cmap_cd.set_bad(color="#d9d9d9")
    norm_cd = TwoSlopeNorm(vmin=-4.0, vcenter=0.0, vmax=4.0)
    tick_pos = list(range(0, n_syn, 3))

    # Panel (c): Planted Oracle Interaction Weights
    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(np.ma.masked_invalid(planted_mat), cmap=cmap_cd, norm=norm_cd)
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels([syn_short[i] for i in tick_pos], rotation=90, fontsize=6.8)
    ax2.set_yticks(tick_pos)
    ax2.set_yticklabels([syn_short[i] for i in tick_pos], fontsize=6.8)
    ax2.set_title("(c) Planted Oracle Interaction Weights", fontsize=8.6, fontweight="bold", pad=4)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03, ticks=[-4, -2, 0, 2, 4])
    cbar2.ax.tick_params(labelsize=6.8)

    # Panel (d): Stored Fitted Interaction Coefficients
    ax3 = fig.add_subplot(gs[1, 1])
    im3 = ax3.imshow(np.ma.masked_invalid(fitted_mat), cmap=cmap_cd, norm=norm_cd)
    ax3.set_xticks(tick_pos)
    ax3.set_xticklabels([syn_short[i] for i in tick_pos], rotation=90, fontsize=6.8)
    ax3.set_yticks(tick_pos)
    ax3.set_yticklabels([syn_short[i] for i in tick_pos], fontsize=6.8)
    ax3.set_title("(d) Stored Fitted Coeffs. $\\hat{s}_{ij}$ (Gray: Not Stored)", fontsize=8.4, fontweight="bold", pad=4)
    cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.03, ticks=[-4, -2, 0, 2, 4])
    cbar3.ax.tick_params(labelsize=6.8)

    fig.subplots_adjust(left=0.05, right=0.96, top=0.93, bottom=0.08)
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "fig0_credit_coupling.pdf")
        fig.savefig(out_dir / "fig0_credit_coupling.png", dpi=260)
    plt.close(fig)


def export_fig0_workflow_diagram(out_dirs: list[Path]) -> None:
    # Native 5.5-inch width so that 7.2-8.8pt fonts render at exact 1:1 scale in the paper
    fig, ax = plt.subplots(figsize=(5.5, 3.65))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")

    def draw_box(x, y, w, h, title, subtitle="", fc="#ffffff", ec="#222222", lw=1.0, title_color="#111111", sub_color="#333333", fs_t=7.8, fs_s=6.8, ls="-"):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.45", facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=3)
        ax.add_patch(patch)
        if subtitle:
            ax.text(x + w / 2, y + h * 0.67, title, ha="center", va="center", fontsize=fs_t, fontweight="bold", color=title_color, zorder=4)
            ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center", fontsize=fs_s, color=sub_color, zorder=4)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs_t, fontweight="bold", color=title_color, zorder=4)

    def draw_arrow(p1, p2, color="#222222", lw=1.05, rad=0.0, ls="-"):
        arr = FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=8.5,
            linewidth=lw,
            color=color,
            linestyle=ls,
            connectionstyle=f"arc3,rad={rad}",
            zorder=4,
        )
        ax.add_patch(arr)

    # --- ROW 1 (Top): Forward Routing & Rollout ---
    draw_box(
        1.0, 49.0, 20.5, 11.5,
        "1. Episode Input",
        "Query $q_t$, context $c_t$\nActive bank $\\mathcal{M}_t$",
        fc="#f8f9fa", ec="#444444",
    )
    draw_box(
        24.5, 49.0, 21.5, 11.5,
        "2. Relevance Gate",
        "Clause split & contrast\ngate $\\tilde{g}_{t,i}$ (Eq. 4–5)",
        fc="#ebf3f9", ec="#1f4e78", title_color="#1f4e78",
    )
    draw_box(
        49.0, 49.0, 26.5, 11.5,
        "3. Lookahead Routing",
        "Gate $G_t$, conflict & early-stop\nguards (0 Manager LLM calls)",
        fc="#e8f1fb", ec="#1f618d", lw=1.2, title_color="#154360",
    )
    draw_box(
        78.5, 49.0, 20.5, 11.5,
        "4. Rollout",
        "Fixed bundle $S_t$\nOutcome $r_t \\in \\{0, 1\\}$",
        fc="#ffffff", ec="#333333",
    )
    draw_arrow((21.5, 54.75), (24.5, 54.75), color="#1f4e78")
    draw_arrow((46.0, 54.75), (49.0, 54.75), color="#1f618d")
    draw_arrow((75.5, 54.75), (78.5, 54.75), color="#1f618d")

    # --- ROW 2 (Middle, right-to-left return loop): Fit -> Optional Distill -> Conditional Prune ---
    draw_box(
        68.5, 32.5, 30.5, 11.5,
        "5. 2nd-Order Laplace Fit",
        "Window $\\mathcal{H}_t$ + hist. cols. (Eq. 6–8)\nUpdate $(\\hat{\\alpha}_c, \\hat{\\theta}_i, \\hat{s}_{ij}, \\hat{\\tau})$",
        fc="#fdecea", ec="#b03a2e", lw=1.2, title_color="#922b21",
    )
    draw_box(
        35.0, 32.5, 30.0, 11.5,
        "6. Optional Distillation",
        "If $r_t=1$: distill $\\leq 1$ rule\nFormat & 15–32 tok admission",
        fc="#fef9e7", ec="#9a7d0a", title_color="#7d6608",
    )
    draw_box(
        1.0, 32.5, 30.5, 11.5,
        "7. Conditional Pruning",
        "If bank over budget: evict $\\arg\\min$\n$\\mathrm{Score}_{\\mathrm{prune}}(i)$ (Eq. 12–13) $\\to \\mathcal{M}_{t+1}$",
        fc="#fdecea", ec="#b03a2e", lw=1.15, title_color="#922b21",
    )

    # Sequential control arrows: 4 -> 5 -> 6 -> 7 -> 1
    draw_arrow((88.75, 49.0), (88.75, 44.0), color="#b03a2e")
    draw_arrow((68.5, 38.25), (65.0, 38.25), color="#b03a2e")
    draw_arrow((35.0, 38.25), (31.5, 38.25), color="#b03a2e")
    draw_arrow((11.25, 44.0), (11.25, 49.0), color="#b03a2e")
    ax.text(6.2, 46.5, "$\\mathcal{M}_{t+1}$", ha="center", va="center", fontsize=7.0, color="#922b21", fontweight="bold")

    # Clean feedback arrow from Box 5 (top-left) to Box 3 (bottom) without crossing any box
    draw_arrow((72.0, 44.0), (62.25, 49.0), color="#b03a2e", lw=1.0, ls="--")
    ax.text(60.5, 46.3, "Posterior $(\\hat{\\alpha}_c, \\hat{\\theta}_i, \\hat{s}_{ij}, \\hat{\\tau})$", ha="center", va="center", fontsize=6.6, color="#922b21", fontweight="bold")

    # --- LOWER DASHED INSETS (Mechanism Details I & II) ---
    inset_l = FancyBboxPatch((1.0, 1.2), 47.5, 28.0, boxstyle="round,pad=0.35", facecolor="#fbfdff", edgecolor="#1f618d", linewidth=0.95, linestyle="--", zorder=2)
    ax.add_patch(inset_l)
    ax.text(24.75, 27.1, "Detail I: Warm-Up Gate Scope (Eq. 9)", ha="center", va="center", fontsize=7.6, fontweight="bold", color="#154360")

    draw_box(
        2.5, 15.2, 44.5, 9.8,
        "Gated by $G_t(i,j) = \\mathbb{I}[N_{c_t}\\geq 5]\\mathbb{I}[n_i\\geq 2]\\mathbb{I}[n_j\\geq 2]$",
        "Controls cold-start pair UCB, second-chance pair UCB,\nand random pair exploration at $S_t=\\varnothing$ (Sec. 3.3)",
        fc="#e8f1fb", ec="#1f618d", title_color="#154360", fs_t=7.1, fs_s=6.5,
    )
    draw_box(
        2.5, 2.8, 44.5, 10.4,
        "Not Disabled by Warm-Up Gate",
        "Pairwise Laplace fit $\\hat{s}_{ij}$, Hard Conflict Guard (Eq. 10),\nand confirmation bonus (own eligibility rules in Sec. 3.3)",
        fc="#f4f6f7", ec="#555555", title_color="#222222", fs_t=7.1, fs_s=6.5,
    )

    inset_r = FancyBboxPatch((50.5, 1.2), 48.5, 28.0, boxstyle="round,pad=0.35", facecolor="#fffafa", edgecolor="#b03a2e", linewidth=0.95, linestyle="--", zorder=2)
    ax.add_patch(inset_r)
    ax.text(74.75, 27.1, "Detail II: Refit Columns & Readiness (Eq. 12–13)", ha="center", va="center", fontsize=7.6, fontweight="bold", color="#922b21")

    draw_box(
        52.0, 15.2, 45.5, 9.8,
        "Historical Item Column Retention in $\\mathbf{X}_t$",
        "Retains main-effect columns for all $i \\in \\mathcal{I}(\\mathcal{H}_t)$ in window $W$\n(incl. pruned rules) to reduce main-effect reassignment",
        fc="#fdecea", ec="#b03a2e", title_color="#922b21", fs_t=7.1, fs_s=6.5,
    )
    draw_box(
        52.0, 2.8, 45.5, 10.4,
        "Heuristic Readiness Guard $\\mathrm{Ready}(i)$",
        "For $1\\leq n_i\\leq 2$: checks $n_i^{\\mathrm{solo}}\\geq 1$, $(n_i\\geq 2 \\wedge |\\mathcal{P}_{\\mathrm{co}}(i)|\\geq 2)$,\nor solo-tested partner; limits negative pruning credit",
        fc="#f4f6f7", ec="#555555", title_color="#222222", fs_t=7.1, fs_s=6.5,
    )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "fig0_workflow_diagram.pdf")
        fig.savefig(out_dir / "fig0_workflow_diagram.png", dpi=260)
    plt.close(fig)


def main() -> None:
    out_dirs = [REPO_ROOT / "paper" / "figures", REPO_ROOT / "paper_artifacts" / "figures"]
    manifest_out = REPO_ROOT / "paper_artifacts" / "method_diagrams_manifest.json"
    export_fig0_credit_coupling(out_dirs, manifest_out)
    export_fig0_workflow_diagram(out_dirs)
    print("Exported native-width fig0_credit_coupling.pdf/.png, fig0_workflow_diagram.pdf/.png, and method_diagrams_manifest.json")


if __name__ == "__main__":
    main()
