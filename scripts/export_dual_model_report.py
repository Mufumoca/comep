"""Export 4-backbone (DeepSeek-V4.1-Flash, Qwen3-14B-AWQ, Qwen3-8B, Llama-3.1-8B-AWQ) LaTeX tables, vector figures, and comprehensive markdown reports."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

from scripts.export_paper_artifacts import (
    COLORS,
    METHOD_LABELS,
    plot_ablation_and_4cell,
    plot_learning_and_pareto,
    render_4cell_table,
    render_ablation_table,
    render_main_table,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ORDER = ["no-memory", "full-memory", "individual", "mmr", "additive", "expel-derived-online", "evomemguard", "comep"]


def load_stage3_summary(stage3_dir: Path) -> dict:
    summary = {}
    for env_k in ("alfworld", "scienceworld"):
        gp = stage3_dir / env_k / "grid_summary.json"
        if gp.is_file():
            summary[env_k] = json.loads(gp.read_text(encoding="utf-8"))
    if len(summary) < 2 and (stage3_dir / "stage3_main_summary.json").is_file():
        raw = json.loads((stage3_dir / "stage3_main_summary.json").read_text(encoding="utf-8"))
        for env_k in ("alfworld", "scienceworld"):
            if env_k not in summary and env_k in raw:
                summary[env_k] = raw[env_k]
    return summary


def compute_env_mean_score(env_dir: Path, m: str) -> float | None:
    ep_files = sorted(env_dir.glob(f"{m}_seed*/episodes.jsonl"))
    if not ep_files:
        return None
    scores = []
    for f in ep_files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if "score" in rec and rec["score"] is not None:
                    scores.append(float(rec["score"]))
    return float(np.mean(scores)) if scores else None


def compute_holm_map(raw_ps: dict[str, float]) -> dict[str, float]:
    sorted_ms = sorted(raw_ps.keys(), key=lambda x: raw_ps[x])
    holm_ps: dict[str, float] = {}
    running_max = 0.0
    m_tot = len(sorted_ms)
    for rank_idx, bm in enumerate(sorted_ms):
        adj = min(1.0, raw_ps[bm] * (m_tot - rank_idx))
        running_max = max(running_max, adj)
        holm_ps[bm] = running_max
    return holm_ps


def compute_pooled_row(grid_summary: dict, m: str, stage3_dir: Path | None = None) -> dict | None:
    if "alfworld" not in grid_summary or "scienceworld" not in grid_summary:
        return None
    alf_m = grid_summary["alfworld"]["methods"]
    sci_m = grid_summary["scienceworld"]["methods"]
    if m not in alf_m or m not in sci_m:
        return None
    a, s = alf_m[m], sci_m[m]
    n_alf = int(a["total_episodes"])
    n_sci = int(s["total_episodes"])
    n_tot = n_alf + n_sci
    w_alf = round(a["mean_sr"] * n_alf)
    w_sci = round(s["mean_sr"] * n_sci)
    pooled_sr = (w_alf + w_sci) / max(1, n_tot)
    macro_sr = 0.5 * (a["mean_sr"] + s["mean_sr"])
    macro_late = 0.5 * (a["mean_late_sr"] + s["mean_late_sr"])
    macro_tok = 0.5 * (a["mean_retrieval_tokens"] + s["mean_retrieval_tokens"])
    pooled_tok = (a["mean_retrieval_tokens"] * n_alf + s["mean_retrieval_tokens"] * n_sci) / max(1, n_tot)
    mgr = int(a["manager_llm_calls"]) + int(s["manager_llm_calls"])
    dist = int(a["distillation_llm_calls"]) + int(s["distillation_llm_calls"])
    tot_cost = float(a["total_cost_usd"]) + float(s["total_cost_usd"])

    p_val = 1.0
    b10 = 0
    b01 = 0
    alf_p = 1.0
    sci_p = 1.0
    alf_b10, alf_b01 = 0, 0
    sci_b10, sci_b01 = 0, 0
    if m != "comep":
        pa = grid_summary["alfworld"].get("paired_vs_comep", {}).get(f"comep_vs_{m}", {})
        ps = grid_summary["scienceworld"].get("paired_vs_comep", {}).get(f"comep_vs_{m}", {})
        alf_b10 = int(pa.get("comep_wins_discordant", 0))
        alf_b01 = int(pa.get("baseline_wins_discordant", 0))
        sci_b10 = int(ps.get("comep_wins_discordant", 0))
        sci_b01 = int(ps.get("baseline_wins_discordant", 0))
        if (alf_b10 + alf_b01) > 0:
            alf_p = float(binomtest(alf_b10, alf_b10 + alf_b01, 0.5).pvalue)
        if (sci_b10 + sci_b01) > 0:
            sci_p = float(binomtest(sci_b10, sci_b10 + sci_b01, 0.5).pvalue)
        b10 = alf_b10 + sci_b10
        b01 = alf_b01 + sci_b01
        if (b10 + b01) > 0:
            p_val = float(binomtest(b10, b10 + b01, 0.5).pvalue)

    sci_score = compute_env_mean_score(stage3_dir / "scienceworld", m) if stage3_dir is not None else None
    seed_pooled_srs = []
    if stage3_dir is not None:
        seeds = grid_summary["alfworld"].get("seeds", [0, 1, 2])
        for sd in seeds:
            a_sum_p = stage3_dir / "alfworld" / f"{m}_seed{sd}" / "summary.json"
            s_sum_p = stage3_dir / "scienceworld" / f"{m}_seed{sd}" / "summary.json"
            if a_sum_p.is_file() and s_sum_p.is_file():
                asum = json.loads(a_sum_p.read_text(encoding="utf-8"))
                ssum = json.loads(s_sum_p.read_text(encoding="utf-8"))
                na_s = int(asum["completed_episodes"])
                ns_s = int(ssum["completed_episodes"])
                wa_s = round(float(asum["success_rate"]) * na_s)
                ws_s = round(float(ssum["success_rate"]) * ns_s)
                seed_pooled_srs.append((wa_s + ws_s) / max(1, na_s + ns_s))
    pooled_std = float(np.std(seed_pooled_srs, ddof=1)) if len(seed_pooled_srs) > 1 else 0.0
    return {
        "alf_sr": a["mean_sr"],
        "alf_std": a["std_sr"],
        "alf_late": a["mean_late_sr"],
        "alf_tok": a["mean_retrieval_tokens"],
        "alf_seeds": int(a.get("seeds", 1)),
        "alf_n": n_alf,
        "alf_w": w_alf,
        "alf_p": alf_p,
        "alf_b10": alf_b10,
        "alf_b01": alf_b01,
        "sci_sr": s["mean_sr"],
        "sci_std": s["std_sr"],
        "sci_late": s["mean_late_sr"],
        "sci_tok": s["mean_retrieval_tokens"],
        "sci_seeds": int(s.get("seeds", 1)),
        "sci_n": n_sci,
        "sci_w": w_sci,
        "sci_p": sci_p,
        "sci_b10": sci_b10,
        "sci_b01": sci_b01,
        "sci_score": sci_score,
        "pooled_sr": pooled_sr,
        "pooled_std": pooled_std,
        "seed_pooled_srs": seed_pooled_srs,
        "macro_sr": macro_sr,
        "macro_late": macro_late,
        "macro_tok": macro_tok,
        "pooled_tok": pooled_tok,
        "mgr_calls": mgr,
        "dist_calls": dist,
        "tot_cost": tot_cost,
        "p_val": p_val,
        "b10": b10,
        "b01": b01,
        "n_tot": n_tot,
        "w_tot": w_alf + w_sci,
    }


def _fmt_sr_cell(sr: float, std: float, seeds: int, n_tot: int, is_best: bool) -> str:
    txt = f"{sr*100:.1f}% ± {std*100:.1f}%" if seeds > 1 else f"{sr*100:.1f}% (n={n_tot}, 1s)"
    return f"**{txt}**" if is_best else txt


def _smoothed_logit_contrast(cr: dict, n_cell: int) -> float:
    def _lgt(r: float) -> float:
        k = r * n_cell
        p = (k + 0.5) / (n_cell + 1.0)
        return math.log(p / (1.0 - p))

    return _lgt(cr["AB"]) - _lgt(cr["A"]) - _lgt(cr["B"]) + _lgt(cr["none"])


def compute_family_breakdown(env_dir: Path, methods: list[str]) -> dict[str, dict[str, tuple[int, int]]]:
    breakdown: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
    for m in methods:
        ep_files = sorted(env_dir.glob(f"{m}_seed*/episodes.jsonl"))
        if not ep_files:
            continue
        by_ctx: dict[str, list[int]] = defaultdict(list)
        for ep_file in ep_files:
            for line in ep_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    by_ctx[str(rec.get("context", ""))].append(int(rec.get("reward", 0)))
        for ctx, rews in by_ctx.items():
            breakdown[ctx][m] = (sum(rews), len(rews))
    return dict(sorted(breakdown.items()))


def render_dual_backbone_table(ds_summary: dict, qwen_summary: dict, out_path: Path):
    d_rows = {m: compute_pooled_row(ds_summary, m) for m in ORDER}
    q_rows = {m: compute_pooled_row(qwen_summary, m) for m in ORDER}
    valid = [m for m in ORDER if d_rows.get(m) and q_rows.get(m)]
    if not valid:
        return

    max_d_alf = max(round(d_rows[m]["alf_sr"] * 100, 1) for m in valid)
    max_d_sci = max(round(d_rows[m]["sci_sr"] * 100, 1) for m in valid)
    max_d_pool = max(round(d_rows[m]["pooled_sr"] * 100, 1) for m in valid)
    max_q_alf = max(round(q_rows[m]["alf_sr"] * 100, 1) for m in valid)
    max_q_sci = max(round(q_rows[m]["sci_sr"] * 100, 1) for m in valid)
    max_q_pool = max(round(q_rows[m]["pooled_sr"] * 100, 1) for m in valid)

    q_sample = q_rows[valid[0]]
    q_seed_lbl = "seed" if q_sample["alf_seeds"] == 1 else "seeds"
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{Online memory routing \\& pruning under unified CoMeP v2.1 across \\textbf{{DeepSeek-V4.1-Flash}} (90-token budget, $\\beta_{{\\text{{pair}}}}=0.25$, 3 seeds: $n=270$ ALFWorld / $225$ ScienceWorld) and \\textbf{{Qwen3-8B}} (90-token budget, $\\beta_{{\\text{{pair}}}}=0.25$, {q_sample['alf_seeds']} {q_seed_lbl}: $n={q_sample['alf_n']}$ ALFWorld / ${q_sample['sci_n']}$ ScienceWorld) on held-out \\textbf{{ALFWorld}} (\\texttt{{valid\\_unseen}}) and \\textbf{{ScienceWorld}} (\\texttt{{test}}). Column maxima are bolded; LLM calls report Manager Critic / Distillation calls separately for each model.}}",
        "\\label{tab:dual_backbone_results}",
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        "& \\multicolumn{4}{c}{\\textbf{DeepSeek-V4.1-Flash (CoMeP v2.1)}} & \\multicolumn{4}{c}{\\textbf{Qwen3-8B Open-Weights (CoMeP v2.1)}} \\\\",
        "\\cmidrule(lr){2-5} \\cmidrule(lr){6-9}",
        "Method & ALFWorld & ScienceWorld & Pooled SR (Macro Tok) & Mgr/Dist & ALFWorld & ScienceWorld & Pooled SR (Macro Tok) & Mgr/Dist \\\\",
        "\\midrule",
    ]
    for m in valid:
        dr, qr = d_rows[m], q_rows[m]
        d_alf = f"{dr['alf_sr']*100:.1f}\\%"
        d_sci = f"{dr['sci_sr']*100:.1f}\\%"
        d_pool = f"{dr['pooled_sr']*100:.1f}\\% ({dr['macro_tok']:.1f})"
        q_alf = f"{qr['alf_sr']*100:.1f}\\%"
        q_sci = f"{qr['sci_sr']*100:.1f}\\%"
        q_pool = f"{qr['pooled_sr']*100:.1f}\\% ({qr['macro_tok']:.1f})"
        if round(dr["alf_sr"] * 100, 1) == max_d_alf:
            d_alf = f"\\textbf{{{d_alf}}}"
        if round(dr["sci_sr"] * 100, 1) == max_d_sci:
            d_sci = f"\\textbf{{{d_sci}}}"
        if round(dr["pooled_sr"] * 100, 1) == max_d_pool:
            d_pool = f"\\textbf{{{d_pool}}}"
        if round(qr["alf_sr"] * 100, 1) == max_q_alf:
            q_alf = f"\\textbf{{{q_alf}}}"
        if round(qr["sci_sr"] * 100, 1) == max_q_sci:
            q_sci = f"\\textbf{{{q_sci}}}"
        if round(qr["pooled_sr"] * 100, 1) == max_q_pool:
            q_pool = f"\\textbf{{{q_pool}}}"
        d_calls = f"{dr['mgr_calls']} / {dr['dist_calls']}"
        q_calls = f"{qr['mgr_calls']} / {qr['dist_calls']}"
        lines.append(f"{METHOD_LABELS.get(m, m)} & {d_alf} & {d_sci} & {d_pool} & {d_calls} & {q_alf} & {q_sci} & {q_pool} & {q_calls} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_four_backbone_table(model_specs: list[tuple[str, str, dict, Path]], out_path: Path):
    """Render a 4-backbone Stage 3 LaTeX comparison table across all completed models."""
    rows_by_model: dict[str, dict[str, dict]] = {}
    active_models: list[tuple[str, str]] = []
    for key, label, summ, s3_dir in model_specs:
        m_map = {m: compute_pooled_row(summ, m, s3_dir) for m in ORDER}
        if any(m_map.values()):
            rows_by_model[key] = m_map
            active_models.append((key, label))
    if not active_models:
        return

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Four-backbone evaluation of \\textbf{CoMeP v2.1} across API and open-weight models on held-out \\textbf{ALFWorld} (\\texttt{valid\\_unseen}) and \\textbf{ScienceWorld} (\\texttt{test}), 3 seeds per model. Each cell reports ALFWorld SR / ScienceWorld SR / Combined Pooled SR (Macro Retrieved Tokens). $^{\\star}$ indicates 7-comparison Holm-adjusted paired McNemar $p_{\\text{Holm}} < 0.05$ in favor of \\textbf{CoMeP} on the combined pooled benchmark.}",
        "\\label{tab:four_backbone_results}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l" + "ccc" * len(active_models) + "}",
        "\\toprule",
    ]
    header_top = [""]
    cmid = []
    header_sub = ["Method"]
    for idx, (key, label) in enumerate(active_models):
        c_start = 2 + idx * 3
        c_end = c_start + 2
        header_top.append(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{label}}}}}")
        cmid.append(f"\\cmidrule(lr){{{c_start}-{c_end}}}")
        header_sub.extend(["ALF (\\%)", "SciW (\\%)", "Pooled (Tok)"])
    lines.append(" & ".join(header_top) + " \\\\")
    lines.append(" ".join(cmid))
    lines.append(" & ".join(header_sub) + " \\\\")
    lines.append("\\midrule")

    holm_by_model: dict[str, dict[str, float]] = {}
    max_by_model: dict[str, tuple[float, float, float]] = {}
    for key, _ in active_models:
        m_map = rows_by_model[key]
        valid_ms = [m for m in ORDER if m_map.get(m)]
        max_a = max((round(m_map[m]["alf_sr"] * 100, 1) for m in valid_ms), default=0.0)
        max_s = max((round(m_map[m]["sci_sr"] * 100, 1) for m in valid_ms), default=0.0)
        max_p = max((round(m_map[m]["pooled_sr"] * 100, 1) for m in valid_ms), default=0.0)
        max_by_model[key] = (max_a, max_s, max_p)
        raw_ps = {m: m_map[m]["p_val"] for m in valid_ms if m != "comep"}
        holm_by_model[key] = compute_holm_map(raw_ps)

    for m in ORDER:
        row_cells = [METHOD_LABELS.get(m, m)]
        for key, _ in active_models:
            r = rows_by_model[key].get(m)
            if not r:
                row_cells.extend(["---", "---", "---"])
                continue
            max_a, max_s, max_p = max_by_model[key]
            a_s = f"{r['alf_sr']*100:.1f}"
            s_s = f"{r['sci_sr']*100:.1f}"
            star = "$^{\\star}$" if (m != "comep" and holm_by_model[key].get(m, 1.0) < 0.05 and r["b10"] > r["b01"]) else ""
            p_s = f"{r['pooled_sr']*100:.1f}{star} ({r['macro_tok']:.0f})"
            if round(r["alf_sr"] * 100, 1) == max_a:
                a_s = f"\\textbf{{{a_s}}}"
            if round(r["sci_sr"] * 100, 1) == max_s:
                s_s = f"\\textbf{{{s_s}}}"
            if round(r["pooled_sr"] * 100, 1) == max_p:
                p_s = f"\\textbf{{{p_s}}}"
            row_cells.extend([a_s, s_s, p_s])
        lines.append(" & ".join(row_cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_backbone_bar_grid(active: list[tuple[str, str, dict, Path]], out_pdf: Path, out_png: Path):
    short_labels = ["No-Mem", "Full-Mem", "Indiv.", "MMR-style", "Additive", "ExpeL-der.", "EvoMemG.-st.", "CoMeP"]
    if not active:
        return
    n_m = len(active)
    ncols = 2
    nrows = math.ceil(n_m / 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.8, 4.5 * nrows), squeeze=False)

    for idx, (key, label, summ, sdir) in enumerate(active):
        ax = axes[idx // ncols, idx % ncols]
        vals = []
        stds = []
        seed_pts = []
        toks = []
        cols = []
        for m in ORDER:
            r = compute_pooled_row(summ, m, sdir)
            vals.append((r["pooled_sr"] * 100.0) if r else 0.0)
            stds.append((r.get("pooled_std", 0.0) * 100.0) if r else 0.0)
            seed_pts.append([sp * 100.0 for sp in r.get("seed_pooled_srs", [])] if r else [])
            toks.append(r["macro_tok"] if r else 0.0)
            cols.append(COLORS.get(m, "#333333"))
        x = np.arange(len(ORDER))
        bars = ax.bar(
            x,
            vals,
            yerr=stds,
            capsize=3.2,
            error_kw={"ecolor": "#222222", "elinewidth": 1.0, "capthick": 1.0},
            color=cols,
            alpha=0.86,
            width=0.62,
            zorder=3,
        )
        for xi, pts in zip(x, seed_pts):
            if pts:
                offsets = np.linspace(-0.10, 0.10, len(pts)) if len(pts) > 1 else [0.0]
                ax.scatter(
                    [xi + off for off in offsets],
                    pts,
                    s=16,
                    color="#111111",
                    alpha=0.65,
                    zorder=4,
                )
        max_v = max(vals, default=0.0)
        max_top = max((v + sd for v, sd in zip(vals, stds)), default=10.0)
        y_pad = max(0.35, max_top * 0.025)
        for bar, v, sd, t in zip(bars, vals, stds, toks):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                v + sd + y_pad,
                f"{v:.1f}±{sd:.1f}%\n({t:.0f}t)",
                ha="center",
                va="bottom",
                fontsize=7.2,
                fontweight="bold" if abs(v - max_v) < 0.06 else "normal",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=22, ha="right", fontsize=8.4)
        ax.set_ylim(0, max_top * 1.28)
        ax.set_ylabel("Pooled Task Success Rate (%)")
        ax.set_title(f"{label} (Held-Out Pooled SR ± 3-Seed SD)", fontsize=10.8, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)

    for idx in range(n_m, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")

    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_four_backbone_comparison(model_specs: list[tuple[str, str, dict, Path]], fig_dir: Path):
    active = [(k, lbl, summ, sdir) for k, lbl, summ, sdir in model_specs if "alfworld" in summ and "scienceworld" in summ]
    dual = [item for item in active if item[0] in ("deepseek_v41_flash", "qwen3_8b")]
    if dual:
        _render_backbone_bar_grid(dual, fig_dir / "fig4_dual_backbone_comparison.pdf", fig_dir / "fig4_dual_backbone_comparison.png")
    if active:
        _render_backbone_bar_grid(active, fig_dir / "fig4_four_backbone_comparison.pdf", fig_dir / "fig4_four_backbone_comparison.png")


def compute_4cell_overlap_audit(stage3_dir: Path, stage4_4cell_dir: Path) -> list[dict]:
    """Compute discovery-to-evaluation task overlap between comep_seed0 and Stage 4B 4-cell tasks."""
    rows = []
    if not stage4_4cell_dir.is_file() and not stage4_4cell_dir.is_dir():
        return rows
    for env_k in ("alfworld", "scienceworld"):
        seed0_ep = stage3_dir / env_k / "comep_seed0" / "episodes.jsonl"
        seed0_ids: set[str] = set()
        if seed0_ep.is_file():
            for ln in seed0_ep.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    seed0_ids.add(str(json.loads(ln).get("task_id", "")))
        all_seed_ids: set[str] = set()
        for ep_f in sorted((stage3_dir / env_k).glob("comep_seed*/episodes.jsonl")):
            for ln in ep_f.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    all_seed_ids.add(str(json.loads(ln).get("task_id", "")))
        env_4c = stage4_4cell_dir / env_k
        if not env_4c.is_dir():
            continue
        for pair_dir in sorted(env_4c.glob("pair_*")):
            none_ep = pair_dir / "cell_none" / "episodes.jsonl"
            if not none_ep.is_file():
                continue
            eval_ids = [
                str(json.loads(ln).get("task_id", ""))
                for ln in none_ep.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            ov0 = sum(1 for tid in eval_ids if tid in seed0_ids)
            ov_all = sum(1 for tid in eval_ids if tid in all_seed_ids)
            rows.append({
                "env": env_k,
                "pair_dir": pair_dir.name,
                "n_tasks": len(eval_ids),
                "overlap_seed0": ov0,
                "overlap_all_seeds": ov_all,
            })
    return rows


def _fmt_p(p_val: float, bold: bool = False) -> str:
    txt = f"{p_val:.4f}" if p_val >= 0.0001 else "<1e-4"
    return f"**{txt}**" if bold else txt


def render_four_model_markdown_report(
    model_entries: list[dict],
    ds_abl: dict | None,
    out_md_path: Path,
):
    lines = [
        "# CoMeP v2.1 四模型完整实验报告：组合信用归因与协同感知记忆剪枝",
        "",
        "## 1. 评测总览、冻结协议与方法论披露（4 Model Configurations × 2 Benchmarks × 8 Methods）",
        "",
        "本报告汇总 **CoMeP v2.1 (`Combinatorial Credit Assignment & Synergy-Aware Memory Pruning`)** 在 **3 个模型家族、4 个不同规模与量化精度配置的基础模型**上的完整留出集评测结果（Stage 3 主实验共 `11,880` 局，Stage 4A 六变体消融 `1,200` 局，Stage 4B 四格反事实干预共 `520` 次 rollout）：",
        "1. **`DeepSeek-V4.1-Flash`**（`opencode-go/deepseek-v4.1-flash`，`thinking=\"disabled\"`，Stage 3: ALFWorld `90×3=270` 局 + ScienceWorld `75×3=225` 局 = `495` 局/方法，总计 `3,960` 局；Stage 4A 六变体消融 `50×2×2×6=1,200` 局；Stage 4B 四格反事实干预 `22` 个独立任务 × 4 格 = `88` 次 rollout）。",
        "2. **`Qwen/Qwen3-14B-AWQ`**（Colab L4 `vLLM` 4-bit AWQ，`thinking=\"disabled\"`，Stage 2 `dev` `392` 局；Stage 3: ALFWorld `60×3=180` 局 + ScienceWorld `50×3=150` 局 = `330` 局/方法，总计 `2,640` 局；Stage 4B 四格反事实干预 `30` 个独立任务、共 `160` 次 rollout）。",
        "3. **`Qwen/Qwen3-8B`**（Colab L4 `vLLM` BF16，`thinking=\"disabled\"`，Stage 2 `dev` `392` 局；Stage 3: ALFWorld `60×3=180` 局 + ScienceWorld `50×3=150` 局 = `330` 局/方法，总计 `2,640` 局；Stage 4B 四格反事实干预 `40` 个独立任务 × 4 格 = `160` 次 rollout）。",
        "4. **`hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`**（Colab L4 `vLLM` 4-bit AWQ，`max_output_tokens=128`，Stage 2 `dev` `392` 局；Stage 3: ALFWorld `60×3=180` 局 + ScienceWorld `50×3=150` 局 = `330` 局/方法，总计 `2,640` 局；Stage 4B 四格反事实干预 `28` 个独立任务 × 4 格 = `112` 次 rollout）。",
        "",
        "### 1.1 算法与评测边界披露（Audit Disclosures）",
        "- **跨任务划分零泄漏（Zero Train/Dev/Test Leakage）**：ALFWorld（`train=36`, `dev=24`, `test_ood=134`）与 ScienceWorld（`train=30`, `dev=25`, `test=100`）三划分任务 ID 完全互斥（重叠为 `0`）。两个环境的初始热启动记忆池（`configs/alfworld_warmstart_pool.json` 与 `configs/scienceworld_warmstart_pool.json`）各含 **18 条**仅由 `train` 集轨迹提炼的通用规则。",
        "- **在线终身评测与统计相关结构说明**：Stage 3 评测为测试流内的在线序列评测（记忆权重与在线蒸馏随已完成的测试集成败反馈更新，方法间与环境间严格隔离）。表内 $p_{\\text{raw}} / p_{\\text{Holm}}$ 来自同一随机种子与任务位置上的双侧精确配对 McNemar 检验及 7 项基线对比的 Holm-Bonferroni 校正。需注意：所报 $p$ 值按逐 episode 配对二元结果计算；在线轨迹内共享记忆状态造成的序列依赖及跨 seed 重复抽样有限留出集任务可能影响其校准，因此作为探索性比较指标，不保证在该依赖结构下仍具有名义显著性水平。Holm 校正仅调整多重比较，不解决基础 $p$ 值的依赖问题。",
        "- **模型规模与量化精度正交边界说明**：4 组模型覆盖 3 个模型家族；其中 `Qwen3-8B` 为 BF16 精度，而 `Qwen3-14B-AWQ` 与 `Llama-3.1-8B-Instruct-AWQ-INT4` 为 4-bit AWQ 量化（且 Llama 设 `max_output_tokens=128` 以抑制非推理模型长输出跑偏）。因此跨 `Qwen3-8B` 与 `Qwen3-14B-AWQ` 的对比同时包含参数规模与量化精度的复合影响。",
        "- **ScienceWorld `Score` 指标定义**：表内 `Score` 为环境终局原始分数的算术均值（含失败局 `-100` 终局惩罚），作为探索性辅助参考；主评估指标为二元任务成功率 `SR` 与后半程成功率 `LateSR`。",
        "- **基线适配与运行期审计透明披露**：",
        "  - `full-memory (Top-k)`：保留不剪枝的全量记忆库（18 条初始热启动规则 + 在线蒸馏规则；在 `DeepSeek-V4.1-Flash` ALFWorld `seed=0` 末期达 89 条 / 2,322 tokens，ScienceWorld `seed=0` 达 43 条，`Qwen3-14B-AWQ` ALFWorld 达 47 条），每集按查询相关性在 `90` token 预算内执行贪心 Top-$k$ 检索（避免全库直接拼接超出 90-token 预算）。",
        "  - `expel-derived-online`：在线适配版 ExpeL（Zhao et al., 2024），关闭逐集启发式蒸馏（`distill=False`）；在每次调用时优先选择同任务的正负对比轨迹对，否则当待处理池中存在至少 2 条成功轨迹时选择其中 2 条组成 `success_batch`（输入为完整轨迹 JSON，超过 `20,000` 字节则跳过记为 `skipped_oversize`；输出为逐行 `ADD/EDIT/REMOVE/AGREE` 操作文本）。运行期 `critic.jsonl` 精确审计如下：在 `DeepSeek-V4.1-Flash` Stage 3 中共触发 `116` 次实际 LLM 反思调用（ALFWorld `114` 次调用：`113` 次成功应用操作批次 `applied`、`1` 次 `skipped_truncated`；ScienceWorld `2` 次调用：`2` 次 `applied`，另有 `44` 次因超过 20KB 记为 `skipped_oversize` 未发起调用）；在 `Qwen3-14B-AWQ` 中触发 `59` 次 ALFWorld 调用（`53` 次 `applied`、`6` 次 `skipped_truncated`）与 `0` 次 ScienceWorld 调用（`3` 次 `skipped_oversize`）；在 `Qwen3-8B` 中触发 `38` 次 ALFWorld 调用（`38` 次 `applied`、`0` 次 `skipped_truncated`）与 `0` 次 ScienceWorld 调用；在 `Llama-3.1-8B-AWQ` 中触发 `8` 次 ALFWorld 调用，均因 `max_output_tokens=128` 导致操作文本输出被截断而记为 `skipped_truncated`（`0` 次 `applied`）。",
        "  - `evomemguard`：架构对齐的 NLI 冲突图 + 加性平滑经验效用 $(s + 1.5)/(n + 2.5)$ 信用归因与剪枝基线（Wang et al., 2026）。",
        "",
        "---",
        "",
        "## 2. 四模型 Stage 3 留出集核心汇总对比（ALFWorld OOD + ScienceWorld `test`，3 Seeds）",
        "",
        "| 模型 (Backbone) | 方法 (Method) | ALFWorld OOD SR (`LateSR`) | ScienceWorld `test` SR (`LateSR` / `Score`) | 跨环境 Pooled SR (`Wins/N`) | 跨环境 Macro SR (`Macro LateSR`) | 检索 Token (Pool / Macro) | Mgr / Dist Calls | Pooled Raw / Holm $p$ vs `comep` |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for entry in model_entries:
        m_name = entry["label"]
        s3 = entry["s3_summary"]
        s3_dir = entry["s3_dir"]
        if "alfworld" not in s3 or "scienceworld" not in s3:
            continue
        rows = {m: compute_pooled_row(s3, m, s3_dir) for m in ORDER}
        valid = [m for m in ORDER if rows.get(m)]
        max_alf = max((round(rows[m]["alf_sr"] * 100, 1) for m in valid), default=0.0)
        max_sci = max((round(rows[m]["sci_sr"] * 100, 1) for m in valid), default=0.0)
        max_pool = max((round(rows[m]["pooled_sr"] * 100, 1) for m in valid), default=0.0)
        max_macro = max((round(rows[m]["macro_sr"] * 100, 1) for m in valid), default=0.0)
        holm_pool = compute_holm_map({m: rows[m]["p_val"] for m in valid if m != "comep"})

        for m in valid:
            r = rows[m]
            alf_txt = f"{r['alf_sr']*100:.1f}% ± {r['alf_std']*100:.1f}% ({r['alf_late']*100:.1f}%)"
            if round(r["alf_sr"] * 100, 1) == max_alf:
                alf_txt = f"**{alf_txt}**"
            sc_part = f" / {r['sci_score']:+.1f}" if r["sci_score"] is not None else ""
            sci_txt = f"{r['sci_sr']*100:.1f}% ± {r['sci_std']*100:.1f}% ({r['sci_late']*100:.1f}%{sc_part})"
            if round(r["sci_sr"] * 100, 1) == max_sci:
                sci_txt = f"**{sci_txt}**"
            pool_txt = f"{r['pooled_sr']*100:.1f}% ({r['w_tot']}/{r['n_tot']})"
            if round(r["pooled_sr"] * 100, 1) == max_pool:
                pool_txt = f"**{pool_txt}**"
            macro_txt = f"{r['macro_sr']*100:.1f}% ({r['macro_late']*100:.1f}%)"
            if round(r["macro_sr"] * 100, 1) == max_macro:
                macro_txt = f"**{macro_txt}**"
            tok_txt = f"{r['pooled_tok']:.1f} / {r['macro_tok']:.1f}"
            calls_txt = f"{r['mgr_calls']} / {r['dist_calls']}"
            if m == "comep":
                p_txt = "---"
            else:
                hp = holm_pool[m]
                p_str = f"{_fmt_p(r['p_val'])} / {_fmt_p(hp, bold=(hp < 0.05))}"
                p_txt = f"{p_str} (`{r['b10']}:{r['b01']}`)"
            lines.append(
                f"| `{m_name}` | `{m}` | {alf_txt} | {sci_txt} | {pool_txt} | {macro_txt} | {tok_txt} | {calls_txt} | {p_txt} |"
            )

    lines.extend([
        "",
        "> **分环境统计显著性补充说明**：",
        "> - **`DeepSeek-V4.1-Flash`**：ALFWorld 单环境上 `comep` (`88.9%`) 显著优于 `no-memory` ($p_{\\text{Holm}}<10^{-4}$)、`mmr` ($p_{\\text{Holm}}<10^{-4}$)、`individual` ($p_{\\text{Holm}}=0.0040$)、`evomemguard` ($p_{\\text{Holm}}=0.0086$)、`full-memory` ($p_{\\text{Holm}}=0.0157$)，与 `additive` (`88.9%`, $p=1.0000$) 并列第一；ScienceWorld 单环境上 `comep` (`57.8%`, `LateSR=66.7%` 居首) 显著优于 `expel-derived-online` ($p_{\\text{Holm}}=0.0001$)、`mmr` ($p_{\\text{Holm}}=0.0005$)、`full-memory` ($p_{\\text{Holm}}=0.0029$)。",
        "> - **`Qwen3-14B-AWQ`**：ALFWorld 单环境上 `comep` (`59.4%`, `LateSR=62.2%`) 显著优于 `no-memory` (`46.1%`, discordant `45:21`, $p_{\\text{raw}}=0.0043, p_{\\text{Holm}}=0.0299$)；跨环境 `Macro LateSR` 达 **`35.8%`**（8 方法第 1），Pooled SR (`36.7%`) 显著优于 `no-memory` (`28.5%`, discordant `55:28`, $p_{\\text{raw}}=0.0040, p_{\\text{Holm}}=0.0283$)，与 `expel-derived-online` (`38.8%`)、Full-Memory/MMR/Additive 等基线的配对差异均无统计显著性 ($p_{\\text{Holm}}=1.0000$)。",
        "> - **`Qwen3-8B`**：`comep` 在 ScienceWorld (`12.7%`) 与跨环境 Pooled SR (`31.2%`)、Macro SR (`29.7%`) 上均居第 1，Pooled 显著优于 `no-memory` ($p_{\\text{Holm}}=0.0075$)、`evomemguard` ($p_{\\text{Holm}}=0.0075$) 与 `individual` ($p_{\\text{Holm}}=0.0261$)。",
        "> - **`Llama-3.1-8B-AWQ`**：在 ScienceWorld 上 `comep` (`6.0% ± 3.5%`, `LateSR=8.0%`) 居第 1 并显著优于 `no-memory` (`0.0%`, discordant `9:0`, $p_{\\text{raw}}=0.0039, p_{\\text{Holm}}=0.0273$)；而在 ALFWorld 上该配置下 `comep` 的成功率较低（`3.9%` vs. `expel-derived-online` `10.0%`、`full-memory` `8.9%`，单环境与合并后 $p_{\\text{Holm}}$ 均无显著差异；其是否源于 4-bit 8B 模型极低基础胜率与高频负回报下的剪枝敏感性尚属待验证假设）。",
    ])

    # Stage 4A Ablations
    if ds_abl and "alfworld" in ds_abl and "scienceworld" in ds_abl:
        lines.extend([
            "",
            "---",
            "",
            "## 3. Stage 4A：`DeepSeek-V4.1-Flash` CoMeP v2.1 六变体消融实验 (`outputs/stage4-ablations-v21`)",
            "",
            "| 变体 (Ablation Variant) | ALFWorld OOD SR (`LateSR`) | ScienceWorld `test` SR (`LateSR`) | 跨环境 Macro SR (`Macro LateSR`) | 检索 Token (ALF / SciW / Macro) | Pooled Discordant (`comep:var`, Raw / Holm $p$) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ])
        abl_order = [
            "comep",
            "comep-no-hard-conflicts",
            "comep-no-pair-optimism",
            "comep-no-early-stop",
            "comep-no-coalition-pruning",
            "comep-sgd-v1",
        ]
        abl_raw_ps: dict[str, float] = {}
        abl_disc: dict[str, tuple[int, int]] = {}
        for m in abl_order:
            if m == "comep" or m not in ds_abl["alfworld"]["methods"] or m not in ds_abl["scienceworld"]["methods"]:
                continue
            pa = ds_abl["alfworld"].get("paired_vs_comep", {}).get(f"comep_vs_{m}", {})
            ps = ds_abl["scienceworld"].get("paired_vs_comep", {}).get(f"comep_vs_{m}", {})
            b10 = int(pa.get("comep_wins_discordant", 0)) + int(ps.get("comep_wins_discordant", 0))
            b01 = int(pa.get("baseline_wins_discordant", 0)) + int(ps.get("baseline_wins_discordant", 0))
            abl_disc[m] = (b10, b01)
            abl_raw_ps[m] = float(binomtest(b10, b10 + b01, 0.5).pvalue) if (b10 + b01) > 0 else 1.0
        abl_holm_ps = compute_holm_map(abl_raw_ps)

        for m in abl_order:
            if m not in ds_abl["alfworld"]["methods"] or m not in ds_abl["scienceworld"]["methods"]:
                continue
            ra = ds_abl["alfworld"]["methods"][m]
            rs = ds_abl["scienceworld"]["methods"][m]
            mac_sr = 0.5 * (ra["mean_sr"] + rs["mean_sr"])
            mac_late = 0.5 * (ra["mean_late_sr"] + rs["mean_late_sr"])
            mac_tok = 0.5 * (ra["mean_retrieval_tokens"] + rs["mean_retrieval_tokens"])
            if m == "comep":
                p_col = "---"
            else:
                b10, b01 = abl_disc[m]
                rp, hp = abl_raw_ps[m], abl_holm_ps[m]
                p_col = f"`{b10}:{b01}` ({_fmt_p(rp)} / {_fmt_p(hp, bold=(hp < 0.05))})"
            lines.append(
                f"| `{m}` | {ra['mean_sr']*100:.1f}% ± {ra['std_sr']*100:.1f}% ({ra['mean_late_sr']*100:.1f}%) | "
                f"{rs['mean_sr']*100:.1f}% ± {rs['std_sr']*100:.1f}% ({rs['mean_late_sr']*100:.1f}%) | "
                f"**{mac_sr*100:.1f}%** ({mac_late*100:.1f}%) | "
                f"{ra['mean_retrieval_tokens']:.1f} / {rs['mean_retrieval_tokens']:.1f} / **{mac_tok:.1f}** | {p_col} |"
            )
        lines.extend([
            "",
            "> **Stage 4A 消融诊断说明**：Stage 4A 采用短序列预算（每环境 `50` 局 × `2` 种子 = `100` 局/变体，跨环境合并 `200` 局/变体）。在各单环境内部（5 项对比 Holm 校正后），各变体与完整 `comep` 均未达统计显著差异（ALFWorld 单环境最小为 `comep-no-coalition-pruning` 的 $p_{\\text{raw}}=0.0703, p_{\\text{Holm}}=0.3516$ 及 `comep-no-early-stop` 的 $p_{\\text{raw}}=0.2101, p_{\\text{Holm}}=0.8405$，其余 $p_{\\text{Holm}}=1.0000$；ScienceWorld 单环境最小为 `comep-no-pair-optimism` 的 $p_{\\text{raw}}=0.0192, p_{\\text{Holm}}=0.0961$，以及 `comep-no-hard-conflicts` 和 `comep-sgd-v1` 的 $p_{\\text{raw}}=0.0347, p_{\\text{Holm}}=0.1388$）。当跨两环境探索性合并（`200` 局）时，`comep` 相对 `comep-no-pair-optimism` 的不一致对为 `7:22`（$p_{\\text{raw}}=0.0081, p_{\\text{Holm}}=0.0407$，方向为去除 pair optimism 更高）：在 50 局短序列实验中，去除 pair optimism scale 的点估计更高，探索性合并检验也呈同方向；该现象与短期探索代价假设相容，但尚不能区分探索开销、任务构成、轨迹随机性及其他机制，是否存在长程补偿收益有待匹配任务与种子的长序列对照实验验证。此外，移除早停（`comep-no-early-stop`）使 Macro 检索开销由 `72.1` 升至 `80.1` tokens。",
        ])

    # Stage 4B 4-Cell Counterfactual Check across all 4 models
    lines.extend([
        "",
        "---",
        "",
        "## 4. Stage 4B：四模型 $2\\times 2$ `{none, A, B, AB}` 反事实干预检验与零重叠审计",
        "",
        "| 模型 (Backbone) | 环境 (`target_context`) | 候选类型 (`selection_source`) | $n$/格 (与 `seed=0` 重叠) | 在线 $\\hat{s}_{AB}$ | $\\varnothing$ | $A$ | $B$ | $AB$ | 线性 $\\hat{\\Delta}_{AB}$ [95% CI] | $\\hat{\\Delta}_{\\text{logit}}$ / $\\hat{\\Delta}_{\\text{score}}$ [95% CI] |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for entry in model_entries:
        m_name = entry["label"]
        c_dict = entry.get("s4_4cell")
        if not c_dict:
            continue
        ov_rows = compute_4cell_overlap_audit(entry["s3_dir"], entry["s4_4cell_dir"])
        ov_map = {(r["env"], r["pair_dir"]): r for r in ov_rows}
        for env_k, items in c_dict.items():
            for idx, rep in enumerate(items):
                cr = rep["cell_rates"]
                ci = rep["ci95_delta_AB"]
                n_c = int(rep.get("episodes_per_cell", 10))
                src = rep.get("selection_source", "posterior_s_ij")
                s_val = f"`{rep['learned_s_ij']:+.2f}`" if rep.get("learned_s_ij") is not None else "`NA`"
                lgt_d = _smoothed_logit_contrast(cr, n_c)
                zero_var_ci = abs(float(ci[0])) < 1e-12 and abs(float(ci[1])) < 1e-12
                ci_str = f"[{ci[0]*100:+.1f}, {ci[1]*100:+.1f}]*" if zero_var_ci else f"[{ci[0]*100:+.1f}, {ci[1]*100:+.1f}]"
                sc_d = rep.get("score_interaction_delta_AB")
                sc_ci = rep.get("ci95_score_delta_AB")
                if sc_d is not None and env_k == "scienceworld":
                    sc_ci_s = f" [{sc_ci[0]:+.1f}, {sc_ci[1]:+.1f}]" if sc_ci else ""
                    lgt_sc_str = f"`{lgt_d:+.2f}` / `{sc_d:+.1f}pt`{sc_ci_s}"
                else:
                    lgt_sc_str = f"`{lgt_d:+.2f}`"
                pair_dir_name = f"pair_{idx}_{rep['pair_kind']}"
                ov_info = ov_map.get((env_k, pair_dir_name))
                ov_str = f"{n_c} (`{ov_info['overlap_seed0']}/{ov_info['n_tasks']}`)" if ov_info else f"{n_c}"
                lines.append(
                    f"| `{m_name}` | `{env_k}` (`{rep.get('target_context', '')}`) | `{rep['pair_kind']}` (`{src}`) | {ov_str} | {s_val} | "
                    f"{cr['none']*100:.1f}% | {cr['A']*100:.1f}% | {cr['B']*100:.1f}% | {cr['AB']*100:.1f}% | "
                    f"**{rep['causal_interaction_delta_AB']*100:+.1f}pp** {ci_str} | {lgt_sc_str} |"
                )

    lines.extend([
        "",
        "> **Stage 4B 审计与统计注记**：",
        "> 1. **零重叠反事实隔离**：全部 4 个模型共 15 组候选规则对的评估任务均严格从 `comep_seed0` 未访问过的留出集尾部 (`disjoint_tail`) 中按 `target_context` 抽样，与发现轨迹 `comep_seed0` 的任务重叠率均为 **`0%`**（`DeepSeek-V4.1-Flash` 共 `22` 个独立任务 / `88` 次 rollout；`Qwen3-14B-AWQ` 共 `30` 个独立任务 / `160` 次 rollout，其中 ScienceWorld 正负候选对共享 `inclined-plane-determine-angle` 的 10 个未见变体；`Qwen3-8B` 共 `40` 个独立任务 / `160` 次 rollout；`Llama-3.1-8B-AWQ` 共 `28` 个独立任务 / `112` 次 rollout，因 `comep_seed0` 在 ScienceWorld 上无 $\\hat{s}_{ij}>0.08$ 的正协同对故仅评测 1 组负冲突对）。",
        "> 2. **候选提取阈值说明**：`extract_top_discovered_pairs` 对负冲突对按三级优先级提取：① 强后验负冲突（$\\hat{s}_{ij} < -0.08$）；② 在线已被协同剪枝归档的低胜率规则对（`archived_co_win`）；③ 弱后验负冲突回退阈值（$-0.08 \\le \\hat{s}_{ij} < -0.02$，仅在无前两类候选时触发，如 `DeepSeek-V4.1-Flash` ALFWorld `two_obj` 的 $\\hat{s}_{AB}=-0.0526$）。",
        "> 3. **退化自助法置信区间 (`*`) 说明**：带 `*` 的 `[+0.0, +0.0]*` 表示在该小样本格子内，逐任务二元交互项 $\\Delta_t = r_{AB,t} - r_{A,t} - r_{B,t} + r_{\\varnothing,t}$ 在所有被测任务上恒为 `0`（例如四格二元成功率全为 `0%`，或每个任务在四格上的对比完全抵消），导致经验任务配对 Bootstrap 样本方差为 `0`；这仅反映小样本二元观测的离散退化，不代表总体交互效应严格为零（故同步报告平滑对数几率对比 $\\hat{\\Delta}_{\\text{logit}}$ 与 ScienceWorld 连续分数交互项 $\\hat{\\Delta}_{\\text{score}}$）。",
    ])

    out_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepseek-stage3", default=str(REPO_ROOT / "outputs/stage3-main"))
    parser.add_argument("--deepseek-ablations", default=str(REPO_ROOT / "outputs/stage4-ablations-v21"))
    parser.add_argument("--deepseek-4cell", default=str(REPO_ROOT / "outputs/stage4-4cell-v21"))
    parser.add_argument("--qwen14b-stage3", default=str(REPO_ROOT / "outputs/stage3-qwen3-14b"))
    parser.add_argument("--qwen14b-4cell", default=str(REPO_ROOT / "outputs/stage4-4cell-qwen3-14b"))
    parser.add_argument("--qwen-stage2-dev", default=str(REPO_ROOT / "outputs/qwen3-8b-stage2-dev"))
    parser.add_argument("--qwen-stage3", default=str(REPO_ROOT / "outputs/stage3-qwen3-8b"))
    parser.add_argument("--qwen-4cell", default=str(REPO_ROOT / "outputs/stage4-4cell-qwen3-8b"))
    parser.add_argument("--llama-stage3", default=str(REPO_ROOT / "outputs/stage3-llama31-8b"))
    parser.add_argument("--llama-4cell", default=str(REPO_ROOT / "outputs/stage4-4cell-llama31-8b"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "paper_artifacts"))
    parser.add_argument("--report-out", default=str(REPO_ROOT / "DUAL_MODEL_FINAL_REPORT_ZH_20261105.md"))
    parser.add_argument("--four-model-report-out", default=str(REPO_ROOT / "FOUR_MODEL_FINAL_REPORT_ZH_20261107.md"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    tab_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    tab_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    ds_s3 = load_stage3_summary(Path(args.deepseek_stage3))
    q14_s3 = load_stage3_summary(Path(args.qwen14b_stage3))
    q8_s3 = load_stage3_summary(Path(args.qwen_stage3))
    ll_s3 = load_stage3_summary(Path(args.llama_stage3))

    abl_path = Path(args.deepseek_ablations) / "stage4_ablations_summary.json"
    ds_abl = json.loads(abl_path.read_text(encoding="utf-8")) if abl_path.is_file() else None

    ds_4c_dir = Path(args.deepseek_4cell)
    ds_4c = json.loads((ds_4c_dir / "stage4_4cell_summary.json").read_text(encoding="utf-8")) if (ds_4c_dir / "stage4_4cell_summary.json").is_file() else None

    q14_4c_dir = Path(args.qwen14b_4cell)
    q14_4c = json.loads((q14_4c_dir / "stage4_4cell_summary.json").read_text(encoding="utf-8")) if (q14_4c_dir / "stage4_4cell_summary.json").is_file() else None

    q8_4c_dir = Path(args.qwen_4cell)
    q8_4c = json.loads((q8_4c_dir / "stage4_4cell_summary.json").read_text(encoding="utf-8")) if (q8_4c_dir / "stage4_4cell_summary.json").is_file() else None

    ll_4c_dir = Path(args.llama_4cell)
    ll_4c = json.loads((ll_4c_dir / "stage4_4cell_summary.json").read_text(encoding="utf-8")) if (ll_4c_dir / "stage4_4cell_summary.json").is_file() else None

    # Export main DeepSeek tables & figures to top-level paper_artifacts
    if "alfworld" in ds_s3 and "scienceworld" in ds_s3:
        render_main_table(ds_s3, tab_dir / "main_results.tex")
        plot_learning_and_pareto(Path(args.deepseek_stage3), ds_s3, fig_dir)
    if ds_abl is not None:
        render_ablation_table(ds_abl, tab_dir / "ablations.tex")
    if ds_4c is not None:
        render_4cell_table(ds_4c, tab_dir / "counterfactual_4cell.tex")
    plot_ablation_and_4cell(ds_abl, ds_4c, fig_dir)

    if "alfworld" in q8_s3 and "scienceworld" in q8_s3:
        render_dual_backbone_table(ds_s3, q8_s3, tab_dir / "dual_backbone_main_results.tex")

    model_specs = [
        ("deepseek_v41_flash", "DeepSeek-V4.1-Flash", ds_s3, Path(args.deepseek_stage3)),
        ("qwen3_14b", "Qwen3-14B-AWQ", q14_s3, Path(args.qwen14b_stage3)),
        ("qwen3_8b", "Qwen3-8B", q8_s3, Path(args.qwen_stage3)),
        ("llama31_8b", "Llama-3.1-8B-AWQ", ll_s3, Path(args.llama_stage3)),
    ]
    render_four_backbone_table(model_specs, tab_dir / "four_backbone_main_results.tex")
    plot_four_backbone_comparison(model_specs, fig_dir)

    model_entries = [
        {"key": "deepseek_v41_flash", "label": "DeepSeek-V4.1-Flash", "s3_summary": ds_s3, "s3_dir": Path(args.deepseek_stage3), "s4_4cell": ds_4c, "s4_4cell_dir": ds_4c_dir, "s4_abl": ds_abl},
        {"key": "qwen3_14b", "label": "Qwen3-14B-AWQ", "s3_summary": q14_s3, "s3_dir": Path(args.qwen14b_stage3), "s4_4cell": q14_4c, "s4_4cell_dir": q14_4c_dir, "s4_abl": None},
        {"key": "qwen3_8b", "label": "Qwen3-8B", "s3_summary": q8_s3, "s3_dir": Path(args.qwen_stage3), "s4_4cell": q8_4c, "s4_4cell_dir": q8_4c_dir, "s4_abl": None},
        {"key": "llama31_8b", "label": "Llama-3.1-8B-AWQ", "s3_summary": ll_s3, "s3_dir": Path(args.llama_stage3), "s4_4cell": ll_4c, "s4_4cell_dir": ll_4c_dir, "s4_abl": None},
    ]
    for entry in model_entries:
        s3 = entry["s3_summary"]
        if "alfworld" not in s3 or "scienceworld" not in s3:
            continue
        sub = out_dir / entry["key"]
        (sub / "tables").mkdir(parents=True, exist_ok=True)
        (sub / "figures").mkdir(parents=True, exist_ok=True)
        render_main_table(s3, sub / "tables" / f"main_results_{entry['key']}.tex")
        plot_learning_and_pareto(entry["s3_dir"], s3, sub / "figures")
        if entry["s4_4cell"] is not None:
            render_4cell_table(entry["s4_4cell"], sub / "tables" / f"counterfactual_4cell_{entry['key']}.tex")
            plot_ablation_and_4cell(entry["s4_abl"], entry["s4_4cell"], sub / "figures")

    render_four_model_markdown_report(model_entries, ds_abl, Path(args.four_model_report_out))
    render_four_model_markdown_report(model_entries, ds_abl, Path(args.report_out))
    print(f"4-backbone artifacts exported successfully to {out_dir}, {args.four_model_report_out}, and {args.report_out}")


if __name__ == "__main__":
    main()
