"""Export publication-grade LaTeX tables and vector PDF/PNG figures from Stage 3 & Stage 4 benchmark outputs."""
from __future__ import annotations

import argparse
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


METHOD_LABELS = {
    "no-memory": "No-Memory",
    "full-memory": "Full-Memory (Top-$k$)",
    "individual": "Individual Credit",
    "mmr": "MMR Diversity",
    "additive": "Additive Laplace UCB",
    "expel-derived-online": "ExpeL-Derived Online$^{\\dagger}$",
    "evomemguard": "EvoMemGuard$^{\\ddagger}$",
    "comep": "\\textbf{CoMeP (Ours)}",
}

ABLATION_LABELS = {
    "comep": "\\textbf{Full CoMeP}",
    "comep-no-hard-conflicts": "\\quad w/o Hard Conflict Guard",
    "comep-no-pair-optimism": "\\quad w/o Pairwise Optimism ($\\beta_{\\text{pair}}=0$)",
    "comep-no-early-stop": "\\quad w/o Marginal Gain/Token Early Stop",
    "comep-no-coalition-pruning": "\\quad w/o Coalition-Aware Pruning",
    "comep-sgd-v1": "\\quad w/o Batch Laplace MAP (Online SGD v1)",
}

COLORS = {
    "comep": "#d62728",
    "additive": "#1f77b4",
    "evomemguard": "#2ca02c",
    "expel-derived-online": "#9467bd",
    "mmr": "#ff7f0e",
    "individual": "#8c564b",
    "full-memory": "#7f7f7f",
    "no-memory": "#bcbd22",
}


def render_main_table(stage3_summary: dict, out_path: Path):
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Main online memory routing \\& pruning results on held-out \\textbf{ALFWorld} (\\texttt{valid\\_unseen}) and \\textbf{ScienceWorld} (\\texttt{test}). $\\dagger$: matched-budget online critic adaptation (Zhao et al., 2024); $\\ddagger$: architecture-grounded NLI conflict graph + smoothed empirical utility adaptation (Wang et al., 2026). $p$-values report unadjusted / 7-comparison Holm-adjusted (within each panel) exact two-sided paired McNemar tests vs.\\ \\textbf{CoMeP} over sequential online evaluation episodes.}",
        "\\label{tab:main_results}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Method & Success (\\%) & Late SR (\\%) & Ret.\\ Tok/Ep & Mgr/Dist Calls & API Cost (\\$) & Raw / Holm $p$ vs.\\ CoMeP \\\\",
        "\\midrule",
    ]
    for env_key, env_title in [("alfworld", "ALFWorld (\\texttt{valid\\_unseen})"), ("scienceworld", "ScienceWorld (\\texttt{test})")]:
        if env_key not in stage3_summary:
            continue
        grid = stage3_summary[env_key]
        n_seeds = len(grid["seeds"])
        seed_lbl = "seed" if n_seeds == 1 else "seeds"
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{env_title} ({grid['episodes_per_seed']} tasks $\\times$ {n_seeds} {seed_lbl})}}}} \\\\")
        methods = grid["methods"]
        paired = grid.get("paired_vs_comep", {})
        max_sr = max((round(methods[m]["mean_sr"] * 100, 1) for m in methods), default=0.0)
        max_late = max((round(methods[m]["mean_late_sr"] * 100, 1) for m in methods), default=0.0)
        from scipy.stats import binomtest

        base_ms = [m for m in ["no-memory", "full-memory", "individual", "mmr", "additive", "expel-derived-online", "evomemguard"] if m in methods]
        raw_ps = {}
        for m in base_ms:
            pm = paired.get(f"comep_vs_{m}", {})
            b10 = int(pm.get("comep_wins_discordant", 0))
            b01 = int(pm.get("baseline_wins_discordant", 0))
            raw_ps[m] = float(binomtest(b10, b10 + b01, 0.5).pvalue) if (b10 + b01) > 0 else 1.0
        sorted_ms = sorted(base_ms, key=lambda x: raw_ps[x])
        holm_ps = {}
        running_max = 0.0
        for rank_idx, bm in enumerate(sorted_ms):
            adj = min(1.0, raw_ps[bm] * (len(sorted_ms) - rank_idx))
            running_max = max(running_max, adj)
            holm_ps[bm] = running_max
        for m in ["no-memory", "full-memory", "individual", "mmr", "additive", "expel-derived-online", "evomemguard", "comep"]:
            if m not in methods:
                continue
            row = methods[m]
            sr_str = f"{row['mean_sr']*100:.1f} $\\pm$ {row['std_sr']*100:.1f}" if n_seeds > 1 else f"{row['mean_sr']*100:.1f}"
            late_str = f"{row['mean_late_sr']*100:.1f}"
            tok_str = f"{row['mean_retrieval_tokens']:.1f}"
            calls_str = f"{row['manager_llm_calls']} / {row['distillation_llm_calls']}"
            cost_str = f"{row['total_cost_usd']:.2f}" if row["total_cost_usd"] > 0 else "N/A"
            if round(row["mean_sr"] * 100, 1) == max_sr:
                sr_str = f"\\textbf{{{sr_str}}}"
            if round(row["mean_late_sr"] * 100, 1) == max_late:
                late_str = f"\\textbf{{{late_str}}}"
            if m == "comep":
                p_str = "---"
            else:
                p_val = raw_ps[m]
                h_val = holm_ps[m]
                r_s = f"{p_val:.4f}" if p_val >= 0.0001 else "$<10^{-4}$"
                h_s = f"{h_val:.4f}" if h_val >= 0.0001 else "$<10^{-4}$"
                p_str = f"{r_s} / {h_s}"
            lines.append(f"{METHOD_LABELS.get(m, m)} & {sr_str} & {late_str} & {tok_str} & {calls_str} & {cost_str} & {p_str} \\\\")
        lines.append("\\midrule")
    if "alfworld" in stage3_summary and "scienceworld" in stage3_summary:
        alf_m = stage3_summary["alfworld"]["methods"]
        sci_m = stage3_summary["scienceworld"]["methods"]
        n_alf = stage3_summary["alfworld"]["episodes_per_seed"] * len(stage3_summary["alfworld"]["seeds"])
        n_sci = stage3_summary["scienceworld"]["episodes_per_seed"] * len(stage3_summary["scienceworld"]["seeds"])
        n_tot = n_alf + n_sci
        from scipy.stats import binomtest

        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{Combined Held-Out Benchmark ({n_tot} paired episodes/method; Pooled SR / Macro Late SR / Macro Ret.\\ Tok)}}}} \\\\")
        valid_comb = [m for m in ["no-memory", "full-memory", "individual", "mmr", "additive", "expel-derived-online", "evomemguard", "comep"] if m in alf_m and m in sci_m]
        max_comb_sr = max((round(((round(alf_m[m]["mean_sr"] * n_alf) + round(sci_m[m]["mean_sr"] * n_sci)) / n_tot) * 100, 1) for m in valid_comb), default=0.0)
        max_comb_late = max((round(0.5 * (alf_m[m]["mean_late_sr"] + sci_m[m]["mean_late_sr"]) * 100, 1) for m in valid_comb), default=0.0)
        comb_base = [m for m in valid_comb if m != "comep"]
        comb_raw_ps = {}
        for bm in comb_base:
            pa = stage3_summary["alfworld"]["paired_vs_comep"].get(f"comep_vs_{bm}", {})
            ps = stage3_summary["scienceworld"]["paired_vs_comep"].get(f"comep_vs_{bm}", {})
            b10 = int(pa.get("comep_wins_discordant", 0)) + int(ps.get("comep_wins_discordant", 0))
            b01 = int(pa.get("baseline_wins_discordant", 0)) + int(ps.get("baseline_wins_discordant", 0))
            comb_raw_ps[bm] = float(binomtest(b10, b10 + b01, 0.5).pvalue) if (b10 + b01) > 0 else 1.0
        sorted_cb = sorted(comb_base, key=lambda x: comb_raw_ps[x])
        comb_holm_ps = {}
        running_max = 0.0
        for rank_idx, bm in enumerate(sorted_cb):
            adj = min(1.0, comb_raw_ps[bm] * (len(sorted_cb) - rank_idx))
            running_max = max(running_max, adj)
            comb_holm_ps[bm] = running_max
        for m in valid_comb:
            a, s = alf_m[m], sci_m[m]
            pooled_sr = (round(a["mean_sr"] * n_alf) + round(s["mean_sr"] * n_sci)) / n_tot
            macro_late = 0.5 * (a["mean_late_sr"] + s["mean_late_sr"])
            macro_tok = 0.5 * (a["mean_retrieval_tokens"] + s["mean_retrieval_tokens"])
            mgr = a["manager_llm_calls"] + s["manager_llm_calls"]
            dist = a["distillation_llm_calls"] + s["distillation_llm_calls"]
            tot_cost = a["total_cost_usd"] + s["total_cost_usd"]
            cost_str = f"{tot_cost:.2f}" if tot_cost > 0 else "N/A"
            sr_str = f"\\textbf{{{pooled_sr*100:.1f}}}" if round(pooled_sr * 100, 1) == max_comb_sr else f"{pooled_sr*100:.1f}"
            late_str = f"\\textbf{{{macro_late*100:.1f}}}" if round(macro_late * 100, 1) == max_comb_late else f"{macro_late*100:.1f}"
            if m == "comep":
                p_str = "---"
            else:
                p_val = comb_raw_ps[m]
                h_val = comb_holm_ps[m]
                r_s = f"{p_val:.4f}" if p_val >= 0.0001 else "$<10^{-4}$"
                h_s = f"{h_val:.4f}" if h_val >= 0.0001 else "$<10^{-4}$"
                p_str = f"{r_s} / {h_s}"
            lines.append(
                f"{METHOD_LABELS.get(m, m)} & {sr_str} & {late_str} & {macro_tok:.1f} & {mgr} / {dist} & {cost_str} & {p_str} \\\\"
            )
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    lines.extend(["\\end{tabular}%", "}", "\\end{table*}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_ablation_table(stage4_abl: dict, out_path: Path):
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Ablation study of \\textbf{CoMeP} components on held-out environments.}",
        "\\label{tab:ablations}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Variant & Success (\\%) & Late SR (\\%) & Ret.\\ Tok/Ep \\\\",
        "\\midrule",
    ]
    for env_key, env_title in [("alfworld", "ALFWorld"), ("scienceworld", "ScienceWorld")]:
        if env_key not in stage4_abl:
            continue
        grid = stage4_abl[env_key]
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{env_title}}}}} \\\\")
        for m, label in ABLATION_LABELS.items():
            if m not in grid["methods"]:
                continue
            r = grid["methods"][m]
            lines.append(f"{label} & {r['mean_sr']*100:.1f} $\\pm$ {r['std_sr']*100:.1f} & {r['mean_late_sr']*100:.1f} & {r['mean_retrieval_tokens']:.1f} \\\\")
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    lines.extend(["\\end{tabular}", "\\end{table}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_4cell_table(stage4_4cell: dict, out_path: Path):
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Exploratory $2\\times 2$ counterfactual intervention check ($\\{\\varnothing, A, B, AB\\}$) on candidate positive and negative pairs. Task-paired bootstrap 95\\% CIs ($5{,}000$ resamples, seed 77) are reported in percentage points (pp); $^{*}$ denotes a degenerate $[0.0, 0.0]$ empirical bootstrap interval caused by zero cross-task variance in per-task interaction $\\Delta_t = r_{AB,t} - r_{A,t} - r_{B,t} + r_{\\varnothing,t}$ (e.g., all-zero binary cell outcomes or identical per-task cell outcomes across all four conditions; does not establish zero population effect).}",
        "\\label{tab:counterfactual_4cell}",
        "\\begin{tabular}{llllcccccc}",
        "\\toprule",
        "Env (Context) & Candidate & Source & $n$/cell & $\\hat{s}_{AB}$ & $\\varnothing$ & $A$ & $B$ & $AB$ & Linear $\\hat{\\Delta}_{AB}$ [95\\% CI] \\\\",
        "\\midrule",
    ]
    for env_key, items in stage4_4cell.items():
        for rep in items:
            cr = rep["cell_rates"]
            ci = rep["ci95_delta_AB"]
            n_c = int(rep.get("episodes_per_cell", 16))
            ctx_lbl = rep.get("target_context", "").replace("_", "\\_")
            p_type = "Pos.\\ Cand." if "pos" in rep["pair_kind"] else "Neg.\\ Cand."
            src = "Posterior" if rep.get("selection_source", "posterior_s_ij") == "posterior_s_ij" else "Archived"
            s_val = f"{rep['learned_s_ij']:+.2f}" if rep.get("learned_s_ij") is not None else "NA"
            zero_var_ci = abs(float(ci[0])) < 1e-12 and abs(float(ci[1])) < 1e-12
            ci_str = f"[{ci[0]*100:+.1f}, {ci[1]*100:+.1f}]$^{{*}}$" if zero_var_ci else f"[{ci[0]*100:+.1f}, {ci[1]*100:+.1f}]"
            lines.append(
                f"{env_key} (\\texttt{{{ctx_lbl}}}) & {p_type} & {src} & {n_c} & {s_val} & {cr['none']*100:.1f}\\% & {cr['A']*100:.1f}\\% & "
                f"{cr['B']*100:.1f}\\% & {cr['AB']*100:.1f}\\% & ${rep['causal_interaction_delta_AB']*100:+.1f}$ pp {ci_str} \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_learning_and_pareto(stage3_dir: Path, stage3_summary: dict, fig_dir: Path):
    envs = [e for e in ("alfworld", "scienceworld") if e in stage3_summary]
    if not envs:
        return

    # Figure 1: Rolling Learning Curves
    fig, axes = plt.subplots(1, len(envs), figsize=(6.2 * len(envs), 4.4), squeeze=False)
    for idx, env_key in enumerate(envs):
        ax = axes[0, idx]
        env_dir = stage3_dir / env_key
        seeds = stage3_summary[env_key]["seeds"]
        eps = stage3_summary[env_key]["episodes_per_seed"]
        win = max(8, eps // 6)
        for m in ["comep", "additive", "evomemguard", "expel-derived-online", "mmr", "individual", "full-memory", "no-memory"]:
            curves = []
            for s in seeds:
                ep_file = env_dir / f"{m}_seed{s}" / "episodes.jsonl"
                if not ep_file.is_file():
                    continue
                recs = sorted((json.loads(l) for l in ep_file.read_text(encoding="utf-8").splitlines() if l.strip()), key=lambda r: r["episode"])
                y = np.array([float(r["reward"]) for r in recs])
                if len(y) >= win:
                    csum = np.cumsum(np.insert(y, 0, 0.0))
                    rolling = (csum[win:] - csum[:-win]) / win
                    curves.append(rolling * 100.0)
            if curves:
                arr_c = np.array(curves, dtype=float)
                mean_c = np.mean(arr_c, axis=0)
                std_c = np.std(arr_c, axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean_c)
                xs = np.arange(win, win + len(mean_c))
                col = COLORS.get(m, "#333333")
                lw = 2.6 if m == "comep" else 1.5
                alpha = 1.0 if m in ("comep", "additive", "evomemguard", "expel-derived-online") else 0.72
                if m in ("comep", "additive") and len(curves) > 1:
                    ax.fill_between(xs, np.clip(mean_c - std_c, 0.0, 100.0), np.clip(mean_c + std_c, 0.0, 100.0), color=col, alpha=0.13)
                if m == "comep" and len(curves) > 1:
                    for s_curve in curves:
                        ax.plot(xs, s_curve, color=col, linewidth=0.75, linestyle=":", alpha=0.42)
                ax.plot(xs, mean_c, label=m, color=col, linewidth=lw, alpha=alpha)
        ax.set_title(f"{env_key.upper()} Online Task Success (Rolling-{win})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Online Evaluation Episode")
        ax.set_ylabel("Success Rate (%)")
        ax.grid(True, alpha=0.25)
        if idx == len(envs) - 1:
            ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_learning_curves.pdf")
    fig.savefig(fig_dir / "fig1_learning_curves.png", dpi=220)
    plt.close(fig)

    # Figure 2: Retrieval Token Efficiency vs Success Rate (Pareto Frontier)
    pareto_offsets = {
        "alfworld": {
            "comep": (8, 8, "left", "bottom", True),
            "additive": (-10, 0, "right", "center", False),
            "expel-derived-online": (-9, 0, "right", "center", False),
            "full-memory": (-9, 4, "right", "bottom", False),
            "evomemguard": (-9, -1, "right", "top", False),
            "individual": (-14, -13, "right", "top", True),
            "mmr": (-9, 0, "right", "center", False),
            "no-memory": (8, 1, "left", "center", False),
        },
        "scienceworld": {
            "individual": (-9, 1, "right", "center", False),
            "comep": (-9, 4, "right", "bottom", False),
            "additive": (-9, -5, "right", "top", False),
            "evomemguard": (-9, 0, "right", "center", False),
            "no-memory": (8, 1, "left", "center", False),
            "full-memory": (-9, 1, "right", "center", False),
            "mmr": (-9, 1, "right", "center", False),
            "expel-derived-online": (-9, -5, "right", "top", False),
        },
    }
    fig, axes = plt.subplots(1, len(envs), figsize=(6.2 * len(envs), 4.4), squeeze=False)
    for idx, env_key in enumerate(envs):
        ax = axes[0, idx]
        methods = stage3_summary[env_key]["methods"]
        for m, row in methods.items():
            x = row["mean_retrieval_tokens"]
            y = row["mean_sr"] * 100.0
            ax.scatter(x, y, s=120 if m == "comep" else 75, color=COLORS.get(m, "#333333"), zorder=4 if m == "comep" else 3)
            dx, dy, ha, va, use_leader = pareto_offsets.get(env_key, {}).get(
                m, ((-7, 4, "right", "bottom", False) if x > 74 else (6, 4, "left", "bottom", False))
            )
            arrow_kw = dict(arrowstyle="-", color="#777777", lw=0.65, shrinkA=0, shrinkB=4) if use_leader else None
            ax.annotate(
                m,
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=8.5,
                fontweight="bold" if m == "comep" else "normal",
                color="#900000" if m == "comep" else "#111111",
                arrowprops=arrow_kw,
            )
        ax.set_xlim(-6, 97 if env_key == "alfworld" else 92)
        if env_key == "alfworld":
            ax.set_ylim(71.8, 92.2)
        elif env_key == "scienceworld":
            ax.set_ylim(38.5, 62.0)
        else:
            ax.margins(y=0.10)
        ax.set_title(f"{env_key.upper()} Memory Token Budget vs. Success Rate", fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean Retrieved Memory Tokens / Episode (Budget = 90)")
        ax.set_ylabel("Task Success Rate (%)")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_pareto_cost_vs_sr.pdf")
    fig.savefig(fig_dir / "fig2_pareto_cost_vs_sr.png", dpi=220)
    plt.close(fig)


def plot_ablation_and_4cell(stage4_abl: dict | None, stage4_4cell: dict | None, fig_dir: Path):
    if not stage4_abl and not stage4_4cell:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))

    # Left panel: Macro Retrieval Tokens vs Macro Success Rate across Ablations (or ScienceWorld 4-Cell if no ablations)
    ax0 = axes[0]
    if stage4_abl and "alfworld" in stage4_abl and "scienceworld" in stage4_abl:
        short_names = {
            "comep": "Full CoMeP",
            "comep-no-hard-conflicts": "w/o HardConflict",
            "comep-no-pair-optimism": "w/o PairOptimism",
            "comep-no-early-stop": "w/o EarlyStop",
            "comep-no-coalition-pruning": "w/o CoalPrune",
            "comep-sgd-v1": "Online SGD v1",
        }
        for m, sname in short_names.items():
            if m in stage4_abl["alfworld"]["methods"] and m in stage4_abl["scienceworld"]["methods"]:
                ra = stage4_abl["alfworld"]["methods"][m]
                rs = stage4_abl["scienceworld"]["methods"][m]
                tok = 0.5 * (ra["mean_retrieval_tokens"] + rs["mean_retrieval_tokens"])
                sr = 0.5 * (ra["mean_sr"] + rs["mean_sr"]) * 100.0
                col = "#d62728" if m == "comep" else ("#7f7f7f" if m == "comep-sgd-v1" else "#1f77b4")
                ax0.scatter(tok, sr, s=120 if m == "comep" else 80, color=col, zorder=4)
                dx, ha = (-7, "right") if tok > 74 else (6, "left")
                ax0.annotate(sname, (tok, sr), xytext=(dx, 4), textcoords="offset points", ha=ha, fontsize=8.5, fontweight="bold" if m == "comep" else "normal")
        ax0.margins(x=0.18, y=0.18)
        ax0.set_title("Stage 4A: Ablation Macro-SR vs. Retrieved Tokens", fontsize=11, fontweight="bold")
        ax0.set_xlabel("Macro Mean Retrieved Tokens / Episode")
        ax0.set_ylabel("Macro Success Rate (%)")
        ax0.grid(True, alpha=0.25)
    elif stage4_4cell and "scienceworld" in stage4_4cell and len(stage4_4cell["scienceworld"]) >= 2:
        cells = ["none", "A", "B", "AB"]
        x = np.arange(len(cells))
        width = 0.35
        syn = stage4_4cell["scienceworld"][0]
        con = stage4_4cell["scienceworld"][1]
        y_syn = [syn.get("cell_mean_scores", syn["cell_rates"])[c] for c in cells]
        y_con = [con.get("cell_mean_scores", con["cell_rates"])[c] for c in cells]
        s_syn = f"{syn['learned_s_ij']:+.2f}" if syn.get("learned_s_ij") is not None else "NA"
        s_con = f"{con['learned_s_ij']:+.2f}" if con.get("learned_s_ij") is not None else "NA"
        d_syn = syn.get("score_interaction_delta_AB", syn["causal_interaction_delta_AB"] * 100.0)
        d_con = con.get("score_interaction_delta_AB", con["causal_interaction_delta_AB"] * 100.0)
        ax0.bar(x - width / 2, y_syn, width, label=f"Pos. Cand. (s={s_syn}, Δscore={d_syn:+.1f})", color="#2ca02c", alpha=0.88)
        ax0.bar(x + width / 2, y_con, width, label=f"Neg. Cand. (s={s_con}, Δscore={d_con:+.1f})", color="#d62728", alpha=0.88)
        ax0.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
        ax0.set_xticks(x)
        ax0.set_xticklabels(["∅ (No Mem)", "Rule A Only", "Rule B Only", "A + B Joint"])
        ax0.set_ylabel("Mean Final Environment Score (incl. -100 penalty)")
        ax0.set_title("Stage 4B: ScienceWorld 2×2 Intervention Check (Score)", fontsize=11, fontweight="bold")
        ax0.legend(fontsize=8, loc="upper left")
        ax0.grid(True, axis="y", alpha=0.25)

    # Right panel: ALFWorld 2x2 Counterfactual Candidate Intervention Check
    ax1 = axes[1]
    if stage4_4cell and "alfworld" in stage4_4cell and len(stage4_4cell["alfworld"]) >= 2:
        cells = ["none", "A", "B", "AB"]
        x = np.arange(len(cells))
        width = 0.35
        syn = stage4_4cell["alfworld"][0]
        con = stage4_4cell["alfworld"][1]
        y_syn = [syn["cell_rates"][c] * 100.0 for c in cells]
        y_con = [con["cell_rates"][c] * 100.0 for c in cells]
        s_syn = f"{syn['learned_s_ij']:+.2f}" if syn.get("learned_s_ij") is not None else "NA"
        s_con = f"{con['learned_s_ij']:+.2f}" if con.get("learned_s_ij") is not None else "NA"
        ax1.bar(x - width / 2, y_syn, width, label=f"Pos. Cand. (s={s_syn}, Δ={syn['causal_interaction_delta_AB']*100:+.1f}pp)", color="#2ca02c", alpha=0.88)
        ax1.bar(x + width / 2, y_con, width, label=f"Neg. Cand. (s={s_con}, Δ={con['causal_interaction_delta_AB']*100:+.1f}pp)", color="#d62728", alpha=0.88)
        ax1.set_xticks(x)
        ax1.set_xticklabels(["∅ (No Mem)", "Rule A Only", "Rule B Only", "A + B Joint"])
        ax1.set_ylim(0, 108)
        ax1.set_ylabel("Task Success Rate (%)")
        ax1.set_title("Stage 4B: ALFWorld 2×2 Intervention Check (SR %)", fontsize=11, fontweight="bold")
        ax1.legend(fontsize=8, loc="upper right")
        ax1.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_ablation_and_4cell.pdf")
    fig.savefig(fig_dir / "fig3_ablation_and_4cell.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-dir", required=True)
    parser.add_argument("--stage4-ablations-dir", default=None)
    parser.add_argument("--stage4-4cell-dir", default=None)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    tab_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    tab_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    stage3_dir = Path(args.stage3_dir)
    s3_summary = json.loads((stage3_dir / "stage3_main_summary.json").read_text(encoding="utf-8"))
    render_main_table(s3_summary, tab_dir / "main_results.tex")
    plot_learning_and_pareto(stage3_dir, s3_summary, fig_dir)

    s4a = None
    s4c = None
    if args.stage4_ablations_dir:
        s4a = json.loads((Path(args.stage4_ablations_dir) / "stage4_ablations_summary.json").read_text(encoding="utf-8"))
        render_ablation_table(s4a, tab_dir / "ablations.tex")
    if args.stage4_4cell_dir:
        s4c = json.loads((Path(args.stage4_4cell_dir) / "stage4_4cell_summary.json").read_text(encoding="utf-8"))
        render_4cell_table(s4c, tab_dir / "counterfactual_4cell.tex")
    plot_ablation_and_4cell(s4a, s4c, fig_dir)

    print(f"Exported tables to {tab_dir} and figures to {fig_dir}")


if __name__ == "__main__":
    main()
