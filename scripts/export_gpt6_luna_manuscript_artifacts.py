"""Deterministic exporter for the GPT-6-Luna six-backbone manuscript edition
(replacing MiMo-V2.6-Flash non-destructively while preserving all historical MiMo artifacts).

Exports:
  - paper/neurocomputing/figures/Figure_8_gpt6_luna.pdf (6.02 x 2.82 in, Item-8 single-Axes spec)
  - paper/figures/fig4_six_backbone_comparison_gpt6_luna.{pdf,png} (14.4 x 11.7 in, 6-panel arXiv spec)
  - paper_artifacts/figures/fig4_six_backbone_comparison_gpt6_luna.{pdf,png}
  - paper_artifacts/routing_latency_audit_gpt6_luna.json
  - paper_artifacts/reproducibility_mapping_gpt6_luna.json
  - paper_artifacts/deployment_provenance/deployment_audit_gpt6_luna.json
"""
from __future__ import annotations

import argparse
import json
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

DEFAULT_REPO_ROOT = Path("/root/.paseo/worktrees/1s5ku8sn/lucid-bumblebee")
sys.path.insert(0, str(DEFAULT_REPO_ROOT))
sys.path.insert(0, str(DEFAULT_REPO_ROOT / "scripts"))

from comep.core import _ACTION_CLAUSE_STARTERS
from campaign_source_resolver import (
    MAIN_METHODS,
    STAGE4_ABLATION_SPEC,
    compute_six_backbone_aggregates,
    resolve_and_verify_campaign_dir,
    sha256_file,
    verify_and_load_stage3_campaign,
    verify_source_hashes,
)

NO_ADDITIVE_METHODS: tuple[str, ...] = (
    "no-memory",
    "full-memory",
    "individual",
    "mmr",
    "expel-derived-online",
    "evomemguard",
    "comep",
)

STAGE3_GPT6_CAMPAIGN_SPECS: tuple[dict, ...] = (
    {
        "key": "deepseek_v41_flash",
        "label": "Stage 3: DeepSeek-V4.1-Flash (ungated)",
        "backbone_name": "DeepSeek-V4.1-Flash",
        "rel_dir": "outputs/stage3-main",
        "n_alf": 90,
        "n_sci": 75,
        "warmup_gate": False,
        "pinned_hashes": {
            "stage3_main_summary.json": "7651e9056cc014f98196a1b2b894ba40a34b85f17de19df4df988048331f4f86",
        },
    },
    {
        "key": "glm53_flash_v22",
        "label": "Stage 3: GLM-5.3-Flash (final gated)",
        "backbone_name": "GLM-5.3-Flash",
        "rel_dir": "outputs/stage3-glm53-flash-v22",
        "n_alf": 90,
        "n_sci": 75,
        "warmup_gate": True,
        "pinned_hashes": {
            "stage3_main_summary.json": "848cb5b896964b3c8fb773fe90c6b82d5c04e61ecef65ae8a01d20d0c8c874cf",
            "gateway_provenance.json": "0fbfaedc2d9ea4309e43b49543562d329de06d07df1def9814cc7b149c8e6ad9",
            "orchestrator_retry_ledger.json": "204c7b96919e34ca415d6605bc59945c263591bd205ab703fe2f4e03d3af8801",
        },
    },
    {
        "key": "gpt6_luna_v22",
        "label": "Stage 3: GPT-6-Luna (final gated)",
        "backbone_name": "GPT-6-Luna",
        "rel_dir": "outputs/stage3-gpt6-luna-v22",
        "n_alf": 90,
        "n_sci": 75,
        "warmup_gate": True,
        "pinned_hashes": {
            "stage3_main_summary.json": "e9ad9cebf4d1caa0caabacbbb66fce26be376e96589a120caec243c3d7abfbfd",
            "gateway_provenance.json": "a3d11052f8249dabbe8322e29770f7da0e8d62615d8cb0fccd4d6b5511d6babc",
            "alfworld/grid_summary.json": "cc9220ed4ffac942dc02dceacbb711f69c02ad24f68c1d162b999a7961bd59ed",
            "scienceworld/grid_summary.json": "0b0c138157c23d64b86e6c89a4c8347df626ad2e05472050d0c26edd20737b7b",
            "orchestrator_retry_ledger.json": "dd2f29492af35fd50a28e5fcbe90ffbd11db02c92fcaa3139373b63b681abae1",
        },
    },
    {
        "key": "qwen3_14b_awq_v22",
        "label": "Stage 3: Qwen3-14B-AWQ (final gated)",
        "backbone_name": "Qwen3-14B-AWQ",
        "rel_dir": "outputs/stage3-qwen3-14b-v22",
        "n_alf": 60,
        "n_sci": 50,
        "warmup_gate": True,
        "pinned_hashes": {
            "stage3_main_summary.json": "604353f2d088c681fc7451ef417d38e00b690a6dba9fb0c41f87c2a67b68a0d5",
        },
    },
    {
        "key": "gemma4_12b_it_v22",
        "label": "Stage 3: Gemma-4-12B-it (final gated)",
        "backbone_name": "Gemma-4-12B-it",
        "rel_dir": "outputs/stage3-gemma4-12b",
        "n_alf": 60,
        "n_sci": 50,
        "warmup_gate": True,
        "pinned_hashes": {
            "stage3_main_summary.json": "2a27cb496876523405b97cb396a96ad311271027895a943c6a79aad1d973c34b",
        },
    },
    {
        "key": "qwen3_8b",
        "label": "Stage 3: Qwen3-8B (ungated)",
        "backbone_name": "Qwen3-8B",
        "rel_dir": "outputs/stage3-qwen3-8b",
        "n_alf": 60,
        "n_sci": 50,
        "warmup_gate": False,
        "pinned_hashes": {
            "stage3_main_summary.json": "92bc4ffa27bf7e2476d28806dea4e1f122b0a5f682876dbc009eedd87f61ab05",
        },
    },
)

# --- ArXiv 6-panel style (matching scripts/export_six_backbone_figure.py) ---
ARXIV_METHODS = [
    ("no-memory", "No-\nMemory", "#95a5a6"),
    ("full-memory", "Full-Memory\n(Top-k)", "#7f8c8d"),
    ("individual", "Individual\nCredit", "#8c564b"),
    ("mmr", "MMR-style\nDiversity", "#e67e22"),
    ("additive", "Additive\nLaplace UCB", "#2980b9"),
    ("expel-derived-online", "ExpeL-\nderived\nOnline", "#8e44ad"),
    ("evomemguard", "EvoMem-\nGuard-\nstyle", "#27ae60"),
    ("comep", "CoMeP\n(Ours)", "#c0392b"),
]

ARXIV_PANEL_METADATA = {
    "deepseek_v41_flash": ("(a) DeepSeek-V4.1-Flash (495 eps/method)", 96),
    "glm53_flash_v22": ("(b) GLM-5.3-Flash (495 eps/method; OpenCode Go API)", 84),
    "gpt6_luna_v22": ("(c) GPT-6-Luna (495 eps/method; Official OpenAI API)", 84),
    "qwen3_14b_awq_v22": ("(d) Qwen3-14B-AWQ (330 eps/method; 4-bit AWQ)", 55),
    "gemma4_12b_it_v22": ("(e) Gemma-4-12B-it (330 eps/method; W4A16 QAT + Proxy Bias)", 38),
    "qwen3_8b": ("(f) Qwen3-8B (330 eps/method; BF16)", 46),
}

# --- Neurocomputing Item-8 standalone style (matching Figure_6..11.pdf) ---
JOURNAL_PALETTE = {
    "no-memory": "#94A3B8",
    "full-memory": "#64748B",
    "individual": "#8D6E63",
    "mmr": "#D9822B",
    "additive": "#2B6CB0",
    "expel-derived-online": "#7E6B9E",
    "evomemguard": "#3D9970",
    "comep": "#C81E1E",
}

JOURNAL_METHODS_BAR = [
    ("no-memory", "No-\nMemory"),
    ("full-memory", "Full-Memory\n(Top-k)"),
    ("individual", "Individual\nCredit"),
    ("mmr", "MMR-style\nDiversity"),
    ("additive", "Additive\nLaplace UCB"),
    ("expel-derived-online", "ExpeL-\nderived\nOnline"),
    ("evomemguard", "EvoMem-\nGuard-style"),
    ("comep", "CoMeP\n(Ours)"),
]


def draw_arxiv_panel_ax(ax, methods_stats: dict[str, dict], title: str, ymax: float) -> None:
    xs = np.arange(len(ARXIV_METHODS))
    means = [methods_stats[m]["pooled_sr_pct"] for m, _, _ in ARXIV_METHODS]
    sds = [methods_stats[m]["pooled_sd_pct"] for m, _, _ in ARXIV_METHODS]
    colors = [c for _, _, c in ARXIV_METHODS]
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
    for idx, (m_key, _, _) in enumerate(ARXIV_METHODS):
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
            fontsize=7.3,
            fontweight="bold" if m_key == "comep" else "normal",
            color="#900000" if m_key == "comep" else "#222222",
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for _, lbl, _ in ARXIV_METHODS], fontsize=8.2)
    for tick_lbl in ax.get_xticklabels():
        if "CoMeP" in tick_lbl.get_text():
            tick_lbl.set_fontweight("bold")
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Combined Pooled Success Rate (%)", fontsize=9.5)
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=6)
    ax.grid(axis="y", linestyle="--", alpha=0.32)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def export_journal_figure_8(methods_stats: dict[str, dict], out_pdf: Path) -> None:
    title = "GPT-6-Luna (495 eps/method; Official OpenAI API)"
    ymax = 88.0
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


def export_arxiv_six_backbone_figure(loaded_stage3: list[dict], out_dirs: list[Path]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14.4, 11.7))
    axes_flat = axes.flatten()
    for ax, camp in zip(axes_flat, loaded_stage3):
        ckey = camp["spec"]["key"]
        panel_title, ymax = ARXIV_PANEL_METADATA[ckey]
        draw_arxiv_panel_ax(ax, camp["methods"], panel_title, ymax)

    fig.tight_layout(pad=1.3, w_pad=1.8, h_pad=1.6)
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_dir / "fig4_six_backbone_comparison_gpt6_luna.pdf",
            metadata={"CreationDate": None},
        )
        fig.savefig(out_dir / "fig4_six_backbone_comparison_gpt6_luna.png", dpi=220)
    plt.close(fig)


def export_routing_latency_audit_gpt6(loaded_stage3: list[dict], out_path: Path) -> dict:
    s4_files = ["stage4_ablations_summary.json"]
    for m_name in ("comep-warmup-gate", "comep"):
        for env_name in ("alfworld", "scienceworld"):
            for s in (0, 1):
                s4_files.append(f"{env_name}/{m_name}_seed{s}/episodes.jsonl")
    _, s4_base = resolve_and_verify_campaign_dir(
        rel_dir=STAGE4_ABLATION_SPEC["rel_dir"],
        required_rel_files=s4_files,
        pinned_hashes=STAGE4_ABLATION_SPEC["pinned_hashes"],
    )

    specs = [
        (c["spec"]["label"], c["spec"]["rel_dir"], Path(c["resolved_dir"]), "comep", (0, 1, 2))
        for c in loaded_stage3
    ] + [
        (
            "Stage 4A: Full CoMeP (final gated)",
            STAGE4_ABLATION_SPEC["rel_dir"],
            s4_base,
            "comep-warmup-gate",
            (0, 1),
        ),
        (
            "Stage 4A: w/o Warm-up Gate (ungated ref.)",
            STAGE4_ABLATION_SPEC["rel_dir"],
            s4_base,
            "comep",
            (0, 1),
        ),
    ]

    rows = []
    source_hashes = {}
    for label, rel_dir, base_dir, m_name, seeds in specs:
        for env_name in ("alfworld", "scienceworld"):
            vals_ms = []
            for s in seeds:
                p = base_dir / env_name / f"{m_name}_seed{s}" / "episodes.jsonl"
                rel_p = f"{rel_dir}/{env_name}/{m_name}_seed{s}/episodes.jsonl"
                source_hashes[rel_p] = sha256_file(p)
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        rec = json.loads(line)
                        vals_ms.append(float(rec["route_seconds"]) * 1000.0)
            arr = np.array(vals_ms, dtype=float)
            rows.append(
                {
                    "campaign": label,
                    "output_dir": rel_dir,
                    "method_key": m_name,
                    "environment": env_name,
                    "seeds": list(seeds),
                    "episodes": int(len(arr)),
                    "mean_ms": round(float(np.mean(arr)), 1),
                    "median_ms": round(float(np.median(arr)), 1),
                    "p95_ms": round(float(np.percentile(arr, 95)), 1),
                    "max_ms": round(float(np.max(arr)), 1),
                    "full_precision_ms": {
                        "mean": float(np.mean(arr)),
                        "median": float(np.median(arr)),
                        "p95": float(np.percentile(arr, 95)),
                        "max": float(np.max(arr)),
                    },
                }
            )
    payload = {
        "kind": "routing_latency_audit_gpt6_luna",
        "timer_scope": "Wall-clock duration of manager.choose(...) recorded as route_seconds in episodes.jsonl by execute_worker (comep/campaign.py:358; query/context extraction occurs prior to timer start). Covers relevance gating and two-element lookahead bundle selection. Post-episode parameter refitting (manager.observe at comep/campaign.py:440), optional post-episode rule distillation on successful episodes (comep/campaign.py:464), and bank pruning occur after episode completion outside this timer and are not timed separately in saved run logs.",
        "caveat": "Descriptive wall-clock measurements under the concurrent worker loads and host/Colab execution environments of each campaign; not an isolated single-thread microbenchmark.",
        "rows": rows,
        "source_hashes": source_hashes,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def export_reproducibility_mapping_gpt6(
    loaded_stage3: list[dict], verified_src: dict[str, str], out_path: Path
) -> dict:
    starters_sorted = sorted(_ACTION_CLAUSE_STARTERS)
    assert len(starters_sorted) == 31, f"Expected 31 starters, got {len(starters_sorted)}"
    by_key = {c["spec"]["key"]: c for c in loaded_stage3}
    agg_full = compute_six_backbone_aggregates(loaded_stage3)
    agg_full["observation_unit_note"] = (
        "2,475 denotes descriptive backbone-seed-episode observations per method "
        "(N_b=495 on DeepSeek-V4.1-Flash, GLM-5.3-Flash, and GPT-6-Luna; "
        "N_b=330 on Qwen3-14B-AWQ, Gemma-4-12B-it, and Qwen3-8B), not independent unique tasks."
    )
    agg_displayed_7 = {
        "displayed_main_table_episodes": sum(
            len(NO_ADDITIVE_METHODS) * int(c["methods"]["comep"]["episodes_pooled"])
            for c in loaded_stage3
        ),
        "full_campaign_episodes": agg_full["total_main_benchmark_episodes"],
        "backbone_seed_episode_observations_per_method": agg_full[
            "backbone_seed_episode_observations_per_method"
        ],
        "observation_unit_note": agg_full["observation_unit_note"],
        "methods": {m: agg_full["methods"][m] for m in NO_ADDITIVE_METHODS},
    }

    payload = {
        "kind": "reproducibility_mapping_gpt6_luna",
        "description": "Exact mapping from paper mathematical symbols, hyperparameter names, and the retrospectively selected six-backbone campaign subset (DeepSeek-V4.1-Flash, GLM-5.3-Flash, GPT-6-Luna, Qwen3-14B-AWQ, Gemma-4-12B-it, Qwen3-8B) to codebase symbols, CLI flags, status codes, current exporter-compatible verified source hashes, and archived output directories.",
        "verified_current_exporter_compatible_source_hashes": verified_src,
        "verified_source_hashes_scope_note": "SHA256 digests of the current exporter-compatible comep/*.py files verified identically across both worktrees (matching the exact frozen execution snapshot of outputs/stage3-gpt6-luna-v22; earlier historical campaigns were executed under their own archived per-run worker_spec.json / manifest.json source snapshots before later client.py wire adaptations were added).",
        "backbone_subset_selection_disclosure": {
            "selection_type": "retrospective_post_hoc_subset_selection (not a preregistered exclusion)",
            "selected_six_backbone_subset": [
                "deepseek_v41_flash",
                "glm53_flash_v22",
                "gpt6_luna_v22",
                "qwen3_14b_awq_v22",
                "gemma4_12b_it_v22",
                "qwen3_8b",
            ],
            "excluded_completed_campaign": {
                "key": "mimo_v26_flash_v22",
                "backbone_name": "MiMo-V2.6-Flash",
                "output_dir": "outputs/stage3-mimo-v26-flash-v22",
                "stage3_main_summary_sha256": "d73d3e773c98933070cefa629fe67748a58a2e881561f9c52041a6042144ff75",
                "total_runs": 48,
                "total_episodes": 3960,
                "comep_summary": {
                    "alfworld_sr_pct": 20.0,
                    "scienceworld_sr_pct": 52.89,
                    "pooled_sr_pct": 34.95,
                    "boundary_note": "On MiMo-V2.6-Flash, CoMeP achieved 20.0% on ALFWorld, 52.9% on ScienceWorld, and 34.9% pooled, trailing leading pruning-baseline pooled point estimates (EvoMemGuard-style 41.0%, Individual Credit 40.4%); archive retained intact on disk.",
                },
            },
            "aggregate_scope_note": "The 19,800 full-campaign / 17,325 displayed main-table episode totals and the 53.90% episode-weighted / 50.08% equal-backbone macro aggregates summarize the selected six-backbone subset above, not every completed Stage-3 campaign.",
        },
        "symbol_to_code_mapping": {
            "B_ret": {"code_symbol": "retrieval_tokens", "stage3_4a": 90, "stage1_diag": 64, "stage4b_4cell": 120, "tokenizer": "cl100k_base"},
            "B_bank": {"code_symbol": "bank_tokens", "stage3_4a": 560, "stage1_diag": 440, "exceptions": "Unbounded in Full-Memory and No-Memory"},
            "K": {"code_symbol": "max_items", "stage3_4a": 3, "stage1_diag": 2, "stage4b_4cell": 2},
            "M_bank": {"code_symbol": "max_bank_items", "stage3_4a": 22, "stage1_diag": 18, "exceptions": "Unbounded in Full-Memory"},
            "mu_0, lambda_theta": {"code_symbol": ["prior_mean", "prior_precision"], "stage3_4a": [0.25, 1.0], "stage1_diag": [0.25, 1.0], "notes": "v2_stack=True in comep and additive; 0.0 prior mean in v1 SGD"},
            "lambda_s, lambda_1,0": {"code_symbol": ["pair_prior_precision", "pair_l1"], "stage3_4a": [1.2, 0.01], "stage1_diag": [1.2, 0.01], "effective_l1_lambda_1": 0.05},
            "sigma_0^2": {"code_symbol": "sampling_scale", "stage3_4a": 0.30, "stage1_diag": 0.30},
            "beta_ucb, beta_pair": {"code_symbol": ["ucb_scale", "pair_ucb_scale"], "stage3_4a": [0.55, 0.25], "stage1_diag": [0.65, 0.55], "ablation_zero": "beta_pair=0 in comep-no-pair-optimism"},
            "M, beta_prune": {"code_symbol": ["top_pair_candidates", "pruning_uncertainty"], "stage3_4a": [6, 0.25], "stage1_diag": [8, 0.25]},
            "epsilon_0": {"code_symbol": "exploration", "stage3_4a": 0.08, "stage1_diag": 0.10, "decay_schedule": "epsilon_0 / (1 + 0.08 * N_obs)"},
            "rho_ret, rho_prune": {"code_symbol": "redundancy", "stage3_4a": 0.15, "stage1_diag": 0.15, "stage4b_4cell": 0.0},
            "delta_conf, z": {"code_symbol": ["conflict_threshold", "conflict_z"], "stage3_4a": [0.20, 1.645], "stage1_diag": [0.20, 1.00]},
            "n_min": {"code_symbol": "min_pair_observations", "stage3_4a": 3, "stage1_diag": 3},
            "kappa_stop": {"code_symbol": "min_prob_gain_per_token", "stage3_4a": 0.002, "stage1_diag": 0.003, "ablation_zero": "0.0 in comep-no-early-stop"},
            "W, P_max": {"code_symbol": ["history_window", "max_active_pairs"], "stage3_4a": [600, 48], "stage1_diag": [600, 48]},
            "tau_nli": {"code_symbol": "nli_conflict_threshold", "stage3_4a": 0.55, "stage1_diag": 0.55, "nli_model": "DeBERTa-v3-base-mnli-fever-anli"},
            "H_max, H_hist": {"code_symbol": ["max_steps", "history_turns"], "alfworld": [30, 8], "scienceworld": [35, 8], "distilled_rule_tokens": "15-32 cl100k_base tokens"},
            "G_t(i,j), retries, cache": {"code_symbol": ["warmup_gate", "max_format_retries", "use_shared_cache"], "definition": "I[N_{c_t}>=5]*I[n_i>=2]*I[n_j>=2]"},
            "temperature, max_output_tokens": {"code_symbol": ["temperature", "max_output_tokens"], "stage3_4a": [0.0, 512]},
            "c_0": {"code_symbol": "\"default\"", "description": "Default context label when no specific family is assigned"},
            "P_t": {"code_symbol": "active_pairs in MemoryManager._refit_laplace (comep/core.py:741)", "description": "Fitted pair columns in current regression refit (capped at P_max=48)"},
            "P_t^stored": {"code_symbol": "self.pairs", "description": "All active-item pairs holding a stored parameter record"},
            "E_syn, E_pos, E_rel": {"code_symbol": ["has_syn", "tested_positive", "high_rel_complement"], "location": "comep/core.py:652-664"},
        },
        "benchmarks_and_splits": {
            "alfworld": {
                "env_class": "AlfredTWEnv",
                "splits": {"train": 36, "dev": 24, "test_valid_unseen": 134},
                "warmstart_pool": "configs/alfworld_warmstart_pool.json (11 modular sub-skills + 7 failure heuristics)",
            },
            "scienceworld": {
                "families": [
                    "find-living-thing",
                    "measure-melting-point-known-substance",
                    "test-conductivity",
                    "use-thermometer",
                    "inclined-plane-determine-angle",
                ],
                "splits": {"train": 30, "dev": 25, "test": 100},
                "warmstart_pool": "configs/scienceworld_warmstart_pool.json (10 modular sub-skills + 8 failure heuristics)",
            },
            "clause_split_rule_comep_core_py_15_31": {
                "function": "_split_scope_action (comep/core.py:23-31)",
                "logic": "Split rule text on commas into non-empty stripped clauses. If at least 2 clauses exist, scan clauses after the first (idx >= 1) and split at the first clause whose first word (lowercased) is in _ACTION_CLAUSE_STARTERS; if no subsequent clause matches, split at the first comma boundary (parts[0] vs. rest); if no comma exists, return (text, text).",
                "action_clause_starters_count": len(starters_sorted),
                "action_clause_starters_sorted": starters_sorted,
            },
            "sentence_encoder": "BAAI/bge-small-en-v1.5",
        },
        "expel_derived_online_protocol": {
            "resolved_ordinary_distillation": "In campaign worker spec resolution (comep/campaign.py:258), standard post-episode distillation is resolved to False when method_label is expel-derived-online, even if the general distill option is True.",
            "trigger_modes": "Although code supports same-task opposite-outcome pair reflection and success_batch reflection (>=2 new successes), in the saved logs of all six Stage 3 campaigns every submitted LLM critic call was triggered in success_batch mode.",
            "prompt_byte_limit": 20000,
            "output_format_and_strictness": "Line-based operations (AGREE <n>: <existing rule>., REMOVE <n>: <existing rule>., EDIT <n>: <new rule>., ADD: <new rule>., each rule ending with a period) or literal NO_CHANGE, executed with strict_format=False (comep/campaign.py:158, comep/critic.py:138-163, comep/insights.py:7-109); malformed lines are ignored while accepted operations from other valid lines still apply.",
            "status_code_meanings": {
                "applied": "LLM critic call submitted and completed without length truncation, producing at least one valid accepted line operation (AGREE/REMOVE/EDIT/ADD) applied to the InsightBank (malformed lines or rejected operations alongside accepted ones are recorded in operations metadata).",
                "skipped_truncated": "LLM critic call submitted, but finish_reason=='length' at max_output_tokens; under strict_format=False no bank edits are applied.",
                "skipped_malformed": "LLM critic call submitted and completed without length truncation and was not literal NO_CHANGE, but yielded zero valid accepted line operations (either no parseable operation lines or all parsed line operations had rejected targets such as existing_substring, invalid_edit_index, or no_matching_text).",
                "no_change": "LLM critic call submitted and completed, returning literal NO_CHANGE (after stripping optional outer code fences under strict_format=False).",
                "skipped_oversize / skipped_oversize_prompt": "Reflection opportunity skipped prior to API submission because the serialized trajectory reflection prompt exceeded the 20,000-byte limit (0 LLM calls made).",
            },
        },
        "six_backbone_displayed_seven_method_aggregates": agg_displayed_7,
        "six_backbone_full_eight_method_aggregates": agg_full,
        "campaign_directories_and_serving": {
            "stage1_synthetic": "outputs/stage1-gate1-benchmark",
            "stage3_deepseek_v41_flash": {
                "output_dir": "outputs/stage3-main",
                "resolved_dir": by_key["deepseek_v41_flash"]["resolved_dir"],
                "stage3_main_summary_sha256": by_key["deepseek_v41_flash"]["summary_sha256"],
                "endpoint_model": "deepseek-v4.1-flash",
                "settings": {"thinking": "disabled", "temperature": 0.0, "warmup_gate": False, "max_format_retries": 0, "use_shared_cache": False},
            },
            "stage3_glm53_flash_v22": {
                "output_dir": "outputs/stage3-glm53-flash-v22",
                "resolved_dir": by_key["glm53_flash_v22"]["resolved_dir"],
                "stage3_main_summary_sha256": by_key["glm53_flash_v22"]["summary_sha256"],
                "gateway_provenance_sha256": by_key["glm53_flash_v22"]["spec"]["pinned_hashes"]["gateway_provenance.json"],
                "orchestrator_retry_ledger_sha256": by_key["glm53_flash_v22"]["spec"]["pinned_hashes"]["orchestrator_retry_ledger.json"],
                "endpoint_model": "opencode-go-2/glm-5.3-flash",
                "vendor_canonical_id": "zhipuai/glm-5.3-flash",
                "gateway": {"software": "magpie 0.1.4366", "provider_id": "opencode-go-2", "noStandIn": True, "cli_fallback_disabled": True},
                "request_format_adaptation": "Configured thinking={'type': 'disabled'} when thinking='disabled'; 4 accepted live responses across 48 runs reported non-zero reasoning_tokens (778 tokens total: 232+448+81+17).",
                "accounting_audit": {
                    "canonical_logical_calls": 67334,
                    "accepted_live_responses": 57131,
                    "cache_hits": 10203,
                    "physical_http_started": 57132,
                    "physical_http_ok": 57131,
                    "physical_http_discarded_invalid_reply": 1,
                    "observed_action_format_retries": 185,
                    "observed_distillation_format_retries": 0,
                    "usage_complete_runs": 48,
                    "http_usage_complete_runs": 48,
                    "billing_note": "OpenCode Go subscription accounting; configured prices are uncached catalog equivalents, not actual billed charges.",
                },
                "settings": {"warmup_gate": True, "max_format_retries": 2, "use_shared_cache": True, "observed_format_retries": 185},
            },
            "stage3_gpt6_luna_v22": {
                "output_dir": "outputs/stage3-gpt6-luna-v22",
                "resolved_dir": by_key["gpt6_luna_v22"]["resolved_dir"],
                "stage3_main_summary_sha256": by_key["gpt6_luna_v22"]["summary_sha256"],
                "gateway_provenance_sha256": by_key["gpt6_luna_v22"]["spec"]["pinned_hashes"]["gateway_provenance.json"],
                "alfworld_grid_summary_sha256": by_key["gpt6_luna_v22"]["spec"]["pinned_hashes"]["alfworld/grid_summary.json"],
                "scienceworld_grid_summary_sha256": by_key["gpt6_luna_v22"]["spec"]["pinned_hashes"]["scienceworld/grid_summary.json"],
                "orchestrator_retry_ledger_sha256": by_key["gpt6_luna_v22"]["spec"]["pinned_hashes"]["orchestrator_retry_ledger.json"],
                "endpoint_model": "gpt-6-luna",
                "vendor_canonical_id": "openai/gpt-6-luna",
                "endpoint": {"provider": "openai_official_direct", "base_url": "https://api.openai.com/v1", "no_redirect_handler": True, "proxy_env_stripped": True},
                "request_format_adaptation": "Endpoint-preflight-verified official OpenAI wire adaptation in comep/client.py on https://api.openai.com/v1 -> gpt-6-luna: sends temperature=0, max_completion_tokens=512, and reasoning_effort='none' (without max_tokens or thinking) when thinking='disabled'.",
                "accounting_audit": {
                    "single_dispatch_runs": 48,
                    "resumed_runs": 0,
                    "rolled_back_logical_events": 0,
                    "transient_orchestrator_retries": 0,
                    "canonical_logical_calls": 80798,
                    "accepted_live_responses": 72019,
                    "cache_hits": 8779,
                    "physical_http_started": 72020,
                    "physical_http_finished": 72020,
                    "physical_http_ok": 72019,
                    "physical_http_discarded_invalid_reply": 1,
                    "transport_error_retries": 0,
                    "canonical_finish_reason_stop": 80798,
                    "canonical_finish_reason_length": 0,
                    "observed_action_format_retries": 18,
                    "observed_distillation_format_retries": 0,
                    "observed_action_parse_fallbacks": 0,
                    "explicit_zero_reasoning_tokens_saved_unique_responses": 72019,
                    "explicit_zero_reasoning_tokens_logical_events": 80798,
                    "unverified_discarded_physical_responses": 1,
                    "provider_reported_physical_http_prompt_tokens": 76949103,
                    "provider_reported_physical_http_completion_tokens": 1374548,
                    "provider_reported_physical_http_total_tokens": 78323651,
                    "usage_complete_runs": 48,
                    "http_usage_complete_runs": 48,
                    "total_provider_reported_cost_usd": None,
                    "billing_note": "Official OpenAI API provider-reported token usage only; per-million prices were left unset (None) so all monetary fields remain null without catalog-price or prompt-caching discount assumptions.",
                },
                "settings": {"warmup_gate": True, "max_format_retries": 2, "use_shared_cache": True, "observed_format_retries": 18},
            },
            "stage3_qwen3_14b_awq": {
                "output_dir": "outputs/stage3-qwen3-14b-v22",
                "resolved_dir": by_key["qwen3_14b_awq_v22"]["resolved_dir"],
                "stage3_main_summary_sha256": by_key["qwen3_14b_awq_v22"]["summary_sha256"],
                "hf_checkpoint": "Qwen/Qwen3-14B-AWQ",
                "system_fingerprint": "vllm-0.33.0-fa57fb2b",
                "vllm_flags": ["--quantization", "awq_marlin", "--dtype", "half", "--max-model-len", "8192", "--gpu-memory-utilization", "0.90", "--enable-prefix-caching", "--default-chat-template-kwargs", "{\"enable_thinking\": false}"],
                "settings": {"warmup_gate": True, "max_format_retries": 2, "use_shared_cache": True, "observed_format_retries": 2},
            },
            "stage3_gemma4_12b_it": {
                "output_dir": "outputs/stage3-gemma4-12b",
                "resolved_dir": by_key["gemma4_12b_it_v22"]["resolved_dir"],
                "stage3_main_summary_sha256": by_key["gemma4_12b_it_v22"]["summary_sha256"],
                "served_model_name": "google/gemma-4-12B-it",
                "hf_checkpoint": "google/gemma-4-12B-it-qat-w4a16-ct",
                "system_fingerprint": "vllm-0.33.0-f29d5c03",
                "vllm_flags": ["--max-model-len", "8192", "--gpu-memory-utilization", "0.90", "--enable-prefix-caching", "--default-chat-template-kwargs", "{\"enable_thinking\": false}"],
                "proxy_adaptation": {
                    "script": "paper_artifacts/deployment_provenance/gemma4_12b_proxy.py",
                    "logit_bias": {"98": -100, "100": -100, "101": -100, "45518": -100},
                    "operator_reported_tokens": {"98": "<|think|>", "100": "<|channel>", "101": "<channel|>", "45518": "thought"},
                },
                "settings": {"warmup_gate": True, "max_format_retries": 2, "use_shared_cache": True, "observed_format_retries": 0},
            },
            "stage3_qwen3_8b": {
                "output_dir": "outputs/stage3-qwen3-8b",
                "resolved_dir": by_key["qwen3_8b"]["resolved_dir"],
                "stage3_main_summary_sha256": by_key["qwen3_8b"]["summary_sha256"],
                "hf_checkpoint": "Qwen/Qwen3-8B",
                "settings": {"dtype": "bfloat16", "thinking": "disabled", "warmup_gate": False, "max_format_retries": 0, "use_shared_cache": False},
            },
            "stage4a_ablations": "outputs/stage4-ablations-v22",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def export_deployment_audit_gpt6(
    loaded_stage3: list[dict], repo_prov_dir: Path, out_path: Path
) -> dict:
    archived_names = [
        "gemma4_12b_proxy.log",
        "gemma4_12b_proxy.py",
        "inspect_gemma4_tok.py",
        "print_gemma4_template.py",
        "qwen14b_v22_proxy.py",
        "setup_qwen14b_l4.py",
        "start_vllm_gemma4_12b.py",
        "watchdog_qwen14b_l4.log",
        "watchdog_qwen14b_l4.py",
    ]
    archived_files = {}
    for fn in archived_names:
        p = repo_prov_dir / fn
        if not p.is_file():
            raise FileNotFoundError(f"Missing archived provenance file {p}")
        archived_files[fn] = {
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }

    by_key = {c["spec"]["key"]: c for c in loaded_stage3}
    glm_base = Path(by_key["glm53_flash_v22"]["resolved_dir"])
    gpt_base = Path(by_key["gpt6_luna_v22"]["resolved_dir"])
    glm_prov = json.loads((glm_base / "gateway_provenance.json").read_text(encoding="utf-8"))
    gpt_prov = json.loads((gpt_base / "gateway_provenance.json").read_text(encoding="utf-8"))

    payload = {
        "kind": "additional_backbone_deployment_provenance_audit_gpt6_luna",
        "models": {
            "qwen3_14b_awq_v22": {
                "output_dir": "outputs/stage3-qwen3-14b-v22",
                "hf_model_id": "Qwen/Qwen3-14B-AWQ",
                "system_fingerprint_in_api_logs": "vllm-0.33.0-fa57fb2b",
                "vllm_launch_flags": [
                    "--model",
                    "Qwen/Qwen3-14B-AWQ",
                    "--quantization",
                    "awq_marlin",
                    "--dtype",
                    "half",
                    "--max-model-len",
                    "8192",
                    "--gpu-memory-utilization",
                    "0.90",
                    "--enable-prefix-caching",
                    "--default-chat-template-kwargs",
                    "{\"enable_thinking\": false}",
                ],
                "proxy_adaptation": "Pass-through HTTP reverse proxy (qwen14b_v22_proxy.py) to cloudflared tunnel; thinking disabled via client payload chat_template_kwargs={\"enable_thinking\": False}.",
                "total_runs": 48,
                "total_episodes": 2640,
                "unique_alfworld_tasks_across_3_seeds": by_key["qwen3_14b_awq_v22"]["unique_alfworld_tasks"],
                "unique_scienceworld_tasks_across_3_seeds": by_key["qwen3_14b_awq_v22"]["unique_scienceworld_tasks"],
                "format_retry_calls": by_key["qwen3_14b_awq_v22"]["total_format_retry_calls"],
            },
            "gemma4_12b_it_v22": {
                "output_dir": "outputs/stage3-gemma4-12b",
                "served_model_name": "google/gemma-4-12B-it",
                "weights_repo_id": "google/gemma-4-12B-it-qat-w4a16-ct",
                "system_fingerprint_in_api_logs": "vllm-0.33.0-f29d5c03",
                "vllm_launch_flags": [
                    "--model",
                    "google/gemma-4-12B-it-qat-w4a16-ct",
                    "--served-model-name",
                    "google/gemma-4-12B-it",
                    "google/gemma-4-12B-it-qat-w4a16-ct",
                    "--max-model-len",
                    "8192",
                    "--gpu-memory-utilization",
                    "0.90",
                    "--enable-prefix-caching",
                    "--default-chat-template-kwargs",
                    "{\"enable_thinking\": false}",
                ],
                "note_on_dtype": "No explicit --dtype flag was passed in start_vllm_gemma4_12b.py; vLLM auto-resolved dtype from the compressed-tensors W4A16 QAT checkpoint config.",
                "proxy_decoding_adaptation": {
                    "script": "gemma4_12b_proxy.py",
                    "injected_fields": {
                        "chat_template_kwargs": {"enable_thinking": False},
                        "logit_bias": {"98": -100, "100": -100, "101": -100, "45518": -100},
                    },
                    "token_id_mapping": {
                        "98": "<|think|> (operator-reported control/channel token from tokenizer inspection)",
                        "100": "<|channel> (operator-reported control/channel token from tokenizer inspection)",
                        "101": "<channel|> (operator-reported control/channel token from tokenizer inspection)",
                        "45518": "thought (operator-reported lowercase text token from tokenizer inspection)",
                    },
                    "scope": "The archived proxy implementation applies this adaptation uniformly to POST /v1/chat/completions requests (action, distillation, and critic calls).",
                    "fingerprint_note": "Because injection occurred in the local reverse proxy downstream of ChatClient, spec_fingerprint and request-cache keys do not include logit_bias. The operators report that the proxy script remained unchanged throughout the campaign; this is not independently attested by per-request proxy-version records.",
                },
                "total_runs": 48,
                "total_episodes": 2640,
                "unique_alfworld_tasks_across_3_seeds": by_key["gemma4_12b_it_v22"]["unique_alfworld_tasks"],
                "unique_scienceworld_tasks_across_3_seeds": by_key["gemma4_12b_it_v22"]["unique_scienceworld_tasks"],
                "format_retry_calls": by_key["gemma4_12b_it_v22"]["total_format_retry_calls"],
            },
            "glm53_flash_v22": {
                "output_dir": "outputs/stage3-glm53-flash-v22",
                "client_model_id": glm_prov["frozen_invariants"]["client_model_id"],
                "vendor_canonical_id": glm_prov["frozen_invariants"]["vendor_canonical_id"],
                "gateway_software": glm_prov["gateway"]["software"],
                "gateway_binary_sha256": glm_prov["gateway"]["binary_sha256"],
                "listen_base_url": glm_prov["gateway"]["listen_base_url"],
                "provider_id": glm_prov["gateway"]["provider_id"],
                "provider_config_sha256": glm_prov["gateway"]["provider_config_sha256"],
                "provider_api_key": "<REDACTED>",
                "no_stand_in_verified": bool(glm_prov["gateway"]["no_stand_in_verified"]),
                "cli_no_fallback_verified": bool(glm_prov["gateway"]["cli_no_fallback_verified"]),
                "request_format_adaptation": glm_prov["request_format_adaptation"],
                "reasoning_token_audit": "Configured with thinking={'type': 'disabled'}, though 4 accepted live responses reported non-zero reasoning_tokens totaling 778 tokens (232+448+81+17 across comep_seed0 ALFWorld call 882, evomemguard_seed2 ScienceWorld call 843, and no-memory_seed1 ScienceWorld calls 807 & 809).",
                "billing_and_price_semantics": glm_prov["billing_and_price_semantics"]["accounting_note"],
                "stage3_main_summary_sha256": by_key["glm53_flash_v22"]["summary_sha256"],
                "gateway_provenance_sha256": sha256_file(glm_base / "gateway_provenance.json"),
                "orchestrator_retry_ledger_sha256": sha256_file(glm_base / "orchestrator_retry_ledger.json"),
                "total_runs": 48,
                "total_episodes": 3960,
                "unique_alfworld_tasks_across_3_seeds": by_key["glm53_flash_v22"]["unique_alfworld_tasks"],
                "unique_scienceworld_tasks_across_3_seeds": by_key["glm53_flash_v22"]["unique_scienceworld_tasks"],
                "format_retry_calls": by_key["glm53_flash_v22"]["total_format_retry_calls"],
            },
            "gpt6_luna_v22": {
                "output_dir": "outputs/stage3-gpt6-luna-v22",
                "client_model_id": gpt_prov["frozen_invariants"]["client_model_id"],
                "vendor_canonical_id": gpt_prov["frozen_invariants"]["vendor_canonical_id"],
                "endpoint_provider": gpt_prov["endpoint"]["provider"],
                "endpoint_base_url": gpt_prov["endpoint"]["base_url"],
                "direct_https_no_proxy_verified": bool(gpt_prov["frozen_invariants"]["direct_https_no_proxy_verified"]),
                "no_redirect_handler": bool(gpt_prov["endpoint"]["no_redirect_handler"]),
                "proxy_env_stripped": list(gpt_prov["endpoint"]["proxy_env_stripped"]),
                "provider_api_key": "<REDACTED>",
                "orchestrator_script_sha256": gpt_prov["frozen_invariants"]["orchestrator_script_sha256"],
                "preflight_sha256": gpt_prov["frozen_invariants"]["preflight_sha256"],
                "request_format_adaptation": gpt_prov["request_format_adaptation"],
                "reasoning_token_audit": "All 72,019 saved unique accepted responses (80,798 overlapping canonical logical events including 8,779 cache hits) explicitly report reasoning_tokens=0, with 0 non-zero and 0 missing entries; out of 72,020 physical HTTP responses, 1 unsaved discarded_invalid_reply response does not persist completion_tokens_details and is recorded as unverified.",
                "dispatch_and_http_audit": "48/48 single-dispatch runs (0 resumes, 0 rolled-back logical events, 0 transient orchestrator retries, 0 transport-error retries). 80,798 canonical logical calls = 72,019 accepted live responses + 8,779 shared-cache hits; physical HTTP started=finished=72,020 = 72,019 ok + 1 discarded_invalid_reply content retry; provider-reported physical HTTP token usage = 76,949,103 prompt + 1,374,548 completion = 78,323,651 total tokens (48/48 runs usage_complete=true and http_usage_complete=true; all monetary fields null).",
                "billing_and_price_semantics": gpt_prov["billing_and_price_semantics"]["accounting_note"],
                "stage3_main_summary_sha256": by_key["gpt6_luna_v22"]["summary_sha256"],
                "gateway_provenance_sha256": sha256_file(gpt_base / "gateway_provenance.json"),
                "alfworld_grid_summary_sha256": sha256_file(gpt_base / "alfworld" / "grid_summary.json"),
                "scienceworld_grid_summary_sha256": sha256_file(gpt_base / "scienceworld" / "grid_summary.json"),
                "orchestrator_retry_ledger_sha256": sha256_file(gpt_base / "orchestrator_retry_ledger.json"),
                "total_runs": 48,
                "total_episodes": 3960,
                "unique_alfworld_tasks_across_3_seeds": by_key["gpt6_luna_v22"]["unique_alfworld_tasks"],
                "unique_scienceworld_tasks_across_3_seeds": by_key["gpt6_luna_v22"]["unique_scienceworld_tasks"],
                "format_retry_calls": by_key["gpt6_luna_v22"]["total_format_retry_calls"],
            },
        },
        "archived_files": archived_files,
    }
    serialized = json.dumps(payload, indent=2) + "\n"
    assert "sk-" not in serialized, "Secret-like token found in serialized deployment audit!"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(serialized, encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        default=str(DEFAULT_REPO_ROOT),
        help="Target root directory (staging directory during pre-review, or worktree root upon promotion).",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Regenerate only the three provenance/audit JSON files while leaving existing figure files untouched.",
    )
    args = parser.parse_args()
    out_root = Path(args.out_root).resolve()

    verified_src = verify_source_hashes()
    loaded_stage3 = [verify_and_load_stage3_campaign(spec) for spec in STAGE3_GPT6_CAMPAIGN_SPECS]
    by_key = {c["spec"]["key"]: c for c in loaded_stage3}

    if not args.json_only:
        export_journal_figure_8(
            by_key["gpt6_luna_v22"]["methods"],
            out_root / "paper" / "neurocomputing" / "figures" / "Figure_8_gpt6_luna.pdf",
        )
        export_arxiv_six_backbone_figure(
            loaded_stage3,
            [
                out_root / "paper" / "figures",
                out_root / "paper_artifacts" / "figures",
            ],
        )
    export_routing_latency_audit_gpt6(
        loaded_stage3,
        out_root / "paper_artifacts" / "routing_latency_audit_gpt6_luna.json",
    )
    export_reproducibility_mapping_gpt6(
        loaded_stage3,
        verified_src,
        out_root / "paper_artifacts" / "reproducibility_mapping_gpt6_luna.json",
    )
    export_deployment_audit_gpt6(
        loaded_stage3,
        DEFAULT_REPO_ROOT / "paper_artifacts" / "deployment_provenance",
        out_root / "paper_artifacts" / "deployment_provenance" / "deployment_audit_gpt6_luna.json",
    )
    print(f"Exported all GPT-6-Luna figures and provenance JSONs cleanly under {out_root}")


if __name__ == "__main__":
    main()
