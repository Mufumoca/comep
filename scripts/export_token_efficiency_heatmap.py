"""Provenance-verified price-agnostic API token-efficiency heatmap exporter for DeepSeek-V4.1-Flash and GLM-5.3-Flash.

Exports a single publication-grade 7x6 heatmap matching the original CoMeP manuscript style
(STIXGeneral, pdf.fonttype=42, CoMeP highlight #900000), showing relative change in finalized
logical API tokens per win versus Full-Memory (Top-k) = 0% on a shared symmetric [-60%, +60%] scale.
"""
from __future__ import annotations

import hashlib
import json
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
    STAGE3_CAMPAIGN_SPECS,
    resolve_and_verify_campaign_dir,
    sha256_file,
    verify_source_hashes,
)

METHODS_7: tuple[tuple[str, str], ...] = (
    ("no-memory", "No-Memory"),
    ("full-memory", "Full-Memory (Top-$k$)"),
    ("individual", "Individual Credit"),
    ("mmr", "MMR-style Diversity"),
    ("expel-derived-online", "ExpeL-derived Online"),
    ("evomemguard", "EvoMemGuard-style"),
    ("comep", "CoMeP (Ours)"),
)

EXPECTED_EPISODES: dict[str, int] = {"alfworld": 90, "scienceworld": 75}
REQUIRE_FINGERPRINT: dict[str, bool] = {
    "deepseek_v41_flash": False,  # Documented legacy archive format lacking spec_fingerprint
    "glm53_flash_v22": True,      # Mandatory non-empty matching fingerprint across summary/manifest/worker_spec
}


def load_and_reconcile_backbone(spec: dict) -> dict:
    key = spec["key"]
    req_files = list(spec["pinned_hashes"].keys()) + [
        f"{env}/{m[0]}_seed{s}/{fname}"
        for env in ("alfworld", "scienceworld")
        for m in METHODS_7
        for s in (0, 1, 2)
        for fname in ("summary.json", "episodes.jsonl", "api.jsonl", "manifest.json", "worker_spec.json")
    ]
    _, cdir = resolve_and_verify_campaign_dir(spec["rel_dir"], req_files, spec["pinned_hashes"])

    consumed_hashes: dict[str, str] = {
        fname: sha256_file(cdir / fname) for fname in spec["pinned_hashes"]
    }
    task_ref: dict[tuple[str, int], list[str]] = {}
    fp_checked = 0
    fp_legacy_absent = 0

    per_method_env: dict[str, dict[str, list[dict]]] = {
        m[0]: {"alfworld": [], "scienceworld": []} for m in METHODS_7
    }

    for env in ("alfworld", "scienceworld"):
        exp_eps = EXPECTED_EPISODES[env]
        for seed in (0, 1, 2):
            for method_key, _ in METHODS_7:
                run_dir = cdir / env / f"{method_key}_seed{seed}"
                if not run_dir.is_dir():
                    raise FileNotFoundError(f"Missing run directory: {run_dir}")

                for fname in ("summary.json", "episodes.jsonl", "api.jsonl", "manifest.json", "worker_spec.json"):
                    fpath = run_dir / fname
                    if not fpath.is_file():
                        raise FileNotFoundError(f"Missing run artifact: {fpath}")
                    consumed_hashes[f"{env}/{method_key}_seed{seed}/{fname}"] = sha256_file(fpath)
                http_path = run_dir / "api_http_attempts.jsonl"
                if http_path.is_file():
                    consumed_hashes[f"{env}/{method_key}_seed{seed}/api_http_attempts.jsonl"] = sha256_file(http_path)

                summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
                manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
                worker_spec = json.loads((run_dir / "worker_spec.json").read_text(encoding="utf-8"))

                if summary.get("status") != "completed":
                    raise ValueError(f"Incomplete run: {run_dir}")
                if int(summary["completed_episodes"]) != exp_eps:
                    raise ValueError(f"Episode count mismatch in {run_dir}: {summary['completed_episodes']} != {exp_eps}")

                # Strict fingerprint verification
                fps = (
                    summary.get("spec_fingerprint"),
                    manifest.get("spec_fingerprint"),
                    worker_spec.get("spec_fingerprint"),
                )
                if REQUIRE_FINGERPRINT[key]:
                    if not all(isinstance(fp, str) and fp.strip() for fp in fps) or not (fps[0] == fps[1] == fps[2]):
                        raise ValueError(f"Required spec_fingerprint missing/mismatched in {run_dir}: {fps!r}")
                    fp_checked += 1
                else:
                    if any(fp is not None for fp in fps):
                        if not all(isinstance(fp, str) and fp.strip() for fp in fps) or not (fps[0] == fps[1] == fps[2]):
                            raise ValueError(f"spec_fingerprint mismatch in {run_dir}: {fps!r}")
                        fp_checked += 1
                    else:
                        fp_legacy_absent += 1

                # Verify episodes.jsonl task alignment and exact integer wins
                ep_rows = [json.loads(line) for line in (run_dir / "episodes.jsonl").open(encoding="utf-8") if line.strip()]
                if len(ep_rows) != exp_eps:
                    raise ValueError(f"episodes.jsonl length mismatch in {run_dir}: {len(ep_rows)} != {exp_eps}")
                task_ids = [str(r["task_id"]) for r in ep_rows]
                if (env, seed) not in task_ref:
                    task_ref[(env, seed)] = task_ids
                elif task_ref[(env, seed)] != task_ids:
                    raise ValueError(f"Task alignment mismatch in {run_dir}")

                wins = sum(1 for r in ep_rows if int(r["reward"]) == 1)
                if abs(wins / exp_eps - float(summary["success_rate"])) > 1e-9:
                    raise ValueError(f"Win count / SR mismatch in {run_dir}")
                ep_ret_tokens = sum(int(r["retrieval_tokens"]) for r in ep_rows)
                if ep_ret_tokens != int(summary["retrieval_tokens"]):
                    raise ValueError(f"Retrieval tokens mismatch in {run_dir}")

                # Strict API log reconciliation: final occurrence per call ID vs raw append-only
                s_api = summary["api"]
                if s_api.get("usage_complete") is not True:
                    raise ValueError(f"usage_complete is not True in {run_dir}")
                if "http_usage_complete" in s_api and s_api["http_usage_complete"] is not True:
                    raise ValueError(f"http_usage_complete is not True in {run_dir}")

                final_by_call: dict[int, dict] = {}
                max_seen_call = -1
                had_rewind = False
                raw_records = 0
                raw_prompt = 0
                raw_completion = 0
                raw_local_cache_hits = 0

                with (run_dir / "api.jsonl").open(encoding="utf-8") as fh:
                    for line_no, line in enumerate(fh, start=1):
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        cid = rec.get("call")
                        if isinstance(cid, bool) or not isinstance(cid, int):
                            raise ValueError(f"Invalid call ID in {run_dir}/api.jsonl:{line_no}")
                        if cid <= max_seen_call:
                            had_rewind = True
                        else:
                            max_seen_call = cid
                        usage = rec.get("usage")
                        if not isinstance(usage, dict):
                            raise ValueError(f"Missing usage dict in {run_dir}/api.jsonl:{line_no}")
                        pt = usage.get("prompt_tokens")
                        ct = usage.get("completion_tokens")
                        if (
                            isinstance(pt, bool)
                            or not isinstance(pt, int)
                            or pt < 0
                            or isinstance(ct, bool)
                            or not isinstance(ct, int)
                            or ct < 0
                        ):
                            raise ValueError(f"Invalid token usage in {run_dir}/api.jsonl:{line_no}: pt={pt!r}, ct={ct!r}")
                        if rec.get("cache_hit") is True:
                            raw_local_cache_hits += 1
                        raw_records += 1
                        raw_prompt += pt
                        raw_completion += ct
                        final_by_call[cid] = rec

                fin_calls = len(final_by_call)
                fin_prompt = sum(int(r["usage"]["prompt_tokens"]) for r in final_by_call.values())
                fin_completion = sum(int(r["usage"]["completion_tokens"]) for r in final_by_call.values())
                fin_local_cache_hits = sum(1 for r in final_by_call.values() if r.get("cache_hit") is True)

                if fin_calls != int(s_api["attempted_calls"]):
                    raise ValueError(f"Finalized call mismatch in {run_dir}: {fin_calls} != {s_api['attempted_calls']}")
                if fin_prompt != int(s_api["prompt_tokens"]) or fin_completion != int(s_api["completion_tokens"]):
                    raise ValueError(f"Finalized token mismatch in {run_dir}")

                per_method_env[method_key][env].append(
                    {
                        "seed": seed,
                        "episodes": exp_eps,
                        "wins": wins,
                        "step_limit_episodes": int(summary["step_limit_episodes"]),
                        "manager_calls": int(summary["manager_llm_calls"]),
                        "distillation_calls": int(summary["distillation_llm_calls"]),
                        "finalized_calls": fin_calls,
                        "finalized_prompt_tokens": fin_prompt,
                        "finalized_completion_tokens": fin_completion,
                        "finalized_total_tokens": fin_prompt + fin_completion,
                        "finalized_local_cache_hits": fin_local_cache_hits,
                        "raw_records": raw_records,
                        "repeated_records": raw_records - fin_calls,
                        "had_rewind": had_rewind,
                        "raw_prompt_tokens": raw_prompt,
                        "raw_completion_tokens": raw_completion,
                        "raw_total_tokens": raw_prompt + raw_completion,
                        "raw_local_cache_hits": raw_local_cache_hits,
                    }
                )

    if REQUIRE_FINGERPRINT[key] and (fp_checked != 42 or fp_legacy_absent != 0):
        raise ValueError(f"Fingerprint verification incomplete for {key}: checked={fp_checked}, absent={fp_legacy_absent}")

    # Aggregate per env and combined (summing tokens and wins across both environments BEFORE taking ratios)
    def agg_runs(runs: list[dict]) -> dict:
        eps = sum(r["episodes"] for r in runs)
        wins = sum(r["wins"] for r in runs)
        fin_tok = sum(r["finalized_total_tokens"] for r in runs)
        raw_tok = sum(r["raw_total_tokens"] for r in runs)
        return {
            "episodes": eps,
            "wins": wins,
            "sr_pct": 100.0 * wins / eps,
            "step_limit_episodes": sum(r["step_limit_episodes"] for r in runs),
            "manager_calls": sum(r["manager_calls"] for r in runs),
            "distillation_calls": sum(r["distillation_calls"] for r in runs),
            "finalized_calls": sum(r["finalized_calls"] for r in runs),
            "finalized_prompt_tokens": sum(r["finalized_prompt_tokens"] for r in runs),
            "finalized_completion_tokens": sum(r["finalized_completion_tokens"] for r in runs),
            "finalized_total_tokens": fin_tok,
            "finalized_tokens_per_win": fin_tok / wins,
            "finalized_local_cache_hits": sum(r["finalized_local_cache_hits"] for r in runs),
            "raw_records": sum(r["raw_records"] for r in runs),
            "repeated_records": sum(r["repeated_records"] for r in runs),
            "runs_with_rewind": sum(1 for r in runs if r["had_rewind"]),
            "raw_total_tokens": raw_tok,
            "raw_tokens_per_win": raw_tok / wins,
        }

    by_env: dict[str, dict[str, dict]] = {"alfworld": {}, "scienceworld": {}, "combined": {}}
    for m_key, _ in METHODS_7:
        alf_runs = per_method_env[m_key]["alfworld"]
        sci_runs = per_method_env[m_key]["scienceworld"]
        by_env["alfworld"][m_key] = agg_runs(alf_runs)
        by_env["scienceworld"][m_key] = agg_runs(sci_runs)
        by_env["combined"][m_key] = agg_runs(alf_runs + sci_runs)

    # Column-specific normalization vs Full-Memory (Top-k) = 0%
    for env in ("alfworld", "scienceworld", "combined"):
        fm_tpw = by_env[env]["full-memory"]["finalized_tokens_per_win"]
        fm_raw_tpw = by_env[env]["full-memory"]["raw_tokens_per_win"]
        for m_key, _ in METHODS_7:
            d = by_env[env][m_key]
            d["rel_change_vs_full_memory_pct"] = 100.0 * (d["finalized_tokens_per_win"] / fm_tpw - 1.0)
            d["raw_rel_change_vs_full_memory_pct"] = 100.0 * (d["raw_tokens_per_win"] / fm_raw_tpw - 1.0)

    comb = by_env["combined"]
    return {
        "backbone_key": key,
        "backbone_name": spec["backbone_name"],
        "resolved_dir": str(cdir),
        "provenance_audit": {
            "runs_verified": 42,
            "fingerprint_required": REQUIRE_FINGERPRINT[key],
            "fingerprints_checked_and_matched": fp_checked,
            "fingerprints_absent_legacy_runs": fp_legacy_absent,
            "total_finalized_calls_7methods": sum(comb[m[0]]["finalized_calls"] for m in METHODS_7),
            "total_raw_api_records_7methods": sum(comb[m[0]]["raw_records"] for m in METHODS_7),
            "total_repeated_call_records_7methods": sum(comb[m[0]]["repeated_records"] for m in METHODS_7),
            "runs_with_counter_rewind_7methods": sum(comb[m[0]]["runs_with_rewind"] for m in METHODS_7),
            "total_local_response_cache_hits_7methods": sum(comb[m[0]]["finalized_local_cache_hits"] for m in METHODS_7),
            "consumed_file_sha256_count": len(consumed_hashes),
            "consumed_file_sha256_digest": hashlib.sha256(
                json.dumps(consumed_hashes, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "consumed_file_hashes": consumed_hashes,
        },
        "by_env": by_env,
    }


def render_heatmap(ds: dict, glm: dict, out_pdf: Path, out_png: Path) -> None:
    cols = [
        ("ALFWorld", ds["by_env"]["alfworld"]),
        ("ScienceWorld", ds["by_env"]["scienceworld"]),
        ("Combined", ds["by_env"]["combined"]),
        ("ALFWorld", glm["by_env"]["alfworld"]),
        ("ScienceWorld", glm["by_env"]["scienceworld"]),
        ("Combined", glm["by_env"]["combined"]),
    ]
    m_keys = [m[0] for m in METHODS_7]
    row_labels = [m[1] for m in METHODS_7]
    col_labels = [c[0] for c in cols]

    mat = np.array([[col_d[k]["rel_change_vs_full_memory_pct"] for _, col_d in cols] for k in m_keys])
    if float(np.min(mat)) < -60.0 or float(np.max(mat)) > 60.0:
        raise ValueError(f"Matrix values [{np.min(mat)}, {np.max(mat)}] exceed shared [-60, +60] color scale!")

    fig = plt.figure(figsize=(6.8, 3.85))
    ax = fig.add_axes([0.225, 0.215, 0.745, 0.605])

    im = ax.imshow(mat, cmap="RdBu_r", vmin=-60.0, vmax=60.0, aspect="auto")

    for r in range(len(m_keys)):
        for c in range(len(cols)):
            v = mat[r, c]
            txt = "0.0%" if m_keys[r] == "full-memory" else f"{v:+.1f}%"
            txt_color = "white" if abs(v) >= 34.0 else "#1a1a1a"
            ax.text(
                c,
                r,
                txt,
                ha="center",
                va="center",
                fontsize=8.6,
                fontweight="bold" if m_keys[r] == "comep" else "normal",
                color=txt_color,
            )

    # Subtle internal cell gridlines for crisp readability
    ax.set_xticks(np.arange(-0.5, len(cols), 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, len(m_keys), 1.0), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.7, alpha=0.8)
    ax.tick_params(which="minor", length=0)

    # Backbone divider & CoMeP separator matching original paper highlight style
    ax.axvline(2.5, color="#222222", linewidth=1.35)
    ax.axhline(5.5, color="#111111", linewidth=1.45)

    ax.set_yticks(np.arange(len(m_keys)))
    ax.set_yticklabels(row_labels, fontsize=8.8)
    co_tick = ax.get_yticklabels()[-1]
    co_tick.set_fontweight("bold")
    co_tick.set_color("#900000")

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(col_labels, fontsize=8.4)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", pad=3, length=3)
    ax.tick_params(axis="y", pad=4, length=3)

    # Backbone group headers cleanly above the two 3-column blocks
    ax.text(
        1.0,
        1.145,
        "DeepSeek-V4.1-Flash",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=9.4,
        fontweight="bold",
        color="#111111",
    )
    ax.text(
        4.0,
        1.145,
        "GLM-5.3-Flash",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=9.4,
        fontweight="bold",
        color="#111111",
    )
    # Clean horizontal bracket rules under the two group titles
    ax.plot([-0.42, 2.42], [1.125, 1.125], transform=ax.get_xaxis_transform(), color="#333333", linewidth=0.9, clip_on=False)
    ax.plot([2.58, 5.42], [1.125, 1.125], transform=ax.get_xaxis_transform(), color="#333333", linewidth=0.9, clip_on=False)

    for sp in ax.spines.values():
        sp.set_edgecolor("#444444")
        sp.set_linewidth(0.75)

    # Clean horizontal colorbar centered directly under the heatmap axes
    cax = fig.add_axes([0.310, 0.125, 0.575, 0.036])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-60, -40, -20, 0, 20, 40, 60])
    cb.ax.set_xticklabels(["-60%", "-40%", "-20%", "0%", "+20%", "+40%", "+60%"], fontsize=8.0)
    cb.set_label(
        "Relative Change in Finalized API Tokens / Win vs. Full-Memory (Blue = Fewer Tokens)",
        fontsize=8.3,
        labelpad=3,
    )
    cb.outline.set_edgecolor("#444444")
    cb.outline.set_linewidth(0.65)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf")
    fig.savefig(out_png, format="png", dpi=300)
    plt.close(fig)


def main() -> None:
    verify_source_hashes()
    spec_by_key = {s["key"]: s for s in STAGE3_CAMPAIGN_SPECS}
    ds = load_and_reconcile_backbone(spec_by_key["deepseek_v41_flash"])
    glm = load_and_reconcile_backbone(spec_by_key["glm53_flash_v22"])

    # Compute exact unrounded DeepSeek Combined CoMeP reduction vs each of the 6 baselines
    ds_comb = ds["by_env"]["combined"]
    ds_comep_tpw = ds_comb["comep"]["finalized_tokens_per_win"]
    ds_vs_baselines_pct = {
        m_key: 100.0 * (ds_comb[m_key]["finalized_tokens_per_win"] - ds_comep_tpw) / ds_comb[m_key]["finalized_tokens_per_win"]
        for m_key, _ in METHODS_7
        if m_key != "comep"
    }
    min_red = min(ds_vs_baselines_pct.values())
    max_red = max(ds_vs_baselines_pct.values())

    exporter_path = Path(__file__).resolve()
    audit_payload = {
        "schema": "token_efficiency_heatmap_audit_v1",
        "exporter_path": str(exporter_path.relative_to(REPO_ROOT)),
        "exporter_sha256": sha256_file(exporter_path),
        "scope": "7 main-table methods (6 external baselines + CoMeP), 495 paired episodes/method (270 ALFWorld + 225 ScienceWorld across 3 seeds), price-agnostic (no dollar values)",
        "normalization_formula": "Delta_{m,c} = 100 * ((T_{m,c}/W_{m,c}) / (T_{Full-Memory,c}/W_{Full-Memory,c}) - 1), where T is finalized logical prompt+completion tokens and W is integer wins. Combined columns sum T and W across both environments before taking ratios. Each column normalizes against its own Full-Memory reference (0.0%) on a shared symmetric [-60%, +60%] color scale.",
        "accounting_notes": {
            "deepseek_repeated_calls": (
                f"{ds['provenance_audit']['runs_with_counter_rewind_7methods']} of {ds['provenance_audit']['runs_verified']} "
                f"selected DeepSeek runs contain append-only repeated call IDs from counter rewinds "
                f"({ds['provenance_audit']['total_repeated_call_records_7methods']} repeated records across "
                f"{ds['provenance_audit']['total_raw_api_records_7methods']:,} raw records vs. "
                f"{ds['provenance_audit']['total_finalized_calls_7methods']:,} finalized calls). Both finalized and raw append-only "
                f"totals are preserved per method and per column in this audit."
            ),
            "glm_logical_usage": (
                f"GLM-5.3-Flash reports finalized logical usage counting local response-cache hits "
                f"({glm['provenance_audit']['total_local_response_cache_hits_7methods']:,} across the 7 displayed methods) "
                f"at their recorded token usage (logical request totals, not transmitted HTTP usage or a separately executed uncached campaign)."
            ),
            "local_compute_exclusion": "API token accounting excludes local embedding, NLI, routing, and Laplace-fitting compute symmetrically across all methods.",
            "ranking_scope": "All comparisons and rankings are descriptive within the seven-method main-table family.",
        },
        "derived_comparisons": {
            "deepseek_combined_comep_reduction_vs_baselines_pct": ds_vs_baselines_pct,
            "deepseek_combined_comep_reduction_min_pct": min_red,
            "deepseek_combined_comep_reduction_max_pct": max_red,
            "deepseek_combined_comep_reduction_range_1dp": f"{min_red:.1f}%--{max_red:.1f}%",
        },
        "backbones": {
            "deepseek_v41_flash": ds,
            "glm53_flash_v22": glm,
        },
    }

    audit_path = REPO_ROOT / "paper_artifacts" / "token_efficiency_heatmap_audit.json"
    audit_path.write_text(json.dumps(audit_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    art_pdf = REPO_ROOT / "paper_artifacts" / "figures" / "fig_token_efficiency_heatmap.pdf"
    art_png = REPO_ROOT / "paper_artifacts" / "figures" / "fig_token_efficiency_heatmap.png"
    render_heatmap(ds, glm, art_pdf, art_png)

    arxiv_pdf = REPO_ROOT / "paper" / "figures" / "fig_token_efficiency_heatmap.pdf"
    nc_pdf = REPO_ROOT / "paper" / "neurocomputing" / "figures" / "Figure_16.pdf"
    shutil.copy2(art_pdf, arxiv_pdf)
    shutil.copy2(art_pdf, nc_pdf)

    print("SUCCESS: Exported original-style STIXGeneral token-efficiency heatmap & provenance audit.")
    print(f"  DeepSeek Combined CoMeP reduction vs 6 baselines: {min_red:.4f}% .. {max_red:.4f}% ({min_red:.1f}%--{max_red:.1f}%)")
    print(f"  Audit JSON: {audit_path}")
    print(f"  Vector PDFs: {art_pdf}, {arxiv_pdf}, {nc_pdf}")


if __name__ == "__main__":
    main()
