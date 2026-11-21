"""Shared deterministic, provenance-bound campaign source resolution and raw-log verification."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

WORKTREES: tuple[Path, ...] = (
    Path("/root/.paseo/worktrees/1s5ku8sn/lucid-bumblebee"),
    Path("/root/.paseo/worktrees/1s5ku8sn/comep-v2"),
)

MAIN_METHODS: tuple[str, ...] = (
    "no-memory",
    "full-memory",
    "individual",
    "mmr",
    "additive",
    "expel-derived-online",
    "evomemguard",
    "comep",
)

PINNED_SOURCE_HASHES: dict[str, str] = {
    "comep/core.py": "7dcf81a1f8ca00fca00a538d50976c8837bcbd3fc18a6b12bb4e64ee05c616e9",
    "comep/client.py": "59c5576e1e2ff30c224161813635c003089b8cde773507b65b04dff3774088c3",
    "comep/runner.py": "a6fcab27725d66f92180e8f74f109de4d552dbc7a1a1bde5fa5baaaf78264d35",
    "comep/campaign.py": "fb67a2872dae00f36d927e7de20ebb1094b562b208476cf4ac60a00437a743ba",
}

STAGE3_CAMPAIGN_SPECS: tuple[dict, ...] = (
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
        "key": "mimo_v26_flash_v22",
        "label": "Stage 3: MiMo-V2.6-Flash (final gated)",
        "backbone_name": "MiMo-V2.6-Flash",
        "rel_dir": "outputs/stage3-mimo-v26-flash-v22",
        "n_alf": 90,
        "n_sci": 75,
        "warmup_gate": True,
        "pinned_hashes": {
            "stage3_main_summary.json": "d73d3e773c98933070cefa629fe67748a58a2e881561f9c52041a6042144ff75",
            "gateway_provenance.json": "bb7025d9bd44831ef39224b94d714cf00a2887b6d12cd4a24ab235c60bb83ddb",
            "orchestrator_retry_ledger.json": "3ffa9cdc69c2b0f10eb13edbca09089502c17fed90661d9d288323409af2c505",
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

STAGE4_ABLATION_SPEC: dict = {
    "rel_dir": "outputs/stage4-ablations-v22",
    "pinned_hashes": {
        "stage4_ablations_summary.json": "9729d740e59d756ff9c0b89396cff34e15417ec8e6a46cda5840cb3589791b20",
    },
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_hashes(worktrees: tuple[Path, ...] = WORKTREES) -> dict[str, str]:
    verified: dict[str, str] = {}
    existing_trees = [wt for wt in worktrees if wt.is_dir()]
    if not existing_trees:
        raise FileNotFoundError("No configured worktrees exist on disk.")
    for rel_src, expected_sha in PINNED_SOURCE_HASHES.items():
        hashes_seen: dict[str, str] = {}
        for wt in existing_trees:
            p = wt / rel_src
            if not p.is_file():
                raise FileNotFoundError(f"Missing pinned source file: {p}")
            h = sha256_file(p)
            if h != expected_sha:
                raise RuntimeError(
                    f"Source hash mismatch for {p}: expected {expected_sha}, got {h}"
                )
            hashes_seen[str(wt)] = h
        if len(set(hashes_seen.values())) != 1:
            raise RuntimeError(f"Cross-worktree source hash divergence on {rel_src}: {hashes_seen}")
        verified[rel_src] = expected_sha
    return verified


def resolve_and_verify_campaign_dir(
    rel_dir: str,
    required_rel_files: list[str],
    pinned_hashes: dict[str, str] | None = None,
    worktrees: tuple[Path, ...] = WORKTREES,
) -> tuple[Path, Path]:
    """Resolve a complete campaign directory from configured worktrees.

    Fails if any candidate tree contains only a partial set of required files,
    or if multiple worktrees contain the campaign directory with differing file hashes,
    or if any pinned SHA256 hash does not match.
    """
    candidates: list[tuple[Path, Path]] = []
    for wt in worktrees:
        cdir = wt / rel_dir
        if cdir.exists():
            missing = [rf for rf in required_rel_files if not (cdir / rf).is_file()]
            if missing:
                raise RuntimeError(
                    f"Partial campaign directory detected at {cdir}; missing {len(missing)} required files "
                    f"(e.g. {missing[:3]}). Refusing to combine partial campaigns across worktrees."
                )
            candidates.append((wt, cdir))

    if not candidates:
        raise FileNotFoundError(
            f"Campaign directory {rel_dir!r} not found in any configured worktree: {worktrees}"
        )

    if len(candidates) > 1:
        ref_wt, ref_dir = candidates[0]
        for other_wt, other_dir in candidates[1:]:
            for rf in required_rel_files:
                h_ref = sha256_file(ref_dir / rf)
                h_oth = sha256_file(other_dir / rf)
                if h_ref != h_oth:
                    raise RuntimeError(
                        f"Cross-worktree artifact mismatch for {rel_dir}/{rf}: "
                        f"{ref_wt} has {h_ref} vs {other_wt} has {h_oth}"
                    )

    chosen_wt, chosen_dir = candidates[0]
    if pinned_hashes:
        for rf, exp_sha in pinned_hashes.items():
            p = chosen_dir / rf
            if not p.is_file():
                raise FileNotFoundError(f"Missing pinned file {p}")
            got_sha = sha256_file(p)
            if got_sha != exp_sha:
                raise RuntimeError(
                    f"Pinned SHA256 mismatch for {p}: expected {exp_sha}, got {got_sha}"
                )
    return chosen_wt, chosen_dir


def verify_and_load_stage3_campaign(spec: dict) -> dict:
    rel_dir = spec["rel_dir"]
    n_alf = int(spec["n_alf"])
    n_sci = int(spec["n_sci"])
    seeds = (0, 1, 2)

    req_files = ["stage3_main_summary.json"]
    req_files.extend(spec.get("pinned_hashes", {}).keys())
    for env_name in ("alfworld", "scienceworld"):
        for m in MAIN_METHODS:
            for s in seeds:
                req_files.append(f"{env_name}/{m}_seed{s}/summary.json")
                req_files.append(f"{env_name}/{m}_seed{s}/episodes.jsonl")
        for s in seeds:
            req_files.append(f"{env_name}/expel-derived-online_seed{s}/critic.jsonl")
    req_files = sorted(set(req_files))

    wt, base = resolve_and_verify_campaign_dir(
        rel_dir=rel_dir,
        required_rel_files=req_files,
        pinned_hashes=spec.get("pinned_hashes"),
    )

    n_tot_seed = n_alf + n_sci
    n_tot_method = 3 * n_tot_seed

    task_seq_ref: dict[tuple[str, int], list[str]] = {}
    unique_tasks: dict[str, set[str]] = {"alfworld": set(), "scienceworld": set()}
    methods_stats: dict[str, dict] = {}
    format_retry_calls = 0

    for m in MAIN_METHODS:
        seed_pooled_srs: list[float] = []
        seed_alf_srs: list[float] = []
        seed_sci_srs: list[float] = []
        tok_alf_seeds: list[float] = []
        tok_sci_seeds: list[float] = []
        wins_alf_total = 0
        wins_sci_total = 0

        for env_name, n_exp in (("alfworld", n_alf), ("scienceworld", n_sci)):
            for s in seeds:
                run_dir = base / env_name / f"{m}_seed{s}"
                summ = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
                if summ.get("status") != "completed":
                    raise RuntimeError(f"Uncompleted run at {run_dir}: status={summ.get('status')}")
                if int(summ.get("completed_episodes", -1)) != n_exp or int(summ.get("requested_episodes", -1)) != n_exp:
                    raise RuntimeError(f"Episode count mismatch in {run_dir}/summary.json")
                format_retry_calls += int(summ.get("action_format_retry_calls", 0)) + int(
                    summ.get("distillation_format_retry_calls", 0)
                )

                ep_recs = [
                    json.loads(ln)
                    for ln in (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
                    if ln.strip()
                ]
                ep_recs.sort(key=lambda r: int(r["episode"]))
                if [int(r["episode"]) for r in ep_recs] != list(range(n_exp)):
                    raise RuntimeError(f"Non-contiguous episodes in {run_dir}/episodes.jsonl")

                t_seq = [str(r["task_id"]) for r in ep_recs]
                unique_tasks[env_name].update(t_seq)
                ref_key = (env_name, s)
                if ref_key not in task_seq_ref:
                    task_seq_ref[ref_key] = t_seq
                elif task_seq_ref[ref_key] != t_seq:
                    raise RuntimeError(f"Task sequence misalignment in {run_dir} vs {ref_key}")

                wins_run = sum(int(round(float(r["reward"]))) for r in ep_recs)
                exp_wins_from_sr = int(round(float(summ["success_rate"]) * n_exp))
                if wins_run != exp_wins_from_sr:
                    raise RuntimeError(
                        f"Win-count mismatch in {run_dir}: episodes={wins_run} vs summary={exp_wins_from_sr}"
                    )

                if env_name == "alfworld":
                    wins_alf_total += wins_run
                    seed_alf_srs.append((wins_run / n_exp) * 100.0)
                    tok_alf_seeds.append(float(summ["mean_retrieval_tokens"]))
                else:
                    wins_sci_total += wins_run
                    seed_sci_srs.append((wins_run / n_exp) * 100.0)
                    tok_sci_seeds.append(float(summ["mean_retrieval_tokens"]))

        for s_idx in range(3):
            wa = round(seed_alf_srs[s_idx] * n_alf / 100.0)
            ws = round(seed_sci_srs[s_idx] * n_sci / 100.0)
            seed_pooled_srs.append(((wa + ws) / n_tot_seed) * 100.0)

        mean_tok_alf = float(np.mean(tok_alf_seeds))
        mean_tok_sci = float(np.mean(tok_sci_seeds))
        env_macro_tok = 0.5 * (mean_tok_alf + mean_tok_sci)
        ep_pooled_tok = (n_alf * mean_tok_alf + n_sci * mean_tok_sci) / n_tot_seed

        methods_stats[m] = {
            "wins_alfworld": wins_alf_total,
            "episodes_alfworld": 3 * n_alf,
            "wins_scienceworld": wins_sci_total,
            "episodes_scienceworld": 3 * n_sci,
            "wins_pooled": wins_alf_total + wins_sci_total,
            "episodes_pooled": n_tot_method,
            "pooled_sr_pct": ((wins_alf_total + wins_sci_total) / n_tot_method) * 100.0,
            "pooled_sd_pct": float(np.std(seed_pooled_srs, ddof=1)),
            "seed_pooled_srs_pct": seed_pooled_srs,
            "alfworld_mean_retrieval_tokens": mean_tok_alf,
            "scienceworld_mean_retrieval_tokens": mean_tok_sci,
            "environment_macro_retrieval_tokens": env_macro_tok,
            "episode_pooled_retrieval_tokens": ep_pooled_tok,
        }

    return {
        "spec": spec,
        "resolved_worktree": str(wt),
        "resolved_dir": str(base),
        "summary_sha256": sha256_file(base / "stage3_main_summary.json"),
        "unique_alfworld_tasks": len(unique_tasks["alfworld"]),
        "unique_scienceworld_tasks": len(unique_tasks["scienceworld"]),
        "total_format_retry_calls": format_retry_calls,
        "methods": methods_stats,
    }


def compute_six_backbone_aggregates(campaigns: list[dict]) -> dict:
    if len(campaigns) != 6:
        raise ValueError(f"Expected 6 Stage 3 campaigns, got {len(campaigns)}")
    per_method: dict[str, dict] = {}
    for m in MAIN_METHODS:
        tot_wins = sum(int(c["methods"][m]["wins_pooled"]) for c in campaigns)
        tot_obs = sum(int(c["methods"][m]["episodes_pooled"]) for c in campaigns)
        per_backbone_rates = [
            int(c["methods"][m]["wins_pooled"]) / int(c["methods"][m]["episodes_pooled"])
            for c in campaigns
        ]
        per_method[m] = {
            "total_wins": tot_wins,
            "total_backbone_seed_episode_observations": tot_obs,
            "episode_weighted_sr": tot_wins / tot_obs,
            "episode_weighted_sr_pct": (tot_wins / tot_obs) * 100.0,
            "equal_backbone_macro_sr": float(np.mean(per_backbone_rates)),
            "equal_backbone_macro_sr_pct": float(np.mean(per_backbone_rates)) * 100.0,
        }
    return {
        "total_main_benchmark_episodes": sum(
            len(MAIN_METHODS) * int(c["methods"]["comep"]["episodes_pooled"]) for c in campaigns
        ),
        "backbone_seed_episode_observations_per_method": sum(
            int(c["methods"]["comep"]["episodes_pooled"]) for c in campaigns
        ),
        "observation_unit_note": (
            "2,475 denotes descriptive backbone-seed-episode observations per method "
            "(N_b=495 on DeepSeek-V4.1-Flash, GLM-5.3-Flash, and MiMo-V2.6-Flash; "
            "N_b=330 on Qwen3-14B-AWQ, Gemma-4-12B-it, and Qwen3-8B), not independent unique tasks."
        ),
        "definitions": {
            "episode_weighted_aggregate": "sum_b wins_b / sum_b N_b across the six Stage 3 backbones",
            "equal_backbone_macro_aggregate": "(1/6) * sum_{b=1}^6 (wins_b / N_b) across the six Stage 3 backbones",
        },
        "methods": per_method,
    }
