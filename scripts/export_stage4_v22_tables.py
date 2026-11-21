"""Audit and export Stage 4A v2.2 revised-protocol performance and telemetry tables."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import sys
import tempfile
import numpy as np
from scipy.stats import binomtest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from comep.campaign import ABLATION_SPECS, DEFAULT_MANAGER_CONFIG, compute_spec_fingerprint
from comep.core import Config

V22_ORDER = [
    "comep-warmup-gate",
    "comep",
    "additive",
    "comep-no-hard-conflicts",
    "comep-no-pair-optimism",
    "comep-no-early-stop",
    "comep-no-coalition-pruning",
    "comep-v2-online-sgd",
]

V22_LATEX_LABELS = {
    "comep-warmup-gate": "\\textbf{Full CoMeP (Ours)}",
    "comep": "\\quad w/o Warm-up Gate (Statistical Ref.)",
    "additive": "\\quad Additive Laplace UCB",
    "comep-no-hard-conflicts": "\\quad w/o Hard Conflict Guard",
    "comep-no-pair-optimism": "\\quad w/o Cold/Second-Chance Pair UCB ($\\beta_{\\text{pair}}=0$)",
    "comep-no-early-stop": "\\quad w/o Gain/Token Early Stop",
    "comep-no-coalition-pruning": "\\quad w/o Coalition-Aware Pruning",
    "comep-v2-online-sgd": "\\quad v2-Routed Online-SGD Estimator",
}

REQUIRED_TRIGGER_KEYS = (
    "episodes_routed",
    "hard_conflict_blocked_episodes",
    "early_stop_trimmed_episodes",
    "early_stop_loop_break_episodes",
    "selected_pair_cold_ucb_episodes",
    "selected_pair_second_chance_ucb_episodes",
    "selected_pair_confirmation_episodes",
    "selected_pair_random_explore_episodes",
    "warmup_gate_blocked_episodes",
    "prune_evictions",
    "coalition_prune_decision_changes",
)

EPISODE_BOUNDED_TRIGGER_KEYS = (
    "hard_conflict_blocked_episodes",
    "early_stop_trimmed_episodes",
    "early_stop_loop_break_episodes",
    "selected_pair_cold_ucb_episodes",
    "selected_pair_second_chance_ucb_episodes",
    "selected_pair_confirmation_episodes",
    "selected_pair_random_explore_episodes",
    "warmup_gate_blocked_episodes",
)

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def _is_nonneg_int(val: object) -> bool:
    return isinstance(val, int) and not isinstance(val, bool) and val >= 0


def _atomic_write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def compute_holm(raw_ps: dict[str, float]) -> dict[str, float]:
    sorted_keys = sorted(raw_ps, key=lambda k: (raw_ps[k], k))
    m = len(sorted_keys)
    holm = {}
    running_max = 0.0
    for idx, k in enumerate(sorted_keys):
        adj = min(1.0, raw_ps[k] * (m - idx))
        running_max = max(running_max, adj)
        holm[k] = running_max
    return holm


def _fmt_p(p: float, bold_sig: bool = False) -> str:
    s = f"{p:.4f}" if p >= 0.0001 else "$<10^{-4}$"
    if bold_sig and p < 0.05:
        return f"\\textbf{{{s}}}"
    return s


def _audit_http_ledger(ledger_path: Path) -> tuple[bool, int, int, int, bool]:
    """Strictly validate two-phase HTTP attempt ledger structure and token types."""
    if not ledger_path.is_file():
        return True, 0, 0, 0, True
    started_ids = []
    started_set = set()
    finished_map = {}
    for raw_line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        try:
            rec = json.loads(raw_line)
        except Exception:
            return False, 0, 0, 0, False
        if not isinstance(rec, dict):
            return False, 0, 0, 0, False
        aid = rec.get("attempt_id")
        if not isinstance(aid, str) or not aid:
            return False, 0, 0, 0, False
        phase = rec.get("phase")
        if phase == "started":
            if aid in started_set:
                # Duplicate started for same attempt_id is invalid
                return False, 0, 0, 0, False
            started_set.add(aid)
            started_ids.append(aid)
        elif phase == "finished":
            # Orphan finished (without prior started) or duplicate finished is invalid
            if aid not in started_set or aid in finished_map:
                return False, 0, 0, 0, False
            uc = rec.get("usage_complete")
            if not isinstance(uc, bool):
                return False, 0, 0, 0, False
            if uc is True:
                pt = rec.get("prompt_tokens")
                ct = rec.get("completion_tokens")
                if not (_is_nonneg_int(pt) and _is_nonneg_int(ct)):
                    return False, 0, 0, 0, False
            finished_map[aid] = rec
        else:
            # Unknown phase is invalid
            return False, 0, 0, 0, False

    pt_sum = 0
    ct_sum = 0
    complete_flag = True
    for aid in started_ids:
        f_rec = finished_map.get(aid)
        if f_rec is not None and f_rec["usage_complete"] is True:
            pt_sum += f_rec["prompt_tokens"]
            ct_sum += f_rec["completion_tokens"]
        else:
            complete_flag = False
    return True, len(started_ids), pt_sum, ct_sum, complete_flag


def get_frozen_checkout_hashes(expected_rules_by_env: dict[str, list[dict]], expected_manifests: dict[str, Path]) -> dict:
    src_hashes = {}
    for src_name in ("core.py", "client.py", "runner.py", "campaign.py"):
        src_p = REPO_ROOT / "comep" / src_name
        src_hashes[src_name] = hashlib.sha256(src_p.read_bytes()).hexdigest() if src_p.is_file() else None
    man_hashes = {
        env: hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
        for env, p in expected_manifests.items()
    }
    rule_hashes = {
        env: hashlib.sha256(json.dumps(rules, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        for env, rules in expected_rules_by_env.items()
    }
    return {
        "note": "Exporter requires the frozen run checkout; spec_fingerprint is recomputed against these live source, manifest, and warm-start pool hashes.",
        "source_hashes": src_hashes,
        "manifest_hashes": man_hashes,
        "warmstart_pool_hashes": rule_hashes,
    }


def audit_and_load_v22(
    v22_dir: Path,
    expected_episodes: int = 50,
    seeds: tuple[int, ...] = (0, 1),
    expected_protocol_version: str = "v2.2-revised",
    expected_model: str = "opencode-go/deepseek-v4.1-flash",
    expected_input_price: float = 0.30,
    expected_output_price: float = 1.20,
    expected_rules_by_env: dict[str, list[dict]] | None = None,
    expected_manifests_by_env: dict[str, Path] | None = None,
) -> tuple[bool, list[str], dict]:
    if expected_rules_by_env is None:
        expected_rules_by_env = {
            "alfworld": json.loads((REPO_ROOT / "configs" / "alfworld_warmstart_pool.json").read_text(encoding="utf-8")),
            "scienceworld": json.loads((REPO_ROOT / "configs" / "scienceworld_warmstart_pool.json").read_text(encoding="utf-8")),
        }
    if expected_manifests_by_env is None:
        expected_manifests_by_env = {
            "alfworld": REPO_ROOT / "configs" / "alfworld_test_ood_134.json",
            "scienceworld": REPO_ROOT / "configs" / "scienceworld_test_100.json",
        }

    errors = []
    data: dict[str, dict] = {
        "_checkout_identity": get_frozen_checkout_hashes(expected_rules_by_env, expected_manifests_by_env)
    }
    for env_name in ("alfworld", "scienceworld"):
        env_dir = v22_dir / env_name
        data[env_name] = {"runs": {}, "task_seq": {}}
        exp_split = "eval_out_of_distribution" if env_name == "alfworld" else "test"
        exp_max_steps = 30 if env_name == "alfworld" else 35
        exp_rules = expected_rules_by_env[env_name]
        exp_man_path = str(expected_manifests_by_env[env_name])

        for seed in seeds:
            ref_seq = None
            for m in V22_ORDER:
                run_dir = env_dir / f"{m}_seed{seed}"
                sum_p = run_dir / "summary.json"
                ep_p = run_dir / "episodes.jsonl"
                spec_p = run_dir / "worker_spec.json"
                man_p = run_dir / "manifest.json"
                state_p = run_dir / "state.json"
                ckpt_p = run_dir / "checkpoint.pkl"
                ledger_p = run_dir / "api_http_attempts.jsonl"

                missing_files = [
                    p.name for p in (sum_p, ep_p, spec_p, man_p, state_p) if not p.is_file()
                ]
                if missing_files:
                    errors.append(f"Missing required files {missing_files} in {run_dir}")
                    continue

                summ = json.loads(sum_p.read_text(encoding="utf-8"))
                sp = json.loads(spec_p.read_text(encoding="utf-8"))
                mp = json.loads(man_p.read_text(encoding="utf-8"))
                st = json.loads(state_p.read_text(encoding="utf-8"))

                fp_s = summ.get("spec_fingerprint")
                fp_w = sp.get("spec_fingerprint")
                fp_m = mp.get("spec_fingerprint")
                if not all(isinstance(x, str) and bool(_HEX64_RE.match(x)) for x in (fp_s, fp_w, fp_m)):
                    errors.append(f"Missing or invalid SHA256 spec_fingerprint in {run_dir}")
                    continue
                if not (fp_s == fp_w == fp_m):
                    errors.append(f"Fingerprint mismatch across spec/manifest/summary in {run_dir}")
                    continue
                recomputed_fp = compute_spec_fingerprint(sp)
                if recomputed_fp != fp_s:
                    errors.append(f"Recomputed spec_fingerprint mismatch against frozen checkout in {run_dir}: {recomputed_fp} != {fp_s}")
                    continue
                if ckpt_p.is_file():
                    try:
                        with ckpt_p.open("rb") as fh:
                            ckpt = pickle.load(fh)
                        if ckpt.get("spec_fingerprint") != fp_s or int(ckpt.get("next_episode", -1)) != expected_episodes:
                            errors.append(f"Residual checkpoint mismatch in {run_dir}")
                            continue
                    except Exception as exc:
                        errors.append(f"Corrupt residual checkpoint in {run_dir}: {exc}")
                        continue

                # Full contract checks on worker_spec.json
                api_spec = sp.get("api") or {}
                if (
                    sp.get("method") != m
                    or sp.get("env_name") != env_name
                    or sp.get("split") != exp_split
                    or str(sp.get("manifest_path")) != exp_man_path
                    or sp.get("seed") != seed
                    or sp.get("episodes") != expected_episodes
                    or sp.get("max_steps") != exp_max_steps
                    or sp.get("distill") is not True
                    or sp.get("initial_rules") != exp_rules
                    or (sp.get("manager") or {}) != {}
                    or api_spec.get("model") != expected_model
                    or api_spec.get("thinking") != "disabled"
                    or api_spec.get("max_output_tokens") != 512
                    or api_spec.get("max_format_retries") != 2
                    or api_spec.get("protocol_version") != expected_protocol_version
                    or not api_spec.get("cache_dir")
                    or api_spec.get("cache_namespace") != f"{env_name}:seed_{seed}"
                    or float(api_spec.get("input_price_per_million", -1.0)) != expected_input_price
                    or float(api_spec.get("output_price_per_million", -1.0)) != expected_output_price
                ):
                    errors.append(f"Protocol contract mismatch in worker_spec.json of {run_dir}")
                    continue

                # Full resolved manager configuration check on state.json
                base_method, expected_overrides = ABLATION_SPECS[m]
                expected_full_cfg = asdict(Config(**{**DEFAULT_MANAGER_CONFIG, **expected_overrides}))
                st_cfg = st.get("config")
                if (
                    st.get("method") != base_method
                    or st.get("observations") != expected_episodes
                    or st_cfg != expected_full_cfg
                ):
                    errors.append(f"Full resolved state.config or method mismatch in {run_dir}")
                    continue

                if summ.get("status") != "completed" or summ.get("completed_episodes") != expected_episodes:
                    errors.append(f"Incomplete run in {run_dir}: status={summ.get('status')}, eps={summ.get('completed_episodes')}")
                    continue

                ep_lines = [json.loads(line) for line in ep_p.read_text(encoding="utf-8").splitlines() if line.strip()]
                ep_lines.sort(key=lambda r: int(r["episode"]))
                if len(ep_lines) != expected_episodes or [r.get("episode") for r in ep_lines] != list(range(expected_episodes)):
                    errors.append(f"Non-contiguous or wrong episode count in {run_dir}")
                    continue

                rewards = []
                ret_toks = []
                bad_ep = False
                for r in ep_lines:
                    rew = r.get("reward")
                    if isinstance(rew, bool) or not isinstance(rew, int) or rew not in (0, 1):
                        errors.append(f"Non-binary reward {rew!r} in {run_dir}")
                        bad_ep = True
                        break
                    rt = r.get("retrieval_tokens")
                    if not _is_nonneg_int(rt):
                        errors.append(f"Invalid retrieval_tokens {rt!r} in {run_dir}")
                        bad_ep = True
                        break
                    rewards.append(rew)
                    ret_toks.append(rt)
                if bad_ep:
                    continue

                wins = sum(rewards)
                late_rewards = rewards[expected_episodes // 2 :]
                late_wins = sum(late_rewards)
                calc_sr = wins / expected_episodes
                calc_late = late_wins / len(late_rewards)
                calc_tok = sum(ret_toks) / expected_episodes
                if (
                    abs(float(summ.get("success_rate", -1.0)) - calc_sr) > 1e-6
                    or abs(float(summ.get("late_success_rate", -1.0)) - calc_late) > 1e-6
                    or summ.get("retrieval_tokens") != sum(ret_toks)
                    or abs(float(summ.get("mean_retrieval_tokens", -1.0)) - calc_tok) > 1e-6
                ):
                    errors.append(f"Summary metric mismatch vs raw episodes.jsonl in {run_dir}")
                    continue

                tc = summ.get("trigger_counts")
                if not isinstance(tc, dict) or tc != st.get("trigger_counts"):
                    errors.append(f"Missing or inconsistent trigger_counts between summary and state in {run_dir}")
                    continue
                missing_tk = [k for k in REQUIRED_TRIGGER_KEYS if k not in tc or not _is_nonneg_int(tc[k])]
                if missing_tk:
                    errors.append(f"Missing or invalid trigger keys {missing_tk} in {run_dir}")
                    continue
                if tc["episodes_routed"] != expected_episodes:
                    errors.append(f"episodes_routed={tc['episodes_routed']} != {expected_episodes} in {run_dir}")
                    continue
                if any(not (0 <= tc[k] <= expected_episodes) for k in EPISODE_BOUNDED_TRIGGER_KEYS):
                    errors.append(f"Out-of-range episode trigger count in {run_dir}: {tc}")
                    continue
                if not (0 <= tc["coalition_prune_decision_changes"] <= tc["prune_evictions"]):
                    errors.append(f"Invalid coalition_prune_decision_changes vs prune_evictions in {run_dir}: {tc}")
                    continue

                # Strict API & HTTP ledger audit + independent cost recomputation
                api = summ.get("api")
                if not isinstance(api, dict):
                    errors.append(f"Missing summary.api dict in {run_dir}")
                    continue
                uc_flag = api.get("usage_complete")
                huc_flag = api.get("http_usage_complete")
                if not isinstance(uc_flag, bool) or not isinstance(huc_flag, bool):
                    errors.append(f"Missing or non-boolean usage_complete flags in {run_dir}")
                    continue
                pt_log = api.get("prompt_tokens")
                ct_log = api.get("completion_tokens")
                att_calls = api.get("attempted_calls")
                c_hits = api.get("cache_hits")
                if not all(_is_nonneg_int(x) for x in (pt_log, ct_log, att_calls, c_hits)):
                    errors.append(f"Invalid logical token or call counters in {run_dir}")
                    continue

                l_valid, l_reqs, l_pt, l_ct, l_ok = _audit_http_ledger(ledger_p)
                if not l_valid:
                    errors.append(f"Malformed or structurally invalid api_http_attempts.jsonl in {run_dir}")
                    continue
                if (
                    api.get("http_requests") != l_reqs
                    or api.get("provider_reported_prompt_tokens") != l_pt
                    or api.get("provider_reported_completion_tokens") != l_ct
                    or huc_flag != l_ok
                ):
                    errors.append(f"HTTP ledger mismatch vs summary.api in {run_dir}")
                    continue

                # Recompute logical and provider-reported costs independently from tokens and unit prices
                recomp_logical_cost = (
                    (pt_log * expected_input_price + ct_log * expected_output_price) / 1_000_000
                    if uc_flag
                    else None
                )
                recomp_prov_cost = (
                    (l_pt * expected_input_price + l_ct * expected_output_price) / 1_000_000
                    if l_ok
                    else None
                )
                sum_lc = api.get("estimated_cost_usd")
                sum_pc = api.get("provider_reported_cost_usd")
                if recomp_logical_cost is None:
                    if sum_lc is not None:
                        errors.append(f"Expected None estimated_cost_usd when usage_complete=False in {run_dir}")
                        continue
                else:
                    if not isinstance(sum_lc, (int, float)) or isinstance(sum_lc, bool) or not math.isfinite(sum_lc) or sum_lc < 0 or abs(float(sum_lc) - recomp_logical_cost) > 1e-8:
                        errors.append(f"Logical cost mismatch in {run_dir}: {sum_lc} != {recomp_logical_cost}")
                        continue
                if recomp_prov_cost is None:
                    if sum_pc is not None:
                        errors.append(f"Expected None provider_reported_cost_usd when http_usage_complete=False in {run_dir}")
                        continue
                else:
                    if not isinstance(sum_pc, (int, float)) or isinstance(sum_pc, bool) or not math.isfinite(sum_pc) or sum_pc < 0 or abs(float(sum_pc) - recomp_prov_cost) > 1e-8:
                        errors.append(f"Provider-reported cost mismatch in {run_dir}: {sum_pc} != {recomp_prov_cost}")
                        continue

                seq = [(int(r["episode"]), str(r["task_id"])) for r in ep_lines]
                if ref_seq is None:
                    ref_seq = seq
                    data[env_name]["task_seq"][seed] = seq
                elif seq != ref_seq:
                    errors.append(f"Task sequence mismatch for {env_name} {m} seed={seed}")
                    continue

                data[env_name]["runs"][(m, seed)] = {
                    "summary": summ,
                    "episodes": ep_lines,
                    "rewards": rewards,
                    "late_rewards": late_rewards,
                    "wins": wins,
                    "late_wins": late_wins,
                    "mean_tok": calc_tok,
                    "recomp_logical_cost": recomp_logical_cost,
                    "recomp_prov_cost": recomp_prov_cost,
                }
    complete = len(errors) == 0 and all(
        len(data[e]["runs"]) == len(V22_ORDER) * len(seeds) for e in ("alfworld", "scienceworld")
    )
    return complete, errors, data


def build_incomplete_preview(data: dict, errors: list[str], seeds: tuple[int, ...] = (0, 1)) -> dict:
    progress = {}
    for env_name in ("alfworld", "scienceworld"):
        runs = data.get(env_name, {}).get("runs", {})
        progress[env_name] = {
            f"{m}_seed{s}": {
                "completed": (m, s) in runs,
                "wins": runs[(m, s)]["wins"] if (m, s) in runs else None,
            }
            for m in V22_ORDER
            for s in seeds
        }
    return {
        "kind": "stage4_v22_incomplete_preview",
        "complete": False,
        "status_stamp": "INCOMPLETE_PREVIEW",
        "checkout_identity": data.get("_checkout_identity"),
        "audit_errors": errors,
        "completed_run_count": sum(len(data.get(e, {}).get("runs", {})) for e in ("alfworld", "scienceworld")),
        "expected_run_count": len(V22_ORDER) * len(seeds) * 2,
        "progress": progress,
    }


def compute_metrics_table(data: dict, expected_episodes: int = 50, seeds: tuple[int, ...] = (0, 1)) -> dict:
    metrics: dict[str, dict] = {"alfworld": {}, "scienceworld": {}, "pooled": {}}
    for env_name in ("alfworld", "scienceworld"):
        runs = data[env_name]["runs"]
        ref_rews = [y for s in seeds for y in runs[("comep", s)]["rewards"]]
        raw_ps = {}
        disc_map = {}
        for m in V22_ORDER:
            seed_srs = [runs[(m, s)]["wins"] / expected_episodes for s in seeds]
            seed_lates = [runs[(m, s)]["late_wins"] / (expected_episodes - expected_episodes // 2) for s in seeds]
            seed_toks = [runs[(m, s)]["mean_tok"] for s in seeds]
            m_rews = [y for s in seeds for y in runs[(m, s)]["rewards"]]

            mean_sr = float(np.mean(seed_srs))
            std_sr = float(np.std(seed_srs, ddof=1)) if len(seed_srs) > 1 else 0.0
            mean_late = float(np.mean(seed_lates))
            mean_tok = float(np.mean(seed_toks))

            trig_sum: dict[str, int] = {k: 0 for k in REQUIRED_TRIGGER_KEYS}
            logical_calls = 0
            http_reqs = 0
            cache_hits = 0
            act_retries = 0
            dist_retries = 0
            logical_cost_raw = 0.0
            prov_cost_raw = 0.0
            logical_ok = True
            prov_ok = True
            for s in seeds:
                r_obj = runs[(m, s)]
                sm = r_obj["summary"]
                api = sm["api"]
                tc = sm["trigger_counts"]
                for tk in REQUIRED_TRIGGER_KEYS:
                    trig_sum[tk] += int(tc[tk])
                logical_calls += int(api["attempted_calls"])
                http_reqs += int(api["http_requests"])
                cache_hits += int(api["cache_hits"])
                act_retries += int(sm["action_format_retry_calls"])
                dist_retries += int(sm["distillation_format_retry_calls"])
                if r_obj["recomp_logical_cost"] is not None:
                    logical_cost_raw += r_obj["recomp_logical_cost"]
                else:
                    logical_ok = False
                if r_obj["recomp_prov_cost"] is not None:
                    prov_cost_raw += r_obj["recomp_prov_cost"]
                else:
                    prov_ok = False

            if m != "comep":
                c_arr = np.array(ref_rews)
                b_arr = np.array(m_rews)
                b10 = int(np.sum((c_arr == 1) & (b_arr == 0)))
                b01 = int(np.sum((c_arr == 0) & (b_arr == 1)))
                disc_map[m] = (b10, b01)
                raw_ps[m] = float(binomtest(b10, b10 + b01, 0.5).pvalue) if (b10 + b01) > 0 else 1.0

            metrics[env_name][m] = {
                "seed_srs": seed_srs,
                "mean_sr": mean_sr,
                "std_sr": std_sr,
                "mean_late": mean_late,
                "mean_tok": mean_tok,
                "rewards": m_rews,
                "triggers": trig_sum,
                "logical_calls": logical_calls,
                "http_requests": http_reqs,
                "cache_hits": cache_hits,
                "action_retries": act_retries,
                "distill_retries": dist_retries,
                "logical_cost_raw": logical_cost_raw if logical_ok else None,
                "provider_reported_cost_raw": prov_cost_raw if prov_ok else None,
                "logical_cost_usd": round(logical_cost_raw, 4) if logical_ok else None,
                "provider_reported_cost_usd": round(prov_cost_raw, 4) if prov_ok else None,
            }
        holm_ps = compute_holm(raw_ps)
        ref_sr = metrics[env_name]["comep"]["mean_sr"]
        for m in V22_ORDER:
            metrics[env_name][m]["delta_sr_pp"] = (metrics[env_name][m]["mean_sr"] - ref_sr) * 100.0
            if m != "comep":
                metrics[env_name][m]["b10"], metrics[env_name][m]["b01"] = disc_map[m]
                metrics[env_name][m]["raw_p"] = raw_ps[m]
                metrics[env_name][m]["holm_p"] = holm_ps[m]

    raw_ps_pool = {}
    disc_pool = {}
    ref_sr_pool = 0.5 * (metrics["alfworld"]["comep"]["mean_sr"] + metrics["scienceworld"]["comep"]["mean_sr"])
    for m in V22_ORDER:
        ma = metrics["alfworld"][m]
        ms = metrics["scienceworld"][m]
        seed_pooled = [0.5 * (a_s + s_s) for a_s, s_s in zip(ma["seed_srs"], ms["seed_srs"])]
        pooled_sr = float(np.mean(seed_pooled))
        pooled_std = float(np.std(seed_pooled, ddof=1)) if len(seed_pooled) > 1 else 0.0
        pooled_late = 0.5 * (ma["mean_late"] + ms["mean_late"])
        pooled_tok = 0.5 * (ma["mean_tok"] + ms["mean_tok"])
        if m != "comep":
            b10 = ma["b10"] + ms["b10"]
            b01 = ma["b01"] + ms["b01"]
            disc_pool[m] = (b10, b01)
            raw_ps_pool[m] = float(binomtest(b10, b10 + b01, 0.5).pvalue) if (b10 + b01) > 0 else 1.0
        trig_comb = {k: ma["triggers"][k] + ms["triggers"][k] for k in REQUIRED_TRIGGER_KEYS}
        lc_raw = (
            ma["logical_cost_raw"] + ms["logical_cost_raw"]
            if (ma["logical_cost_raw"] is not None and ms["logical_cost_raw"] is not None)
            else None
        )
        pc_raw = (
            ma["provider_reported_cost_raw"] + ms["provider_reported_cost_raw"]
            if (ma["provider_reported_cost_raw"] is not None and ms["provider_reported_cost_raw"] is not None)
            else None
        )
        metrics["pooled"][m] = {
            "seed_srs": seed_pooled,
            "mean_sr": pooled_sr,
            "std_sr": pooled_std,
            "mean_late": pooled_late,
            "mean_tok": pooled_tok,
            "delta_sr_pp": (pooled_sr - ref_sr_pool) * 100.0,
            "triggers": trig_comb,
            "logical_calls": ma["logical_calls"] + ms["logical_calls"],
            "http_requests": ma["http_requests"] + ms["http_requests"],
            "cache_hits": ma["cache_hits"] + ms["cache_hits"],
            "action_retries": ma["action_retries"] + ms["action_retries"],
            "distill_retries": ma["distill_retries"] + ms["distill_retries"],
            "logical_cost_usd": round(lc_raw, 4) if lc_raw is not None else None,
            "provider_reported_cost_usd": round(pc_raw, 4) if pc_raw is not None else None,
        }
    holm_pool = compute_holm(raw_ps_pool)
    for m in disc_pool:
        metrics["pooled"][m]["b10"], metrics["pooled"][m]["b01"] = disc_pool[m]
        metrics["pooled"][m]["raw_p"] = raw_ps_pool[m]
        metrics["pooled"][m]["holm_p"] = holm_pool[m]
    return metrics


def render_v22_performance_table(metrics: dict, out_path: Path):
    alf = metrics["alfworld"]
    sci = metrics["scienceworld"]
    pool = metrics["pooled"]

    def _rank_marks(vals: dict[str, float]) -> tuple[float | None, float | None]:
        uniq = sorted({round(v, 1) for v in vals.values()}, reverse=True)
        best = uniq[0] if uniq else None
        second = uniq[1] if len(uniq) > 1 else None
        return best, second

    best_a_sr, sec_a_sr = _rank_marks({m: alf[m]["mean_sr"] * 100 for m in V22_ORDER})
    best_a_lt, sec_a_lt = _rank_marks({m: alf[m]["mean_late"] * 100 for m in V22_ORDER})
    best_s_sr, sec_s_sr = _rank_marks({m: sci[m]["mean_sr"] * 100 for m in V22_ORDER})
    best_s_lt, sec_s_lt = _rank_marks({m: sci[m]["mean_late"] * 100 for m in V22_ORDER})
    best_p_sr, sec_p_sr = _rank_marks({m: pool[m]["mean_sr"] * 100 for m in V22_ORDER})
    best_p_lt, sec_p_lt = _rank_marks({m: pool[m]["mean_late"] * 100 for m in V22_ORDER})

    def _style_val(val_str: str, v_num: float, best: float | None, sec: float | None) -> str:
        rv = round(v_num, 1)
        if best is not None and rv == best:
            return f"\\bst{{{val_str}}}"
        if sec is not None and rv == sec:
            return f"\\subbst{{{val_str}}}"
        return val_str

    lines = [
        "\\begin{table*}[t]",
        "  \\caption{Stage 4A ablation and estimator evaluation on DeepSeek-V4.1-Flash ($50$ episodes $\\times 2$ seeds $= 100$ eps/condition per environment; $1{,}600$ total episodes across $8$ conditions under up to two extra format-validation retries and per-replicate complete-request response caching). Full CoMeP (Ours, with the warm-up gate $G_t(i,j)$) is shown in the top row; because the four component removals and the v2-Routed Online-SGD estimator comparison were executed on the ungated configuration, w/o Warm-up Gate serves as the statistical reference row ($\\Delta_{\\text{SR}} = \\text{SR}_{\\text{row}} - \\text{SR}_{\\text{ungated ref.}}$, and $b_{10}:b_{01}$ denotes ungated-reference unique wins : row unique wins paired on environment, seed, episode, and task identifier).}\\label{tab:ablation_study}",
        "  \\vskip -0.06in",
        "  \\renewcommand{\\arraystretch}{0.94}",
        "  \\setlength{\\tabcolsep}{3.4pt}",
        "  \\centering",
        "  \\scriptsize",
        "  \\resizebox{\\textwidth}{!}{%",
        "  \\begin{threeparttable}",
        "  \\begin{tabular}{l|ccc|ccc|cccc}",
        "    \\toprule",
        "    \\multirow{2}{*}{\\textbf{Condition}} & \\multicolumn{3}{c|}{\\textbf{ALFWorld OOD} ($100$ eps)} & \\multicolumn{3}{c|}{\\textbf{ScienceWorld Test} ($100$ eps)} & \\multicolumn{4}{c}{\\textbf{Combined Pooled Benchmark} ($200$ eps)} \\\\",
        "    \\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-11}",
        "    & SR (\\%) & Late (\\%) & Tok & SR (\\%) & Late (\\%) & Tok & Pooled SR (Late) & $\\Delta_{\\text{SR}}$ (pp) & Tok & Disc.\\ (Raw / Holm $p$) \\\\",
        "    \\midrule",
    ]
    for m in V22_ORDER:
        if m == "comep-no-hard-conflicts":
            lines.append("    \\midrule")
            lines.append("    \\multicolumn{11}{l}{\\textit{Component removals \\& estimator comparison on the ungated configuration ($G_t(i,j)\\equiv 1$):}} \\\\")
        ra, rs, rp = alf[m], sci[m], pool[m]
        a_sr_s = _style_val(f"{ra['mean_sr']*100:.1f}$\\pm${ra['std_sr']*100:.1f}", ra["mean_sr"] * 100, best_a_sr, sec_a_sr)
        a_lt_s = _style_val(f"{ra['mean_late']*100:.1f}", ra["mean_late"] * 100, best_a_lt, sec_a_lt)
        s_sr_s = _style_val(f"{rs['mean_sr']*100:.1f}$\\pm${rs['std_sr']*100:.1f}", rs["mean_sr"] * 100, best_s_sr, sec_s_sr)
        s_lt_s = _style_val(f"{rs['mean_late']*100:.1f}", rs["mean_late"] * 100, best_s_lt, sec_s_lt)
        p_sr_inner = _style_val(f"{rp['mean_sr']*100:.1f}", rp["mean_sr"] * 100, best_p_sr, sec_p_sr)
        p_lt_inner = _style_val(f"{rp['mean_late']*100:.1f}", rp["mean_late"] * 100, best_p_lt, sec_p_lt)
        if m == "comep":
            d_str = "---"
            p_str = "---"
        else:
            d_str = f"{rp['delta_sr_pp']:+.1f}"
            p_str = f"{rp['b10']}:{rp['b01']} ({_fmt_p(rp['raw_p'])} / {_fmt_p(rp['holm_p'], bold_sig=True)})"
        lines.append(
            f"    {V22_LATEX_LABELS.get(m, m)} & {a_sr_s} & {a_lt_s} & {ra['mean_tok']:.1f} & "
            f"{s_sr_s} & {s_lt_s} & {rs['mean_tok']:.1f} & "
            f"{p_sr_inner} ({p_lt_inner}) & {d_str} & {rp['mean_tok']:.1f} & {p_str} \\\\"
        )
    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\begin{tablenotes}",
        "    \\item \\scriptsize \\textit{Note}: $\\pm$ denotes the descriptive 2-seed sample SD computed from exact integer win counts before rounding; Late SR is fixed to the second half (episodes $25$--$49$) of each 50-episode stream. This table displays the 7-comparison Holm-adjusted exact two-sided McNemar $p$-values relative to the ungated reference row (stratified single-environment 7-comparison tests are reported in Appendix~\\ref{app:v22_telemetry}). Because two seeds, online state dependence, and cross-seed task overlap limit calibration, $p$-values serve as exploratory comparative indicators.",
        "  \\end{tablenotes}",
        "  \\end{threeparttable}%",
        "  }",
        "  \\vskip -0.06in",
        "\\end{table*}",
    ])
    _atomic_write_text(out_path, "\n".join(lines) + "\n")


def render_v22_telemetry_table(metrics: dict, out_path: Path):
    pool = metrics["pooled"]
    lines = [
        "\\begin{table*}[ht]",
        "  \\caption{Companion protocol, caching, cost, and mechanism-activation telemetry for the Stage 4A ablation and estimator evaluation matrix ($200$ pooled episodes per condition across ALFWorld and ScienceWorld). Routing activations are reported as deduplicated episode counts over $200$ routed episodes; pruning changes report single-step local counterfactual eviction decision changes over total pruning evictions (\\texttt{NA} when zero evictions occur).}\\label{tab:ablation_telemetry_v22}",
        "  \\vskip -0.06in",
        "  \\renewcommand{\\arraystretch}{0.94}",
        "  \\setlength{\\tabcolsep}{3.0pt}",
        "  \\centering",
        "  \\scriptsize",
        "  \\resizebox{\\textwidth}{!}{%",
        "  \\begin{threeparttable}",
        "  \\begin{tabular}{l|ccc|cccccc}",
        "    \\toprule",
        "    \\multirow{2}{*}{\\textbf{Condition}} & \\multicolumn{3}{c|}{\\textbf{API \\& Shared-Cache Accounting}} & \\multicolumn{6}{c}{\\textbf{Mechanism Activation Telemetry} ($200$ routed episodes)} \\\\",
        "    \\cmidrule(lr){2-4} \\cmidrule(lr){5-10}",
        "    & Log.\\ Calls (Retries) & Hits / Live HTTP & Log.\\ / Prov.\\ Cost (\\$) & Hard-Conf. & Early Trim / Brk. & Pair Cold / 2nd / Conf. & Rand.\\ Pair & Gate Blk. & Coal.\\ Prune Chg./Evict \\\\",
        "    \\midrule",
    ]
    for m in V22_ORDER:
        if m == "comep-no-hard-conflicts":
            lines.append("    \\midrule")
            lines.append("    \\multicolumn{10}{l}{\\textit{Component removals \\& estimator comparison on the ungated configuration ($G_t(i,j)\\equiv 1$):}} \\\\")
        rp = pool[m]
        tc = rp["triggers"]
        tot_ret = rp["action_retries"] + rp["distill_retries"]
        lc_s = f"{rp['logical_cost_usd']:.2f}" if rp["logical_cost_usd"] is not None else "UNK"
        pc_s = f"{rp['provider_reported_cost_usd']:.2f}" if rp["provider_reported_cost_usd"] is not None else "UNK"
        ev = int(tc["prune_evictions"])
        chg = int(tc["coalition_prune_decision_changes"])
        prune_s = f"{chg}/{ev} ({chg/ev*100:.0f}\\%)" if ev > 0 else "0/0 (NA)"
        lines.append(
            f"    {V22_LATEX_LABELS.get(m, m)} & {rp['logical_calls']} ({tot_ret}) & {rp['cache_hits']} / {rp['http_requests']} & "
            f"{lc_s} / {pc_s} & {tc['hard_conflict_blocked_episodes']}/200 & "
            f"{tc['early_stop_trimmed_episodes']} / {tc['early_stop_loop_break_episodes']} & "
            f"{tc['selected_pair_cold_ucb_episodes']} / {tc['selected_pair_second_chance_ucb_episodes']} / {tc['selected_pair_confirmation_episodes']} & "
            f"{tc['selected_pair_random_explore_episodes']}/200 & {tc['warmup_gate_blocked_episodes']}/200 & {prune_s} \\\\"
        )
    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\begin{tablenotes}",
        "    \\item \\scriptsize \\textit{Note}: Within each replicate ($\\text{env}:\\text{seed}$), the request cache shares responses only when the complete request payload (\\texttt{messages}, generation parameters, \\texttt{thinking}), \\texttt{purpose}, protocol version, and episode/step/format-attempt context match identically (for distillation calls without a step index, keyed by episode, format-attempt index, and full distillation prompt). Logical cost values logical token usage under the configured unit prices, including cached responses; it is not an independently incurred bill. Provider-reported cost reflects observable live-request usage and cache-population order (UNK indicates incomplete observable usage, including unfinished attempts or HTTP/network errors without provider-reported token usage). Coal.\\ Prune Chg./Evict counts single-step local counterfactual eviction differences at the moment of pruning rather than full-trajectory causal effects.",
        "  \\end{tablenotes}",
        "  \\end{threeparttable}%",
        "  }",
        "\\end{table*}",
    ])
    _atomic_write_text(out_path, "\n".join(lines) + "\n")


def run_self_tests():
    """Offline regression tests covering all 9 audit failure modes required by Codex."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "v22"
        man_file = Path(tmp) / "manifest_tasks.json"
        man_file.write_text(json.dumps([{"task_id": f"t{i}"} for i in range(4)]), encoding="utf-8")
        rules_fixture = [{"id": "r0", "text": "Rule 0"}]
        rules_by_env = {"alfworld": rules_fixture, "scienceworld": rules_fixture}
        mans_by_env = {"alfworld": man_file, "scienceworld": man_file}

        def _populate_valid_grid(ep_count: int = 4):
            for env_name in ("alfworld", "scienceworld"):
                for seed in (0, 1):
                    for m in V22_ORDER:
                        rdir = root / env_name / f"{m}_seed{seed}"
                        rdir.mkdir(parents=True, exist_ok=True)
                        base_m, ov = ABLATION_SPECS[m]
                        spec = {
                            "env_name": env_name,
                            "split": "eval_out_of_distribution" if env_name == "alfworld" else "test",
                            "manifest_path": str(man_file),
                            "seed": seed,
                            "episodes": ep_count,
                            "max_steps": 30 if env_name == "alfworld" else 35,
                            "method": m,
                            "initial_rules": rules_fixture,
                            "distill": True,
                            "admissible_limit": 0,
                            "action_instructions": "Act",
                            "manager": {},
                            "output_dir": str(rdir),
                            "api": {
                                "base_url": "http://127.0.0.1:3425/v1",
                                "model": "opencode-go/deepseek-v4.1-flash",
                                "key_env": "OPENCODE_API_KEY",
                                "thinking": "disabled",
                                "max_output_tokens": 512,
                                "max_format_retries": 2,
                                "protocol_version": "v2.2-revised",
                                "cache_dir": str(root / ".cache" / env_name / f"seed_{seed}"),
                                "cache_namespace": f"{env_name}:seed_{seed}",
                                "input_price_per_million": 0.3,
                                "output_price_per_million": 1.2,
                            },
                        }
                        fp = compute_spec_fingerprint(spec)
                        (rdir / "worker_spec.json").write_text(json.dumps({**spec, "spec_fingerprint": fp}), encoding="utf-8")
                        (rdir / "manifest.json").write_text(json.dumps({"spec_fingerprint": fp}), encoding="utf-8")
                        tc = {k: 0 for k in REQUIRED_TRIGGER_KEYS}
                        tc["episodes_routed"] = ep_count
                        tc["prune_evictions"] = 2
                        tc["coalition_prune_decision_changes"] = 1
                        full_cfg = asdict(Config(**{**DEFAULT_MANAGER_CONFIG, **ov}))
                        (rdir / "state.json").write_text(
                            json.dumps({"method": base_m, "observations": ep_count, "config": full_cfg, "trigger_counts": tc}),
                            encoding="utf-8",
                        )
                        eps = [{"episode": i, "task_id": f"t{i}", "reward": 1 if i % 2 == 0 else 0, "retrieval_tokens": 60} for i in range(ep_count)]
                        (rdir / "episodes.jsonl").write_text("\n".join(json.dumps(e) for e in eps) + "\n", encoding="utf-8")
                        is_primary = m == "comep"
                        if is_primary:
                            (rdir / "api_http_attempts.jsonl").write_text(
                                json.dumps({"phase": "started", "attempt_id": "a1"}) + "\n"
                                + json.dumps({"phase": "finished", "attempt_id": "a1", "usage_complete": True, "prompt_tokens": 1000, "completion_tokens": 100}) + "\n",
                                encoding="utf-8",
                            )
                        else:
                            (rdir / "api_http_attempts.jsonl").write_text("", encoding="utf-8")
                        summ = {
                            "status": "completed",
                            "spec_fingerprint": fp,
                            "completed_episodes": ep_count,
                            "success_rate": 0.5,
                            "late_success_rate": 0.5,
                            "retrieval_tokens": 60 * ep_count,
                            "mean_retrieval_tokens": 60.0,
                            "action_format_retry_calls": 0,
                            "distillation_format_retry_calls": 0,
                            "trigger_counts": tc,
                            "api": {
                                "attempted_calls": 10,
                                "http_requests": 1 if is_primary else 0,
                                "cache_hits": 0 if is_primary else 10,
                                "prompt_tokens": 1000,
                                "completion_tokens": 100,
                                "provider_reported_prompt_tokens": 1000 if is_primary else 0,
                                "provider_reported_completion_tokens": 100 if is_primary else 0,
                                "usage_complete": True,
                                "http_usage_complete": True,
                                "estimated_cost_usd": 0.00042,
                                "provider_reported_cost_usd": 0.00042 if is_primary else 0.0,
                            },
                        }
                        (rdir / "summary.json").write_text(json.dumps(summ), encoding="utf-8")

        def _audit():
            return audit_and_load_v22(
                root,
                expected_episodes=4,
                expected_rules_by_env=rules_by_env,
                expected_manifests_by_env=mans_by_env,
            )

        _populate_valid_grid(4)
        ok, errs, data = _audit()
        assert ok and not errs, f"Expected valid grid to pass, got {errs}"
        m_tab = compute_metrics_table(data, expected_episodes=4)
        assert m_tab["pooled"]["additive"]["provider_reported_cost_usd"] == 0.0
        assert m_tab["pooled"]["comep"]["provider_reported_cost_usd"] > 0.0

        target_dir = root / "alfworld" / "comep_seed0"

        # 1. Missing manifest file must fail audit
        (target_dir / "manifest.json").unlink()
        ok1, errs1, _ = _audit()
        assert not ok1 and any("Missing required files" in e for e in errs1)
        _populate_valid_grid(4)

        # 2. All-None or invalid SHA256 fingerprint must fail audit
        for fn in ("worker_spec.json", "manifest.json", "summary.json"):
            d = json.loads((target_dir / fn).read_text(encoding="utf-8"))
            d["spec_fingerprint"] = None
            (target_dir / fn).write_text(json.dumps(d), encoding="utf-8")
        ok2, errs2, _ = _audit()
        assert not ok2 and any("invalid SHA256" in e for e in errs2)
        _populate_valid_grid(4)

        # 3. Non-binary reward (e.g. 0.9) must fail audit
        eps_bad = [{"episode": i, "task_id": f"t{i}", "reward": 0.9 if i == 0 else 0, "retrieval_tokens": 60} for i in range(4)]
        (target_dir / "episodes.jsonl").write_text("\n".join(json.dumps(e) for e in eps_bad) + "\n", encoding="utf-8")
        ok3, errs3, _ = _audit()
        assert not ok3 and any("Non-binary reward" in e for e in errs3)
        _populate_valid_grid(4)

        # 4. Incomplete Full seeds must not crash preview builder
        (root / "alfworld" / "comep_seed1" / "summary.json").unlink()
        ok4, errs4, data4 = _audit()
        assert not ok4
        prev = build_incomplete_preview(data4, errs4)
        assert prev["status_stamp"] == "INCOMPLETE_PREVIEW" and prev["completed_run_count"] == 31
        _populate_valid_grid(4)

        # 5. Missing telemetry key must fail audit
        s_bad = json.loads((target_dir / "summary.json").read_text(encoding="utf-8"))
        del s_bad["trigger_counts"]["warmup_gate_blocked_episodes"]
        (target_dir / "summary.json").write_text(json.dumps(s_bad), encoding="utf-8")
        ok5, errs5, _ = _audit()
        assert not ok5 and any("trigger" in e for e in errs5)
        _populate_valid_grid(4)

        # 6. Unknown HTTP usage (unfinished attempt in ledger) propagates None to provider_reported_cost_usd
        with (target_dir / "api_http_attempts.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"phase": "started", "attempt_id": "unfinished_99"}) + "\n")
        s_unk = json.loads((target_dir / "summary.json").read_text(encoding="utf-8"))
        s_unk["api"]["http_requests"] = 2
        s_unk["api"]["http_usage_complete"] = False
        s_unk["api"]["provider_reported_cost_usd"] = None
        (target_dir / "summary.json").write_text(json.dumps(s_unk), encoding="utf-8")
        ok6, errs6, data6 = _audit()
        assert ok6 and not errs6
        m_tab6 = compute_metrics_table(data6, expected_episodes=4)
        assert m_tab6["pooled"]["comep"]["provider_reported_cost_usd"] is None
        _populate_valid_grid(4)

        # 7. Full CoMeP with unexpected warmup_gate=True or altered token budget in state["config"] must fail audit
        st_bad = json.loads((target_dir / "state.json").read_text(encoding="utf-8"))
        st_bad["config"]["warmup_gate"] = True
        (target_dir / "state.json").write_text(json.dumps(st_bad), encoding="utf-8")
        ok7a, errs7a, _ = _audit()
        assert not ok7a and any("state.config" in e for e in errs7a)
        _populate_valid_grid(4)
        st_bad2 = json.loads((target_dir / "state.json").read_text(encoding="utf-8"))
        st_bad2["config"]["retrieval_tokens"] = 120
        (target_dir / "state.json").write_text(json.dumps(st_bad2), encoding="utf-8")
        ok7b, errs7b, _ = _audit()
        assert not ok7b and any("state.config" in e for e in errs7b)
        _populate_valid_grid(4)

        # 8. Orphan finished, duplicate finished, or float token in HTTP ledger must fail audit
        (target_dir / "api_http_attempts.jsonl").write_text(
            json.dumps({"phase": "finished", "attempt_id": "orphan_1", "usage_complete": True, "prompt_tokens": 10, "completion_tokens": 5}) + "\n",
            encoding="utf-8",
        )
        ok8a, errs8a, _ = _audit()
        assert not ok8a and any("api_http_attempts.jsonl" in e for e in errs8a)
        (target_dir / "api_http_attempts.jsonl").write_text(
            json.dumps({"phase": "started", "attempt_id": "a1"}) + "\n"
            + json.dumps({"phase": "finished", "attempt_id": "a1", "usage_complete": True, "prompt_tokens": 10.5, "completion_tokens": 5}) + "\n",
            encoding="utf-8",
        )
        ok8b, errs8b, _ = _audit()
        assert not ok8b and any("api_http_attempts.jsonl" in e for e in errs8b)
        _populate_valid_grid(4)

        # 9. Correct token counts but wrong provider_reported_cost_usd or mutated manifest/source identity must fail audit
        s_cost_bad = json.loads((target_dir / "summary.json").read_text(encoding="utf-8"))
        s_cost_bad["api"]["provider_reported_cost_usd"] = 0.9999
        (target_dir / "summary.json").write_text(json.dumps(s_cost_bad), encoding="utf-8")
        ok9a, errs9a, _ = _audit()
        assert not ok9a and any("Provider-reported cost mismatch" in e for e in errs9a)
        _populate_valid_grid(4)

        # Mutate manifest file content after run to simulate workspace drift -> recomputed_fp mismatch
        man_file.write_text(json.dumps([{"task_id": f"mutated_{i}"} for i in range(4)]), encoding="utf-8")
        ok9b, errs9b, _ = _audit()
        assert not ok9b and any("Recomputed spec_fingerprint mismatch" in e for e in errs9b)
        print("ALL 9 STAGE4-V22 EXPORTER SELF-TESTS PASSED!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v22-dir", default=str(REPO_ROOT / "outputs/stage4-ablations-v22"))
    parser.add_argument("--out-json", default=str(REPO_ROOT / "paper_artifacts/stage4_v22_audited_metrics.json"))
    parser.add_argument("--preview-json", default=str(REPO_ROOT / "paper_artifacts/stage4_v22_incomplete_preview.json"))
    parser.add_argument("--out-perf-tex", default=str(REPO_ROOT / "paper_artifacts/tables/ablations_v22_perf.tex"))
    parser.add_argument("--out-telem-tex", default=str(REPO_ROOT / "paper_artifacts/tables/ablations_v22_telemetry.tex"))
    parser.add_argument("--allow-incomplete-preview", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_tests()
        return

    complete, errors, data = audit_and_load_v22(Path(args.v22_dir))
    if not complete:
        preview = build_incomplete_preview(data, errors)
        _atomic_write_text(Path(args.preview_json), json.dumps(preview, indent=2, ensure_ascii=False) + "\n")
        print(f"Status: INCOMPLETE_PREVIEW ({preview['completed_run_count']}/{preview['expected_run_count']} runs; errors: {len(errors)})")
        if not args.allow_incomplete_preview:
            raise SystemExit(1)
        return

    metrics = compute_metrics_table(data)
    out = {
        "kind": "stage4_v22_audited_metrics",
        "complete": True,
        "status_stamp": "COMPLETE",
        "checkout_identity": data.get("_checkout_identity"),
        "audit_errors": [],
        "metrics": metrics,
    }
    _atomic_write_text(Path(args.out_json), json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    render_v22_performance_table(metrics, Path(args.out_perf_tex))
    render_v22_telemetry_table(metrics, Path(args.out_telem_tex))
    print(f"Status: COMPLETE — exported audited metrics to {args.out_json}, {args.out_perf_tex}, and {args.out_telem_tex}")


if __name__ == "__main__":
    main()
