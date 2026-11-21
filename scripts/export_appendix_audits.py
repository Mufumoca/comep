"""Export routing_latency_audit.json, reproducibility_mapping.json, and deployment_audit.json
deterministically from verified frozen campaign artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from comep.core import _ACTION_CLAUSE_STARTERS
from campaign_source_resolver import (
    STAGE3_CAMPAIGN_SPECS,
    STAGE4_ABLATION_SPEC,
    compute_six_backbone_aggregates,
    resolve_and_verify_campaign_dir,
    sha256_file,
    verify_and_load_stage3_campaign,
    verify_source_hashes,
)


def export_routing_latency_audit(loaded_stage3: list[dict]) -> dict:
    stage4_req = [STAGE4_ABLATION_SPEC["pinned_hashes"].keys()]
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
        "kind": "routing_latency_audit",
        "timer_scope": "Wall-clock duration of manager.choose(...) recorded as route_seconds in episodes.jsonl by execute_worker (comep/campaign.py:358; query/context extraction occurs prior to timer start). Covers relevance gating and two-element lookahead bundle selection. Post-episode parameter refitting (manager.observe at comep/campaign.py:440), optional post-episode rule distillation on successful episodes (comep/campaign.py:464), and bank pruning occur after episode completion outside this timer and are not timed separately in saved run logs.",
        "caveat": "Descriptive wall-clock measurements under the concurrent worker loads and host/Colab execution environments of each campaign; not an isolated single-thread microbenchmark.",
        "rows": rows,
        "source_hashes": source_hashes,
    }
    out_path = REPO_ROOT / "paper_artifacts" / "routing_latency_audit.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def export_reproducibility_mapping(loaded_stage3: list[dict], verified_src: dict[str, str]) -> dict:
    starters_sorted = sorted(_ACTION_CLAUSE_STARTERS)
    assert len(starters_sorted) == 31, f"Expected 31 starters, got {len(starters_sorted)}"
    by_key = {c["spec"]["key"]: c for c in loaded_stage3}
    agg = compute_six_backbone_aggregates(loaded_stage3)

    payload = {
        "kind": "reproducibility_mapping",
        "description": "Exact mapping from paper mathematical symbols, hyperparameter names, and six-backbone campaign descriptions to codebase symbols, CLI flags, status codes, verified source hashes, and archived output directories.",
        "verified_source_hashes": verified_src,
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
            "status_code_meanings": {
                "applied": "LLM critic call submitted and returned valid JSON operations that modified the memory bank.",
                "skipped_truncated": "LLM critic call submitted, but the model output was truncated at max_output_tokens and no bank edits were applied.",
                "skipped_malformed": "LLM critic call submitted and completed without length truncation, but returned invalid/unparseable JSON operations and no bank edits were applied.",
                "no_change": "LLM critic call submitted and completed, returning zero effective bank changes.",
                "skipped_oversize / skipped_oversize_prompt": "Reflection opportunity skipped prior to API submission because the serialized trajectory reflection prompt exceeded the 20,000-byte limit (0 LLM calls made).",
            },
        },
        "six_backbone_descriptive_aggregates": agg,
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
            "stage3_mimo_v26_flash_v22": {
                "output_dir": "outputs/stage3-mimo-v26-flash-v22",
                "resolved_dir": by_key["mimo_v26_flash_v22"]["resolved_dir"],
                "stage3_main_summary_sha256": by_key["mimo_v26_flash_v22"]["summary_sha256"],
                "gateway_provenance_sha256": by_key["mimo_v26_flash_v22"]["spec"]["pinned_hashes"]["gateway_provenance.json"],
                "orchestrator_retry_ledger_sha256": by_key["mimo_v26_flash_v22"]["spec"]["pinned_hashes"]["orchestrator_retry_ledger.json"],
                "endpoint_model": "opencode-go-2/mimo-v2.6-flash",
                "vendor_canonical_id": "xiaomi/mimo-v2.6-flash",
                "gateway": {"software": "magpie 0.1.4366", "provider_id": "opencode-go-2", "noStandIn": True, "cli_fallback_disabled": True},
                "request_format_adaptation": "Endpoint-verified wire adaptation in comep/client.py sending reasoning={'enabled': False} (without 'thinking' key) on http://127.0.0.1:3425/v1 -> opencode-go-2/mimo-v2.6-flash when thinking='disabled'; 0 non-zero and 0 missing reasoning_tokens across all 89,934 raw response records.",
                "accounting_audit": {
                    "canonical_logical_calls": 89919,
                    "canonical_live_responses": 78189,
                    "cache_hits": 11730,
                    "raw_api_jsonl_rows": 89934,
                    "rolled_back_pre_resume_responses": 15,
                    "resumed_run": "alfworld/full-memory_seed2 (paused at episode=85, resumed as dispatch 2; 0 transient orchestrator retries)",
                    "physical_http_started": 78226,
                    "physical_http_ok": 78204,
                    "physical_http_network_error": 21,
                    "physical_http_unfinished": 1,
                    "canonical_finish_reason_length": 1918,
                    "observed_action_format_retries": 327,
                    "observed_distillation_format_retries": 0,
                    "observed_action_parse_fallbacks": 1,
                    "usage_complete_runs": 48,
                    "http_usage_complete_false_runs": 18,
                    "total_provider_reported_cost_usd": None,
                    "billing_note": "OpenCode Go subscription accounting; configured prices are uncached catalog equivalents, not actual billed charges.",
                },
                "settings": {"warmup_gate": True, "max_format_retries": 2, "use_shared_cache": True, "observed_format_retries": 327},
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
    out_path = REPO_ROOT / "paper_artifacts" / "reproducibility_mapping.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def export_deployment_audit(loaded_stage3: list[dict]) -> dict:
    prov_dir = REPO_ROOT / "paper_artifacts" / "deployment_provenance"
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
        p = prov_dir / fn
        if not p.is_file():
            raise FileNotFoundError(f"Missing archived provenance file {p}")
        archived_files[fn] = {
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }

    by_key = {c["spec"]["key"]: c for c in loaded_stage3}
    glm_base = Path(by_key["glm53_flash_v22"]["resolved_dir"])
    mimo_base = Path(by_key["mimo_v26_flash_v22"]["resolved_dir"])
    glm_prov = json.loads((glm_base / "gateway_provenance.json").read_text(encoding="utf-8"))
    mimo_prov = json.loads((mimo_base / "gateway_provenance.json").read_text(encoding="utf-8"))

    payload = {
        "kind": "additional_backbone_deployment_provenance_audit",
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
                        "chat_template_kwargs": {
                            "enable_thinking": False,
                        },
                        "logit_bias": {
                            "98": -100,
                            "100": -100,
                            "101": -100,
                            "45518": -100,
                        },
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
            "mimo_v26_flash_v22": {
                "output_dir": "outputs/stage3-mimo-v26-flash-v22",
                "client_model_id": mimo_prov["frozen_invariants"]["client_model_id"],
                "vendor_canonical_id": mimo_prov["frozen_invariants"]["vendor_canonical_id"],
                "gateway_software": mimo_prov["gateway"]["software"],
                "gateway_binary_sha256": mimo_prov["gateway"]["binary_sha256"],
                "listen_base_url": mimo_prov["gateway"]["listen_base_url"],
                "provider_id": mimo_prov["gateway"]["provider_id"],
                "provider_config_sha256": mimo_prov["gateway"]["provider_config_sha256"],
                "provider_api_key": "<REDACTED>",
                "no_stand_in_verified": bool(mimo_prov["gateway"]["no_stand_in_verified"]),
                "cli_no_fallback_verified": bool(mimo_prov["gateway"]["cli_no_fallback_verified"]),
                "request_format_adaptation": mimo_prov["request_format_adaptation"],
                "reasoning_token_audit": "0 non-zero and 0 missing reasoning_tokens across all 89,934 raw response records.",
                "resume_and_http_audit": "89,919 canonical logical calls = 78,189 canonical live responses + 11,730 cache hits; raw api.jsonl has 89,934 rows including 15 rolled-back pre-resume responses from intentional checkpoint pause/resume at episode 85 on alfworld/full-memory_seed2 (0 transient orchestrator retries). Physical HTTP: 78,226 started = 78,204 ok + 21 network_error + 1 unfinished; 18/48 runs have http_usage_complete=false (campaign total_provider_reported_cost_usd=null), 48/48 runs have logical usage_complete=true.",
                "billing_and_price_semantics": mimo_prov["billing_and_price_semantics"]["accounting_note"],
                "stage3_main_summary_sha256": by_key["mimo_v26_flash_v22"]["summary_sha256"],
                "gateway_provenance_sha256": sha256_file(mimo_base / "gateway_provenance.json"),
                "orchestrator_retry_ledger_sha256": sha256_file(mimo_base / "orchestrator_retry_ledger.json"),
                "total_runs": 48,
                "total_episodes": 3960,
                "unique_alfworld_tasks_across_3_seeds": by_key["mimo_v26_flash_v22"]["unique_alfworld_tasks"],
                "unique_scienceworld_tasks_across_3_seeds": by_key["mimo_v26_flash_v22"]["unique_scienceworld_tasks"],
                "format_retry_calls": by_key["mimo_v26_flash_v22"]["total_format_retry_calls"],
            },
        },
        "archived_files": archived_files,
    }
    out_path = prov_dir / "deployment_audit.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    verified_src = verify_source_hashes()
    loaded_stage3 = [verify_and_load_stage3_campaign(spec) for spec in STAGE3_CAMPAIGN_SPECS]
    export_routing_latency_audit(loaded_stage3)
    export_reproducibility_mapping(loaded_stage3, verified_src)
    export_deployment_audit(loaded_stage3)
    print("Exported verified routing_latency_audit.json, reproducibility_mapping.json, and deployment_audit.json")
