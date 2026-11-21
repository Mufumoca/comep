"""CLI entrypoint for Stage 2 calibration, Stage 3 main benchmarks, and Stage 4 ablations + 4-cell causal checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from comep.campaign import (
    ABLATION_SPECS,
    MAIN_METHODS,
    REPO_ROOT,
    build_warmstart_pool,
    default_api_dict,
    run_counterfactual_4cell,
    run_grid,
)
from comep.experiments import write_json


def _write_merged_summary(summary_path: Path, new_results: dict) -> dict:
    merged = {}
    if summary_path.is_file():
        merged = json.loads(summary_path.read_text(encoding="utf-8"))
    merged.update(new_results)
    write_json(summary_path, merged)
    return merged


def extract_top_discovered_pairs(
    grid_dir: Path,
    top_k_pos: int = 1,
    top_k_neg: int = 1,
    seeds: list[int] | None = None,
) -> list[dict]:
    """Read CoMeP's final state.json (and episodes.jsonl for already-pruned conflicts) across seeds to pick top pairs."""
    from collections import Counter
    from itertools import combinations

    pos_candidates = []
    neg_candidates = []
    if seeds is not None:
        state_files = [grid_dir / f"comep_seed{s}" / "state.json" for s in seeds if (grid_dir / f"comep_seed{s}" / "state.json").is_file()]
    else:
        state_files = sorted(grid_dir.glob("comep_seed*/state.json"))
    for state_file in state_files:
        state = json.loads(state_file.read_text(encoding="utf-8"))
        archived = {a["memory"]["identifier"]: a["memory"]["text"] for a in state.get("archive", []) if "memory" in a}
        memories = {m["identifier"]: m["text"] for m in state.get("memories", [])}
        all_texts = {**archived, **memories}

        ep_file = state_file.parent / "episodes.jsonl"
        pair_stats: dict[tuple[str, str], list[int]] = {}
        pair_ctxs: dict[tuple[str, str], Counter] = {}
        if ep_file.is_file():
            for line in ep_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                for a_id, b_id in combinations(sorted(rec.get("selected", [])), 2):
                    if a_id in all_texts and b_id in all_texts:
                        pair_stats.setdefault((a_id, b_id), []).append(int(rec["reward"]))
                        pair_ctxs.setdefault((a_id, b_id), Counter())[str(rec.get("context", ""))] += 1

        for p in state.get("pairs", []):
            ids = p.get("identifiers", p.get("pair", []))
            if len(ids) != 2 or ids[0] not in all_texts or ids[1] not in all_texts:
                continue
            key = tuple(sorted((ids[0], ids[1])))
            mean_val = float(p.get("mean", 0.0))
            rews = pair_stats.get(key, [])
            cnt = max(int(p.get("count", 0)), len(rews))
            if cnt < 2:
                continue
            dom_ctx = pair_ctxs[key].most_common(1)[0][0] if key in pair_ctxs and pair_ctxs[key] else ""
            entry = {
                "id_a": ids[0],
                "id_b": ids[1],
                "rule_a": all_texts[ids[0]],
                "rule_b": all_texts[ids[1]],
                "learned_s_ij": round(mean_val, 4),
                "count": cnt,
                "co_wins": sum(rews),
                "selection_source": "posterior_s_ij",
                "target_context": dom_ctx,
            }
            if mean_val > 0.08:
                pos_candidates.append({**entry, "kind": "positive_synergy"})
            elif mean_val < -0.02:
                neg_candidates.append({**entry, "kind": "negative_conflict"})

        active_pair_keys = {tuple(sorted(p.get("identifiers", p.get("pair", [])))) for p in state.get("pairs", [])}
        for (a_id, b_id), rews in pair_stats.items():
            if (a_id, b_id) in active_pair_keys:
                continue
            if len(rews) >= 2 and (sum(rews) / len(rews)) <= 0.25 and (a_id in archived or b_id in archived):
                dom_ctx = pair_ctxs[(a_id, b_id)].most_common(1)[0][0] if pair_ctxs.get((a_id, b_id)) else ""
                neg_candidates.append(
                    {
                        "id_a": a_id,
                        "id_b": b_id,
                        "rule_a": all_texts[a_id],
                        "rule_b": all_texts[b_id],
                        "learned_s_ij": None,
                        "count": len(rews),
                        "co_wins": sum(rews),
                        "selection_source": f"archived_co_win_{sum(rews)}/{len(rews)}",
                        "target_context": dom_ctx,
                        "kind": "negative_conflict",
                    }
                )
    pos_candidates.sort(key=lambda x: (-(1 if x["count"] >= 3 else 0), -(x["learned_s_ij"] or 0.0), -x["count"]))
    neg_candidates.sort(
        key=lambda x: (
            -(
                1
                if (x["selection_source"] == "posterior_s_ij" and (x["learned_s_ij"] or 0.0) < -0.08)
                else (0 if x["selection_source"] != "posterior_s_ij" else -1)
            ),
            -(1 if x["count"] >= 3 else 0),
            (x["learned_s_ij"] if x["learned_s_ij"] is not None else 0.0),
            -x["count"],
        )
    )
    chosen = []
    seen = set()
    for item in pos_candidates:
        sig = tuple(sorted((item["rule_a"], item["rule_b"])))
        if sig not in seen and len([c for c in chosen if c["kind"] == "positive_synergy"]) < top_k_pos:
            seen.add(sig)
            chosen.append(item)
    for item in neg_candidates:
        sig = tuple(sorted((item["rule_a"], item["rule_b"])))
        if sig not in seen and len([c for c in chosen if c["kind"] == "negative_conflict"]) < top_k_neg:
            seen.add(sig)
            chosen.append(item)
    return chosen


def main():
    parser = argparse.ArgumentParser(description="CoMeP v2 Stage 2-4 Benchmark Runner")
    parser.add_argument("stage", choices=["stage2-pool", "stage2-dev", "stage3-main", "stage4-ablations", "stage4-4cell"])
    parser.add_argument("--env", choices=["alfworld", "scienceworld", "both"], default="both")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--episodes", type=int, default=None, help="Optional override for episode count")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Optional override for seed list")
    parser.add_argument("--methods", nargs="+", default=None, help="Optional override for method list")
    parser.add_argument("--warmup-gate", action="store_true", help="Enable Warm-up Gate (G_t(i,j)) in manager_overrides for stage3-main")
    parser.add_argument("--use-shared-cache", action="store_true", help="Enable per-replicate request-level shared cache in stage3-main")
    parser.add_argument("--no-shared-cache", action="store_true", help="Disable per-replicate request-level shared cache in stage4-ablations")
    parser.add_argument("--protocol-version", default="v2.2", help="Protocol version tag for request cache and manifests")
    parser.add_argument("--source-grid", default=None, help="Grid directory for stage4-4cell pair discovery")
    args = parser.parse_args()

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    api = default_api_dict(base_url=args.base_url, model=args.model, key_env=args.key_env)
    envs = ["alfworld", "scienceworld"] if args.env == "both" else [args.env]

    results = {}
    if args.stage == "stage2-pool":
        for env_name in envs:
            if env_name == "alfworld":
                manifest = REPO_ROOT / "configs" / "alfworld_train_36.json"
                pool_path = REPO_ROOT / "configs" / "alfworld_warmstart_pool.json"
                eps = args.episodes or 24
                steps = 30
            else:
                manifest = REPO_ROOT / "configs" / "scienceworld_train_30.json"
                pool_path = REPO_ROOT / "configs" / "scienceworld_warmstart_pool.json"
                eps = args.episodes or 25
                steps = 35
            results[env_name] = build_warmstart_pool(
                env_name=env_name,
                train_manifest=manifest,
                output_dir=out_root,
                pool_out_path=pool_path,
                api_dict=api,
                episodes=eps,
                max_steps=steps,
                target_pool_size=20,
            )
        _write_merged_summary(out_root / "stage2_pool_summary.json", results)

    elif args.stage == "stage2-dev":
        seeds = args.seeds or [0]
        for env_name in envs:
            if env_name == "alfworld":
                manifest = REPO_ROOT / "configs" / "alfworld_dev_24.json"
                split = "eval_in_distribution"
                pool_path = REPO_ROOT / "configs" / "alfworld_warmstart_pool.json"
                eps = args.episodes or 24
                steps = 30
            else:
                manifest = REPO_ROOT / "configs" / "scienceworld_dev_25.json"
                split = "dev"
                pool_path = REPO_ROOT / "configs" / "scienceworld_warmstart_pool.json"
                eps = args.episodes or 25
                steps = 35
            results[env_name] = run_grid(
                stage_name="stage2-dev",
                env_name=env_name,
                split=split,
                manifest_path=manifest,
                initial_rules_path=pool_path,
                methods=MAIN_METHODS,
                seeds=seeds,
                episodes=eps,
                max_steps=steps,
                output_dir=out_root / env_name,
                workers=args.workers,
                api_dict=api,
                distill=True,
            )
        _write_merged_summary(out_root / "stage2_dev_summary.json", results)

    elif args.stage == "stage3-main":
        seeds = args.seeds or [0, 1, 2]
        main_methods = args.methods or MAIN_METHODS
        mgr_overrides = {"warmup_gate": True} if args.warmup_gate else {}
        cache_dir = (out_root / ".shared_request_cache") if args.use_shared_cache else None
        for env_name in envs:
            if env_name == "alfworld":
                manifest = REPO_ROOT / "configs" / "alfworld_test_ood_134.json"
                split = "eval_out_of_distribution"
                pool_path = REPO_ROOT / "configs" / "alfworld_warmstart_pool.json"
                eps = args.episodes or 134
                steps = 30
            else:
                manifest = REPO_ROOT / "configs" / "scienceworld_test_100.json"
                split = "test"
                pool_path = REPO_ROOT / "configs" / "scienceworld_warmstart_pool.json"
                eps = args.episodes or 100
                steps = 35
            results[env_name] = run_grid(
                stage_name="stage3-main",
                env_name=env_name,
                split=split,
                manifest_path=manifest,
                initial_rules_path=pool_path,
                methods=main_methods,
                seeds=seeds,
                episodes=eps,
                max_steps=steps,
                output_dir=out_root / env_name,
                workers=args.workers,
                api_dict=api,
                distill=True,
                manager_overrides=mgr_overrides,
                shared_cache_dir=cache_dir,
                protocol_version=args.protocol_version,
            )
        _write_merged_summary(out_root / "stage3_main_summary.json", results)

    elif args.stage == "stage4-ablations":
        seeds = args.seeds or [0, 1]
        ablation_methods = args.methods or list(ABLATION_SPECS.keys())
        cache_dir = None if args.no_shared_cache else (out_root / ".shared_request_cache")
        for env_name in envs:
            if env_name == "alfworld":
                manifest = REPO_ROOT / "configs" / "alfworld_test_ood_134.json"
                split = "eval_out_of_distribution"
                pool_path = REPO_ROOT / "configs" / "alfworld_warmstart_pool.json"
                eps = args.episodes or 50
                steps = 30
            else:
                manifest = REPO_ROOT / "configs" / "scienceworld_test_100.json"
                split = "test"
                pool_path = REPO_ROOT / "configs" / "scienceworld_warmstart_pool.json"
                eps = args.episodes or 50
                steps = 35
            results[env_name] = run_grid(
                stage_name="stage4-ablations",
                env_name=env_name,
                split=split,
                manifest_path=manifest,
                initial_rules_path=pool_path,
                methods=ablation_methods,
                seeds=seeds,
                episodes=eps,
                max_steps=steps,
                output_dir=out_root / env_name,
                workers=args.workers,
                api_dict=api,
                distill=True,
                shared_cache_dir=cache_dir,
                protocol_version=args.protocol_version,
            )
        _write_merged_summary(out_root / "stage4_ablations_summary.json", results)

    elif args.stage == "stage4-4cell":
        if not args.source_grid:
            raise ValueError("--source-grid required for stage4-4cell")
        src_root = Path(args.source_grid)
        alf_ctx_substr = {
            "alfworld:cool": "pick_cool_then_place",
            "alfworld:heat": "pick_heat_then_place",
            "alfworld:clean": "pick_clean_then_place",
            "alfworld:two_obj": "pick_two_obj",
            "alfworld:examine": "look_at_obj",
            "alfworld:put": "pick_and_place_simple",
        }
        for env_name in envs:
            grid_dir = src_root / env_name
            seed0_ep_file = grid_dir / "comep_seed0" / "episodes.jsonl"
            used_task_ids: set[str] = set()
            if seed0_ep_file.is_file():
                for line in seed0_ep_file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        used_task_ids.add(str(json.loads(line).get("task_id", "")))
            pairs = extract_top_discovered_pairs(grid_dir, top_k_pos=1, top_k_neg=1, seeds=[0])
            env_reports = []
            for idx, pair_entry in enumerate(pairs):
                pair_out_dir = out_root / env_name / f"pair_{idx}_{pair_entry['kind']}"
                pair_out_dir.mkdir(parents=True, exist_ok=True)
                target_ctx = pair_entry.get("target_context", "")
                if env_name == "alfworld":
                    base_manifest = REPO_ROOT / "configs" / "alfworld_test_ood_134.json"
                    split = "eval_out_of_distribution"
                    steps = 30
                    raw = json.loads(base_manifest.read_text(encoding="utf-8"))
                    games = raw["games"] if isinstance(raw, dict) and "games" in raw else raw
                    sub = alf_ctx_substr.get(target_ctx, "")
                    if used_task_ids:
                        disjoint_tail = [g for g in games if (g["game"] if isinstance(g, dict) else g) not in used_task_ids]
                    else:
                        disjoint_tail = games[60:]
                    matched = [g for g in disjoint_tail if sub and sub in (g["game"] if isinstance(g, dict) else g)]
                    chosen_items = matched if matched else disjoint_tail
                    manifest = pair_out_dir / "task_matched_manifest.json"
                    if not manifest.is_file():
                        write_json(manifest, chosen_items)
                    else:
                        chosen_items = json.loads(manifest.read_text(encoding="utf-8"))
                    eps = min(args.episodes or 10, len(chosen_items))
                else:
                    base_manifest = REPO_ROOT / "configs" / "scienceworld_test_100.json"
                    split = "test"
                    steps = 35
                    tasks = json.loads(base_manifest.read_text(encoding="utf-8"))
                    if used_task_ids:
                        disjoint_tail = [
                            t for t in tasks
                            if f"{t['task']}:{t['variation']}:{t.get('split', 'test')}" not in used_task_ids
                        ]
                    else:
                        disjoint_tail = tasks[50:]
                    matched = [t for t in disjoint_tail if target_ctx and t.get("task") == target_ctx]
                    chosen_items = matched if matched else disjoint_tail
                    manifest = pair_out_dir / "task_matched_manifest.json"
                    if not manifest.is_file():
                        write_json(manifest, chosen_items)
                    else:
                        chosen_items = json.loads(manifest.read_text(encoding="utf-8"))
                    eps = min(args.episodes or 10, len(chosen_items))
                rep = run_counterfactual_4cell(
                    env_name=env_name,
                    split=split,
                    manifest_path=manifest,
                    pair_entry=pair_entry,
                    episodes=eps,
                    max_steps=steps,
                    output_dir=pair_out_dir,
                    workers=min(3, args.workers),
                    api_dict=api,
                )
                for meta_k in ("target_context", "selection_source", "count", "co_wins", "id_a", "id_b"):
                    if meta_k in pair_entry:
                        rep[meta_k] = pair_entry[meta_k]
                write_json(pair_out_dir / "counterfactual_summary.json", rep)
                env_reports.append(rep)
            results[env_name] = env_reports
        _write_merged_summary(out_root / "stage4_4cell_summary.json", results)

    print(json.dumps({"stage": args.stage, "completed_envs": list(results.keys())}, indent=2))


if __name__ == "__main__":
    main()
