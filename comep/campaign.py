"""End-to-end Stage 2–4 benchmark campaign, warm-start pool extraction, and causal 4-cell verification."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import numpy as np

from .client import APIConfig, BudgetExceeded, ChatClient, complete_and_validate, uses_verified_magpie_mimo_reasoning_field
from .core import Config, MemoryManager
from .environments import ALFWorldAdapter, ScienceWorldAdapter, load_alfworld_game_files
from .experiments import write_json
from .runner import (
    ACTION_FORMAT_CORRECTION_PROMPT,
    DISTILLATION_FORMAT_CORRECTION_PROMPT,
    AgentConfig,
    _parse_json_object,
    parse_action,
    validate_distillation_json,
)
from .text import LocalEncoder, TokenCounter, similarity


REPO_ROOT = Path(__file__).resolve().parent.parent
ALFWORLD_PYTHON = str(REPO_ROOT / ".venv-alfworld" / "bin" / "python")
SCIENCEWORLD_PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")
BGE_MODEL_PATH = str(REPO_ROOT / "models" / "bge-small-en-v1.5")
NLI_MODEL_PATH = str(REPO_ROOT / "models" / "deberta-v3-base-mnli-fever-anli")

MAIN_METHODS = [
    "no-memory",
    "full-memory",
    "individual",
    "mmr",
    "additive",
    "expel-derived-online",
    "evomemguard",
    "comep",
]

ABLATION_SPECS: dict[str, tuple[str, dict]] = {
    "comep": ("comep", {}),
    "comep-warmup-gate": ("comep", {"warmup_gate": True}),
    "additive": ("additive", {}),
    "comep-no-hard-conflicts": ("comep", {"hard_conflicts": False}),
    "comep-no-pair-optimism": ("comep", {"pair_ucb_scale": 0.0}),
    "comep-no-early-stop": ("comep", {"min_prob_gain_per_token": 0.0}),
    "comep-no-coalition-pruning": ("comep", {"coalition_pruning": False}),
    "comep-v2-online-sgd": ("comep", {"batch_laplace": False, "v2_routing_and_features": True}),
    "comep-sgd-v1": ("comep", {"batch_laplace": False, "v2_routing_and_features": False}),
}

ALFWORLD_ACTION_INSTRUCTIONS = (
    "Keep Thought under 25 words. Valid ALFWorld commands (never repeat the number; always `go to` a receptacle before opening, examining, or taking from it): "
    "look | inventory | go to cabinet 1 | open cabinet 1 | close cabinet 1 | "
    "take mug 1 from cabinet 1 | move mug 1 to countertop 1 | examine mug 1 | "
    "use desklamp 1 | clean mug 1 with sinkbasin 1 | heat mug 1 with microwave 1 | "
    "cool mug 1 with fridge 1. Substitute exact object and receptacle names/numbers seen in observations."
)

SCIENCEWORLD_ACTION_INSTRUCTIONS = (
    "Keep Thought under 25 words. ScienceWorld command templates (use exact observed names): "
    "look around | inventory | look at <obj> | open <door_or_container> | teleport to <room> | go to <room> | "
    "pick up <obj> | move <obj> to <container_or_device> | focus on <obj> | use <instrument> in inventory on <target> | "
    "connect <component A terminal> to <component B terminal> | activate <device> | deactivate <device> | wait1 | wait. "
    "Follow the task's required focus rules carefully."
)

DEFAULT_MANAGER_CONFIG = {
    "retrieval_tokens": 90,
    "bank_tokens": 560,
    "max_items": 3,
    "max_bank_items": 22,
    "prior_precision": 1.0,
    "pair_prior_precision": 1.2,
    "prior_mean": 0.25,
    "sampling_scale": 0.3,
    "exploration": 0.08,
    "redundancy": 0.15,
    "synergy": 1.0,
    "conflict_threshold": 0.2,
    "conflict_z": 1.645,
    "min_pair_observations": 3,
    "pruning_uncertainty": 0.25,
    "pair_lookahead": True,
    "hard_conflicts": True,
    "matched_pair_lookahead": True,
    "batch_laplace": True,
    "history_window": 600,
    "min_active_pair_count": 1,
    "max_active_pairs": 48,
    "top_pair_candidates": 6,
    "ucb_scale": 0.55,
    "pair_ucb_scale": 0.25,
    "min_prob_gain_per_token": 0.002,
    "normalize_relevance": True,
    "coalition_pruning": True,
    "nli_model": NLI_MODEL_PATH,
    "nli_conflict_threshold": 0.55,
}


def extract_routing_query(initial_text: str, env_name: str) -> str:
    """Extract the task goal statement from initial observation so room clutter does not drown query semantics."""
    lines = [ln.strip() for ln in initial_text.splitlines() if ln.strip()]
    if env_name == "alfworld":
        for ln in lines:
            if "your task is to:" in ln.lower():
                return ln
    elif env_name == "scienceworld":
        goal_lines = []
        capture = False
        for ln in lines:
            low = ln.lower()
            if "task description:" in low:
                capture = True
                continue
            if low.startswith(("this room is called", "this outside location is called", "here you see:")):
                break
            if capture or low.startswith("your task is"):
                goal_lines.append(ln)
        if goal_lines:
            return " ".join(goal_lines)
    return initial_text


def permute_items(items: list, seed: int) -> list:
    if seed == 0:
        return list(items)
    rng = np.random.default_rng(10_000 + int(seed))
    order = rng.permutation(len(items))
    return [items[int(i)] for i in order]


def build_manager(method_label: str, manager_cfg_dict: dict, seed: int, initial_rules: list[dict]) -> MemoryManager | object:
    base_method, overrides = ABLATION_SPECS.get(method_label, (method_label, {}))
    merged = {**DEFAULT_MANAGER_CONFIG, **manager_cfg_dict, **overrides}
    cfg = Config(**merged)
    encoder = LocalEncoder(BGE_MODEL_PATH)
    counter = TokenCounter("cl100k_base")
    if base_method == "expel-derived-online":
        from .critic import OnlineInsightManager

        return OnlineInsightManager(
            [(r["text"], 2) for r in initial_rules],
            config=cfg,
            encoder=encoder,
            counter=counter,
            max_prompt_bytes=20000,
            strict_format=False,
        )
    mgr = MemoryManager(base_method, cfg, seed=seed, encoder=encoder, counter=counter)
    for r in initial_rules:
        mgr.add(r["id"], r["text"], step=0, prune=False)
    mgr.prune()
    return mgr


def compute_spec_fingerprint(spec: dict) -> str:
    """Compute a deterministic SHA256 fingerprint over all resolved protocol, source, task, rule, manager, and API settings."""
    method_label = str(spec["method"])
    base_method, overrides = ABLATION_SPECS.get(method_label, (method_label, {}))
    merged_manager = {**DEFAULT_MANAGER_CONFIG, **(spec.get("manager") or {}), **overrides}
    resolved_manager = asdict(Config(**merged_manager))
    manifest_p = Path(spec["manifest_path"])
    manifest_sha = hashlib.sha256(manifest_p.read_bytes()).hexdigest() if manifest_p.is_file() else str(manifest_p)
    ordered_rules = [{"id": str(r.get("id", "")), "text": str(r.get("text", ""))} for r in spec.get("initial_rules", [])]
    distill = bool(spec.get("distill", True))
    if method_label in ("no-memory", "expel-derived-online"):
        distill = False
    api_raw = dict(spec.get("api") or {})
    api_raw.setdefault("base_url", "https://api.deepseek.com")
    api_raw.setdefault("model", "deepseek-flash")
    api_obj = APIConfig(**api_raw)
    source_hashes = {}
    for src_name in ("core.py", "client.py", "runner.py", "campaign.py"):
        src_p = REPO_ROOT / "comep" / src_name
        if src_p.is_file():
            source_hashes[src_name] = hashlib.sha256(src_p.read_bytes()).hexdigest()
    canonical = {
        "code_protocol": "comep-campaign-v2.2",
        "source_hashes": source_hashes,
        "env_name": str(spec["env_name"]),
        "split": str(spec["split"]),
        "manifest_sha256": manifest_sha,
        "seed": int(spec["seed"]),
        "episodes": int(spec["episodes"]),
        "max_steps": int(spec["max_steps"]),
        "history_turns": int(spec.get("history_turns", 8)),
        "max_rule_tokens": int(spec.get("max_rule_tokens", 32)),
        "admissible_limit": spec.get("admissible_limit", 0),
        "action_instructions": str(spec["action_instructions"]),
        "method_label": method_label,
        "base_method": base_method,
        "distill": distill,
        "ordered_initial_rules": ordered_rules,
        "resolved_manager": resolved_manager,
        "api_protocol": {
            "base_url": api_obj.base_url.rstrip("/"),
            "model": api_obj.model,
            "thinking": api_obj.thinking,
            "max_output_tokens": api_obj.max_output_tokens,
            "max_format_retries": api_obj.max_format_retries,
            "protocol_version": api_obj.protocol_version,
            "shared_cache_enabled": bool(api_obj.cache_dir),
            "cache_dir": str(api_obj.cache_dir) if api_obj.cache_dir else None,
            "cache_namespace": api_obj.cache_namespace,
        },
    }
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def execute_worker(spec_path: str | Path) -> dict:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    expected_fp = compute_spec_fingerprint(spec)
    out_dir = Path(spec["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    manifest_out_path = out_dir / "manifest.json"
    ckpt_path = out_dir / "checkpoint.pkl"
    episodes = int(spec["episodes"])

    if manifest_out_path.is_file():
        existing_man = json.loads(manifest_out_path.read_text(encoding="utf-8"))
        old_fp = existing_man.get("spec_fingerprint")
        if old_fp and old_fp != expected_fp:
            raise RuntimeError(f"Manifest spec_fingerprint mismatch in {out_dir}: {old_fp} != {expected_fp}")

    if summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        sum_fp = existing.get("spec_fingerprint")
        if sum_fp != expected_fp:
            raise RuntimeError(f"Refusing to reuse completed summary in {out_dir} with missing or mismatched spec_fingerprint")
        if existing.get("status") == "completed" and existing.get("completed_episodes") == episodes:
            state_path = out_dir / "state.json"
            if state_path.is_file() and spec.get("distill") is False and spec.get("initial_rules"):
                st = json.loads(state_path.read_text(encoding="utf-8"))
                st_texts = [m["text"] for m in st.get("memories", [])]
                req_texts = [r["text"] for r in spec.get("initial_rules", [])]
                if st_texts != req_texts:
                    raise RuntimeError(f"Existing cell state memories in {out_dir} do not match requested initial_rules")
            return existing

    env_name = spec["env_name"]
    split = spec["split"]
    seed = int(spec["seed"])
    max_steps = int(spec["max_steps"])
    method_label = spec["method"]
    initial_rules = spec.get("initial_rules", [])
    distill = bool(spec.get("distill", True))
    if method_label in ("no-memory", "expel-derived-online"):
        distill = False

    agent = AgentConfig(
        episodes=episodes,
        max_steps=max_steps,
        history_turns=int(spec.get("history_turns", 8)),
        distill=distill,
        max_rule_tokens=int(spec.get("max_rule_tokens", 32)),
        seed=seed,
        admissible_limit=spec.get("admissible_limit", 0),
        action_instructions=spec["action_instructions"],
        evidence_distillation=False,
        fallback_action="look" if env_name == "alfworld" else "look around",
    )
    api = APIConfig(**spec["api"])
    client = ChatClient(api, out_dir / "api.jsonl")

    if ckpt_path.is_file():
        with ckpt_path.open("rb") as fh:
            ckpt = pickle.load(fh)
        if ckpt.get("spec_fingerprint") != expected_fp:
            raise RuntimeError(f"Refusing to resume checkpoint in {out_dir} with missing or mismatched spec_fingerprint")
        start_ep = int(ckpt["next_episode"])
        manager = ckpt["manager"]
        if hasattr(manager, "max_prompt_bytes"):
            manager.max_prompt_bytes = min(int(manager.max_prompt_bytes), 20000)
        records = list(ckpt["records"])
        with (out_dir / "episodes.jsonl").open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, allow_nan=False) + "\n")
        for aux_name in ("transitions.jsonl", "distillation.jsonl", "critic.jsonl"):
            aux_p = out_dir / aux_name
            if aux_p.is_file():
                kept = [
                    ln for ln in aux_p.read_text(encoding="utf-8").splitlines()
                    if ln.strip() and int(json.loads(ln).get("episode", -1)) < start_ep
                ]
                with aux_p.open("w", encoding="utf-8") as fh:
                    for ln in kept:
                        fh.write(ln + "\n")
        client.calls = int(ckpt["client_calls"])
        client.cache_hits = int(ckpt.get("cache_hits", 0))
        client.prompt_tokens = int(ckpt["prompt_tokens"])
        client.completion_tokens = int(ckpt["completion_tokens"])
        client.purpose_calls = dict(ckpt["purpose_calls"])
        # Non-rollbackable live HTTP usage is authoritative from append-only ledger on disk
        client.sync_http_ledger()
    else:
        start_ep = 0
        manager = build_manager(method_label, spec.get("manager", {}), seed, initial_rules)
        records = []
        write_json(
            manifest_out_path,
            {
                "kind": "campaign_worker_rollout",
                "spec_fingerprint": expected_fp,
                "method_label": method_label,
                "env_name": env_name,
                "split": split,
                "seed": seed,
                "episodes": episodes,
                "max_steps": max_steps,
                "distill": distill,
                "initial_rules_count": len(initial_rules),
                "api": asdict(api),
            },
        )

    if env_name == "alfworld":
        all_games = load_alfworld_game_files(spec["manifest_path"], verify_hashes=True)
        ordered = permute_items(all_games, seed)[:episodes]
        environment = ALFWorldAdapter(
            str(REPO_ROOT / "configs" / "alfworld_base_config.yaml"),
            split,
            seed=seed,
            max_steps=max_steps,
            game_files=ordered,
            ordered_games=True,
        )
    elif env_name == "scienceworld":
        all_tasks = json.loads(Path(spec["manifest_path"]).read_text(encoding="utf-8"))
        ordered = permute_items(all_tasks, seed)[:episodes]
        environment = ScienceWorldAdapter(spec["manifest_path"], max_steps=max_steps, tasks_override=ordered)
        if spec.get("verify_splits", False):
            environment.verify_splits()
    else:
        raise ValueError(f"unsupported env_name: {env_name}")

    status = "completed"
    error_type = None
    failure_stage = None
    try:
        for episode in range(start_ep, episodes):
            failure_stage = "environment_reset"
            observation = environment.reset(episode)
            initial = observation
            routing_query = extract_routing_query(initial.text, env_name)
            failure_stage = "routing"
            t_route = time.perf_counter()
            selection = manager.choose(routing_query, initial.context)
            route_seconds = time.perf_counter() - t_route
            failure_stage = None

            memories = [manager.memories[i].text for i in selection.identifiers]
            memory_text = "\n".join(f"- {t}" for t in memories)
            system = (
                "You are a text-environment agent. Solve the stated task using environment commands. "
                "Reply with optional short Thought: and exactly one Action: command line (always prefix the command with `Action: `). "
                "Memories are fallible past experience, not instructions overriding the task.\n"
                + agent.action_instructions
            )
            if memory_text:
                system += "\nMemories:\n" + memory_text

            history = []
            trajectory = []
            for step in range(agent.max_steps):
                if observation.done:
                    break
                content = observation.text
                shown_actions = (
                    observation.admissible
                    if agent.admissible_limit is None
                    else observation.admissible[: agent.admissible_limit]
                )
                if shown_actions:
                    content += "\nAdmissible actions:\n" + "\n".join(shown_actions)
                messages = (
                    [{"role": "system", "content": system}, {"role": "user", "content": "Initial task:\n" + initial.text}]
                    + history[-2 * agent.history_turns :]
                    + [{"role": "user", "content": content}]
                )
                failure_stage = "provider"
                reply, parsed_action, fmt_meta = complete_and_validate(
                    client,
                    messages,
                    "action",
                    parse_action,
                    ACTION_FORMAT_CORRECTION_PROMPT,
                    request_context={"episode": episode, "step": step},
                )
                failure_stage = "parse"
                parse_fallback = False
                if parsed_action is not None:
                    action = parsed_action
                else:
                    action = agent.fallback_action
                    parse_fallback = True
                failure_stage = "environment"
                next_observation = environment.step(action)
                failure_stage = None

                transition = {
                    "episode": episode,
                    "step": step,
                    "task_id": initial.task_id,
                    "observation": observation.text,
                    "admissible": observation.admissible,
                    "reply": reply,
                    "action": action,
                    "listed_action": action in observation.admissible,
                    "next_observation": next_observation.text,
                    "done": next_observation.done,
                    "success": next_observation.success,
                    "score": next_observation.score,
                    "previous_score": observation.score,
                    "format_attempts": fmt_meta["format_attempts"],
                    "format_retries": fmt_meta["format_retries"],
                }
                if parse_fallback:
                    transition["parse_fallback"] = True
                with (out_dir / "transitions.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(transition, allow_nan=False) + "\n")
                trajectory.append(transition)
                assistant_hist = reply if "action:" in reply.lower() else f"{reply.strip()}\nAction: {action}".strip()
                history.extend([{"role": "user", "content": content}, {"role": "assistant", "content": assistant_hist}])
                observation = next_observation

            reward = int(observation.success)
            termination = "success" if observation.success else ("environment_terminal" if observation.done else "step_limit")
            prediction = manager.observe(selection, reward)
            record = {
                "episode": episode,
                "task_id": initial.task_id,
                "context": initial.context,
                "reward": reward,
                "score": observation.score,
                "steps": len(trajectory),
                "truncated": not observation.done,
                "termination": termination,
                "selected": selection.identifiers,
                "retrieval_tokens": selection.tokens,
                "route_seconds": route_seconds,
                "predicted_probability": prediction,
                "explored": selection.explored,
            }
            records.append(record)
            with (out_dir / "episodes.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, allow_nan=False) + "\n")

            if hasattr(manager, "after_episode"):
                failure_stage = "insight_critic"
                manager.after_episode(record, trajectory, client, out_dir)
                failure_stage = None
            elif agent.distill and getattr(manager, "method", "") != "no-memory":
                dist_fmt_meta = {"format_attempts": 0, "format_retries": 0, "format_fallback": False}
                if reward == 1:
                    distilled_traj = [
                        {
                            "step": tr["step"],
                            "action": tr["action"],
                            "next_observation": tr["next_observation"][:280],
                            "score": tr["score"],
                        }
                        for tr in trajectory[-12:]
                    ]
                    prompt = (
                        f"Context: {initial.context}\nTask: {initial.text[:300]}\n"
                        f"Winning trajectory tail: {json.dumps(distilled_traj, ensure_ascii=False)}\n"
                        "Extract ONE task-scoped procedural rule (16 to 28 tokens, starting with 'For <task-type> tasks, ...') "
                        "preserving the exact command verbs that solved this task while abstracting specific item/color names. "
                        "If trivial or generic, return {\"rule\": null}."
                    )
                    reply, parsed_dist, dist_fmt_meta = complete_and_validate(
                        client,
                        [
                            {"role": "system", "content": "Extract one task-scoped procedural memory rule (16-28 tokens); return JSON only."},
                            {"role": "user", "content": prompt},
                        ],
                        "distillation",
                        validate_distillation_json,
                        DISTILLATION_FORMAT_CORRECTION_PROMPT,
                        request_context={"episode": episode},
                    )
                    distilled = parsed_dist if parsed_dist is not None else {"rule": None}
                else:
                    distilled = {"rule": None}
                rule = distilled.get("rule")
                accepted = (
                    isinstance(rule, str)
                    and bool(rule.strip())
                    and 15 <= manager.counter.count(rule.strip()) <= agent.max_rule_tokens
                )
                removed = manager.add(f"distilled-s{seed}-e{episode}", rule.strip(), episode + 1) if accepted else []
                with (out_dir / "distillation.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "episode": episode,
                                "rule": rule,
                                "accepted": accepted,
                                "removed": removed,
                                "retained_after_pruning": accepted and f"distilled-s{seed}-e{episode}" in manager.memories,
                                "format_attempts": dist_fmt_meta["format_attempts"],
                                "format_retries": dist_fmt_meta["format_retries"],
                                "format_fallback": dist_fmt_meta["format_fallback"],
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )

            # Save atomic episode-level checkpoint
            tmp_ckpt = out_dir / "checkpoint.pkl.tmp"
            with tmp_ckpt.open("wb") as fh:
                pickle.dump(
                    {
                        "spec_fingerprint": expected_fp,
                        "next_episode": episode + 1,
                        "manager": manager,
                        "records": records,
                        "client_calls": client.calls,
                        "cache_hits": client.cache_hits,
                        "prompt_tokens": client.prompt_tokens,
                        "completion_tokens": client.completion_tokens,
                        "purpose_calls": client.purpose_calls,
                    },
                    fh,
                )
            os.replace(tmp_ckpt, ckpt_path)
            write_json(out_dir / "state.json", manager.snapshot())
    except BudgetExceeded:
        status = "budget_exhausted"
        failure_stage = "budget"
    except Exception as err:
        status = "error"
        error_type = type(err).__name__
        raise
    finally:
        environment.close()
        snap = manager.snapshot()
        write_json(out_dir / "state.json", snap)
        distill_calls = client.purpose_calls.get("distillation", 0) + client.purpose_calls.get("distillation_format_retry", 0)
        action_retry_calls = client.purpose_calls.get("action_format_retry", 0)
        distill_retry_calls = client.purpose_calls.get("distillation_format_retry", 0)
        critic_calls = getattr(manager, "manager_llm_calls", 0)
        summary = {
            "kind": "environment_rollout",
            "spec_fingerprint": expected_fp,
            "method_label": method_label,
            "env_name": env_name,
            "split": split,
            "seed": seed,
            "status": status,
            "error_type": error_type,
            "failure_stage": failure_stage,
            "completed_episodes": len(records),
            "requested_episodes": episodes,
            "step_limit_episodes": sum(r["termination"] == "step_limit" for r in records),
            "success_rate": sum(r["reward"] for r in records) / len(records) if records else None,
            "late_success_rate": (
                sum(r["reward"] for r in records[len(records) // 2 :]) / max(1, len(records) - len(records) // 2)
                if records
                else None
            ),
            "mean_score": sum(r["score"] for r in records) / len(records) if records else None,
            "retrieval_tokens": sum(r["retrieval_tokens"] for r in records),
            "mean_retrieval_tokens": sum(r["retrieval_tokens"] for r in records) / len(records) if records else 0.0,
            "manager_llm_calls": critic_calls,
            "distillation_llm_calls": distill_calls,
            "action_format_retry_calls": action_retry_calls,
            "distillation_format_retry_calls": distill_retry_calls,
            "total_memory_overhead_calls": critic_calls + distill_calls,
            "trigger_counts": snap.get("trigger_counts", {}),
            "api": client.summary(),
        }
        write_json(summary_path, summary)
        if status == "completed" and ckpt_path.is_file():
            ckpt_path.unlink(missing_ok=True)
    return summary


def default_api_dict(base_url: str = "https://api.deepseek.com", model: str = "deepseek-flash", key_env: str = "DEEPSEEK_API_KEY", max_calls: int = 12000, max_total_tokens: int = 35_000_000) -> dict:
    m_low = model.lower()
    is_deepseek = "deepseek" in base_url.lower() or "deepseek" in m_low
    if is_deepseek or "qwen3" in m_low or "glm-5" in m_low or uses_verified_magpie_mimo_reasoning_field(base_url, model):
        thinking = "disabled"
    else:
        thinking = None
    return {
        "base_url": base_url,
        "model": model,
        "key_env": key_env,
        "thinking": thinking,
        "max_calls": max_calls,
        "max_total_tokens": max_total_tokens,
        "max_output_tokens": 128 if "llama" in m_low else 512,
        "timeout_seconds": 90.0,
        "input_price_per_million": 0.30 if is_deepseek else 0.0,
        "output_price_per_million": 1.20 if is_deepseek else 0.0,
        "max_retries": 5,
        "max_format_retries": 2,
    }


def invoke_worker_subprocess(spec: dict) -> dict:
    python_bin = ALFWORLD_PYTHON if spec["env_name"] == "alfworld" else SCIENCEWORLD_PYTHON
    out_dir = Path(spec["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = out_dir / "worker_spec.json"
    summary_path = out_dir / "summary.json"
    ckpt_path = out_dir / "checkpoint.pkl"
    manifest_out_path = out_dir / "manifest.json"
    expected_fp = compute_spec_fingerprint(spec)

    if spec_path.is_file():
        existing_spec = json.loads(spec_path.read_text(encoding="utf-8"))
        old_fp = existing_spec.get("spec_fingerprint") or compute_spec_fingerprint(existing_spec)
        if old_fp != expected_fp and (summary_path.is_file() or ckpt_path.is_file() or manifest_out_path.is_file()):
            raise RuntimeError(f"Refusing to overwrite existing run/checkpoint in {out_dir} with mismatched spec_fingerprint")
    if summary_path.is_file():
        existing_sum = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing_sum.get("spec_fingerprint") != expected_fp:
            raise RuntimeError(f"Refusing to reuse summary in {out_dir} with missing or mismatched spec_fingerprint")
    if ckpt_path.is_file():
        with ckpt_path.open("rb") as fh:
            ckpt = pickle.load(fh)
        if ckpt.get("spec_fingerprint") != expected_fp:
            raise RuntimeError(f"Refusing to resume checkpoint in {out_dir} with missing or mismatched spec_fingerprint")

    spec_to_write = {**spec, "spec_fingerprint": expected_fp}
    write_json(spec_path, spec_to_write)
    disk_tmp = REPO_ROOT / "outputs" / ".tmp_worker"
    disk_tmp.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["TMPDIR"] = str(disk_tmp)
    env["TEMP"] = str(disk_tmp)
    env["TMP"] = str(disk_tmp)
    env["JAVA_TOOL_OPTIONS"] = f"-Xmx512m -XX:-UsePerfData -Djava.io.tmpdir={disk_tmp}"
    proc = None
    for sub_attempt in range(3):
        proc = subprocess.run(
            [python_bin, "-m", "comep.campaign", "worker", "--spec", str(spec_path)],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            break
        if sub_attempt < 2:
            time.sleep(3.0 * (sub_attempt + 1))
    if proc.returncode != 0:
        raise RuntimeError(
            f"Worker failed ({spec['env_name']} / {spec['method']} / seed={spec['seed']}):\nSTDOUT:\n{proc.stdout[-1500:]}\nSTDERR:\n{proc.stderr[-1500:]}"
        )
    return json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))


def build_warmstart_pool(
    env_name: str,
    train_manifest: str | Path,
    output_dir: str | Path,
    pool_out_path: str | Path,
    api_dict: dict,
    episodes: int = 24,
    max_steps: int = 30,
    target_pool_size: int = 20,
) -> dict:
    """Run no-memory on official train split ONLY and extract modular + naive candidate memory pool."""
    out_dir = Path(output_dir) / f"{env_name}_train_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_out_path = Path(pool_out_path)

    # Verify manifest belongs strictly to train split
    raw_manifest = json.loads(Path(train_manifest).read_text(encoding="utf-8"))
    if env_name == "scienceworld":
        if any(entry.get("split") != "train" for entry in raw_manifest):
            raise ValueError("warm-start pool extraction strictly requires train split only")
    elif env_name == "alfworld":
        if any("/train/" not in (entry["game"] if isinstance(entry, dict) else entry) for entry in raw_manifest):
            raise ValueError("ALFWorld warm-start pool extraction strictly requires train split only")

    spec = {
        "env_name": env_name,
        "split": "train",
        "manifest_path": str(train_manifest),
        "seed": 0,
        "episodes": episodes,
        "max_steps": max_steps,
        "method": "no-memory",
        "initial_rules": [],
        "distill": False,
        "admissible_limit": 0,
        "action_instructions": ALFWORLD_ACTION_INSTRUCTIONS if env_name == "alfworld" else SCIENCEWORLD_ACTION_INSTRUCTIONS,
        "output_dir": str(out_dir / "no_memory_rollout"),
        "api": api_dict,
    }
    rollout_summary = invoke_worker_subprocess(spec)

    # Read episodes and transitions from train rollout to distill modular sub-skills & naive reflections
    ep_lines = (out_dir / "no_memory_rollout" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ep_records = [json.loads(line) for line in ep_lines if line.strip()]
    tr_by_ep: dict[int, list[dict]] = {}
    if (out_dir / "no_memory_rollout" / "transitions.jsonl").is_file():
        for line in (out_dir / "no_memory_rollout" / "transitions.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                tr = json.loads(line)
                tr_by_ep.setdefault(int(tr["episode"]), []).append(tr)

    extractor_client = ChatClient(APIConfig(**api_dict), out_dir / "pool_extraction_api.jsonl")
    encoder = LocalEncoder(BGE_MODEL_PATH)
    counter = TokenCounter("cl100k_base")

    candidates: list[dict] = []
    for rec in ep_records:
        ep = int(rec["episode"])
        trs = tr_by_ep.get(ep, [])
        if not trs:
            continue
        compact_trs = [
            {"step": t["step"], "action": t["action"], "obs": t["next_observation"][:220], "score": t["score"]}
            for t in trs[:18]
        ]
        task_header = trs[0]["observation"][:320]
        if rec["reward"] == 1 or rec["score"] >= 50:
            # Extract two complementary modular sub-skills (locator/precondition + operator/command)
            prompt = (
                f"Train episode (context={rec['context']}, success={rec['reward']}, score={rec['score']}):\n"
                f"Initial observation: {task_header}\n"
                f"Trajectory: {json.dumps(compact_trs, ensure_ascii=False)}\n"
                "Decompose what worked into TWO separate, concise, complementary rules (each 12 to 26 tokens): "
                "rule_a = where to search or precondition setup for this task family; "
                "rule_b = exact command sequence or device/instrument operation for this task family. "
                "Return JSON: {\"rule_a\": \"...\", \"rule_b\": \"...\"}."
            )
            reply = extractor_client.complete(
                [{"role": "system", "content": "Extract two concise modular sub-skill rules; return JSON only."}, {"role": "user", "content": prompt}],
                "train_pool_modular",
            )
            try:
                parsed = _parse_json_object(reply)
                for slot in ("rule_a", "rule_b"):
                    txt = parsed.get(slot)
                    if isinstance(txt, str) and 6 <= counter.count(txt.strip()) <= 30:
                        candidates.append(
                            {
                                "text": txt.strip(),
                                "source_episode": ep,
                                "source_task_id": rec["task_id"],
                                "context": rec["context"],
                                "source_outcome": rec["reward"],
                                "extraction_type": f"modular_{slot}",
                            }
                        )
            except Exception:
                pass
        else:
            # Extract uncurated naive agent hypothesis from failed/partial train attempt (realistic memory noise/shortcuts)
            prompt = (
                f"Failed train episode (context={rec['context']}, termination={rec['termination']}, score={rec['score']}):\n"
                f"Initial observation: {task_header}\n"
                f"Trajectory: {json.dumps(compact_trs, ensure_ascii=False)}\n"
                "Write ONE concise heuristic rule (12 to 26 tokens) that a naive continual-learning agent might record from this attempt "
                "(e.g. an assertive workflow shortcut or unverified assumption about locations/commands for this task type). "
                "Return JSON: {\"rule\": \"...\"}."
            )
            reply = extractor_client.complete(
                [{"role": "system", "content": "Write one concise naive continual-agent memory rule; return JSON only."}, {"role": "user", "content": prompt}],
                "train_pool_naive",
            )
            try:
                parsed = _parse_json_object(reply)
                txt = parsed.get("rule")
                if isinstance(txt, str) and 6 <= counter.count(txt.strip()) <= 30:
                    candidates.append(
                        {
                            "text": txt.strip(),
                            "source_episode": ep,
                            "source_task_id": rec["task_id"],
                            "context": rec["context"],
                            "source_outcome": rec["reward"],
                            "extraction_type": "naive_failed_reflection",
                        }
                    )
            except Exception:
                pass

    # Deduplicate near-identical candidates (cosine >= 0.88) while balancing modular (70%) and naive/noisy (30%) rules
    modular = [c for c in candidates if c["extraction_type"].startswith("modular_")]
    naive = [c for c in candidates if c["extraction_type"] == "naive_failed_reflection"]

    selected_entries: list[dict] = []
    selected_vecs: list[np.ndarray] = []

    def try_admit(pool_slice: list[dict], limit: int):
        for cand in pool_slice:
            if len(selected_entries) >= limit:
                break
            vec = encoder.encode(cand["text"])
            if any(similarity(vec, v) >= 0.88 for v in selected_vecs):
                continue
            selected_entries.append(cand)
            selected_vecs.append(vec)

    modular_quota = int(round(target_pool_size * 0.70))
    try_admit(modular, modular_quota)
    try_admit(naive, target_pool_size)
    if len(selected_entries) < target_pool_size:
        try_admit(modular, target_pool_size)

    final_pool = []
    provenance = []
    for idx, entry in enumerate(selected_entries):
        rid = f"{env_name[:3]}-train-{idx:02d}"
        final_pool.append({"id": rid, "text": entry["text"]})
        provenance.append({"id": rid, "tokens": counter.count(entry["text"]), **entry})

    write_json(pool_out_path, final_pool)
    prov_summary = {
        "env_name": env_name,
        "split": "train",
        "split_leakage_check": "PASSED (100% train split)",
        "train_no_memory_episodes": episodes,
        "train_no_memory_success_rate": rollout_summary["success_rate"],
        "pool_size": len(final_pool),
        "modular_count": sum(p["extraction_type"].startswith("modular_") for p in provenance),
        "naive_count": sum(p["extraction_type"] == "naive_failed_reflection" for p in provenance),
        "extraction_api": extractor_client.summary(),
        "rollout_api": rollout_summary["api"],
        "rules": provenance,
    }
    write_json(out_dir / "pool_provenance.json", prov_summary)
    return prov_summary


def paired_statistics(comep_rewards: list[int], base_rewards: list[int], seed: int = 42) -> dict:
    n = min(len(comep_rewards), len(base_rewards))
    if n == 0:
        return {"n": 0, "diff": 0.0, "ci95": [0.0, 0.0], "mcnemar_p": 1.0}
    c = np.array(comep_rewards[:n], dtype=float)
    b = np.array(base_rewards[:n], dtype=float)
    diff = float(np.mean(c - b))

    # Paired bootstrap 95% CI
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(5000, n))
    boot_diffs = np.mean(c[idx] - b[idx], axis=1)
    ci_low, ci_high = float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))

    # Exact two-sided McNemar test on discordant pairs
    b10 = int(np.sum((c == 1) & (b == 0)))
    b01 = int(np.sum((c == 0) & (b == 1)))
    disc = b10 + b01
    if disc == 0:
        p_val = 1.0
    else:
        from scipy.stats import binomtest

        p_val = float(binomtest(b10, disc, 0.5, alternative="two-sided").pvalue)
    return {
        "n_paired": n,
        "diff": round(diff, 4),
        "ci95": [round(ci_low, 4), round(ci_high, 4)],
        "comep_wins_discordant": b10,
        "baseline_wins_discordant": b01,
        "mcnemar_p": round(p_val, 6),
    }


def run_grid(
    stage_name: str,
    env_name: str,
    split: str,
    manifest_path: str | Path,
    initial_rules_path: str | Path,
    methods: list[str],
    seeds: list[int],
    episodes: int,
    max_steps: int,
    output_dir: str | Path,
    workers: int,
    api_dict: dict,
    distill: bool = True,
    manager_overrides: dict | None = None,
    shared_cache_dir: str | Path | None = None,
    protocol_version: str = "v2.2",
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    initial_rules = json.loads(Path(initial_rules_path).read_text(encoding="utf-8")) if initial_rules_path else []

    specs = []
    for seed in seeds:
        for method in methods:
            run_dir = out_dir / f"{method}_seed{seed}"
            run_api = dict(api_dict)
            run_api["protocol_version"] = protocol_version
            if shared_cache_dir is not None:
                run_api["cache_dir"] = str(Path(shared_cache_dir) / env_name / f"seed_{seed}")
                run_api["cache_namespace"] = f"{env_name}:seed_{seed}"
            specs.append(
                {
                    "env_name": env_name,
                    "split": split,
                    "manifest_path": str(manifest_path),
                    "seed": int(seed),
                    "episodes": int(episodes),
                    "max_steps": int(max_steps),
                    "method": method,
                    "initial_rules": initial_rules,
                    "distill": distill,
                    "admissible_limit": 0,
                    "action_instructions": ALFWORLD_ACTION_INSTRUCTIONS if env_name == "alfworld" else SCIENCEWORLD_ACTION_INSTRUCTIONS,
                    "manager": manager_overrides or {},
                    "output_dir": str(run_dir),
                    "api": run_api,
                }
            )

    summaries = []
    if shared_cache_dir is not None and "comep" in methods:
        primary_specs = [s for s in specs if s["method"] == "comep"]
        secondary_specs = [s for s in specs if s["method"] != "comep"]
        batches = [primary_specs, secondary_specs]
    else:
        batches = [specs]

    for batch in batches:
        if not batch:
            continue
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            fut_map = {pool.submit(invoke_worker_subprocess, s): s for s in batch}
            for fut in as_completed(fut_map):
                s = fut_map[fut]
                res = fut.result()
                summaries.append(res)
                hits = res["api"].get("cache_hits", 0)
                live = res["api"].get("http_requests", res["api"].get("attempted_calls", 0))
                print(
                    f"[{stage_name}][{env_name}] DONE {s['method']:26s} seed={s['seed']} "
                    f"SR={res['success_rate']*100:5.1f}% LateSR={res['late_success_rate']*100:5.1f}% "
                    f"RetTok={res['mean_retrieval_tokens']:4.1f} Hits={hits:4d} Live={live:4d} "
                    f"Cost=${res['api']['estimated_cost_usd'] or 0:.3f}",
                    flush=True,
                )

    # Load episode-level rewards per (method, seed) and assert strict (episode, task_id) paired alignment
    ep_rewards: dict[tuple[str, int], list[int]] = {}
    seed_task_sequences: dict[int, list[tuple[int, str]]] = {}
    for method in methods:
        for seed in seeds:
            ep_file = out_dir / f"{method}_seed{seed}" / "episodes.jsonl"
            lines = ep_file.read_text(encoding="utf-8").splitlines()
            recs = [json.loads(line) for line in lines if line.strip()]
            recs.sort(key=lambda r: r["episode"])
            if len(recs) != int(episodes):
                raise RuntimeError(f"Episode count mismatch for {method} seed={seed}: got {len(recs)}, expected {episodes}")
            ep_indices = [int(r["episode"]) for r in recs]
            if ep_indices != list(range(int(episodes))):
                raise RuntimeError(f"Non-contiguous or duplicate episodes for {method} seed={seed}")
            task_seq = [(int(r["episode"]), str(r["task_id"])) for r in recs]
            if seed not in seed_task_sequences:
                seed_task_sequences[seed] = task_seq
            elif seed_task_sequences[seed] != task_seq:
                raise RuntimeError(f"Paired (episode, task_id) sequence mismatch for {method} on seed={seed}")
            ep_rewards[(method, seed)] = [int(r["reward"]) for r in recs]

    agg_methods = {}
    for method in methods:
        m_runs = [r for r in summaries if r["method_label"] == method]
        m_runs.sort(key=lambda r: r["seed"])
        srs = [float(r["success_rate"]) for r in m_runs]
        late_srs = [float(r["late_success_rate"]) for r in m_runs]
        ret_toks = [float(r["mean_retrieval_tokens"]) for r in m_runs]
        prompt_toks = sum(int(r["api"]["prompt_tokens"]) for r in m_runs)
        comp_toks = sum(int(r["api"]["completion_tokens"]) for r in m_runs)
        prov_prompt_toks = sum(int(r["api"].get("provider_reported_prompt_tokens", r["api"]["prompt_tokens"])) for r in m_runs)
        prov_comp_toks = sum(int(r["api"].get("provider_reported_completion_tokens", r["api"]["completion_tokens"])) for r in m_runs)
        http_reqs = sum(int(r["api"].get("http_requests", r["api"]["attempted_calls"])) for r in m_runs)
        cache_hits = sum(int(r["api"].get("cache_hits", 0)) for r in m_runs)
        logical_complete = all(bool(r["api"].get("usage_complete", True)) for r in m_runs)
        http_complete = all(bool(r["api"].get("http_usage_complete", True)) for r in m_runs)

        logical_costs = [r["api"].get("estimated_cost_usd") for r in m_runs]
        cost_usd = round(sum(float(c) for c in logical_costs), 4) if (logical_complete and all(c is not None for c in logical_costs)) else None

        prov_costs = []
        for r in m_runs:
            pc = r["api"].get("provider_reported_cost_usd")
            if pc is None and "provider_reported_cost_usd" not in r["api"]:
                pc = r["api"].get("estimated_cost_usd")
            prov_costs.append(pc)
        prov_cost_usd = round(sum(float(c) for c in prov_costs), 4) if (http_complete and all(c is not None for c in prov_costs)) else None

        mgr_calls = sum(int(r["manager_llm_calls"]) for r in m_runs)
        dist_calls = sum(int(r["distillation_llm_calls"]) for r in m_runs)
        act_retry_calls = sum(int(r.get("action_format_retry_calls", 0)) for r in m_runs)
        dist_retry_calls = sum(int(r.get("distillation_format_retry_calls", 0)) for r in m_runs)
        pooled = [y for s in seeds for y in ep_rewards[(method, s)]]

        agg_triggers: dict[str, int] = {}
        for r in m_runs:
            for tk, tv in (r.get("trigger_counts") or {}).items():
                agg_triggers[tk] = agg_triggers.get(tk, 0) + int(tv)

        rng = np.random.default_rng(100)
        p_arr = np.array(pooled, dtype=float)
        idx = rng.integers(0, len(p_arr), size=(5000, len(p_arr)))
        ci95 = [
            round(float(np.percentile(np.mean(p_arr[idx], axis=1), 2.5)), 4),
            round(float(np.percentile(np.mean(p_arr[idx], axis=1), 97.5)), 4),
        ]

        agg_methods[method] = {
            "seeds": len(m_runs),
            "total_episodes": len(pooled),
            "per_seed_sr": [round(x, 4) for x in srs],
            "per_seed_late_sr": [round(x, 4) for x in late_srs],
            "mean_sr": round(float(np.mean(srs)), 4),
            "std_sr": round(float(np.std(srs, ddof=1)) if len(srs) > 1 else 0.0, 4),
            "ci95_sr": ci95,
            "mean_late_sr": round(float(np.mean(late_srs)), 4),
            "mean_retrieval_tokens": round(float(np.mean(ret_toks)), 2),
            "total_prompt_tokens": prompt_toks,
            "total_completion_tokens": comp_toks,
            "total_provider_tokens": prompt_toks + comp_toks,
            "provider_reported_prompt_tokens": prov_prompt_toks,
            "provider_reported_completion_tokens": prov_comp_toks,
            "http_requests": http_reqs,
            "cache_hits": cache_hits,
            "usage_complete": logical_complete,
            "http_usage_complete": http_complete,
            "manager_llm_calls": mgr_calls,
            "distillation_llm_calls": dist_calls,
            "action_format_retry_calls": act_retry_calls,
            "distillation_format_retry_calls": dist_retry_calls,
            "trigger_counts": agg_triggers,
            "total_cost_usd": cost_usd,
            "provider_reported_cost_usd": prov_cost_usd,
            "per_episode_cost_usd": round(cost_usd / max(1, len(pooled)), 6) if cost_usd is not None else None,
        }

    comparisons = {}
    if "comep" in methods:
        comep_pooled = [y for s in seeds for y in ep_rewards[("comep", s)]]
        for method in methods:
            if method == "comep":
                continue
            base_pooled = [y for s in seeds for y in ep_rewards[(method, s)]]
            comparisons[f"comep_vs_{method}"] = paired_statistics(comep_pooled, base_pooled)

    all_logical_costs = [v["total_cost_usd"] for v in agg_methods.values()]
    all_prov_costs = [v["provider_reported_cost_usd"] for v in agg_methods.values()]
    total_cost_usd = round(sum(c for c in all_logical_costs), 4) if all(c is not None for c in all_logical_costs) else None
    total_prov_cost_usd = round(sum(c for c in all_prov_costs), 4) if all(c is not None for c in all_prov_costs) else None
    grid_report = {
        "kind": "benchmark_grid_report",
        "protocol_version": protocol_version,
        "shared_request_cache": shared_cache_dir is not None,
        "stage": stage_name,
        "env_name": env_name,
        "split": split,
        "episodes_per_seed": episodes,
        "seeds": seeds,
        "methods": agg_methods,
        "paired_vs_comep": comparisons,
        "total_grid_cost_usd": total_cost_usd,
        "total_provider_reported_cost_usd": total_prov_cost_usd,
    }
    write_json(out_dir / "grid_summary.json", grid_report)
    return grid_report


def run_counterfactual_4cell(
    env_name: str,
    split: str,
    manifest_path: str | Path,
    pair_entry: dict,
    episodes: int,
    max_steps: int,
    output_dir: str | Path,
    workers: int,
    api_dict: dict,
) -> dict:
    """Evaluate {none, A, B, AB} counterfactual cells on identical tasks to verify causal pair interaction."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rule_a = pair_entry["rule_a"]
    rule_b = pair_entry["rule_b"]
    cells = {
        "none": [],
        "A": [{"id": "A", "text": rule_a}],
        "B": [{"id": "B", "text": rule_b}],
        "AB": [{"id": "A", "text": rule_a}, {"id": "B", "text": rule_b}],
    }
    specs = []
    for cell_name, rules in cells.items():
        specs.append(
            {
                "env_name": env_name,
                "split": split,
                "manifest_path": str(manifest_path),
                "seed": 0,
                "episodes": int(episodes),
                "max_steps": int(max_steps),
                "method": "no-memory" if cell_name == "none" else "full-memory",
                "initial_rules": rules,
                "distill": False,
                "admissible_limit": 0,
                "action_instructions": ALFWORLD_ACTION_INSTRUCTIONS if env_name == "alfworld" else SCIENCEWORLD_ACTION_INSTRUCTIONS,
                "manager": {"retrieval_tokens": 120, "max_items": 2, "redundancy": 0.0},
                "output_dir": str(out_dir / f"cell_{cell_name}"),
                "api": api_dict,
            }
        )

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        list(pool.map(invoke_worker_subprocess, specs))

    cell_rewards: dict[str, list[int]] = {}
    cell_scores: dict[str, list[float]] = {}
    cell_costs = 0.0
    for cell_name in ("none", "A", "B", "AB"):
        c_dir = out_dir / f"cell_{cell_name}"
        summ = json.loads((c_dir / "summary.json").read_text(encoding="utf-8"))
        cell_costs += float(summ["api"]["estimated_cost_usd"] or 0.0)
        lines = (c_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        recs = sorted((json.loads(l) for l in lines if l.strip()), key=lambda r: r["episode"])
        cell_rewards[cell_name] = [int(r["reward"]) for r in recs]
        cell_scores[cell_name] = [float(r.get("score", r["reward"])) for r in recs]

    n = min(len(v) for v in cell_rewards.values())
    y0 = np.array(cell_rewards["none"][:n], dtype=float)
    ya = np.array(cell_rewards["A"][:n], dtype=float)
    yb = np.array(cell_rewards["B"][:n], dtype=float)
    yab = np.array(cell_rewards["AB"][:n], dtype=float)

    s0 = np.array(cell_scores["none"][:n], dtype=float)
    sa = np.array(cell_scores["A"][:n], dtype=float)
    sb = np.array(cell_scores["B"][:n], dtype=float)
    sab = np.array(cell_scores["AB"][:n], dtype=float)

    contrasts = yab - ya - yb + y0
    delta_hat = float(np.mean(contrasts))
    rng = np.random.default_rng(77)
    idx = rng.integers(0, n, size=(5000, n))
    boot = np.mean(contrasts[idx], axis=1)
    ci95 = [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)]

    sc_contrasts = sab - sa - sb + s0
    sc_delta_hat = float(np.mean(sc_contrasts))
    sc_boot = np.mean(sc_contrasts[idx], axis=1)
    sc_ci95 = [round(float(np.percentile(sc_boot, 2.5)), 2), round(float(np.percentile(sc_boot, 97.5)), 2)]

    report = {
        "kind": "causal_4cell_verification",
        "env_name": env_name,
        "split": split,
        "pair_kind": pair_entry.get("kind", "discovered"),
        "id_a": pair_entry.get("id_a"),
        "id_b": pair_entry.get("id_b"),
        "learned_s_ij": pair_entry.get("learned_s_ij"),
        "count": pair_entry.get("count"),
        "co_wins": pair_entry.get("co_wins"),
        "selection_source": pair_entry.get("selection_source", "posterior_s_ij"),
        "target_context": pair_entry.get("target_context", ""),
        "rule_a": rule_a,
        "rule_b": rule_b,
        "episodes_per_cell": n,
        "cell_rates": {
            "none": round(float(np.mean(y0)), 4),
            "A": round(float(np.mean(ya)), 4),
            "B": round(float(np.mean(yb)), 4),
            "AB": round(float(np.mean(yab)), 4),
        },
        "causal_interaction_delta_AB": round(delta_hat, 4),
        "ci95_delta_AB": ci95,
        "cell_mean_scores": {
            "none": round(float(np.mean(s0)), 2),
            "A": round(float(np.mean(sa)), 2),
            "B": round(float(np.mean(sb)), 2),
            "AB": round(float(np.mean(sab)), 2),
        },
        "score_interaction_delta_AB": round(sc_delta_hat, 2),
        "ci95_score_delta_AB": sc_ci95,
        "total_cost_usd": round(cell_costs, 4),
    }
    write_json(out_dir / "counterfactual_summary.json", report)
    return report


def main(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(prog="comep.campaign")
    sub = parser.add_subparsers(dest="cmd", required=True)
    w_parser = sub.add_parser("worker")
    w_parser.add_argument("--spec", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "worker":
        execute_worker(args.spec)


if __name__ == "__main__":
    main()
