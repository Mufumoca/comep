"""Zero-API-cost offline replay & counterfactual routing validation on historical real DeepSeek trajectories."""
from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np

from comep.core import Config, MemoryManager, Selection
from comep.experiments import write_json
from comep.text import LocalEncoder, TokenCounter, similarity


HIST_ROOT = Path("/root/.paseo/worktrees/1s5ku8sn/lucid-bumblebee/outputs")
OUT_DIR = Path("/root/.paseo/worktrees/1s5ku8sn/comep-v2/outputs/stage2-offline-real-replay")
BGE_PATH = "/root/.paseo/worktrees/1s5ku8sn/comep-v2/models/bge-small-en-v1.5"
NLI_PATH = "/root/.paseo/worktrees/1s5ku8sn/comep-v2/models/deberta-v3-base-mnli-fever-anli"


def brier_and_logloss(preds: list[tuple[float, int]]) -> dict[str, float]:
    if not preds:
        return {"n": 0, "brier": float("nan"), "logloss": float("nan")}
    brier = sum((p - y) ** 2 for p, y in preds) / len(preds)
    ll = -sum(
        y * math.log(min(max(p, 1e-6), 1.0 - 1e-6)) + (1 - y) * math.log(1.0 - min(max(p, 1e-6), 1.0 - 1e-6))
        for p, y in preds
    ) / len(preds)
    return {"n": len(preds), "brier": round(brier, 5), "logloss": round(ll, 5)}


def evaluate_procedural_pair_counterfactual(encoder: LocalEncoder, counter: TokenCounter) -> dict:
    summary_path = HIST_ROOT / "procedural-pair-live-20261103" / "summary.json"
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [r for r in data["rows"] if r.get("status") == "completed"]

    rule_a = None
    rule_b = None
    cell_outcomes: dict[str, list[int]] = {"none": [], "A": [], "B": [], "AB": []}
    for r in rows:
        cell = r["cell"]
        reward = int(r["episode"]["reward"])
        if cell == "A" and r["rules"]:
            rule_a = r["rules"][0]
            cell_outcomes["A"].append(reward)
        elif cell == "B" and r["rules"]:
            rule_b = r["rules"][0]
            cell_outcomes["B"].append(reward)
        elif cell in ("AB", "BA") and len(r["rules"]) == 2:
            cell_outcomes["AB"].append(reward)
        elif cell == "none":
            cell_outcomes["none"].append(reward)

    means = {k: round(float(np.mean(v)), 4) for k, v in cell_outcomes.items()}
    interaction_contrast = round(means["AB"] - means["A"] - means["B"] + means["none"], 4)

    # Load real ScienceWorld general distractors so the bank has realistic clutter
    sw_rules_path = Path("/root/.paseo/worktrees/1s5ku8sn/comep-v2/configs/scienceworld-initial-rules.json")
    distractors = json.loads(sw_rules_path.read_text(encoding="utf-8"))

    query = "Task: Your task is to turn on the green light bulb by powering it using a renewable or non-renewable power source, wires, and a switch."
    context = "power-component"

    # Prequential log-loss / Brier over actual chronological 4-cell observations
    preq_results = {}
    for variant, cfg, method, enc in [
        ("comep-v1-legacy", Config(batch_laplace=False, retrieval_tokens=95, max_items=2), "comep", LocalEncoder()),
        ("additive-v2", Config(batch_laplace=True, retrieval_tokens=95, max_items=2, matched_pair_lookahead=True), "additive", encoder),
        ("comep-v2", Config(batch_laplace=True, retrieval_tokens=95, max_items=2), "comep", encoder),
    ]:
        mgr = MemoryManager(method=method, config=cfg, seed=0, encoder=enc, counter=counter)
        mgr.add("A", rule_a, prune=False)
        mgr.add("B", rule_b, prune=False)
        q_vec = enc.encode(query)
        rel = {mid: similarity(q_vec, mgr.vectors[mid]) for mid in mgr.memories}
        preds = []
        for r in rows:
            cell = r["cell"]
            if cell not in ("none", "A", "B", "AB", "BA"):
                continue
            selected = [] if cell == "none" else (["A"] if cell == "A" else (["B"] if cell == "B" else ["A", "B"]))
            tokens = sum(mgr.memories[m].tokens for m in selected)
            sel = Selection(selected, rel, context, tokens, False)
            pred = mgr.observe(sel, int(r["episode"]["reward"]))
            preds.append((float(pred), int(r["episode"]["reward"])))
        preq_results[variant] = {
            **brier_and_logloss(preds),
            "learned_pair_mean_AB": round(mgr.pair_mean("A", "B"), 4),
            "learned_ind_A": round(mgr.individual["A"].mean, 4),
            "learned_ind_B": round(mgr.individual["B"].mean, 4),
        }

    # Counterfactual online bandit benchmark grounded in empirical real DeepSeek 4-cell table
    methods = ["no-memory", "full-memory", "individual", "mmr", "additive", "evomemguard", "comep", "comep-v1-legacy"]
    cf_summary = {}
    episodes = 60
    seeds = 10
    for m_label in methods:
        seed_srs = []
        seed_late_srs = []
        seed_toks = []
        seed_ab_rates = []
        for s in range(seeds):
            rng = np.random.default_rng(1000 + s)
            if m_label == "comep-v1-legacy":
                cfg = Config(batch_laplace=False, retrieval_tokens=90, bank_tokens=400, max_items=2, max_bank_items=12)
                mgr = MemoryManager("comep", cfg, seed=s, encoder=LocalEncoder(), counter=TokenCounter())
            else:
                cfg = Config(
                    batch_laplace=True,
                    retrieval_tokens=90,
                    bank_tokens=400,
                    max_items=2,
                    max_bank_items=12,
                    matched_pair_lookahead=True,
                    nli_model=NLI_PATH,
                )
                mgr = MemoryManager(m_label, cfg, seed=s, encoder=encoder, counter=counter)
            mgr.add("A", rule_a, prune=False)
            mgr.add("B", rule_b, prune=False)
            for d in distractors[:6]:
                mgr.add(d["id"], d["text"], prune=False)
            mgr.prune()

            rewards = []
            toks = []
            ab_hits = 0
            for ep in range(episodes):
                sel = mgr.choose(query, context)
                has_a = "A" in sel.identifiers
                has_b = "B" in sel.identifiers
                if has_a and has_b:
                    pool = cell_outcomes["AB"]
                    ab_hits += 1
                elif has_a:
                    pool = cell_outcomes["A"]
                elif has_b:
                    pool = cell_outcomes["B"]
                else:
                    pool = cell_outcomes["none"]
                reward = int(rng.choice(pool))
                mgr.observe(sel, reward)
                rewards.append(reward)
                toks.append(sel.tokens)
            seed_srs.append(float(np.mean(rewards)))
            seed_late_srs.append(float(np.mean(rewards[episodes // 2 :])))
            seed_toks.append(float(np.mean(toks)))
            seed_ab_rates.append(ab_hits / episodes)
        cf_summary[m_label] = {
            "mean_sr": round(float(np.mean(seed_srs)), 4),
            "std_sr": round(float(np.std(seed_srs, ddof=1)), 4),
            "late_sr": round(float(np.mean(seed_late_srs)), 4),
            "mean_tokens": round(float(np.mean(seed_toks)), 2),
            "pair_AB_selection_rate": round(float(np.mean(seed_ab_rates)), 4),
        }

    return {
        "source": str(summary_path),
        "task_family": "ScienceWorld power-component (real deepseek-flash rollouts)",
        "empirical_4cell_rates": means,
        "empirical_cell_counts": {k: len(v) for k, v in cell_outcomes.items()},
        "empirical_interaction_contrast_delta_AB": interaction_contrast,
        "prequential_fit": preq_results,
        "counterfactual_online_routing_10seeds": cf_summary,
    }


def evaluate_alfworld_online_replay(encoder: LocalEncoder, counter: TokenCounter) -> dict:
    base = HIST_ROOT / "alfworld-online-live-20261104"
    summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
    # Replay the actual comep stream from trial-000..trial-029
    # Collect distilled rules in chronological order from comep sub-runs
    comep_rows = [r for r in summary["rows"] if r["method"] == "comep" and r["status"] == "completed"]
    comep_rows.sort(key=lambda r: r["trial"])

    # Extract initial task observation and distilled rule from each trial directory
    stream = []
    for row in comep_rows:
        t_idx = row["trial"]
        trial_dir = base / f"trial-{t_idx:03d}" / "comep"
        obs_text = row["task_type"]
        if (trial_dir / "transitions.jsonl").is_file():
            first_tr = json.loads((trial_dir / "transitions.jsonl").read_text(encoding="utf-8").splitlines()[0])
            obs_text = first_tr["observation"]
        distilled_rule = None
        if (trial_dir / "distillation.jsonl").is_file():
            lines = (trial_dir / "distillation.jsonl").read_text(encoding="utf-8").splitlines()
            if lines:
                d_rec = json.loads(lines[-1])
                if d_rec.get("accepted") and d_rec.get("rule"):
                    distilled_rule = d_rec["rule"]
        stream.append(
            {
                "trial": t_idx,
                "task_type": row["task_type"],
                "obs_text": obs_text,
                "selected": row["episode"]["selected"],
                "reward": int(row["episode"]["reward"]),
                "logged_pred": float(row["episode"]["predicted_probability"]),
                "distilled_rule": distilled_rule,
            }
        )

    # Compare logged v1 predictions vs v2 Laplace + BGE + family context prequential predictions
    logged_preds = [(item["logged_pred"], item["reward"]) for item in stream]
    mgr_v2 = MemoryManager("comep", Config(batch_laplace=True), seed=0, encoder=encoder, counter=counter)
    v2_preds = []
    from comep.environments import _infer_alfworld_context

    for item in stream:
        ctx = _infer_alfworld_context(item["obs_text"])
        q_vec = encoder.encode(item["obs_text"])
        rel = {mid: similarity(q_vec, mgr_v2.vectors[mid]) for mid in mgr_v2.memories}
        valid_sel = [mid for mid in item["selected"] if mid in mgr_v2.memories]
        toks = sum(mgr_v2.memories[mid].tokens for mid in valid_sel)
        sel = Selection(valid_sel, rel, ctx, toks, False)
        pred = mgr_v2.observe(sel, item["reward"])
        v2_preds.append((float(pred), item["reward"]))
        if item["distilled_rule"]:
            mgr_v2.add(f"distilled-{item['trial']}", item["distilled_rule"], step=item["trial"] + 1, prune=False)

    # Also measure within-family vs cross-family similarity separation: lexical vs BGE-small
    lex_enc = LocalEncoder()
    rules_by_family = [(item["task_type"], item["distilled_rule"]) for item in stream if item["distilled_rule"]]
    within_lex, cross_lex, within_bge, cross_bge = [], [], [], []
    for i in range(len(rules_by_family)):
        for j in range(i + 1, len(rules_by_family)):
            fi, ti = rules_by_family[i]
            fj, tj = rules_by_family[j]
            s_lex = similarity(lex_enc.encode(ti), lex_enc.encode(tj))
            s_bge = similarity(encoder.encode(ti), encoder.encode(tj))
            if fi == fj:
                within_lex.append(s_lex)
                within_bge.append(s_bge)
            else:
                cross_lex.append(s_lex)
                cross_bge.append(s_bge)

    return {
        "source": str(base),
        "episodes_replayed": len(stream),
        "distilled_rules_count": len(rules_by_family),
        "logged_comep_v1_prequential": brier_and_logloss(logged_preds),
        "upgraded_comep_v2_prequential": brier_and_logloss(v2_preds),
        "embedding_family_discrimination": {
            "lexical_within_mean": round(float(np.mean(within_lex)), 4),
            "lexical_cross_mean": round(float(np.mean(cross_lex)), 4),
            "lexical_gap": round(float(np.mean(within_lex) - np.mean(cross_lex)), 4),
            "bge_within_mean": round(float(np.mean(within_bge)), 4),
            "bge_cross_mean": round(float(np.mean(cross_bge)), 4),
            "bge_gap": round(float(np.mean(within_bge) - np.mean(cross_bge)), 4),
        },
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    encoder = LocalEncoder(BGE_PATH)
    counter = TokenCounter("cl100k_base")
    proc_report = evaluate_procedural_pair_counterfactual(encoder, counter)
    alf_report = evaluate_alfworld_online_replay(encoder, counter)
    report = {
        "kind": "stage2_zero_cost_real_deepseek_replay",
        "procedural_scienceworld_4cell": proc_report,
        "alfworld_online_stream": alf_report,
    }
    write_json(OUT_DIR / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
