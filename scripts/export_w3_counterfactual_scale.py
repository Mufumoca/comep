"""Compute and persist W3 model-predicted 4-cell scale diagnostics (fixed seed=0 full-bank gates)."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from comep.campaign import BGE_MODEL_PATH, extract_routing_query
from comep.core import _split_scope_action, sigmoid
from comep.experiments import write_json
from comep.text import LocalEncoder, similarity


def compute_model_4cell_scale_audit(model_key: str, model_label: str, stage3_dir: Path, stage4_dir: Path, encoder: LocalEncoder) -> list[dict]:
    s4_sum_path = stage4_dir / "stage4_4cell_summary.json"
    s4_sum = json.loads(s4_sum_path.read_text(encoding="utf-8"))
    results = []

    for env_name in ("alfworld", "scienceworld"):
        state_path = stage3_dir / env_name / "comep_seed0" / "state.json"
        state_bytes = state_path.read_bytes()
        state_sha = hashlib.sha256(state_bytes).hexdigest()
        state = json.loads(state_bytes.decode("utf-8"))

        active_memories = {m["identifier"]: m["text"] for m in state.get("memories", [])}
        ind_means = {m["identifier"]: float(m["parameter"]["mean"]) for m in state.get("memories", [])}
        ctx_means = {k: float(v["mean"]) for k, v in state.get("contexts", {}).items()}
        synergy_gamma = float(state.get("config", {}).get("synergy", 1.0))
        state_pairs = {}
        for p_item in state.get("pairs", []):
            ids = p_item.get("identifiers", p_item.get("pair", []))
            if len(ids) == 2:
                state_pairs[tuple(sorted((str(ids[0]), str(ids[1]))))] = float(p_item["mean"])

        # Pre-encode final active bank strictly as frozen at end of comep_seed0
        bank_vecs = {i: encoder.encode(t) for i, t in active_memories.items()}
        bank_scopes = {i: encoder.encode(_split_scope_action(t)[0]) for i, t in active_memories.items()}

        for idx, p_rep in enumerate(s4_sum.get(env_name, [])):
            id_a = str(p_rep["id_a"])
            id_b = str(p_rep["id_b"])
            pair_key = tuple(sorted((id_a, id_b)))
            kind = str(p_rep["pair_kind"])
            s_ab_summary = p_rep.get("learned_s_ij")
            in_active_bank = (id_a in active_memories) and (id_b in active_memories) and (pair_key in state_pairs)

            pair_dir = stage4_dir / env_name / f"pair_{idx}_{kind}"
            trs0 = [
                json.loads(line)
                for line in (pair_dir / "cell_none" / "transitions.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip() and int(json.loads(line)["step"]) == 0
            ]
            eps0 = [
                json.loads(line)
                for line in (pair_dir / "cell_none" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(trs0) != len(eps0):
                raise RuntimeError(f"Step-0 transition count mismatch in {pair_dir}")

            if not in_active_bank:
                results.append(
                    {
                        "model_key": model_key,
                        "model_label": model_label,
                        "env_name": env_name,
                        "pair_index": idx,
                        "pair_kind": kind,
                        "selection_source": p_rep.get("selection_source"),
                        "target_context": p_rep.get("target_context"),
                        "state_path": str(state_path),
                        "state_sha256": state_sha,
                        "id_a": id_a,
                        "id_b": id_b,
                        "in_final_active_bank": False,
                        "learned_s_ij": s_ab_summary,
                        "n_tasks": len(eps0),
                        "mean_gated_logit_contrast": None,
                        "mean_pred_prob_contrast": None,
                        "mean_pred_prob_contrast_pp": None,
                        "mean_pred_cells": None,
                        "empirical_cell_rates": p_rep.get("cell_rates"),
                        "empirical_linear_delta_AB": p_rep.get("causal_interaction_delta_AB"),
                        "tasks": [],
                        "note": "Candidate includes archived rule or NA posterior s_ij; excluded from final active-bank fixed-gate prediction.",
                    }
                )
                continue

            s_ab = float(state_pairs[pair_key])
            if s_ab_summary is not None and abs(round(s_ab, 4) - float(s_ab_summary)) > 1e-4:
                raise RuntimeError(f"State pair mean {s_ab} does not match summary {s_ab_summary} for {pair_key}")
            th_a = float(ind_means[id_a])
            th_b = float(ind_means[id_b])
            task_rows = []
            logit_contrasts = []
            prob_contrasts = []
            pred_none, pred_a, pred_b, pred_ab = [], [], [], []

            for ep_rec, tr0 in zip(eps0, trs0):
                if str(ep_rec["task_id"]) != str(tr0["task_id"]):
                    raise RuntimeError(f"Task ID mismatch at episode {ep_rec['episode']} in {pair_dir}")
                ctx = str(ep_rec["context"])
                q = extract_routing_query(str(tr0["observation"]), env_name)
                qv = encoder.encode(q)
                if ctx and ctx != "default":
                    cv = encoder.encode(ctx.replace(":", " ").replace("-", " "))
                    raw_rel = {
                        i: (
                            0.65 * similarity(qv, v)
                            + 0.15 * similarity(qv, bank_scopes[i])
                            + 0.20 * similarity(cv, bank_scopes[i])
                        )
                        for i, v in bank_vecs.items()
                    }
                else:
                    raw_rel = {
                        i: 0.80 * similarity(qv, v) + 0.20 * similarity(qv, bank_scopes[i])
                        for i, v in bank_vecs.items()
                    }
                vals = sorted(raw_rel.values())
                peak = max(vals, default=0.0)
                if peak <= 1e-9:
                    eff_rel = {k: 0.0 for k in raw_rel}
                elif len(vals) >= 4:
                    bg = float(np.median(vals))
                    if peak > bg + 1e-6:
                        ref = max(0.5 * peak + 0.5 * vals[-2], bg + 1e-6)
                        span = ref - bg
                        eff_rel = {
                            k: float(np.clip((max(0.0, v - bg) / span) ** 0.65, 0.03, 1.0))
                            for k, v in raw_rel.items()
                        }
                    else:
                        eff_rel = {k: float((v / peak) ** 2) for k, v in raw_rel.items()}
                else:
                    eff_rel = {k: float((v / peak) ** 2) for k, v in raw_rel.items()}

                g_a = float(eff_rel[id_a])
                g_b = float(eff_rel[id_b])
                alpha_c = float(ctx_means.get(ctx, 0.0))

                eta_0 = alpha_c
                eta_a = alpha_c + g_a * th_a
                eta_b = alpha_c + g_b * th_b
                gated_logit = synergy_gamma * math.sqrt(g_a * g_b) * s_ab
                eta_ab = alpha_c + g_a * th_a + g_b * th_b + gated_logit

                p_0 = float(sigmoid(eta_0))
                p_a = float(sigmoid(eta_a))
                p_b = float(sigmoid(eta_b))
                p_ab = float(sigmoid(eta_ab))

                lgt_c = float(eta_ab - eta_a - eta_b + eta_0)
                prb_c = float(p_ab - p_a - p_b + p_0)

                logit_contrasts.append(lgt_c)
                prob_contrasts.append(prb_c)
                pred_none.append(p_0)
                pred_a.append(p_a)
                pred_b.append(p_b)
                pred_ab.append(p_ab)

                task_rows.append(
                    {
                        "episode": int(ep_rec["episode"]),
                        "task_id": str(ep_rec["task_id"]),
                        "context": ctx,
                        "routing_query": q,
                        "g_A": round(g_a, 6),
                        "g_B": round(g_b, 6),
                        "alpha_c": round(alpha_c, 6),
                        "theta_A": round(th_a, 6),
                        "theta_B": round(th_b, 6),
                        "s_AB": round(s_ab, 6),
                        "eta_none": round(eta_0, 6),
                        "eta_A": round(eta_a, 6),
                        "eta_B": round(eta_b, 6),
                        "eta_AB": round(eta_ab, 6),
                        "pred_p_none": round(p_0, 6),
                        "pred_p_A": round(p_a, 6),
                        "pred_p_B": round(p_b, 6),
                        "pred_p_AB": round(p_ab, 6),
                        "gated_logit_contrast": round(lgt_c, 6),
                        "pred_prob_contrast": round(prb_c, 6),
                    }
                )

            results.append(
                {
                    "model_key": model_key,
                    "model_label": model_label,
                    "env_name": env_name,
                    "pair_index": idx,
                    "pair_kind": kind,
                    "selection_source": p_rep.get("selection_source"),
                    "target_context": p_rep.get("target_context"),
                    "state_path": str(state_path),
                    "state_sha256": state_sha,
                    "id_a": id_a,
                    "id_b": id_b,
                    "in_final_active_bank": True,
                    "theta_A": round(th_a, 6),
                    "theta_B": round(th_b, 6),
                    "learned_s_ij": round(s_ab, 6),
                    "n_tasks": len(task_rows),
                    "mean_gated_logit_contrast": round(float(np.mean(logit_contrasts)), 6),
                    "mean_pred_prob_contrast": round(float(np.mean(prob_contrasts)), 6),
                    "mean_pred_prob_contrast_pp": round(float(np.mean(prob_contrasts) * 100.0), 4),
                    "mean_pred_cells": {
                        "none": round(float(np.mean(pred_none)), 6),
                        "A": round(float(np.mean(pred_a)), 6),
                        "B": round(float(np.mean(pred_b)), 6),
                        "AB": round(float(np.mean(pred_ab)), 6),
                    },
                    "empirical_cell_rates": p_rep.get("cell_rates"),
                    "empirical_linear_delta_AB": p_rep.get("causal_interaction_delta_AB"),
                    "tasks": task_rows,
                }
            )
    return results


def main():
    encoder = LocalEncoder(BGE_MODEL_PATH)
    rows = []
    rows.extend(
        compute_model_4cell_scale_audit(
            "deepseek_v41_flash",
            "DeepSeek-V4.1-Flash",
            REPO_ROOT / "outputs" / "stage3-main",
            REPO_ROOT / "outputs" / "stage4-4cell-v21",
            encoder,
        )
    )
    rows.extend(
        compute_model_4cell_scale_audit(
            "qwen3_8b",
            "Qwen3-8B",
            REPO_ROOT / "outputs" / "stage3-qwen3-8b",
            REPO_ROOT / "outputs" / "stage4-4cell-qwen3-8b",
            encoder,
        )
    )
    out_path = REPO_ROOT / "paper_artifacts" / "w3_counterfactual_scale_audit.json"
    write_json(out_path, {"kind": "w3_counterfactual_scale_audit", "pairs": rows})
    for r in rows:
        gl = f"{r['mean_gated_logit_contrast']:+.3f}" if r["mean_gated_logit_contrast"] is not None else "NA"
        pp = f"{r['mean_pred_prob_contrast_pp']:+.1f}pp" if r["mean_pred_prob_contrast_pp"] is not None else "NA"
        emp = f"{r['empirical_linear_delta_AB']*100:+.1f}pp"
        print(f"{r['model_label']:20s} | {r['env_name']:12s} | {r['pair_kind']:17s} | {r['id_a']}+{r['id_b']} | GatedLogit={gl:6s} | PredDeltaP={pp:7s} | EmpDeltaP={emp}")


if __name__ == "__main__":
    main()
