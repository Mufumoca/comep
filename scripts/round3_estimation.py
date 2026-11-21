import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from comep.core import Config, MemoryManager, Selection
from comep.experiments import new_directory, write_json
from comep.joint import fit_joint, probabilities


CELLS = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]], dtype=float)
SEEDS = [701, 702, 703, 704, 705]


def schedule_for(seed, count, balanced=True):
    rng = np.random.default_rng(seed)
    schedule = np.tile(np.arange(4), count//4) if balanced else np.full(count, 3)
    if balanced:
        for offset in range(0, count, 4):
            rng.shuffle(schedule[offset:offset+4])
    return schedule, rng.random(count)


def summarize_fit(fit, evaluation, target, scale, variant, metadata):
    prediction = fit.predict(evaluation)
    contrast = np.array([0, 0, 0, scale[-1]]) if len(scale) == 4 else None
    return {**metadata, "variant": variant, "probabilities": prediction.tolist(), "target_probabilities": target.tolist(), "effective_coefficients": (fit.coefficients*scale).tolist(), "cell_rmse": float(np.sqrt(np.mean((prediction-target)**2))), "rank": fit.rank, "converged": fit.converged, "gradient_norm": fit.gradient_norm, "iterations": fit.iterations, "pair_evidence": fit.contrast(contrast) if contrast is not None else None}


def run(output):
    output = new_directory(output)
    config = Config(exploration=0, sampling_scale=0, redundancy=0)
    write_json(output/"manifest.json", {"kind": "round3_estimator_development", "seeds": SEEDS, "checkpoints": [200,1000,4000], "gates": [1.0,0.25], "effects": [0.0,-3.0], "ridge": 0.1, "prior": "summed logistic loss; L2 in raw or RMS-normalized coordinates as named", "legacy_config": asdict(config), "api_calls": 0, "protocol": "refine-logs/ROUND3_PLAN_20261103.md"})
    rows = []
    for seed in SEEDS:
        for effect in [0.0,-3.0]:
            target = probabilities(CELLS @ np.array([-1,1,1,effect]))
            for exposure in ["balanced","always-both"]:
                schedule, uniform = schedule_for(seed,4000,exposure=="balanced")
                rewards = (uniform < target[schedule]).astype(int)
                np.savez_compressed(output/f"data-{seed}-{effect}-{exposure}.npz",schedule=schedule,uniform=uniform,rewards=rewards)
                for gate in [1.0,0.25]:
                    scale = np.array([1,gate,gate,gate**2])
                    design = CELLS[schedule]*scale
                    manager = MemoryManager("comep",config,seed)
                    for identifier in ["first","second"]:
                        manager.add(identifier,identifier+" rule")
                    for index,cell in enumerate(schedule):
                        identifiers=[identifier for identifier,active in zip(["first","second"],CELLS[cell,1:3]) if active]
                        manager.observe(Selection(identifiers,{"first":gate,"second":gate},"context",len(identifiers)*2,False),int(rewards[index]))
                        count=index+1
                        if count not in [200,1000,4000]:
                            continue
                        metadata={"seed":seed,"true_effect":effect,"exposure":exposure,"gate":gate,"observations":count}
                        for variant,normalize in [("raw-joint",False),("normalized-joint",True)]:
                            fitted=fit_joint(design[:count],rewards[:count],normalize=normalize)
                            rows.append(summarize_fit(fitted,CELLS*scale,target,scale,variant,metadata))
                        fitted=fit_joint(design[:count,:3],rewards[:count],normalize=True)
                        rows.append(summarize_fit(fitted,(CELLS*scale)[:,:3],target,scale[:3],"normalized-additive",metadata))
                        effective=np.array([manager.contexts["context"].mean,manager.individual["first"].mean,manager.individual["second"].mean,manager.pair_mean("first","second")])*scale
                        prediction=probabilities(CELLS@effective)
                        rows.append({**metadata,"variant":"legacy-sgd","probabilities":prediction.tolist(),"effective_coefficients":effective.tolist(),"cell_rmse":float(np.sqrt(np.mean((prediction-target)**2))),"legacy_conflict":manager.conflict("first","second")})
    shift=[]
    for seed in SEEDS:
        schedule,uniform=schedule_for(seed,2000)
        effect=np.where(np.arange(2000)<1000,-3.0,3.0)
        base=-1+CELLS[schedule,1]+CELLS[schedule,2]+effect*CELLS[schedule,3]
        rewards=(uniform<probabilities(base)).astype(int)
        np.savez_compressed(output/f"shift-{seed}.npz",schedule=schedule,uniform=uniform,rewards=rewards)
        for count in [1000,1200,1600,2000]:
            target=probabilities(CELLS@np.array([-1,1,1,-3 if count<=1000 else 3]))
            for window in [None,400]:
                start=0 if window is None else max(0,count-window)
                fit=fit_joint(CELLS[schedule[start:count]],rewards[start:count])
                shift.append(summarize_fit(fit,CELLS,target,np.ones(4),"cumulative" if window is None else "window400",{"seed":seed,"observations":count,"window_start":start}))
    normalized=[row for row in rows if row["variant"]=="normalized-joint"]
    max_difference=0.0
    for row in normalized:
        if row["gate"]!=1:
            continue
        other=next(item for item in normalized if item["gate"]==0.25 and all(item[key]==row[key] for key in ["seed","true_effect","exposure","observations"]))
        max_difference=max(max_difference,float(np.max(np.abs(np.array(row["probabilities"])-other["probabilities"]))))
    gates={"scale_max_prediction_difference":max_difference,"scale_invariance_pass":max_difference<1e-8,"always_both_all_unknown":all(row["pair_evidence"]["status"]=="unknown" for row in normalized if row["exposure"]=="always-both"),"zero_effect_final_negative_declarations":sum(row["pair_evidence"]["status"]=="negative" for row in normalized if row["true_effect"]==0 and row["observations"]==4000 and row["exposure"]=="balanced"),"all_joint_fits_converged":all(row["converged"] for row in rows if row["variant"]!="legacy-sgd")}
    write_json(output/"results.json",{"rows":rows,"shift":shift,"diagnostic_gates":gates,"warning":"Development checks, approximate Laplace evidence; no calibrated coverage or causal/benchmark claim."})
    print(json.dumps(gates,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args().output)
