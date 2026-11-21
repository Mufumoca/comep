from collections import deque
from time import perf_counter, process_time

import numpy as np

from comep.joint import fit_joint
from comep.policy import FixedBankPolicy


def forecast_loss(prediction, reward):
    probability = float(np.clip(prediction, 1e-12, 1-1e-12))
    return -float(reward*np.log(probability)+(1-reward)*np.log1p(-probability))


def paired_gate(records):
    count = len(records)
    if count < 200:
        return {"count": count, "deploy_joint": False, "mean_gain": None, "threshold": None,
                "exploration_count": sum(record[1] for record in records)}
    values = np.array([record[0] for record in records])
    mean = float(np.mean(values))
    threshold = max(.005, float(1.96*np.std(values, ddof=1)/np.sqrt(count)))
    exploratory = [gain for gain, exploring in records if exploring]
    exploitative = [gain for gain, exploring in records if not exploring]
    return {"count": count, "deploy_joint": mean > threshold, "mean_gain": mean, "threshold": threshold,
            "exploration_count": len(exploratory),
            "exploration_mean_gain": float(np.mean(exploratory)) if exploratory else None,
            "exploitation_mean_gain": float(np.mean(exploitative)) if exploitative else None}


class PrequentialPolicy(FixedBankPolicy):
    def __init__(self, mode, costs, window=4000, refit_every=100, ridge=1.):
        if mode not in ("additive", "joint", "gated"):
            raise ValueError("unknown prequential policy mode")
        super().__init__("normalized-joint-soft", costs, window=window, refit_every=refit_every, ridge=ridge)
        self.mode = mode
        self.additive_fit = None
        self.joint_fit = None
        self.loss_history = deque(maxlen=400)
        self.gate = paired_gate(self.loss_history)
        self.pending = None
        self.last_refit = None

    def choose(self, context, gates, exploration_index=None):
        if self.pending is not None:
            raise RuntimeError("observe the pending action before another choose")
        wall_start, cpu_start = perf_counter(), process_time()
        design = self.design(context, gates)
        main_dimension = 2+len(self.costs)
        use_joint = self.mode == "joint" or (self.mode == "gated" and self.gate["deploy_joint"])
        selected_fit = self.joint_fit if use_joint else self.additive_fit
        selected_design = design if use_joint else design[:, :main_dimension]
        scores = selected_design @ selected_fit.coefficients if selected_fit is not None else np.zeros(len(design))
        if exploration_index is not None:
            if not 0 <= exploration_index < len(self.subsets):
                raise ValueError("invalid exploration index")
            chosen = int(exploration_index)
        else:
            chosen = int(np.argmax(scores))
        forecasts = {}
        for name, fitted, features in (("additive", self.additive_fit, design[chosen, :main_dimension]),
                                       ("joint", self.joint_fit, design[chosen])):
            forecasts[name] = float(fitted.predict(features)) if fitted is not None and fitted.converged else None
        row = design[chosen].copy()
        self.pending = {"row": row, "chosen": chosen, "forecasts": forecasts,
                        "exploration": exploration_index is not None}
        mean_pair = float(self.joint_fit.coefficients[main_dimension]*gates[0]*gates[1]) if self.joint_fit is not None else None
        metadata = {"forecasts": forecasts, "gate": dict(self.gate), "selected_model": "joint" if use_joint else "additive",
                    "last_fit_observation": self.observations-self.observations % self.refit_every,
                    "pair01_mean": mean_pair, "route_wall_seconds": perf_counter()-wall_start,
                    "route_cpu_seconds": process_time()-cpu_start}
        return chosen, metadata

    def observe(self, subset_index, reward):
        if self.pending is None or subset_index != self.pending["chosen"] or reward not in (0, 1):
            raise ValueError("reward must match pending executed action")
        pending = self.pending
        self.pending = None
        forecasts = pending["forecasts"]
        difference = None
        if self.mode == "gated" and all(value is not None for value in forecasts.values()):
            difference = forecast_loss(forecasts["additive"], reward)-forecast_loss(forecasts["joint"], reward)
            self.loss_history.append((difference, pending["exploration"]))
        self.history.append((pending["row"], reward, subset_index))
        self.observations += 1
        self.last_refit = None
        if self.observations % self.refit_every == 0:
            wall_start, cpu_start = perf_counter(), process_time()
            design = np.array([record[0] for record in self.history])
            rewards = np.array([record[1] for record in self.history])
            main_dimension = 2+len(self.costs)
            if self.mode != "joint":
                self.additive_fit = fit_joint(design[:, :main_dimension], rewards, ridge=self.ridge, max_iterations=100)
            if self.mode != "additive":
                self.joint_fit = fit_joint(design, rewards, ridge=self.ridge, max_iterations=100)
            if self.mode == "gated":
                self.gate = paired_gate(self.loss_history)
                if not self.additive_fit.converged or not self.joint_fit.converged:
                    self.gate = {**self.gate, "deploy_joint": False, "reason": "nonconverged"}
            self.last_refit = {"after_episode": self.observations, "history_count": len(self.history),
                               "wall_seconds": perf_counter()-wall_start, "cpu_seconds": process_time()-cpu_start,
                               "gate": dict(self.gate), "fits": {}}
            for name, fitted in (("additive", self.additive_fit), ("joint", self.joint_fit)):
                if fitted is not None:
                    self.last_refit["fits"][name] = {"coefficients": fitted.coefficients.tolist(), "rank": fitted.rank,
                                                    "converged": fitted.converged, "iterations": fitted.iterations,
                                                    "gradient_norm": fitted.gradient_norm}
            if self.joint_fit is not None:
                contrast = np.zeros(design.shape[1])
                contrast[main_dimension] = .25
                self.last_refit["pair01_effect"] = self.joint_fit.contrast(contrast)
        return {"paired_logloss_gain": difference}
