from time import perf_counter

import numpy as np

from comep.joint import fit_joint
from comep.policy import FixedBankPolicy


def binary_losses(predicted, rewards):
    predicted = np.clip(predicted, 1e-12, 1-1e-12)
    return -(rewards*np.log(predicted)+(1-rewards)*np.log1p(-predicted))


class SparseValidatedPolicy(FixedBankPolicy):
    def __init__(self, costs, **kwargs):
        super().__init__("normalized-joint-soft", costs, **kwargs)
        self.active_pair = None
        self.candidate_fit = None
        self.selection = {"candidate_pair": None, "validation_pass": False, "reason": "insufficient_history"}

    def choose(self, context, gates, exploration_index=None):
        started = perf_counter()
        design = self.design(context, gates)
        if exploration_index is not None:
            if not 0 <= exploration_index < len(self.subsets):
                raise ValueError("invalid exploration index")
            chosen = exploration_index
        else:
            if self.active_pair is not None:
                columns = list(range(2+len(self.costs))) + [2+len(self.costs)+self.active_pair]
                scores = design[:, columns] @ self.candidate_fit.coefficients
            elif self.fit is not None:
                scores = design[:, :2+len(self.costs)] @ self.fit.coefficients
            else:
                scores = np.zeros(len(self.subsets))
            chosen = int(np.argmax(scores))
        return chosen, {"route_seconds": perf_counter()-started, "active_pair": self.active_pair,
                        "selection": dict(self.selection), "rank": self.fit.rank if self.fit else 0}

    def observe(self, context, gates, subset_index, reward):
        if reward not in (0, 1) or not 0 <= subset_index < len(self.subsets):
            raise ValueError("valid executed subset and binary reward required")
        self.history.append((self.design(context, gates)[subset_index], reward, subset_index))
        self.observations += 1
        self.last_fit_seconds = 0.
        if self.observations % self.refit_every:
            return
        started = perf_counter()
        design = np.array([record[0] for record in self.history])
        rewards = np.array([record[1] for record in self.history])
        offset = 2+len(self.costs)
        self.fit = fit_joint(design[:, :offset], rewards, ridge=self.ridge)
        self.active_pair = None
        self.candidate_fit = None
        self.selection = {"candidate_pair": None, "validation_pass": False, "reason": "insufficient_history",
                          "all_fits_converged": self.fit.converged}
        if len(self.history) >= 200:
            split = len(self.history)//2
            main = design[:split, :offset]
            discover = fit_joint(main, rewards[:split], ridge=self.ridge)
            predicted = discover.predict(main)
            weights = predicted*(1-predicted)
            normalized = main/discover.scales
            pair = design[:split, offset:]
            pair_scales = np.sqrt(np.mean(pair**2, axis=0))
            pair_scales[pair_scales == 0] = 1.
            pair = pair/pair_scales
            information = normalized.T @ (weights[:, None]*normalized) + self.ridge*np.eye(offset)
            residualized = pair-normalized @ np.linalg.solve(information, normalized.T @ (weights[:, None]*pair))
            scores = np.abs(residualized.T @ (rewards[:split]-predicted))/np.sqrt(np.sum(weights[:, None]*residualized**2, axis=0)+1.)
            candidate = int(np.argmax(scores))
            columns = list(range(offset)) + [offset+candidate]
            proposed = fit_joint(design[:split, columns], rewards[:split], ridge=self.ridge)
            improvement = binary_losses(discover.predict(design[split:, :offset]), rewards[split:])-binary_losses(proposed.predict(design[split:, columns]), rewards[split:])
            mean = float(np.mean(improvement))
            threshold = max(.01, float(1.96*np.std(improvement, ddof=1)/np.sqrt(len(improvement))))
            converged = self.fit.converged and discover.converged and proposed.converged
            passed = converged and mean > threshold
            self.selection = {"candidate_pair": candidate, "validation_pass": bool(passed),
                              "reason": "passed" if passed else "nonconverged" if not converged else "validation_no_gain",
                              "discovery_score": float(scores[candidate]), "validation_improvement": mean,
                              "validation_threshold": threshold, "discovery_observations": split,
                              "validation_observations": len(improvement), "all_fits_converged": bool(converged)}
            if passed:
                self.candidate_fit = fit_joint(design[:, columns], rewards, ridge=self.ridge)
                if self.candidate_fit.converged:
                    self.active_pair = candidate
                else:
                    self.selection.update(validation_pass=False, reason="full_refit_nonconverged", all_fits_converged=False)
        self.last_fit_seconds = perf_counter()-started
