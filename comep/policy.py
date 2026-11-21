from collections import deque
from itertools import combinations
from time import perf_counter

import numpy as np

from comep.joint import fit_joint


class FixedBankPolicy:
    methods = ("individual", "normalized-additive", "normalized-joint-soft", "normalized-joint-hard")

    def __init__(self, method, costs, limit=3, budget=24, window=400, refit_every=20, ridge=.1):
        if method not in self.methods:
            raise ValueError("unknown method")
        if not costs or min(costs) <= 0 or limit < 0 or budget < 0 or window < 1 or refit_every < 1:
            raise ValueError("invalid fixed-bank settings")
        self.method = method
        self.costs = np.asarray(costs, dtype=int)
        self.pairs = list(combinations(range(len(costs)), 2))
        self.subsets = [subset for size in range(min(limit, len(costs))+1)
                        for subset in combinations(range(len(costs)), size)
                        if sum(self.costs[list(subset)]) <= budget]
        self.membership = np.zeros((len(self.subsets), len(costs)))
        for index, subset in enumerate(self.subsets):
            self.membership[index, list(subset)] = 1
        self.pair_membership = np.array([self.membership[:, first] * self.membership[:, second]
                                         for first, second in self.pairs]).T
        self.joint = method.startswith("normalized-joint")
        self.history = deque(maxlen=window)
        self.refit_every = refit_every
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("positive finite ridge required")
        self.ridge = ridge
        self.observations = 0
        self.fit = None
        self.last_fit_seconds = 0.0

    def design(self, context, gates):
        gates = np.asarray(gates, dtype=float)
        if gates.shape != self.costs.shape or not np.isfinite(gates).all() or np.any(gates < 0):
            raise ValueError("finite nonnegative gates required")
        main = self.membership * gates
        parts = [np.ones((len(self.subsets), 1)), np.full((len(self.subsets), 1), context), main]
        if self.joint:
            parts.append(np.array([main[:, first] * main[:, second] for first, second in self.pairs]).T)
        return np.concatenate(parts, axis=1)

    def pair_evidence(self, gates):
        if not self.joint or self.fit is None:
            return ["unknown"] * len(self.pairs)
        states = []
        offset = 2 + len(self.costs)
        for index, (first, second) in enumerate(self.pairs):
            contrast = np.zeros(len(self.fit.coefficients))
            contrast[offset+index] = gates[first] * gates[second]
            states.append(self.fit.contrast(contrast)["status"] if contrast[offset+index] else "unknown")
        return states

    def choose(self, context, gates, exploration_index=None):
        started = perf_counter()
        design = self.design(context, gates)
        states = self.pair_evidence(gates)
        if exploration_index is not None:
            if exploration_index < 0 or exploration_index >= len(self.subsets):
                raise ValueError("invalid exploration subset index")
            chosen = exploration_index
        else:
            if self.method == "individual":
                totals = np.zeros(len(self.costs))
                counts = np.zeros(len(self.costs))
                for _, reward, subset_index in self.history:
                    totals += self.membership[subset_index] * (reward-.5)
                    counts += self.membership[subset_index]
                quality = np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)
                scores = self.membership @ (np.asarray(gates) * quality)
            else:
                scores = design @ self.fit.coefficients if self.fit is not None else np.zeros(len(design))
            if self.method == "normalized-joint-hard":
                negative = np.array([state == "negative" for state in states])
                scores[(self.pair_membership[:, negative] > 0).any(axis=1)] = -np.inf
            chosen = int(np.argmax(scores))
        return chosen, {"pair_states": states, "rank": self.fit.rank if self.fit else 0,
                        "converged": self.fit.converged if self.fit else None,
                        "route_seconds": perf_counter()-started}

    def observe(self, context, gates, subset_index, reward):
        if reward not in (0, 1) or not 0 <= subset_index < len(self.subsets):
            raise ValueError("valid executed subset and binary reward required")
        self.history.append((self.design(context, gates)[subset_index], reward, subset_index))
        self.observations += 1
        self.last_fit_seconds = 0.0
        if self.method != "individual" and self.observations % self.refit_every == 0:
            started = perf_counter()
            self.fit = fit_joint(np.array([record[0] for record in self.history]),
                                 np.array([record[1] for record in self.history]), max_iterations=100, ridge=self.ridge)
            self.last_fit_seconds = perf_counter()-started
