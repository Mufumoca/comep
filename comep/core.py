from collections import deque
from dataclasses import asdict, dataclass
from itertools import combinations
import math
import re

import numpy as np

from .evomemguard import EvoMemGuardMixin
from .text import LocalEncoder, TokenCounter, similarity


METHODS = ("no-memory", "full-memory", "individual", "mmr", "additive", "comep", "proposal", "evomemguard")

_ACTION_CLAUSE_STARTERS = frozenset({
    "check", "go", "hold", "place", "search", "open", "wash", "chill", "move",
    "take", "leave", "focus", "wire", "larger", "pick", "navigate", "compare",
    "treat", "immediately", "complete", "use", "run", "wait1", "look", "first",
    "then", "never", "avoid", "do", "activate", "connect",
})


def _split_scope_action(text: str) -> tuple[str, str]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) >= 2:
        for idx in range(1, len(parts)):
            words = re.findall(r"\w+", parts[idx].lower())
            if words and words[0] in _ACTION_CLAUSE_STARTERS:
                return ", ".join(parts[:idx]), ", ".join(parts[idx:])
        return parts[0], ", ".join(parts[1:])
    return text, text


def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-float(np.clip(value, -40, 40))))


def _sigmoid_vec(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


@dataclass
class Config:
    retrieval_tokens: int = 120
    bank_tokens: int = 600
    max_items: int = 5
    max_bank_items: int = 50
    learning_rate: float = 0.15
    pair_rate: float = 0.1
    regularization: float = 0.01
    pair_l1: float = 0.01
    prior_precision: float = 1.0
    pair_prior_precision: float = 1.2
    prior_mean: float = 0.25
    sampling_scale: float = 0.3
    exploration: float = 0.1
    redundancy: float = 0.15
    synergy: float = 1.0
    conflict_threshold: float = 0.2
    conflict_z: float = 1.645
    min_pair_observations: int = 5
    pruning_uncertainty: float = 0.25
    pair_lookahead: bool = True
    hard_conflicts: bool = True
    admission_grace_episodes: int = 0
    retrieval_redundancy: float | None = None
    pruning_redundancy: float | None = None
    matched_pair_lookahead: bool = False
    batch_laplace: bool = True
    history_window: int = 600
    min_active_pair_count: int = 1
    max_active_pairs: int = 48
    top_pair_candidates: int = 6
    ucb_scale: float = 0.65
    pair_ucb_scale: float = 0.25
    min_prob_gain_per_token: float = 0.0
    normalize_relevance: bool = True
    coalition_pruning: bool = True
    v2_routing_and_features: bool = True
    warmup_gate: bool = False
    warmup_min_context_obs: int = 5
    warmup_min_item_obs: int = 2
    pair_confirmation_bonus: bool = True
    pair_random_explore: bool = True
    nli_model: str | None = None
    nli_conflict_threshold: float = 0.55

    def __post_init__(self):
        for flag in (
            "matched_pair_lookahead",
            "batch_laplace",
            "normalize_relevance",
            "coalition_pruning",
            "pair_lookahead",
            "hard_conflicts",
            "v2_routing_and_features",
            "warmup_gate",
            "pair_confirmation_bonus",
            "pair_random_explore",
        ):
            if not isinstance(getattr(self, flag), bool):
                raise ValueError(f"{flag} must be boolean")
        for name in (
            "retrieval_tokens",
            "bank_tokens",
            "max_items",
            "max_bank_items",
            "min_pair_observations",
            "history_window",
            "min_active_pair_count",
            "max_active_pairs",
            "top_pair_candidates",
            "warmup_min_context_obs",
            "warmup_min_item_obs",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "learning_rate",
            "pair_rate",
            "regularization",
            "pair_l1",
            "prior_precision",
            "pair_prior_precision",
            "prior_mean",
            "sampling_scale",
            "redundancy",
            "synergy",
            "conflict_threshold",
            "conflict_z",
            "pruning_uncertainty",
            "ucb_scale",
            "pair_ucb_scale",
            "min_prob_gain_per_token",
            "nli_conflict_threshold",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.prior_precision <= 0 or self.pair_prior_precision <= 0 or not 0 <= self.exploration <= 1:
            raise ValueError("positive prior precision and exploration in [0,1] required")
        if not isinstance(self.admission_grace_episodes, int) or isinstance(self.admission_grace_episodes, bool) or self.admission_grace_episodes < 0:
            raise ValueError("admission_grace_episodes must be a nonnegative integer")
        for value in (self.retrieval_redundancy, self.pruning_redundancy):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("redundancy overrides must be finite and nonnegative")


@dataclass(frozen=True)
class Memory:
    identifier: str
    text: str
    tokens: int
    created_at: int = 0


@dataclass
class Parameter:
    mean: float = 0.0
    precision: float = 1.0
    count: int = 0


@dataclass
class Selection:
    identifiers: list[str]
    relevance: dict[str, float]
    context: str
    tokens: int
    explored: bool


class MemoryManager(EvoMemGuardMixin):
    def __init__(self, method="comep", config=None, seed=0, encoder=None, counter=None):
        if method not in METHODS:
            raise ValueError(f"unknown method: {method}")
        self.method = method
        self.config = config or Config()
        self.rng = np.random.default_rng(seed)
        self.encoder = encoder or LocalEncoder()
        self.counter = counter or TokenCounter()
        self.memories = {}
        self.vectors = {}
        self.scope_vectors = {}
        self.action_vectors = {}
        self.individual = {}
        self.pairs = {}
        self.contexts = {}
        self.archive = []
        self.seen_ids = set()
        self.reward_sum = 0.0
        self.observations = 0
        self.last_prediction = None
        self.inserted_at_observation = {}
        self.history = deque(maxlen=self.config.history_window)
        self._pair_counts = {}
        self._pair_wins = {}
        self._pair_contexts = {}
        self._item_contexts = {}
        self._born_context = {}
        self._last_context = "default"
        self._solo_counts = {}
        self._warm_coefs = {}
        self.trigger_counts = {
            "episodes_routed": 0,
            "hard_conflict_blocked_episodes": 0,
            "early_stop_trimmed_episodes": 0,
            "early_stop_loop_break_episodes": 0,
            "selected_pair_cold_ucb_episodes": 0,
            "selected_pair_second_chance_ucb_episodes": 0,
            "selected_pair_confirmation_episodes": 0,
            "selected_pair_random_explore_episodes": 0,
            "warmup_gate_blocked_episodes": 0,
            "prune_evictions": 0,
            "coalition_prune_decision_changes": 0,
            "eval_hard_conflict_blocks": 0,
            "eval_pair_ucb_cold_calls": 0,
            "eval_pair_ucb_second_chance_calls": 0,
            "eval_pair_confirmation_calls": 0,
            "eval_warmup_gate_blocks": 0,
            "eval_coalition_prune_pos_items": 0,
            "eval_coalition_prune_neg_items": 0,
            "eval_coalition_prune_blame_guard_items": 0,
        }
        if self.method == "evomemguard":
            self._init_evomemguard(self.config.nli_model, self.config.nli_conflict_threshold)

    @property
    def interactions(self):
        return self.method in ("comep", "proposal")

    @property
    def model_based(self):
        return self.method in ("comep", "proposal", "additive")

    @property
    def v2_stack(self):
        return self.config.v2_routing_and_features and self.method in ("comep", "additive")

    @property
    def laplace_based(self):
        return self.config.batch_laplace and self.method in ("comep", "additive")

    @property
    def bank_tokens(self):
        return sum(memory.tokens for memory in self.memories.values())

    def parameter(self):
        init_mean = self.config.prior_mean if (self.v2_stack and self.config.sampling_scale > 0) else 0.0
        return Parameter(mean=init_mean, precision=self.config.prior_precision)

    @staticmethod
    def key(first, second):
        return tuple(sorted((first, second)))

    def pair_mean(self, first, second):
        parameter = self.pairs.get(self.key(first, second))
        return parameter.mean if parameter else 0.0

    def pair_count(self, first, second):
        key = self.key(first, second)
        parameter = self.pairs.get(key)
        return max(parameter.count if parameter else 0, self._pair_counts.get(key, 0))

    def add(self, identifier, text, step=0, prune=True):
        if not identifier or identifier in self.seen_ids:
            raise ValueError("memory identifier must be nonempty and globally unique")
        if not text.strip():
            raise ValueError("memory text must not be empty")
        self.seen_ids.add(identifier)
        if self.method == "no-memory":
            return []
        if self.method == "evomemguard":
            return self._evomemguard_admit(identifier, text, step, prune)
        memory = Memory(identifier, text, self.counter.count(text), step)
        self.memories[identifier] = memory
        self.vectors[identifier] = self.encoder.encode(text)
        scope_txt, act_txt = _split_scope_action(text)
        self.scope_vectors[identifier] = self.encoder.encode(scope_txt)
        self.action_vectors[identifier] = self.encoder.encode(act_txt)
        self.individual[identifier] = self.parameter()
        self.inserted_at_observation[identifier] = self.observations
        if self.observations > 0 and getattr(self, "_last_context", "default") != "default":
            self._born_context[identifier] = self._last_context
        return self.prune() if prune else []

    def conflict(self, first, second):
        key = self.key(first, second)
        if self.method == "evomemguard":
            return key in self.nli_conflicts
        parameter = self.pairs.get(key)
        if not self.interactions or parameter is None:
            return False
        upper = parameter.mean + self.config.conflict_z / math.sqrt(max(parameter.precision, 1e-8))
        cnt = self.pair_count(first, second)
        if cnt < self.config.min_pair_observations:
            return False
        if upper < -self.config.conflict_threshold:
            return True
        if self.v2_stack and key in self._pair_counts:
            wins = self._pair_wins.get(key, 0)
            if wins / cnt <= 0.25 and parameter.mean < -1.75 * self.config.conflict_threshold:
                return True
        return False

    def _effective_relevance(self, raw_relevance: dict[str, float]) -> dict[str, float]:
        if not raw_relevance or not self.config.normalize_relevance or self.method not in ("comep", "additive"):
            return raw_relevance
        vals = list(raw_relevance.values())
        peak = max(vals, default=0.0)
        if peak <= 1e-9:
            return {k: 0.0 for k in raw_relevance}
        if len(vals) >= 4:
            sorted_vals = sorted(vals)
            bg = float(np.median(sorted_vals))
            if peak > bg + 1e-6:
                ref = max(0.5 * peak + 0.5 * sorted_vals[-2], bg + 1e-6)
                span = ref - bg
                return {k: float(np.clip((max(0.0, v - bg) / span) ** 0.65, 0.03, 1.0)) for k, v in raw_relevance.items()}
        return {k: float((v / peak) ** 2) for k, v in raw_relevance.items()}

    def choose(self, query, context="default"):
        self.trigger_counts["episodes_routed"] += 1
        ep_flags = {
            "hard_conflict": False,
            "early_trim": False,
            "early_break": False,
            "warmup_block": False,
            "random_pair": False,
        }
        committed_bonus_kinds = set()
        query_vector = self.encoder.encode(query)
        dense_encoder = getattr(self.encoder, "model", None) is not None
        scope_map = getattr(self, "scope_vectors", {})
        if self.v2_stack and dense_encoder and scope_map:
            if context and context != "default":
                ctx_vector = self.encoder.encode(context.replace(":", " ").replace("-", " "))
                relevance = {
                    identifier: (
                        0.65 * similarity(query_vector, vector)
                        + 0.15 * similarity(query_vector, scope_map.get(identifier, vector))
                        + 0.20 * similarity(ctx_vector, scope_map.get(identifier, vector))
                    )
                    for identifier, vector in self.vectors.items()
                }
            else:
                relevance = {
                    identifier: (
                        0.80 * similarity(query_vector, vector)
                        + 0.20 * similarity(query_vector, scope_map.get(identifier, vector))
                    )
                    for identifier, vector in self.vectors.items()
                }
        else:
            relevance = {identifier: similarity(query_vector, vector) for identifier, vector in self.vectors.items()}
        eff_rel = self._effective_relevance(relevance)
        candidates = sorted(self.memories)
        selected = []
        tokens = 0
        eff_explore = (
            self.config.exploration / (1.0 + 0.08 * self.observations)
            if self.v2_stack
            else self.config.exploration
        )
        explored = self.model_based and self.rng.random() < eff_explore

        # Top-M query-relevant candidates eligible for pairwise optimism in CoMeP
        ranked_by_rel = sorted(candidates, key=lambda ident: (-relevance[ident], ident))
        top_pair_pool = set(ranked_by_rel[: self.config.top_pair_candidates])

        sampled = {}
        optimism_active = self.config.sampling_scale > 0
        item_contexts_map = getattr(self, "_item_contexts", {})
        born_context_map = getattr(self, "_born_context", {})
        for identifier in candidates:
            parameter = self.individual[identifier]
            if self.method == "evomemguard":
                sampled[identifier] = self._evomemguard_utility(identifier)
            elif self.model_based:
                variance = 1.0 / (1.0 + parameter.count) if self.method == "proposal" else 1.0 / max(parameter.precision, 1e-6)
                if self.v2_stack:
                    ucb = (self.config.ucb_scale * math.sqrt(self.config.sampling_scale * variance)) if optimism_active else 0.0
                    noise_scale = 0.25 if explored else 0.02
                    noise = self.rng.normal(0.0, noise_scale * math.sqrt(self.config.sampling_scale * variance)) if optimism_active else 0.0
                    item_ctxs = item_contexts_map.get(identifier)
                    born_ctx = born_context_map.get(identifier)
                    if context != "default" and born_ctx and born_ctx != context and (not item_ctxs or context not in item_ctxs):
                        eff_mean = min(parameter.mean, 0.0)
                    elif context != "default" and item_ctxs and context not in item_ctxs:
                        eff_mean = min(parameter.mean, self.config.prior_mean)
                    else:
                        eff_mean = parameter.mean
                    sampled[identifier] = eff_mean + ucb + noise
                else:
                    ucb = 0.0
                    noise = self.rng.normal(0.0, math.sqrt(self.config.sampling_scale * variance)) if optimism_active else 0.0
                    sampled[identifier] = parameter.mean + ucb + noise
            else:
                sampled[identifier] = parameter.mean

        def feasible(bundle):
            if len(selected) + len(bundle) > self.config.max_items:
                return False
            if tokens + sum(self.memories[identifier].tokens for identifier in bundle) > self.config.retrieval_tokens:
                return False
            if self.method == "evomemguard":
                return not any(self.conflict(first, second) for first, second in combinations(selected + list(bundle), 2))
            if self.config.hard_conflicts:
                if any(self.conflict(first, second) for first, second in combinations(selected + list(bundle), 2)):
                    self.trigger_counts["eval_hard_conflict_blocks"] += 1
                    ep_flags["hard_conflict"] = True
                    return False
            return True

        ctx_obs = self.contexts[context].count if context in self.contexts else 0
        pair_cold_decay = 1.0 / (1.0 + 0.06 * ctx_obs)

        def warmup_ready(first, second):
            if not self.config.warmup_gate:
                return True
            return (
                ctx_obs >= self.config.warmup_min_context_obs
                and self.individual[first].count >= self.config.warmup_min_item_obs
                and self.individual[second].count >= self.config.warmup_min_item_obs
            )

        def pair_term(first, second, coalition_size=1, record_eval=True, return_bonus_kind=False):
            key = self.key(first, second)
            raw_mean = self.pair_mean(first, second)
            cnt = self.pair_count(first, second)
            shrinkage_ratio = (self.config.pair_prior_precision + cnt) / max(self.config.prior_precision + cnt, 1e-6)
            min_marginal = min(self.individual[first].mean, self.individual[second].mean) + shrinkage_ratio * raw_mean
            # Suppress compensatory positive s_ij when adding a harmful rule still degrades its partner
            mean_val = 0.0 if (self.method == "comep" and raw_mean > 0.0 and min_marginal <= 0.0) else raw_mean
            bonus_kind = None
            if self.method == "comep" and optimism_active:
                if cnt > 0:
                    wins = self._pair_wins.get(key, 0)
                    solo_sep = max(
                        max(0, self.individual[first].count - cnt),
                        max(0, self.individual[second].count - cnt),
                    )
                    # Active collinearity-breaking: if a co-selected pair has 0 wins and zero solo separation, split once to identify credit
                    if solo_sep < 1 and wins == 0:
                        val = mean_val - 0.85 * math.sqrt(self.config.sampling_scale)
                        return (val, None) if return_bonus_kind else val
                    # Posterior confirmation bonus strictly for net-positive marginal coalitions
                    if (
                        self.config.pair_confirmation_bonus
                        and mean_val > 0.05
                        and min_marginal > 0.05
                        and wins >= 1
                        and (cnt >= 2 or solo_sep >= 1)
                        and key in self.pairs
                    ):
                        post_std = 1.0 / math.sqrt(max(self.pairs[key].precision, 1e-6))
                        mean_val += 1.25 * math.sqrt(self.config.sampling_scale) * post_std
                        if record_eval:
                            self.trigger_counts["eval_pair_confirmation_calls"] += 1
                        bonus_kind = "confirmation"
                    # Single second-chance UCB after solo separation when at least one partner lacks standalone mastery
                    elif (
                        coalition_size <= 1
                        and cnt < self.config.min_pair_observations
                        and raw_mean > -0.12
                        and min(self.individual[first].mean, self.individual[second].mean) < 0.35
                        and self.individual[first].mean > -0.12
                        and self.individual[second].mean > -0.12
                    ):
                        if warmup_ready(first, second):
                            bonus = self.config.pair_ucb_scale * pair_cold_decay * math.sqrt(self.config.sampling_scale / (1.0 + cnt))
                            if bonus > 0:
                                mean_val += bonus
                                if record_eval:
                                    self.trigger_counts["eval_pair_ucb_second_chance_calls"] += 1
                                bonus_kind = "second_chance"
                        elif record_eval:
                            self.trigger_counts["eval_warmup_gate_blocks"] += 1
                            ep_flags["warmup_block"] = True
                elif (
                    coalition_size <= 1
                    and first in top_pair_pool
                    and second in top_pair_pool
                    and eff_rel[first] >= 0.35
                    and eff_rel[second] >= 0.35
                    and self.individual[first].mean > -0.05
                    and self.individual[second].mean > -0.05
                    and (self.inserted_at_observation.get(first, 0) == 0 or self.individual[first].count >= 1)
                    and (self.inserted_at_observation.get(second, 0) == 0 or self.individual[second].count >= 1)
                ):
                    if warmup_ready(first, second):
                        bonus = self.config.pair_ucb_scale * pair_cold_decay * math.sqrt(self.config.sampling_scale)
                        if bonus > 0:
                            mean_val += bonus
                            if record_eval:
                                self.trigger_counts["eval_pair_ucb_cold_calls"] += 1
                            bonus_kind = "cold"
                    elif record_eval:
                        self.trigger_counts["eval_warmup_gate_blocks"] += 1
                        ep_flags["warmup_block"] = True
            if self.method == "comep" and mean_val > 0.0:
                query_matched = (
                    first in top_pair_pool
                    and second in top_pair_pool
                    and eff_rel[first] >= 0.35
                    and eff_rel[second] >= 0.35
                )
                observed_ctxs = self._pair_contexts.get(key)
                if observed_ctxs and context != "default":
                    if context not in observed_ctxs or eff_rel[first] < 0.25 or eff_rel[second] < 0.25:
                        return (0.0, None) if return_bonus_kind else 0.0
                elif not query_matched:
                    return (0.0, None) if return_bonus_kind else 0.0
            active_kind = bonus_kind if mean_val > 0.0 else None
            return (mean_val, active_kind) if return_bonus_kind else mean_val

        def gain(identifier, existing):
            if self.method in ("full-memory", "mmr"):
                value = relevance[identifier]
            elif self.method == "individual":
                value = relevance[identifier] * (1.0 + sampled[identifier])
            elif self.method == "evomemguard":
                value = relevance[identifier] * (sampled[identifier] - 0.18)
            elif self.method in ("comep", "additive"):
                value = eff_rel[identifier] * sampled[identifier]
            else:
                value = relevance[identifier] * sampled[identifier]

            if self.interactions:
                coalition_size = len(existing)
                for other in existing:
                    if self.method == "proposal":
                        gate = 1.0
                    else:
                        gate = math.sqrt(eff_rel[identifier] * eff_rel[other])
                    value += self.config.synergy * gate * pair_term(identifier, other, coalition_size)

            if self.method in ("mmr", "comep", "proposal", "additive", "evomemguard"):
                penalty = self.config.redundancy if self.config.retrieval_redundancy is None else self.config.retrieval_redundancy
                if existing and penalty > 0:
                    if self.method in ("comep", "additive"):
                        act_map = getattr(self, "action_vectors", {})
                        sims = []
                        for other in existing:
                            raw_sim = similarity(self.vectors[identifier], self.vectors[other])
                            if dense_encoder and identifier in act_map and other in act_map and raw_sim < 0.96:
                                adj_sim = min(raw_sim, similarity(act_map[identifier], act_map[other]))
                            else:
                                adj_sim = max(0.0, raw_sim - 0.5 * relevance[identifier] * relevance[other]) if raw_sim < 0.92 else raw_sim
                            if self.method == "comep":
                                pos_syn = max(0.0, self.pair_mean(identifier, other))
                                attenuation = math.exp(-2.0 * pos_syn) if raw_sim < 0.96 else 1.0
                                sims.append(adj_sim * attenuation)
                            else:
                                sims.append(adj_sim)
                        value -= penalty * max(sims, default=0.0)
                    else:
                        value -= penalty * max((similarity(self.vectors[identifier], self.vectors[other]) for other in existing), default=0.0)
            return value

        ctx_intercept = self.contexts[context].mean if context in self.contexts else 0.0

        def current_prob(subset):
            latent = ctx_intercept
            for ident in subset:
                latent += eff_rel[ident] * self.individual[ident].mean
            if self.method == "comep":
                for f_id, s_id in combinations(subset, 2):
                    latent += self.config.synergy * math.sqrt(eff_rel[f_id] * eff_rel[s_id]) * self.pair_mean(f_id, s_id)
            return sigmoid(latent)

        while candidates and len(selected) < self.config.max_items:
            singles = [(identifier,) for identifier in candidates if feasible((identifier,))]
            if not singles:
                break
            if explored:
                if self.method == "comep" and self.config.pair_random_explore and not selected:
                    raw_feasible_pairs = [
                        (f_id, s_id)
                        for f_id, s_id in combinations(candidates, 2)
                        if (
                            f_id in top_pair_pool
                            and s_id in top_pair_pool
                            and eff_rel[f_id] >= 0.35
                            and eff_rel[s_id] >= 0.35
                            and feasible((f_id, s_id))
                        )
                    ]
                    feasible_pairs = [(f_id, s_id) for f_id, s_id in raw_feasible_pairs if warmup_ready(f_id, s_id)]
                    if raw_feasible_pairs and not feasible_pairs:
                        self.trigger_counts["eval_warmup_gate_blocks"] += 1
                        ep_flags["warmup_block"] = True
                    if feasible_pairs:
                        p_weights = np.array([
                            max(1e-3, (eff_rel[f_id] * eff_rel[s_id]) ** 1.5) / ((1.0 + self.pair_count(f_id, s_id)) ** 1.5)
                            for f_id, s_id in feasible_pairs
                        ], dtype=float)
                        p_weights /= p_weights.sum()
                        bundle = feasible_pairs[int(self.rng.choice(len(feasible_pairs), p=p_weights))]
                        ep_flags["random_pair"] = True
                        selected.extend(bundle)
                        tokens += sum(self.memories[identifier].tokens for identifier in bundle)
                        candidates = [identifier for identifier in candidates if identifier not in bundle]
                        continue
                # Relevance-weighted exploration toward under-exposed items
                weights = np.array([
                    max(1e-3, eff_rel[b[0]]) / math.sqrt(1.0 + self.individual[b[0]].count)
                    for b in singles
                ], dtype=float)
                weights /= weights.sum()
                bundle = singles[int(self.rng.choice(len(singles), p=weights))]
            else:
                choices = [(gain(bundle[0], selected), bundle) for bundle in singles]
                pair_search = self.interactions or (self.method == "additive" and self.config.matched_pair_lookahead)
                if pair_search and self.config.pair_lookahead and self.method != "proposal":
                    lookahead_pool = (
                        candidates
                        if len(candidates) <= 10
                        else [b[0] for _, b in sorted(choices, key=lambda item: (-item[0], item[1][0]))[:10]]
                    )
                    for first, second in combinations(lookahead_pool, 2):
                        first_id, second_id = (first, second) if first < second else (second, first)
                        if feasible((first_id, second_id)):
                            g1 = gain(first_id, selected)
                            g2 = gain(second_id, selected)
                            forward = g1 + gain(second_id, selected + [first_id])
                            backward = g2 + gain(first_id, selected + [second_id])
                            ordered_pair = (first_id, second_id) if g1 >= g2 else (second_id, first_id)
                            choices.append((max(forward, backward), ordered_pair))
                best_gain, bundle = max(choices, key=lambda item: item[0])
                if self.method not in ("full-memory", "individual") and best_gain <= 0:
                    break
                stop_after_bundle = False
                if self.config.min_prob_gain_per_token > 0 and self.method in ("comep", "additive"):
                    if len(bundle) == 2 and not selected and all(self.individual[i].count >= 3 for i in bundle):
                        f_id, s_id = bundle
                        keep_second = (
                            (self.method == "comep" and self.pair_mean(f_id, s_id) > 0.0)
                            or self.individual[s_id].mean >= self.config.prior_mean
                        )
                        if not keep_second:
                            p_first = current_prob([f_id])
                            p_both = current_prob([f_id, s_id])
                            if (p_both - p_first) < self.config.min_prob_gain_per_token * self.memories[s_id].tokens and gain(f_id, []) > 0:
                                bundle = (f_id,)
                                stop_after_bundle = True
                                ep_flags["early_trim"] = True
                    elif len(selected) >= 2 or (selected and all(self.individual[i].count >= 3 for i in bundle)):
                        bundle_tokens = sum(self.memories[i].tokens for i in bundle)
                        prob_delta = current_prob(selected + list(bundle)) - current_prob(selected)
                        has_syn = self.method == "comep" and any(self.pair_mean(i, prev) > 0.10 for i in bundle for prev in selected)
                        tested_positive = all(
                            self.individual[i].count >= 2 and self.individual[i].mean > self.config.prior_mean
                            for i in bundle
                        )
                        high_rel_complement = (
                            self.method in ("comep", "additive")
                            and all(
                                i in top_pair_pool and eff_rel[i] >= 0.48 and self.individual[i].mean >= 0.0
                                for i in bundle
                            )
                        )
                        if len(selected) >= 2 and not (has_syn or tested_positive or high_rel_complement):
                            ep_flags["early_break"] = True
                            break
                        if not (has_syn or high_rel_complement) and prob_delta < self.config.min_prob_gain_per_token * bundle_tokens:
                            ep_flags["early_break"] = True
                            break
                if self.method == "comep":
                    if len(bundle) == 1:
                        b0 = bundle[0]
                        c_size = len(selected)
                        for other in selected:
                            _, b_kind = pair_term(b0, other, coalition_size=c_size, record_eval=False, return_bonus_kind=True)
                            if b_kind:
                                committed_bonus_kinds.add(b_kind)
                    elif len(bundle) == 2:
                        b0, b1 = bundle
                        fwd = gain(b0, selected) + gain(b1, selected + [b0])
                        bwd = gain(b1, selected) + gain(b0, selected + [b1])
                        first_add, second_add = (b0, b1) if fwd >= bwd else (b1, b0)
                        c_size0 = len(selected)
                        for other in selected:
                            _, b_kind = pair_term(first_add, other, coalition_size=c_size0, record_eval=False, return_bonus_kind=True)
                            if b_kind:
                                committed_bonus_kinds.add(b_kind)
                        ext_sel = selected + [first_add]
                        c_size1 = len(ext_sel)
                        for other in ext_sel:
                            _, b_kind = pair_term(second_add, other, coalition_size=c_size1, record_eval=False, return_bonus_kind=True)
                            if b_kind:
                                committed_bonus_kinds.add(b_kind)
            selected.extend(bundle)
            tokens += sum(self.memories[identifier].tokens for identifier in bundle)
            candidates = [identifier for identifier in candidates if identifier not in bundle]
            if not explored and stop_after_bundle:
                break

        if ep_flags["hard_conflict"]:
            self.trigger_counts["hard_conflict_blocked_episodes"] += 1
        if ep_flags["early_trim"]:
            self.trigger_counts["early_stop_trimmed_episodes"] += 1
        if ep_flags["early_break"]:
            self.trigger_counts["early_stop_loop_break_episodes"] += 1
        if ep_flags["warmup_block"]:
            self.trigger_counts["warmup_gate_blocked_episodes"] += 1
        if ep_flags["random_pair"]:
            self.trigger_counts["selected_pair_random_explore_episodes"] += 1

        if not explored and len(selected) >= 2:
            if "cold" in committed_bonus_kinds:
                self.trigger_counts["selected_pair_cold_ucb_episodes"] += 1
            if "second_chance" in committed_bonus_kinds:
                self.trigger_counts["selected_pair_second_chance_ucb_episodes"] += 1
            if "confirmation" in committed_bonus_kinds:
                self.trigger_counts["selected_pair_confirmation_episodes"] += 1

        return Selection(selected, relevance, context, tokens, explored)

    def predict(self, selection):
        baseline = self.contexts.get(selection.context)
        latent = baseline.mean if baseline else 0.0
        if self.v2_stack:
            eff_rel = self._effective_relevance(selection.relevance)
            for identifier in selection.identifiers:
                latent += eff_rel.get(identifier, 1.0) * self.individual[identifier].mean
            if self.method == "comep":
                for first, second in combinations(selection.identifiers, 2):
                    gate = math.sqrt(eff_rel.get(first, 1.0) * eff_rel.get(second, 1.0))
                    latent += self.config.synergy * gate * self.pair_mean(first, second)
            return sigmoid(latent)
        for identifier in selection.identifiers:
            latent += selection.relevance[identifier] * self.individual[identifier].mean
        if self.interactions:
            for first, second in combinations(selection.identifiers, 2):
                gate = 1.0 if self.method == "proposal" else selection.relevance[first] * selection.relevance[second]
                latent += self.config.synergy * gate * self.pair_mean(first, second)
        return sigmoid(latent)

    def _refit_laplace(self):
        if not self.history:
            return
        n_obs = len(self.history)
        ctx_list = sorted({row[0] for row in self.history})
        ctx_index = {c: idx for idx, c in enumerate(ctx_list)}

        # Retain all historical item columns present in window to prevent retroactive blame after pruning
        item_set = set()
        for _, ids, _, _ in self.history:
            item_set.update(ids)
        item_list = sorted(item_set)
        item_index = {ident: len(ctx_list) + idx for idx, ident in enumerate(item_list)}

        active_pairs = []
        if self.method == "comep":
            window_pair_counts = {}
            for _, ids, _, eff_gates in self.history:
                active_ids = [i for i in ids if i in self.memories and eff_gates.get(i, 1.0) >= 0.25]
                for f_id, s_id in combinations(sorted(active_ids), 2):
                    pair = (f_id, s_id)
                    window_pair_counts[pair] = window_pair_counts.get(pair, 0) + 1
            threshold = 1 if n_obs < 4 else self.config.min_active_pair_count
            eligible = [
                pair for pair, cnt in window_pair_counts.items()
                if cnt >= threshold or pair in self.pairs
            ]
            if len(eligible) > self.config.max_active_pairs:
                eligible.sort(key=lambda p: (-window_pair_counts[p], p))
                eligible = eligible[: self.config.max_active_pairs]
            active_pairs = sorted(eligible)

        pair_offset = len(ctx_list) + len(item_list)
        n_params = pair_offset + len(active_pairs)

        design = np.zeros((n_obs, n_params), dtype=float)
        rewards = np.zeros(n_obs, dtype=float)
        pair_lookup = {pair: pair_offset + idx for idx, pair in enumerate(active_pairs)}

        for row_idx, (ctx, ids, rew, eff_gates) in enumerate(self.history):
            rewards[row_idx] = float(rew)
            design[row_idx, ctx_index[ctx]] = 1.0
            for ident in ids:
                design[row_idx, item_index[ident]] = eff_gates.get(ident, 1.0)
            if active_pairs and len(ids) >= 2:
                for f_id, s_id in combinations(sorted(ids), 2):
                    col = pair_lookup.get((f_id, s_id))
                    if col is not None:
                        design[row_idx, col] = math.sqrt(eff_gates.get(f_id, 1.0) * eff_gates.get(s_id, 1.0))

        priors = np.empty(n_params, dtype=float)
        priors[: len(ctx_list)] = max(1.0, self.config.prior_precision)
        priors[len(ctx_list) : pair_offset] = self.config.prior_precision
        priors[pair_offset:] = self.config.pair_prior_precision

        prior_targets = np.zeros(n_params, dtype=float)
        prior_targets[len(ctx_list) : pair_offset] = self.config.prior_mean
        if len(ctx_list) > 1:
            for idx in range(len(ctx_list)):
                ctx_rows = design[:, idx] > 0
                if np.any(ctx_rows):
                    rate = float((np.sum(rewards[ctx_rows]) + 0.5) / (np.sum(ctx_rows) + 1.0))
                    mean_gate_sum = float(np.mean(np.sum(design[ctx_rows, len(ctx_list) : pair_offset], axis=1)))
                    eb_target = math.log(rate / (1.0 - rate)) - mean_gate_sum * self.config.prior_mean
                    prior_targets[idx] = 0.65 * eb_target

        l1_vec = np.zeros(n_params, dtype=float)
        if active_pairs and self.config.pair_l1 > 0:
            l1_vec[pair_offset:] = min(0.15, self.config.pair_l1 * 5.0)

        theta = np.zeros(n_params, dtype=float)
        for idx, ctx in enumerate(ctx_list):
            key = ("ctx", ctx)
            if key in self._warm_coefs:
                theta[idx] = self._warm_coefs[key]
            else:
                ctx_rows = design[:, idx] > 0
                rate = (np.sum(rewards[ctx_rows]) + 0.5) / (np.sum(ctx_rows) + 1.0)
                theta[idx] = math.log(rate / (1.0 - rate))
        for idx, ident in enumerate(item_list):
            key = ("item", ident)
            theta[len(ctx_list) + idx] = self._warm_coefs.get(key, self.config.prior_mean)
        for idx, pair in enumerate(active_pairs):
            key = ("pair", pair)
            if key in self._warm_coefs:
                theta[pair_offset + idx] = self._warm_coefs[key]

        def objective(vec):
            latent = np.clip(design @ vec, -35.0, 35.0)
            nll = float(np.sum(np.logaddexp(0.0, latent) - rewards * latent))
            diff = vec - prior_targets
            reg = 0.5 * float(np.sum(priors * (diff ** 2))) + float(np.sum(l1_vec * np.abs(vec)))
            return nll + reg

        max_iters = 5 if (self._warm_coefs and n_obs > 25) else 12
        for _ in range(max_iters):
            latent = np.clip(design @ theta, -35.0, 35.0)
            probs = _sigmoid_vec(latent)
            weights = np.maximum(probs * (1.0 - probs), 1e-6)
            grad_smooth = design.T @ (probs - rewards) + priors * (theta - prior_targets)
            # Smooth Huber-like subgradient for small L1 on pair coordinates
            subgrad = grad_smooth + l1_vec * np.tanh(theta * 20.0)
            if float(np.max(np.abs(subgrad))) < 1e-5:
                break
            hess = (design.T * weights) @ design
            hess[np.diag_indices_from(hess)] += priors + l1_vec * 20.0 * (1.0 - np.tanh(theta * 20.0) ** 2)
            try:
                step_dir = np.linalg.solve(hess, subgrad)
            except np.linalg.LinAlgError:
                step_dir = np.linalg.lstsq(hess, subgrad, rcond=None)[0]
            base_obj = objective(theta)
            alpha = 1.0
            while alpha >= 1.0 / 64.0:
                cand = theta - alpha * step_dir
                if active_pairs and self.config.pair_l1 > 0:
                    # Exact proximal soft-thresholding on very small pair values
                    small = np.abs(cand[pair_offset:]) < (l1_vec[pair_offset:] / (priors[pair_offset:] + np.diag(hess)[pair_offset:])) * 0.5
                    cand[pair_offset:][small] = 0.0
                if objective(cand) <= base_obj + 1e-9:
                    theta = cand
                    break
                alpha *= 0.5
            else:
                break

        latent = np.clip(design @ theta, -35.0, 35.0)
        probs = _sigmoid_vec(latent)
        weights = np.maximum(probs * (1.0 - probs), 1e-6)
        hess = (design.T * weights) @ design
        hess[np.diag_indices_from(hess)] += priors
        try:
            cov = np.linalg.inv(hess)
            post_prec = 1.0 / np.maximum(np.diag(cov), 1e-6)
        except np.linalg.LinAlgError:
            post_prec = np.diag(hess)

        self._warm_coefs.clear()
        for idx, ctx in enumerate(ctx_list):
            self._warm_coefs[("ctx", ctx)] = float(theta[idx])
            prev_cnt = self.contexts[ctx].count if ctx in self.contexts else 0
            self.contexts[ctx] = Parameter(mean=float(theta[idx]), precision=float(post_prec[idx]), count=prev_cnt)

        for idx, ident in enumerate(item_list):
            col = len(ctx_list) + idx
            self._warm_coefs[("item", ident)] = float(theta[col])
            if ident in self.individual:
                self.individual[ident].mean = float(theta[col])
                self.individual[ident].precision = float(post_prec[col])

        if self.method == "comep":
            new_pairs = {}
            for idx, pair in enumerate(active_pairs):
                col = pair_offset + idx
                val = float(theta[col])
                self._warm_coefs[("pair", pair)] = val
                cnt = self._pair_counts.get(pair, 0)
                if pair in self.pairs and cnt == 0:
                    cnt = self.pairs[pair].count
                new_pairs[pair] = Parameter(mean=val, precision=float(post_prec[col]), count=cnt)
            # Keep manually injected pairs that haven't been overwritten yet
            for pair, param in self.pairs.items():
                if pair not in new_pairs and pair[0] in self.memories and pair[1] in self.memories:
                    new_pairs[pair] = param
            self.pairs = new_pairs

    def observe(self, selection, reward):
        if reward not in (0, 1):
            raise ValueError("Bernoulli reward must be 0 or 1")
        if len(set(selection.identifiers)) != len(selection.identifiers) or any(identifier not in self.memories for identifier in selection.identifiers):
            raise ValueError("selection contains duplicate or inactive memories")
        prediction = self.predict(selection)
        self.last_prediction = prediction
        identifiers = selection.identifiers

        if len(identifiers) == 1:
            self._solo_counts[identifiers[0]] = self._solo_counts.get(identifiers[0], 0) + 1
        for identifier in identifiers:
            self.individual[identifier].count += 1
            if hasattr(self, "_item_contexts"):
                self._item_contexts.setdefault(identifier, set()).add(selection.context)
            if self.method == "evomemguard":
                self.individual[identifier].mean += float(reward)
        for first, second in combinations(identifiers, 2):
            key = self.key(first, second)
            self._pair_counts[key] = self._pair_counts.get(key, 0) + 1
            self._pair_wins[key] = self._pair_wins.get(key, 0) + int(reward)
            self._pair_contexts.setdefault(key, set()).add(selection.context)
            if key in self.pairs:
                self.pairs[key].count = max(self.pairs[key].count + 1, self._pair_counts[key])

        ctx_param = self.contexts.setdefault(selection.context, self.parameter())
        ctx_param.count += 1
        self._last_context = selection.context
        self.reward_sum += reward
        self.observations += 1

        if self.method == "evomemguard":
            return prediction

        if self.laplace_based:
            eff_gates = self._effective_relevance(selection.relevance)
            self.history.append((selection.context, tuple(identifiers), int(reward), {i: eff_gates.get(i, 1.0) for i in identifiers}))
            self._refit_laplace()
            return prediction

        residual = reward - prediction
        curvature = prediction * (1 - prediction)
        ctx_param.mean += self.config.learning_rate * (residual - self.config.regularization * ctx_param.mean)
        ctx_param.precision += curvature
        if self.method == "proposal" and identifiers:
            baseline = (self.reward_sum - reward) / (self.observations - 1) if self.observations > 1 else 0.5
            logs = np.array([-math.log1p(self.individual[identifier].count - 1) - self.individual[identifier].mean for identifier in identifiers])
            weights = np.exp(logs - logs.max())
            weights /= weights.sum()
            previous_means = {identifier: self.individual[identifier].mean for identifier in identifiers}
            for identifier, weight in zip(identifiers, weights):
                self.individual[identifier].mean += self.config.learning_rate * float(weight) * (reward - baseline)
            for first, second in combinations(identifiers, 2):
                parameter = self.pairs.setdefault(self.key(first, second), self.parameter())
                parameter.mean += self.config.pair_rate * (reward - sigmoid(previous_means[first] + previous_means[second]))
                parameter.precision = 1 + parameter.count
        elif self.model_based:
            eff_gates = self._effective_relevance(selection.relevance) if self.v2_stack else selection.relevance
            prior_target = self.config.prior_mean if (self.v2_stack and self.config.sampling_scale > 0) else 0.0
            for identifier in identifiers:
                parameter = self.individual[identifier]
                feature = eff_gates.get(identifier, 1.0)
                parameter.mean += self.config.learning_rate * (feature * residual - self.config.regularization * (parameter.mean - prior_target))
                parameter.precision += curvature * feature ** 2
            if self.interactions:
                for first, second in combinations(identifiers, 2):
                    default_pair_param = (
                        Parameter(mean=0.0, precision=self.config.pair_prior_precision)
                        if self.v2_stack
                        else self.parameter()
                    )
                    parameter = self.pairs.setdefault(self.key(first, second), default_pair_param)
                    if self.v2_stack:
                        feature = self.config.synergy * math.sqrt(eff_gates.get(first, 1.0) * eff_gates.get(second, 1.0))
                    else:
                        feature = self.config.synergy * selection.relevance[first] * selection.relevance[second]
                    updated = parameter.mean + self.config.pair_rate * feature * residual
                    parameter.mean = math.copysign(max(0.0, abs(updated) - self.config.pair_rate * self.config.pair_l1), updated)
                    parameter.precision += curvature * feature ** 2
        elif self.method == "individual":
            for identifier in identifiers:
                self.individual[identifier].mean += 1 if reward else -1
        return prediction

    def _compute_pruning_score(self, identifier, use_coalition_pruning: bool, record_triggers: bool = False):
        parameter = self.individual[identifier]
        if self.method == "individual":
            return parameter.mean
        if self.method == "evomemguard":
            return self._evomemguard_utility(identifier) / self.memories[identifier].tokens
        if self.method == "mmr":
            return -max((similarity(self.vectors[identifier], vector) for other, vector in self.vectors.items() if other != identifier), default=0.0)
        ins_at = self.inserted_at_observation.get(identifier, 0)
        blame_identified = True
        if self.method == "comep" and use_coalition_pruning and 0 < parameter.count <= 2:
            solo_cnt = self._solo_counts.get(identifier, 0)
            active_co_pairs = [p for p, c in self._pair_counts.items() if c > 0 and identifier in p]
            partner_separated = any(
                self._solo_counts.get(other, 0) >= 1
                for p in active_co_pairs
                for other in p
                if other != identifier
            )
            blame_identified = (solo_cnt >= 1) or (parameter.count >= 2 and len(active_co_pairs) >= 2) or partner_separated

        if self.method == "proposal":
            neighbors = [value.mean for pair, value in self.pairs.items() if identifier in pair]
            pair_value = sum(neighbors) / len(neighbors) if neighbors else 0.0
        elif self.method == "comep" and use_coalition_pruning:
            # Coalition-aware retention: protect strictly marginal-positive partnerships, penalize conflict liability once identified
            pos_partners = []
            for (first, second), value in self.pairs.items():
                if identifier in (first, second) and value.mean > 0:
                    other = second if first == identifier else first
                    cnt = self.pair_count(first, second)
                    shrinkage_ratio = (self.config.pair_prior_precision + cnt) / max(self.config.prior_precision + cnt, 1e-6)
                    eff_pair = shrinkage_ratio * value.mean
                    if (
                        other in self.individual
                        and (parameter.mean + eff_pair) > 0.05
                        and (self.individual[other].mean + eff_pair) > 0.05
                    ):
                        pos_partners.append(value.mean)
            neg_conflicts = []
            if blame_identified:
                vulnerability = sigmoid(-(parameter.mean - 0.35) * 3.0)
                for (first, second), value in self.pairs.items():
                    if identifier in (first, second) and value.mean < 0:
                        other = second if first == identifier else first
                        if other in self.individual and self.individual[other].mean >= parameter.mean - 0.1:
                            neg_conflicts.append(vulnerability * value.mean)
            if record_triggers:
                if pos_partners:
                    self.trigger_counts["eval_coalition_prune_pos_items"] += 1
                if neg_conflicts:
                    self.trigger_counts["eval_coalition_prune_neg_items"] += 1
                if not blame_identified and parameter.mean < 0.0:
                    self.trigger_counts["eval_coalition_prune_blame_guard_items"] += 1
            pair_value = self.config.synergy * (max(pos_partners, default=0.0) + 0.5 * sum(neg_conflicts))
        else:
            pair_value = 0.0

        def dominates(other: str) -> bool:
            if other == identifier or self.pair_mean(identifier, other) > 0.1:
                return False
            other_param = self.individual[other]
            if self.v2_stack and parameter.count == 0:
                if ins_at > 0:
                    other_ins = self.inserted_at_observation.get(other, 0)
                    return (other_param.count > 0 and other_param.mean >= 0.0) or (
                        other_param.count == 0 and (other_ins, other) < (ins_at, identifier)
                    )
                return other_param.count >= 2 and other_param.mean > self.config.prior_mean + 0.10
            return other_param.mean > parameter.mean

        raw_redundancy = max(
            (similarity(self.vectors[identifier], vector) for other, vector in self.vectors.items() if dominates(other)),
            default=0.0,
        )
        redundancy = raw_redundancy if (not self.v2_stack or raw_redundancy >= 0.72) else 0.25 * raw_redundancy
        if not self.v2_stack:
            effective_mean = parameter.mean
        elif parameter.count == 0:
            effective_mean = 0.0
        elif not blame_identified and parameter.mean < 0.0:
            effective_mean = 0.0
        else:
            effective_mean = parameter.mean

        if self.method == "proposal":
            uncertainty = 0.0
        elif self.v2_stack and parameter.count == 0:
            uncertainty = 0.55 * self.config.pruning_uncertainty
        else:
            uncertainty = self.config.pruning_uncertainty / math.sqrt(max(parameter.precision, 1e-6))
        penalty = self.config.redundancy if self.config.pruning_redundancy is None else self.config.pruning_redundancy
        effective_tokens = max(16, self.memories[identifier].tokens) if self.v2_stack else self.memories[identifier].tokens
        return (effective_mean + pair_value - penalty * redundancy + uncertainty) / effective_tokens

    def pruning_score(self, identifier):
        return self._compute_pruning_score(identifier, self.config.coalition_pruning, record_triggers=False)

    def _evict_single(self, victim: str, reason: str = "budget_prune"):
        score = self.pruning_score(victim) if victim in self.memories else None
        self.archive.append({
            "memory": asdict(self.memories[victim]),
            "parameter": asdict(self.individual[victim]),
            "score": score,
            "reason": reason,
            "observed_at": self.observations,
        })
        del self.memories[victim]
        del self.individual[victim]
        del self.vectors[victim]
        getattr(self, "scope_vectors", {}).pop(victim, None)
        getattr(self, "action_vectors", {}).pop(victim, None)
        getattr(self, "_born_context", {}).pop(victim, None)
        del self.inserted_at_observation[victim]
        self._solo_counts.pop(victim, None)
        self.pairs = {pair: parameter for pair, parameter in self.pairs.items() if victim not in pair}
        if self.method == "evomemguard":
            self.triples.pop(victim, None)
            self.nli_conflicts = {pair for pair in self.nli_conflicts if victim not in pair}

    def prune(self):
        removed = []
        if self.method in ("no-memory", "full-memory"):
            return removed
        while self.bank_tokens > self.config.bank_tokens or len(self.memories) > self.config.max_bank_items:
            oversized = [identifier for identifier, memory in self.memories.items() if memory.tokens > self.config.bank_tokens]
            mature = [identifier for identifier in self.memories if self.observations - self.inserted_at_observation[identifier] >= self.config.admission_grace_episodes]
            if oversized and self.config.admission_grace_episodes:
                worst = min(oversized)
            elif mature:
                worst = min(
                    mature,
                    key=lambda identifier: (
                        self._compute_pruning_score(identifier, self.config.coalition_pruning, record_triggers=True),
                        self.memories[identifier].created_at,
                        identifier,
                    ),
                )
                if self.method == "comep" and self.config.coalition_pruning:
                    alt_worst = min(
                        mature,
                        key=lambda identifier: (
                            self._compute_pruning_score(identifier, False, record_triggers=False),
                            self.memories[identifier].created_at,
                            identifier,
                        ),
                    )
                    if alt_worst != worst:
                        self.trigger_counts["coalition_prune_decision_changes"] += 1
            else:
                worst = min(self.memories, key=lambda identifier: (self.inserted_at_observation[identifier], self.memories[identifier].created_at, identifier))
            self.trigger_counts["prune_evictions"] += 1
            self._evict_single(worst, reason="budget_prune")
            removed.append(worst)
        return removed

    def snapshot(self):
        snap = {
            "method": self.method,
            "config": asdict(self.config),
            "token_counter": self.counter.label,
            "encoder": self.encoder.label,
            "observations": self.observations,
            "bank_tokens": self.bank_tokens,
            "memories": [asdict(memory) | {"parameter": asdict(self.individual[identifier])} for identifier, memory in self.memories.items()],
            "pairs": [{"identifiers": list(pair), **asdict(parameter), "conflict": self.conflict(*pair)} for pair, parameter in sorted(self.pairs.items())],
            "contexts": {key: asdict(value) for key, value in self.contexts.items()},
            "archive": self.archive,
            "trigger_counts": dict(self.trigger_counts),
            "manager_llm_calls": 0,
        }
        if self.method == "evomemguard":
            snap["nli_encoder"] = self.nli.label
            snap["nli_conflicts"] = [list(pair) for pair in sorted(self.nli_conflicts)]
            snap["guard_events_count"] = len(self.guard_events)
        return snap
