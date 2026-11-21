from dataclasses import dataclass
import math
import re

import numpy as np

from .text import similarity


NEGATION_MARKERS = frozenset({"not", "never", "avoid", "without", "do_not", "dont", "forbid", "prohibit", "skip", "ignore"})
ANTONYM_PAIRS = (
    frozenset({"heat", "cool"}),
    frozenset({"hot", "cold"}),
    frozenset({"open", "close"}),
    frozenset({"closed", "opened"}),
    frozenset({"empty", "holding"}),
    frozenset({"raw", "cooked"}),
    frozenset({"microwave", "fridge"}),
    frozenset({"first", "last"}),
    frozenset({"always", "never"}),
)


@dataclass(frozen=True)
class ExperienceTriple:
    identifier: str
    context_clause: str
    action_clause: str
    outcome_polarity: str
    raw_text: str
    tokens: int
    created_at: int = 0


def parse_experience_triple(identifier: str, text: str, tokens: int, created_at: int = 0) -> ExperienceTriple:
    cleaned = text.strip()
    lower = cleaned.lower()
    context_clause = "general"
    action_clause = cleaned
    for splitter in (" then ", ", ", " before ", " when "):
        if splitter in lower:
            idx = lower.find(splitter)
            if splitter in (" when ", " before "):
                action_clause = cleaned[:idx].strip(" ,.")
                context_clause = cleaned[idx + len(splitter):].strip(" ,.")
            else:
                context_clause = cleaned[:idx].strip(" ,.")
                action_clause = cleaned[idx + len(splitter):].strip(" ,.")
            break
    words = set(re.findall(r"\w+", lower))
    polarity = "negative_constraint" if (words & NEGATION_MARKERS) else "positive_guidance"
    return ExperienceTriple(
        identifier=identifier,
        context_clause=context_clause or "general",
        action_clause=action_clause or cleaned,
        outcome_polarity=polarity,
        raw_text=cleaned,
        tokens=tokens,
        created_at=created_at,
    )


class LocalNLIDetector:
    """Bidirectional policy-compatibility NLI detector for EvoMemGuard."""

    _MODEL_CACHE = {}
    _PAIR_CACHE: dict[tuple[str | None, str, str], tuple[float, float]] = {}

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.label_map = {"contradiction": 0, "entailment": 1, "neutral": 2}
        if model_path:
            if model_path not in self._MODEL_CACHE:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
                tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                mod = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
                mod.eval()
                id2label = {int(k): str(v).lower() for k, v in mod.config.id2label.items()}
                mapping = {}
                for idx, label in id2label.items():
                    if "contradict" in label:
                        mapping["contradiction"] = idx
                    elif "entail" in label:
                        mapping["entailment"] = idx
                    else:
                        mapping["neutral"] = idx
                self._MODEL_CACHE[model_path] = (tok, mod, mapping)
            self.tokenizer, self.model, self.label_map = self._MODEL_CACHE[model_path]

    @property
    def label(self) -> str:
        return f"nli_transformer:{self.model_path}" if self.model is not None else "lexical_nli_fallback"

    def score_pairs_batch(self, pairs: list[tuple[str, str]]) -> list[tuple[float, float]]:
        """Batch-evaluate (policy_contradiction_score, bidirectional_entailment_score) with caching."""
        if not pairs:
            return []
        missing_keys: list[tuple[str | None, str, str]] = []
        seen_missing = set()
        ordered_keys = []
        for text_a, text_b in pairs:
            first, second = (text_a, text_b) if text_a <= text_b else (text_b, text_a)
            ck = (self.model_path, first, second)
            ordered_keys.append(ck)
            if ck not in self._PAIR_CACHE and ck not in seen_missing:
                missing_keys.append(ck)
                seen_missing.add(ck)

        if missing_keys:
            if self.model is not None:
                import torch

                c_idx = self.label_map["contradiction"]
                e_idx = self.label_map["entailment"]
                chunk_size = 16
                for start in range(0, len(missing_keys), chunk_size):
                    chunk = missing_keys[start : start + chunk_size]
                    premises = []
                    hypotheses = []
                    for _, first, second in chunk:
                        pol_a = f'An agent manual includes the guideline: "{first}"'
                        pol_b = f'The same agent manual also includes the guideline: "{second}"'
                        premises.extend([pol_a, pol_b, first, second])
                        hypotheses.extend([pol_b, pol_a, second, first])
                    inputs = self.tokenizer(premises, hypotheses, padding=True, truncation=True, max_length=192, return_tensors="pt")
                    with torch.no_grad():
                        probs = torch.softmax(self.model(**inputs).logits, dim=-1).cpu().numpy()
                    for i, ck in enumerate(chunk):
                        base = 4 * i
                        contradiction = float(math.sqrt(max(0.0, float(probs[base, c_idx]) * float(probs[base + 1, c_idx]))))
                        entailment = float(min(probs[base + 2, e_idx], probs[base + 3, e_idx]))
                        self._PAIR_CACHE[ck] = (contradiction, entailment)
            else:
                for ck in missing_keys:
                    _, first, second = ck
                    words_a = set(re.findall(r"\w+", first.lower()))
                    words_b = set(re.findall(r"\w+", second.lower()))
                    union = words_a | words_b
                    jaccard = len(words_a & words_b) / len(union) if union else 0.0
                    neg_a = bool(words_a & NEGATION_MARKERS)
                    neg_b = bool(words_a & NEGATION_MARKERS)
                    has_antonym = any(
                        len(pair & union) == 2 and bool(pair & words_a) and bool(pair & words_b) and (pair & words_a) != (pair & words_b)
                        for pair in ANTONYM_PAIRS
                    )
                    contradiction = float(np.clip(0.85 if jaccard >= 0.22 and (neg_a != neg_b or has_antonym) else 0.04, 0.0, 1.0))
                    entailment = float(np.clip(jaccard if contradiction < 0.5 else 0.0, 0.0, 1.0))
                    self._PAIR_CACHE[ck] = (contradiction, entailment)

        return [self._PAIR_CACHE[ck] for ck in ordered_keys]

    def score_pair(self, text_a: str, text_b: str) -> tuple[float, float]:
        return self.score_pairs_batch([(text_a, text_b)])[0]


class EvoMemGuardMixin:
    """Architecture-grounded adaptation of EvoMemGuard (Wang et al., 2026, DOI: 10.13140/RG.2.2.33763.72483):
    1. Experience Triple structuring (context / action / outcome polarity)
    2. Memory Conflict Detector (bidirectional NLI policy-contradiction guard on admission & retrieval)
    3. Memory Utility Scorer (Laplace-smoothed empirical utility filtering)
    4. Memory Compressor (entailment/duplicate condensation + utility-per-token budget eviction)
    """

    def _init_evomemguard(self, nli_model: str | None = None, nli_conflict_threshold: float = 0.55, nli_entail_threshold: float = 0.82):
        self.nli = LocalNLIDetector(nli_model)
        self.nli_conflict_threshold = nli_conflict_threshold
        self.nli_entail_threshold = nli_entail_threshold
        self.triples: dict[str, ExperienceTriple] = {}
        self.nli_conflicts: set[tuple[str, str]] = set()
        self.guard_events: list[dict] = []

    def _evomemguard_utility(self, identifier: str) -> float:
        param = self.individual[identifier]
        successes = param.mean
        trials = param.count
        return (successes + 1.5) / (trials + 2.5)

    def _evomemguard_admit(self, identifier: str, text: str, step: int, prune: bool) -> list[str]:
        tokens = self.counter.count(text)
        triple = parse_experience_triple(identifier, text, tokens, step)
        new_vec = self.encoder.encode(text)
        removed: list[str] = []

        min_sim = 0.52 if getattr(self.encoder, "model", None) is not None else 0.25
        scored_incumbents = [
            (similarity(new_vec, self.vectors[other_id]), other_id)
            for other_id in self.memories
        ]
        related = [
            (rel, other_id)
            for rel, other_id in sorted(scored_incumbents, key=lambda x: (-x[0], x[1]))[:6]
            if rel >= min_sim
        ]
        nli_scores = self.nli.score_pairs_batch([(text, self.memories[other_id].text) for _, other_id in related])

        for (rel, other_id), (contradiction, entailment) in zip(related, nli_scores):
            if other_id not in self.memories:
                continue
            # Module 3: Memory Compressor — entailment / near-duplicate condensation
            if entailment >= self.nli_entail_threshold or rel >= 0.92:
                incumbent_u = self._evomemguard_utility(other_id)
                if incumbent_u >= 0.5 and self.memories[other_id].tokens <= tokens:
                    self.guard_events.append({"event": "compressed_rejected_duplicate", "candidate": identifier, "incumbent": other_id, "entailment": entailment})
                    return removed
                self._evict_single(other_id, reason="evomemguard_compressor")
                removed.append(other_id)
                continue
            # Module 1: Memory Conflict Detector — NLI policy contradiction check
            if contradiction >= self.nli_conflict_threshold:
                incumbent_u = self._evomemguard_utility(other_id)
                incumbent_n = self.individual[other_id].count
                if incumbent_n >= 2 and incumbent_u >= 0.55:
                    self.guard_events.append({"event": "conflict_blocked_newcomer", "candidate": identifier, "incumbent": other_id, "contradiction": contradiction})
                    return removed
                if incumbent_n >= 2 and incumbent_u < 0.42:
                    self._evict_single(other_id, reason="evomemguard_conflict_prune")
                    removed.append(other_id)
                else:
                    self.nli_conflicts.add(self.key(identifier, other_id))
                    self.guard_events.append({"event": "conflict_edge_registered", "pair": list(self.key(identifier, other_id)), "contradiction": contradiction})

        from .core import Memory, Parameter

        self.memories[identifier] = Memory(identifier, text, tokens, step)
        self.triples[identifier] = triple
        self.vectors[identifier] = new_vec
        self.individual[identifier] = Parameter(mean=0.0, precision=2.0, count=0)
        self.inserted_at_observation[identifier] = self.observations
        if prune:
            removed.extend(self.prune())
        return removed
