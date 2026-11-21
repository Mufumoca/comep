import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from comep.core import Config, MemoryManager, Parameter


class FixtureEncoder:
    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, text):
        return self.vectors[text]


class FixtureCounter:
    def __init__(self, costs):
        self.costs = costs

    def count(self, text):
        return self.costs[text]


def manager(method, config, costs, means, vectors):
    instance = MemoryManager(method, config, seed=71, encoder=FixtureEncoder(vectors), counter=FixtureCounter(costs))
    for identifier in sorted(costs):
        instance.add(identifier, identifier)
        instance.individual[identifier].mean = means[identifier]
    return instance


def run():
    config = Config(retrieval_tokens=2, max_items=2, sampling_scale=0, exploration=0, redundancy=0)
    costs = {'A': 2, 'B': 1, 'C': 1}
    means = {'A': 6, 'B': 4, 'C': 4}
    vectors = {identifier: np.array([1.0]) for identifier in [*costs, 'query']}
    legacy = manager('additive', config, costs, means, vectors).choose('query')
    joint = manager('comep', config, costs, means, vectors).choose('query')
    matched_config = replace(config, matched_pair_lookahead=True)
    matched = manager('additive', matched_config, costs, means, vectors).choose('query')
    assert legacy.identifiers == ['A']
    assert joint.identifiers == matched.identifiers == ['B', 'C']
    disabled = manager('additive', replace(matched_config, pair_lookahead=False), costs, means, vectors).choose('query')
    assert disabled.identifiers == legacy.identifiers
    tight = replace(matched_config, max_items=1)
    assert manager('additive', tight, costs, means, vectors).choose('query').identifiers == ['A']
    random = np.random.default_rng(20261104)
    for case in range(100):
        identifiers = [f'rule-{index}' for index in range(int(random.integers(2, 9)))]
        costs_case = {identifier: int(random.integers(1, 6)) for identifier in identifiers}
        means_case = {identifier: float(random.normal()) for identifier in identifiers}
        vectors_case = {}
        for identifier in [*identifiers, 'query']:
            vector = random.normal(size=8)
            vectors_case[identifier] = vector / np.linalg.norm(vector)
        settings = replace(matched_config, retrieval_tokens=int(random.integers(1, 16)),
            max_items=int(random.integers(1, 6)), redundancy=float(random.random()))
        additive = manager('additive', settings, costs_case, means_case, vectors_case).choose('query')
        comep = manager('comep', settings, costs_case, means_case, vectors_case).choose('query')
        assert additive.identifiers == comep.identifiers, case
        assert additive.tokens == comep.tokens <= settings.retrieval_tokens
        assert len(additive.identifiers) <= settings.max_items
    complementary_costs = {'A': 1, 'B': 1}
    complementary_means = {'A': -.1, 'B': -.1}
    additive = manager('additive', matched_config, complementary_costs, complementary_means, vectors)
    comep = manager('comep', matched_config, complementary_costs, complementary_means, vectors)
    comep.pairs[('A', 'B')] = Parameter(mean=1)
    assert additive.choose('query').identifiers == []
    assert comep.choose('query').identifiers == ['A', 'B']
    for invalid in (0, 1, None, 'false'):
        try:
            replace(config, matched_pair_lookahead=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('nonboolean option accepted')
    return {
        'kind': 'deterministic_search_contract_not_environment_efficacy',
        'witness': {'costs': costs, 'assigned_main_values': means, 'legacy_additive': legacy.identifiers,
            'joint_zero_interactions': joint.identifiers, 'matched_additive': matched.identifiers},
        'zero_interaction_parity_fixtures': 100,
        'positive_complementarity_fixture_passed': True,
        'budget_and_disabled_option_checks_passed': True,
        'claim_supported': {'C1': False, 'C2': False},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as handle:
            json.dump(result, handle, indent=2)
            handle.write('\n')
    print('ROUTING_PARITY_WITNESS legacy_difference,matched100,budget,disabled,complementarity; noAPI')


if __name__ == '__main__':
    main()
