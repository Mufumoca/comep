from itertools import product
import math

from audit_online_information import audit, bigram_jaccard, impossible_conflict_bound, lexical_cosine, words


def main():
    assert words('Heat, the EGG!') == words('heat the egg')
    assert words('heat egg') != words('cool egg')
    assert math.isclose(lexical_cosine('heat egg', 'egg heat'), 1)
    assert bigram_jaccard('heat egg', 'egg heat') == 0
    assert lexical_cosine('', 'egg') == bigram_jaccard('', '') == 0
    for probabilities in product((.01, .5, .99), repeat=4):
        for rewards in product((0, 1), repeat=4):
            mean, precision = 0.0, 1.0
            for count, (probability, reward) in enumerate(zip(probabilities, rewards), 1):
                feature = (count % 3 + 1) / 3
                value = mean + .1 * feature * (reward - probability)
                mean = math.copysign(max(0, abs(value) - .001), value)
                precision += probability * (1 - probability) * feature ** 2
                bound = impossible_conflict_bound(count, 1, .1, .01, 1, 1.96)
                assert mean + 1.96 / math.sqrt(precision) >= bound - 1e-12
    assert all(impossible_conflict_bound(count, 1, .1, .01, 1, 1.96) >= -.2 for count in range(12))
    assert impossible_conflict_bound(12, 1, .1, .01, 1, 1.96) < -.2
    result = audit('outputs/alfworld-online-live-20261104')
    assert [row['accepted'] for row in result['arms']] == [27, 26, 28]
    assert all(row['normalized_exact_repeated_arrivals'] == 0 for row in result['arms'])
    assert [row['distinct_exposed_pairs'] for row in result['arms']] == [6, 23, 29]
    for arm in result['arms']:
        lifetimes = [row for row in result['lifetimes'] if row['method'] == arm['method']]
        assert all(row['future_trial_opportunities'] >= row['selected_episodes'] for row in lifetimes)
        assert arm['never_selected_all_arrivals'] == arm['never_selected_with_future_opportunity'] + arm['zero_future_opportunity_arrivals']
    info = result['information']
    assert info['pair_occurrences'] == info['zero_start_l1_survival_occurrences'] == 46
    assert math.isclose(info['total_diagonal_curvature_increment'], .2410170780328912, abs_tol=1e-12)
    assert info['earliest_count_not_excluded_by_optimistic_bound'] == 12
    assert all(row['post_precision'] >= row['prior_precision'] and row['post_upper'] > 0 for row in result['pair_updates'])
    print('ONLINE_INFORMATION_VERIFY: bounded-update fixtures, lexical counterexamples,90cell provenance and46factual pair updates; no API')


if __name__ == '__main__':
    main()
