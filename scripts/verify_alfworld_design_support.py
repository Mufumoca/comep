from itertools import combinations
import json
from pathlib import Path

from audit_alfworld_design_support import audit


def main():
    result = audit()
    assert result['total_lexical_tokens'] == 164
    assert result['native_scope_pair_count'] == 0
    assert result['native_scope_max_candidates'] == 1
    assert result['all_pair_count'] == 10
    assert result['dry_run_pruned_ids'] == []
    assert result['ungated_feasible_subset_count'] == 1 + 5 + 10 + 10
    assert max(len(subset) for subset in result['ungated_feasible_subsets']) == 3
    assert not result['ungated_full_bank_retrievable']
    assert all(not row['online_admission_permitted'] for row in result['candidates'])
    feasible = result['ungated_feasible_subsets']
    identifiers = [row['id'] for row in result['candidates']]
    assert all(list(pair) in feasible for pair in combinations(identifiers, 2))
    assert result['historical_exposed_candidate_pairs'] == 0
    assert not result['live_admission'] and not result['paid_followon_authorized']
    saved = Path('outputs/alfworld-design-support-20261104/audit.json')
    if saved.exists():
        assert json.loads(saved.read_text()) == result
    print('ALFWORLD_DESIGN_SUPPORT_WITNESS native0pairs,ungated10pairs,164tokens,no_bank_pressure,26feasible; noAPI')


if __name__ == '__main__':
    main()
