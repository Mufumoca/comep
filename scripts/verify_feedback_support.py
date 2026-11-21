from copy import deepcopy
import json
from pathlib import Path

from audit_feedback_support import audit, summarize_episode


def main():
    record = {'episode': 0, 'task_id': 'fixture', 'steps': 3, 'reward': 0, 'score': -100, 'selected': ['a', 'b']}
    transitions = [
        {'step': 0, 'task_id': 'fixture', 'score': 10},
        {'step': 1, 'task_id': 'fixture', 'score': 20},
        {'step': 2, 'task_id': 'fixture', 'previous_score': 20, 'score': -100},
    ]
    result = summarize_episode(record, transitions)
    assert result['positive_score_steps'] == 1
    assert result['negative_score_steps'] == 1
    assert result['unknown_initial_previous_score'] == 1
    assert result['inferred_previous_from_adjacent_step'] == 1
    assert result['progress_before_failure']
    assert not result['rule_level_credit_identified']
    explicit = deepcopy(transitions)
    explicit[0]['previous_score'] = 0
    assert summarize_episode(record, explicit)['positive_score_steps'] == 2
    broken = deepcopy(transitions)
    broken[2]['previous_score'] = 19
    try:
        summarize_episode(record, broken)
    except ValueError:
        pass
    else:
        raise AssertionError('score discontinuity accepted')
    result = audit()
    assert len(result['episodes']) == 185
    assert len(result['campaigns']) == 11
    assert sum(row['evidence_type'] == 'learned-policy' for row in result['episodes']) == 18
    assert all(not row['rule_level_credit_identified'] for row in result['episodes'])
    assert sum(row['incomplete_episode_transition_rows_excluded'] for row in result['campaigns']) > 0
    saved = Path('outputs/feedback-support-20261104-v2/audit.json')
    if saved.exists():
        assert json.loads(saved.read_text()) == result
    print('FEEDBACK_SUPPORT_WITNESS missing-initial-not-zero,adjacent-reconstruction,continuity,185completed,18learned,no-rule-credit')


if __name__ == '__main__':
    main()
