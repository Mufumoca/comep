from copy import deepcopy
import json
from pathlib import Path

from comep.alfworld_rules import parse_goal, source_recipe
from audit_alfworld_rules import audit


def source(trial):
    root = Path('outputs/alfworld-pilot-live-20261104') / f'trial-{trial:03d}'
    return json.loads((root / 'episodes.jsonl').read_text()), [json.loads(line) for line in (root / 'transitions.jsonl').read_text().splitlines()]


def rejected(record, trajectory):
    assert not source_recipe(record, trajectory)['accepted']


def continuous(trajectory):
    for previous, current in zip(trajectory, trajectory[1:]):
        current['observation'] = previous['next_observation']
    return trajectory


def main():
    result = audit(Path('outputs/alfworld-pilot-live-20261104'))
    assert result['accepted'] == 5 and result['total'] == 6
    wrong = result['candidates'][2]
    assert wrong['reason'] == 'unsuccessful_source_with_off_target_actions'
    assert wrong['off_target_event_steps'] == [20, 22, 24]
    assert not any(row['usable_as_online_memory'] or row['semantic_generalization_verified'] for row in result['candidates'])
    record, trajectory = source(3)
    assert source_recipe(record, trajectory)['accepted']
    for step, action, feedback in (
        (15, 'heat potato 4 with microwave 1', 'You heat the potato 4 using the microwave 1.'),
        (15, 'heat potato 3 with fridge 1', 'You heat the potato 3 using the fridge 1.'),
        (17, 'move potato 3 to bed 1', 'You move the potato 3 to the bed 1.'),
        (12, 'take potato 4 from countertop 1', 'You pick up the potato 4 from the countertop 1.'),
    ):
        changed = deepcopy(trajectory)
        changed[step].update(action=action, next_observation=feedback)
        rejected(record, changed)
    changed = deepcopy(trajectory)
    changed[15]['next_observation'] = 'Nothing happens.'
    rejected(record, changed)
    changed = deepcopy(trajectory)
    changed[12]['action'], changed[15]['action'] = changed[15]['action'], changed[12]['action']
    changed[12]['next_observation'], changed[15]['next_observation'] = changed[15]['next_observation'], changed[12]['next_observation']
    rejected(record, changed)
    changed = deepcopy(trajectory)
    changed[0]['observation'] = 'Your task is to: do something unsupported.'
    rejected(record, changed)
    rejected({**record, 'reward': 0}, trajectory)
    changed = deepcopy(trajectory)
    changed[-1]['success'] = False
    rejected({**record, 'reward': 0}, changed)
    record, trajectory = source(5)
    changed = deepcopy(trajectory)
    for row in changed:
        row['action'] = row['action'].replace('book 2', 'book 1')
        row['next_observation'] = row['next_observation'].replace('book 2', 'book 1')
    rejected(record, changed)
    record, trajectory = source(1)
    changed = deepcopy(trajectory)
    changed[14].update(action='use floorlamp 1', next_observation='You turn on the floorlamp 1.')
    rejected(record, changed)
    assert parse_goal('Your task is to: put two hot egg in bowl.') is None
    record, trajectory = source(0)
    for bad in ('false', None, 1):
        for field in ('done', 'success'):
            changed = deepcopy(trajectory)
            changed[-1][field] = bad
            rejected(record, changed)
    for field, bad in (('score', 0), ('truncated', True), ('termination', 'environment_terminal')):
        rejected({**record, field: bad}, trajectory)
    for suffix in ('\nYour task is to: put a clean egg in countertop.', '\nThen clean an egg.'):
        changed = deepcopy(trajectory)
        changed[0]['observation'] += suffix
        rejected(record, changed)
    changed = deepcopy(trajectory)
    del changed[1]['action']
    rejected(record, changed)
    changed = deepcopy(trajectory)
    changed[1]['next_observation'] = None
    rejected(record, changed)
    record, trajectory = source(5)
    changed = deepcopy(trajectory)
    changed[7].update(action='take book 1 from bed 1', next_observation='You pick up the book 1 from the bed 1.')
    rejected(record, changed)
    record, trajectory = source(3)
    changed = deepcopy(trajectory)
    changed[16].update(action='cool potato 3 with fridge 1', next_observation='You cool the potato 3 using the fridge 1.')
    rejected(record, changed)
    record, trajectory = source(1)
    changed = deepcopy(trajectory)
    changed[13].update(action='put tissuebox 1 in/on sidetable 1', next_observation='You put the tissuebox 1 in/on the sidetable 1.')
    rejected(record, changed)
    record, trajectory = source(3)
    for step, action, feedback, reason in (
        (14, 'move potato 4 to countertop 1', 'You move the potato 4 to the countertop 1.', 'move_without_supported_possession'),
        (16, 'heat potato 3 with fridge 1', 'You heat the potato 3 using the fridge 1.', 'unsupported_transform_tool'),
        (16, 'cool potato 3 with fridge 1', 'You cool the potato 3 using the fridge 1.', 'unsupported_intervening_transform'),
    ):
        changed = deepcopy(trajectory)
        changed[step].update(action=action, next_observation=feedback)
        assert source_recipe(record, continuous(changed))['reason'] == reason
    changed = deepcopy(trajectory)
    changed[13]['observation'] = 'inconsistent feedback'
    assert source_recipe(record, changed)['reason'] == 'observation_continuity_mismatch'
    record, trajectory = source(5)
    changed = deepcopy(trajectory)
    changed[7].update(action='take book 1 from bed 1', next_observation='You pick up the book 1 from the bed 1.')
    assert source_recipe(record, continuous(changed))['reason'] == 'contradictory_successful_possession'
    saved = Path('outputs/alfworld-rule-grounding-20261104-v3/audit.json')
    if saved.exists():
        assert json.loads(saved.read_text()) == result
    print('ALFWORLD_RULE_WITNESS sixsource,wrongobject,reorder,quote,tool,target,distinctinstances,unsupported; noAPI')


if __name__ == '__main__':
    main()
