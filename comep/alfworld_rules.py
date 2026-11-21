import re


def parse_goal(observation):
    if not isinstance(observation, str) or observation.count('Your task is to:') != 1:
        return None
    match = re.search(r'Your task is to: ([^\n]+)\s*\Z', observation)
    if not match:
        return None
    goal = match.group(1).strip().removesuffix('.')
    examine = re.fullmatch(r'examine the (\w+) with the (\w+)', goal)
    if examine:
        return {'kind': 'examine', 'object_type': examine[1], 'tool_type': examine[2], 'count': 1}
    placement = re.fullmatch(r'put (some|a|two) (?:(clean|hot|cool) )?(\w+) (in|on) (\w+)', goal)
    if not placement or (placement[1] == 'two' and placement[2]):
        return None
    return {'kind': {'clean': 'clean', 'hot': 'heat', 'cool': 'cool'}.get(placement[2], 'place'),
        'object_type': placement[3], 'target_type': placement[5], 'count': 2 if placement[1] == 'two' else 1}


def entity_type(entity):
    match = re.fullmatch(r'(\w+) ([1-9][0-9]*)', entity)
    return match[1] if match else None


def parse_event(transition):
    action = transition['action']
    feedback = transition['next_observation']
    patterns = (
        ('take', r'take (.+?) from (.+)', 'You pick up the {object} from the {other}.'),
        ('move', r'move (.+?) to (.+)', 'You move the {object} to the {other}.'),
        ('heat', r'heat (.+?) with (.+)', 'You heat the {object} using the {other}.'),
        ('cool', r'cool (.+?) with (.+)', 'You cool the {object} using the {other}.'),
        ('clean', r'clean (.+?) with (.+)', 'You clean the {object} using the {other}.'),
    )
    for kind, pattern, template in patterns:
        match = re.fullmatch(pattern, action)
        if match and entity_type(match[1]) and entity_type(match[2]):
            expected = template.format(object=match[1], other=match[2])
            if feedback.strip() == expected:
                return {'kind': kind, 'object': match[1], 'other': match[2], 'step': transition['step'], 'quote': feedback}
    match = re.fullmatch(r'use (.+)', action)
    if match and entity_type(match[1]) and feedback.strip() == f'You turn on the {match[1]}.':
        return {'kind': 'use', 'object': None, 'other': match[1], 'step': transition['step'], 'quote': feedback}
    return None


def source_recipe(record, trajectory):
    result = {'accepted': False, 'reason': None, 'rule': None, 'evidence': [],
        'origin': 'author-designed closed-template extraction from observed trajectory',
        'semantic_generalization_verified': False, 'causal_necessity_verified': False,
        'usable_as_online_memory': False}
    if not isinstance(record, dict) or not isinstance(trajectory, list):
        return {**result, 'reason': 'malformed_source'}
    if type(record.get('reward')) is not int or record['reward'] not in (0, 1) or type(record.get('steps')) is not int or not isinstance(record.get('task_id'), str):
        return {**result, 'reason': 'malformed_record'}
    for row in trajectory:
        if not isinstance(row, dict) or type(row.get('step')) is not int or any(not isinstance(row.get(key), str) for key in ('task_id', 'observation', 'action', 'next_observation')) or any(type(row.get(key)) is not bool for key in ('done', 'success')):
            return {**result, 'reason': 'malformed_transition'}
    if not trajectory:
        return {**result, 'reason': 'empty_trajectory'}
    if [row['step'] for row in trajectory] != list(range(len(trajectory))):
        return {**result, 'reason': 'noncontiguous_steps'}
    if record['steps'] != len(trajectory) or any(row['task_id'] != record['task_id'] for row in trajectory):
        return {**result, 'reason': 'source_identity_or_length_mismatch'}
    if any(current['observation'] != previous['next_observation'] for previous, current in zip(trajectory, trajectory[1:])):
        return {**result, 'reason': 'observation_continuity_mismatch'}
    if any(row['done'] or row['success'] for row in trajectory[:-1]) or record['reward'] != int(trajectory[-1]['success']):
        return {**result, 'reason': 'inconsistent_terminal_outcome'}
    if record['reward'] == 1 and (record.get('score') != 1 or record.get('truncated') is not False or record.get('termination') != 'success' or trajectory[-1].get('score') != 1):
        return {**result, 'reason': 'inconsistent_success_record'}
    goal = parse_goal(trajectory[0]['observation'])
    if goal is None:
        return {**result, 'reason': 'unsupported_goal'}
    result['goal'] = goal
    events = [event for row in trajectory if (event := parse_event(row)) is not None]
    mismatches = [event for event in events if event['object'] is not None and entity_type(event['object']) != goal['object_type']]
    result['off_target_event_steps'] = [event['step'] for event in mismatches]
    if record['reward'] != 1 or not trajectory[-1]['done']:
        return {**result, 'reason': 'unsuccessful_source_with_off_target_actions' if mismatches else 'unsuccessful_source'}
    event_steps = {event['step'] for event in events}
    for row in trajectory:
        if row['step'] in event_steps or row['next_observation'].strip() == 'Nothing happens.':
            continue
        if not re.fullmatch(r'(?:go to|open|close|examine) \w+ [1-9][0-9]*|look|inventory', row['action']):
            return {**result, 'reason': 'unsupported_intervening_action'}
    held = None
    held_evidence = []
    chains = {}
    tool_type = {'heat': 'microwave', 'cool': 'fridge', 'clean': 'sinkbasin'}.get(goal['kind'])
    for event in events:
        kind = event['kind']
        if kind == 'take':
            chains.pop(event['object'], None)
            if held is not None:
                return {**result, 'reason': 'contradictory_successful_possession'}
            held = event['object']
            held_evidence = [event]
        elif kind in ('heat', 'cool', 'clean'):
            if held != event['object']:
                return {**result, 'reason': 'transform_without_supported_possession'}
            if kind != goal['kind']:
                return {**result, 'reason': 'unsupported_intervening_transform'}
            if entity_type(event['other']) != tool_type:
                return {**result, 'reason': 'unsupported_transform_tool'}
            if held == event['object'] and kind == goal['kind'] and entity_type(event['other']) == tool_type:
                held_evidence.append(event)
        elif kind == 'move':
            if held != event['object']:
                return {**result, 'reason': 'move_without_supported_possession'}
            if held == event['object'] and entity_type(held) == goal['object_type'] and entity_type(event['other']) == goal.get('target_type'):
                transformed = goal['kind'] == 'place' or any(item['kind'] == goal['kind'] for item in held_evidence)
                if transformed:
                    chains[held] = held_evidence + [event]
            if held == event['object']:
                held, held_evidence = None, []
        elif kind == 'use' and goal['kind'] == 'examine':
            if held is not None and entity_type(held) == goal['object_type'] and entity_type(event['other']) == goal['tool_type']:
                chains[held] = held_evidence + [event]
    if len(chains) < goal['count']:
        return {**result, 'reason': 'required_goal_aligned_chain_missing'}
    selected = sorted(chains.values(), key=lambda chain: chain[-1]['step'])[:goal['count']]
    if selected[-1][-1]['step'] != trajectory[-1]['step']:
        return {**result, 'reason': 'recipe_not_at_observed_success'}
    templates = {
        'place': 'For an ordinary placement goal, take an observed instance of the requested object from its observed source, then move that same instance to the requested destination. For a two-object goal, repeat with a distinct instance.',
        'heat': 'For a hot-object placement goal, take an observed requested-object instance, heat that same instance with an observed microwave, then move it to the requested destination.',
        'cool': 'For a cool-object placement goal, take an observed requested-object instance, cool that same instance with an observed fridge, then move it to the requested destination.',
        'clean': 'For a clean-object placement goal, take an observed requested-object instance, clean that same instance with an observed sinkbasin, then move it to the requested destination.',
        'examine': 'For an examine-with-lamp goal, take an observed requested-object instance and use the requested observed lamp while retaining that instance.',
    }
    if goal['kind'] == 'place' and goal['count'] == 1:
        rule = templates['place'].split(' For a two-object')[0]
    else:
        rule = templates[goal['kind']]
    if goal['kind'] == 'examine' and goal['tool_type'] not in ('desklamp', 'floorlamp'):
        return {**result, 'reason': 'unsupported_examine_tool'}
    return {**result, 'accepted': True, 'reason': 'observed_goal_aligned_success_chain', 'rule': rule,
        'evidence': [event for chain in selected for event in chain],
        'transfer_assumptions': ['Object/tool/destination bindings must be observed anew.',
            'Navigation/open/close/examine/look/inventory and literal no-op feedback are assumed to preserve tracked possession and treatment; hidden state is not verified.',
            'Omitted navigation, search, opening and state preconditions are not proved unnecessary.',
            'A successful source chain does not prove future benefit or pair synergy.']}
