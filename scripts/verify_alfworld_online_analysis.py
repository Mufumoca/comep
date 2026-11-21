from copy import deepcopy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import sys

from analyze_alfworld_online import analyze, main as analysis_main
from run_alfworld_online import METHODS


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


def lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def fixture(root, count=90, extraction_failure=False, unknown=False):
    trials = [{'trial': index, 'game': f'train/game-{index}', 'task_type': f'family-{index % 6}'} for index in range(30)]
    save(root / 'campaign.json', {'trials': trials})
    rows, api = [], []
    for trial in trials:
        index = trial['trial']
        for method in METHODS[index % 3:] + METHODS[:index % 3]:
            if len(rows) == count:
                break
            directory = root / f'trial-{index:03d}' / method
            memory = {'identifier': 'past', 'text': 'past rule'}
            save(directory / 'state-before.json', {'observations': index, 'memories': [memory] if index else []})
            failed = len(rows) == count - 1 and extraction_failure
            reward = 0 if method == 'individual' else int(index % 2 == (method == 'comep'))
            episode = {'episode': index, 'task_id': trial['game'], 'reward': reward, 'steps': 1,
                'selected': ['past'] if index else [], 'retrieval_tokens': int(index > 0),
                'route_seconds': 0, 'predicted_probability': 0.0}
            lines(directory / 'episodes.jsonl', [episode])
            requests = []
            for purpose in ('action', 'distillation'):
                call = {'call': len(api) + 1, 'purpose': purpose, 'usage': {'prompt_tokens': 10, 'completion_tokens': 2}}
                if failed and unknown and purpose == 'distillation':
                    del call['usage']
                api.append(call)
                requests.append({'next_call': call['call'], 'purpose': purpose})
            lines(directory / 'requests.jsonl', requests)
            distilled = [] if failed else [{'accepted': index % 3 != 0, 'retained_after_pruning': index % 3 == 1,
                'removed': ['old'] if index % 3 == 2 else []}]
            lines(directory / 'distillation.jsonl', distilled)
            removed_count = sum(prior % 3 == 2 for prior in range(index + int(not failed)))
            save(directory / 'state.json', {'archive': [{}] * removed_count, 'bank_tokens': 1,
                'memories': [memory], 'pairs': []})
            usage = {'attempted_calls': 2,
                **{field: sum(call.get('usage', {}).get(field, 0) for call in api[-2:]) for field in ('prompt_tokens', 'completion_tokens')}}
            rows.append({**trial, 'method': method, 'status': 'error' if failed else 'completed', 'episode': episode,
                'usage': usage, 'usage_complete': not (failed and unknown)})
    totals = {'attempted_calls': len(api), 'usage_complete': not unknown,
        **{field: sum(call.get('usage', {}).get(field, 0) for call in api) for field in ('prompt_tokens', 'completion_tokens')}}
    save(root / 'summary.json', {'rows': rows, 'status': 'error' if extraction_failure else 'completed', 'api': totals})
    lines(root / 'trials.jsonl', rows)
    lines(root / 'api.jsonl', api)


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        full = root / 'full'
        fixture(full)
        result = analyze(full)
        assert len(result['cells']) == 90 and len(result['paired_tasks']) == 30
        assert result['paired_success_count_difference'] == {'additive': 0, 'individual': 15}
        for method in result['methods']:
            assert method['outcomes'] == method['complete_update_cells'] == 30
            assert method['successes'] == (0 if method['method'] == 'individual' else 15)
            assert method['success_rate'] == (0 if method['method'] == 'individual' else .5)
            assert method['action_calls'] == method['distillation_calls'] == 30
            assert method['prompt_tokens'] == 600 and method['completion_tokens'] == 120
            assert method['accepted_rules'] == method['admitted_rules'] == 20
            assert method['retained_arrivals'] == method['pruned_rules'] == method['archived_rules'] == 10
            if method['method'] == 'individual':
                assert method['tokens_per_success'] is None
        assert all(cell['predicted_probability'] == 0.0 for cell in result['cells'])
        assert all(block['paired_tasks'] == 6 and block['individual'] == 0 and block['additive'] == block['comep'] == 3 for block in result['blocks'])
        for unknown in (False, True):
            partial = root / f'partial-{unknown}'
            fixture(partial, count=3, extraction_failure=True, unknown=unknown)
            result = analyze(partial)
            assert len(result['cells']) == 90 and len(result['paired_tasks']) == 1
            assert sum(cell['reward'] is None for cell in result['cells']) == 87
            comep = next(method for method in result['methods'] if method['method'] == 'comep')
            assert comep['outcomes'] == 1 and comep['successes'] == 0 and comep['complete_update_cells'] == 0
            assert comep['admitted_rules'] == 0
            assert (comep['estimated_usd'] is None) == unknown
            assert result['usage_complete'] == (not unknown)
            destination = root / f'report-{unknown}'
            with patch.object(sys, 'argv', ['analyze_alfworld_online.py', '--source', str(partial), '--output', str(destination)]), \
                    patch('builtins.print'):
                analysis_main()
            assert (destination / 'REPORT.md').exists()
            if unknown:
                assert 'unknown' in (destination / 'REPORT.md').read_text()
        empty = root / 'empty'
        fixture(empty, count=0)
        result = analyze(empty)
        assert not result['paired_tasks'] and all(method['success_rate'] is None for method in result['methods'])
        no_outcome = root / 'no-outcome'
        fixture(no_outcome, count=3, extraction_failure=True)
        summary_path = no_outcome / 'summary.json'
        summary = json.loads(summary_path.read_text())
        summary['rows'][-1]['episode'] = None
        save(summary_path, summary)
        lines(no_outcome / 'trials.jsonl', summary['rows'])
        lines(no_outcome / 'trial-000' / 'comep' / 'episodes.jsonl', [])
        result = analyze(no_outcome)
        assert not result['paired_tasks']
        assert sum(cell['reward'] is None for cell in result['cells']) == 88
        assert next(method for method in result['methods'] if method['method'] == 'comep')['outcomes'] == 0
        api_path = full / 'api.jsonl'
        original = [json.loads(line) for line in api_path.read_text().splitlines()]
        for mutation in ('duplicate', 'tokens'):
            changed = deepcopy(original)
            if mutation == 'duplicate':
                changed[1]['call'] = changed[0]['call']
            else:
                changed[0]['usage']['prompt_tokens'] += 1
            lines(api_path, changed)
            try:
                analyze(full)
            except ValueError:
                pass
            else:
                raise AssertionError(f'corrupted {mutation} accounting accepted')
        lines(api_path, original)
        request_path = full / 'trial-000' / 'additive' / 'requests.jsonl'
        lines(request_path, [{'next_call': 1, 'purpose': 'action'}, {'next_call': 2, 'purpose': 'distillation'}])
        try:
            analyze(full)
        except ValueError:
            pass
        else:
            raise AssertionError('duplicated request assignment accepted')
    print('ONLINE_ANALYSIS_VERIFY: synthetic 90 cells, partial extraction, unknown usage, zeros, blocks, admissions/pruning, accounting corruption; no live results or network')


if __name__ == '__main__':
    main()
