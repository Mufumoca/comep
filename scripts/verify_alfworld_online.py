from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import sys
from unittest.mock import patch

from comep.client import BudgetExceeded
from comep.environments import Observation
from comep.factorial import CostGuardClient
import run_alfworld_online as online
from run_alfworld_online import CAP, METHODS, SCOPE, run, validate_launch, validate_packet
from run_alfworld_pilot import IdentityEnvironment, digest
from verify_factorial import MockClient


class Client(MockClient):
    def complete(self, messages, purpose='action'):
        if purpose == 'action':
            assert all('game.tw-pddl' not in message['content'] and 'json_2.1.1/' not in message['content'] for message in messages)
        super().complete(messages, purpose)
        request_log = getattr(self, 'request_log', None)
        if request_log is None:
            return json.dumps({'rule': 'Inspect the room first.'}) if purpose == 'distillation' else 'Action: look around'
        method = Path(request_log).parent.name
        system = messages[0]['content']
        for other in METHODS:
            if other != method:
                assert f'private-{other}' not in system
        if purpose == 'action':
            index = int(Path(self.request_log).parent.parent.name.split('-')[1])
            if index == 0:
                assert 'Memories:' not in system
        return json.dumps({'rule': f'private-{method}: If carrying the target, place it in the requested container.'}) if purpose == 'distillation' else 'Action: look around'


class Environment:
    def __init__(self, trial):
        self.trial = trial
        self.closed = False

    def reset(self, episode):
        assert episode == self.trial['trial']
        return Observation('Your task is: put the target in the container.', self.trial['game'], 'alfworld', admissible=['look around'])

    def step(self, action):
        assert action == 'look around'
        return Observation('Task complete.', self.trial['game'], 'alfworld', True, True, 1)

    def close(self):
        self.closed = True


def main():
    packet_path = Path('outputs/alfworld-online-preflight-20261104-v2/packet.json')
    packet = json.loads(packet_path.read_text())
    validate_packet(packet)
    approval = {'scope': SCOPE, 'operator_authorized': True, 'credential_reuse_authorized': True,
        'cap_usd': CAP, 'model': 'deepseek-flash', 'input_usd_per_million': .30, 'output_usd_per_million': 1.20,
        'pricing_source': 'fixture', 'pricing_checked_at_utc': datetime.now(timezone.utc).isoformat()}
    review = {'scope': SCOPE, 'scientific_pilot_justified': True, 'code_protocol_ready': True, 'packet_sha256': digest(packet_path)}
    ledger = {'cumulative_peak_estimate_usd': 1.5947499, 'other_reserved_usd': 0}
    validate_launch(packet, approval, review, packet_path, ledger)
    for changed in ({**ledger, 'other_reserved_usd': .1}, {**ledger, 'cumulative_peak_estimate_usd': 3},
                    {**ledger, 'other_unknown_usage': True}, {**ledger, 'alfworld_online_unknown_usage': True},
                    {**ledger, 'cumulative_peak_estimate_usd': float('nan')},
                    {**ledger, 'cumulative_peak_estimate_usd': True},
                    {**ledger, 'alfworld_online_actual_peak_estimate_usd': 0}):
        try:
            validate_launch(packet, approval, review, packet_path, changed)
        except ValueError:
            pass
        else:
            raise AssertionError('unsafe ledger accepted')
    for field in ('scientific_pilot_justified', 'code_protocol_ready'):
        try:
            validate_launch(packet, approval, {**review, field: False}, packet_path, ledger)
        except ValueError:
            pass
        else:
            raise AssertionError('missing scoped review accepted')
    for changed in ({**review, 'scope': 'wrong'}, {**review, 'packet_sha256': 'wrong'}):
        try:
            validate_launch(packet, approval, changed, packet_path, ledger)
        except ValueError:
            pass
        else:
            raise AssertionError('unbound review accepted')
    for field, value in [('manager', {**packet['manager'], 'matched_pair_lookahead': False}), ('estimated_cap_usd', 5)]:
        changed = deepcopy(packet)
        changed[field] = value
        try:
            validate_packet(changed)
        except ValueError:
            pass
        else:
            raise AssertionError('protocol drift accepted')
    environments = []

    def factory(trial):
        environment = Environment(trial)
        environments.append(environment)
        return environment

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        output = root / 'complete'
        rows = run(packet, Client(limit=180), factory, output)
        assert len(rows) == 90 and all(row['status'] == 'completed' for row in rows)
        assert all(environment.closed for environment in environments)
        for index in range(30):
            expected = METHODS[index % 3:] + METHODS[:index % 3]
            assert tuple(row['method'] for row in rows[index*3:index*3+3]) == expected
            for method in METHODS:
                directory = output / f'trial-{index:03d}' / method
                before = json.loads((directory / 'state-before.json').read_text())
                distilled = json.loads((directory / 'distillation.jsonl').read_text())
                assert before['observations'] == index
                assert distilled['episode'] == index and distilled['accepted']
                assert len(before['memories']) == index
                assert {memory['identifier'] for memory in before['memories']} == {f'distilled-{prior}' for prior in range(index)}
                assert all(f'private-{method}' in memory['text'] for memory in before['memories'])
                assert all(memory['created_at'] == int(memory['identifier'].split('-')[1]) + 1 for memory in before['memories'])
        limited = run(packet, Client(limit=3), factory, root / 'limited')
        assert len(limited) == 2 and limited[-1]['status'] == 'budget_exhausted'
        assert limited[-1]['episode'] is not None
        assert json.loads((root / 'limited' / 'summary.json').read_text())['status'] == 'budget_exhausted'
        class BrokenClient(Client):
            def complete(self, messages, purpose='action'):
                raise ValueError('fixture provider error')
        broken = run(packet, BrokenClient(), factory, root / 'broken')
        assert len(broken) == 1 and broken[0]['episode'] is None and broken[0]['status'] == 'error'

        def wrong_factory(trial):
            return IdentityEnvironment(factory(trial), 'wrong-game')

        wrong_client = Client()
        wrong = run(packet, wrong_client, wrong_factory, root / 'wrong-task')
        assert len(wrong) == 1 and wrong[0]['status'] == 'error' and wrong[0]['episode'] is None
        assert wrong_client.calls == 0

        def failed_factory(trial):
            raise RuntimeError('fixture environment construction failure')

        failed = run(packet, Client(), failed_factory, root / 'factory-failure')
        assert len(failed) == 1 and failed[0]['status'] == 'error' and failed[0]['episode'] is None

        class MetadataClient(Client):
            mode = 'truncation'

            def complete(self, messages, purpose='action'):
                response = super().complete(messages, purpose)
                self.last_response = {'finish_reason': 'length' if self.mode == 'truncation' else 'stop',
                    'response_model': 'mock', 'system_fingerprint': 'changed' if self.calls > 1 else 'fixed'}
                if self.mode == 'parse':
                    return 'not an action'
                return response

        for mode in ('truncation', 'identity', 'parse'):
            raw = MetadataClient()
            raw.mode = mode
            guarded = CostGuardClient(raw, CAP)
            failed = run(packet, guarded, factory, root / mode)
            assert len(failed) == 1 and failed[0]['status'] == 'error'
            assert raw.calls == (2 if mode == 'identity' else 1)
            assert (failed[0]['episode'] is not None) == (mode == 'identity')

        guarded = CostGuardClient(Client(), 1e-12)
        blocked = run(packet, guarded, factory, root / 'dollar-limit')
        assert len(blocked) == 1 and blocked[0]['status'] == 'budget_exhausted'
        assert guarded.summary()['attempted_calls'] == 0

        class UnknownClient(Client):
            def summary(self):
                result = super().summary()
                if self.calls:
                    result.update(usage_complete=False, estimated_cost_usd=None)
                return result

            def complete(self, messages, purpose='action'):
                super().complete(messages, purpose)
                raise ValueError('fixture missing provider usage')

        for name, raw in (('known', Client()), ('unknown', UnknownClient())):
            ledger_path = root / f'{name}-ledger.json'
            ledger_path.write_text(json.dumps(ledger))
            approval_path = root / 'approval.json'
            review_path = root / 'review.json'
            approval_path.write_text(json.dumps(approval))
            review_path.write_text(json.dumps(review))
            destination = root / f'cli-{name}'
            arguments = ['run_alfworld_online.py', 'execute', '--packet', str(packet_path),
                '--approval', str(approval_path), '--review', str(review_path), '--output', str(destination), '--execute-paid']

            def dispatch(packet, guarded, factory, output):
                reserved = json.loads(ledger_path.read_text())
                assert reserved['alfworld_online_reserved_usd'] == CAP
                assert abs(reserved['remaining_unreserved_to_internal_stop_usd'] - .5052501) < 1e-9
                output.mkdir()
                guarded.request_log = output / 'trial-000' / METHODS[0] / 'requests.jsonl'
                guarded.request_log.parent.mkdir(parents=True)
                try:
                    guarded.complete([{'role': 'system', 'content': 'fixture'}])
                except ValueError:
                    assert name == 'unknown'

            with patch.object(online, 'LEDGER', ledger_path), patch.object(online, 'ChatClient', return_value=raw), \
                    patch.object(online, 'run', side_effect=dispatch), patch.object(sys, 'argv', arguments), \
                    patch('comep.environments.ALFWorldAdapter'):
                online.main()
            settled = json.loads(ledger_path.read_text())
            if name == 'known':
                assert settled['alfworld_online_reserved_usd'] == 0
                assert settled['alfworld_online_actual_peak_estimate_usd'] == raw.summary()['estimated_cost_usd']
                assert settled['cumulative_peak_estimate_usd'] == ledger['cumulative_peak_estimate_usd'] + raw.summary()['estimated_cost_usd']
            else:
                assert settled['alfworld_online_reserved_usd'] == CAP
                assert settled['alfworld_online_unknown_usage'] is True
                assert settled['cumulative_peak_estimate_usd'] == ledger['cumulative_peak_estimate_usd']
        assert all(environment.closed for environment in environments)
    print('ALFWORLD_ONLINE_VERIFY: 90 isolated own-history cells, cyclic order, unique IDs, bounded failures, review gates, CLI ledger settlement; no remote/API calls')


if __name__ == '__main__':
    main()
