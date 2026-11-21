from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

from comep.environments import Observation
from comep.factorial import CostGuardClient
from run_alfworld_pilot import IdentityEnvironment, SCOPE, digest, run, validate_launch, validate_packet
from verify_factorial import MockClient
from analyze_alfworld_pilot import analyze


class Environment:
    def __init__(self, identity='fixture'):
        self.identity = identity
        self.closed = False

    def reset(self, episode):
        return Observation('Your task is: put the object on the table.', self.identity, 'alfworld', admissible=['look around'])

    def step(self, action):
        return Observation('Task complete.', self.identity, 'alfworld', True, True, 1)

    def close(self):
        self.closed = True


def main():
    path = Path('outputs/alfworld-pilot-preflight-20261104/packet.json')
    packet = json.loads(path.read_text())
    validate_packet(packet)
    approval = {'scope': SCOPE, 'operator_authorized': True, 'credential_reuse_authorized': True,
        'cap_usd': .15, 'model': 'deepseek-flash', 'prior_spent_and_other_reserved_usd': 1.5394785,
        'input_usd_per_million': .30, 'output_usd_per_million': 1.20, 'pricing_source': 'fixture',
        'pricing_checked_at_utc': datetime.now(timezone.utc).isoformat()}
    review = {'review_scope': SCOPE, 'code_protocol_ready': True, 'scientific_pilot_justified': True, 'packet_sha256': digest(path)}
    validate_launch(packet, approval, review, path)
    for field in ('code_protocol_ready', 'scientific_pilot_justified'):
        try:
            validate_launch(packet, approval, {**review, field: False}, path)
        except ValueError:
            pass
        else:
            raise AssertionError('missing review gate accepted')
    for altered in ({**review, 'packet_sha256': 'wrong'}, {**review, 'review_scope': 'wrong'}):
        try:
            validate_launch(packet, approval, altered, path)
        except ValueError:
            pass
        else:
            raise AssertionError('wrong review scope/hash accepted')
    for prior in (None, True, -1, float('nan'), float('inf'), 4.35):
        try:
            validate_launch(packet, {**approval, 'prior_spent_and_other_reserved_usd': prior}, review, path)
        except ValueError:
            pass
        else:
            raise AssertionError('unsafe budget accepted')
    changed = deepcopy(packet)
    changed['trials'][0]['game'] = changed['trials'][1]['game']
    try:
        validate_packet(changed)
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate task accepted')
    identity = IdentityEnvironment(Environment('actual'), 'expected')
    try:
        identity.reset(0)
    except ValueError:
        pass
    else:
        raise AssertionError('wrong executed identity accepted')
    environments = []

    def factory(trial):
        environment = Environment()
        environments.append(environment)
        return environment

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        class LoggedClient(MockClient):
            def __init__(self, output, limit=1000):
                super().__init__(limit=limit)
                self.output = output

            def complete(self, messages, purpose='action'):
                response = super().complete(messages, purpose)
                with (self.output / 'api.jsonl').open('a') as handle:
                    handle.write(json.dumps({'usage': {'prompt_tokens': 100, 'completion_tokens': 10}})+'\n')
                return response

        client = LoggedClient(root / 'full')
        def matched_factory(trial):
            environment = Environment(trial['game'])
            environments.append(environment)
            return environment

        rows = run(packet, CostGuardClient(client, .15), matched_factory, root / 'full')
        assert len(rows) == 6 and all(row['episode']['reward'] == 1 for row in rows)
        assert client.calls == 6
        assert analyze(root / 'full')['successes'] == 6
        assert all('Memories:' not in message['content'] for messages in client.messages for message in messages)
        assert all('game.tw-pddl' not in message['content'] for messages in client.messages for message in messages)
        partial = run(packet, CostGuardClient(LoggedClient(root / 'partial', limit=2), .15), matched_factory, root / 'partial')
        assert len(partial) == 3 and partial[-1]['episode'] is None
        assert partial[-1]['status'] == 'budget_exhausted'
        summary = json.loads((root / 'partial' / 'summary.json').read_text())
        assert summary['expected_trials'] == 6
        analyzed = analyze(root / 'partial')
        assert analyzed['completed'] == 2 and not analyzed['complete']
        assert analyzed['rows'][-1]['success'] is None
        blocked = run(packet, CostGuardClient(MockClient(), 1e-12), factory, root / 'no-dispatch')
        assert blocked[0]['episode'] is None

        class TruncatedClient(MockClient):
            def complete(self, messages, purpose='action'):
                response = super().complete(messages, purpose)
                self.last_response = {'finish_reason': 'length', 'response_model': 'mock', 'system_fingerprint': 'fixed'}
                return response

        truncated_client = TruncatedClient()
        truncated = run(packet, CostGuardClient(truncated_client, .15), factory, root / 'truncated')
        assert len(truncated) == 1 and truncated[0]['episode'] is None and truncated[0]['status'] == 'error'
        assert truncated_client.calls == 1
    assert all(environment.closed for environment in environments)
    print('ALFWORLD_PILOT_WITNESS sixmock,nomemory,nopathprompts,identity,budget,missingness,closures; noAPI')


if __name__ == '__main__':
    main()
