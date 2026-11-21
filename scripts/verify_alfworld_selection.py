import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from comep.environments import ALFWorldAdapter


class Environment:
    def seed(self, seed):
        self.seed_value = seed

    def close(self):
        pass


class Wrapper:
    initialized = []

    def __init__(self, settings, train_eval):
        self.game_files = ['/eligible/train/b', '/eligible/train/a']
        self.num_games = 2

    def init_env(self, batch_size):
        self.initialized.append(list(self.game_files))
        return Environment()


def main():
    with tempfile.TemporaryDirectory() as temporary:
        config = Path(temporary) / 'config.json'
        config.write_text(json.dumps({'env': {}, 'general': {}, 'rl': {}}))
        with patch('alfworld.agents.environment.get_environment', return_value=Wrapper):
            default = ALFWorldAdapter(config, 'train')
            assert default.available_episodes == 2
            assert Wrapper.initialized[-1] == ['/eligible/train/a', '/eligible/train/b']
            chosen = ALFWorldAdapter(config, 'train', game_files=['/eligible/train/b'])
            assert chosen.available_episodes == 1
            assert Wrapper.initialized[-1] == ['/eligible/train/b']
            count = len(Wrapper.initialized)
            for invalid in ([], ['/eligible/train/a', '/eligible/train/a'], ['/other/valid/game']):
                try:
                    ALFWorldAdapter(config, 'train', game_files=invalid)
                except ValueError:
                    pass
                else:
                    raise AssertionError('ineligible or duplicate manifest accepted')
                assert len(Wrapper.initialized) == count
            default.close()
            chosen.close()
    print('ALFWORLD_SELECTION_WITNESS splitmembership,duplicate,empty,legacyorder; mockonly')


if __name__ == '__main__':
    main()
