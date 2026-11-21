from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path


def load_alfworld_game_files(manifest_path, verify_hashes=True) -> list[str]:
    entries = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if isinstance(entries, dict) and "games" in entries:
        entries = entries["games"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("ALFWorld manifest must be a nonempty list")
    paths = []
    for entry in entries:
        if isinstance(entry, str):
            paths.append(entry)
        elif isinstance(entry, dict) and isinstance(entry.get("game"), str):
            game_path = Path(entry["game"])
            if verify_hashes and entry.get("sha256"):
                digest = hashlib.sha256(game_path.read_bytes()).hexdigest()
                if digest != entry["sha256"]:
                    raise ValueError(f"SHA256 mismatch for ALFWorld game: {game_path}")
            paths.append(str(game_path))
        else:
            raise ValueError("invalid ALFWorld manifest entry")
    return paths


@dataclass
class Observation:
    text: str
    task_id: str
    context: str
    done: bool = False
    success: bool = False
    score: float = 0.0
    admissible: list[str] = field(default_factory=list)
    environment_metadata: dict = field(default_factory=dict)


def _infer_alfworld_context(text: str) -> str:
    """Infer ALFWorld task family strictly from visible initial task instructions (zero path leakage)."""
    import re

    lower = text.lower()
    marker = "your task is to:"
    if marker not in lower:
        return "alfworld"
    goal = lower.split(marker, 1)[1].splitlines()[0]
    if re.search(r"\b(two|2)\b", goal):
        return "alfworld:two_obj"
    if re.search(r"\b(clean|cleaned|wash|washed|rinse|rinsed)\b", goal):
        return "alfworld:clean"
    if re.search(r"\b(heat|heated|hot|warm|warmed)\b", goal):
        return "alfworld:heat"
    if re.search(r"\b(cool|cooled|chilled|cold)\b", goal):
        return "alfworld:cool"
    if re.search(r"\b(examine|look at)\b", goal):
        return "alfworld:examine"
    return "alfworld:put"


def _ensure_java_environment():
    try:
        import shutil

        if shutil.which("java"):
            return
        import jdk4py

        os.environ["JAVA_HOME"] = str(jdk4py.JAVA_HOME)
        java_bin = str(jdk4py.JAVA_HOME / "bin")
        if java_bin not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = java_bin + os.pathsep + os.environ.get("PATH", "")
    except ImportError:
        pass


class ALFWorldAdapter:
    def __init__(self, config_path, split, seed=0, max_steps=50, game_files=None, ordered_games=False):
        import yaml
        from alfworld.agents.environment import get_environment

        if split not in ("train", "eval_in_distribution", "eval_out_of_distribution"):
            raise ValueError("explicit ALFWorld split required")
        settings = yaml.safe_load(Path(config_path).read_text())
        settings["env"]["type"] = "AlfredTWEnv"
        settings.setdefault("general", {})["training_method"] = "dqn"
        settings.setdefault("rl", {}).setdefault("training", {})["max_nb_steps_per_episode"] = max_steps
        self._settings = settings
        self._split = split
        self._seed = seed
        self._ordered_games = bool(ordered_games and game_files is not None)
        wrapper = get_environment("AlfredTWEnv")(settings, train_eval=split)
        wrapper.game_files = sorted(wrapper.game_files)
        if game_files is not None:
            requested = [str(Path(path).resolve()) for path in game_files]
            eligible = {str(Path(path).resolve()) for path in wrapper.game_files}
            if not requested or len(set(requested)) != len(requested) or not set(requested) <= eligible:
                raise ValueError("requested ALFWorld games must be distinct members of the declared split")
            wrapper.game_files = requested
            wrapper.num_games = len(requested)
        if not wrapper.game_files:
            raise ValueError("ALFWorld split contains no games")
        self._game_list = list(wrapper.game_files)
        self._wrapper = wrapper
        self.available_episodes = len(self._game_list)
        self.env = None if self._ordered_games else wrapper.init_env(batch_size=1)
        if self.env is not None:
            self.env.seed(seed)
        self.current = None
        self._episode_context = "alfworld"

    def unpack(self, observations, infos, done=False):
        game = infos.get("extra.gamefile", [None])[0]
        if game is None:
            raise ValueError("ALFWorld did not expose extra.gamefile; cannot audit task identity")
        success = bool(infos.get("won", [False])[0])
        obs_text = str(observations[0])
        ctx = getattr(self, "_episode_context", "alfworld")
        if "your task is to:" in obs_text.lower():
            ctx = _infer_alfworld_context(obs_text)
            self._episode_context = ctx
        self.current = Observation(obs_text, str(game), ctx, bool(done) or success, success, float(success), list(infos.get("admissible_commands", [[]])[0]))
        return self.current

    def reset(self, episode):
        self._episode_context = "alfworld"
        if getattr(self, "_ordered_games", False):
            if self.env is not None:
                self.env.close()
            target_game = self._game_list[episode]
            self._wrapper.game_files = [target_game]
            self._wrapper.num_games = 1
            self.env = self._wrapper.init_env(batch_size=1)
            self.env.seed(self._seed + int(episode))
        observations, infos = self.env.reset()
        return self.unpack(observations, infos)

    def step(self, action):
        if self.current is None:
            raise ValueError("ALFWorld must be reset before stepping")
        observations, rewards, dones, infos = self.env.step([action])
        if infos.get("extra.gamefile", [None])[0] is None:
            infos["extra.gamefile"] = [self.current.task_id]
        elif infos["extra.gamefile"][0] != self.current.task_id:
            raise ValueError("ALFWorld game identity changed within an episode")
        return self.unpack(observations, infos, dones[0])

    def close(self):
        if getattr(self, "env", None) is not None:
            self.env.close()
            self.env = None


class ScienceWorldAdapter:
    def __init__(self, tasks_path, max_steps=50, tasks_override=None):
        _ensure_java_environment()
        from scienceworld import ScienceWorldEnv

        self.tasks = list(tasks_override) if tasks_override is not None else json.loads(Path(tasks_path).read_text())
        if not isinstance(self.tasks, list) or not self.tasks:
            raise ValueError("ScienceWorld requires a nonempty task manifest")
        for task in self.tasks:
            if not isinstance(task.get("task"), str) or not task["task"].strip() or not isinstance(task.get("variation"), int) or isinstance(task["variation"], bool) or task["variation"] < 0 or not isinstance(task.get("split"), str) or not task["split"].strip():
                raise ValueError("each task requires task, integer variation and declared split")
        self.available_episodes = len(self.tasks)
        self.env = ScienceWorldEnv(envStepLimit=max_steps)
        self.current_task = None
        self.split_verified = False

    def verify_splits(self):
        accessors = {"train": "get_variations_train", "dev": "get_variations_dev", "validation": "get_variations_dev", "test": "get_variations_test"}
        for task in self.tasks:
            if task["split"] not in accessors:
                raise ValueError("ScienceWorld split must be train/dev/validation/test for verification")
            self.env.load(task["task"], task["variation"], simplificationStr=task.get("simplifications", ""), generateGoldPath=False)
            variations = getattr(self.env, accessors[task["split"]])()
            if task["variation"] not in variations:
                raise ValueError(f"variation does not belong to declared split: {task}")
        self.split_verified = True

    def unpack(self, text, info, done=False):
        task = self.current_task
        score = float(info["score"])
        description = info.get("taskDesc") or self.env.taskdescription()
        if not isinstance(description, str) or not description.strip():
            raise ValueError("ScienceWorld returned an empty task objective")
        admissible = info["valid"] if "valid" in info else self.env.get_valid_action_object_combinations()
        metadata = {
            "moves": info.get("moves"),
            "move_limit": getattr(self.env, "envStepLimit", None),
            "server_completed": info.get("server_completed"),
            "wrapper_done": bool(done),
            "negative_score": score < 0,
        }
        metadata["move_limit_exceeded"] = metadata["moves"] > metadata["move_limit"] if metadata["moves"] is not None and metadata["move_limit"] is not None else None
        return Observation(f"Task: {description}\n{text}", f"{task['task']}:{task['variation']}:{task['split']}", task["task"], bool(done) or score >= 100, score >= 100, score, list(admissible), metadata)

    def reset(self, episode):
        self.current_task = self.tasks[episode]
        self.env.load(self.current_task["task"], self.current_task["variation"], simplificationStr=self.current_task.get("simplifications", ""), generateGoldPath=False)
        text, info = self.env.reset()
        info = {**info, "server_completed": bool(self.env.server.getCompleted())}
        return self.unpack(text, info)

    def step(self, action):
        text, reward_delta, done, info = self.env.step(action)
        info = {**info, "server_completed": bool(self.env.server.getCompleted())}
        return self.unpack(text, info, done)

    def close(self):
        self.env.close()
