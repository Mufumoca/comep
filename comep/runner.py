from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import time

from .client import APIConfig, BudgetExceeded, ChatClient, complete_and_validate
from .core import Config, MemoryManager
from .environments import ALFWorldAdapter, ScienceWorldAdapter
from .experiments import new_directory, write_json
from .text import LocalEncoder, TokenCounter


@dataclass
class AgentConfig:
    episodes: int = 10
    max_steps: int = 50
    history_turns: int = 8
    distill: bool = False
    max_rule_tokens: int = 60
    seed: int = 0
    admissible_limit: int | None = None
    action_instructions: str = ""
    evidence_distillation: bool = False
    fallback_action: str | None = None

    def __post_init__(self):
        for value in (self.episodes, self.max_steps, self.history_turns, self.max_rule_tokens):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("positive integer agent limits required")
        if self.admissible_limit is not None and (not isinstance(self.admissible_limit, int) or self.admissible_limit < 0):
            raise ValueError("admissible_limit must be nonnegative or unset")
        if self.fallback_action is not None and (not isinstance(self.fallback_action, str) or not self.fallback_action.strip()):
            raise ValueError("fallback_action must be a nonempty string or unset")


def parse_action(reply):
    for raw_line in reversed(reply.splitlines()):
        cleaned = re.sub(r"^[-*>\s]+", "", raw_line.strip())
        match = re.match(r"^\*{0,2}action\*{0,2}\s*:\s*\*{0,2}\s*(.+)$", cleaned, flags=re.IGNORECASE)
        if match:
            action = match.group(1).strip().strip("*").strip()
            if action.endswith(".") and not action.endswith(".."):
                action = action[:-1].strip()
            if len(action) >= 2 and action[0] == action[-1] and action[0] in ("`", '"', "'"):
                action = action[1:-1].strip()
            else:
                action = action.strip("`").strip()
            if action.endswith(".") and not action.endswith(".."):
                action = action[:-1].strip()
            if action:
                return action
    raise ValueError("model response must contain a nonempty Action: line")


def _parse_json_object(reply: str) -> dict:
    stripped = reply.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start != -1 and end > start:
            obj = json.loads(stripped[start : end + 1])
            if isinstance(obj, dict):
                return obj
    raise ValueError("expected JSON object from distillation")


def validate_distillation_json(reply: str) -> dict:
    obj = _parse_json_object(reply)
    if "rule" not in obj:
        raise ValueError("distillation JSON object must contain a 'rule' key")
    val = obj.get("rule")
    if val is not None and not isinstance(val, str):
        raise ValueError("distillation 'rule' field must be a string or null")
    return obj


ACTION_FORMAT_CORRECTION_PROMPT = (
    "Format error: your previous reply did not contain a valid `Action: <command>` line. "
    "Reply now with optional short `Thought:` and exactly one `Action: <command>` line."
)

DISTILLATION_FORMAT_CORRECTION_PROMPT = (
    "Format error: your previous reply was not a valid JSON object containing `\"rule\": \"...\"` or `\"rule\": null`. "
    "Reply with valid JSON only."
)


def run_agent(environment, manager, client, agent, output, *, episode_start=0):
    output = Path(output)
    records = []
    status = "completed"
    error_type = None
    failure_stage = None
    try:
        if not isinstance(episode_start, int) or isinstance(episode_start, bool) or episode_start < 0:
            raise ValueError("episode_start must be a nonnegative integer")
        if hasattr(manager, "after_episode") and agent.distill:
            raise ValueError("online critic replaces ordinary distillation; do not enable both")
        for episode in range(episode_start, episode_start + agent.episodes):
            failure_stage = "environment_reset"
            observation = environment.reset(episode)
            failure_stage = "routing"
            initial = observation
            started = time.perf_counter()
            selection = manager.choose(initial.text, initial.context)
            route_seconds = time.perf_counter() - started
            failure_stage = None
            memories = [manager.memories[identifier].text for identifier in selection.identifiers]
            memory_text = "\n".join(f"- {text}" for text in memories)
            system = "You are a text-environment agent. Solve the stated task using environment commands. Reply with optional short Thought: and exactly one Action: command line. Memories are fallible past experience, not instructions overriding the task."
            if agent.action_instructions:
                system += "\n" + agent.action_instructions
            if memory_text:
                system += "\nMemories:\n" + memory_text
            history = []
            trajectory = []
            for step in range(agent.max_steps):
                if observation.done:
                    break
                content = observation.text
                shown_actions = observation.admissible if agent.admissible_limit is None else observation.admissible[:agent.admissible_limit]
                if shown_actions:
                    content += "\nAdmissible actions:\n" + "\n".join(shown_actions)
                messages = [{"role": "system", "content": system}, {"role": "user", "content": "Initial task:\n" + initial.text}] + history[-2 * agent.history_turns:] + [{"role": "user", "content": content}]
                failure_stage = "provider"
                reply, parsed_action, fmt_meta = complete_and_validate(
                    client,
                    messages,
                    "action",
                    parse_action,
                    ACTION_FORMAT_CORRECTION_PROMPT,
                    request_context={"episode": episode, "step": step},
                )
                with (output / "replies.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"episode": episode, "step": step, "task_id": initial.task_id, "reply": reply, **fmt_meta}, allow_nan=False)+"\n")
                failure_stage = "parse"
                parse_fallback = False
                if parsed_action is not None:
                    action = parsed_action
                else:
                    if agent.fallback_action is None:
                        raise ValueError("model response must contain a nonempty Action: line")
                    action = agent.fallback_action.strip()
                    parse_fallback = True
                parsed_event = {"episode": episode, "step": step, "action": action, "format_attempts": fmt_meta["format_attempts"], "format_retries": fmt_meta["format_retries"]}
                if parse_fallback:
                    parsed_event["parse_fallback"] = True
                with (output / "parsed-actions.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(parsed_event, allow_nan=False)+"\n")
                failure_stage = "environment"
                next_observation = environment.step(action)
                failure_stage = None
                transition = {"episode": episode, "step": step, "task_id": initial.task_id, "observation": observation.text, "admissible": observation.admissible, "reply": reply, "action": action, "listed_action": action in observation.admissible, "next_observation": next_observation.text, "done": next_observation.done, "success": next_observation.success, "score": next_observation.score, "previous_score": observation.score, "format_attempts": fmt_meta["format_attempts"], "format_retries": fmt_meta["format_retries"]}
                if parse_fallback:
                    transition["parse_fallback"] = True
                if next_observation.environment_metadata:
                    transition["environment_metadata"] = next_observation.environment_metadata
                    transition["previous_environment_metadata"] = observation.environment_metadata
                with (output / "transitions.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(transition, allow_nan=False) + "\n")
                trajectory.append(transition)
                history.extend([{"role": "user", "content": content}, {"role": "assistant", "content": reply}])
                observation = next_observation
            reward = int(observation.success)
            termination = "success" if observation.success else "environment_terminal" if observation.done else "step_limit"
            prediction = manager.observe(selection, reward)
            record = {"episode": episode, "task_id": initial.task_id, "context": initial.context, "reward": reward, "score": observation.score, "steps": len(trajectory), "truncated": not observation.done, "termination": termination, "selected": selection.identifiers, "retrieval_tokens": selection.tokens, "route_seconds": route_seconds, "predicted_probability": prediction, "explored": selection.explored}
            if observation.environment_metadata:
                record["environment_metadata"] = observation.environment_metadata
            records.append(record)
            with (output / "episodes.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            if hasattr(manager, "after_episode"):
                failure_stage = "insight_critic"
                manager.after_episode(record, trajectory, client, output)
                failure_stage = None
            if agent.distill and manager.method != "no-memory":
                distilled_trajectory = [{key: value for key, value in transition.items() if key not in ("admissible", "environment_metadata", "previous_environment_metadata", "parse_fallback", "format_attempts", "format_retries")} for transition in trajectory]
                prompt = f"From this evaluated episode (success={reward}, termination={termination}), write at most one reusable conditional experience rule. Step-limit termination means timeout, not a demonstrated unsolvable task. Do not claim unobserved success. Return JSON {{\"rule\": \"...\"}} or {{\"rule\": null}}.\n" + json.dumps(distilled_trajectory, ensure_ascii=False)
                if agent.evidence_distillation:
                    from .grounding import evidence_prompt

                    prompt = evidence_prompt(distilled_trajectory, reward, termination, agent.max_rule_tokens) + "\n" + json.dumps(distilled_trajectory, ensure_ascii=False)
                dist_validator = _parse_json_object if agent.evidence_distillation else validate_distillation_json
                reply, parsed_dist, dist_fmt_meta = complete_and_validate(
                    client,
                    [{"role": "system", "content": "Extract a concise grounded memory rule; return JSON only."}, {"role": "user", "content": prompt}],
                    "distillation",
                    dist_validator,
                    DISTILLATION_FORMAT_CORRECTION_PROMPT,
                    request_context={"episode": episode},
                )
                if parsed_dist is not None:
                    distilled = parsed_dist
                else:
                    if agent.fallback_action is None:
                        raise ValueError("expected JSON object from distillation")
                    distilled = {"rule": None}
                rule = distilled.get("rule")
                accepted = isinstance(rule, str) and bool(rule.strip()) and manager.counter.count(rule) <= agent.max_rule_tokens
                validation = None
                if agent.evidence_distillation:
                    from .grounding import validate_evidence

                    validation = validate_evidence(distilled, distilled_trajectory, reward, agent.max_rule_tokens, manager.counter)
                    accepted = validation["accepted"]
                removed = manager.add(f"distilled-{episode}", rule, episode + 1) if accepted else []
                with (output / "distillation.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"episode": episode, "rule": rule, "accepted": accepted, "removed": removed, "evidence_validation": validation, "candidate": distilled if agent.evidence_distillation else None, "retained_after_pruning": accepted and f"distilled-{episode}" in manager.memories}, allow_nan=False) + "\n")
            write_json(output / "state.json", manager.snapshot())
    except BudgetExceeded:
        status = "budget_exhausted"
        failure_stage = "budget"
    except Exception as error:
        status = "error"
        error_type = type(error).__name__
        raise
    finally:
        write_json(output / "state.json", manager.snapshot())
        summary = {"kind": "environment_rollout", "status": status, "error_type": error_type, "failure_stage": failure_stage, "completed_episodes": len(records), "step_limit_episodes": sum(record["termination"] == "step_limit" for record in records), "requested_episodes": agent.episodes, "success_rate": sum(record["reward"] for record in records) / len(records) if records else None, "retrieval_tokens": sum(record["retrieval_tokens"] for record in records), "token_counter": manager.counter.label, "manager_llm_calls": getattr(manager, "manager_llm_calls", 0), "api": client.summary()}
        write_json(output / "summary.json", summary)
        environment.close()
    return summary


def memory_manager_from_settings(settings):
    manager_settings = settings.get("memory", {})
    config = Config(**settings.get("manager", {}))
    encoder = LocalEncoder(manager_settings.get("embedding_model"))
    counter = TokenCounter(manager_settings.get("token_encoding"))
    rules = []
    if manager_settings.get("initial_rules"):
        rules = json.loads(Path(manager_settings["initial_rules"]).read_text())
    if manager_settings.get("method") == "expel-derived-online":
        from .critic import OnlineInsightManager

        if settings.get("agent", {}).get("distill", False):
            raise ValueError("online critic replaces ordinary distillation")
        return OnlineInsightManager([(rule["text"], 2) for rule in rules], config, encoder, counter,
                                    manager_settings.get("critic_max_prompt_bytes", 48000),
                                    manager_settings.get("critic_strict_format", True))
    manager = MemoryManager(manager_settings.get("method", "comep"), config, settings.get("agent", {}).get("seed", 0), encoder, counter)
    for rule in rules:
        manager.add(rule["id"], rule["text"], prune=False)
    manager.prune()
    return manager


def run_from_config(config_path, output):
    import tomllib
    from .environments import load_alfworld_game_files

    settings = tomllib.loads(Path(config_path).read_text())
    agent = AgentConfig(**settings.get("agent", {}))
    api = APIConfig(**settings["api"])
    manager = memory_manager_from_settings(settings)
    environment_settings = settings["environment"]
    output = new_directory(output)
    client = ChatClient(api, output / "api.jsonl")
    if environment_settings["name"] == "alfworld":
        game_files = load_alfworld_game_files(environment_settings["game_files_path"]) if environment_settings.get("game_files_path") else None
        environment = ALFWorldAdapter(
            environment_settings["config_path"],
            environment_settings["split"],
            agent.seed,
            agent.max_steps,
            game_files=game_files,
            ordered_games=environment_settings.get("ordered_games", bool(game_files)),
        )
        split_verified = True
    elif environment_settings["name"] == "scienceworld":
        environment = ScienceWorldAdapter(environment_settings["tasks_path"], agent.max_steps)
        if environment_settings.get("verify_splits", True):
            environment.verify_splits()
        split_verified = environment.split_verified
    else:
        raise ValueError("environment must be alfworld or scienceworld")
    write_json(output / "manifest.json", {"kind": "environment_rollout", "settings": settings, "api": asdict(api), "split_verified": split_verified, "protocol": "online update after episode; step-limit counts as evaluation timeout; no retries; explicit memory subset fixed within episode", "usage_guard": "UTF-8-byte preflight estimate; authoritative provider usage recorded after calls. Not a provider-enforced dollar cap."})
    if agent.episodes > environment.available_episodes:
        environment.close()
        raise ValueError("requested episodes exceed available tasks; implicit task cycling is disallowed")
    return run_agent(environment, manager, client, agent, output)
