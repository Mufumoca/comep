import math


KINDS = {"task_success", "progress", "observation", "negative_outcome"}


def evidence_prompt(trajectory, reward, termination, max_rule_tokens):
    return (
        f"Extract at most one conditional rule from this evaluated episode. Success={reward}; "
        f"termination={termination}. Keep the rule within {max_rule_tokens} lexical tokens "
        "(words and punctuation each count); shorter is safer. Return JSON with rule, kind, "
        "and evidence. Example schema: {\"rule\":\"...\",\"kind\":\"observation\","
        "\"evidence\":[{\"step\":0,\"quote\":\"exact substring of next_observation\"}]}. "
        "Use rule:null and evidence:[] to abstain. kind is task_success, progress, observation, "
        "or negative_outcome. Do not turn an unsuccessful terminal action into positive advice. "
        "If an action immediately terminates unsuccessfully, any rule using that action as "
        "evidence must describe the observed negative outcome, not recommend it. A timeout "
        "does not prove an action or task impossible. Preserve task-specific conditions; "
        "parser acceptance does not prove task completion or universal usefulness. "
        "Every evidence quote must be exact visible feedback at the stated step. "
        "Do not invent unobserved causes. These checks establish provenance, not rule truth."
    )


def validate_evidence(candidate, trajectory, reward, max_rule_tokens, counter):
    def reject(reason):
        return {"accepted": False, "reason": reason, "semantic_verified": False}

    if not isinstance(candidate, dict):
        return reject("invalid_schema")
    rule = candidate.get("rule")
    if rule is None:
        return reject("abstained")
    if not isinstance(rule, str) or not rule.strip():
        return reject("empty_or_invalid_rule")
    if counter.count(rule) > max_rule_tokens:
        return reject("oversize")
    kind = candidate.get("kind")
    evidence = candidate.get("evidence")
    if not isinstance(kind, str) or kind not in KINDS or not isinstance(evidence, list) or not evidence:
        return reject("missing_evidence_or_kind")
    source = {transition["step"]: transition for transition in trajectory}
    steps = []
    selected = []
    for reference in evidence:
        if not isinstance(reference, dict):
            return reject("invalid_reference")
        step = reference.get("step")
        quote = reference.get("quote")
        if not isinstance(step, int) or isinstance(step, bool) or step not in source or step in steps:
            return reject("unknown_or_duplicate_step")
        if not isinstance(quote, str) or not quote.strip() or quote not in source[step]["next_observation"]:
            return reject("quote_not_in_feedback")
        steps.append(step)
        selected.append(source[step])
    failed_terminal = any(item["done"] and not item["success"] for item in selected)
    if kind != "negative_outcome" and failed_terminal:
        return reject("positive_evidence_from_failed_terminal")
    if kind == "task_success" and (reward != 1 or not any(item["success"] for item in selected)):
        return reject("task_success_not_observed")
    if kind == "negative_outcome" and not failed_terminal:
        return reject("negative_terminal_not_observed")
    if kind == "progress":
        improved = False
        for item in selected:
            previous = item.get("previous_score")
            current = item.get("score")
            if isinstance(previous, (int, float)) and isinstance(current, (int, float)) and math.isfinite(previous) and math.isfinite(current) and current > previous:
                improved = True
        if not improved:
            return reject("score_progress_not_observed")
    return {"accepted": True, "reason": "provenance_checks_passed", "kind": kind, "source_steps": steps, "semantic_verified": False, "warning": "Exact source/outcome consistency only; natural-language implication and generalization remain unverified."}
