from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import tempfile

from collect_diverse_experience import prepare, validate, SCOPE
from comep.collection import run_collection
from comep.factorial import CostGuardClient
from verify_collection import MockClient, MockEnvironment


def main():
    packet = prepare()
    approval = {"scope": SCOPE, "status": "approved", "operator_authorized": True,
        "credential_reuse_authorized": True, "credential_reuse_authorization_source": "explicit_user_instruction",
        "model": packet["model"], "pricing_source": "fixture", "pricing_checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "prior_spent_and_other_reserved_usd": 1.1868249, **packet["budget_proposal"]}
    review = {"review_scope": SCOPE, "code_protocol_ready": True}
    validate(packet, approval, review, True)
    for prior in (None, True, -1, float("nan"), float("inf"), 4.1, 5):
        invalid = {**approval, "prior_spent_and_other_reserved_usd": prior}
        try:
            validate(packet, invalid, review, True)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid budget accepted")
    for field, value in (("scope", "six_train_experience_collection_v2"), ("max_calls", 247), ("max_output_tokens", 256)):
        try:
            validate(packet, {**approval, field: value}, review, True)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong approval accepted")
    changed = deepcopy(packet)
    changed["stages"]["experience_train"][0]["variation"] = 1
    try:
        validate(changed, approval, review, True)
    except ValueError:
        pass
    else:
        raise AssertionError("source task change accepted")
    with tempfile.TemporaryDirectory() as temporary:
        for mode in ("valid", "invalid", "abstain"):
            environments = []
            client = MockClient(mode=mode)

            def factory(tasks):
                environment = MockEnvironment(tasks)
                environments.append(environment)
                return environment

            result = run_collection(packet, CostGuardClient(client, .40), factory, Path(temporary) / mode)
            assert result["completed_episodes"] == 6 and client.calls == 12
            assert all(environment.closed for environment in environments)
            assert all("Memories:" not in str(messages) and "ORACLE_SENTINEL" not in str(messages) for purpose, messages in client.requests)
    print("DIVERSE_COLLECTION_WITNESS 36mockcalls, invalid/abstain retained, global budget, packet and scope guards, no memories/oracle, closures; noAPI")


if __name__ == "__main__":
    main()
