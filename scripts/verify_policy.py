import numpy as np

from comep.joint import JointFit
from comep.policy import FixedBankPolicy


def main():
    for method in FixedBankPolicy.methods:
        policy = FixedBankPolicy(method, [8]*8)
        assert len(policy.subsets) == 93
        assert policy.design(1, np.ones(8)).shape == (93, 38 if policy.joint else 10)
        assert policy.choose(1, np.ones(8))[0] == 0
        for episode in range(420):
            chosen, _ = policy.choose(1, np.ones(8), episode % 93)
            assert chosen == episode % 93
            policy.observe(1, np.ones(8), chosen, episode % 2)
        assert len(policy.history) == 400
        if method != "individual":
            assert policy.fit is not None and policy.fit.converged
            assert all(state == "unknown" for state in policy.pair_evidence(np.zeros(8)))
        assert max(sum(policy.costs[list(subset)]) for subset in policy.subsets) <= 24
    soft = FixedBankPolicy("normalized-joint-soft", [8, 8], limit=2, budget=16)
    hard = FixedBankPolicy("normalized-joint-hard", [8, 8], limit=2, budget=16)
    fit = JointFit(np.array([0., 0., 2., 2., -1.]), np.ones(5), np.eye(5)*.0001,
                   np.eye(5), np.full(5, 100), 5, 1, 0., True)
    soft.fit = fit
    hard.fit = fit
    assert soft.subsets[soft.choose(1, np.ones(2))[0]] == (0, 1)
    assert hard.subsets[hard.choose(1, np.ones(2))[0]] == (0,)
    assert hard.subsets[hard.choose(1, np.ones(2), 3)[0]] == (0, 1)
    fit.converged = False
    assert hard.subsets[hard.choose(1, np.ones(2))[0]] == (0, 1)
    print("POLICY_WITNESS candidates/budgets, own-observation refits, window expiry, evidence exclusion and exploration override passed")


if __name__ == "__main__":
    main()
