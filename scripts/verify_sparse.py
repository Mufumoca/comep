import numpy as np

from comep.policy import FixedBankPolicy
from comep.sparse import SparseValidatedPolicy


def main():
    sparse = SparseValidatedPolicy([8]*8, ridge=1.)
    additive = FixedBankPolicy("normalized-additive", [8]*8, ridge=1.)
    random = np.random.default_rng(77)
    for episode in range(199):
        gates = np.where(np.arange(8)%2 == episode%2, 1., .25)
        context = 2*(episode%2)-1
        assert sparse.choose(context, gates)[0] == additive.choose(context, gates)[0]
        chosen = int(random.integers(93))
        reward = int(random.integers(2))
        sparse.observe(context, gates, chosen, reward)
        additive.observe(context, gates, chosen, reward)
        assert sparse.active_pair is None
    for episode in range(199, 420):
        sparse.observe(1, np.ones(8), int(random.integers(93)), int(random.integers(2)))
    assert len(sparse.history) == 400
    assert sparse.selection["candidate_pair"] in range(28)
    assert sparse.choose(1, np.ones(8), 0)[0] == 0
    assert sparse.selection["validation_observations"] == 200
    interacting = SparseValidatedPolicy([8, 8], ridge=1.)
    for episode in range(400):
        chosen = episode % 4
        interacting.observe(1, np.ones(2), chosen, int(chosen in (1, 2)))
    assert interacting.active_pair == 0
    assert interacting.selection["validation_pass"]
    assert interacting.candidate_fit.coefficients[-1] < 0
    assert interacting.choose(1, np.ones(2))[0] in (1, 2)
    assert interacting.choose(1, np.ones(2), 3)[0] == 3
    print("SPARSE_WITNESS cold fallback, window split, candidate bounds, positive validation path and exploration passed")


if __name__ == "__main__":
    main()
