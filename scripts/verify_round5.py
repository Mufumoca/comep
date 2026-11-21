import numpy as np

from comep.policy import FixedBankPolicy
from round5_diagnostic import draw_stream, logloss


def main():
    policy = FixedBankPolicy("normalized-joint-soft", [8]*8)
    first = draw_stream(901, policy)
    second = draw_stream(901, policy)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert first["heldout_uniforms"].shape == (4, 2, 93)
    assert set(first["sizes"]) == {0, 1, 2, 3}
    assert all(len(policy.subsets[index]) == size for index, size in zip(first["choices"], first["sizes"]))
    np.testing.assert_allclose(logloss(np.array([.5, .5]), np.array([0, 1])), np.log(2))
    assert np.isfinite(logloss(np.array([0., 1.]), np.array([1, 0])))
    print("ROUND5_WITNESS shared draws deterministic, subset-size support, heldout shape, logloss passed")


if __name__ == "__main__":
    main()
