import numpy as np

from comep.joint import probabilities
from comep.policy import FixedBankPolicy
from round7_information import design_weights, expected_information


def main():
    policy = FixedBankPolicy("normalized-joint-soft", [8]*8)
    distributions = design_weights(policy)
    for weights in distributions.values():
        np.testing.assert_allclose(weights.sum(), 1.)
        assert np.all(weights >= 0)
    np.testing.assert_allclose(distributions["uniform-size"] @ policy.pair_membership, 1/28)
    np.testing.assert_allclose(distributions["uniform-subset"] @ policy.pair_membership, 7/93)
    np.testing.assert_allclose(distributions["balanced-pair-probes"] @ policy.pair_membership, 1/112)
    design = np.concatenate([policy.design(2*context-1, np.where(np.arange(8)%2 == context, 1., .25)) for context in (0, 1)])
    coefficients = np.linspace(-.2, .2, 38)
    probability = probabilities(design @ coefficients)
    weights = np.tile(distributions["uniform-size"]/2, 2)
    information = design.T @ ((weights*probability*(1-probability))[:, None]*design)
    finite = np.zeros_like(information)
    for index in range(38):
        perturbation = np.zeros(38)
        perturbation[index] = 1e-5
        plus = design.T @ (weights*(probabilities(design @ (coefficients+perturbation))-probability))
        minus = design.T @ (weights*(probabilities(design @ (coefficients-perturbation))-probability))
        finite[:, index] = (plus-minus)/(2e-5)
    np.testing.assert_allclose(finite, information, atol=1e-10, rtol=1e-8)
    contrast = np.zeros(38)
    contrast[10] = .25
    original = expected_information(design, probability, weights, contrast)
    scale = np.linspace(.1, 3., 38)
    np.testing.assert_allclose(probability, probabilities((design*scale) @ (coefficients/scale)), atol=1e-14)
    scaled = expected_information(design*scale, probability, weights, contrast*scale)
    np.testing.assert_allclose(original["variance_factor"], scaled["variance_factor"], rtol=1e-10)
    np.testing.assert_allclose(original["variance_factor"], contrast @ np.linalg.solve(information, contrast), rtol=1e-10)
    assert original["rank"] == 38
    confounded = expected_information(np.ones((10, 4)), np.full(10, .5), np.full(10, .1), [0, 0, 0, 1])
    assert confounded["rank"] == 1 and not confounded["identifiable"] and confounded["variance_factor"] is None
    estimable = expected_information(np.ones((10, 4)), np.full(10, .5), np.full(10, .1), [1, 1, 1, 1])
    assert estimable["identifiable"]
    np.testing.assert_allclose(estimable["variance_factor"], 4.)
    both_index = policy.subsets.index((0, 1))
    both_rows = design[[both_index, 93+both_index]]
    both = expected_information(both_rows, np.full(2, .5), np.full(2, .5), contrast)
    assert both["rank"] == 2 and not both["identifiable"]
    for name, subset_weights in distributions.items():
        assert expected_information(design, probability, np.tile(subset_weights/2, 2), contrast)["rank"] == 38, name
    print("INFORMATION_WITNESS exact distributions, finite-difference Hessian, coordinate invariance, inverse and rank checks passed")


if __name__ == "__main__":
    main()
