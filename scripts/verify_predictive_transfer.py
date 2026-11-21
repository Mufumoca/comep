import numpy as np

from predictive_transfer import CELLS, MODELS, feature, fit, load, objective, probability


def main():
    full = np.array([feature(cell) for cell in CELLS], dtype=float)
    assert np.linalg.matrix_rank(full) == 6
    assert np.array_equal(full[3, :5], full[4, :5])
    assert full[5, 1:5].tolist() == [1, 1, 1, 1]
    design = np.tile(full, (3, 1))
    labels = np.array([0, 0, 1, 1, 0, 1] * 3)
    for model, dimension in MODELS.items():
        matrix = design[:, :dimension]
        result = fit(matrix, labels)
        coefficients = np.array(result["coefficients"])
        numeric = []
        for coordinate in range(dimension):
            delta = np.zeros(dimension)
            delta[coordinate] = 1e-5
            numeric.append((objective(matrix, labels, coefficients + delta) - objective(matrix, labels, coefficients - delta)) / 2e-5)
        assert max(abs(value) for value in numeric) < 1e-6
        reversed_fit = fit(matrix[::-1], labels[::-1])
        np.testing.assert_allclose(result["coefficients"], reversed_fit["coefficients"], atol=1e-9)
    intercept = fit(np.ones((4, 1)), [0, 0, 0, 1])
    assert abs(float(probability(intercept["coefficients"][0])) - .25) < 1e-8
    assert len(load()) == 72
    print("PREDICTIVE_TRANSFER_WITNESS rank6,sharedfeatures,unpenalizedintercept,finitegradient,permutation,raw72cellreconcile; noAPI")


if __name__ == "__main__":
    main()
