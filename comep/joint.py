from dataclasses import dataclass

import numpy as np


def probabilities(latent):
    return 1.0 / (1.0 + np.exp(-np.clip(latent, -40, 40)))


@dataclass
class JointFit:
    coefficients: np.ndarray
    scales: np.ndarray
    covariance: np.ndarray
    row_projector: np.ndarray
    exposures: np.ndarray
    rank: int
    iterations: int
    gradient_norm: float
    converged: bool

    def predict(self, design):
        return probabilities(np.asarray(design, dtype=float) @ self.coefficients)

    def contrast(self, contrast, threshold=0.2, z_value=1.96, minimum_exposures=8):
        contrast = np.asarray(contrast, dtype=float)
        if contrast.shape != self.coefficients.shape or not np.isfinite(contrast).all() or not np.any(contrast):
            raise ValueError("contrast must be a finite nonzero coefficient vector")
        if threshold < 0 or z_value < 0 or minimum_exposures < 1:
            raise ValueError("invalid declaration settings")
        normalized = contrast / self.scales
        residual = float(np.linalg.norm(normalized - self.row_projector @ normalized) / np.linalg.norm(normalized))
        mean = float(contrast @ self.coefficients)
        standard_error = float(np.sqrt(max(0.0, normalized @ self.covariance @ normalized)))
        upper = mean + z_value * standard_error
        exposed = int(np.min(self.exposures[contrast != 0]))
        identifiable = residual <= 1e-8
        if not self.converged or not identifiable or exposed < minimum_exposures:
            status = "unknown"
        else:
            status = "negative" if upper < -threshold else "not_declared"
        return {"status": status, "mean": mean, "approximate_standard_error": standard_error, "approximate_upper": upper, "identifiable": identifiable, "rowspace_relative_residual": residual, "minimum_coordinate_exposures": exposed, "interpretation": "Ridge-Laplace approximation; not a calibrated confidence bound or causal certificate."}


def fit_joint(design, rewards, normalize=True, ridge=0.1, max_iterations=100, tolerance=1e-8):
    design = np.asarray(design, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    if design.ndim != 2 or not design.shape[0] or not design.shape[1] or rewards.shape != (design.shape[0],):
        raise ValueError("nonempty 2D design and matching rewards required")
    if not np.isfinite(design).all() or not np.isfinite(rewards).all() or not np.all(np.isin(rewards, [0, 1])):
        raise ValueError("finite design and binary rewards required")
    if not np.isfinite(ridge) or ridge <= 0 or max_iterations < 1 or tolerance <= 0:
        raise ValueError("positive finite ridge, iteration budget and tolerance required")
    scales = np.sqrt(np.mean(design**2, axis=0)) if normalize else np.ones(design.shape[1])
    scales[scales == 0] = 1.0
    normalized = design / scales
    coefficients = np.zeros(design.shape[1])
    identity = np.eye(design.shape[1])

    def loss(values):
        latent = normalized @ values
        return float(np.sum(np.logaddexp(0, latent) - rewards * latent) + ridge * np.dot(values, values) / 2)

    iteration = 0
    for iteration in range(1, max_iterations + 1):
        predicted = probabilities(normalized @ coefficients)
        gradient = normalized.T @ (predicted - rewards) + ridge * coefficients
        if np.linalg.norm(gradient, ord=np.inf) <= tolerance:
            break
        curvature = normalized.T @ ((predicted * (1-predicted))[:, None] * normalized) + ridge * identity
        direction = np.linalg.solve(curvature, gradient)
        previous = loss(coefficients)
        step = 1.0
        accepted = False
        while step >= 2**-30:
            candidate = coefficients - step * direction
            if loss(candidate) <= previous - 1e-4 * step * float(gradient @ direction) + 1e-12:
                coefficients = candidate
                accepted = True
                break
            step /= 2
        if not accepted:
            break
    predicted = probabilities(normalized @ coefficients)
    gradient = normalized.T @ (predicted - rewards) + ridge * coefficients
    curvature = normalized.T @ ((predicted * (1-predicted))[:, None] * normalized) + ridge * identity
    gradient_norm = float(np.linalg.norm(gradient, ord=np.inf))
    _, singular, right = np.linalg.svd(normalized, full_matrices=False)
    cutoff = max(normalized.shape) * np.finfo(float).eps * singular[0] if singular.size else 0
    rank = int(np.sum(singular > cutoff))
    row_projector = right[:rank].T @ right[:rank]
    return JointFit(coefficients/scales, scales, np.linalg.inv(curvature), row_projector, np.count_nonzero(design, axis=0), rank, iteration, gradient_norm, gradient_norm <= max(tolerance, 1e-6))
