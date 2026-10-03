"""
Mathematical transformations and landmark stabilization utilities.
"""

import numpy as np


class OneEuroFilter:
    """1€ Filter: eliminates high-frequency jitter at rest while preserving low latency during fast motion."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, x: np.ndarray, dt: float) -> np.ndarray:
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x
        dx = (x - self.x_prev) / max(dt, 1e-6)
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx_prev = a_d * dx + (1.0 - a_d) * self.dx_prev
        a = self._alpha(self.min_cutoff + self.beta * np.abs(self.dx_prev), dt)
        self.x_prev = a * x + (1.0 - a) * self.x_prev
        return self.x_prev

    def reset(self):
        self.x_prev = None
        self.dx_prev = None


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Computes 2D similarity transform matrix (Umeyama SVD) mapping src -> dst.
    Returns:
        2x3 affine matrix np.ndarray
    """
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = d_c.T @ s_c / n
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / (s_c ** 2).sum() * n
    t = mu_d - scale * R @ mu_s
    return np.hstack([scale * R, t[:, None]]).astype(np.float32)
