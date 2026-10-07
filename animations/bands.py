"""Finite-N sampling bands for the running estimates of <phibar^2> and U,
from independent draws of the quartic (Starobinsky-Yokoyama) density.
The zero mode is redrawn exactly every trajectory (tau = 0.5), so successive
values are independent to good accuracy."""
import numpy as np

QS = (2.5, 16.0, 84.0, 97.5)


def simulate(K, pred_var, reps=20000, seed=11, chunk=2000):
    """Running (var, U) paths, shape (reps, K), for exp(-a x^4) scaled to pred_var."""
    from scipy.special import gamma
    rng = np.random.default_rng(seed)
    unit_var = gamma(0.75) / gamma(0.25)            # <x^2> for exp(-x^4)
    n = np.arange(1, K + 1)
    V, U = [], []
    for _ in range(reps // chunk):
        x2 = np.sqrt(rng.gamma(0.25, size=(chunk, K))) * (pred_var / unit_var)
        v = np.cumsum(x2, 1) / n
        u = 1 - (np.cumsum(x2**2, 1) / n) / (3 * v**2)
        V.append(v.astype(np.float32)); U.append(u.astype(np.float32))
    return np.concatenate(V), np.concatenate(U)


def bands(K, pred_var, npts=48, **kw):
    V, U = simulate(K, pred_var, **kw)
    N = np.unique(np.round(np.geomspace(4, K, npts)).astype(int))
    qv = np.percentile(V[:, N - 1], QS, axis=0)
    qu = np.percentile(U[:, N - 1], QS, axis=0)
    return N, qv, qu, V, U
