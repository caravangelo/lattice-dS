"""Error estimation for Monte Carlo time series.

Small, dependency-free helpers used by the analysis: integrated autocorrelation
time, the error of a mean, the Binder cumulant with a blocked-jackknife error,
and block-bootstrap resampling indices.
"""

from __future__ import annotations

import math

import numpy as np


def tau_int(x, cmax=5000):
    """Integrated autocorrelation time with an automatic window.

    tau = 1/2 + sum_{t=1}^{W} rho(t), with the window W the first lag at which
    W >= 6 tau (the self-consistent truncation of Madras-Sokal and Wolff). The
    autocorrelation function is computed by FFT. Returns at least 0.5, the
    value for uncorrelated data.
    """
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    if n < 4:
        return 0.5
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n].real
    if ac[0] <= 0:
        return 0.5
    ac /= ac[0]
    t = 0.5
    for w in range(1, min(cmax, n)):
        t += ac[w]
        if w >= 6 * t:
            break
    return max(t, 0.5)


def tau_int_first_negative(x, wmax=300):
    """Integrated autocorrelation time summed up to the first negative rho(t).

    A cruder window than `tau_int`, used for the mean force of the Set A
    constrained chains (Fig. 3, open symbols), whose autocorrelation is short.
    """
    x = np.asarray(x, float) - np.mean(x)
    v = np.mean(x * x)
    t = 0.5
    if v == 0:
        return t
    for k in range(1, wmax):
        r = np.mean(x[:-k] * x[k:]) / v
        if r < 0:
            break
        t += r
    return t


def error_of_mean(x, tau):
    """Standard error of the mean of `x` for integrated autocorrelation time `tau`."""
    n_eff = max(len(x) / (2.0 * tau), 1.0)
    return float(np.std(x, ddof=1) / np.sqrt(n_eff))


def binder(c):
    """Binder cumulant U = 1 - <c^4> / (3 <c^2>^2); 0 for a Gaussian."""
    c2 = np.mean(c ** 2)
    return 1 - np.mean(c ** 4) / (3 * c2 ** 2)


def binder_jackknife_error(c, n_blocks=100):
    """Blocked-jackknife error of the Binder cumulant, `n_blocks` blocks."""
    nb = n_blocks
    n = len(c) // nb * nb
    b = c[:n].reshape(nb, -1)
    s2 = (b ** 2).sum(1)
    s4 = (b ** 4).sum(1)
    T2, T4 = s2.sum(), s4.sum()
    m = n - n // nb
    U = 1 - ((T4 - s4) / m) / (3 * ((T2 - s2) / m) ** 2)
    return math.sqrt((nb - 1) / nb * ((U - U.mean()) ** 2).sum())


def block_bootstrap_indices(rng, n, block=500):
    """Indices of one block-bootstrap resample of a series of length `n`.

    The series is cut into n // block consecutive blocks (a remainder at the end
    is dropped) and that many blocks are drawn with replacement.
    """
    nb = n // block
    return (rng.integers(nb, size=nb)[:, None] * block + np.arange(block)).reshape(-1)
