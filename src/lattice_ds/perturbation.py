"""Analytic reference quantities: loop sums, one-loop tuning, Starobinsky-Yokoyama.

Conventions (fixed by `action.py`; unit sphere, H = 1, Vol = Omega = 8 pi^2/3):

    S = 1/2 sum_A (lambda_n + m^2) c_A^2 + (lam4 / 4) int phi^4
    V(phi) = m^2 phi^2 / 2 + lam4 phi^4 / 4
    phibar = c_0 / sqrt(Omega)          (the zero mode, a constant field)

Everything is written in this convention rather than quoted from the
literature, because the normalisation of the quartic (lam/4 against lam/4!)
differs between sources.
"""

from __future__ import annotations

import math

import numpy as np

from .grid import VOL_S4
from .harmonics import degeneracy, eigenvalue

#: sqrt(lam4) <phibar^2> for the massless Starobinsky-Yokoyama distribution, 0.13176.
SY_PHIBAR2 = math.gamma(0.75) / math.gamma(0.25) * math.sqrt(4.0 / VOL_S4)

#: Binder cumulant U = 1 - <phibar^4>/(3 <phibar^2>^2) of the same distribution, 0.27052.
SY_BINDER = 1.0 - math.gamma(0.25) ** 2 / (12.0 * math.gamma(0.75) ** 2)


# ---------------------------------------------------------------- loop sums

def loop_sum(k, n_max):
    """S_k(n_max) = sum_{n=1}^{n_max} d_n / lambda_n^k (the zero mode is excluded)."""
    return float(sum(int(degeneracy(n)) / float(eigenvalue(n)) ** k
                     for n in range(1, n_max + 1)))


def screening_coefficient(n_max):
    """u = 9 S_2 / Omega, the one-loop coefficient of the coupling renormalisation.

    lambda_R = lambda (1 - u lambda) at one loop; in the paper's Fig. 3,
    x = u lambda and y = u lambda_R. u grows like (9 / 8 pi^2) ln n_max.
    """
    return 9.0 * loop_sum(2, n_max) / VOL_S4


def coincident_propagator(n_max, m2):
    """G(0) = <phi(x)^2> of the free theory at cutoff n_max, zero mode INCLUDED.

        G(0) = (1/Vol) sum_{n<=n_max} d_n / (n(n+3) + m^2)

    It diverges like n_max^2 as the cutoff is removed. It is also infrared
    divergent as m^2 -> 0, since the n = 0 term is 1/(Vol m^2): use it only for
    a heavy field. Anything concerning a light field or the zero-mode
    distribution needs `coincident_propagator_nonzero`.
    """
    n = np.arange(n_max + 1)
    return float(np.sum(degeneracy(n) / (eigenvalue(n) + m2)) / VOL_S4)


def coincident_propagator_nonzero(n_max, m2):
    """G'(0): the coincident propagator with the constant mode EXCLUDED.

    This is the loop that dresses the zero mode: the constant mode is treated
    exactly and the others perturbatively. Including the constant mode in the
    loop that corrects its own potential would double count it. The choice of
    sum range is the choice between the standard effective potential (n >= 0)
    and the constraint effective potential (n >= 1) used in the paper.

    At m^2 = 0 this is S_1 / Omega.
    """
    n = np.arange(1, n_max + 1)
    return float(np.sum(degeneracy(n) / (eigenvalue(n) + m2)) / VOL_S4)


def one_loop_bare_mass(n_max, lam4):
    """Bare mass m^2 = -3 lam4 G'(0) at which m_R^2 vanishes at one loop (Set A)."""
    return -3.0 * lam4 * coincident_propagator_nonzero(n_max, 0.0)


def one_loop_mass_shift(n_max, m2, lam4):
    """delta m^2 = 3 lam4 G(0), the one-loop tadpole for a heavy field.

    Writing phi = phibar + dphi in (lam4/4) phi^4, the term quadratic in phibar
    is (3 lam4 / 2) phibar^2 dphi^2; averaging dphi^2 -> G(0) gives
    m^2 -> m^2 + 3 lam4 G(0).
    """
    return 3.0 * lam4 * coincident_propagator(n_max, m2)


# ---------------------------------------------------------- free propagator

def gegenbauer_ratio(n_max, x, nu=1.5):
    """C_n^nu(x) / C_n^nu(1) for n = 0..n_max, by upward recurrence.

    With nu = 3/2 this is the angular dependence of a level on S^4, through the
    addition theorem sum_I Y_nI(x) Y_nI(y) = d_n/Vol * C_n^{3/2}(x.y)/C_n^{3/2}(1).
    """
    C = np.zeros(n_max + 1)
    C1 = np.zeros(n_max + 1)
    C[0] = C1[0] = 1.0
    if n_max >= 1:
        C[1] = 2.0 * nu * x
        C1[1] = 2.0 * nu
    for n in range(2, n_max + 1):
        C[n] = (2.0 * (n + nu - 1.0) * x * C[n - 1] - (n + 2.0 * nu - 2.0) * C[n - 2]) / n
        C1[n] = (2.0 * (n + nu - 1.0) * C1[n - 1] - (n + 2.0 * nu - 2.0) * C1[n - 2]) / n
    return C / C1


def propagator_closed_form(theta, m2):
    """Continuum free propagator on S^4 at geodesic angle theta.

        G(theta) = Gamma(D+) Gamma(D-) / (4 pi)^2 * 2F1(D+, D-; 2; cos^2(theta/2)),
        D+- = 3/2 +- sqrt(9/4 - m^2).

    Valid for 0 < m^2 < 9/4 (real D+-).
    """
    from scipy.special import gamma, hyp2f1

    s = np.sqrt(2.25 - m2)
    dp, dm = 1.5 + s, 1.5 - s
    pref = gamma(dp) * gamma(dm) / ((4.0 * np.pi) ** 2 * gamma(2.0))
    return pref * hyp2f1(dp, dm, 2.0, np.cos(theta / 2.0) ** 2)


def propagator_mode_difference(theta, m2a, m2b, n_max=400_000):
    """G(theta; m2a) - G(theta; m2b) from the mode sum up to n_max.

    The individual sums diverge at coincident points; the difference between
    two masses converges absolutely, which makes it a usable numerical check of
    the spectral data (eigenvalues, degeneracies, addition theorem) against the
    closed form.
    """
    n = np.arange(n_max + 1)
    r = gegenbauer_ratio(n_max, np.cos(theta))
    d = degeneracy(n).astype(float)
    lam = eigenvalue(n).astype(float)
    return float(np.sum(d / VOL_S4 * (1.0 / (lam + m2a) - 1.0 / (lam + m2b)) * r))


# ------------------------------------------- mass from the mode variances

def effective_mass_from_shell_power(shell_power, n_max):
    """Per-level estimate m^2_eff(n) = 1/<c_A^2>_n - lambda_n.

    At one loop the interaction acts as a pure mass shift, so these agree across
    levels. Disagreement means the correction is not a simple mass shift at the
    coupling in question.
    """
    n = np.arange(n_max + 1)
    return 1.0 / np.asarray(shell_power[: n_max + 1]) - eigenvalue(n)


def effective_mass_weighted(shell_power, n_max, n_samples, tau=1.0, n_min=1):
    """Inverse-variance combination of the per-level estimates, with an error.

    For level n, <c^2> has relative error sqrt(2 tau / (d_n N)), so m^2_eff
    inherits an absolute error (lambda_n + m^2_eff) times that.

    `n_min` defaults to 1: the zero mode is excluded. It is a single degree of
    freedom whose variance is shifted non-perturbatively (that shift is the SY
    physics), and it would dominate the weight, so leaving it in makes a
    perturbative mass measurement come out systematically high.
    """
    n = np.arange(n_min, n_max + 1)
    est = effective_mass_from_shell_power(shell_power, n_max)[n_min:]
    rel = np.sqrt(2.0 * tau / (degeneracy(n) * n_samples))
    sig = (eigenvalue(n) + est) * rel
    w = 1.0 / sig**2
    mean = float(np.sum(w * est) / np.sum(w))
    err = float(1.0 / np.sqrt(np.sum(w)))
    return mean, err, est, sig


# ------------------------------------------------- zero-mode distributions

def _zero_mode_span(m2, lam4, vol):
    """Half-width of the c_0 integration range: wide enough that the density is
    numerically dead at the ends for a positive, zero or negative bare mass."""
    scale = max((4.0 * vol / max(lam4, 1e-30)) ** 0.25,          # quartic width
                np.sqrt(40.0 / m2) if m2 > 0 else 0.0,            # Gaussian width
                np.sqrt(-m2 * vol / max(lam4, 1e-30)) if m2 < 0 else 0.0)   # position of the minima
    return 8.0 * max(scale, 1.0)


def _check_tails(p, name):
    if max(p[0], p[-1]) > 1e-12 * p.max():
        raise ValueError(f"{name}: the density is not negligible at the ends of the "
                         "integration range; pass a larger span")


def sy_moments(m2, lam4, vol=VOL_S4, n_grid=400001, span=None):
    """Moments of the Starobinsky-Yokoyama zero-mode distribution.

    With only the constant mode present, phi = c_0 / sqrt(Vol) and

        -log P(c_0) = Vol * V(phi) = m^2 c_0^2 / 2 + lam4 c_0^4 / (4 Vol),

    which is the SY weight exp(-(8 pi^2 / 3 H^4) V(phi)). Computed by numerical
    integration, so the convention cannot drift from the action.

    Returns a dict: `phi2`, `phi4` (moments of phibar), `c0_2` (of the
    coefficient), and the tabulated density `c0_grid`, `pdf_c0`.
    """
    if span is None:
        span = _zero_mode_span(m2, lam4, vol)
    c = np.linspace(-span, span, n_grid)
    logp = -(0.5 * m2 * c**2 + 0.25 * lam4 * c**4 / vol)
    p = np.exp(logp - logp.max())
    _check_tails(p, "sy_moments")
    norm = np.trapezoid(p, c)
    phi = c / np.sqrt(vol)
    return {
        "phi2": float(np.trapezoid(p * phi**2, c) / norm),
        "phi4": float(np.trapezoid(p * phi**4, c) / norm),
        "c0_2": float(np.trapezoid(p * c**2, c) / norm),
        "c0_grid": c,
        "pdf_c0": p / norm,
    }


def sy_phi2_massless(lam4, vol=VOL_S4):
    """Closed form of <phibar^2> under SY at m^2 = 0: 0.13176 / sqrt(lam4).

    P ~ exp(-a c_0^4) with a = lam4 / (4 Vol), so <c_0^2> = Gamma(3/4)/Gamma(1/4)
    * a^{-1/2} and <phibar^2> = <c_0^2> / Vol. The 1/sqrt(lam4) scaling is
    non-perturbative: no order of perturbation theory produces it.
    """
    from scipy.special import gamma

    a = lam4 / (4.0 * vol)
    c0_2 = gamma(0.75) / gamma(0.25) / np.sqrt(a)
    return float(c0_2 / vol)


def one_loop_zero_mode_log_density(c0, n_max, m2, lam4, vol=VOL_S4):
    """log P(c_0) with the nonzero modes integrated out at one loop.

        -log P = Vol V(c_0/sqrt(Vol))
                 + (1/2) sum_{n>=1} d_n log[(lambda_n + M^2(c_0)) / (lambda_n + m^2)]

    with M^2(c_0) = V'' = m^2 + 3 lam4 c_0^2 / Vol the curvature in the constant
    background. The determinant depends on c_0 and is not merely a mass shift.
    This is the dashed curve of Fig. 1(a) of the paper.

    Returned unnormalised (up to an additive constant).
    """
    c0 = np.atleast_1d(np.asarray(c0, dtype=float))
    n = np.arange(1, n_max + 1)
    deg = degeneracy(n).astype(float)
    lam_n = eigenvalue(n).astype(float)

    tree = 0.5 * m2 * c0**2 + 0.25 * lam4 * c0**4 / vol
    curv = m2 + 3.0 * lam4 * c0[:, None] ** 2 / vol
    det = 0.5 * np.sum(deg * np.log((lam_n + curv) / (lam_n + m2)), axis=1)
    return -(tree + det)


def one_loop_zero_mode_moments(n_max, m2, lam4, vol=VOL_S4, n_grid=200001,
                               span=None):
    """<c_0^2> and <c_0^4> from the one-loop zero-mode effective potential."""
    if span is None:
        span = _zero_mode_span(m2, lam4, vol)
    c = np.linspace(-span, span, n_grid)
    logp = one_loop_zero_mode_log_density(c, n_max, m2, lam4, vol)
    p = np.exp(logp - logp.max())
    _check_tails(p, "one_loop_zero_mode_moments")
    norm = np.trapezoid(p, c)
    return {
        "c0_2": float(np.trapezoid(p * c**2, c) / norm),
        "c0_4": float(np.trapezoid(p * c**4, c) / norm),
        "c0_grid": c,
        "pdf_c0": p / norm,
    }
