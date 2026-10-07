"""Exact conditional (Gibbs) update for the constant mode on S^4.

Why this exists
---------------
For a light field the constant mode on S^4 carries the non-perturbative
physics: with the other modes switched off its distribution is exactly the
Starobinsky-Yokoyama (SY) weight `exp(-Vol(S^4) V(phi))`. Two things make plain
HMC a poor sampler for it:

- Its free curvature is `q_0 = m^2`, which is *small* precisely in the regime of
  interest, so it is the slowest mode in the chain.
- Mass preconditioning uses `M_A = q_A`, so as `m^2 -> 0` the preconditioner
  `1/q_0` diverges and the drift step blows up. `action.default_minv` floors it
  for this reason; the floor keeps the integrator stable but does nothing for
  decorrelation.

Conditioned on every other mode, the constant mode has a one-dimensional
density that can be sampled directly, which removes it from the autocorrelation
budget entirely.

The conditional density
-----------------------
Write the field as `f_a = u c_0 + r_a`, where `u = Y_0 = Vol^{-1/2}` is the
(constant) zero-mode harmonic and `r` is the contribution of all other modes.
Expanding the quartic and dropping `c_0`-independent terms,

    S(c_0 | rest) = a4 c_0^4 + a3 c_0^3 + a2 c_0^2 + a1 c_0

    a4 = lam4 / (4 Vol)
    a3 = lam4 u^3 M1
    a2 = q_0 / 2 + (3/2) lam4 u^2 M2
    a1 = lam4 u M3            with  Mk = sum_a w_a r_a^k

`a4 > 0` for `lam4 > 0`, so the density is normalisable; it may be bimodal when
`a2 < 0`, which is exactly the symmetry-broken case the SY distribution
describes, and is why a single Gaussian proposal is not good enough.

Sampling
--------
The density is tabulated on an adaptively widened interval, a candidate is
drawn by inverting the trapezoid CDF of the table, and the move is then
accepted by an independence Metropolis-Hastings step against that same
proposal. Inverting the trapezoid CDF draws from the piecewise-constant
density (p_i + p_{i+1})/2 on each cell, and it is this density, not a linear
interpolation of the table, that enters the MH ratio. The MH step makes the
update **exact regardless of tabulation error**: the tabulation only affects
the acceptance rate, which in practice sits at ~1.

Composing this Gibbs step with HMC preserves the target, since each kernel
preserves it separately.
"""

from __future__ import annotations

import numpy as np

ZERO_MODE_INDEX = 0  # labels are ordered (n, L, l, m); (0,0,0,0) comes first


def conditional_coeffs(coeffs, basis, w, q, lam4, vol):
    """Quartic coefficients (a1, a2, a3, a4) of S(c_0 | rest).

    Costs one `to_grid` plus three weighted reductions, i.e. a small fraction of
    one HMC trajectory.
    """
    from .action import _to_grid

    rest = np.array(coeffs, dtype=float)
    rest[ZERO_MODE_INDEX] = 0.0
    r = _to_grid(rest, basis)

    u = 1.0 / np.sqrt(vol)
    m1 = float(np.sum(w * r))
    m2 = float(np.sum(w * r**2))
    m3 = float(np.sum(w * r**3))

    a4 = lam4 / (4.0 * vol)
    a3 = lam4 * u**3 * m1
    a2 = 0.5 * q[ZERO_MODE_INDEX] + 1.5 * lam4 * u**2 * m2
    a1 = lam4 * u * m3
    return a1, a2, a3, a4


def _quartic(c, a1, a2, a3, a4):
    return ((a4 * c + a3) * c + a2) * c * c + a1 * c


def _bracket(a1, a2, a3, a4, depth=40.0):
    """Interval containing essentially all of exp(-S), by stationary points."""
    if a4 <= 0.0:
        if a2 <= 0.0:
            raise ValueError("conditional density is not normalisable")
        c0 = -a1 / (2.0 * a2)
        halfwidth = np.sqrt(depth / a2)
        return c0 - halfwidth, c0 + halfwidth

    roots = np.roots([4.0 * a4, 3.0 * a3, 2.0 * a2, a1])
    real = np.real(roots[np.abs(np.imag(roots)) < 1e-9 * (1.0 + np.abs(roots))])
    if real.size == 0:
        real = np.array([0.0])
    c_min = real[np.argmin(_quartic(real, a1, a2, a3, a4))]

    # Width from whichever term dominates at the required depth.
    w_quartic = (depth / a4) ** 0.25
    w_quad = np.sqrt(depth / a2) if a2 > 0 else 0.0
    halfwidth = max(w_quartic, w_quad, np.ptp(real) if real.size > 1 else 0.0) * 1.5 + 1e-3
    return c_min - halfwidth, c_min + halfwidth


def _tabulate(a1, a2, a3, a4, n_grid=2048, max_widen=8):
    """Grid, normalised density and CDF for exp(-S), widened until the tails die."""
    lo, hi = _bracket(a1, a2, a3, a4)
    for _ in range(max_widen):
        c = np.linspace(lo, hi, n_grid)
        s = _quartic(c, a1, a2, a3, a4)
        p = np.exp(-(s - s.min()))
        if max(p[0], p[-1]) <= 1e-13 * p.max():
            break
        mid = 0.5 * (lo + hi)
        lo, hi = mid + 1.6 * (lo - mid), mid + 1.6 * (hi - mid)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(c))])
    cdf /= cdf[-1]
    return c, p, cdf


def sample_conditional(rng, c_current, a1, a2, a3, a4, n_grid=2048):
    """One exact draw of c_0 from its conditional, via tabulated-proposal MH.

    Returns (c_new, accepted). The proposal is close to perfect, so acceptance
    is ~1; the MH step is what makes tabulation error irrelevant to correctness.
    """
    c, p, cdf = _tabulate(a1, a2, a3, a4, n_grid=n_grid)
    cand = float(np.interp(rng.random(), cdf, c))

    # Proposal density: the inverse-CDF draw samples the piecewise-constant
    # density (p_i + p_{i+1})/2 of the cell (unnormalised is fine; the
    # normalisation cancels). Outside the table the proposal density is zero.
    def g(x):
        if x <= c[0] or x >= c[-1]:
            return 0.0
        i = int(np.searchsorted(c, x, side="right")) - 1
        return 0.5 * (p[i] + p[i + 1])

    g_cand = g(cand)
    g_cur = g(c_current)
    if g_cand <= 0.0:
        return c_current, False
    if g_cur <= 0.0:
        return cand, True

    s_cand = _quartic(cand, a1, a2, a3, a4)
    s_cur = _quartic(c_current, a1, a2, a3, a4)
    log_alpha = -(s_cand - s_cur) + (np.log(g_cur) - np.log(g_cand))
    if np.log(rng.random()) < log_alpha:
        return cand, True
    return c_current, False


def update(rng, coeffs, basis, w, q, lam4, vol, n_grid=2048):
    """Replace coeffs[0] by an exact draw from its conditional. In place."""
    if lam4 == 0.0:
        # Free theory: the conditional is exactly N(0, 1/q_0).
        coeffs[ZERO_MODE_INDEX] = rng.standard_normal() / np.sqrt(q[ZERO_MODE_INDEX])
        return True
    a1, a2, a3, a4 = conditional_coeffs(coeffs, basis, w, q, lam4, vol)
    new, acc = sample_conditional(rng, coeffs[ZERO_MODE_INDEX], a1, a2, a3, a4, n_grid)
    coeffs[ZERO_MODE_INDEX] = new
    return acc


def sy_log_density(c0, m2, lam4, vol):
    """The Starobinsky-Yokoyama weight, as a function of the zero-mode coefficient.

    With only the constant mode present, phi = c_0 / sqrt(Vol) and

        S = Vol * V(phi) = Vol * (m^2 phi^2 / 2 + lam4 phi^4 / 4)
          = m^2 c_0^2 / 2 + lam4 c_0^4 / (4 Vol),

    which is the a4/a2 pair above with M1 = M2 = M3 = 0.
    """
    return -(0.5 * m2 * c0**2 + 0.25 * lam4 * c0**4 / vol)
