"""Regulated Euclidean action of lambda phi^4 on S^4, its gradient, and leapfrog.

The degrees of freedom are the real mode coefficients c_A, A = (n, L, l, m),
with n <= n_max. The free part is diagonal; the quartic term is evaluated
pseudo-spectrally on the quadrature grid (Appendix A of the paper):

    S[c] = 1/2 sum_A q_A c_A^2 + (lam4 / 4) sum_a w_a phi_a^4,   phi = Y^T c
    dS/dc_A = q_A c_A + lam4 (Y (w * phi^3))_A

with q_A = lambda_n + m^2 the free inverse propagator, lambda_n = n(n+3), and
units H = 1 (unit sphere). The potential is V = m^2 phi^2/2 + lam4 phi^4/4.

Names used throughout the package:

    lam    per-mode Laplacian eigenvalue lambda_n   (array, one entry per mode)
    lam4   quartic coupling lambda                  (float)
    m2     bare mass squared m^2                    (float)
    q      lam + m2                                 (array)
    minv   inverse of the HMC mass matrix, 1/M_n    (array)

On a grid of degree 4 n_max (`grid.for_degree(4 * n_max)`) the action and its
gradient are exact within the retained mode space and the regulated theory is
exactly SO(5) invariant. A coarser grid leaves an aliasing error that breaks
the symmetry at the percent level; see `reproduction/fig_so5.py`.

The functions below take plain arrays, so they can be called with either a
dense basis matrix Y of shape (n_modes, n_points) or a `transform.S4Transform`.
`ActionContext` bundles the arrays; `unpack` turns it into the argument tuple.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _to_grid(coeffs, basis):
    """Mode coefficients -> grid values, for a dense Y or an S4Transform."""
    if hasattr(basis, "to_grid"):
        return basis.to_grid(coeffs)
    return coeffs @ basis


def _to_modes(field, basis, w):
    """Quadrature projection Y (w * field), for a dense Y or an S4Transform.

    `S4Transform.to_modes` already folds the quadrature weights into its factor
    tables, so both branches compute the same thing.
    """
    if hasattr(basis, "to_modes"):
        return basis.to_modes(field)
    return basis @ (w * field)


@dataclass(frozen=True)
class ActionContext:
    """The arrays that define one simulation point. Use `unpack` to call the primitives.

    `Y` is either a dense (n_modes, n_points) array or a
    `transform.S4Transform`. The dense form is for validation and small n_max;
    the separable transform is what simulations use, since a dense Y is about
    5 GB at n_max = 10 while the factor tables are under 1 MB.
    """

    Y: np.ndarray          # dense basis, or an S4Transform
    w: np.ndarray          # (n_points,) quadrature weights
    lam: np.ndarray        # (n_modes,) -Laplacian eigenvalue per mode
    m2: float
    lam4: float

    @property
    def n_modes(self):
        if hasattr(self.Y, "n_modes"):
            return int(self.Y.n_modes)          # S4Transform
        return int(self.Y.shape[0])             # dense (n_modes, n_points)

    @property
    def quadratic(self):
        return self.lam + self.m2


def unpack(ctx):
    """(Y, w, q, lam4) -- the argument tuple every primitive below takes."""
    return ctx.Y, ctx.w, ctx.quadratic, ctx.lam4


def action(coeffs, basis, w, q, lam4):
    """S[c] for the mode coefficients `coeffs`."""
    quad = 0.5 * np.sum(q * coeffs**2)
    if lam4 == 0.0:
        return quad
    f = _to_grid(coeffs, basis)
    return quad + 0.25 * lam4 * np.sum(w * f**4)


def grad(coeffs, basis, w, q, lam4):
    """dS/dc_A."""
    g = q * coeffs
    if lam4 == 0.0:
        return g
    f = _to_grid(coeffs, basis)
    return g + lam4 * _to_modes(f**3, basis, w)


def hamiltonian(coeffs, mom, basis, w, q, lam4, minv):
    """S[c] + 1/2 sum_A p_A^2 / M_A, with minv = 1 / M."""
    return action(coeffs, basis, w, q, lam4) + 0.5 * np.sum(minv * mom**2)


def leapfrog(coeffs, mom, eps, n_steps, basis, w, q, lam4, minv):
    """Leapfrog integration of `n_steps` steps of size `eps` with mass matrix M = 1/minv.

    With M_A = q_A (see `default_minv`) every mode of the free theory is a
    unit-frequency oscillator, so a single step size is uniformly accurate
    across the whole spectrum (Fourier acceleration). Without it the energy
    error grows with the number of modes and with the spread of q, and the
    acceptance collapses at any usable step size.
    """
    mom = mom - 0.5 * eps * grad(coeffs, basis, w, q, lam4)
    for _ in range(n_steps - 1):
        coeffs = coeffs + eps * (minv * mom)
        mom = mom - eps * grad(coeffs, basis, w, q, lam4)
    coeffs = coeffs + eps * (minv * mom)
    mom = mom - 0.5 * eps * grad(coeffs, basis, w, q, lam4)
    return coeffs, mom


def default_minv(q, q_floor=None):
    """HMC mass matrix M = diag(max(q, q_floor)); returns 1/M. Used for Set A.

    With M = q the free theory maps onto identical unit-frequency oscillators.
    The floor matters for a light field. The zero mode has q_0 = m^2, which is
    small (or negative) in the regime of interest, while its true curvature is
    q_0 + 3 lam4 c_0^2 / Vol. Preconditioning with M_0 = q_0 would give that
    mode an effective frequency sqrt(1 + 3 lam4 c_0^2 / (q_0 Vol)), which
    diverges as m^2 -> 0 and destroys the integrator.

    By default the floor is the second smallest distinct entry of q, i.e. the
    n = 1 level: no mode is preconditioned more aggressively than the first
    non-constant one. The zero mode is then slightly under-preconditioned in
    HMC but stable, and `zeromode.update` decorrelates it exactly.
    """
    q = np.asarray(q)
    if q_floor is None:
        distinct = np.unique(q)
        cand = float(distinct[1]) if distinct.size > 1 else 0.0
        if cand <= 0.0:
            # Tuning the renormalised mass to zero makes the bare mass negative,
            # and at strong coupling several low levels can have q < 0. The floor
            # must stay positive or sqrt(1/minv) is NaN. Fall back to the lowest
            # positive curvature present.
            pos = distinct[distinct > 0.0]
            cand = float(pos[0]) if pos.size else 1.0
        q_floor = cand
    return 1.0 / np.maximum(q, q_floor)


def renormalised_minv(lam, m2_ren=0.0, floor=None):
    """HMC mass matrix M = diag(max(lambda_n + m_R^2, floor)); returns 1/M. Used for Set B.

    `default_minv` is built from the bare inverse propagator q = lambda_n + m^2.
    With the mass tuned to m_R^2 = 0 the bare mass is large and negative,
    m^2 ~ -3 lam4 G'(0), so q_1 = 4 + m^2 passes through zero at
    lam4 = 4/(3 G'(0)) (5.6, 3.2, 2.0, 1.4 at n_max = 4, 6, 8, 10). There the
    floor of `default_minv` collapses, the low modes are given a near-zero mass,
    and the integrator goes unstable.

    The cure is to precondition with the renormalised curvature. After the
    tadpole is absorbed, the quadratic form seen by mode n is lambda_n + m_R^2,
    not lambda_n + m^2: the bare mass is large only because it cancels a loop.
    Flooring at lambda_1 = 4 keeps the constant mode, whose renormalised
    curvature is zero by construction, at a finite frequency.

    The mass matrix changes only the momentum distribution, never the target
    distribution, so chains run with either choice sample the same theory.
    """
    lam = np.asarray(lam, dtype=float)
    if floor is None:
        floor = 4.0                      # lambda_1 = n(n+3) at n = 1, in H = 1
    return 1.0 / np.maximum(lam + m2_ren, floor)
