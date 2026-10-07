"""Constrained simulations: the zero mode held fixed (Appendix D1 of the paper).

Writing phi = phibar + psi, with psi the nonzero modes, the constraint effective
potential V_C of the zero mode is defined by

    exp(-Omega V_C(phibar)) = int D psi exp(-S[phibar + psi]).

A constrained simulation samples psi at fixed phibar: the coefficient
c_0 = sqrt(Omega) phibar is frozen and HMC runs on the other N - 1
coefficients, with no zero-mode update. Two estimators are then measured on
each configuration.

The force, whose average is the slope of V_C ("mean force"):

    f = (1/Omega) dS/dphibar
      = m^2 phibar + lam4 phibar^3 + (3 lam4 phibar / Omega) I_2 + (lam4 / Omega) I_3,
    V_C'(phibar) = <f>,          I_k = int psi^k.

The coefficients of the action as a polynomial in c_0,

    S = S[psi] + A c_0 + B c_0^2 / 2 + lam4 c_0^4 / (4 Omega),
    A = lam4 I_3 / sqrt(Omega),   B = m^2 + 3 lam4 I_2 / Omega,

which give the curvature at the origin from a simulation at phibar = 0:

    m_R^2 = V_C''(0) = <B> - <A^2>.

Since I_2 = (B - m^2) Omega / (3 lam4), storing (A, B) is also enough to
reweight a constrained chain to another bare mass (`meanforce`).
"""

from __future__ import annotations

import numpy as np

from . import action as A_
from . import grid as G
from . import harmonics as H
from . import zeromode
from .transform import S4Transform


class FixedZeroMode:
    """Embedding of the N - 1 nonzero-mode coefficients at fixed c_0.

    It exposes the `to_grid` / `to_modes` interface of `S4Transform` for the
    reduced coefficient vector, so `action`, `grad` and `hmc.run` work on it
    unchanged: `to_grid` inserts the fixed c_0, and `to_modes` drops the
    zero-mode component of the gradient.
    """

    def __init__(self, basis, c0):
        self.base = basis
        self.c0 = c0
        self.n_modes = basis.n_modes - 1

    def full(self, c):
        """Reduced coefficients -> full coefficient vector, with c_0 inserted."""
        return np.concatenate(([self.c0], np.asarray(c)))

    def to_grid(self, c):
        return self.base.to_grid(self.full(c))

    def to_modes(self, f):
        return self.base.to_modes(f)[1:]


def context(n_max, m2, lam4, phibar):
    """Set up a constrained simulation at zero mode `phibar`.

    Returns (ctx, basis, eig, vol): the `ActionContext` of the reduced system,
    the full `S4Transform`, the eigenvalue of every mode of the full system,
    and the volume of the sphere as integrated by the grid.
    """
    grid = G.for_degree(4 * n_max)
    basis = S4Transform(grid, n_max)
    vol = float(grid.weights.sum())
    eig = H.eigenvalue(basis.shell_index.astype(float))
    wrapped = FixedZeroMode(basis, np.sqrt(vol) * phibar)
    ctx = A_.ActionContext(Y=wrapped, w=grid.weights, lam=eig[1:], m2=m2, lam4=lam4)
    return ctx, basis, eig, vol


def zero_mode_coefficients(full_coeffs, basis, w, q, lam4, vol):
    """(A, B, f) for one configuration, given its full coefficient vector.

    `q` is the full array lambda_n + m^2. Works for constrained and
    unconstrained configurations alike; in the latter, <f | phibar> binned in
    phibar is again V_C'(phibar).
    """
    a1, a2, a3, a4 = zeromode.conditional_coeffs(full_coeffs, basis, w, q, lam4, vol)
    v = full_coeffs[0]
    force = (a1 + 2 * a2 * v + 3 * a3 * v * v + 4 * a4 * v ** 3) / np.sqrt(vol)
    return a1, 2 * a2, force


def measure(samples, ctx, basis, eig, m2, lam4, vol):
    """Series of A, B and f over the stored configurations of a constrained chain."""
    ab = []
    force = []
    for c in samples:
        full = ctx.Y.full(c)
        a, b, f = zero_mode_coefficients(full, basis, ctx.w, eig + m2, lam4, vol)
        ab.append([a, b])
        force.append(f)
    ab = np.array(ab)
    return ab[:, 0], ab[:, 1], np.array(force)


def curvature(A, B):
    """m_R^2 = V_C''(0) = <B> - <A^2> from a chain at phibar = 0.

    <A> = 0 by the symmetry phi -> -phi and is imposed, not estimated.
    """
    A = np.asarray(A)
    return float(np.mean(B) - np.mean(A * A))
