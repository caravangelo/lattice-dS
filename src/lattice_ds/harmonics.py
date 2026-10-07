"""Real hyperspherical harmonics on S^4 and the mode/grid transforms.

The harmonics factorise along the chain S^4 > S^3 > S^2 > S^1:

    Y_{n L l m}(chi, psi, theta, varphi)
        = R4_{n L}(chi) * R3_{L l}(psi) * Y_{l m}(theta, varphi)

    R4_{n L}(chi) proportional to sin^L(chi)  C^{L + 3/2}_{n - L}(cos chi)
    R3_{L l}(psi) proportional to sin^l(psi)  C^{l + 1}_{L - l}(cos psi)

with labels ordered n >= L >= l >= |m|. Counting these gives

    sum_{L=0}^{n} sum_{l=0}^{L} (2l + 1) = (n+1)(n+2)(2n+3)/6 = deg(n),

which is the S^4 scalar degeneracy, so complete shells are exactly the sets
0 <= n <= n_max.

The two radial factors are normalised **numerically**, using the same 1-D Gauss
rules that build the grid, rather than from closed-form constants. This is
deliberate: a mis-transcribed normalisation constant is silently wrong, whereas
numerical normalisation is verified end to end by the addition theorem

    sum_{L, l, m} Y_{n L l m}(x) Y_{n L l m}(y)
        = deg(n) / Vol(S^4) * C_n^{3/2}(x . y) / C_n^{3/2}(1),

which no consistent-but-wrong normalisation can satisfy (Appendix A of the
paper; checked in `tests/test_spectral.py`).

Mode ordering. A flat coefficient vector lists the modes by n, then L, then l,
then m (`label_list`), so index 0 is the constant mode (0, 0, 0, 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, lpmv, roots_chebyu, roots_jacobi

from .grid import S4Grid


def degeneracy(n):
    """Number of harmonics at level n, d_n = (n+1)(n+2)(2n+3)/6."""
    return (2 * n + 3) * (n + 1) * (n + 2) // 6


def eigenvalue(n):
    """-Laplacian eigenvalue on the unit S^4."""
    return n * (n + 3)


def n_modes(n_max):
    """Number of modes with n <= n_max."""
    return int(sum(degeneracy(n) for n in range(n_max + 1)))


def gegenbauer_table(alpha, k_max, x):
    """C^alpha_k(x) for k = 0..k_max, shape (k_max+1, len(x)), by recurrence."""
    x = np.asarray(x, dtype=float)
    out = np.zeros((k_max + 1, x.size))
    out[0] = 1.0
    if k_max >= 1:
        out[1] = 2.0 * alpha * x
    for k in range(2, k_max + 1):
        out[k] = (
            2.0 * (k + alpha - 1.0) * x * out[k - 1] - (k + 2.0 * alpha - 2.0) * out[k - 2]
        ) / k
    return out


def _radial_norms(n_max):
    """Normalisation constants for the two radial families, grid-independently.

    The constants are NOT computed on the grid used for the simulation. That
    would be circular: an under-resolved grid would renormalise the harmonics
    to be orthonormal with respect to its own inexact quadrature, hiding the
    aliasing error. They are computed once on a dedicated high-order 1-D rule
    that is exact for these integrands, so orthonormality on the simulation
    grid is a real test and the harmonics can be evaluated at arbitrary points
    (see `evaluate_at`).
    """
    n_ref = 4 * n_max + 12
    u_chi, w_chi = roots_jacobi(n_ref, 1.0, 1.0)     # weight sin^3 chi dchi
    u_psi, w_psi = roots_chebyu(n_ref)               # weight sin^2 psi dpsi
    chi, psi = np.arccos(u_chi), np.arccos(u_psi)

    n_l = n_max + 1
    norm4 = np.zeros((n_l, n_l))
    s_chi = np.sin(chi)
    for L in range(n_max + 1):
        geg = gegenbauer_table(L + 1.5, n_max - L, u_chi)
        sinL = s_chi**L
        for n in range(L, n_max + 1):
            f = sinL * geg[n - L]
            norm4[n, L] = 1.0 / np.sqrt(np.einsum("i,i->", w_chi, f * f))

    norm3 = np.zeros((n_l, n_l))
    s_psi = np.sin(psi)
    for l in range(n_max + 1):
        geg = gegenbauer_table(l + 1.0, n_max - l, u_psi)
        sinl = s_psi**l
        for L in range(l, n_max + 1):
            f = sinl * geg[L - l]
            norm3[L, l] = 1.0 / np.sqrt(np.einsum("i,i->", w_psi, f * f))

    return norm4, norm3


def real_sph_harm(l, m, theta, phi):
    """Real orthonormal spherical harmonic on S^2, as an outer product grid.

    Returns shape (len(theta), len(phi)). Sign conventions are irrelevant here;
    only orthonormality and shell completeness matter.
    """
    am = abs(m)
    k = np.sqrt(
        (2 * l + 1) / (4.0 * np.pi) * np.exp(gammaln(l - am + 1) - gammaln(l + am + 1))
    )
    p = lpmv(am, l, np.cos(theta))
    if m == 0:
        return np.outer(k * p, np.ones_like(phi))
    if m > 0:
        return np.sqrt(2.0) * np.outer(k * p, np.cos(am * phi))
    return np.sqrt(2.0) * np.outer(k * p, np.sin(am * phi))


@dataclass
class S4Harmonics:
    """Dense harmonic basis evaluated on a grid, plus the transforms.

    `Y` has shape (n_modes, n_points): row A is Y_A evaluated at every grid
    point. `labels[A]` is the tuple (n, L, l, m).
    """

    grid: S4Grid
    n_max: int
    Y: np.ndarray
    labels: np.ndarray
    shell_index: np.ndarray  # n for each mode, i.e. labels[:, 0]

    @property
    def lam(self):
        """-Laplacian eigenvalue for each mode."""
        return eigenvalue(self.shell_index.astype(float))

    def to_grid(self, coeffs):
        """Mode coefficients -> field values on the grid."""
        return coeffs @ self.Y

    def to_modes(self, field):
        """Field values on the grid -> mode coefficients (quadrature projection)."""
        return self.Y @ (self.grid.weights * field)

    def shell_power(self, coeffs):
        """Sum of |coeff|^2 within each n shell, length n_max+1."""
        out = np.zeros(self.n_max + 1)
        np.add.at(out, self.shell_index, coeffs**2)
        return out


def factor_tables(grid: S4Grid, n_max: int):
    """The 1-D factors the harmonics are built from, as padded dense arrays.

    Returns a dict with, using n_l = n_max+1 and n_m = 2 n_max + 1 (m offset by
    n_max so index mi = m + n_max):

        R4    (n_max+1, n_max+1, n_chi)   R4[n, L]   zero where L > n
        R3    (n_max+1, n_max+1, n_psi)   R3[L, l]   zero where l > L
        Theta (n_max+1, n_m,     n_theta) Theta[l, mi] zero where |m| > l
        Phi   (n_m,     n_phi)            Phi[mi]

    such that

        Y_{n L l m}(chi, psi, theta, varphi)
            = R4[n, L, i] R3[L, l, j] Theta[l, mi, k] Phi[mi, p].

    Both `build` (dense) and `transform.S4Transform` (separable) consume these,
    so the two representations cannot drift apart.
    """
    chi, psi, theta, phi = grid.chi, grid.psi, grid.theta, grid.phi
    n_chi, n_psi, n_theta, n_phi = grid.shape
    n_l = n_max + 1
    n_m = 2 * n_max + 1

    norm4, norm3 = _radial_norms(n_max)

    R4 = np.zeros((n_l, n_l, n_chi))
    s_chi, c_chi = np.sin(chi), np.cos(chi)
    for L in range(n_max + 1):
        geg = gegenbauer_table(L + 1.5, n_max - L, c_chi)
        sinL = s_chi**L
        for n in range(L, n_max + 1):
            R4[n, L] = norm4[n, L] * sinL * geg[n - L]

    R3 = np.zeros((n_l, n_l, n_psi))
    s_psi, c_psi = np.sin(psi), np.cos(psi)
    for l in range(n_max + 1):
        geg = gegenbauer_table(l + 1.0, n_max - l, c_psi)
        sinl = s_psi**l
        for L in range(l, n_max + 1):
            R3[L, l] = norm3[L, l] * sinl * geg[L - l]

    # Real spherical harmonic split Y_lm(theta, varphi) = Theta_{l|m|} Phi_m.
    Theta = np.zeros((n_l, n_m, n_theta))
    Phi = np.zeros((n_m, n_phi))
    for m in range(-n_max, n_max + 1):
        am = abs(m)
        mi = m + n_max
        if m == 0:
            Phi[mi] = 1.0
        elif m > 0:
            Phi[mi] = np.sqrt(2.0) * np.cos(am * phi)
        else:
            Phi[mi] = np.sqrt(2.0) * np.sin(am * phi)
        for l in range(am, n_max + 1):
            k = np.sqrt((2 * l + 1) / (4.0 * np.pi)
                        * np.exp(gammaln(l - am + 1) - gammaln(l + am + 1)))
            Theta[l, mi] = k * lpmv(am, l, np.cos(theta))

    return {"R4": R4, "R3": R3, "Theta": Theta, "Phi": Phi,
            "n_l": n_l, "n_m": n_m, "n_max": n_max}


def label_list(n_max):
    """Canonical mode ordering: n, then L, then l, then m."""
    return [
        (n, L, l, m)
        for n in range(n_max + 1)
        for L in range(n + 1)
        for l in range(L + 1)
        for m in range(-l, l + 1)
    ]


def build(grid: S4Grid, n_max: int) -> S4Harmonics:
    """Dense basis on `grid`. For validation and small n_max only.

    The array has n_modes * n_points entries (about 5 GB at n_max = 10 on the
    de-aliased grid); simulations use `transform.S4Transform` instead.
    """
    n_chi, n_psi, n_theta, n_phi = grid.shape
    t = factor_tables(grid, n_max)
    R4, R3, Theta, Phi = t["R4"], t["R3"], t["Theta"], t["Phi"]

    labels = [
        (n, L, l, m)
        for n in range(n_max + 1)
        for L in range(n + 1)
        for l in range(L + 1)
        for m in range(-l, l + 1)
    ]
    n_mode = len(labels)
    assert n_mode == n_modes(n_max), "label enumeration does not match deg(n) sum"

    Y = np.empty((n_mode, n_chi, n_psi, n_theta, n_phi))
    for A, (n, L, l, m) in enumerate(labels):
        mi = m + n_max
        Y[A] = (
            R4[n, L][:, None, None, None]
            * R3[L, l][None, :, None, None]
            * (Theta[l, mi][:, None] * Phi[mi][None, :])[None, None, :, :]
        )

    return S4Harmonics(
        grid=grid,
        n_max=n_max,
        Y=Y.reshape(n_mode, -1),
        labels=np.array(labels, dtype=int),
        shell_index=np.array([lab[0] for lab in labels], dtype=int),
    )


def angles_from_points(points):
    """Inverse of the S^4 embedding used by `grid.make_grid`.

    points : (N, 5) unit vectors -> (chi, psi, theta, varphi), each length N.
    """
    p = np.asarray(points, dtype=float)
    x0, x1, x2, x3, x4 = (p[:, i] for i in range(5))
    chi = np.arccos(np.clip(x0, -1.0, 1.0))
    s1 = np.sin(chi)
    safe1 = np.where(s1 > 1e-13, s1, 1.0)
    psi = np.arccos(np.clip(x1 / safe1, -1.0, 1.0))
    s2 = s1 * np.sin(psi)
    safe2 = np.where(s2 > 1e-13, s2, 1.0)
    theta = np.arccos(np.clip(x2 / safe2, -1.0, 1.0))
    varphi = np.arctan2(x4, x3) % (2.0 * np.pi)
    return chi, psi, theta, varphi


def evaluate_at(points, n_max):
    """Dense harmonics (n_modes, N) evaluated at arbitrary points on S^4.

    Used by the SO(5) rotation test, which needs the same band-limited field
    sampled on a rotated copy of the grid. Normalisation comes from
    `_radial_norms`, so this agrees with `build` without reference to any grid.
    """
    chi, psi, theta, varphi = angles_from_points(points)
    npts = chi.size
    norm4, norm3 = _radial_norms(n_max)

    c_chi, s_chi = np.cos(chi), np.sin(chi)
    c_psi, s_psi = np.cos(psi), np.sin(psi)
    ct = np.cos(theta)

    R4 = {}
    for L in range(n_max + 1):
        geg = gegenbauer_table(L + 1.5, n_max - L, c_chi)
        sinL = s_chi**L
        for n in range(L, n_max + 1):
            R4[(n, L)] = norm4[n, L] * sinL * geg[n - L]
    R3 = {}
    for l in range(n_max + 1):
        geg = gegenbauer_table(l + 1.0, n_max - l, c_psi)
        sinl = s_psi**l
        for L in range(l, n_max + 1):
            R3[(L, l)] = norm3[L, l] * sinl * geg[L - l]

    S2 = {}
    for l in range(n_max + 1):
        for m in range(-l, l + 1):
            am = abs(m)
            k = np.sqrt((2 * l + 1) / (4.0 * np.pi)
                        * np.exp(gammaln(l - am + 1) - gammaln(l + am + 1)))
            p = k * lpmv(am, l, ct)
            if m == 0:
                S2[(l, m)] = p
            elif m > 0:
                S2[(l, m)] = np.sqrt(2.0) * p * np.cos(am * varphi)
            else:
                S2[(l, m)] = np.sqrt(2.0) * p * np.sin(am * varphi)

    labels = label_list(n_max)
    Y = np.empty((len(labels), npts))
    for A, (n, L, l, m) in enumerate(labels):
        Y[A] = R4[(n, L)] * R3[(L, l)] * S2[(l, m)]
    return Y
