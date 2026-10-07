"""Product Gauss quadrature grid on Euclidean S^4 (unit radius).

Coordinates and measure:

    ds^2 = dchi^2 + sin^2(chi) [ dpsi^2 + sin^2(psi) ( dtheta^2 + sin^2(theta) dvarphi^2 ) ]
    dOmega_4 = sin^3(chi) sin^2(psi) sin(theta) dchi dpsi dtheta dvarphi

with chi, psi, theta in [0, pi] and varphi in [0, 2 pi).

Each polar angle gets the Gauss rule whose weight function is exactly its
Jacobian factor, so no weight is ever approximated by a generic rule:

    chi   : u = cos(chi),   sin^3(chi) dchi = (1 - u^2) du    -> Gauss-Jacobi(1, 1)
    psi   : u = cos(psi),   sin^2(psi) dpsi = sqrt(1-u^2) du  -> Gauss-Chebyshev (2nd kind)
    theta : u = cos(theta), sin(theta) dtheta = du            -> Gauss-Legendre
    varphi: uniform nodes, exact for trigonometric polynomials

An n-node Gauss rule is exact for polynomial degree 2n - 1 against its weight,
and the uniform varphi rule with n_phi nodes is exact for trigonometric degree
< n_phi. `S4Grid.for_degree` sizes the rule from a target exactness degree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import roots_chebyu, roots_jacobi, roots_legendre

VOL_S4 = 8.0 * np.pi**2 / 3.0


@dataclass(frozen=True)
class S4Grid:
    """A product quadrature grid on S^4.

    Attributes
    ----------
    points : (N, 5) array
        Unit vectors in the R^5 embedding, in the order used by `weights`.
    weights : (N,) array
        Quadrature weights; they sum to Vol(S^4).
    chi, psi, theta, phi : 1-D node arrays for each angle
    w_chi, w_psi, w_theta, w_phi : 1-D weight arrays for each angle
    degree : int
        Polynomial degree this rule was built to integrate exactly.
    """

    points: np.ndarray
    weights: np.ndarray
    chi: np.ndarray
    psi: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    w_chi: np.ndarray
    w_psi: np.ndarray
    w_theta: np.ndarray
    w_phi: np.ndarray
    degree: int

    @property
    def shape(self):
        return (len(self.chi), len(self.psi), len(self.theta), len(self.phi))

    @property
    def n_points(self):
        return int(self.points.shape[0])


def _polar_rules(n_chi, n_psi, n_theta, n_phi):
    u1, w1 = roots_jacobi(n_chi, 1.0, 1.0)      # weight (1-u)(1+u) = sin^3 chi dchi
    u2, w2 = roots_chebyu(n_psi)                 # weight sqrt(1-u^2) = sin^2 psi dpsi
    u3, w3 = roots_legendre(n_theta)             # weight 1          = sin theta dtheta
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    w4 = np.full(n_phi, 2.0 * np.pi / n_phi)
    # Map u -> angle (arccos is decreasing, so the angles run from pi to 0;
    # the weights follow the nodes, so the ordering is immaterial).
    chi, psi, theta = np.arccos(u1), np.arccos(u2), np.arccos(u3)
    return (chi, w1), (psi, w2), (theta, w3), (phi, w4)


def make_grid(n_chi, n_psi, n_theta, n_phi, degree=-1):
    """Product grid with the given number of nodes in each angle."""
    (chi, w_chi), (psi, w_psi), (theta, w_theta), (phi, w_phi) = _polar_rules(
        n_chi, n_psi, n_theta, n_phi
    )

    # Embedding into R^5.
    sc, cc = np.sin(chi), np.cos(chi)
    sp, cp = np.sin(psi), np.cos(psi)
    st, ct = np.sin(theta), np.cos(theta)
    sf, cf = np.sin(phi), np.cos(phi)

    C = cc[:, None, None, None]
    S = sc[:, None, None, None]
    CP = cp[None, :, None, None]
    SP = sp[None, :, None, None]
    CT = ct[None, None, :, None]
    ST = st[None, None, :, None]
    CF = cf[None, None, None, :]
    SF = sf[None, None, None, :]

    ones = np.ones((n_chi, n_psi, n_theta, n_phi))
    x0 = C * ones
    x1 = S * CP * ones
    x2 = S * SP * CT * ones
    x3 = S * SP * ST * CF
    x4 = S * SP * ST * SF
    points = np.stack([x0, x1, x2, x3, x4], axis=-1).reshape(-1, 5)

    weights = (
        w_chi[:, None, None, None]
        * w_psi[None, :, None, None]
        * w_theta[None, None, :, None]
        * w_phi[None, None, None, :]
    ).reshape(-1)

    return S4Grid(
        points=points,
        weights=weights,
        chi=chi,
        psi=psi,
        theta=theta,
        phi=phi,
        w_chi=w_chi,
        w_psi=w_psi,
        w_theta=w_theta,
        w_phi=w_phi,
        degree=int(degree),
    )


def for_degree(degree):
    """Grid exact for band-limited functions of harmonic degree `degree`.

    An n-node Gauss rule integrates polynomial degree 2n-1 exactly, so
    n = degree // 2 + 1 suffices in each polar angle; varphi needs
    degree + 1 nodes for trigonometric exactness.

    For free-theory work (orthonormality of levels up to n_max) use
    degree = 2 * n_max. For the phi^4 interaction use degree = 4 * n_max, which
    evaluates the action and its gradient without aliasing and keeps SO(5)
    exact (Appendix A of the paper). At n_max = 10 that is 21^3 x 41 = 379 701
    points for 1716 modes.
    """
    n_pol = degree // 2 + 1
    n_phi = degree + 1
    return make_grid(n_pol, n_pol, n_pol, n_phi, degree=degree)
