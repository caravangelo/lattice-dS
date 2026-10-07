"""Separable (factorised) harmonic transform on S^4.

The dense transform stores `Y` of shape (n_modes, n_points) and costs
O(n_modes * n_points) per application. That is fatal well before it is slow:
at n_max = 12 with a de-aliased grid, `Y` is about 2.4e9 entries, roughly
19.5 GB in float64, touched twice per leapfrog step.

Because the quadrature is a *product* grid and the harmonics factorise,

    Y_{n L l m}(chi, psi, theta, varphi)
        = R4[n, L](chi) R3[L, l](psi) Theta[l, m](theta) Phi[m](varphi),

the transform decomposes into four one-dimensional contractions applied in
sequence, each a batched GEMM:

    grid -> modes :  varphi -> theta -> psi -> chi
    modes -> grid :  chi -> psi -> theta -> varphi   (the transpose, reversed)

`Y` is never formed. Only the four factor tables are stored, a few MB, and the
cost drops from O(n_modes * n_points) to O(n_points * n_max) up to constants.
This is what makes useful n_max reachable, on CPU or anywhere else.

Intermediate arrays are rectangular and padded: the label triangle
n >= L >= l >= |m| is enforced by zeros in the factor tables (R4 is zero where
L > n, R3 where l > L, Theta where |m| > l), so no masking is needed and every
stage stays a dense contraction. `pack` and `unpack` convert between the padded
(n, L, l, m) box and the flat coefficient vector whose ordering matches
`harmonics.build`.

Correctness is pinned by `tests/test_transform.py`, which requires agreement
with the dense path to machine precision in both directions.
"""

from __future__ import annotations

import numpy as np

from .grid import S4Grid
from .harmonics import factor_tables, label_list


class S4Transform:
    """Separable mode/grid transform for complete shells n <= n_max."""

    def __init__(self, grid: S4Grid, n_max: int):
        self.grid = grid
        self.n_max = n_max
        t = factor_tables(grid, n_max)
        self.n_l = t["n_l"]
        self.n_m = t["n_m"]
        self.grid_shape = grid.shape
        n_chi, n_psi, n_theta, n_phi = grid.shape

        # Weighted tables for the forward (grid -> modes) direction, and bare
        # tables for the backward direction.
        self.R4 = t["R4"]                                  # (n, L, chi)
        self.R3 = t["R3"]                                  # (L, l, psi)
        self.Theta = t["Theta"]                            # (l, mi, theta)
        self.Phi = t["Phi"]                                # (mi, phi)

        self.R4w = self.R4 * grid.w_chi                    # (n, L, chi)
        self.R3w = self.R3 * grid.w_psi                    # (L, l, psi)
        self.Thetaw = self.Theta * grid.w_theta            # (l, mi, theta)
        self.Phiw = self.Phi * grid.w_phi                  # (mi, phi)

        # Backward direction batches over a different leading index, so the
        # tables are stored pre-transposed and contiguous.
        self.R4b = np.ascontiguousarray(self.R4.transpose(1, 0, 2))      # (L, n, chi)
        self.R3b = np.ascontiguousarray(self.R3.transpose(1, 0, 2))      # (l, L, psi)
        self.Thetab = np.ascontiguousarray(self.Theta.transpose(1, 0, 2))  # (mi, l, theta)

        labels = label_list(n_max)
        self.labels = np.array(labels, dtype=int)
        self.n_modes = len(labels)
        self.idx = (
            self.labels[:, 0],
            self.labels[:, 1],
            self.labels[:, 2],
            self.labels[:, 3] + n_max,
        )
        self.shell_index = self.labels[:, 0].copy()

    # ---- packing between the padded box and the flat coefficient vector ----

    def pack(self, padded):
        return padded[self.idx]

    def unpack(self, coeffs):
        out = np.zeros((self.n_l, self.n_l, self.n_l, self.n_m))
        out[self.idx] = coeffs
        return out

    # ---- the two directions ----

    def to_modes(self, field):
        """Grid values (flat, or grid-shaped) -> flat mode coefficients."""
        n_chi, n_psi, n_theta, n_phi = self.grid_shape
        n_l, n_m = self.n_l, self.n_m
        F = np.asarray(field).reshape(n_chi, n_psi, n_theta, n_phi)

        # varphi: (.., n_phi) @ (n_phi, n_m)
        A = (F.reshape(-1, n_phi) @ self.Phiw.T).reshape(n_chi, n_psi, n_theta, n_m)

        # theta: batched over mi
        A2 = np.ascontiguousarray(A.transpose(3, 0, 1, 2)).reshape(n_m, n_chi * n_psi, n_theta)
        B2 = A2 @ np.ascontiguousarray(self.Thetaw.transpose(1, 2, 0))       # (n_m, ., n_l)
        B = B2.reshape(n_m, n_chi, n_psi, n_l).transpose(1, 2, 3, 0)

        # psi: batched over l
        B3 = np.ascontiguousarray(B.transpose(2, 0, 3, 1)).reshape(n_l, n_chi * n_m, n_psi)
        C3 = B3 @ np.ascontiguousarray(self.R3w.transpose(1, 2, 0))          # (n_l, ., n_L)
        C = C3.reshape(n_l, n_chi, n_m, n_l).transpose(1, 3, 0, 2)

        # chi: batched over L
        C4 = np.ascontiguousarray(C.transpose(1, 2, 3, 0)).reshape(n_l, n_l * n_m, n_chi)
        c4 = C4 @ np.ascontiguousarray(self.R4w.transpose(1, 2, 0))          # (n_L, ., n_n)
        padded = c4.reshape(n_l, n_l, n_m, n_l).transpose(3, 0, 1, 2)

        return self.pack(padded)

    def to_grid(self, coeffs):
        """Flat mode coefficients -> grid values, flattened in grid-point order."""
        n_chi, n_psi, n_theta, n_phi = self.grid_shape
        n_l, n_m = self.n_l, self.n_m
        padded = self.unpack(np.asarray(coeffs))

        # chi^T
        c4 = np.ascontiguousarray(padded.transpose(1, 2, 3, 0)).reshape(n_l, n_l * n_m, n_l)
        C4 = c4 @ self.R4b                                                   # (n_L, ., n_chi)
        C = C4.reshape(n_l, n_l, n_m, n_chi).transpose(3, 0, 1, 2)

        # psi^T
        C3 = np.ascontiguousarray(C.transpose(2, 0, 3, 1)).reshape(n_l, n_chi * n_m, n_l)
        B3 = C3 @ self.R3b                                                   # (n_l, ., n_psi)
        B = B3.reshape(n_l, n_chi, n_m, n_psi).transpose(1, 3, 0, 2)

        # theta^T
        B2 = np.ascontiguousarray(B.transpose(3, 0, 1, 2)).reshape(n_m, n_chi * n_psi, n_l)
        A2 = B2 @ self.Thetab                                                # (n_m, ., n_theta)
        A = A2.reshape(n_m, n_chi, n_psi, n_theta).transpose(1, 2, 3, 0)

        # varphi^T
        F = np.ascontiguousarray(A).reshape(-1, n_m) @ self.Phi
        return F.reshape(-1)

    # ---- diagnostics ----

    @property
    def factor_bytes(self):
        return int(sum(a.nbytes for a in
                       (self.R4, self.R3, self.Theta, self.Phi,
                        self.R4w, self.R3w, self.Thetaw, self.Phiw,
                        self.R4b, self.R3b, self.Thetab)))

    def dense_bytes(self):
        """Bytes the equivalent dense Y would occupy, for comparison."""
        return int(self.n_modes * int(np.prod(self.grid_shape)) * 8)
