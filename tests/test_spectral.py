"""Tests for the spectral S^4 regulator (grid, harmonics, action, sampler).

Grid, harmonics, action and gradient, leapfrog, and the free-theory sampler.
"""

import numpy as np
import pytest

from lattice_ds import action as A
from lattice_ds import grid as G
from lattice_ds import harmonics as H
from lattice_ds import hmc


@pytest.fixture(scope="module")
def basis():
    n_max = 4
    g = G.for_degree(4 * n_max)
    return g, H.build(g, n_max), n_max


def test_grid_integrates_volume():
    for d in (4, 8, 16):
        g = G.for_degree(d)
        assert g.weights.sum() == pytest.approx(G.VOL_S4, rel=1e-13)
        assert np.abs(np.linalg.norm(g.points, axis=1) - 1.0).max() < 1e-13


def test_mode_counting_matches_degeneracy():
    for n_max in (0, 1, 4, 8):
        g = G.for_degree(2 * n_max)
        h = H.build(g, n_max)
        assert h.Y.shape[0] == H.n_modes(n_max)
        for n in range(n_max + 1):
            assert int((h.shell_index == n).sum()) == H.degeneracy(n)


def test_orthonormality(basis):
    g, h, _ = basis
    M = h.Y @ (g.weights[:, None] * h.Y.T)
    assert np.abs(M - np.eye(M.shape[0])).max() < 1e-11


def test_addition_theorem(basis):
    """Cannot be satisfied by a consistent-but-wrong normalisation."""
    g, h, n_max = basis
    rng = np.random.default_rng(0)
    idx = rng.choice(g.n_points, 40, replace=False)
    dots = g.points[idx] @ g.points[idx].T
    for n in range(n_max + 1):
        sel = h.shell_index == n
        lhs = h.Y[sel][:, idx].T @ h.Y[sel][:, idx]
        c = H.gegenbauer_table(1.5, n, dots.ravel())[n].reshape(dots.shape)
        c1 = H.gegenbauer_table(1.5, n, np.array([1.0]))[n][0]
        rhs = H.degeneracy(n) / G.VOL_S4 * c / c1
        assert np.abs(lhs - rhs).max() / np.abs(rhs).max() < 1e-11


def test_transforms_roundtrip(basis):
    g, h, _ = basis
    rng = np.random.default_rng(1)
    c = rng.standard_normal(h.Y.shape[0])
    assert np.allclose(h.to_modes(h.to_grid(c)), c, atol=1e-11)


def test_grad_matches_finite_differences(basis):
    g, h, _ = basis
    ctx = A.ActionContext(Y=h.Y, w=g.weights, lam=h.lam, m2=0.4, lam4=0.7)
    Y, w, q, l4 = A.unpack(ctx)
    rng = np.random.default_rng(2)
    c = 0.3 * rng.standard_normal(ctx.n_modes)
    ga = A.grad(c, Y, w, q, l4)
    eps = 1e-6
    for i in rng.choice(ctx.n_modes, 15, replace=False):
        cp, cm = c.copy(), c.copy()
        cp[i] += eps
        cm[i] -= eps
        fd = (A.action(cp, Y, w, q, l4) - A.action(cm, Y, w, q, l4)) / (2 * eps)
        assert fd == pytest.approx(ga[i], rel=1e-5, abs=1e-8)


def test_quartic_is_exact_on_dealiased_grid():
    """S must not change when the grid is refined past degree 4 n_max."""
    n_max = 3
    rng = np.random.default_rng(3)
    vals = []
    for extra in (0, 8):
        g = G.for_degree(4 * n_max + extra)
        h = H.build(g, n_max)
        ctx = A.ActionContext(Y=h.Y, w=g.weights, lam=h.lam, m2=0.4, lam4=0.9)
        Y, w, q, l4 = A.unpack(ctx)
        c = 0.4 * np.random.default_rng(3).standard_normal(ctx.n_modes)
        vals.append(A.action(c, Y, w, q, l4))
    assert vals[0] == pytest.approx(vals[1], rel=1e-12)


def test_leapfrog_reversible_and_second_order(basis):
    g, h, _ = basis
    ctx = A.ActionContext(Y=h.Y, w=g.weights, lam=h.lam, m2=0.4, lam4=0.7)
    Y, w, q, l4 = A.unpack(ctx)
    minv = A.default_minv(q)
    rng = np.random.default_rng(4)
    c = 0.3 * rng.standard_normal(ctx.n_modes)
    p = rng.standard_normal(ctx.n_modes) * np.sqrt(q)

    c1, p1 = A.leapfrog(c, p, 0.02, 10, Y, w, q, l4, minv)
    c2, _ = A.leapfrog(c1, -p1, 0.02, 10, Y, w, q, l4, minv)
    assert np.abs(c2 - c).max() < 1e-12

    # Second-order convergence holds at *fixed trajectory length* T = eps * n.
    # Varying eps at fixed n_steps also shortens T, which is a different limit.
    def dH(eps, n_steps):
        h0 = A.hamiltonian(c, p, Y, w, q, l4, minv)
        ca, pa = A.leapfrog(c, p, eps, n_steps, Y, w, q, l4, minv)
        return abs(A.hamiltonian(ca, pa, Y, w, q, l4, minv) - h0)

    for eps, n in [(0.16, 2), (0.08, 4), (0.04, 8)]:
        assert dH(eps, n) / dH(eps / 2, 2 * n) == pytest.approx(4.0, rel=0.1)


def test_free_hmc_reproduces_exact_propagator():
    n_max, m2 = 4, 0.7
    g = G.for_degree(2 * n_max)
    h = H.build(g, n_max)
    ctx = A.ActionContext(Y=h.Y, w=g.weights, lam=h.lam, m2=m2, lam4=0.0)
    r = hmc.run(ctx, h.shell_index, n_max, eps=0.3, n_steps=11,
                n_therm=500, n_meas=8000, seed=11)
    assert r["acceptance"] > 0.8
    assert r["exp_negdH"] == pytest.approx(1.0, abs=0.01)
    ns = r["samples"].shape[0]
    for n in range(n_max + 1):
        exact = 1.0 / (H.eigenvalue(n) + m2)
        sig = exact * np.sqrt(2.0 / (H.degeneracy(n) * ns)) * 6.0
        assert abs(r["shell_power"][n] - exact) < 5 * sig
