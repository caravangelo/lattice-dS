"""The separable transform must be indistinguishable from the dense basis.

The dense path is the one validated against analytic results in
`test_spectral.py` (orthonormality, addition theorem, exact free
propagator). These tests pin the separable transform to it, so the simulation
path inherits that validation rather than needing its own.
"""

import numpy as np
import pytest

from lattice_ds import action as A
from lattice_ds import grid as G
from lattice_ds import harmonics as H
from lattice_ds import hmc
from lattice_ds.transform import S4Transform


@pytest.mark.parametrize("n_max", [2, 3, 4, 6])
def test_label_order_matches_dense(n_max):
    g = G.for_degree(2 * n_max)
    h = H.build(g, n_max)
    t = S4Transform(g, n_max)
    assert np.array_equal(t.labels, h.labels)
    assert np.array_equal(t.shell_index, h.shell_index)


@pytest.mark.parametrize("n_max", [2, 3, 4, 6])
def test_to_grid_matches_dense(n_max):
    g = G.for_degree(2 * n_max)
    h = H.build(g, n_max)
    t = S4Transform(g, n_max)
    c = np.random.default_rng(0).standard_normal(t.n_modes)
    a, b = h.to_grid(c), t.to_grid(c)
    assert np.abs(a - b).max() / np.abs(a).max() < 1e-12


@pytest.mark.parametrize("n_max", [2, 3, 4, 6])
def test_to_modes_matches_dense(n_max):
    g = G.for_degree(2 * n_max)
    h = H.build(g, n_max)
    t = S4Transform(g, n_max)
    f = np.random.default_rng(1).standard_normal(g.n_points)
    a, b = h.to_modes(f), t.to_modes(f)
    assert np.abs(a - b).max() / np.abs(a).max() < 1e-12


def test_roundtrip_is_identity_on_band_limited_fields():
    n_max = 5
    t = S4Transform(G.for_degree(2 * n_max), n_max)
    c = np.random.default_rng(2).standard_normal(t.n_modes)
    assert np.abs(t.to_modes(t.to_grid(c)) - c).max() < 1e-11


def test_pack_unpack_roundtrip():
    n_max = 4
    t = S4Transform(G.for_degree(2 * n_max), n_max)
    c = np.random.default_rng(3).standard_normal(t.n_modes)
    assert np.array_equal(t.pack(t.unpack(c)), c)


def test_action_and_grad_agree_with_dense():
    n_max = 4
    g = G.for_degree(4 * n_max)          # de-aliased: quartic must be exact
    h = H.build(g, n_max)
    t = S4Transform(g, n_max)
    q = h.lam + 0.4
    c = 0.3 * np.random.default_rng(4).standard_normal(t.n_modes)
    assert A.action(c, h.Y, g.weights, q, 0.7) == pytest.approx(
        A.action(c, t, g.weights, q, 0.7), rel=1e-12)
    gd = A.grad(c, h.Y, g.weights, q, 0.7)
    gs = A.grad(c, t, g.weights, q, 0.7)
    assert np.abs(gd - gs).max() < 1e-11


def test_hmc_chains_are_identical_across_bases():
    """Same seed, both bases: the chains must agree bit for bit."""
    n_max, m2 = 4, 0.5
    g = G.for_degree(2 * n_max)
    h = H.build(g, n_max)
    t = S4Transform(g, n_max)
    out = []
    for basis in (h.Y, t):
        ctx = A.ActionContext(Y=basis, w=g.weights, lam=h.lam, m2=m2, lam4=0.0)
        out.append(hmc.run(ctx, t.shell_index, n_max, eps=0.3, n_steps=9,
                           n_therm=100, n_meas=400, seed=17))
    assert np.abs(out[0]["samples"] - out[1]["samples"]).max() == 0.0
    assert out[0]["acceptance"] == out[1]["acceptance"]


def test_context_n_modes_for_both_bases():
    n_max = 3
    g = G.for_degree(2 * n_max)
    h = H.build(g, n_max)
    t = S4Transform(g, n_max)
    for basis in (h.Y, t):
        ctx = A.ActionContext(Y=basis, w=g.weights, lam=h.lam, m2=0.5, lam4=0.0)
        assert ctx.n_modes == H.n_modes(n_max)


def test_memory_advantage_is_real():
    """Sanity check on the claim that motivates this module."""
    n_max = 10
    t = S4Transform(G.for_degree(4 * n_max), n_max)
    assert t.dense_bytes() > 4e9          # dense Y would be multiple GB
    assert t.factor_bytes < 5e6           # factor tables are a few MB
