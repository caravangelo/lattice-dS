"""Tests for the exact zero-mode conditional update and the SO(5) diagnostic."""

import numpy as np
import pytest

from lattice_ds import action as A
from lattice_ds import grid as G
from lattice_ds import harmonics as H
from lattice_ds import hmc
from lattice_ds import zeromode as Z
from lattice_ds.transform import S4Transform


@pytest.fixture(scope="module")
def setup():
    n_max = 3
    g = G.for_degree(4 * n_max)
    h = H.build(g, n_max)
    t = S4Transform(g, n_max)
    return g, h, t, n_max


def test_zero_mode_is_first_label(setup):
    _, h, _, _ = setup
    assert tuple(h.labels[Z.ZERO_MODE_INDEX]) == (0, 0, 0, 0)


def test_zero_mode_harmonic_is_constant(setup):
    g, h, _, _ = setup
    y0 = h.Y[Z.ZERO_MODE_INDEX]
    assert np.allclose(y0, 1.0 / np.sqrt(G.VOL_S4), atol=1e-13)


def test_conditional_reproduces_full_action_dependence(setup):
    """S(c_0|rest) from the quartic coefficients must differ from the full
    action by a constant, exactly."""
    g, h, t, _ = setup
    q = h.lam + 0.3
    lam4 = 0.8
    c = 0.4 * np.random.default_rng(0).standard_normal(t.n_modes)
    a1, a2, a3, a4 = Z.conditional_coeffs(c, t, g.weights, q, lam4, G.VOL_S4)
    offsets = []
    for c0 in (-1.7, -0.5, 0.0, 0.6, 1.9):
        cc = c.copy()
        cc[0] = c0
        offsets.append(A.action(cc, t, g.weights, q, lam4)
                       - Z._quartic(c0, a1, a2, a3, a4))
    assert np.ptp(offsets) < 1e-9


def test_free_conditional_is_exact_gaussian(setup):
    g, h, t, _ = setup
    q = h.lam + 0.7
    rng = np.random.default_rng(1)
    c = np.zeros(t.n_modes)
    draws = []
    for _ in range(20000):
        Z.update(rng, c, t, g.weights, q, 0.0, G.VOL_S4)
        draws.append(c[0])
    draws = np.array(draws)
    exact_var = 1.0 / q[0]
    assert abs(draws.mean()) < 5 * np.sqrt(exact_var / len(draws))
    assert draws.var() == pytest.approx(exact_var, rel=0.05)


@pytest.mark.parametrize("coef", [(0.0, 1.5, 0.0, 0.05),      # unimodal
                                  (0.0, -2.0, 0.0, 0.08),     # bimodal
                                  (0.4, -2.5, 0.3, 0.10)])    # skewed bimodal
def test_conditional_sampler_matches_numerical_integration(coef):
    a1, a2, a3, a4 = coef
    rng = np.random.default_rng(2)
    n = 60000
    xs = np.empty(n)
    cur = 0.0
    for i in range(n):
        cur, _ = Z.sample_conditional(rng, cur, a1, a2, a3, a4)
        xs[i] = cur
    c = np.linspace(-40, 40, 400001)
    p = np.exp(-(Z._quartic(c, a1, a2, a3, a4) - Z._quartic(c, a1, a2, a3, a4).min()))
    w = p / np.trapezoid(p, c)
    m2 = np.trapezoid(w * c**2, c)
    assert (xs**2).mean() == pytest.approx(m2, rel=0.05)


def test_conditional_sampler_is_exact_with_a_coarse_table():
    """The MH correction makes the draw exact even when the proposal table is crude (8 cells,
    acceptance about 0.63). With the proposal density evaluated as a linear interpolation of
    the table instead of the piecewise-constant density that the inverse-CDF draw samples,
    this table gives <c_0> = -0.617 and <c_0^2> = 1.23 for the exact -0.538 and 1.082."""
    a1, a2, a3, a4 = 0.3, -1.0, 0.2, 0.5
    rng = np.random.default_rng(5)
    n, cur, acc = 100000, 0.0, 0
    xs = np.empty(n)
    for i in range(n):
        cur, a = Z.sample_conditional(rng, cur, a1, a2, a3, a4, n_grid=8)
        xs[i] = cur
        acc += a
    c = np.linspace(-8, 8, 400001)
    s = Z._quartic(c, a1, a2, a3, a4)
    w = np.exp(-(s - s.min()))
    w /= np.trapezoid(w, c)
    assert acc / n < 0.8                                    # the table is visibly imperfect ...
    assert xs.mean() == pytest.approx(np.trapezoid(w * c, c), abs=0.015)       # ... and the draw still exact
    assert (xs**2).mean() == pytest.approx(np.trapezoid(w * c**2, c), rel=0.02)


def test_minv_floor_protects_soft_zero_mode():
    q = np.array([1e-6, 4.0, 4.0, 10.0])
    minv = A.default_minv(q)
    assert minv[0] == pytest.approx(1.0 / 4.0)     # floored at the n=1 shell
    assert minv[3] == pytest.approx(1.0 / 10.0)    # stiff modes untouched


def test_hmc_with_zero_mode_updates_still_free_exact():
    n_max, m2 = 3, 0.7
    g = G.for_degree(2 * n_max)
    h = H.build(g, n_max)
    t = S4Transform(g, n_max)
    ctx = A.ActionContext(Y=t, w=g.weights, lam=h.lam, m2=m2, lam4=0.0)
    r = hmc.run(ctx, t.shell_index, n_max, eps=0.3, n_steps=11,
                n_therm=500, n_meas=8000, seed=13, zero_mode_updates=2)
    ns = r["samples"].shape[0]
    for n in range(n_max + 1):
        exact = 1.0 / (H.eigenvalue(n) + m2)
        sig = exact * np.sqrt(2.0 / (H.degeneracy(n) * ns)) * 6.0
        assert abs(r["shell_power"][n] - exact) < 5 * sig


def test_angles_roundtrip_and_evaluate_at_matches_build():
    n_max = 3
    g = G.for_degree(2 * n_max)
    h = H.build(g, n_max)
    assert np.abs(H.evaluate_at(g.points, n_max) - h.Y).max() < 1e-11


def test_so5_invariance_on_dealiased_grid():
    """A rotated field has the same int phi^4 on the grid of degree 4 n_max."""
    n_max = 3
    g = G.for_degree(4 * n_max)
    h = H.build(g, n_max)
    rng = np.random.default_rng(5)
    c = rng.standard_normal(H.n_modes(n_max))
    i0 = float(np.sum(g.weights * (c @ h.Y) ** 4))
    a = rng.standard_normal((5, 5))
    Q, R = np.linalg.qr(a)
    Q *= np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    ir = float(np.sum(g.weights * (c @ H.evaluate_at(g.points @ Q.T, n_max)) ** 4))
    assert abs(ir - i0) / abs(i0) < 1e-12
