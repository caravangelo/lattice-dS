"""Constrained simulations: reduction to N - 1 modes, estimators, equilibrium."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp

from lattice_ds import action as A
from lattice_ds import constrained, hmc, zeromode
from lattice_ds.stats import tau_int


@pytest.mark.parametrize("phibar", [0.0, 0.2, -0.2])
def test_reduced_action_gradient_and_reversibility(phibar):
    rng = np.random.default_rng(4200)
    m2, lam4 = -0.8, 3.0
    ctx, basis, eig, vol = constrained.context(2, m2, lam4, phibar)
    u = rng.normal(size=ctx.n_modes) * 0.1
    Y, w, q, lam = A.unpack(ctx)
    c = Y.full(u)
    # reduced action = full action minus the free term of the frozen zero mode
    s_red = A.action(u, Y, w, q, lam)
    s_full = A.action(c, basis, w, eig + m2, lam)
    np.testing.assert_allclose(s_full - s_red, 0.5 * m2 * c[0] ** 2, atol=1e-12)
    g = A.grad(u, Y, w, q, lam)
    for i in (0, 3, len(u) - 1):
        d = np.zeros(len(u))
        d[i] = 1e-6
        numeric = (A.action(u + d, Y, w, q, lam) - A.action(u - d, Y, w, q, lam)) / 2e-6
        np.testing.assert_allclose(g[i], numeric, rtol=2e-7, atol=1e-9)
    minv = A.renormalised_minv(eig[1:])
    p = rng.normal(size=len(u)) / np.sqrt(minv)
    un, pn = A.leapfrog(u, p, 0.04, 7, Y, w, q, lam, minv)
    ub, pb = A.leapfrog(un, -pn, 0.04, 7, Y, w, q, lam, minv)
    np.testing.assert_allclose(ub, u, atol=1e-12)
    np.testing.assert_allclose(pb, -p, atol=1e-12)
    assert Y.full(un)[0] == Y.c0          # the zero mode never moves


def test_force_is_derivative_of_action_in_phibar():
    """f = (1/Omega) dS/dphibar at fixed nonzero modes, by finite differences."""
    rng = np.random.default_rng(7)
    n_max, m2, lam4, phibar = 3, -0.5, 2.0, 0.3
    ctx, basis, eig, vol = constrained.context(n_max, m2, lam4, phibar)
    u = rng.normal(size=ctx.n_modes) * 0.1
    full = ctx.Y.full(u)
    a, b, f = constrained.zero_mode_coefficients(full, basis, ctx.w, eig + m2, lam4, vol)

    def s_at(pb):
        c = full.copy()
        c[0] = np.sqrt(vol) * pb
        return A.action(c, basis, ctx.w, eig + m2, lam4)

    h = 1e-5
    assert f == pytest.approx((s_at(phibar + h) - s_at(phibar - h)) / (2 * h) / vol, rel=1e-7)
    # (A, B): S = S[psi] + A c0 + B c0^2/2 + lam4 c0^4/(4 Omega)
    c0 = full[0]
    c = full.copy()
    c[0] = 0.0
    s0 = A.action(c, basis, ctx.w, eig + m2, lam4)
    assert s_at(phibar) == pytest.approx(s0 + a * c0 + 0.5 * b * c0 ** 2 + lam4 * c0 ** 4 / (4 * vol), rel=1e-11)


def test_curvature_estimator_on_exactly_solvable_mixture():
    """<B> - <A^2> is the second derivative of the free energy W(c0)."""
    ab = np.array([[-1.0, 1.5], [-0.3, 0.8], [0.3, 0.8], [1.0, 1.5]])   # equal-weight (A, B) values

    def W(c):
        return -logsumexp(-ab[:, 0] * c - 0.5 * ab[:, 1] * c * c)

    h = 0.01
    w2 = (W(h) - 2 * W(0) + W(-h)) / h ** 2
    assert constrained.curvature(ab[:, 0], ab[:, 1]) == pytest.approx(w2, abs=2e-4)
    # Gaussian case: the curvature is the bare one
    assert constrained.curvature(np.zeros(10), np.full(10, 0.7)) == pytest.approx(0.7, abs=1e-12)


def test_free_constrained_equilibrium():
    ctx, basis, eig, vol = constrained.context(2, 0.7, 0.0, 0.3)
    minv = 1 / (eig[1:] + 0.7)
    r = hmc.run(ctx, basis.shell_index[1:] - 1, 1, eps=0.15, n_steps=10,
                n_therm=500, n_meas=12000, seed=42001, minv=minv, zero_mode_updates=0)
    scaled = np.mean(r["samples"] ** 2 * (eig[1:] + 0.7), axis=1)
    err = np.std(scaled, ddof=1) * np.sqrt(2 * tau_int(scaled) / len(scaled))
    assert abs(scaled.mean() - 1) < 5 * err
    assert r["acceptance"] > 0.8 and 0.9 < r["exp_negdH"] < 1.1


def test_interacting_constrained_equilibrium_against_exact_integral():
    """n_max = 1 at strong coupling: five modes, and the constrained integral is radial.

    int psi^4 = 15 |c|^4 / (7 Omega) and int psi^3 = 0, so <|c|^2> is a
    one-dimensional integral. A non-Gaussian benchmark for the sampler.
    """
    ctx, basis, eig, vol = constrained.context(1, -8.0, 30.0, 0.0)
    r = hmc.run(ctx, basis.shell_index[1:] - 1, 0, eps=0.04, n_steps=37,
                n_therm=1000, n_meas=30000, seed=42002,
                minv=A.renormalised_minv(eig[1:]), zero_mode_updates=0)
    r2 = np.sum(r["samples"] ** 2, axis=1)

    def radial(k):
        return quad(lambda x: x ** (4 + k) * np.exp(2 * x * x - 15 * 30 * x ** 4 / (28 * vol)), 0, 8, epsabs=1e-10)[0]

    exact = radial(2) / radial(0)
    se = r2.std(ddof=1) * np.sqrt(2 * tau_int(r2) / len(r2))
    assert abs(r2.mean() - exact) < 5 * se
    assert r["acceptance"] > 0.8 and 0.9 < r["exp_negdH"] < 1.1


def test_measure_matches_conditional_coefficients():
    ctx, basis, eig, vol = constrained.context(2, -0.3, 1.5, 0.25)
    r = hmc.run(ctx, basis.shell_index[1:] - 1, 1, eps=0.2, n_steps=7,
                n_therm=50, n_meas=20, seed=3, minv=A.renormalised_minv(eig[1:]), zero_mode_updates=0)
    Aser, Bser, f = constrained.measure(r["samples"], ctx, basis, eig, -0.3, 1.5, vol)
    full = ctx.Y.full(r["samples"][5])
    a1, a2, a3, a4 = zeromode.conditional_coeffs(full, basis, ctx.w, eig - 0.3, 1.5, vol)
    assert Aser[5] == a1 and Bser[5] == 2 * a2
    assert abs(a3) < 1e-12            # int psi = 0
