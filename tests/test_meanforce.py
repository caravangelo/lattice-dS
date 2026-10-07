"""The mean-force analysis on synthetic chains with a known answer."""

import numpy as np
import pytest

from lattice_ds import meanforce
from lattice_ds import perturbation as P
from lattice_ds.grid import VOL_S4 as OM
from lattice_ds.stats import binder, binder_jackknife_error, tau_int


def synthetic_point(n_max=4, lam=2.0, m_s=-1.3, kappa=0.25, s=0.3, n=20000, seed=0):
    """Chains of a model in which the nonzero modes only shift the mass.

    I_2 = kappa * Omega is constant and A is Gaussian noise of width s, so that
    V_C'(phibar)/phibar = m_R^2 + lam phibar^2 with m_R^2 = m_s^2 + 3 lam kappa - s^2.
    The massless point is therefore delta_* = -(m_s^2 + 3 lam kappa - s^2),
    lambda_R = lam, and the zero mode at the massless point follows the SY law.
    """
    rng = np.random.default_rng(seed)
    mR = m_s + 3 * lam * kappa - s * s
    chains = []
    for phibar in (0.0, 0.0, 0.1, 0.15, 0.3, 0.38, 0.45):
        A = s * rng.standard_normal(n)
        B = np.full(n, m_s + 3 * lam * kappa)
        f = mR * phibar + lam * phibar ** 3 + 0.02 * rng.standard_normal(n)
        chains.append(meanforce.constrained_chain(phibar, m_s, lam, n_max, A, B, f))
    # unconstrained ensemble at m_s^2: P(c0) ~ exp(-mR c0^2/2 - lam c0^4/(4 Omega)), by inverse CDF
    x = np.linspace(-12, 12, 200001)
    p = np.exp(-(0.5 * mR * x ** 2 + lam * x ** 4 / (4 * OM)))
    cdf = np.cumsum(p)
    cdf /= cdf[-1]
    c0 = np.interp(rng.random(8 * n), cdf, x)
    unc = meanforce.unconstrained_ensemble(c0, c0 ** 2 / OM + kappa, m_s)
    return chains, unc, mR


def test_massless_point_and_coupling_on_synthetic_chains():
    lam, n_max = 2.0, 4
    chains, unc, mR = synthetic_point(n_max=n_max, lam=lam)
    r = meanforce.massless_point(chains, unc, order=3, xmax=0.5, n_boot=40, seed=1)
    u = P.screening_coefficient(n_max)
    assert r["delta"] == pytest.approx(-mR, abs=5 * r["delta_err"])
    assert 0 < r["delta_err"] < 0.01
    assert r["y_taylor"] == pytest.approx(u * lam, abs=5 * r["y_taylor_err"])
    assert r["sigma"] ** 2 == pytest.approx(P.SY_PHIBAR2 / np.sqrt(lam), rel=0.02)
    assert r["U"] == pytest.approx(P.SY_BINDER, abs=5 * r["U_err"])
    assert r["slope"] == pytest.approx(3.0, abs=0.05)             # pure quartic at the massless point
    assert r["y_eff"] == pytest.approx(u * lam, abs=5 * r["y_eff_err"])
    assert r["n_chains"] == 7 and r["n_fit"] == 7


def test_short_chains_are_refused_with_a_clear_message():
    chains, unc, _ = synthetic_point(n=20000)
    short = meanforce.constrained_chain(0.2, chains[0]["mass"], 2.0, 4, np.zeros(700), np.zeros(700), np.zeros(700))
    with pytest.raises(ValueError, match="two blocks.*phibar = 0.2"):
        meanforce.massless_point(chains + [short], unc, n_boot=2)
    with pytest.raises(ValueError, match="unconstrained ensemble"):
        meanforce.massless_point(chains, unc, n_boot=2, block=100000)


def test_fit_window_and_order():
    chains, unc, mR = synthetic_point()
    r = meanforce.massless_point(chains, unc, order=2, xmax=0.15, n_boot=10, seed=1)
    assert r["n_fit"] == 4 and r["n_chains"] == 7
    assert r["delta"] == pytest.approx(-mR, abs=6 * r["delta_err"])


def test_reweighting_identities():
    rng = np.random.default_rng(3)
    n, lam, m_s, phibar = 5000, 1.0, -0.4, 0.2
    B = m_s + 0.5 + 0.1 * rng.standard_normal(n)
    c = meanforce.constrained_chain(phibar, m_s, lam, 4, np.zeros(n), B, rng.standard_normal(n))
    f0, ess0 = meanforce.reweighted_force(c, 0.0)
    assert f0 == pytest.approx(c["f"].mean()) and ess0 == pytest.approx(1.0)
    # explicit shift of the force, and a reduced effective sample for delta != 0
    f1, ess1 = meanforce.reweighted_force(c, 0.3)
    w = np.exp(-0.15 * c["i2"])
    assert f1 == pytest.approx(np.sum(w * (c["f"] + 0.3 * phibar)) / w.sum())
    assert ess1 < 1.0


def test_fixed_mass_coupling_on_power_law():
    rng = np.random.default_rng(5)
    lam, n_max, sigma = 0.1, 4, 0.65
    chains = [meanforce.constrained_chain(pb, 0.0, lam, n_max, np.zeros(4000), np.zeros(4000),
                                          lam * pb ** 3 * (1 + 0.01 * rng.standard_normal(4000)))
              for pb in (0.8 * sigma, sigma, 1.2 * sigma)]
    r = meanforce.effective_coupling_fixed_mass(chains, sigma, n_max)
    assert r["slope"] == pytest.approx(3.0, abs=5 * r["slope_err"])
    assert r["y_eff"] == pytest.approx(P.screening_coefficient(n_max) * lam, abs=5 * r["y_eff_err"])


def test_stats_helpers():
    rng = np.random.default_rng(9)
    x = rng.standard_normal(40000)
    assert tau_int(x) == pytest.approx(0.5, abs=0.05)
    # AR(1) with coefficient a has tau = (1 + a) / (2 (1 - a))
    a = 0.8
    y = np.zeros(200000)
    e = rng.standard_normal(len(y))
    for i in range(1, len(y)):
        y[i] = a * y[i - 1] + e[i]
    assert tau_int(y) == pytest.approx((1 + a) / (2 * (1 - a)), rel=0.1)
    assert binder(x) == pytest.approx(0.0, abs=5 * binder_jackknife_error(x))
