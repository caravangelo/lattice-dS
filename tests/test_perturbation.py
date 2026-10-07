"""Tests for the analytic interacting-theory references and the run machinery."""

import numpy as np
import pytest

from lattice_ds import action as A
from lattice_ds import grid as G
from lattice_ds import harmonics as H
from lattice_ds import hmc
from lattice_ds import perturbation as P
from lattice_ds.transform import S4Transform


def test_coincident_propagator_matches_explicit_mode_sum():
    n_max, m2 = 5, 0.4
    g = G.for_degree(2 * n_max)
    h = H.build(g, n_max)
    # G(0) = sum_A Y_A(x)^2 / (lam_A + m^2), position independent by SO(5).
    direct = float(np.sum(h.Y[:, 0] ** 2 / (h.lam + m2)))
    assert P.coincident_propagator(n_max, m2) == pytest.approx(direct, rel=1e-10)


def test_nonzero_propagator_is_total_minus_zero_mode():
    n_max, m2 = 6, 0.02
    total = P.coincident_propagator(n_max, m2)
    nz = P.coincident_propagator_nonzero(n_max, m2)
    assert total - nz == pytest.approx(1.0 / (m2 * G.VOL_S4), rel=1e-12)
    # At small mass the constant mode dominates the full loop: this is why
    # dressing the zero mode with G(0) rather than G'(0) is badly wrong.
    assert (total - nz) / total > 0.7


def test_sy_moments_match_closed_form_at_zero_mass():
    for lam4 in (0.01, 0.1, 1.0, 10.0):
        assert P.sy_moments(0.0, lam4)["phi2"] == pytest.approx(
            P.sy_phi2_massless(lam4), rel=1e-10)


def test_sy_shows_inverse_sqrt_lambda_scaling():
    vals = [P.sy_moments(0.0, l)["phi2"] * np.sqrt(l) for l in (0.01, 0.1, 1, 10)]
    assert np.ptp(vals) / np.mean(vals) < 1e-9      # exactly 1/sqrt(lam)


def test_one_loop_density_is_even_and_reduces_to_tree():
    c = np.array([-2.0, -0.5, 0.5, 2.0])
    lp = P.one_loop_zero_mode_log_density(c, 4, 0.2, 0.3)
    assert lp[0] == pytest.approx(lp[3], rel=1e-12)
    assert lp[1] == pytest.approx(lp[2], rel=1e-12)
    # At vanishing coupling the determinant term switches off.
    tiny = P.one_loop_zero_mode_moments(4, 0.2, 1e-8)["c0_2"]
    assert tiny == pytest.approx(P.sy_moments(0.2, 1e-8)["c0_2"], rel=1e-4)


def test_one_loop_lies_between_tree_and_mass_shift():
    """The determinant is more than a mass shift, but the two should be close."""
    n_max, m2, lam4 = 4, 0.02, 0.05
    tree = P.sy_moments(m2, lam4)["c0_2"]
    ol = P.one_loop_zero_mode_moments(n_max, m2, lam4)["c0_2"]
    m2s = m2 + 3 * lam4 * P.coincident_propagator_nonzero(n_max, m2)
    shift = P.sy_moments(m2s, lam4)["c0_2"]
    assert ol < tree
    assert ol == pytest.approx(shift, rel=0.05)


def test_effective_mass_recovers_free_theory():
    n_max, m2 = 5, 0.6
    n = np.arange(n_max + 1)
    exact_power = 1.0 / (H.eigenvalue(n) + m2)
    est = P.effective_mass_from_shell_power(exact_power, n_max)
    assert np.allclose(est, m2, rtol=1e-10)
    mean, err, _, _ = P.effective_mass_weighted(exact_power, n_max, 10000)
    assert mean == pytest.approx(m2, rel=1e-10)
    assert err > 0


def test_reduced_observables_consistent_with_modes():
    n_max, m2 = 3, 0.7
    g = G.for_degree(2 * n_max)
    h = H.build(g, n_max)
    t = S4Transform(g, n_max)
    ctx = A.ActionContext(Y=t, w=g.weights, lam=h.lam, m2=m2, lam4=0.0)
    r = hmc.run(ctx, t.shell_index, n_max, eps=0.3, n_steps=11,
                n_therm=200, n_meas=2000, seed=3)
    # (1/Vol) integral phi^2 = (1/Vol) sum_A c_A^2 by orthonormality.
    assert r["phi2"].mean() == pytest.approx(
        r["mode_power"].sum() / G.VOL_S4, rel=1e-10)


def test_checkpoint_resume_reproduces_uninterrupted_chain(tmp_path):
    n_max, m2 = 3, 0.7
    g = G.for_degree(2 * n_max)
    h = H.build(g, n_max)
    t = S4Transform(g, n_max)
    ctx = A.ActionContext(Y=t, w=g.weights, lam=h.lam, m2=m2, lam4=0.2)
    kw = dict(eps=0.3, n_steps=11, n_therm=100, n_meas=1500, seed=8,
              zero_mode_updates=1)
    ref = hmc.run(ctx, t.shell_index, n_max, **kw)
    ck = tmp_path / "ck.npz"
    got = hmc.run(ctx, t.shell_index, n_max, checkpoint_path=str(ck),
                  checkpoint_every=200, **kw)
    assert np.abs(got["c0"] - ref["c0"]).max() == 0.0
    assert np.abs(got["shell_power"] - ref["shell_power"]).max() == 0.0


def test_keep_samples_false_saves_memory_but_keeps_observables():
    n_max, m2 = 3, 0.7
    g = G.for_degree(2 * n_max)
    h = H.build(g, n_max)
    t = S4Transform(g, n_max)
    ctx = A.ActionContext(Y=t, w=g.weights, lam=h.lam, m2=m2, lam4=0.0)
    kw = dict(eps=0.3, n_steps=11, n_therm=100, n_meas=800, seed=9)
    full = hmc.run(ctx, t.shell_index, n_max, keep_samples=True, **kw)
    light = hmc.run(ctx, t.shell_index, n_max, keep_samples=False, **kw)
    assert light["samples"].shape[0] == 0
    assert np.abs(light["c0"] - full["c0"]).max() == 0.0
    assert np.abs(light["shell_power"] - full["shell_power"]).max() == 0.0
