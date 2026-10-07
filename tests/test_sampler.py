"""Run machinery: interruption and resume, step-size calibration, the lattice-ds command."""

import subprocess
import sys

import numpy as np
import pytest

import lattice_ds as L
from lattice_ds import action as A
from lattice_ds import hmc
from lattice_ds import perturbation as P



def test_interrupted_chain_resumes_exactly(tmp_path, monkeypatch):
    """A chain killed mid-run and restarted reproduces the uninterrupted one, diagnostics included."""
    n_max, lam4 = 3, 2.0
    ctx, basis = L.setup(n_max, P.one_loop_bare_mass(n_max, lam4), lam4)
    kw = dict(eps=0.25, n_steps=6, n_therm=60, n_meas=200, seed=4242,
              minv=A.renormalised_minv(ctx.lam, 0.0), zero_mode_updates=2, keep_samples=False)
    ref = hmc.run(ctx, basis.shell_index, n_max, **kw)

    ck = tmp_path / "ck.npz"
    real = hmc.leapfrog
    calls = {"n": 0}

    def dying(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] in (45, 150):          # once during thermalisation, once during measurement
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(hmc, "leapfrog", dying)
    got = None
    for _ in range(3):
        try:
            got = hmc.run(ctx, basis.shell_index, n_max, checkpoint_path=str(ck), checkpoint_every=20, **kw)
            break
        except KeyboardInterrupt:
            assert ck.exists()
    assert got is not None and calls["n"] > 260       # it really was interrupted twice and redid some work
    for k in ("c0", "phi2", "action", "shell_series", "dH"):
        assert np.array_equal(got[k], ref[k]), k
    for k in ("acceptance", "therm_acceptance", "exp_negdH", "exp_negdH_meas", "dH_max"):
        assert got[k] == ref[k], k


def test_checkpoint_refuses_other_arguments_and_carries_observables(tmp_path, monkeypatch):
    """A checkpoint written with one set of arguments is not resumed with another, and the
    per-measurement observables are restored on resume."""
    n_max, lam4 = 3, 2.0
    ctx, basis = L.setup(n_max, P.one_loop_bare_mass(n_max, lam4), lam4)
    obs = lambda c: np.array([c[0], float(np.dot(c, c))])
    kw = dict(eps=0.25, n_steps=6, n_therm=20, n_meas=100, seed=11, zero_mode_updates=1,
              keep_samples=False, observables=obs)
    ref = hmc.run(ctx, basis.shell_index, n_max, **kw)
    assert ref["observables"].shape == (100, 2) and np.array_equal(ref["observables"][:, 0], ref["c0"])

    ck = tmp_path / "ck.npz"
    real = hmc.leapfrog
    calls = {"n": 0}

    def dying(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 70:
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(hmc, "leapfrog", dying)
    with pytest.raises(KeyboardInterrupt):
        hmc.run(ctx, basis.shell_index, n_max, checkpoint_path=str(ck), checkpoint_every=10, **kw)
    with pytest.raises(ValueError, match="different arguments.*eps"):
        hmc.run(ctx, basis.shell_index, n_max, checkpoint_path=str(ck), checkpoint_every=10, **dict(kw, eps=0.2))
    got = hmc.run(ctx, basis.shell_index, n_max, checkpoint_path=str(ck), checkpoint_every=10, **kw)
    assert np.array_equal(got["observables"], ref["observables"])
    assert np.array_equal(got["c0"], ref["c0"])


def test_tune_eps_shrinks_at_fixed_trajectory_length():
    n_max, lam4 = 4, 60.0
    ctx, basis = L.setup(n_max, -28.2, lam4)
    minv = A.renormalised_minv(ctx.lam, 0.0)
    eps, n_steps = hmc.tune_eps(ctx, basis.shell_index, n_max, eps0=0.25, n_steps=6, seed=1,
                                minv=minv, zero_mode_updates=2, n_cal=100)
    assert eps < 0.25
    assert eps * n_steps == pytest.approx(1.5, rel=0.15)


def test_cold_start_is_available_and_hot_is_default():
    ctx, basis = L.setup(2, 0.5, 0.0)
    r = hmc.run(ctx, basis.shell_index, 2, eps=0.25, n_steps=6, n_therm=0, n_meas=1, seed=0, start="cold")
    assert r["therm_acceptance"] is None
    with pytest.raises(ValueError):
        hmc.run(ctx, basis.shell_index, 2, eps=0.25, n_steps=6, n_therm=0, n_meas=1, start="warm")


def _run(command, *args):
    return subprocess.run([sys.executable, "-m", "lattice_ds.cli", command, *map(str, args)],
                          capture_output=True, text=True)


def test_run_command(tmp_path):
    args = ("--n-max", 2, "--lam", 0.1, "--one-loop-mass", "--seeds", 2, "--n-meas", 300,
            "--n-therm", 100, "--workers", 1, "--outdir", tmp_path)
    p = _run("run", *args)
    assert p.returncode == 0, p.stdout + p.stderr
    files = sorted(tmp_path.glob("*.npz"))
    assert len(files) == 2
    z = np.load(files[0])
    assert z["c0"].shape == (300,) and float(z["m2"]) == P.one_loop_bare_mass(2, 0.1)
    from lattice_ds.cli.run import derive_seed
    seeds = {int(np.load(f)["seed"]) for f in files}
    assert seeds == {derive_seed(260910, 2, P.one_loop_bare_mass(2, 0.1), 0.1, s) for s in (0, 1)}
    # the seed depends on the point, not on list positions: a different coupling never shares it
    assert derive_seed(260910, 2, P.one_loop_bare_mass(2, 0.2), 0.2, 0) not in seeds
    # second call: finished chains are skipped; changed settings are refused
    assert "skipped" in _run("run", *args).stdout
    changed = _run("run", *args[:8], 400, *args[9:])
    assert changed.returncode == 1 and "MISMATCH" in changed.stdout


def test_run_constrained_and_run_with_force_commands(tmp_path):
    p = _run("run-constrained", "--tag", "c", "--n-max", 2, "--lam", 1.0, "--m2", -0.3, "--phibar", 0.0,
             "--n-meas", 300, "--n-therm", 100, "--outdir", tmp_path)
    assert p.returncode == 0, p.stdout + p.stderr
    z = np.load(tmp_path / "c.npz")
    assert {"A", "B", "force", "phibar", "m2", "lam4", "n_max", "vol"} <= set(z.files)
    assert "curvature_mR2" in p.stdout
    from lattice_ds import meanforce
    assert meanforce.load_constrained([tmp_path / "c.npz"])[0]["f"].shape == (300,)

    p = _run("run-with-force", "--tag", "u", "--n-max", 2, "--lam", 1.0, "--m2", -0.3,
             "--n-meas", 300, "--n-therm", 100, "--outdir", tmp_path)
    assert p.returncode == 0, p.stdout + p.stderr
    q2, p2, vol, m2 = meanforce.load_unconstrained(tmp_path / "u.npz")
    assert q2.shape == (300,) and m2 == -0.3
