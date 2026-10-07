"""Hybrid Monte Carlo for the regulated lambda phi^4 theory on S^4.

One iteration of `run` is

1. an HMC trajectory on all mode coefficients: Gaussian momenta with variance
   M_A, leapfrog integration (`action.leapfrog`), Metropolis accept/reject on
   the energy change dH;
2. optionally, `zero_mode_updates` exact heat-bath draws of the constant mode
   from its conditional distribution (`zeromode.update`).

Each step preserves the target distribution exp(-S), so their composition does.
The algorithm is described in Appendix B of the paper.

Three features are needed for correct results and are on by default:

- Fourier acceleration: the mass matrix M (`action.default_minv`,
  `action.renormalised_minv`) makes every free mode a unit-frequency oscillator.
- Randomised trajectory length: with a common frequency a fixed length T would
  be resonant near T = pi, so the number of leapfrog steps is drawn uniformly
  within +-30% of its nominal value for each trajectory.
- Hot start: chains start from a free-theory Gaussian draw (`free_start`), not
  from phi = 0.

`tune_eps` calibrates the step size at strong coupling. The health of a chain
is judged by the identity <exp(-dH)> = 1 as well as by the acceptance: a chain
can keep a good acceptance while rare unstable trajectories violate the
identity by orders of magnitude.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import zeromode
from .action import action, default_minv, hamiltonian, leapfrog, unpack


def free_start(rng, q, minv=None):
    """A free-theory equilibrium draw, c_A ~ N(0, 1/max(q_A, min_A M_A)).

    Use this, not phi = 0, to start a chain. The typical action is one half per
    mode, so reaching equilibrium from phi = 0 requires trajectories with
    dS ~ N/2, and the leapfrog energy error at a step size chosen for
    equilibrium then scales with the number of modes N. Past a few hundred
    modes a cold chain does not thermalise at all: every trajectory is
    rejected, while the zero-mode heat bath keeps drawing plausible values of
    c_0 against a frozen background. The only symptom is zero acceptance, which
    `run` reports separately for the thermalisation as `therm_acceptance`.

    The floor is applied to q, so a negative bare mass gives a finite, positive
    variance for the low modes. The zero mode is redrawn exactly by
    `zeromode.update` on the first iteration, so its starting value hardly
    matters.
    """
    q = np.asarray(q, dtype=float)
    minv = default_minv(q) if minv is None else minv
    q_eff = np.maximum(q, 1.0 / np.max(minv))
    return rng.standard_normal(q.size) / np.sqrt(q_eff)


def tune_eps(ctx, shell_index, n_max, *, eps0, n_steps, seed, minv=None,
             zero_mode_updates=0, n_cal=600, dH_cap=None, acc_floor=0.45,
             shrink=0.7, max_iter=12, verbose=False):
    """Reduce the step size until the largest |dH| of a calibration run is bounded.

    At strong coupling the Hessian of the quartic term is 3 lam4 phi(x)^2
    pointwise, and its maximum over the grid is far above the mode-averaged
    value the mass matrix is built from. The leapfrog integrator then becomes
    unstable on rare large-field configurations. Acceptance does not detect
    this: the unstable trajectories return a huge |dH|, the negative ones are
    always accepted, and each contributes exp(+huge) to <exp(-dH)>, which must
    equal 1. The calibration therefore targets max|dH| over the window, with
    the acceptance only as a secondary floor.

    The cap is matched to the length of the window: keeping <exp(-dH)> within
    0.9-1.1 over N samples needs exp(dH_max)/N < 0.1, i.e. dH_max < ln(0.1 N),
    which is 4.09 for the default n_cal = 600.

    Each attempt runs n_cal/2 thermalisation and n_cal measurement
    trajectories; if it fails, eps is multiplied by `shrink`. The trajectory
    length T = eps0 * n_steps is held fixed.

    Returns (eps, n_steps). The calibration run is discarded.
    """
    if dH_cap is None:
        dH_cap = float(np.log(0.1 * n_cal))      # 4.09 at n_cal = 600
    T = eps0 * n_steps
    eps = float(eps0)
    for _ in range(max_iter):
        ns = max(2, int(round(T / eps)))
        r = run(ctx, shell_index, n_max, eps=eps, n_steps=ns,
                n_therm=n_cal // 2, n_meas=n_cal, seed=seed, minv=minv,
                zero_mode_updates=zero_mode_updates, keep_samples=False)
        ok = (np.isfinite(r["dH_max"]) and r["dH_max"] <= dH_cap
              and r["acceptance"] >= acc_floor)
        if verbose:
            print(f"    eps={eps:.4f} n_steps={ns:3d} acc={r['acceptance']:.3f} "
                  f"max|dH|={r['dH_max']:.3g}", flush=True)
        if ok:
            return eps, ns
        eps *= shrink
    return eps, max(2, int(round(T / eps)))


def run(ctx, shell_index, n_max, *, eps, n_steps, n_therm, n_meas,
        seed=0, thin=1, coeffs0=None, minv=None, zero_mode_updates=0,
        vol=None, keep_samples=True, checkpoint_path=None,
        checkpoint_every=0, progress_every=0, randomize_traj=True,
        start="hot", observables=None):
    """Run one chain: n_therm thermalisation iterations, then n_meas measurements.

    Parameters
    ----------
    ctx : action.ActionContext
    shell_index : int array, the level n of each mode (`S4Transform.shell_index`)
    n_max : int, the cutoff (sets the length of the per-level arrays)
    eps, n_steps : leapfrog step size and nominal number of steps
    n_therm, n_meas, thin : thermalisation iterations, number of measurements,
        and iterations between measurements
    seed : seed of the numpy Generator that drives the whole chain
    minv : inverse HMC mass matrix; `action.default_minv(q)` if None
    zero_mode_updates : exact heat-bath draws of the constant mode after each
        trajectory. Each costs one mode-to-grid transform.
    keep_samples : store the full configurations (n_meas x n_modes floats)
    checkpoint_path, checkpoint_every : write the chain state atomically every
        `checkpoint_every` iterations; a chain restarted with the same arguments
        resumes from the file and reproduces the uninterrupted chain exactly.
        The file records the arguments, and a restart with different ones is
        refused.
    randomize_traj : draw the number of leapfrog steps uniformly in
        [0.7 n_steps, 1.3 n_steps] for each trajectory
    start : "hot" (free-theory draw, see `free_start`) or "cold" (phi = 0)
    observables : optional function of the coefficient vector, evaluated on
        every measured configuration and returning a 1-D float array. The
        results are stored as the series `observables` (n_meas x n_obs) and
        checkpointed with the chain, which avoids keeping the configurations.

    Returns
    -------
    dict with the per-measurement series

        c0            zero-mode coefficient c_0 = sqrt(Vol) * phibar
        phi2          (1/Vol) int phi^2 = (1/Vol) sum_A c_A^2
        action        S
        shell_series  mean of c_A^2 over each level n, shape (n_meas, n_max+1)
        samples       configurations, if keep_samples
        observables   values of `observables`, if given

    the averages `shell_power` and `mode_power`, and the diagnostics
    `acceptance`, `therm_acceptance` (thermalisation only), `zero_mode_acceptance`,
    `dH_mean`, `dH_max`, `exp_negdH` (all over every trajectory of the chain),
    `dH` (the energy change of every trajectory, in order) and `exp_negdH_meas`
    (over the trajectories after thermalisation).
    """
    Y, w, q, lam4 = unpack(ctx)
    if vol is None:
        vol = float(np.sum(w))
    minv = default_minv(q) if minv is None else minv
    sqrt_mass = np.sqrt(1.0 / minv)
    total_iter = n_therm + n_meas * thin
    n_modes = ctx.n_modes

    c0_arr = np.zeros(n_meas)
    phi2_arr = np.zeros(n_meas)
    act_arr = np.zeros(n_meas)
    # Power per level for every measurement. The autocorrelation differs
    # strongly between levels: the heat bath drives tau(c_0) to ~0.5 while the
    # nonzero modes stay more correlated, so errors need the per-level series.
    shell_series = np.zeros((n_meas, n_max + 1))
    shell_counts = np.zeros(n_max + 1)
    np.add.at(shell_counts, shell_index, 1.0)
    mode_sq = np.zeros(n_modes)
    samples = np.zeros((n_meas, n_modes)) if keep_samples else None
    obs_arr = None          # allocated at the first measurement, once the width is known
    dH_arr = np.full(total_iter, np.nan)

    # Everything a resumed chain must agree on. Stored in the checkpoint and
    # compared at resume: continuing with other arguments would silently
    # produce a chain that is neither the interrupted one nor a new one.
    settings = dict(eps=float(eps), n_steps=int(n_steps), n_therm=int(n_therm),
                    n_meas=int(n_meas), thin=int(thin), seed=int(seed),
                    zero_mode_updates=int(zero_mode_updates), keep_samples=bool(keep_samples),
                    randomize_traj=bool(randomize_traj), start=str(start),
                    n_modes=int(n_modes), lam4=float(lam4), vol=float(vol),
                    has_observables=observables is not None)

    rng = np.random.default_rng(seed)
    if coeffs0 is not None:
        c = np.array(coeffs0, dtype=float)
    elif start == "hot":
        c = free_start(rng, q, minv)
    elif start == "cold":
        c = np.zeros(n_modes)
    else:
        raise ValueError(f"start must be 'hot' or 'cold', got {start!r}")
    n_acc = n_prop = 0
    n_acc_therm = n_prop_therm = 0
    zm_acc = zm_prop = 0
    dH_sum = 0.0
    dH_exp_sum = 0.0
    dH_max = 0.0
    dH_n = 0
    it0 = 0
    n_stored = 0

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        ck = np.load(checkpoint_path, allow_pickle=True)
        if "settings" in ck.files:
            saved = json.loads(str(ck["settings"]))
            diff = [k for k in settings if saved.get(k) != settings[k]]
            if diff:
                raise ValueError(
                    f"checkpoint {checkpoint_path} was written with different arguments "
                    f"({', '.join(diff)}); resume with the original ones or remove the file")
            if not (np.allclose(ck["q"], q) and np.allclose(ck["minv"], minv)):
                raise ValueError(f"checkpoint {checkpoint_path} belongs to a different action "
                                 "or mass matrix; remove the file to start afresh")
        c = ck["c"]
        rng.bit_generator.state = ck["rng_state"].item()
        it0 = int(ck["it"]) + 1
        n_acc, n_prop = int(ck["n_acc"]), int(ck["n_prop"])
        zm_acc, zm_prop = int(ck["zm_acc"]), int(ck["zm_prop"])
        dH_sum, dH_exp_sum, dH_n = float(ck["dH_sum"]), float(ck["dH_exp_sum"]), int(ck["dH_n"])
        dH_max = float(ck["dH_max"])
        n_acc_therm = int(ck["n_acc_therm"])
        n_prop_therm = int(ck["n_prop_therm"])
        n_stored = int(ck["n_stored"])
        c0_arr[:n_stored] = ck["c0"][:n_stored]
        phi2_arr[:n_stored] = ck["phi2"][:n_stored]
        act_arr[:n_stored] = ck["act"][:n_stored]
        shell_series[:n_stored] = ck["shell_series"][:n_stored]
        mode_sq = ck["mode_sq"]
        dH_arr[:it0] = ck["dH"][:it0]
        if keep_samples and "samples" in ck:
            samples[:n_stored] = ck["samples"][:n_stored]
        if observables is not None and "observables" in ck.files:
            obs_arr = np.array(ck["observables"])

    def _save(it):
        if checkpoint_path is None:
            return
        tmp = str(checkpoint_path) + ".tmp.npz"
        payload = dict(c=c, rng_state=np.array(rng.bit_generator.state, dtype=object),
                       it=it, n_acc=n_acc, n_prop=n_prop, zm_acc=zm_acc,
                       zm_prop=zm_prop, dH_sum=dH_sum, dH_exp_sum=dH_exp_sum,
                       dH_n=dH_n, dH_max=dH_max, n_acc_therm=n_acc_therm,
                       n_prop_therm=n_prop_therm,
                       n_stored=n_stored, c0=c0_arr, phi2=phi2_arr,
                       act=act_arr, mode_sq=mode_sq, shell_series=shell_series,
                       dH=dH_arr, settings=json.dumps(settings), q=q, minv=minv)
        if keep_samples:
            payload["samples"] = samples
        if obs_arr is not None:
            payload["observables"] = obs_arr
        np.savez(tmp, **payload)
        os.replace(tmp, checkpoint_path)   # atomic: never a half-written file

    lo = max(1, int(round(0.7 * n_steps)))
    hi = max(lo + 1, int(round(1.3 * n_steps)))

    for it in range(it0, total_iter):
        p = sqrt_mass * rng.standard_normal(n_modes)
        # Randomised trajectory length: a mixture of valid kernels, so detailed
        # balance is untouched, and no mode sits at the T = pi resonance.
        ns = int(rng.integers(lo, hi + 1)) if randomize_traj else n_steps
        h0 = hamiltonian(c, p, Y, w, q, lam4, minv)
        c_new, p_new = leapfrog(c, p, eps, ns, Y, w, q, lam4, minv)
        h1 = hamiltonian(c_new, p_new, Y, w, q, lam4, minv)
        d = h1 - h0
        n_prop += 1
        dH_sum += d
        dH_exp_sum += np.exp(-d) if d > -700.0 else np.inf
        # A non-finite dH is an unstable trajectory and must fail the health
        # check; max() would silently ignore a NaN.
        dH_max = np.inf if not np.isfinite(d) else max(dH_max, abs(float(d)))
        dH_n += 1
        dH_arr[it] = d
        accepted = np.log(rng.random()) < -d
        if accepted:
            c, n_acc = c_new, n_acc + 1
        if it < n_therm:
            n_prop_therm += 1
            n_acc_therm += int(accepted)

        for _ in range(zero_mode_updates):
            zm_acc += zeromode.update(rng, c, Y, w, q, lam4, vol)
            zm_prop += 1

        if it >= n_therm and (it - n_therm) % thin == 0:
            k = (it - n_therm) // thin
            c0_arr[k] = c[0]
            phi2_arr[k] = float(np.dot(c, c)) / vol
            act_arr[k] = action(c, Y, w, q, lam4)
            mode_sq += c * c
            sp = np.zeros(n_max + 1)
            np.add.at(sp, shell_index, c * c)
            shell_series[k] = sp / shell_counts
            if keep_samples:
                samples[k] = c
            if observables is not None:
                o = np.atleast_1d(np.asarray(observables(c), dtype=float))
                if obs_arr is None:
                    obs_arr = np.zeros((n_meas, o.size))
                obs_arr[k] = o
            n_stored = k + 1

        if progress_every and (it + 1) % progress_every == 0:
            print(f"    iter {it+1}/{total_iter}  acc={n_acc/max(n_prop,1):.3f}",
                  flush=True)
        if checkpoint_every and (it + 1) % checkpoint_every == 0:
            _save(it)

    _save(total_iter - 1)

    mode_power = mode_sq / max(n_stored, 1)
    per_shell = np.zeros(n_max + 1)
    counts = np.zeros(n_max + 1)
    np.add.at(per_shell, shell_index, mode_power)
    np.add.at(counts, shell_index, 1.0)
    shell_power = per_shell / counts

    dH_meas = dH_arr[n_therm:]
    with np.errstate(over="ignore"):
        exp_negdH_meas = float(np.mean(np.exp(-dH_meas))) if dH_meas.size else float("nan")

    return {
        "samples": samples if keep_samples else np.zeros((0, n_modes)),
        "observables": obs_arr if obs_arr is not None else np.zeros((n_meas, 0)),
        "c0": c0_arr,
        "phi2": phi2_arr,
        "action": act_arr,
        "shell_series": shell_series,
        "acceptance": n_acc / max(n_prop, 1),
        "therm_acceptance": (n_acc_therm / n_prop_therm) if n_prop_therm else None,
        "zero_mode_acceptance": zm_acc / zm_prop if zm_prop else None,
        "shell_power": shell_power,
        "mode_power": mode_power,
        "dH": dH_arr,
        "dH_mean": dH_sum / max(dH_n, 1),
        "dH_max": dH_max,
        "dH_max_meas": float(np.max(np.abs(dH_meas))) if dH_meas.size else float("nan"),
        "exp_negdH": dH_exp_sum / max(dH_n, 1),
        "exp_negdH_meas": exp_negdH_meas,
        "action_last": float(action(c, Y, w, q, lam4)),
    }
