"""lattice-ds run-constrained: one chain with the zero mode held fixed at phibar.

HMC on the nonzero modes at fixed zero mode:

    lattice-ds run-constrained --tag n4_l1_m0.00_s1 --n-max 4 --lam 1 \\
        --m2 -0.7116 --phibar 0.0 --n-meas 100000 --outdir runs/constrained

The chain measures, for every configuration, the force f whose average is
V_C'(phibar), and the coefficients (A, B) that give the curvature
V_C''(0) = <B> - <A^2> at phibar = 0 and allow reweighting in the bare mass
(see `lattice_ds.constrained` and Appendix D of the paper). The output
<outdir>/<tag>.npz has the format of the published constrained chains.

Sampler settings are those of Set B: mass matrix max(lambda_n, lambda_1),
trajectory length T = 1.5, step size calibrated over 600 trajectories
(`hmc.tune_eps`), no zero-mode update.

Seeds. The calibration uses --seed and is discarded; the production chain is
then started afresh with seed + 100000. The published files record --seed.

To run many chains in parallel, start one process per chain (for example with
GNU parallel or xargs -P); each uses a single BLAS thread.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lattice_ds import action as A
from lattice_ds import constrained, hmc
from lattice_ds.stats import tau_int


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lattice-ds run-constrained", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="name of the output file")
    ap.add_argument("--n-max", type=int, required=True)
    ap.add_argument("--lam", type=float, required=True, help="quartic coupling")
    ap.add_argument("--m2", type=float, required=True, help="bare mass")
    ap.add_argument("--phibar", type=float, default=0.0, help="value at which the zero mode is held")
    ap.add_argument("--n-meas", type=int, default=10000)
    ap.add_argument("--n-therm", type=int, default=2000)
    ap.add_argument("--eps", type=float, default=0.25, help="starting step size of the calibration")
    ap.add_argument("--seed", type=int, default=420101)
    ap.add_argument("--checkpoint-every", type=int, default=2000)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args(argv)

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{a.tag}.npz"
    if out.exists():
        raise SystemExit(f"{out} exists; choose another --tag")
    t0 = time.monotonic()

    ctx, basis, eig, vol = constrained.context(a.n_max, a.m2, a.lam, a.phibar)
    minv = A.renormalised_minv(eig[1:])
    shell = basis.shell_index[1:] - 1          # levels 1..n_max, relabelled from 0
    eps, n_steps = hmc.tune_eps(ctx, shell, a.n_max - 1, eps0=a.eps, n_steps=max(2, round(1.5 / a.eps)),
                                seed=a.seed, minv=minv, zero_mode_updates=0, verbose=True)
    # (A, B, f) are measured on the fly, so the configurations are never stored.
    q_full = eig + a.m2

    def measure(c):
        return constrained.zero_mode_coefficients(ctx.Y.full(c), basis, ctx.w, q_full, a.lam, vol)

    r = hmc.run(ctx, shell, a.n_max - 1, eps=eps, n_steps=n_steps, n_therm=a.n_therm, n_meas=a.n_meas,
                seed=a.seed + 100000, minv=minv, zero_mode_updates=0, progress_every=1000,
                keep_samples=False, observables=measure,
                checkpoint_path=str(outdir / f"{a.tag}.checkpoint.npz"), checkpoint_every=a.checkpoint_every)
    Aser, Bser, force = r["observables"].T

    # Health is judged on the trajectories after thermalisation.
    therm = r["therm_acceptance"]
    ok = bool(r["acceptance"] >= 0.3 and (therm is None or therm >= 0.05)
              and 0.9 < r["exp_negdH_meas"] < 1.1 and r["dH_max_meas"] <= 25 and np.all(np.isfinite(r["dH"])))
    np.savez_compressed(
        out, A=Aser, B=Bser, force=force, phibar=a.phibar, m2=a.m2, lam4=a.lam, n_max=a.n_max, vol=vol,
        n_meas=a.n_meas, n_therm=a.n_therm, seed=a.seed, eps=eps, n_steps=n_steps,
        acceptance=r["acceptance"], therm_acceptance=-1.0 if therm is None else therm,
        exp_negdH=r["exp_negdH_meas"], dH_max=r["dH_max_meas"], health_pass=ok,
        shell_series=r["shell_series"])
    (outdir / f"{a.tag}.checkpoint.npz").unlink(missing_ok=True)

    summary = dict(tag=a.tag, n_max=a.n_max, lam=a.lam, m2=a.m2, phibar=a.phibar, eps=eps, n_steps=n_steps,
                   acceptance=r["acceptance"], exp_negdH=r["exp_negdH_meas"], dH_max=r["dH_max_meas"],
                   health_pass=ok, mean_force=float(force.mean()), tau_force=tau_int(force),
                   seconds=time.monotonic() - t0)
    if a.phibar == 0:
        summary["curvature_mR2"] = constrained.curvature(Aser, Bser)
    print(json.dumps(summary, indent=2))
    if not ok:
        raise SystemExit("sampler health check failed: do not use this chain")

