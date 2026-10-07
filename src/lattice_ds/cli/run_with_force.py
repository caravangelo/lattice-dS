"""lattice-ds run-with-force: one unconstrained chain that also records the force on the zero mode.

    lattice-ds run-with-force --tag unc_n6_l30 --n-max 6 --lam 30 \\
        --m2 -28.8 --n-meas 50000 --outdir runs/force

Same sampler as `lattice-ds run` with the Set B settings (mass matrix
max(lambda_n, lambda_1), calibrated step size, two zero-mode updates per
trajectory), but the configurations are kept in memory so that the force
f = (1/Omega) dS/dphibar can be evaluated on each of them. Binned in phibar,
<f | phibar> is V_C'(phibar), which cross-checks the constrained chains. In the
paper these chains (files force_*.npz) provide sigma and U at the Set B points
(6, 30), (6, 60), (8, 20), (4, 30) and at the points beyond lambda = 60.

Seeds: as in `lattice-ds run-constrained`, the calibration uses --seed and the
production chain seed + 100000.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import lattice_ds as L
from lattice_ds import action as A
from lattice_ds import constrained, hmc
from lattice_ds.stats import tau_int


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lattice-ds run-with-force", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-max", type=int, required=True)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--m2", type=float, required=True)
    ap.add_argument("--n-meas", type=int, default=100000)
    ap.add_argument("--n-therm", type=int, default=2000)
    ap.add_argument("--eps", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=430001)
    ap.add_argument("--zero-mode-updates", type=int, default=2)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args(argv)

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{a.tag}.npz"
    if out.exists():
        raise SystemExit(f"{out} exists; choose another --tag")
    t0 = time.monotonic()

    ctx, basis = L.setup(a.n_max, a.m2, a.lam)
    vol = float(ctx.w.sum())
    minv = A.renormalised_minv(ctx.lam, 0.0)
    eps, n_steps = hmc.tune_eps(ctx, basis.shell_index, a.n_max, eps0=a.eps, n_steps=max(2, round(1.5 / a.eps)),
                                seed=a.seed, minv=minv, zero_mode_updates=a.zero_mode_updates, verbose=True)
    r = hmc.run(ctx, basis.shell_index, a.n_max, eps=eps, n_steps=n_steps, n_therm=a.n_therm, n_meas=a.n_meas,
                seed=a.seed + 100000, minv=minv, zero_mode_updates=a.zero_mode_updates, keep_samples=True,
                progress_every=5000)
    force = np.array([constrained.zero_mode_coefficients(c, basis, ctx.w, ctx.quadratic, a.lam, vol)[2]
                      for c in r["samples"]])
    np.savez_compressed(
        out, c0=r["c0"], phi2=r["phi2"], force=force, shell_series=r["shell_series"], vol=vol,
        n_max=a.n_max, m2=a.m2, lam4=a.lam, seed=a.seed, n_meas=a.n_meas, n_therm=a.n_therm,
        eps=eps, n_steps=n_steps, acceptance=r["acceptance"], exp_negdH=r["exp_negdH"],
        exp_negdH_meas=r["exp_negdH_meas"], dH_max=r["dH_max"])
    print(json.dumps(dict(tag=a.tag, eps=eps, n_steps=n_steps, acceptance=r["acceptance"],
                          exp_negdH=r["exp_negdH"], dH_max=r["dH_max"], tau_c0=tau_int(r["c0"]),
                          seconds=time.monotonic() - t0), indent=2))

