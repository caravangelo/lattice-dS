"""lattice-ds run: independent chains on a list of points (n_max, lambda, m^2), in parallel.

Each chain is HMC on all modes plus exact zero-mode updates. One .npz file is
written per chain, in the format of the chains published with the paper (reproduction/README.md):
the series c0 and phi2 per measurement, and the run parameters and diagnostics.

Bare mass fixed to the one-loop counterterm m^2 = -3 lambda G'(0), as in Set A of the paper:

    lattice-ds run --n-max 4 6 --lam 0.01 0.1 --one-loop-mass \\
        --seeds 2 --n-meas 25000 --outdir runs/setA

Explicit bare masses, renormalised mass matrix and calibrated step size, as in the Set B scans:

    lattice-ds run --n-max 4 --lam 20 --m2 -11.6 -11.5 -11.4 \\
        --precondition renorm --eps-auto --seeds 6 --n-meas 30000 --outdir runs/scan_n4_l20

Properties:

- Resumable. Every chain checkpoints atomically; a killed run resumes and
  reproduces the uninterrupted chain exactly.
- Idempotent. A finished chain is skipped if it was run with the settings now
  requested, and refused (never overwritten) if it was not.
- Health checked. A chain is flagged SUSPECT unless acceptance >= 0.3,
  thermalisation acceptance >= 0.05, 0.9 < <exp(-dH)> < 1.1 and max|dH| <= 25.
- One BLAS thread per worker: the parallelism is across chains (set by `lattice_ds.cli`).

Seeds. Chain i_seed of the point (n_max, m^2, lambda) is seeded by
numpy.random.SeedSequence([base_seed, n_max, round(1e9 (m^2 + 1000)), round(1e9 lambda), i_seed]),
so chains at different points, or from different invocations with the same
--base-seed, never share a seed. --seed runs one point with explicitly given
seeds, for example to re-run a published chain from the seed recorded in its
file. With --eps-auto the same seed drives the
calibration and then, restarted, the production chain.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

import lattice_ds as L
from lattice_ds import action as A
from lattice_ds import hmc
from lattice_ds import perturbation as P


def chain_tag(n_max, m2, lam4, seed):
    return f"nmax{n_max}_m2{m2:g}_lam{lam4:g}_seed{seed}"


def derive_seed(base, n_max, m2, lam4, i_seed):
    """Seed of chain i_seed at the point (n_max, m2, lam4): a function of the point itself."""
    # SeedSequence wants non-negative integers: the mass is shifted by 10^3 (bare masses are > -1000).
    key = [int(base), int(n_max), int(round(1e9 * (float(m2) + 1000.0))), int(round(1e9 * float(lam4))), int(i_seed)]
    if min(key) < 0:
        raise ValueError(f"cannot derive a seed for n_max={n_max}, m2={m2}, lam4={lam4}, base={base}")
    return int(np.random.SeedSequence(key).generate_state(1, dtype=np.uint32)[0])


def healthy(acc, therm_acc, exp_negdH, dH_max):
    # therm_acc is None (or -1.0 in a file) when the chain had no thermalisation.
    therm_acc = 1.0 if therm_acc is None or therm_acc < 0 else therm_acc
    return bool(acc >= 0.30 and therm_acc >= 0.05 and np.isfinite(exp_negdH)
                and 0.9 < exp_negdH < 1.1 and np.isfinite(dH_max) and dH_max <= 25.0)


def run_one(job):
    n_max, m2, lam4, seed, outdir = job["n_max"], job["m2"], job["lam4"], job["seed"], Path(job["outdir"])
    tag = chain_tag(n_max, m2, lam4, seed)
    out = outdir / f"{tag}.npz"
    settings = {k: job[k] for k in ("n_max", "m2", "lam4", "seed", "n_therm", "n_meas", "eps", "n_steps",
                                    "zero_mode_updates", "precondition", "eps_auto", "randomize_traj")}
    if out.exists():
        z = np.load(out, allow_pickle=True)
        got = json.loads(str(z["settings"]))
        if got != settings:
            diff = [k for k in settings if got.get(k) != settings[k]]
            return dict(tag=tag, status="MISMATCH", detail=",".join(diff))
        ok = healthy(float(z["acceptance"]), float(z["therm_acceptance"]), float(z["exp_negdH"]), float(z["dH_max"]))
        return dict(tag=tag, status="skipped" if ok else "SUSPECT",
                    acceptance=float(z["acceptance"]), exp_negdH=float(z["exp_negdH"]))

    t0 = time.time()
    ctx, basis = L.setup(n_max, m2, lam4)
    minv = A.renormalised_minv(ctx.lam, 0.0) if job["precondition"] == "renorm" else None
    eps, n_steps = job["eps"], job["n_steps"]
    if job["eps_auto"]:
        eps, n_steps = hmc.tune_eps(ctx, basis.shell_index, n_max, eps0=eps, n_steps=n_steps, seed=seed,
                                    minv=minv, zero_mode_updates=job["zero_mode_updates"])
    r = hmc.run(ctx, basis.shell_index, n_max, eps=eps, n_steps=n_steps, minv=minv,
                n_therm=job["n_therm"], n_meas=job["n_meas"], seed=seed,
                zero_mode_updates=job["zero_mode_updates"], keep_samples=False,
                checkpoint_path=str(outdir / f"{tag}.checkpoint.npz"),
                checkpoint_every=job["checkpoint_every"], randomize_traj=job["randomize_traj"])
    therm = r["therm_acceptance"]
    np.savez_compressed(
        out, c0=r["c0"], phi2=r["phi2"], action=r["action"], shell_series=r["shell_series"],
        shell_power=r["shell_power"], mode_power=r["mode_power"], shell_index=basis.shell_index,
        vol=float(ctx.w.sum()), n_max=n_max, m2=m2, lam4=lam4, seed=seed,
        n_meas=job["n_meas"], n_therm=job["n_therm"], eps=eps, n_steps=n_steps,
        acceptance=r["acceptance"], therm_acceptance=-1.0 if therm is None else therm,
        zero_mode_acceptance=-1.0 if r["zero_mode_acceptance"] is None else r["zero_mode_acceptance"],
        exp_negdH=r["exp_negdH"],
        exp_negdH_meas=r["exp_negdH_meas"], dH_max=r["dH_max"],
        settings=json.dumps(settings), minutes=(time.time() - t0) / 60.0)
    (outdir / f"{tag}.checkpoint.npz").unlink(missing_ok=True)
    ok = healthy(r["acceptance"], therm, r["exp_negdH"], r["dH_max"])
    return dict(tag=tag, status="done" if ok else "SUSPECT", acceptance=r["acceptance"],
                exp_negdH=r["exp_negdH"], minutes=(time.time() - t0) / 60.0)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lattice-ds run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-max", type=int, nargs="+", help="cutoffs (required unless --mass-table)")
    ap.add_argument("--lam", type=float, nargs="+", help="quartic couplings (required unless --mass-table)")
    mass = ap.add_mutually_exclusive_group(required=True)
    mass.add_argument("--m2", type=float, nargs="+", help="bare masses (every one is run at every coupling)")
    mass.add_argument("--one-loop-mass", action="store_true",
                      help="m^2 = -3 lambda G'(0; n_max), which makes m_R^2 vanish at one loop")
    mass.add_argument("--mass-table", help="JSON list of {n_max, lam4, m2[, seeds]}; replaces --n-max/--lam/--m2")
    ap.add_argument("--seeds", type=int, default=2, help="independent chains per point")
    ap.add_argument("--base-seed", type=int, default=260910)
    ap.add_argument("--seed", type=int, nargs="+", help="explicit seeds, one chain each; a single point only")
    ap.add_argument("--n-meas", type=int, default=20000)
    ap.add_argument("--n-therm", type=int, default=2000)
    ap.add_argument("--eps", type=float, default=0.25, help="leapfrog step size")
    ap.add_argument("--n-steps", type=int, default=6, help="nominal leapfrog steps; T = n_steps * eps = 1.5 by default")
    ap.add_argument("--eps-auto", action="store_true",
                    help="calibrate eps per chain at fixed T (hmc.tune_eps); needed at strong coupling")
    ap.add_argument("--precondition", choices=("bare", "renorm"), default="bare",
                    help="HMC mass matrix: 'bare' = max(lambda_n + m^2, floor) (Set A); "
                         "'renorm' = max(lambda_n, lambda_1) (Set B, required once the bare mass is large and negative)")
    ap.add_argument("--zero-mode-updates", type=int, default=2)
    ap.add_argument("--fixed-traj", action="store_true", help="disable the randomised trajectory length (not advised)")
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dry-run", action="store_true", help="list the chains and exit")
    a = ap.parse_args(argv)
    if not a.mass_table and not (a.n_max and a.lam):
        ap.error("--n-max and --lam are required unless --mass-table is given")
    if a.seed and (a.mass_table or len(a.n_max) != 1 or len(a.lam) != 1 or (a.m2 and len(a.m2) != 1)):
        ap.error("--seed runs a single point: one --n-max, one --lam and one --m2 (or --one-loop-mass)")

    common = dict(outdir=a.outdir, n_therm=a.n_therm, n_meas=a.n_meas, eps=a.eps, n_steps=a.n_steps,
                  zero_mode_updates=a.zero_mode_updates, precondition=a.precondition, eps_auto=a.eps_auto,
                  randomize_traj=not a.fixed_traj, checkpoint_every=a.checkpoint_every)
    jobs = []
    if a.mass_table:
        for e in json.loads(Path(a.mass_table).read_text()):
            n_max, lam4, m2 = int(e["n_max"]), float(e["lam4"]), float(e["m2"])
            for s in range(int(e.get("seeds", a.seeds))):
                jobs.append(dict(common, n_max=n_max, lam4=lam4, m2=m2, seed=derive_seed(a.base_seed, n_max, m2, lam4, s)))
    else:
        for n_max in a.n_max:
            for m2 in ([None] if a.one_loop_mass else a.m2):
                for lam4 in a.lam:
                    this_m2 = P.one_loop_bare_mass(n_max, lam4) if a.one_loop_mass else m2
                    seeds = a.seed or [derive_seed(a.base_seed, n_max, this_m2, lam4, s) for s in range(a.seeds)]
                    for seed in seeds:
                        jobs.append(dict(common, n_max=n_max, lam4=lam4, m2=this_m2, seed=seed))

    print(f"{len(jobs)} chains -> {a.outdir}  ({a.workers} workers)")
    if a.dry_run:
        for j in jobs:
            print("  ", chain_tag(j["n_max"], j["m2"], j["lam4"], j["seed"]))
        return 0
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = []
    with mp.Pool(a.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_one, jobs), 1):
            results.append(res)
            print(f"[{i}/{len(jobs)}] {res['status']:8s} {res['tag']}"
                  + (f"  acc={res['acceptance']:.3f} <exp(-dH)>={res['exp_negdH']:.4f}" if "acceptance" in res else "")
                  + (f"  ({res['detail']} differ)" if "detail" in res else ""), flush=True)
    bad = [r for r in results if r["status"] in ("MISMATCH", "SUSPECT")]
    print(f"finished in {(time.time() - t0) / 60:.1f} min; {len(bad)} chain(s) need attention")
    if any(r["status"] == "MISMATCH" for r in results):
        print("MISMATCH: an existing file was run with different settings. It was not reused "
              "and not overwritten; use a fresh --outdir.")
    return 1 if bad else 0

