"""Recompute the tables of the paper from the published chains, and compare.

The Monte Carlo chains are archived on Zenodo (see the README). After
downloading and unpacking them,

    python reproduction/reproduce_tables.py --chains /path/to/chains setA
    python reproduction/reproduce_tables.py --chains /path/to/chains setA-coupling
    python reproduction/reproduce_tables.py --chains /path/to/chains setB [--points 8:20 6:60] [--jobs 4]

recompute reproduction/data/{setA_points, setA_effective_coupling, setB_points}.csv
and print, for every column, the largest deviation from the published table.
The recomputed tables are written to --out (default: recomputed/); the
published ones are never modified.

What to expect. Every number is a deterministic function of the chains,
including the bootstrap errors of Set B, whose resampling is seeded. The
recomputed tables are written with the same rounding as the published ones
(FORMATS below) and reproduce them to within that rounding; the script
prints the largest deviation column by column, and how many values are
identical once rounded.

Set A takes seconds. Set B takes one to ten minutes per point on one core,
about two hours in total; --jobs runs points in parallel.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

from lattice_ds import meanforce
from lattice_ds.grid import VOL_S4 as OM
from lattice_ds.perturbation import SY_BINDER, SY_PHIBAR2, screening_coefficient
from lattice_ds.stats import binder, binder_jackknife_error, error_of_mean, tau_int

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(HERE, "data")


def read_table(name):
    with open(os.path.join(TABLES, name)) as fh:
        return list(csv.DictReader(fh))


def tag_of(path):
    return os.path.basename(path)[:-4]


# ============================================================== Set A

def seta_chains(chains, n, lam):
    return sorted(glob.glob(os.path.join(chains, "unconstrained", f"setA_n{n}_lam{lam:g}_seed*.npz")))


def seta_point(chains, n, lam):
    """<phibar^2> and U at one Set A point: inverse-variance mean over the chains."""
    c2, e2, U, eU, n_meas, m2 = [], [], [], [], None, None
    files = seta_chains(chains, n, lam)
    if not files:
        raise FileNotFoundError(f"no Set A chains for n_max={n}, lambda={lam:g} in {chains}/unconstrained "
                                f"(expected setA_n{n}_lam{lam:g}_seed*.npz)")
    for p in files:
        z = np.load(p)
        c0 = z["c0"]
        sq = c0 ** 2
        c2.append(float(np.mean(sq)))
        e2.append(error_of_mean(sq, tau_int(sq)))
        U.append(binder(c0))
        eU.append(binder_jackknife_error(c0))
        n_meas = n_meas or int(z["n_meas"])
        m2 = m2 if m2 is not None else float(z["m2"])
    c2, e2, U, eU = map(np.array, (c2, e2, U, eU))
    wv, wU = 1 / e2 ** 2, 1 / eU ** 2
    return dict(n_max=n, lam=lam, m2_bare=m2, n_chains=len(c2), n_meas_per_chain=n_meas,
                phibar2=(c2 * wv).sum() / wv.sum() / OM, phibar2_err=wv.sum() ** -0.5 / OM,
                phibar2_SY=SY_PHIBAR2 / math.sqrt(lam),
                U=(U * wU).sum() / wU.sum(), U_err=wU.sum() ** -0.5, U_SY=SY_BINDER,
                c0_2_plain_mean=float(c2.mean()))


def do_seta(a):
    pub = read_table("setA_points.csv")
    rows = [dict(seta_point(a.chains, int(r["n_max"]), float(r["lam"])), role=r.get("role", "Set A")) for r in pub]
    write(a.out, "setA_points.csv", rows,
          ["n_max", "lam", "m2_bare", "n_chains", "n_meas_per_chain", "phibar2", "phibar2_err",
           "phibar2_SY", "U", "U_err", "U_SY", "role"])
    compare(pub, rows, [("m2_bare", None), ("n_chains", None), ("n_meas_per_chain", None),
                        ("phibar2", "phibar2_err"), ("phibar2_err", "rel"), ("phibar2_SY", None),
                        ("U", "U_err"), ("U_err", "rel")])


def do_seta_coupling(a):
    """Effective coupling at the Set A points from three constrained chains around sigma."""
    pub = read_table("setA_effective_coupling.csv")
    rows = []
    for r in pub:
        n, lam = int(r["n_max"]), float(r["lam"])
        sigma = math.sqrt(seta_point(a.chains, n, lam)["c0_2_plain_mean"] / OM)
        files = sorted(glob.glob(os.path.join(a.chains, "constrained", f"F6_n{n}_l{lam:g}_f*.npz")))
        ye = meanforce.effective_coupling_fixed_mass(meanforce.load_constrained(files), sigma, n)
        rows.append(dict(n_max=n, lam=lam, x=screening_coefficient(n) * lam, y=ye["y_eff"],
                         y_err=ye["y_eff_err"], local_exponent=ye["slope"], sigma=sigma))
    write(a.out, "setA_effective_coupling.csv", rows, ["n_max", "lam", "x", "y", "y_err", "local_exponent", "sigma"])
    compare(pub, rows, [("x", None), ("y", "y_err"), ("y_err", "rel")])


# ============================================================== Set B

# Which chains enter each Set B point, and which fits are made.
#
#   constrained   name prefixes of the constrained chains (files <prefix>_*.npz)
#   unconstrained "pool": every unconstrained scan chain of this (n_max, lambda) at
#                 the bare mass nearest to that of the constrained chains,
#                 concatenated; otherwise the name of one unconstrained chain
#   central       (order, window) of the fit that gives the quoted values
#   alt           (order, window) of the second determination of m_*^2, which gives
#                 the fit-window systematic error
#
# Bootstrap replicas: 150 for the default layout, 120 for the dense one.
#
# Default layout (lambda <= 20 at n_max = 4, <= 10 at n_max = 6 and 8, <= 5 at
# n_max = 10): chains "F9" at phibar = 0, 0.10, 0.15 (two each) and "F1" at
# 0.8, 1.0, 1.2 sigma; window 1.25 sigma (the `fit_window` column of the table).
# The points listed here have a denser layout and fixed windows.
DENSE = {
    (4, 30.0): dict(constrained=["T2_l30"], unconstrained="force_T1_l30", central=(3, 0.4), alt=(2, 0.15)),
    (4, 40.0): dict(constrained=["T2_l40"], unconstrained="pool", central=(3, 0.4), alt=(2, 0.15)),
    (4, 60.0): dict(constrained=["F5_n4_l60"], unconstrained="pool", central=(3, 0.4), alt=(2, 0.15)),
    (4, 80.0): dict(constrained=["F4_n4_l80", "F5_n4_l80", "F8_n4_l80"], unconstrained="force_F3_unc_n4_l80", central=(3, 0.4), alt=(2, 0.15)),
    (4, 120.0): dict(constrained=["F4_n4_l120", "F5_n4_l120", "F8_n4_l120"], unconstrained="force_F3_unc_n4_l120", central=(3, 0.4), alt=(2, 0.15)),
    (6, 15.0): dict(constrained=["F2_n6_l15"], unconstrained="pool", central=(3, 0.5), alt=(3, 0.4)),
    (6, 20.0): dict(constrained=["F2_n6_l20"], unconstrained="pool", central=(3, 0.5), alt=(3, 0.4)),
    (6, 30.0): dict(constrained=["F7_n6_l30"], unconstrained="force_F7_unc_n6_l30", central=(3, 0.5), alt=(2, 0.15)),
    (6, 40.0): dict(constrained=["F2_n6_l40", "F5_n6_l40"], unconstrained="pool", central=(3, 0.5), alt=(2, 0.15)),
    (6, 60.0): dict(constrained=["F4_n6_l60", "F5_n6_l60", "F8_n6_l60"], unconstrained="force_F8_unc_n6_l60", central=(3, 0.5), alt=(2, 0.15)),
    (8, 20.0): dict(constrained=["F4_n8_l20", "F5_n8_l20", "F8_n8_l20"], unconstrained="force_F3_unc_n8_l20", central=(3, 0.5), alt=(2, 0.15)),
    (10, 15.0): dict(constrained=["F4_n10_l15", "F5_n10_l15", "F8_n10_l15"], unconstrained="pool", central=(3, 0.5), alt=(2, 0.15)),
}


def setb_config(n, lam, fit_window):
    if (n, lam) in DENSE:
        return DENSE[(n, lam)]
    return dict(constrained=[f"F1_n{n}_l{lam:g}", f"F9_n{n}_l{lam:g}"], unconstrained="pool",
                central=(3, fit_window), alt=(2, 0.15))


def pooled_scan_chains(chains, n, lam, m2):
    """Unconstrained chains of (n, lam) at the bare mass nearest to m2.

    The candidates are the Set B mass scans and, where a Set A point exists at
    the same (n, lam), its chains at the one-loop mass.
    """
    unc = os.path.join(chains, "unconstrained")
    files = (glob.glob(os.path.join(unc, f"setB_nmax{n}_m2*_lam{lam:g}_seed*.npz"))
             + glob.glob(os.path.join(unc, f"setA_n{n}_lam{lam:g}_seed*.npz")))
    by_mass = {}
    for p in sorted(files):
        by_mass.setdefault(round(float(np.load(p)["m2"]), 5), []).append(p)
    if not by_mass:
        raise FileNotFoundError(f"no unconstrained scan chains for n_max={n}, lambda={lam:g}")
    return by_mass[min(by_mass, key=lambda x: abs(x - m2))]


def setb_inputs(chains, n, lam, cfg):
    files = sorted({p for pre in cfg["constrained"]
                    for p in glob.glob(os.path.join(chains, "constrained", pre + "_[mf]*.npz"))}, key=tag_of)
    con = meanforce.load_constrained(files)
    if cfg["unconstrained"] == "pool":
        ufiles = pooled_scan_chains(chains, n, lam, min(c["mass"] for c in con))
    else:
        ufiles = [os.path.join(chains, "unconstrained", cfg["unconstrained"] + ".npz")]
    return files, ufiles


def setb_point(args):
    chains, n, lam, fit_window, n_boot_override = args
    cfg = setb_config(n, lam, fit_window)
    files, ufiles = setb_inputs(chains, n, lam, cfg)
    nb = 120 if (n, lam) in DENSE else 150
    out = {}
    for name, (order, xmax) in (("central", cfg["central"]), ("alt", cfg["alt"])):
        con = meanforce.load_constrained(files)          # fresh copy: the fit stores weights in it
        unc = meanforce.load_unconstrained(ufiles)
        out[name] = meanforce.massless_point(con, unc, order=order, xmax=xmax,
                                             n_boot=n_boot_override or nb, seed=3)
    c, alt = out["central"], out["alt"]
    x = screening_coefficient(n) * lam
    row = dict(n_max=n, lam=lam, x=x, y_chain=x / (1 + x), y=c["y_eff"], y_err_stat=c["y_eff_err"],
               y_err_window=abs(c["y_eff"] - alt["y_eff"]) / 2, U=c["U"], U_err=c["U_err"], U_SY=SY_BINDER,
               sigma=c["sigma"], m2_star=c["mass_retuned"], m2_star_err=c["delta_err"], fit_window=c["xmax"],
               local_exponent=c["slope"], y_taylor=c["y_taylor"], y_taylor_err=c["y_taylor_err"],
               chi2=c["fit_chi2"], dof=c["fit_dof"], n_constrained=c["n_chains"], n_unconstrained_files=len(ufiles),
               ess_unconstrained=c["ess_unconstrained"], m2_star_alt=alt["mass_retuned"])
    print(f"  done n_max={n} lambda={lam:g}: y = {row['y']:.4f} +- {row['y_err_stat']:.4f} +- {row['y_err_window']:.4f}, "
          f"m2* = {row['m2_star']:.5f}", flush=True)
    return row


SETB_COLUMNS = ["n_max", "lam", "x", "y_chain", "y", "y_err_stat", "y_err_window", "U", "U_err", "U_SY", "sigma",
                "m2_star", "m2_star_err", "fit_window", "local_exponent", "y_taylor", "y_taylor_err",
                "m2_crossing", "m2_crossing_err", "used_in_paper",
                "chi2", "dof", "n_constrained", "n_unconstrained_files", "ess_unconstrained", "m2_star_alt"]


def do_setb(a):
    pub = read_table("setB_points.csv")
    if a.points:
        want = {(int(p.split(":")[0]), float(p.split(":")[1])) for p in a.points}
        pub = [r for r in pub if (int(r["n_max"]), float(r["lam"])) in want]
    jobs = [(a.chains, int(r["n_max"]), float(r["lam"]), float(r["fit_window"]), a.n_boot) for r in pub]
    if a.jobs > 1:
        with Pool(a.jobs) as pool:
            rows = pool.map(setb_point, jobs, chunksize=1)
    else:
        rows = [setb_point(j) for j in jobs]
    # The histogram estimate of the massless point comes from fit_histograms.py
    # (data/histogram_crossing.csv, usable points only); used_in_paper is a flag of the published table.
    hist = {(int(h["n_max"]), float(h["lam"])): h for h in read_table("histogram_crossing.csv")}
    for r, p in zip(rows, pub):
        h = hist.get((int(p["n_max"]), float(p["lam"])), {})
        ok = h.get("usable") == "1"
        r.update(m2_crossing=h["m2_crossing"] if ok else "", m2_crossing_err=h["m2_crossing_err"] if ok else "",
                 used_in_paper=p.get("used_in_paper", ""))
    write(a.out, "setB_points.csv", rows, SETB_COLUMNS)
    compare(pub, rows, [("x", None), ("y_chain", None), ("y", "y_err_stat"), ("y_err_stat", "rel"),
                        ("y_err_window", "y_err_stat"), ("U", "U_err"), ("U_err", "rel"), ("sigma", None),
                        ("m2_star", "m2_star_err"), ("m2_star_err", "rel"), ("local_exponent", None),
                        ("y_taylor", "y_taylor_err"), ("y_taylor_err", "rel")])


# ============================================================== output

# Rounding of the published tables (data/*.csv), column by column; floats in
# columns not listed here are written with 10 significant digits.
FORMATS = {
    "m2_bare": ".10g", "phibar2": ".6g", "phibar2_err": ".3g", "phibar2_SY": ".6g",
    "U": ".5f", "U_err": ".5f", "U_SY": ".5f",
    "x": ".5g", "y": ".5g", "y_err": ".3g", "y_chain": ".5f", "y_err_stat": ".3g", "y_err_window": ".3g",
    "sigma": ".5f", "m2_star": ".6g", "m2_star_err": ".3g", "local_exponent": ".3f",
    "y_taylor": ".4g", "y_taylor_err": ".3g", "chi2": ".3g", "ess_unconstrained": ".4g", "m2_star_alt": ".6g",
}


def fmt(col, v):
    return f"{v:{FORMATS.get(col, '.10g')}}" if isinstance(v, float) else v


def write(outdir, name, rows, cols):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([fmt(c, r[c]) for c in cols])
    print(f"wrote {path}")


def compare(pub, rows, spec):
    """Largest deviation per column. Unit: the published error column, 'rel' (relative),
    or None (relative to the rounding of the published number)."""
    print(f"\n{'column':16s} {'max deviation':>14s}  unit                at (n_max, lambda)")
    for col, unit in spec:
        worst, where = 0.0, None
        for p, r in zip(pub, rows):
            if p[col] == "":
                continue
            pv, rv = float(p[col]), float(r[col])
            if unit is None:
                digits = len(p[col].split("e")[0].replace("-", "").replace(".", "").lstrip("0")) or 1
                d = abs(rv - pv) / (abs(pv) * 10 ** (-digits + 1) / 2 + 1e-300)   # in units of half the last printed digit
            elif unit == "rel":
                d = abs(rv / pv - 1)
            else:
                d = abs(rv - pv) / float(p[unit])
            if d >= worst:
                worst, where = d, (p["n_max"], p["lam"])
        label = {None: "last printed digit/2", "rel": "relative"}.get(unit, f"published {unit}")
        print(f"{col:16s} {worst:14.3g}  {label:20s}{where}")
    same = tot = 0
    for p, r in zip(pub, rows):
        for col, _ in spec:
            if p[col] == "":
                continue
            tot += 1
            same += str(fmt(col, r[col])) == p[col] or float(fmt(col, r[col])) == float(p[col])
    print(f"\n{same} of {tot} values identical to the published table after rounding to its format"
          + ("" if same == tot else "; inspect the differing values above (a different platform can change the"
             " last printed digit of a few values, anything larger needs an explanation)") + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["setA", "setA-coupling", "setB"])
    ap.add_argument("--chains", required=True, help="directory holding constrained/ and unconstrained/")
    ap.add_argument("--out", default="recomputed")
    ap.add_argument("--points", nargs="*", help="Set B only: restrict to points n_max:lambda, e.g. 8:20")
    ap.add_argument("--n-boot", type=int, default=None, help="Set B only: override the number of bootstrap replicas")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    if not os.path.isdir(os.path.join(a.chains, "constrained")):
        sys.exit(f"{a.chains} does not contain constrained/ and unconstrained/")
    {"setA": do_seta, "setA-coupling": do_seta_coupling, "setB": do_setb}[a.what](a)


if __name__ == "__main__":
    main()
