"""Histogram estimate of the massless point from the unconstrained mass scans (Appendix D3).

At each Set B point (n_max, lambda) the zero-mode histograms of the mass-scan chains are
fitted jointly to

    -ln P_j(phibar) = Omega/2 a(m^2_j) phibar^2 + Omega/4 lambda_R(m^2_j) phibar^4
                      + g6 phibar^6 [+ g8 phibar^8] + c_j ,
    a(m^2) = b + s m^2 [+ q m^4],   lambda_R(m^2) = lambda_R [+ lambda_R' (m^2 - m_ref)],

one normalisation c_j per bare mass, and the massless point m^2_hist is the root of
a(m^2) = 0 nearest the scan. The fit is linear in every parameter (weighted least squares
on -ln of the bin frequencies, bins with more than 30 counts). Model selection is
pre-declared and ordered: the quadratic response, the phi^8 term and the mass running of
lambda_R are each adopted only if a nested test, calibrated by a parametric bootstrap
(multinomial draws from the fitted smaller model), rejects the smaller model at the 1%
level; a point is usable only if the calibrated goodness-of-fit p-value of the selected
model is above 1% and the design is well conditioned. Errors are a chain-level bootstrap,
stratified by mass.

Which chains: at each point, the dense production scan, i.e. every bare mass with at least
MIN_CHAINS chains (this excludes the two-chain wide scans that bracket the window), inside
the window of WINDOWS where one is given (the strong-coupling points at lambda = 40, 80
also carry a coarse scan of 12-chain masses far from the massless point, which the
response model does not describe). One-loop-mass chains (setA_* files) are not used.

    python reproduction/fit_histograms.py --chains reproduction/chains [--points 4:20 8:5] [--out recomputed/]

writes histogram_crossing.csv (n_max, lam, m2_crossing, m2_crossing_err, usable, model,
n_masses, n_chains); reproduce_tables.py takes the m2_crossing columns of setB_points.csv
from data/histogram_crossing.csv.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

from lattice_ds.grid import VOL_S4 as V

P_SEL = 0.01          # nested-test threshold for adopting a larger model
P_GOF = 0.01          # calibrated goodness-of-fit below this -> unusable
COND_MAX = 1e10       # design condition number above this -> non-identifiable
U_ORDER_F = 10.0      # adopt quadratic U interpolation when F exceeds this
MIN_CHAINS = 6        # a bare mass enters the fit with at least this many chains
WINDOWS = {(4, 40.0): (-25.0, -18.5), (4, 80.0): (-47.5, -35.0), (6, 40.0): (-40.5, -31.5)}
QMAX, NBINS, N_NULL, N_BOOT, SEED = 0.90, 60, 120, 300, 20260914


# ------------------------------------------------------------------ loading

def load_point(chains, n_max, lam, min_chains=MIN_CHAINS, window=None):
    """{bare mass: [phibar series per chain]} of the dense scan at (n_max, lambda)."""
    by_mass = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(chains, "unconstrained", f"setB_nmax{n_max}_m2*_lam{lam:g}_seed*.npz"))):
        z = np.load(p)
        if int(z["n_max"]) != n_max or abs(float(z["lam4"]) - lam) > 1e-9:
            continue
        m2 = round(float(z["m2"]), 10)
        if window is not None and not (window[0] <= m2 <= window[1]):
            continue
        by_mass[m2].append(np.asarray(z["c0"], dtype=float) / np.sqrt(float(z["vol"])))
    return {m: v for m, v in by_mass.items() if len(v) >= min_chains}


# ------------------------------------------------------------ histogram fit

def prepare(by_mass, qmax, nbins):
    masses = sorted(by_mass)
    allx = np.concatenate([np.concatenate(v) for v in by_mass.values()])
    hi = float(np.quantile(np.abs(allx), qmax))
    edges = np.linspace(0.0, hi, nbins + 1)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    H, M = {}, {}
    for m2 in masses:
        H[m2] = np.array([np.histogram(np.abs(x), bins=edges)[0] for x in by_mass[m2]], dtype=float)
        M[m2] = np.array([[x.size, np.sum(x ** 2), np.sum(x ** 4)] for x in by_mass[m2]], dtype=float)
    return dict(masses=masses, ctr=ctr, H=H, M=M, hi=hi, mref=float(np.mean(masses)))


class Model:
    def __init__(self, response=1, potential=6, lamR_mass=False, u_order=1):
        self.response, self.potential = response, potential
        self.lamR_mass, self.u_order = lamR_mass, u_order

    def label(self):
        return (f"resp{self.response}/phi^{self.potential}/"
                f"lamR{'(m2)' if self.lamR_mass else 'const'}/U{self.u_order}")

    def with_(self, **kw):
        d = dict(response=self.response, potential=self.potential,
                 lamR_mass=self.lamR_mass, u_order=self.u_order)
        d.update(kw)
        return Model(**d)


def build_design(prep, model, counts, min_count=30):
    masses, ctr, mref = prep["masses"], prep["ctr"], prep["mref"]
    P = len(masses)
    n_resp = model.response + 1
    n_lam = 2 if model.lamR_mass else 1
    pot = [6, 8] if model.potential == 8 else [6]
    ncol = n_resp + n_lam + len(pot) + P
    rows, rhs, wts = [], [], []
    for j, m2 in enumerate(masses):
        h, ntot = counts[m2]
        ok = h > min_count
        if ok.sum() < 8:
            return None
        y = -np.log(h[ok] / ntot)
        w = np.sqrt(h[ok])
        c = ctr[ok]
        block = np.zeros((int(ok.sum()), ncol))
        for k in range(n_resp):
            block[:, k] = V / 2 * (m2 ** k) * c ** 2
        block[:, n_resp] = V / 4 * c ** 4
        if model.lamR_mass:
            block[:, n_resp + 1] = V / 4 * (m2 - mref) * c ** 4
        for i, p in enumerate(pot):
            block[:, n_resp + n_lam + i] = c ** p
        block[:, n_resp + n_lam + len(pot) + j] = 1.0
        rows.append(block); rhs.append(y); wts.append(w)
    return (np.vstack(rows), np.concatenate(rhs), np.concatenate(wts),
            dict(n_resp=n_resp, n_lam=n_lam, pot=pot, ncol=ncol))


def sum_counts(prep, pick=None):
    out = {}
    for m2 in prep["masses"]:
        idx = pick[m2] if pick is not None else slice(None)
        out[m2] = (prep["H"][m2][idx].sum(axis=0), prep["M"][m2][idx][:, 0].sum())
    return out


def crit_mass(a, hint):
    n = len(a)
    if n == 2:
        return float(-a[0] / a[1]) if a[1] else float("nan")
    if n == 3:
        c, b, q = a
        if abs(q) < 1e-14:
            return float(-c / b) if b else float("nan")
        d = b * b - 4 * q * c
        if d < 0:
            return float("nan")
        r = [(-b + np.sqrt(d)) / (2 * q), (-b - np.sqrt(d)) / (2 * q)]
        return float(min(r, key=lambda z: abs(z - hint)))
    return float("nan")


def _bin_probs(prep, model, beta, ix):
    masses, ctr, mref = prep["masses"], prep["ctr"], prep["mref"]
    nr, nl, pot = ix["n_resp"], ix["n_lam"], ix["pot"]
    probs = {}
    for m2 in masses:
        a = sum(beta[k] * m2 ** k for k in range(nr))
        lam = beta[nr] + (beta[nr + 1] * (m2 - mref) if model.lamR_mass else 0.0)
        e = V / 2 * a * ctr ** 2 + V / 4 * lam * ctr ** 4
        for i, p in enumerate(pot):
            e = e + beta[nr + nl + i] * ctr ** p
        w = np.exp(-(e - e.min()))
        probs[m2] = w / w.sum()
    return probs


def fit_model(prep, model, counts=None, min_count=30, want_probs=False):
    counts = counts if counts is not None else sum_counts(prep)
    d = build_design(prep, model, counts, min_count)
    if d is None:
        return None
    A, Y, W, ix = d
    Aw = A * W[:, None]
    beta, *_ = np.linalg.lstsq(Aw, Y * W, rcond=None)
    resid = (Y - A @ beta) * W
    dof = max(1, len(Y) - ix["ncol"])
    nr, nl = ix["n_resp"], ix["n_lam"]
    out = dict(chi2=float(resid @ resid), dof=int(dof), npts=len(Y),
               cond=float(np.linalg.cond(Aw)),
               lam_R=float(beta[nr]),
               g6=float(beta[nr + nl]),
               m2_crit=crit_mass(beta[:nr], float(np.mean(prep["masses"]))))
    if want_probs:
        out["probs"] = _bin_probs(prep, model, beta, ix)
    return out


def binder_counts(prep, m2, idx=None):
    M = prep["M"][m2][idx if idx is not None else slice(None)]
    n, s2, s4 = M[:, 0].sum(), M[:, 1].sum(), M[:, 2].sum()
    return 1 - (s4 / n) / (3 * (s2 / n) ** 2)


def _F(small, big):
    dnu = small["dof"] - big["dof"]
    if dnu <= 0 or big["chi2"] <= 0:
        return float("nan")
    return ((small["chi2"] - big["chi2"]) / dnu) / (big["chi2"] / big["dof"])


def calibrate(prep, model_small, model_big, rng, n_null, n_eff_frac=1.0):
    """Calibrated p-values for a nested comparison and for the small model's fit quality:
    the null distributions of chi^2 and of the F statistic are built by parametric
    bootstrap, drawing multinomial counts from the fitted small model."""
    f_small = fit_model(prep, model_small, want_probs=True)
    if f_small is None:
        return None
    obs_chi2 = f_small["chi2"]
    f_big = fit_model(prep, model_big) if model_big is not None else None
    obs_F = _F(f_small, f_big) if f_big is not None else float("nan")
    ntot = {m2: prep["M"][m2][:, 0].sum() for m2 in prep["masses"]}
    null_chi2, null_F = [], []
    for _ in range(n_null):
        cnt = {}
        for m2 in prep["masses"]:
            n_draw = max(50, int(ntot[m2] * n_eff_frac))
            h = rng.multinomial(n_draw, f_small["probs"][m2]).astype(float)
            cnt[m2] = (h * (ntot[m2] / n_draw), ntot[m2])
        fs = fit_model(prep, model_small, cnt)
        if fs is None:
            continue
        null_chi2.append(fs["chi2"])
        if model_big is not None:
            fb = fit_model(prep, model_big, cnt)
            if fb is not None:
                null_F.append(_F(fs, fb))
    nc = np.asarray(null_chi2, dtype=float)
    nf = np.asarray([x for x in null_F if np.isfinite(x)], dtype=float)
    return dict(chi2=obs_chi2, dof=f_small["dof"],
                p_gof=float(np.mean(nc >= obs_chi2)) if nc.size else float("nan"),
                F=obs_F,
                p_nested=(float(np.mean(nf >= obs_F)) if nf.size and np.isfinite(obs_F) else float("nan")),
                n_null=int(nc.size), cond=f_small["cond"])


def select_model(prep, rng, n_null, n_eff_frac=1.0):
    m, diag = Model(), {}
    steps = [("response_1_vs_2", dict(response=2)),
             ("potential_6_vs_8", dict(potential=8)),
             ("lamR_const_vs_running", dict(lamR_mass=True))]
    for name, kw in steps:
        if len(prep["masses"]) < 3 and "potential" not in kw:
            diag[name] = {"skipped": "needs >= 3 masses"}
            continue
        c = calibrate(prep, m, m.with_(**kw), rng, n_null, n_eff_frac)
        if c is None:
            diag[name] = {"skipped": "fit failed"}
            continue
        diag[name] = c
        if np.isfinite(c["p_nested"]) and c["p_nested"] < P_SEL:
            m = m.with_(**kw)
    masses = np.array(prep["masses"])
    Um = np.array([binder_counts(prep, mm) for mm in masses])
    if len(masses) >= 4:
        r1 = Um - np.polyval(np.polyfit(masses, Um, 1), masses)
        r2 = Um - np.polyval(np.polyfit(masses, Um, 2), masses)
        s1, s2 = float(r1 @ r1), float(r2 @ r2)
        F = ((s1 - s2) / 1) / (s2 / max(1, len(masses) - 3)) if s2 > 0 else np.inf
        if np.isfinite(F) and F > U_ORDER_F:
            m = m.with_(u_order=2)
    final = calibrate(prep, m, None, rng, n_null, n_eff_frac)
    diag["selected_gof"] = final
    ok, reason = True, []
    if final is None:
        ok = False; reason.append("selected model failed to fit")
    else:
        if np.isfinite(final["p_gof"]) and final["p_gof"] < P_GOF:
            ok = False; reason.append(f"calibrated goodness-of-fit p = {final['p_gof']:.3g} < {P_GOF}")
        if final["cond"] > COND_MAX:
            ok = False; reason.append(f"condition number {final['cond']:.2e} > {COND_MAX:.0e}")
    diag["usable"], diag["unusable_reason"] = ok, reason
    return m, diag


def crossing(by_mass, rng, n_boot=N_BOOT, n_null=N_NULL, qmax=QMAX, nbins=NBINS):
    """m^2_hist and its chain-bootstrap error at one point, with the selected model."""
    prep = prepare(by_mass, qmax, nbins)
    model, diag = select_model(prep, rng, n_null)
    f = fit_model(prep, model)
    if f is None or not np.isfinite(f["m2_crit"]):
        return None
    reps = []
    for _ in range(n_boot):
        pick = {m: rng.integers(0, prep["M"][m].shape[0], prep["M"][m].shape[0]) for m in prep["masses"]}
        fb = fit_model(prep, model, sum_counts(prep, pick))
        if fb is not None and np.isfinite(fb["m2_crit"]):
            reps.append(fb["m2_crit"])
    reps = np.asarray(reps)
    return dict(m2_crossing=f["m2_crit"], m2_crossing_err=float(reps.std(ddof=1)) if reps.size > 2 else float("nan"),
                usable=bool(diag["usable"]), unusable_reason="; ".join(diag["unusable_reason"]),
                model=model.label(), chi2=f["chi2"], dof=f["dof"], n_boot=int(reps.size),
                n_masses=len(prep["masses"]), n_chains=int(sum(prep["M"][m].shape[0] for m in prep["masses"])),
                masses=[float(m) for m in prep["masses"]])


# ------------------------------------------------------------------- driver

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chains", required=True, help="folder with unconstrained/ (the Zenodo chains)")
    ap.add_argument("--points", nargs="*", help="subset, as n_max:lambda (default: every row of data/setB_points.csv)")
    ap.add_argument("--out", default=os.path.join(HERE, "recomputed"), help="output folder (default: recomputed/)")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--null", type=int, default=N_NULL)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)

    pub = list(csv.DictReader(open(os.path.join(HERE, "data", "setB_points.csv"))))
    pts = [(int(r["n_max"]), float(r["lam"])) for r in pub]
    if a.points:
        want = {(int(p.split(":")[0]), float(p.split(":")[1])) for p in a.points}
        pts = [p for p in pts if p in want]
    os.makedirs(a.out, exist_ok=True)
    rows, details = [], {}
    for n, lam in pts:
        rng = np.random.default_rng([a.seed, n, int(round(1e6 * lam))])   # one stream per point
        by_mass = load_point(a.chains, n, lam, window=WINDOWS.get((n, lam)))
        if len(by_mass) < 3:
            print(f"  n_max={n} lambda={lam:g}: {len(by_mass)} dense masses, no histogram estimate", flush=True)
            rows.append(dict(n_max=n, lam=f"{lam:g}", m2_crossing="", m2_crossing_err="", usable="", model="", n_masses=len(by_mass), n_chains=""))
            continue
        r = crossing(by_mass, rng, a.boot, a.null)
        if r is None:
            print(f"  n_max={n} lambda={lam:g}: fit failed", flush=True)
            rows.append(dict(n_max=n, lam=f"{lam:g}", m2_crossing="", m2_crossing_err="", usable="", model="", n_masses=len(by_mass), n_chains=""))
            continue
        details[f"{n}:{lam:g}"] = r
        print(f"  n_max={n:2d} lambda={lam:<5g}: m2_hist = {r['m2_crossing']:.5f} +- {r['m2_crossing_err']:.5f}  "
              f"{r['model']:26s} {'usable' if r['usable'] else 'UNUSABLE: ' + r['unusable_reason']}  "
              f"({r['n_masses']} masses, {r['n_chains']} chains)", flush=True)
        rows.append(dict(n_max=n, lam=f"{lam:g}", m2_crossing=f"{r['m2_crossing']:.6g}", m2_crossing_err=f"{r['m2_crossing_err']:.3g}",
                         usable=int(r["usable"]), model=r["model"], n_masses=r["n_masses"], n_chains=r["n_chains"]))
    path = os.path.join(a.out, "histogram_crossing.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["n_max", "lam", "m2_crossing", "m2_crossing_err", "usable", "model", "n_masses", "n_chains"])
        w.writeheader(); w.writerows(rows)
    json.dump(details, open(os.path.join(a.out, "histogram_crossing_details.json"), "w"), indent=1)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
