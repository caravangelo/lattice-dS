"""Massless point and effective coupling from the mean force (Appendix D of the paper).

At one (n_max, lambda) the inputs are

- constrained chains (zero mode held at phibar), all run at one bare mass
  m_s^2, which store per configuration the force f = (1/Omega) dS/dphibar and
  the pair (A, B) of `constrained.zero_mode_coefficients`;
- one unconstrained ensemble at a bare mass near the massless point, with the
  series c_0 and phi2 = (1/Omega) int phi^2.

The analysis then proceeds in four steps.

1. Reweighting. Every chain is moved from m_s^2 to m_s^2 + delta with the exact
   weights exp(-delta I_2 / 2), I_2 = int psi^2 = (B - m_s^2) Omega / (3 lambda).
   The observables shift as f -> f + delta * phibar and B -> B + delta.
2. Renormalised mass. V_C'(phibar)/phibar = m_R^2 + lambda_R phibar^2
   (+ a_6 phibar^4) is fitted to the forces at phibar > 0 together with the
   curvature V_C''(0) = <B> - <A^2> of the phibar = 0 chains, which enters as
   the point at phibar = 0.
3. Massless point. delta_* solves m_R^2(delta) = 0 by bisection;
   m_*^2 = m_s^2 + delta_*.
4. Observables at m_*^2. sigma^2 = <phibar^2> and the Binder cumulant U come
   from the unconstrained ensemble reweighted to m_*^2. The force at sigma is
   interpolated with a local power law between the two constrained values of
   phibar that bracket sigma, and y = u V_C'(sigma) / sigma^3.

Errors are a joint block bootstrap: every chain, constrained and
unconstrained, is resampled in blocks of `block` measurements (500 by
default) and steps 1-4 are repeated for each replica, so the error of y
includes the uncertainty of m_*^2.

Numpy only.
"""

from __future__ import annotations

import math

import numpy as np

from .grid import VOL_S4
from .perturbation import screening_coefficient
from .stats import block_bootstrap_indices

BLOCK = 500   # default block length of the bootstrap, in measurements


# ------------------------------------------------------------------ loading

def constrained_chain(phibar, m2, lam4, n_max, A, B, force, vol=VOL_S4):
    """Bundle the series of one constrained chain for the functions below."""
    ab = np.stack([np.asarray(A, float), np.asarray(B, float)], axis=1)
    return dict(phibar=float(phibar), mass=float(m2), lam=float(lam4), n=int(n_max),
                ab=ab, f=np.asarray(force, float),
                i2=(ab[:, 1] - float(m2)) * vol / (3 * float(lam4)), vol=float(vol))


def load_constrained(paths):
    """Constrained chains from .npz files written by `lattice-ds run-constrained`.

    The same keys are used by the published chains (A, B, force, phibar, m2,
    lam4, n_max, vol).
    """
    chains = []
    for p in paths:
        z = np.load(p)
        chains.append(constrained_chain(z["phibar"], z["m2"], z["lam4"], z["n_max"],
                                        z["A"], z["B"], z["force"], float(z["vol"])))
    return chains


def unconstrained_ensemble(c0, phi2, m2, vol=VOL_S4):
    """Bundle an unconstrained ensemble: (phibar^2 series, phi2 series, vol, bare mass)."""
    return (np.asarray(c0, float) ** 2 / vol, np.asarray(phi2, float), float(vol), float(m2))


def load_unconstrained(paths):
    """One ensemble from one or more unconstrained .npz files at the same bare mass.

    Several files are concatenated in the order given; their bare masses must
    agree to 5 decimals, and that rounded value is used as the mass of the
    ensemble. A single file keeps its mass unrounded.
    """
    if isinstance(paths, (str, bytes)) or hasattr(paths, "__fspath__"):
        paths = [paths]
    c0, phi2, m2, vol = [], [], None, None
    for p in paths:
        z = np.load(p)
        n = int(z["n_meas"]) if "n_meas" in z.files else len(z["c0"])
        c0.append(z["c0"][:n])
        phi2.append(z["phi2"][:n])
        vol = float(z["vol"])
        m = float(z["m2"]) if len(paths) == 1 else round(float(z["m2"]), 5)
        if m2 is not None and m != m2:
            raise ValueError(f"unconstrained chains at different bare masses: {m2}, {m}")
        m2 = m
    return unconstrained_ensemble(np.concatenate(c0), np.concatenate(phi2), m2, vol)


# -------------------------------------------------------------- reweighting

def reweighted_force(c, delta, idx=None):
    """<f + delta * phibar> at bare mass m_s^2 + delta, and the effective sample fraction."""
    i2, f, pb = c["i2"], c["f"], c["phibar"]
    if idx is not None:
        i2, f = i2[idx], f[idx]
    lw = -0.5 * delta * i2
    w = np.exp(lw - lw.max())
    w /= w.sum()
    return float((w * (f + delta * pb)).sum()), float(1 / (w * w).sum() / len(w))


def reweighted_curvature(c, delta, idx=None):
    """V_C''(0) = <B + delta> - <A^2> at bare mass m_s^2 + delta (a phibar = 0 chain)."""
    i2, ab = c["i2"], c["ab"]
    if idx is not None:
        i2, ab = i2[idx], ab[idx]
    lw = -0.5 * delta * i2
    w = np.exp(lw - lw.max())
    w /= w.sum()
    return float((w * (ab[:, 1] + delta)).sum() - (w * ab[:, 0] ** 2).sum()), float(1 / (w * w).sum() / len(w))


# ---------------------------------------------------------------------- fit

def fit_small_field(pts, order):
    """Weighted least squares of V_C'/phibar = m_R^2 + lambda_R phibar^2 (+ a_6 phibar^4).

    `pts` is a list of (phibar, value, error): for phibar > 0 the value is the
    mean force, for phibar = 0 it is the curvature itself. The phibar^4 term is
    included for order >= 3 when there are at least four points.

    Returns (m_R^2, lambda_R, chi2, dof).
    """
    X = np.array([p[0] for p in pts])
    Y = np.array([p[1] / p[0] if p[0] > 0 else p[1] for p in pts])
    E = np.array([p[2] / p[0] if p[0] > 0 else p[2] for p in pts])
    cols = [0, 2] + ([4] if order >= 3 and len(pts) >= 4 else [])
    A = np.stack([X ** k for k in cols], 1)
    W = 1 / E
    p, *_ = np.linalg.lstsq(A * W[:, None], Y * W, rcond=None)
    return p[0], p[1], float((((A @ p - Y) * W) ** 2).sum()), len(pts) - len(p)


# ------------------------------------------------- one (bootstrap) realisation

def _solve(ch, unc, order, idxs=None, delta_max=3.0, ch_all=None, idxs_all=None, idx_unc=None):
    """Steps 1-4 for one realisation of the data.

    `ch` are the chains inside the fit window, `ch_all` all chains at this point
    (used for the interpolation of the force at sigma). `idxs`, `idxs_all` are
    bootstrap index arrays, one per chain, and `idx_unc` the index array of the
    unconstrained ensemble; all None for the original data.
    Returns None if m_R^2(delta) has no zero within |delta| < delta_max.
    """
    ch_all = ch_all or ch
    idxs_all = idxs_all if idxs_all is not None else idxs

    def m_of(delta):
        pts = []
        for k, c in enumerate(ch):
            idx = None if idxs is None else idxs[k]
            if c["phibar"] == 0:
                v, _ = reweighted_curvature(c, delta, idx)
                pts.append((0.0, v, c["err0"]))
            else:
                v, _ = reweighted_force(c, delta, idx)
                pts.append((c["phibar"], v, c["errF"]))
        return fit_small_field(pts, order), pts

    # massless point: zero of m_R^2(delta), by bisection
    lo, hi = -delta_max, delta_max
    (mlo, *_), _ = m_of(lo)
    (mhi, *_), _ = m_of(hi)
    if mlo > 0 or mhi < 0:
        return None
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        (m, *_), _ = m_of(mid)
        if m < 0:
            lo = mid
        else:
            hi = mid
    delta = 0.5 * (lo + hi)
    (m, lamR, chi2, dof), pts = m_of(delta)

    # sigma and U at the massless point, from the unconstrained ensemble
    q2, p2, vol, msrc = unc
    if idx_unc is not None:
        q2, p2 = q2[idx_unc], p2[idx_unc]
    lw = -0.5 * (ch[0]["mass"] + delta - msrc) * vol * p2
    w = np.exp(lw - lw.max())
    w /= w.sum()
    s2 = float((w * q2).sum())
    s4 = float((w * q2 * q2).sum())
    sigma = math.sqrt(s2)
    U = 1 - s4 / (3 * s2 * s2)
    ess_u = float(1 / (w * w).sum() / len(w))

    # force at sigma: local power law between the two bracketing values of phibar
    fp = {}
    for k, c in enumerate(ch_all):
        if c["phibar"] == 0:
            continue
        idx = None if idxs is None else idxs_all[k]
        v, _ = reweighted_force(c, delta, idx)
        if v > 0:
            fp.setdefault(c["phibar"], []).append(v)
    xs = sorted(fp)
    vals = [np.mean(fp[x]) for x in xs]
    lo = max([x for x in xs if x <= sigma] or [xs[0]])
    hi = min([x for x in xs if x >= sigma] or [xs[-1]])
    if lo == hi:
        j = xs.index(lo)
        lo, hi = (xs[j - 1], xs[j]) if j > 0 else (xs[0], xs[1])
    flo, fhi = vals[xs.index(lo)], vals[xs.index(hi)]
    slope = (math.log(fhi) - math.log(flo)) / (math.log(hi) - math.log(lo))
    Fs = math.exp(math.log(flo) + slope * (math.log(sigma) - math.log(lo)))
    u = screening_coefficient(ch[0]["n"])
    return dict(delta=delta, lamR=lamR, chi2=chi2, dof=dof, sigma=sigma, U=U, ess_unc=ess_u,
                F_sigma=Fs, slope=slope, y_eff=u * Fs / sigma ** 3, y_taylor=u * lamR)


# ------------------------------------------------------------------- driver

def massless_point(constrained, unconstrained, *, order=3, xmax=1.0, n_boot=200, seed=3,
                   block=BLOCK, verbose=False):
    """Massless point, sigma, U and effective coupling at one (n_max, lambda).

    Parameters
    ----------
    constrained : list of chains (`load_constrained`), all at one bare mass,
        including at least one chain at phibar = 0
    unconstrained : ensemble tuple (`load_unconstrained`)
    order : 3 fits m_R^2 + lambda_R phibar^2 + a_6 phibar^4, 2 drops the last term
    xmax : fit window, only chains with phibar <= xmax enter the fit. All chains
        are used for the interpolation of the force at sigma.
    n_boot : bootstrap replicas for the errors
    seed : seed of the bootstrap
    block : block length of the bootstrap, in measurements; every chain and
        the unconstrained ensemble must hold at least two blocks

    Returns
    -------
    dict with, among others,

        mass_source, mass_retuned, delta, delta_err   m_s^2, m_*^2, their difference, its error
        sigma, U (+ _err)                             width and Binder cumulant at m_*^2
        y_eff, y_eff_err                              y = u V_C'(sigma)/sigma^3
        slope                                         local exponent d ln V_C' / d ln phibar at sigma
        lamR_smallfield, y_taylor (+ _err)            quartic Taylor coefficient and u times it
        fit_chi2, fit_dof                             of the fit at m_*^2
        ess_unconstrained                             effective sample fraction of the reweighted ensemble
    """
    ch_all = list(constrained)
    ch = [c for c in ch_all if c["phibar"] <= xmax + 1e-9]
    masses = sorted({c["mass"] for c in ch})
    if masses[-1] - masses[0] > 1e-3:
        raise ValueError(f"constrained chains at different bare masses: {masses}")
    for c in ch:
        c["mass"] = masses[0]
    n_unc = len(unconstrained[0])
    short = [f"constrained chain at phibar = {c['phibar']} ({len(c['f'])} measurements)"
             for c in ch_all if len(c["f"]) < 2 * block]
    if n_unc < 2 * block:
        short.append(f"unconstrained ensemble ({n_unc} measurements)")
    if short:
        raise ValueError("the block bootstrap needs at least two blocks of "
                         f"{block} measurements per series; too short: " + "; ".join(short))

    # per-chain errors at delta = 0 (block bootstrap), used as fit weights
    rng = np.random.default_rng(seed)
    for c in ch_all:
        reps = []
        for _ in range(100):
            idx = block_bootstrap_indices(rng, len(c["f"]), block)
            reps.append(reweighted_curvature(c, 0, idx)[0] if c["phibar"] == 0
                        else reweighted_force(c, 0, idx)[0])
        if c["phibar"] == 0:
            c["err0"] = float(np.std(reps, ddof=1))
        else:
            c["errF"] = float(np.std(reps, ddof=1))

    base = _solve(ch, unconstrained, order, ch_all=ch_all)
    if base is None:
        raise RuntimeError("m_R^2(delta) has no zero within +-3 in bare mass; "
                           "simulate closer to the massless point")

    boots = []
    for b in range(n_boot):
        idxs_all = [block_bootstrap_indices(rng, len(c["f"]), block) for c in ch_all]
        idxs = [idxs_all[k] for k, c in enumerate(ch_all) if c["phibar"] <= xmax + 1e-9]
        idx_unc = block_bootstrap_indices(rng, n_unc, block)
        r = _solve(ch, unconstrained, order, idxs, ch_all=ch_all, idxs_all=idxs_all, idx_unc=idx_unc)
        if r:
            boots.append([r["delta"], r["lamR"], r["sigma"], r["U"], r["y_eff"], r["slope"], r["y_taylor"]])
        if verbose and (b + 1) % 20 == 0:
            print(f"    bootstrap {b + 1}/{n_boot}", flush=True)
    b = np.array(boots)
    sd = b.std(axis=0, ddof=1) if len(b) > 2 else np.full(7, np.nan)

    m0 = ch[0]["mass"]
    n = ch[0]["n"]
    lam = ch[0]["lam"]
    x = screening_coefficient(n) * lam
    return dict(n_max=n, lam=lam, x=x, y_chain=x / (1 + x), mass_source=m0,
                mass_retuned=m0 + base["delta"], delta=base["delta"], delta_err=float(sd[0]),
                lamR_smallfield=base["lamR"], lamR_err=float(sd[1]),
                y_taylor=base["y_taylor"], y_taylor_err=float(sd[6]),
                sigma=base["sigma"], sigma_err=float(sd[2]), U=base["U"], U_err=float(sd[3]),
                ess_unconstrained=base["ess_unc"],
                y_eff=base["y_eff"], y_eff_err=float(sd[4]), slope=base["slope"], slope_err=float(sd[5]),
                fit_chi2=base["chi2"], fit_dof=base["dof"], order=order, n_chains=len(ch_all), n_fit=len(ch),
                phibars=sorted(c["phibar"] for c in ch), n_boot_ok=len(b), xmax=xmax, block=block)


# --------------------------------------- effective coupling at a known mass

def effective_coupling_fixed_mass(constrained, sigma, n_max):
    """y = u V_C'(sigma)/sigma^3 without mass tuning (Set A, Fig. 3 open symbols).

    `constrained` are chains at a few values of phibar around sigma, at the
    bare mass of the unconstrained simulation that gave sigma. ln V_C' is
    fitted linearly in ln phibar (a local power law; the slope is 3 for a pure
    quartic) and evaluated at sigma. Errors of the mean force use the
    integrated autocorrelation time and are propagated through the fit.
    """
    from .stats import tau_int_first_negative

    pts = []
    for c in constrained:
        f = c["f"]
        F = float(f.mean())
        E = float(f.std(ddof=1) * math.sqrt(2 * tau_int_first_negative(f) / len(f)))
        if F <= 0:
            continue
        pts.append((c["phibar"], F, E))
    if len(pts) < 2:
        return None
    pts.sort()
    X = np.array([p[0] for p in pts])
    F = np.array([p[1] for p in pts])
    E = np.array([p[2] for p in pts])
    lx, lf = np.log(X), np.log(F)
    w = (F / E) ** 2
    A = np.stack([np.ones_like(lx), lx], 1)
    p, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], lf * np.sqrt(w), rcond=None)
    cov = np.linalg.inv((A * np.sqrt(w)[:, None]).T @ (A * np.sqrt(w)[:, None]))
    ls = math.log(sigma)
    lF = p[0] + p[1] * ls
    var = cov[0, 0] + 2 * ls * cov[0, 1] + ls * ls * cov[1, 1]
    Fs = math.exp(lF)
    Fe = Fs * math.sqrt(max(var, 0))
    u = screening_coefficient(n_max)
    return dict(F_sigma=Fs, F_sigma_err=Fe, slope=float(p[1]), slope_err=float(math.sqrt(cov[1, 1])),
                y_eff=u * Fs / sigma ** 3, y_eff_err=u * Fe / sigma ** 3,
                phibars=X.tolist(), forces=F.tolist(), errs=E.tolist())
