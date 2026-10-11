"""Reproduce Figs. 1-3 of "Lattice Monte Carlo of λφ⁴ on Euclidean de Sitter"
(A. Caravano, 2026) from the released tables.

    python make_figures.py      # writes fig1_main, fig2_running, fig3_triviality (.pdf/.png) into figures/ next to this file

Requires numpy and matplotlib only.  Inputs (all in ./data):
    setA_points.csv               Set A: <phibar^2>, Binder cumulant per (n_max, lambda)      -> Fig. 1(b), Fig. 2, Fig. 3(b)
    fig1_zero_mode_samples.npz    Set A: zero-mode chains at n_max = 10, lambda = 0.01, 0.1, 0.2 -> Fig. 1(a)
    setA_effective_coupling.csv   Set A: effective quartic coupling from the mean force          -> Fig. 3(a), open symbols
    setB_points.csv               Set B: effective coupling, Binder cumulant at the massless point -> Fig. 3, filled symbols
Conventions: H = 1, Omega = 8 pi^2/3, S = 1/2 sum (lam_n + m^2) c^2 + (lam/4) int phi^4, phibar = c_0/sqrt(Omega).
"""
import csv
import logging
from math import gamma, pi, sqrt
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
logging.getLogger("fontTools").setLevel(logging.ERROR)   # silence the font-timestamp notices when writing PDFs
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

HERE = Path(__file__).resolve().parent
TAB = HERE / "data"

# ---------------------------------------------------------------- physics
VOL = 8.0 * pi ** 2 / 3.0
SY_C = sqrt(4.0 / VOL) * gamma(0.75) / gamma(0.25)          # 0.13176...
# pure-quartic (massless SY) moments m_2k = Gamma((2k+1)/4)/Gamma(1/4), a = 1
MOM = {k: gamma((2 * k + 1) / 4.0) / gamma(0.25) for k in (1, 2, 3, 4)}
U_SY = 1.0 - MOM[2] / (3.0 * MOM[1] ** 2)                    # 0.27052...


def deg(n):
    return (2 * n + 3) * (n + 1) * (n + 2) / 6.0


def lam_n(n):
    return n * (n + 3.0)


def S2(n_max):
    """One-loop coupling-screening sum, sum_{n>=1} deg(n)/lam_n^2 ~ ln(n_max)/3."""
    n = np.arange(1, n_max + 1, dtype=float)
    return float(np.sum(deg(n) / lam_n(n) ** 2))


def slope_1loop(n_max):
    """<phibar^2>/<phibar^2>_SY - 1 = (9/2) S2 lam4 / V + O(lam4^{3/2})."""
    return 4.5 * S2(n_max) / VOL


# ---------------------------------------------------------------- style
COL = {4: "#0072B2", 6: "#D55E00", 8: "#009E73", 10: "#CC79A7", 12: "#56B4E9"}
MRK = {4: "o", 6: "s", 8: "^", 10: "v", 12: "D"}
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e6e6e3"
COLW = 3.375                                      # APS single column [in]

RC = {
    "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
    "font.size": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 7.2,
    "xtick.labelsize": 7.8, "ytick.labelsize": 7.8, "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.45, "ytick.minor.width": 0.45,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "lines.linewidth": 1.1, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "axes.unicode_minus": False,
}
EB = dict(lw=0, elinewidth=0.9, capsize=1.8, ms=4.2, mew=1.0, mfc="white")


def grid(ax):
    ax.grid(True, color=GRID, lw=0.45, zorder=0)
    ax.set_axisbelow(True)


def panel(ax, lab):
    ax.text(-0.17, 1.0, lab, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9)


# ---------------------------------------------------------------- data
def read_csv(name):
    with open(TAB / name) as fh:
        return list(csv.DictReader(fh))


def load_setA():
    """data[(n_max, lambda)] = dict(phi2, err, sy, U, Uerr, c0) from setA_points.csv; c0 (zero-mode
    samples, Fig. 1a) only for the three n_max = 10 couplings shipped in fig1_zero_mode_samples.npz."""
    z = np.load(TAB / "fig1_zero_mode_samples.npz")
    data = {}
    for r in read_csv("setA_points.csv"):
        if r["role"] != "Set A":
            continue
        n, lam = int(r["n_max"]), float(r["lam"])
        key = f"lam{lam:g}"
        data[(n, lam)] = dict(phi2=float(r["phibar2"]), err=float(r["phibar2_err"]), sy=float(r["phibar2_SY"]),
                              U=float(r["U"]), Uerr=float(r["U_err"]),
                              c0=(z[key] if (n == int(z["n_max"]) and key in z.files) else None))
    return data


def fit_slope(data, nm, window):
    L = np.array(sorted(l for (n, l) in data if n == nm and l <= window))
    dev = np.array([data[(nm, l)]["phi2"] / data[(nm, l)]["sy"] - 1 for l in L])
    e = np.array([data[(nm, l)]["err"] / data[(nm, l)]["sy"] for l in L])
    c = float(np.sum(L * dev / e ** 2) / np.sum(L ** 2 / e ** 2))
    return c, float(1 / np.sqrt(np.sum(L ** 2 / e ** 2)))


# ---------------------------------------------------------------- figure 1
LAM_COL = {0.01: "#A992D4", 0.02: "#9880CA", 0.05: "#876DC0",
           0.1: "#6E47AD", 0.2: "#3A1670", 0.5: "#240E47"}   # sequential: lambda
LAM_MRK = {0.01: "o", 0.02: "^", 0.05: "v", 0.1: "s", 0.2: "P", 0.5: "D"}
XMAX = 1.25                                    # range of x = lam^{1/4} phibar / H


def sy_pdf(x, r=1.0):
    """Massless SY law in the scaling variable x = lam^{1/4} phibar; r rescales
    the quartic (r = lam_C/lam gives the one-loop constraint potential)."""
    a = r * VOL / 4.0
    Z = 2.0 * gamma(1.25) * a ** -0.25
    return np.exp(-a * x ** 4) / Z


def _clear_top_ylabel(ax, frac=0.07):
    """Blank the topmost y-tick label when it sits in the top `frac` of the
    axes, where it would collide with the panel letter."""
    ax.figure.canvas.draw()
    lo, hi = ax.get_ylim()
    ticks = list(ax.get_yticks())
    labs = [t.get_text() for t in ax.get_yticklabels()]
    keep = ["" if (lo <= t <= hi and t > hi - frac * (hi - lo)) else l
            for t, l in zip(ticks, labs)]
    ax.set_yticks(ticks)
    ax.set_yticklabels(keep)
    ax.set_ylim(lo, hi)


def fig1(data, cutoffs, window, out, hist_nmax=None, hist_lams=(0.01, 0.1, 0.2),
         lmax=0.5, zoom="tail", fname="fig1_main", inset_lam=None):
    """Main figure: (a,b) zero-mode histograms collapsing onto SY, and the
    constraint potential; (c,d) the variance and its residual vs lambda.

    `lmax` caps the couplings shown.  The default 0.5 shows the weak-coupling SY
    recovery together with the first column where the one-loop description
    visibly fails; that departure is the higher-order correction to the mass,
    which this set does not carry because its bare mass is the ANALYTIC
    one-loop counterterm.  Its size is measurable, since Set A and
    Set B share lam = 0.2 and 0.5 and differ in nothing but the bare mass:
    <phibar^2> differs by 2.6% at (n_max=4, lam=0.2), and by -1.8%, +1.4%,
    -0.3% at lam=0.5 for n_max = 4, 6, 8.  Beyond lam ~ 0.2 that contamination
    overtakes the signal and is not of a fixed sign, so the strong-coupling
    story belongs to Set B (fig3), where m_R^2 = 0 is imposed.
    """
    hist_nmax = hist_nmax or max(n for n in cutoffs if data[(n, hist_lams[0])]["c0"] is not None)
    lams = np.array(sorted({l for (_, l) in data if l <= lmax}))
    dodge = {nm: 1 + 0.055 * (i - (len(cutoffs) - 1) / 2) for i, nm in enumerate(cutoffs)}

    fig = plt.figure(figsize=(2 * COLW + 0.3, 3.35))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.15, 1], width_ratios=[1, 1.08],
                          hspace=0.0, wspace=0.30)
    a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[1, 0], sharex=a)
    c = fig.add_subplot(gs[0, 1]); d = fig.add_subplot(gs[1, 1], sharex=c)

    # ---- (a) histograms in the scaling variable: every lambda on one curve
    xs = np.linspace(-XMAX, XMAX, 400)
    edges = np.linspace(-XMAX, XMAX, 43)
    xc, dx = 0.5 * (edges[1:] + edges[:-1]), edges[1] - edges[0]
    a.fill_between(xs, sy_pdf(xs), color="#e9e7e2", lw=0, zorder=1)
    hSY, = a.plot(xs, sy_pdf(xs), color=INK, lw=1.2, zorder=8,
                  label=r"SY, $e^{-\Omega\lambda\bar\phi^4/4}$")
    sig = sqrt(SY_C)
    hG, = a.plot(xs, np.exp(-xs ** 2 / (2 * sig ** 2)) / sqrt(2 * pi * sig ** 2),
                 color=MUTED, lw=0.9, ls=(0, (1.2, 1.4)), zorder=2,
                 label="Gaussian, same variance")
    lbig = inset_lam if inset_lam is not None else max(hist_lams)   # coupling shown in the insets
    rC = 1 - 9 * lbig * S2(hist_nmax) / VOL          # one-loop quartic / tree quartic at lbig
    h1L = Line2D([], [], color=LAM_COL[lbig], lw=0.9, ls=(0, (4, 2)),
                 label=rf"one loop, $\lambda={lbig:g}$ (insets)")
    hl = []
    pot = {}
    for i, l in enumerate(hist_lams):
        x = l ** 0.25 * data[(hist_nmax, l)]["c0"] / sqrt(VOL)
        # every bin is shown; the three couplings use grids offset by dx/3 so
        # their markers interleave instead of piling up on the same x
        e_i = edges + (i - (len(hist_lams) - 1) / 2) * dx / len(hist_lams)
        xc_i = 0.5 * (e_i[1:] + e_i[:-1])
        h, _ = np.histogram(x, bins=e_i)
        n = len(x)
        pdf, epdf = h / (n * dx), np.sqrt(np.maximum(h, 1)) / (n * dx)
        hl.append(a.errorbar(xc_i, pdf, yerr=epdf, marker=LAM_MRK[l],
                             ms=2.0, mfc="white", mew=0.6, color=LAM_COL[l], lw=0,
                             elinewidth=0.6, capsize=0, zorder=4 + i,
                             label=rf"$\lambda={l:g}$"))
        pot[l] = (xc_i, h, n)
    a.set_ylim(0, 1.52)
    a.set_ylabel(r"$P$")
    _h = [hSY, hG] + ([h1L] if h1L is not None else [])
    leg1 = a.legend(_h, [x.get_label() for x in _h],
                    frameon=False, loc="upper left", handletextpad=0.4,
                    borderaxespad=0.2, labelspacing=0.25, fontsize=7.2, handlelength=1.8)
    a.add_artist(leg1)
    a.legend(hl, [h.get_label() for h in hl], frameon=False, loc="upper right",
             title=rf"lattice, $n_{{\max}}={hist_nmax}$", title_fontsize=7.2,
             handletextpad=0.2, borderaxespad=0.2, labelspacing=0.25, fontsize=7.2)
    plt.setp(a.get_xticklabels(), visible=False)
    grid(a)
    a.text(-0.16, 1.0, "(a)", transform=a.transAxes, fontsize=9, va="bottom")

    # ---- (b) the constraint effective potential, Omega V_C = -ln P
    b.plot(xs, VOL * xs ** 4 / 4, color=INK, lw=1.2, zorder=8)
    for i, l in enumerate(hist_lams):
        xcen, h, n = pot[l]
        ok = h >= 30
        # -ln P, zero set by the SY normalisation (not by a noisy bin)
        y = -np.log(h[ok] / (n * dx) / sy_pdf(0.0))
        ey = 1 / np.sqrt(h[ok])
        b.errorbar(xcen[ok], y, yerr=ey, marker=LAM_MRK[l], ms=1.9,
                   mfc="white", mew=0.6, color=LAM_COL[l], lw=0, elinewidth=0.6,
                   capsize=0, zorder=4 + i)
    b.set_ylim(-0.6, 7.2)
    b.set_xlim(-XMAX, XMAX)
    b.set_xlabel(r"$\lambda^{1/4}\,\bar\phi/H$")
    b.set_ylabel(r"$\Omega\,V_{\rm C}$", labelpad=6)
    b.set_yticks([0, 3, 6])
    grid(b)

    # ---- zoom insets at x ~ 0.7: one-loop screening at the largest coupling
    # |x| folded (the distribution is even) in finer bins, weakest and largest coupling only
    if zoom == "tail":
        ZLO, ZHI, ZNB, fold = 0.70, 0.80, 5, 2
    else:                       # around the origin
        ZLO, ZHI, ZNB, fold = -0.25, 0.25, 10, 1
    xz = np.linspace(ZLO, ZHI, 200)
    def folded(l):
        v = l ** 0.25 * data[(hist_nmax, l)]["c0"] / sqrt(VOL)
        if fold == 2:
            v = np.abs(v)
        e = np.linspace(ZLO, ZHI, ZNB + 1); h, _ = np.histogram(v, bins=e)
        w = e[1] - e[0]; n = len(v)
        return 0.5 * (e[1:] + e[:-1]), h / (fold * n * w), np.sqrt(h) / (fold * n * w), h
    zl = (min(hist_lams), lbig)
    ia = a.inset_axes([0.37, 0.07, 0.26, 0.24] if zoom == "tail" else [0.33, 0.10, 0.34, 0.30])
    ia.plot(xz, sy_pdf(xz), color=INK, lw=1.0, zorder=3)
    ia.plot(xz, sy_pdf(xz, rC), color=LAM_COL[lbig], lw=0.9, ls=(0, (4, 2)), zorder=3)
    ib = b.inset_axes([0.37, 0.42, 0.26, 0.24 * 2.15] if zoom == "tail" else [0.33, 0.30, 0.34, 0.30 * 2.15])
    ib.plot(xz, VOL * xz ** 4 / 4, color=INK, lw=1.0, zorder=3)
    ib.plot(xz, -np.log(sy_pdf(xz, rC) / sy_pdf(0.0)), color=LAM_COL[lbig], lw=0.9, ls=(0, (4, 2)), zorder=3)
    for l in zl:
        xc, p, ep, h = folded(l)
        kw = dict(marker=LAM_MRK[l], ms=2.3, mfc="white", mew=0.6, color=LAM_COL[l], lw=0,
                  elinewidth=0.7, capsize=0, zorder=5)
        ia.errorbar(xc, p, ep, **kw)
        ib.errorbar(xc, -np.log(p / sy_pdf(0.0)), 1 / np.sqrt(h), **kw)
    for ax_ in (ia, ib):
        ax_.set_xlim(ZLO, ZHI); ax_.set_xticks([0.70, 0.75, 0.80] if zoom == "tail" else [-0.2, 0.0, 0.2])
        ax_.tick_params(labelsize=6.2, length=2, pad=1.5)
        ax_.grid(False)
    if zoom == "tail":
        yv = np.concatenate([sy_pdf(xz), sy_pdf(xz, rC)]); ia.set_ylim(0.9 * yv.min(), 1.06 * yv.max())
        ib.set_ylim(0.92 * VOL * ZLO ** 4 / 4, 1.04 * VOL * ZHI ** 4 / 4)
        ia.set_yticks([0.1, 0.15]); ib.set_yticks([2.0, 2.5])
    else:
        ia.set_ylim(0.835, 0.905); ia.set_yticks([0.85, 0.89])
        ib.set_ylim(-0.03, 0.06); ib.set_yticks([0.0, 0.04])
    ia.tick_params(labelbottom=False)     # same x range as the lower inset, which carries the labels
    from matplotlib.patches import ConnectionPatch, Rectangle
    for ax_, ins in ((a, ia), (b, ib)):
        y0, y1 = ins.get_ylim()
        ax_.add_patch(Rectangle((ZLO, y0), ZHI - ZLO, y1 - y0, fill=False, ec="#7a7a7a", lw=0.6, zorder=9))
        # same geometry in both panels: inset top-right -> box top-left, inset bottom-left -> box bottom-left
        pairs = (((1, 1), (ZLO, y1)), ((0, 0), (ZLO, y0))) if zoom == "tail" else \
                ((((0, 1), (ZLO, y0)), ((1, 1), (ZHI, y0))) if ax_ is b else (((0, 1), (ZLO, y0)), ((1, 1), (ZHI, y0))))
        for xa, xy in pairs:
            ax_.add_artist(ConnectionPatch(xyA=xa, coordsA=ins.transAxes, xyB=xy, coordsB=ax_.transData,
                                           color="#7a7a7a", lw=0.6, zorder=9))

    # ---- (c) variance vs lambda
    xhi = 1.30 * lams.max()
    LAM_1L = 0.15      # beyond the lam <= 0.1 window where a linear (one-loop) residual is the whole story
    for ax_ in (c, d):
        ax_.axvspan(LAM_1L, xhi, color="#efefec", zorder=0, lw=0)
    lg = np.geomspace(0.007, xhi, 200)
    for nm in cutoffs:
        L = np.array([l for l in lams if (nm, l) in data])
        y = np.array([data[(nm, l)]["phi2"] * sqrt(l) for l in L])
        e = np.array([data[(nm, l)]["err"] * sqrt(l) for l in L])
        c.plot(lg, SY_C * (1 + slope_1loop(nm) * lg), color=COL[nm], lw=0.8,
               ls=(0, (4, 2)), zorder=2)
        c.errorbar(L * dodge[nm], y, yerr=e, marker=MRK[nm], color=COL[nm],
                   zorder=4, label=rf"$n_{{\max}}={nm}$", **{**EB, "ms": 3.8})
    hSYc = c.axhline(SY_C, color=INK, lw=1.2, zorder=3, label="SY")

    c.set_xscale("log")
    c.set_xlim(0.007, xhi)
    _hi = max(data[(nm, l)]["phi2"] * sqrt(l) + data[(nm, l)]["err"] * sqrt(l)
              for nm in cutoffs for l in lams if (nm, l) in data)
    c.set_ylim(SY_C - 0.0011, _hi + 0.0009)
    c.set_ylabel(r"$\sqrt{\lambda}\,\langle\bar\phi^{2}\rangle/H^{2}$")
    _clear_top_ylabel(c)
    c.text(np.sqrt(LAM_1L * xhi), SY_C + 0.00015, "beyond\none loop", fontsize=6.0, color=MUTED,
           ha="center", va="bottom", linespacing=1.1)
    h, lb = c.get_legend_handles_labels()
    k = lb.index("SY")
    h, lb = [h[k]] + h[:k] + h[k + 1:], [lb[k]] + lb[:k] + lb[k + 1:]
    h.append(Line2D([], [], color=MUTED, lw=0.8, ls=(0, (4, 2))))
    lb.append("one loop, no free parameter")
    c.legend(h, lb, frameon=False, loc="upper left", handletextpad=0.4,
             borderaxespad=0.2, labelspacing=0.25, fontsize=7.2, ncol=2,
             columnspacing=0.9)
    plt.setp(c.get_xticklabels(), visible=False)
    grid(c)
    c.text(-0.17, 1.0, "(b)", transform=c.transAxes, fontsize=9, va="bottom")

    # ---- (d) Binder cumulant: the shape, below the width in (c)
    for nm in cutoffs:
        L = np.array([l for l in lams if (nm, l) in data])
        d.errorbar(L * dodge[nm], [data[(nm, l)]["U"] for l in L],
                   yerr=[data[(nm, l)]["Uerr"] for l in L], marker=MRK[nm],
                   color=COL[nm], zorder=4, **{**EB, "ms": 3.8})
    d.axhline(U_SY, color=INK, lw=1.2, zorder=3)
    d.text(0.035, 0.90, rf"SY: $U={U_SY:.4f}$   (Gaussian: $U=0$)",
           transform=d.transAxes, fontsize=7, color=INK, va="top")
    d.set_ylim(0.2605, 0.2865)
    d.set_yticks([0.265, 0.270, 0.275, 0.280])
    d.set_xticks(list(lams))
    d.set_xticklabels([f"{l:g}" for l in lams])
    d.minorticks_off()
    c.minorticks_off()
    d.set_xlabel(r"bare coupling $\lambda$")
    d.set_ylabel(r"$U=1-\frac{\langle\bar\phi^4\rangle}{3\langle\bar\phi^2\rangle^2}$",
                 labelpad=2)
    grid(d)
    fig.align_ylabels([a, b])
    fig.align_ylabels([c, d])
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{fname}.{ext}", **({"dpi": 220} if ext == "png" else {}))
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig2(data, cutoffs, window, out):
    """Result 2: the residual slope against the cutoff.  On a logarithmic
    cutoff axis the one-loop prediction (9/2) S_2/Omega is close to a straight
    line, so what the eye reads off is the logarithmic running of lambda.
    Three fit windows: lambda <= 0.05 (filled), 0.1 (light) and 0.2 (open).
    The excess over one loop grows with the window and with the cutoff, as a
    residual mass of two-loop order (~ lambda^2 S_1) does."""
    fig, ax = plt.subplots(figsize=(COLW, 1.8))
    ns = np.arange(3, 14)
    h1L, = ax.plot(ns, [2 * slope_1loop(n) for n in ns], color=INK, lw=1.2, zorder=3,
                   label=r"one loop, $u=9S_2(n_{\max})/\Omega$")
    windows = ((0.05, -0.04, dict(mfc=None)), (0.1, 0.0, dict(mfc=None, alpha=0.45)), (0.2, 0.04, dict(mfc="white")))
    for nm in cutoffs:
        for w, dx, st in windows:
            c, e = fit_slope(data, nm, w)
            ax.errorbar([nm * (1 + dx)], [2 * c], yerr=[2 * e], marker=MRK[nm], color=COL[nm], zorder=5,
                        **{**EB, "ms": 4.6, **{k: (COL[nm] if v is None else v) for k, v in st.items()}})
    ax.legend(handles=[Line2D([], [], lw=0, marker="o", ms=4.6, color=MUTED, mfc=MUTED, label=r"$\lambda\leq0.05$"),
                       Line2D([], [], lw=0, marker="o", ms=4.6, color=MUTED, mfc=MUTED, alpha=0.45, label=r"$\lambda\leq0.1$"),
                       Line2D([], [], lw=0, marker="o", ms=4.6, color=MUTED, mfc="white", mew=1.0, label=r"$\lambda\leq0.2$")],
              frameon=False, loc="upper left", fontsize=7.5, ncol=3,
              handletextpad=0.35, columnspacing=0.9, borderaxespad=0.25)
    ax.add_artist(ax.get_legend())
    ax.legend(handles=[h1L], frameon=False, loc="lower right", fontsize=7.5,
              handlelength=1.9, handletextpad=0.45, borderaxespad=0.3)
    ax.set_xscale("log"); ax.set_xlim(3.45, 11.8)
    ax.set_xticks(list(cutoffs)); ax.set_xticklabels([str(x) for x in cutoffs])
    ax.minorticks_off()
    ax.set_xlabel(r"cutoff $n_{\max}\simeq\Lambda/H$")
    ax.set_ylabel(r"screening coeff. $u$")
    ax.set_ylim(0.10, 0.46)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4])
    grid(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "fig2_running.pdf")
    fig.savefig(out / "fig2_running.png", dpi=200)
    plt.close(fig)
    return {nm: fit_slope(data, nm, window) for nm in cutoffs}


# ---------------------------------------------------------------- figure 3
def fig3(dataA, out):
    """Effective quartic coupling of the zero mode over four decades of bare coupling (a) and the Binder
    cumulant (b); open symbols: Set A (one-loop mass counterterm), filled: Set B (massless point tuned
    nonperturbatively).  Total error on y in Set B = statistical (+) fit-window systematic in quadrature.
    Axes: x = u lambda, y = u lambda_R^(eff), with u = 9 S_2 / Omega."""
    mk = {4: "s", 6: "o", 8: "^", 10: "D"}
    col = {4: "#1f5fbf", 6: "#0f8a5f", 8: "#d9731a", 10: "#b02a2a"}
    pts, Ub = [], []
    for r in read_csv("setA_effective_coupling.csv"):
        n, lam = int(r["n_max"]), float(r["lam"])
        pts.append((n, float(r["x"]), float(r["y"]), float(r["y_err"]), True))
        d = dataA[(n, lam)]
        Ub.append((n, float(r["x"]), d["U"], d["Uerr"], True))
    for r in read_csv("setB_points.csv"):
        if r.get("used_in_paper", "1") == "0":   # the n_max = 4 points beyond lambda = 60 are tabulated but not used
            continue
        n = int(r["n_max"])
        e = sqrt(float(r["y_err_stat"]) ** 2 + float(r["y_err_window"]) ** 2)
        pts.append((n, float(r["x"]), float(r["y"]), e, False))
        Ub.append((n, float(r["x"]), float(r["U"]), float(r["U_err"]), False))
    matplotlib.rcdefaults()          # this figure does not use the style of Figs. 1 and 2
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm", "axes.linewidth": 0.8})
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.4, 4.25), sharex=True,
                               gridspec_kw=dict(height_ratios=[1.75, 0.62], hspace=0.06))
    xx = np.logspace(-3.2, 1.7, 400)
    a.plot(xx, xx, color="0.55", ls="-.", lw=0.8); a.plot(xx, xx * (1 - xx), "r--", lw=0.8); a.plot(xx, xx / (1 + xx), "k-", lw=1)
    a.axhline(1, color="k", ls=":", lw=0.6)

    def draw(ax, P, ms, lw, cs):
        for n, x, y, e, openm in P:
            ax.errorbar(x, y, e, fmt=mk[n], ms=ms, color=col[n], lw=lw, capsize=cs, mfc=("white" if openm else col[n]), mew=0.9, alpha=0.8, zorder=3)
    draw(a, pts, 3.3, 0.8, 1.5)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(1.5e-3, 40); a.set_ylim(1.5e-3, 25)
    a.set_ylabel(r"$y=u\,\lambda_R^{\rm (eff)}$")
    h1 = [Line2D([], [], color="0.55", ls="-.", lw=0.8, label=r"tree level $\lambda_R^{\rm (eff)}=\lambda$"),
          Line2D([], [], color="r", ls="--", lw=0.8, label=r"one loop $y=x(1-x)$"),
          Line2D([], [], color="k", ls="-", lw=1, label=r"bubble chain $y=x/(1+x)$")]
    h2 = [Line2D([], [], marker=mk[n], color=col[n], ls="", ms=3.3, label=r"$n_{\max}=%d$" % n) for n in (4, 6, 8, 10)]
    l1 = a.legend(handles=h1, frameon=True, fancybox=False, edgecolor="0.45", framealpha=1, fontsize=6.2, loc="upper left",
                  bbox_to_anchor=(0.005, 0.995), handletextpad=0.5, handlelength=1.8, labelspacing=0.3, borderpad=0.45)
    l1.get_frame().set_linewidth(0.2); a.add_artist(l1)
    l2 = a.legend(handles=h2, frameon=True, fancybox=False, edgecolor="0.45", framealpha=1, fontsize=6.2, loc="upper left",
                  bbox_to_anchor=(0.555, 0.995), ncol=1, handletextpad=0.3, labelspacing=0.25, borderpad=0.4)
    l2.get_frame().set_linewidth(0.2)
    ins = inset_axes(a, width="100%", height="100%", loc="lower left", bbox_to_anchor=(0.52, 0.07, 0.45, 0.34),
                     bbox_transform=a.transAxes, borderpad=0)
    ins.plot(xx, xx / (1 + xx), "k-", lw=0.8); ins.axhline(1, color="k", ls=":", lw=0.6)
    draw(ins, [p for p in pts if p[1] >= 0.9], 2.6, 0.6, 1.0)
    ins.set_xscale("log"); ins.set_xlim(0.9, 20); ins.set_ylim(0.35, 1.2); ins.tick_params(labelsize=5.5, pad=1.5, length=2)
    ins.set_yticks([0.4, 0.6, 0.8, 1.0]); ins.set_xticks([1, 10]); ins.set_xticklabels(["1", "10"]); ins.minorticks_off()
    mark_inset(a, ins, loc1=2, loc2=4, fc="none", ec="0.45", lw=0.45, ls="-")
    draw(b, [(n, x, U / 0.27052, e / 0.27052, o) for n, x, U, e, o in Ub], 3.3, 0.8, 1.5)
    b.axhline(1, color="k", lw=1)
    b.set_ylim(0.96, 1.19); b.set_yticks([1.0, 1.1]); b.set_ylabel(r"$U/U_{\rm SY}$"); b.set_xlabel(r"$x=u\lambda$")
    h3 = [Line2D([], [], marker="s", color="0.3", mfc="white", ls="", ms=3.3, label=r"one-loop mass"),
          Line2D([], [], marker="s", color="0.3", mfc="0.3", ls="", ms=3.3, label=r"$m_R^2=0$ tuned nonperturbatively")]
    l3 = b.legend(handles=h3, frameon=True, fancybox=False, edgecolor="0.45", framealpha=1, fontsize=6.2, loc="upper left",
                  bbox_to_anchor=(0.005, 0.995), handletextpad=0.4, labelspacing=0.3, borderpad=0.45)
    l3.get_frame().set_linewidth(0.2)
    for sp in ins.spines.values():
        sp.set_linewidth(0.55)
    for ax, t in ((a, "(a)"), (b, "(b)")):
        ax.text(-0.20, 1.0, t, transform=ax.transAxes, fontsize=9, va="top", ha="left")
    fig.savefig(out / "fig3_triviality.pdf", bbox_inches="tight")
    fig.savefig(out / "fig3_triviality.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    out = HERE / "figures"; out.mkdir(exist_ok=True)
    dataA = load_setA()
    cutoffs = sorted({n for (n, _) in dataA})
    plt.rcParams.update(RC)
    fig1(dataA, cutoffs, 0.2, out)
    fits = fig2(dataA, cutoffs, 0.2, out)
    fig3(dataA, out)
    print(f"SY constant sqrt(lambda)<phibar^2> = {SY_C:.5f};  U_SY = {U_SY:.5f}")
    print("screening coefficient u from <phibar^2>/<phibar^2>_SY - 1 = u lambda/2 (lambda <= 0.1), "
          "against the one-loop value 9 S_2/Omega:")
    for nm in cutoffs:
        c, ce = fit_slope(dataA, nm, 0.1)
        print(f"  n_max={nm:>2}: u = {2 * c:.4f} +- {2 * ce:.4f}   one loop {2 * slope_1loop(nm):.4f}")
    # common fit of the residual slope to kappa times the one-loop value, lambda <= 0.1 (Result 2)
    num = den = 0.0
    for (nm, lam), d in dataA.items():
        if lam <= 0.1 + 1e-12:
            t, e = slope_1loop(nm) * lam, d["err"] / d["sy"]
            num += t * (d["phi2"] / d["sy"] - 1) / e ** 2
            den += t * t / e ** 2
    print(f"common fit u = kappa * 9 S_2/Omega on lambda <= 0.1: kappa = {num / den:.3f} +- {den ** -0.5:.3f}")
    # plateau of the effective coupling at strong coupling (Result 3): constant fit of y at x > 4
    P = [r for r in read_csv("setB_points.csv") if r["used_in_paper"] == "1" and float(r["x"]) > 4]
    w = np.array([1 / (float(r["y_err_stat"]) ** 2 + float(r["y_err_window"]) ** 2) for r in P])
    y = np.array([float(r["y"]) for r in P])
    yinf = float((w * y).sum() / w.sum())
    print(f"plateau y_inf (x > 4, {len(P)} points) = {yinf:.3f} +- {w.sum() ** -0.5:.3f}, "
          f"chi2/dof = {float((w * (y - yinf) ** 2).sum()):.1f}/{len(P) - 1}")
    print("wrote", ", ".join(sorted(p.name for p in out.iterdir())))


if __name__ == "__main__":
    main()
