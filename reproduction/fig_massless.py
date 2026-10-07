"""Appendix figure on the massless point (Appendix D3 of the paper).

(a) V_C'/phibar against phibar^2 at one Set B point, every constrained chain
    reweighted to m_*^2: the curvature at the origin, the mean force at
    phibar > 0, and the fit m_R^2 + lambda_R phibar^2 + a_6 phibar^4.
(b) Pull between the mean-force massless point and the independent estimate
    from histogram fits of unconstrained mass scans, at every Set B point
    where both exist (columns m2_star and m2_crossing of setB_points.csv).

Needs the published chains for panel (a):

    python reproduction/fig_massless.py --chains /path/to/chains [--point 8 20]
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from lattice_ds import meanforce
from lattice_ds.meanforce import BLOCK, reweighted_curvature, reweighted_force
from lattice_ds.stats import block_bootstrap_indices

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reproduce_tables import read_table, setb_config, tag_of  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "mathtext.fontset": "cm", "axes.linewidth": 0.6,
                     "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True})
COL = {4: "#1f5fbf", 6: "#0f8a5f", 8: "#d9731a", 10: "#b02a2a"}
MK = {4: "s", 6: "o", 8: "^", 10: "D"}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--chains", required=True, help="directory holding constrained/ and unconstrained/")
ap.add_argument("--point", nargs=2, type=float, default=[8, 20], metavar=("N_MAX", "LAMBDA"))
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "fig_massless"))
a = ap.parse_args()
n, lam = int(a.point[0]), a.point[1]

table = read_table("setB_points.csv")
rec = [r for r in table if int(r["n_max"]) == n and abs(float(r["lam"]) - lam) < 1e-9][0]
m_star, xm, sig = float(rec["m2_star"]), float(rec["fit_window"]), float(rec["sigma"])
cfg = setb_config(n, lam, xm)
files = sorted({p for pre in cfg["constrained"]
                for p in glob.glob(os.path.join(a.chains, "constrained", pre + "_[mf]*.npz"))}, key=tag_of)
if not files:
    raise SystemExit(f"no constrained chains for n_max={n}, lambda={lam:g} in {a.chains}/constrained "
                     f"(expected files starting with {', '.join(cfg['constrained'])})")
ch = meanforce.load_constrained(files)

# one point per chain: (phibar, V_C'/phibar or curvature, error), reweighted from its own mass to m_*^2
rng = np.random.default_rng(1)
pts = []
for c in ch:
    delta = m_star - c["mass"]
    reps = []
    for _ in range(100):
        idx = block_bootstrap_indices(rng, len(c["f"]), BLOCK)
        reps.append(reweighted_curvature(c, delta, idx)[0] if c["phibar"] == 0 else reweighted_force(c, delta, idx)[0])
    v = reweighted_curvature(c, delta)[0] if c["phibar"] == 0 else reweighted_force(c, delta)[0]
    e = float(np.std(reps, ddof=1))
    p = c["phibar"]
    pts.append((p, v / p if p > 0 else v, e / p if p > 0 else e))

# fit on the individual chains inside the window, as in the analysis
fp = [q for q in pts if q[0] <= xm + 1e-9]
X = np.array([q[0] for q in fp])
Y = np.array([q[1] for q in fp])
W = 1 / np.array([q[2] for q in fp])
A = np.stack([X ** 0, X ** 2, X ** 4], 1)
coef, *_ = np.linalg.lstsq(A * W[:, None], Y * W, rcond=None)
chi2 = float((((A @ coef - Y) * W) ** 2).sum())

# for display only: replicas at the same phibar combined by inverse-variance weighting
grp = {}
for p, v, e in pts:
    grp.setdefault(round(p, 3), []).append((v, e))
disp = []
for p, rows in sorted(grp.items()):
    w = np.array([1 / e ** 2 for _, e in rows])
    vv = np.array([v for v, _ in rows])
    disp.append((p, float((w * vv).sum() / w.sum()), float(w.sum() ** -0.5)))
disp = [q for q in disp if q[0] <= xm + 1e-9]          # show only the chains used in the fit

fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.8, 2.3), gridspec_kw=dict(wspace=0.34))
C = COL[n]


def draw(ax, pmax, ms, legend=False, zero=True):
    xx = np.linspace(0, min(pmax, xm) * 1.02, 200)
    hf, = ax.plot(xx ** 2, coef[0] + coef[1] * xx ** 2 + coef[2] * xx ** 4, "k-", lw=0.9, zorder=2)
    if zero:
        ax.axhline(0, color="0.6", lw=0.5, zorder=0)
    h = {}
    for p, v, e in disp:
        if p > pmax:
            continue
        cc = "#d62728" if p == 0 else C          # curvature in red
        h["c" if p == 0 else "f"] = ax.errorbar(p ** 2, v, e, fmt="s" if p == 0 else "o", ms=ms, color=cc, mfc=cc,
                                                 mew=0.8, lw=0.8, capsize=1.5, zorder=3)
    if legend:
        hs = ax.axvline(sig ** 2, color="0.5", lw=0.6, ls=":", zorder=1)
        ax.legend([h["c"], h["f"], hf, hs], [r"curvature $V_C''(0)$", "mean force", "fit", r"$\bar\phi=\sigma$"],
                  frameon=False, fontsize=6.3, loc="upper left", bbox_to_anchor=(0.0, 1.0), handletextpad=0.4, labelspacing=0.3)


pmax = max(p for p, _, _ in disp)
draw(a1, pmax, 3.0, legend=True)
a1.set_xlim(-0.01, pmax ** 2 * 1.06)
ins = inset_axes(a1, width="100%", height="100%", loc="lower left", bbox_to_anchor=(0.64, 0.22, 0.33, 0.30),
                 bbox_transform=a1.transAxes, borderpad=0)
draw(ins, 0.16, 2.6, zero=False)
ins.set_xlim(-0.002, 0.026)
ys = [v + s * e for p, v, e in disp if p <= 0.16 for s in (-1, 0, 1)]
pad = 0.15 * (max(ys) - min(ys))
ins.set_ylim(min(ys) - pad, max(ys) + pad)
ins.tick_params(labelsize=5.5, pad=1.5, length=2)
ins.set_xticks([0, 0.01, 0.02])
for sp in ins.spines.values():
    sp.set_linewidth(0.5)
a1.set_xlabel(r"$\bar\phi^2$")
a1.set_ylabel(r"$V_C'(\bar\phi)/\bar\phi$")
a1.text(-0.22, 1.0, "(a)", transform=a1.transAxes, fontsize=9, va="bottom")

# (b) pulls
pulls = []
for r in table:
    if r["m2_crossing"] == "" or r["used_in_paper"] != "1":
        continue
    z = (float(r["m2_star"]) - float(r["m2_crossing"])) / np.hypot(float(r["m2_star_err"]), float(r["m2_crossing_err"]))
    pulls.append(z)
    a2.plot(float(r["x"]), z, MK[int(r["n_max"])], color=COL[int(r["n_max"])], ms=3.4)
a2.axhspan(-2, 2, color="0.92", zorder=0)
a2.axhline(0, color="k", lw=0.6)
a2.set_xscale("log")
a2.set_xlim(0.03, 15)
a2.set_ylim(-4.4, 3.5)
a2.set_yticks([-3, -2, -1, 0, 1, 2, 3])
a2.set_xlabel(r"$x=u\lambda$")
a2.set_ylabel(r"pull $z$")
a2.text(-0.22, 1.0, "(b)", transform=a2.transAxes, fontsize=9, va="bottom")
a2.legend(handles=[Line2D([], [], ls="", label=r"$n_{\max}=$")]
          + [Line2D([], [], marker=MK[k], color=COL[k], ls="", ms=3.2, label=r"$%d$" % k) for k in (4, 6, 8, 10)],
          handlelength=0.8, frameon=False, fontsize=6.5, loc="lower center", ncol=5, handletextpad=0.2,
          columnspacing=0.7, borderaxespad=0.3)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"{a.out}.{ext}", dpi=250, bbox_inches="tight")
print(f"wrote {a.out}: {len(ch)} chains, fit chi2/dof {chi2:.1f}/{len(fp) - 3}, m_R^2 at m_*^2 = {coef[0]:+.4f}; "
      f"{len(pulls)} pulls, mean {np.mean(pulls):+.2f} +- {np.std(pulls, ddof=1) / np.sqrt(len(pulls)):.2f}, "
      f"{sum(abs(z) > 2 for z in pulls)} outside |z| <= 2")
