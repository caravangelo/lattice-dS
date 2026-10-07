"""SO(5) invariance of the regulated interaction against the degree D of the quadrature grid.

A band-limited field is fixed in mode space, int phi^4 is evaluated on the
product Gauss grid of degree D and on eight random SO(5) rotations of it, and

    delta = max_R |I4(R) - I4| / |I4|

is recorded together with the orthonormality error of the grid,
eps = max_AB |sum_a w_a Y_A(x_a) Y_B(x_a) - delta_AB|. At D = 4 n_max, delta
drops to double-precision roundoff: the regulated theory is exactly SO(5)
invariant. At D = 2 n_max the symmetry is broken at the percent level although
the harmonics are still orthonormal on the grid (Appendix A of the paper).

    python reproduction/fig_so5.py               # plot from reproduction/data/so5_scan.json
    python reproduction/fig_so5.py --scan        # recompute the scan into recomputed/ first (about 5 minutes)
"""

import argparse
import json
from pathlib import Path

import numpy as np

from lattice_ds import grid as G
from lattice_ds import harmonics as H

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/so5_scan.json"          # the published scan
RECOMPUTED = ROOT / "recomputed/so5_scan.json"  # written by --scan; the published file is never modified
OUT = ROOT / "figures/fig_so5"


def random_so5(rng):
    a = rng.standard_normal((5, 5))
    q, r = np.linalg.qr(a)
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def scan(n_max, n_rot=8, m2=0.5, seed=0, chunk=40000):
    def fields_at(points, coeffs):
        out = [np.empty(len(points)) for _ in coeffs]
        for s in range(0, len(points), chunk):
            Y = H.evaluate_at(points[s:s + chunk], n_max)
            for o, c in zip(out, coeffs):
                o[s:s + chunk] = c @ Y
        return out

    def ortho(g):
        nm = H.n_modes(n_max)
        M = np.zeros((nm, nm))
        for s in range(0, g.n_points, chunk):
            Y = H.evaluate_at(g.points[s:s + chunk], n_max)
            M += Y @ (g.weights[s:s + chunk, None] * Y.T)
        return float(np.abs(M - np.eye(nm)).max())

    rng = np.random.default_rng(seed)
    shells = np.array([lab[0] for lab in H.label_list(n_max)])
    raw = rng.standard_normal(H.n_modes(n_max))
    coeffs = [raw, raw / np.sqrt(H.eigenvalue(shells) + m2)]          # white, free thermal
    rots = [random_so5(rng) for _ in range(n_rot)]
    rows = []
    for D in range(2 * n_max, 4 * n_max + 5, 2):
        g = G.for_degree(D)
        i0 = [float(np.sum(g.weights * f ** 4)) for f in fields_at(g.points, coeffs)]
        dev = [0.0, 0.0]
        for R in rots:
            for k, f in enumerate(fields_at(g.points @ R.T, coeffs)):
                dev[k] = max(dev[k], abs(float(np.sum(g.weights * f ** 4)) - i0[k]) / abs(i0[k]))
        rows.append(dict(n_max=n_max, degree=D, points=g.n_points, ortho=ortho(g), white=dev[0], thermal=dev[1]))
        print(rows[-1], flush=True)
    return rows


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "mathtext.fontset": "cm", "axes.linewidth": 0.6,
                         "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True})
    COL = {4: "#1f5fbf", 6: "#0f8a5f", 8: "#d9731a", 10: "#b02a2a"}
    MK = {4: "s", 6: "o", 8: "^", 10: "D"}
    NS = sorted({r["n_max"] for r in rows})
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.axvline(4, color="0.5", lw=0.6, ls=":")
    for n in NS:
        rr = [r for r in rows if r["n_max"] == n]
        x = np.array([r["degree"] / n for r in rr])
        ax.plot(x, [r["thermal"] for r in rr], "-", marker=MK[n], color=COL[n], ms=3.8, lw=0.9)
        ax.plot(x, [r["white"] for r in rr], "--", marker=MK[n], color=COL[n], mfc="white", ms=3.8, lw=0.7)
        ax.plot(x, [r["ortho"] for r in rr], ":", marker=MK[n], color="0.55", mec="0.55", ms=2.4, lw=0.6)
    ax.set_yscale("log")
    ax.set_ylim(1e-17, 10)
    ax.set_xlim(1.9, 5.1)
    ax.set_xlabel(r"$D/n_{\max}$")
    ax.set_ylabel("deviation")
    h = [Line2D([], [], marker=MK[n], color=COL[n], ls="", ms=3.6, label=r"$n_{\max}=%d$" % n) for n in NS]
    h += [Line2D([], [], color="k", ls="-", lw=0.9, label=r"$\delta$, thermal"),
          Line2D([], [], color="k", ls="--", lw=0.7, label=r"$\delta$, white"),
          Line2D([], [], color="0.55", ls=":", marker="o", mec="0.55", ms=2.4, lw=0.6, label=r"$\epsilon$")]
    ax.legend(handles=h, frameon=False, fontsize=6.5, loc="lower left", bbox_to_anchor=(0.0, 0.172), handletextpad=0.5,
              labelspacing=0.25)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=250, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", action="store_true",
                    help="recompute the scan into reproduction/recomputed/ and plot it (otherwise plot the published one)")
    ap.add_argument("--n-max", nargs="+", type=int, default=[4, 6, 8])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    data = DATA
    if a.scan:
        RECOMPUTED.parent.mkdir(exist_ok=True)
        with open(RECOMPUTED, "w") as fh:
            json.dump(sum([scan(n) for n in a.n_max], []), fh, indent=1)
        data = RECOMPUTED
    with open(data) as fh:
        plot(json.load(fh), a.out)
