"""Static figure for the Monte Carlo appendix, from the same chain as the page.

    python reproduction/fig_mc_sampling.py [chain_nmax10.npz]

Writes reproduction/figures/fig_mc_sampling.{pdf,png} (the figure of Appendix B of the paper).

Left: three configurations of the chain, each shown on 2-spheres of S^4 at
latitudes chi = 30..150 deg (psi = pi/2), drawn to scale (radius sin chi).
Right: the zero mode along the chain with its histogram against the
Starobinsky-Yokoyama density, and the running estimates of <phibar^2> and U.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma, gammaln, lpmv

HERE = Path(__file__).resolve().parent
ANIM = HERE.parent / "animations"          # the chain and bands.py live with the animation
sys.path.insert(0, str(ANIM))
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e3e3e0"
BLUE, WARN = "#0072B2", "#B23A48"
RC = {
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 8,
    "axes.labelsize": 8.5, "axes.titlesize": 8.5, "legend.fontsize": 6.8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.7,
    "figure.dpi": 200, "savefig.dpi": 400, "savefig.bbox": "tight",
}
ORDER = [1, 2, 0, 3, 4]           # chi = 30, 60, 90, 120, 150


def sphere_image(b, n_max, npix=260, tilt=0.42, yaw=0.6):
    """Orthographic image of sum_lm b_lm Y_lm; NaN outside the disc."""
    u = (np.arange(npix) + 0.5) / npix * 2 - 1
    X, Yv = np.meshgrid(u, -u)
    r2 = X**2 + Yv**2
    Z = np.sqrt(np.clip(1 - r2, 0, None))
    by = Yv * np.cos(tilt) + Z * np.sin(tilt)
    bz = -Yv * np.sin(tilt) + Z * np.cos(tilt)
    theta = np.arccos(np.clip(by, -1, 1))
    phi = np.arctan2(X, bz) + yaw
    out = np.zeros_like(X)
    ct = np.cos(theta)
    for l in range(n_max + 1):
        for m in range(-l, l + 1):
            am = abs(m)
            k = np.sqrt((2 * l + 1) / (4 * np.pi)
                        * np.exp(gammaln(l - am + 1) - gammaln(l + am + 1)))
            p = k * lpmv(am, l, ct)
            if m > 0:
                p = np.sqrt(2) * p * np.cos(am * phi)
            elif m < 0:
                p = np.sqrt(2) * p * np.sin(am * phi)
            out += b[l * l + l + m] * p
    out[r2 >= 1] = np.nan
    return out


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "chain_nmax10.npz"
    d = np.load(ANIM / src)
    n_max, vol, lam4 = int(d["n_max"]), float(d["vol"]), float(d["lam4"])
    pb = d["c0_post"] / np.sqrt(vol)
    K = pb.size
    chi = d["chi_deg"]

    # three configurations: low, central and high zero mode (10th/50th/90th pct)
    picks = [int(np.argmin(np.abs(pb - np.percentile(pb, q)))) for q in (8, 50, 92)]
    rms = np.sqrt(np.mean(np.sum(d["slices"][:, 0].astype(float) ** 2, axis=1)) / (4 * np.pi))
    vmax = round(2.4 * float(rms), 1)

    plt.rcParams.update(RC)
    fig = plt.figure(figsize=(7.1, 3.5))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.02], wspace=0.30)
    gl = outer[0].subgridspec(3, 5, wspace=0.02, hspace=0.05,
                              width_ratios=[1, 1, 1, 1, 1])
    for row, k in enumerate(picks):
        for col, s in enumerate(ORDER):
            ax = fig.add_subplot(gl[row, col])
            img = sphere_image(d["slices"][k, s].astype(float), n_max)
            rad = np.sin(np.deg2rad(chi[s]))
            ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      extent=(-rad, rad, -rad, rad), interpolation="bilinear",
                      rasterized=True)
            ax.add_patch(plt.Circle((0, 0), rad, fill=False, lw=0.4, color="#9a9a96"))
            ax.set_xlim(-1.04, 1.04); ax.set_ylim(-1.04, 1.04)
            ax.set_aspect("equal"); ax.axis("off")
            if row == 0:
                ax.set_title(rf"$\chi={chi[s]:.0f}^\circ$", fontsize=7, pad=2, color=INK)
            if col == 0:
                ax.text(-1.25, 0, rf"$\bar\phi={pb[k]:+.2f}$", rotation=90,
                        va="center", ha="center", fontsize=7, color=INK)
    cax = fig.add_axes([0.135, 0.055, 0.30, 0.018])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-vmax, vmax), cmap="RdBu_r"),
                      cax=cax, orientation="horizontal")
    cb.set_label(r"$\phi$  $[H]$", fontsize=7.5, labelpad=1)
    cb.ax.tick_params(labelsize=6.5, width=0.6, length=2)
    cb.outline.set_linewidth(0.6)

    gr = outer[1].subgridspec(2, 1, height_ratios=[1.0, 0.92], hspace=0.62)
    gt = gr[0].subgridspec(1, 2, width_ratios=[4.2, 1], wspace=0.04)
    ax = fig.add_subplot(gt[0]); axh = fig.add_subplot(gt[1], sharey=ax)
    lim = 3.2
    ax.plot(np.arange(1, K + 1), pb, color=BLUE, lw=0.45)
    for k in picks:
        ax.plot([k + 1], [pb[k]], marker="o", ms=3.2, mfc="none", mec=INK, mew=0.8)
    ax.set_xlim(0, K); ax.set_ylim(-lim, lim)
    ax.set_xlabel("trajectory"); ax.set_ylabel(r"$\bar\phi$  $[H]$")
    bins = np.linspace(-lim, lim, 37)
    axh.hist(pb, bins=bins, density=True, orientation="horizontal",
             color=BLUE, alpha=0.25, lw=0, label="chain")
    xg = np.linspace(-lim, lim, 400)
    w = np.exp(-vol * lam4 * xg**4 / 4)
    axh.plot(w / np.trapezoid(w, xg), xg, color=INK, lw=1.0, ls=(0, (4, 2)), label="SY")
    axh.tick_params(labelleft=False); axh.set_xticks([])
    axh.legend(frameon=False, loc="lower right", handlelength=1.6, borderpad=0.0,
               ncol=2, columnspacing=1.0, bbox_to_anchor=(1.02, 1.0),
               labels=["histogram of the chain", "SY density"],
               handles=axh.get_legend_handles_labels()[0])
    for a_ in (ax, axh):
        a_.grid(True, color=GRID, lw=0.5); a_.set_axisbelow(True)

    gb = gr[1].subgridspec(1, 2, wspace=0.85)
    N = np.arange(1, K + 1)
    v = np.cumsum(pb**2) / N
    U = 1 - (np.cumsum(pb**4) / N) / (3 * v**2)
    kurt = gamma(0.25) * gamma(1.25) / gamma(0.75) ** 2
    from lattice_ds.perturbation import one_loop_zero_mode_moments
    pv = one_loop_zero_mode_moments(n_max, float(d["m2"]), lam4, vol)["c0_2"] / vol
    pu = 1 - kurt / 3
    import bands
    bN, qv, qu, _, _ = bands.bands(K, pv)
    for i, (y, ref, q, lab, yl) in enumerate((
            (v, pv, qv, r"$\langle\bar\phi^2\rangle$  $[H^2]$", (0.3, 2.5)),
            (U, pu, qu, r"$U$", (-0.3, 0.7)))):
        a_ = fig.add_subplot(gb[i])
        a_.fill_between(bN, q[0], q[3], color=BLUE, alpha=0.10, lw=0)
        a_.fill_between(bN, q[1], q[2], color=BLUE, alpha=0.18, lw=0,
                        label="68% and 95% of chains")
        a_.axhline(ref, color=INK, lw=1.0, ls=(0, (4, 2)), label="prediction")
        a_.plot(N[3:], y[3:], color=BLUE, lw=1.1, label="running estimate")
        a_.set_xscale("log"); a_.set_xlim(4, K); a_.set_ylim(*yl)
        a_.set_xlabel(r"configurations $N_{\rm meas}$"); a_.set_ylabel(lab)
        a_.grid(True, color=GRID, lw=0.5); a_.set_axisbelow(True)
        if i == 1:
            a_.legend(frameon=False, loc="lower right", handlelength=1.6, ncol=3,
                      borderpad=0.0, columnspacing=1.0, bbox_to_anchor=(1.32, 1.0))

    out = HERE / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_mc_sampling.{ext}")
    print("wrote reproduction/figures/fig_mc_sampling.{pdf,png}; picks", picks, "vmax", vmax)


if __name__ == "__main__":
    main()
