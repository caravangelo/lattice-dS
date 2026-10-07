"""Make the animated banner shown at the top of the README.

    python animations/make_banner.py [chain_nmax10.npz]

Writes banner-light.gif and banner-dark.gif (1600 x 500 pixels), one
for each GitHub colour theme. The banner is drawn directly from the chain, not
recorded from the web page: five 2-spheres of S^4 showing the field
configurations along the chain, and the histogram of the zero mode filling in
under the Starobinsky-Yokoyama distribution.

Timing. The first trajectories are shown slowly, with the field moved along the
free Hamiltonian flow that joins two successive configurations (the same
display-only interpolation as on the web page, see animations/README.md) and
the heat-bath redraw of the zero mode as a uniform shift. Then one
configuration per frame, then many per frame until the chain is complete.

Needs numpy, scipy, matplotlib and Pillow.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from PIL import Image
from scipy.spatial import cKDTree
from scipy.special import gammaln, lpmv

HERE = Path(__file__).resolve().parent
W, H, DPI = 1600, 500, 100
ORDER = [1, 2, 0, 3, 4]            # chain index of the spheres at chi = 30, 60, 90, 120, 150 degrees
R_MAX = 140                        # pixel radius of the equatorial sphere
VMAX = 4.0                         # colour range of the field, in units of H
COLOURS = 160                      # size of the GIF palette
TRANSPARENT = 255                  # palette index used for unchanged pixels
N_FULL = 60                        # the histogram is normalised to unit area from this many samples on
LEVELS = 20                        # colour steps (flat bands compress well in a GIF)
THEMES = {
    "light": dict(surface="#ffffff", ink="#1f2328", muted="#59636e", grid="#d1d9e0",
                  bars="#9bb8d9", cmap="RdBu_r"),
    "dark": dict(surface="#0d1117", ink="#f0f6fc", muted="#9198a1", grid="#3d444d",
                 bars="#3f6ea3", cmap="RdBu_r"),
}


def sphere_basis(n_max, radius, tilt=0.42, yaw=0.6):
    """Matrix (pixels in the disc, (n_max+1)^2) of real spherical harmonics, orthographic view."""
    n = 2 * radius
    u = (np.arange(n) + 0.5) / n * 2 - 1
    X, Y = np.meshgrid(u, -u)
    r2 = X ** 2 + Y ** 2
    inside = r2 < 1
    x, y = X[inside], Y[inside]
    z = np.sqrt(1 - r2[inside])
    by = y * np.cos(tilt) + z * np.sin(tilt)
    bz = -y * np.sin(tilt) + z * np.cos(tilt)
    ct = np.clip(by, -1, 1)
    phi = np.arctan2(x, bz) + yaw
    B = np.zeros((x.size, (n_max + 1) ** 2), dtype=np.float32)
    for l in range(n_max + 1):
        for m in range(-l, l + 1):
            am = abs(m)
            k = math.sqrt((2 * l + 1) / (4 * math.pi) * math.exp(gammaln(l - am + 1) - gammaln(l + am + 1)))
            p = k * lpmv(am, l, ct)
            if m > 0:
                p = math.sqrt(2) * p * np.cos(am * phi)
            elif m < 0:
                p = math.sqrt(2) * p * np.sin(am * phi)
            B[:, l * l + l + m] = p
    return B, inside


def schedule(K, T, acc):
    """Frames as (index a, index b, flow time s or None, zero-mode stage, samples shown, ms)."""
    frames = []
    slow, single, sub = 12, 60, 7
    for k in range(1, slow + 1):                     # config k-1 -> k, interpolated
        for j in range(1, sub + 1):
            frames.append((k - 1, k, T[k - 1] * j / sub if acc[k - 1] else None, j / sub, k - 1, 50))
        frames.append((k, k, None, 2.0, k, 90))      # after the zero-mode redraw
    for k in range(slow + 1, single + 1):
        frames.append((k, k, None, 2.0, k, 80))
    for k in np.unique(np.round(np.geomspace(single + 1, K, 56)).astype(int)):
        frames.append((int(k), int(k), None, 2.0, int(k), 60))
    a, b, s, z, nshow, _ = frames[-1]
    frames[-1] = (a, b, s, z, nshow, 2600)           # hold the completed chain
    return frames


def render(theme, d):
    th = THEMES[theme]
    n_max, vol, lam = int(d["n_max"]), float(d["vol"]), float(d["lam4"])
    K = d["slices"].shape[0]
    chi = d["chi_deg"]
    # configurations 0..K: slices with the zero mode removed, and the zero mode before/after its redraw
    sl = np.concatenate([d["start_slices"][None], d["slices"]]).astype(np.float64)
    pb_post = np.concatenate([[float(d["start_c0"])], d["c0_post"]]) / math.sqrt(vol)
    pb_pre = np.concatenate([[float(d["start_c0"])], d["c0_pre"]]) / math.sqrt(vol)
    sl[:, :, 0] -= (pb_post * math.sqrt(4 * math.pi))[:, None]
    T, acc = d["T"], d["acc"]

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Inter", "DejaVu Sans"],
                         "mathtext.fontset": "dejavusans", "font.size": 16, "text.color": th["ink"],
                         "axes.edgecolor": th["grid"], "axes.labelcolor": th["muted"],
                         "xtick.color": th["muted"], "ytick.color": th["muted"]})
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=th["surface"])
    cmap = matplotlib.colormaps[th["cmap"]].resampled(LEVELS)

    # ---- the five spheres, drawn to scale (radius sin chi)
    radii = [int(round(R_MAX * math.sin(math.radians(chi[i])))) for i in ORDER]
    gap, x0, cy = 16, 24, 248
    spheres = []
    x = x0
    for i, r in zip(ORDER, radii):
        ax = fig.add_axes([x / W, 1 - (cy + r) / H, 2 * r / W, 2 * r / H])
        ax.set_axis_off()
        B, inside = sphere_basis(n_max, r)
        img = np.full(inside.shape, np.nan)
        im = ax.imshow(img, cmap=cmap, vmin=-VMAX, vmax=VMAX, interpolation="nearest", extent=(-1, 1, -1, 1))
        ax.add_patch(Circle((0, 0), 1.0, fill=False, ec=th["surface"], lw=3.0))      # hides the pixel edge
        ax.add_patch(Circle((0, 0), 0.992, fill=False, ec=th["grid"], lw=1.0))
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.02, 1.02)
        fig.text((x + r) / W, 1 - (cy + R_MAX + 34) / H, rf"$\chi={chi[i]:.0f}°$", ha="center", va="center",
                 color=th["muted"], fontsize=16)
        spheres.append((i, B, inside, img, im))
        x += 2 * r + gap
    x_end = x - gap
    fig.text(x0 / W, 1 - 38 / H, r"Field $\phi(x)$ on 2-spheres of $S^4$", ha="left", va="center", fontsize=21,
             fontweight="semibold")
    status = fig.text(x_end / W, 1 - 38 / H, "", ha="right", va="center", fontsize=16, color=th["muted"])
    # colour scale
    cax = fig.add_axes([(x0 + 18) / W, 1 - 470 / H, 190 / W, 10 / H])
    cax.imshow(np.linspace(-1, 1, 256)[None], cmap=cmap, aspect="auto", extent=(-VMAX, VMAX, 0, 1))
    cax.set_yticks([])
    cax.set_xticks([-VMAX, 0, VMAX], [f"−{VMAX:.0f}", "0", f"+{VMAX:.0f}"], fontsize=14)
    cax.tick_params(length=0, pad=3)
    for sp in cax.spines.values():
        sp.set_visible(False)
    fig.text((x0 + 236) / W, 1 - 466 / H, r"$\phi/H$", ha="left", va="center", color=th["muted"], fontsize=15)
    fig.text(x_end / W, 1 - 470 / H,
             rf"$n_{{\max}}={n_max}$:  {int(d['n_modes']):,} modes,   $\lambda={lam:g}$".replace(",", " ", 1),
             ha="right", va="center", color=th["muted"], fontsize=15)

    # ---- zero-mode histogram against the Starobinsky-Yokoyama distribution
    hx0, hx1, hy0, hy1 = x_end + 60, W - 24, 138, 404
    ax = fig.add_axes([hx0 / W, 1 - hy1 / H, (hx1 - hx0) / W, (hy1 - hy0) / H], facecolor=th["surface"])
    edges = np.linspace(-3.3, 3.3, 34)
    xs = np.linspace(-3.3, 3.3, 400)
    sy = np.exp(-vol * lam * xs ** 4 / 4)
    sy /= np.trapezoid(sy, xs)
    bars = ax.bar(0.5 * (edges[1:] + edges[:-1]), np.zeros(len(edges) - 1), width=np.diff(edges) * 0.86,
                  color=th["bars"], lw=0)
    ax.plot(xs, sy, color=th["ink"], lw=2.4)
    marker, = ax.plot([], [], marker="v", ms=9, color=th["ink"], ls="", clip_on=False)
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(0, 0.46)
    ax.set_yticks([])
    ax.set_xticks([-2, 0, 2])
    ax.tick_params(length=0, pad=6, labelsize=15)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(r"zero mode $\bar\phi/H$", fontsize=16, labelpad=4)
    fig.text(hx0 / W, 1 - 38 / H, "Zero-mode distribution", ha="left", va="center", fontsize=21, fontweight="semibold")
    # legend line under the title: sampled histogram and the analytic curve
    for row, (label, kind) in enumerate([("Monte Carlo", "bar"), ("Starobinsky–Yokoyama", "line")]):
        ly = 1 - (76 + 26 * row) / H
        if kind == "bar":
            fig.patches.append(plt.Rectangle((hx0 / W, ly - 7 / H), 16 / W, 14 / H, transform=fig.transFigure,
                                             fc=th["bars"], ec="none"))
        else:
            fig.add_artist(plt.Line2D([(hx0 - 3) / W, (hx0 + 19) / W], [ly, ly], color=th["ink"], lw=2.4))
        fig.text((hx0 + 28) / W, ly, label, ha="left", va="center", fontsize=15, color=th["muted"])

    out = []
    for a, b, s, stage, nshow, ms in schedule(K, T, acc):
        if s is None:
            coef = sl[b]
        else:                                        # free Hamiltonian flow from configuration a to b
            Tk = T[a]
            coef = (sl[a] * math.sin(Tk - s) + sl[b] * math.sin(s)) / math.sin(Tk)
        if stage >= 2.0:
            pb = pb_post[b]
        elif acc[a]:                                 # the trajectory also moves the zero mode
            pb = pb_post[a] + (pb_pre[b] - pb_post[a]) * stage
        else:
            pb = pb_post[a]
        for i, B, inside, img, im in spheres:
            img[inside] = B @ coef[i] + pb
            im.set_data(img)
        h, _ = np.histogram(pb_post[1:nshow + 1], bins=edges)
        dens = h / max(nshow, N_FULL) / np.diff(edges)      # the first samples add fixed-size steps
        for rect, v in zip(bars, dens):
            rect.set_height(v)
        marker.set_data([pb], [0.475])
        status.set_text(f"trajectory {max(b, 1) if s is not None else b:,} of {K:,}".replace(",", " "))
        fig.canvas.draw()
        out.append((np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(), ms))
    plt.close(fig)
    return out


def save_gif(frames, path, surface, cmap):
    """One shared palette for all frames and no dithering, which keeps flat areas flat and the file small."""
    sample = np.concatenate([frames[i][0] for i in np.linspace(0, len(frames) - 1, 8).astype(int)], axis=0)
    # palette: the page colour and the colour steps of the field exactly, the rest by median cut
    fixed = np.vstack([[int(surface[i:i + 2], 16) for i in (1, 3, 5)], np.round(cmap(np.arange(LEVELS))[:, :3] * 255)])
    pal = Image.fromarray(sample).quantize(colors=COLOURS - len(fixed), method=Image.Quantize.MEDIANCUT,
                                           dither=Image.Dither.NONE)
    rgb = np.vstack([fixed, np.array(pal.getpalette()[:3 * (COLOURS - len(fixed))]).reshape(-1, 3)]).astype(int)
    tree = cKDTree(rgb)
    index, durations = [], []
    for f, ms in frames:                             # exact nearest-colour mapping
        colours, inverse = np.unique(f.reshape(-1, 3), axis=0, return_inverse=True)
        idx = tree.query(colours)[1].astype(np.uint8)[inverse.reshape(-1)].reshape(f.shape[:2])
        if index and np.array_equal(idx, index[-1]):  # a rejected trajectory: show the same frame for longer
            durations[-1] += ms
        else:
            index.append(idx)
            durations.append(ms)
    # pixels that do not change from one frame to the next are written as transparent
    imgs = []
    for k, idx in enumerate(index):
        im = Image.fromarray(idx if k == 0 else np.where(idx == index[k - 1], TRANSPARENT, idx).astype(np.uint8), "P")
        im.putpalette(rgb.reshape(-1).tolist())
        imgs.append(im)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=durations, loop=0, optimize=False,
                 disposal=1, transparency=TRANSPARENT)
    print(f"wrote {path}  {len(imgs)} frames  {Path(path).stat().st_size / 1e6:.1f} MB")


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "chain_nmax10.npz"
    d = np.load(HERE / src)
    out = HERE
    for theme in THEMES:
        th = THEMES[theme]
        save_gif(render(theme, d), out / f"banner-{theme}.gif", th["surface"],
                 matplotlib.colormaps[th["cmap"]].resampled(LEVELS))


if __name__ == "__main__":
    main()
