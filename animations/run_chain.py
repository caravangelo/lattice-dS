"""Run the short chain shown in the animation, keeping what the page needs.

Same theory and sampler settings as the Set A chains of the paper at
n_max = 10, lambda = 0.01: eps = 0.25, 6 leapfrog steps with randomised length,
the bare mass matrix, 2 exact zero-mode updates per trajectory, hot start. The
loop is the one in `lattice_ds.hmc.run`, written out here only to record, per
trajectory: the accept flag, the trajectory length, the zero mode before and
after its heat-bath update, the power per level, and the configuration
restricted to a few 2-spheres inside S^4.

    python animations/run_chain.py --n-max 10 --n-meas 1500 --seed 20261003 --out animations/chain_nmax10.npz

Several minutes on a laptop.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from lattice_ds import action as A  # noqa: E402
from lattice_ds import grid as G  # noqa: E402
from lattice_ds import harmonics as H  # noqa: E402
from lattice_ds import hmc, zeromode  # noqa: E402
from lattice_ds.transform import S4Transform  # noqa: E402

# 2-spheres shown: fixed chi (latitude x0 = cos chi), psi = pi/2.
CHI_DEG = [90.0, 30.0, 60.0, 120.0, 150.0]


def slice_projector(labels, n_max, chi, psi):
    """(n_s2, n_modes) matrix: S^4 coefficients -> real S^2 coefficients b_lm
    of the field restricted to the 2-sphere at fixed (chi, psi).

    Y_{nLlm} = R4_{nL}(chi) R3_{Ll}(psi) Y_lm(theta, varphi), so the restriction
    is band-limited at l <= n_max and b_lm = sum_{n,L} R4 R3 c_{nLlm}.
    S^2 index: l*l + l + m.
    """
    norm4, norm3 = H._radial_norms(n_max)
    cc, sc = np.cos(chi), np.sin(chi)
    cp, sp = np.cos(psi), np.sin(psi)
    P = np.zeros(((n_max + 1) ** 2, len(labels)))
    g4 = {L: H.gegenbauer_table(L + 1.5, n_max - L, np.array([cc]))[:, 0]
          for L in range(n_max + 1)}
    g3 = {l: H.gegenbauer_table(l + 1.0, n_max - l, np.array([cp]))[:, 0]
          for l in range(n_max + 1)}
    for a, (n, L, l, m) in enumerate(labels):
        r4 = norm4[n, L] * sc**L * g4[L][n - L]
        r3 = norm3[L, l] * sp**l * g3[l][L - l]
        P[l * l + l + m, a] = r4 * r3
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-max", type=int, default=10)
    ap.add_argument("--m2", type=float, default=-0.02822140799315009)
    ap.add_argument("--lam4", type=float, default=0.01)
    ap.add_argument("--eps", type=float, default=0.25)
    ap.add_argument("--n-steps", type=int, default=6)
    ap.add_argument("--zmu", type=int, default=2)
    ap.add_argument("--n-therm", type=int, default=2000)
    ap.add_argument("--n-meas", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--out", default="chain_nmax10.npz")
    a = ap.parse_args()

    n_max = a.n_max
    grid = G.for_degree(4 * n_max)
    basis = S4Transform(grid, n_max)
    lam = H.eigenvalue(basis.shell_index.astype(float))
    ctx = A.ActionContext(Y=basis, w=grid.weights, lam=lam, m2=a.m2, lam4=a.lam4)
    Y, w, q, lam4 = A.unpack(ctx)
    vol = float(np.sum(w))
    minv = A.default_minv(q)
    sqrt_mass = np.sqrt(1.0 / minv)
    n_modes = ctx.n_modes
    shell_index = basis.shell_index
    counts = np.bincount(shell_index).astype(float)

    labels = [tuple(int(v) for v in r) for r in basis.labels]
    projs = [slice_projector(labels, n_max, np.deg2rad(c), np.pi / 2) for c in CHI_DEG]
    # self-check of the projector against harmonics.evaluate_at
    rng0 = np.random.default_rng(1)
    ctest = rng0.standard_normal(n_modes)
    th, ph = 0.7, 2.1
    for cdeg, P in zip(CHI_DEG, projs):
        chi = np.deg2rad(cdeg)
        pt = np.array([[np.cos(chi), 0.0, np.sin(chi) * np.cos(th),
                        np.sin(chi) * np.sin(th) * np.cos(ph),
                        np.sin(chi) * np.sin(th) * np.sin(ph)]])
        direct = float(H.evaluate_at(pt, n_max)[:, 0] @ ctest)
        b = P @ ctest
        val = 0.0
        for l in range(n_max + 1):
            for m in range(-l, l + 1):
                val += b[l * l + l + m] * H.real_sph_harm(l, m, np.array([th]), np.array([ph]))[0, 0]
        assert abs(val - direct) < 1e-9 * max(1, abs(direct)), (cdeg, val, direct)
    print("slice projector agrees with evaluate_at", flush=True)

    rng = np.random.default_rng(a.seed)
    c = hmc.free_start(rng, q, minv)
    lo = max(1, int(round(0.7 * a.n_steps)))
    hi = max(lo + 1, int(round(1.3 * a.n_steps)))
    K = a.n_meas
    n_s2 = (n_max + 1) ** 2
    slices = np.zeros((K, len(CHI_DEG), n_s2), dtype=np.float32)
    acc = np.zeros(K, dtype=bool)
    T = np.zeros(K)
    c0_pre = np.zeros(K)
    c0_post = np.zeros(K)
    shell = np.zeros((K, n_max + 1))
    phi2 = np.zeros(K)
    dH = np.zeros(K)
    act = np.zeros(K)
    start_slices = None
    start_c0 = None

    n_acc = 0
    t0 = time.time()
    total = a.n_therm + K
    for it in range(total):
        if it == a.n_therm:
            start_slices = np.array([P @ c for P in projs], dtype=np.float32)
            start_c0 = c[0]
        p = sqrt_mass * rng.standard_normal(n_modes)
        ns = int(rng.integers(lo, hi + 1))
        h0 = A.hamiltonian(c, p, Y, w, q, lam4, minv)
        c_new, p_new = A.leapfrog(c, p, a.eps, ns, Y, w, q, lam4, minv)
        h1 = A.hamiltonian(c_new, p_new, Y, w, q, lam4, minv)
        d = h1 - h0
        accepted = np.log(rng.random()) < -d
        if accepted:
            c = c_new
        n_acc += int(accepted)
        pre = c[0]
        for _ in range(a.zmu):
            zeromode.update(rng, c, Y, w, q, lam4, vol)
        k = it - a.n_therm
        if k >= 0:
            acc[k] = accepted
            T[k] = a.eps * ns
            dH[k] = d
            c0_pre[k] = pre
            c0_post[k] = c[0]
            phi2[k] = float(c @ c) / vol
            act[k] = A.action(c, Y, w, q, lam4)
            shell[k] = np.bincount(shell_index, weights=c * c) / counts
            for s, P in enumerate(projs):
                slices[k, s] = P @ c
        if (it + 1) % 100 == 0:
            el = time.time() - t0
            print(f"iter {it+1}/{total} acc={n_acc/(it+1):.3f} "
                  f"{el/(it+1):.3f}s/traj eta {el/(it+1)*(total-it-1)/60:.1f} min",
                  flush=True)

    np.savez_compressed(
        a.out, slices=slices, start_slices=start_slices, start_c0=start_c0,
        acc=acc, T=T, dH=dH, c0_pre=c0_pre, c0_post=c0_post, shell=shell,
        phi2=phi2, action=act, chi_deg=np.array(CHI_DEG), n_max=n_max,
        m2=a.m2, lam4=a.lam4, eps=a.eps, n_steps=a.n_steps, zmu=a.zmu,
        seed=a.seed, n_therm=a.n_therm, vol=vol, q=q, shell_index=shell_index,
        grid_points=grid.n_points, n_modes=n_modes,
    )
    print(f"done acc={n_acc/total:.3f} <exp(-dH)>={np.mean(np.exp(-dH)):.4f}")


if __name__ == "__main__":
    main()
