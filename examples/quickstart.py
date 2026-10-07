"""A first example: the Starobinsky-Yokoyama distribution from a Monte Carlo chain.

Simulates lambda phi^4 on S^4 at cutoff n_max = 4 and lambda = 0.1, with the
bare mass set to the one-loop counterterm so that the renormalised mass
vanishes at one loop, and compares the zero mode with the massless
Starobinsky-Yokoyama (SY) law P(phibar) ~ exp(-Omega lambda phibar^4 / 4).

    python examples/quickstart.py

Takes about a minute on one core of a recent laptop.
"""

import math

import lattice_ds as L
from lattice_ds import hmc
from lattice_ds import perturbation as P
from lattice_ds.stats import binder, binder_jackknife_error, error_of_mean, tau_int

n_max, lam = 4, 0.1
m2 = P.one_loop_bare_mass(n_max, lam)                # m^2 = -3 lambda G'(0)
ctx, basis = L.setup(n_max, m2, lam)
print(f"n_max = {n_max}: {ctx.n_modes} modes, {ctx.w.size} quadrature points, bare mass m^2 = {m2:.5f}")

r = hmc.run(ctx, basis.shell_index, n_max,
            eps=0.25, n_steps=6,                     # trajectory length 1.5, randomised by +-30%
            n_therm=1000, n_meas=20000, seed=1,
            zero_mode_updates=2,                     # exact heat-bath draws of the zero mode
            keep_samples=False, progress_every=5000)
print(f"acceptance {r['acceptance']:.3f}, <exp(-dH)> = {r['exp_negdH']:.4f} (must be 1)")

phibar = r["c0"] / math.sqrt(L.VOL_S4)               # zero mode
sq = phibar ** 2
var, err = sq.mean(), error_of_mean(sq, tau_int(sq))
U, U_err = binder(phibar), binder_jackknife_error(phibar)
sy = P.SY_PHIBAR2 / math.sqrt(lam)
one_loop = sy * (1 + 0.5 * P.screening_coefficient(n_max) * lam)

print(f"<phibar^2>      = {var:.4f} +- {err:.4f}")
print(f"  SY            = {sy:.4f}    ({(var - sy) / err:+.1f} sigma)")
print(f"  SY + one loop = {one_loop:.4f}    ({(var - one_loop) / err:+.1f} sigma; coupling renormalised at one loop)")
print(f"Binder cumulant = {U:.4f} +- {U_err:.4f}")
print(f"  SY            = {P.SY_BINDER:.4f}    ({(U - P.SY_BINDER) / U_err:+.1f} sigma; a Gaussian has U = 0)")
