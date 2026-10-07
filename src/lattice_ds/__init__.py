"""lattice-dS: lattice Monte Carlo of quantum field theory on Euclidean de Sitter space (S^4).

The package simulates a real scalar field with a lambda phi^4 interaction. The
field is expanded in hyperspherical harmonics up to level n_max, the quartic
interaction is integrated exactly on a product Gauss grid, and the mode
coefficients are sampled with hybrid Monte Carlo. The regulator keeps the
de Sitter isometry group SO(5) exact at finite cutoff.

Modules
-------
grid          product Gauss quadrature on S^4
harmonics     real hyperspherical harmonics
transform     separable mode <-> grid transform
action        regulated action, gradient, leapfrog, HMC mass matrices
hmc           the sampler
zeromode      exact heat-bath update of the constant mode
constrained   simulations at fixed zero mode; mean force and curvature
meanforce     massless point and effective coupling from the mean force
perturbation  loop sums, one-loop tuning, Starobinsky-Yokoyama reference values
stats         autocorrelation, jackknife and bootstrap helpers

Units: H = 1 (unit sphere), Vol(S^4) = 8 pi^2 / 3.
"""

from . import (action, constrained, grid, harmonics, hmc, meanforce,
               perturbation, stats, transform, zeromode)
from .grid import VOL_S4

__version__ = "1.0.0"


def setup(n_max, m2, lam4):
    """Grid, transform and action context for one simulation point.

    Returns (ctx, basis): an `action.ActionContext` on the de-aliased grid of
    degree 4 n_max, and the `transform.S4Transform`. `basis.shell_index` and
    `basis.labels` give the level n and the labels (n, L, l, m) of every mode.

    >>> ctx, basis = setup(4, m2=0.5, lam4=0.1)
    >>> r = hmc.run(ctx, basis.shell_index, 4, eps=0.25, n_steps=6,
    ...             n_therm=200, n_meas=1000, zero_mode_updates=2,
    ...             keep_samples=False)
    """
    g = grid.for_degree(4 * n_max)
    basis = transform.S4Transform(g, n_max)
    lam = harmonics.eigenvalue(basis.shell_index.astype(float))
    ctx = action.ActionContext(Y=basis, w=g.weights, lam=lam, m2=m2, lam4=lam4)
    return ctx, basis


__all__ = ["setup", "VOL_S4", "action", "constrained", "grid", "harmonics", "hmc",
           "meanforce", "perturbation", "stats", "transform", "zeromode"]
