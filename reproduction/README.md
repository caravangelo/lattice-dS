# Data and analysis of the paper

Data tables, analysis and figure scripts, and chain format for

> A. Caravano, *Lattice Monte Carlo of $\lambda\phi^4$ on Euclidean de Sitter*, arXiv:2610.12547 (2026).

Data are released under CC-BY-4.0 (`LICENSE-DATA` at the top of the repository). If you use them, please cite the paper.

## Contents

```
make_figures.py      Figs. 1-3 of the paper from data/ (numpy and matplotlib only)
reproduce_tables.py  recomputes the tables in data/ from the chains
fit_histograms.py    histogram estimate of the massless point from the mass scans (Appendix D3; needs the chains)
fig_massless.py      appendix figure on the massless point (needs the chains)
fig_so5.py           appendix figure on SO(5) invariance against the grid degree
fig_mc_sampling.py   appendix figure on the Monte Carlo sampling (chain in ../animations/)
data/                per-point results and the data behind the figures
chains/manifest.csv  list of the Monte Carlo chains
figures/             output of the figure scripts
```

The Monte Carlo chains (4041 files, 5.8 GB) are archived on Zenodo, [doi:10.5281/zenodo.23242257](https://doi.org/10.5281/zenodo.23242257); it might take a few days for the record to be available at this DOI. Each file contains, for one chain, the observables measured on every configuration; the field configurations themselves (136 GB) are available on request. Download `chains_constrained.zip` and `chains_unconstrained.zip` and unzip them inside `reproduction/`; this creates `chains/constrained/` and `chains/unconstrained/` next to `chains/manifest.csv`.

## Reproducing the paper

There are three levels, from cheapest to most expensive. Commands are run from the top of the repository.

**1. Figures from the tables** (seconds).

```bash
python reproduction/make_figures.py      # Figs. 1-3, and the numbers quoted in the text
python reproduction/fig_so5.py           # appendix: SO(5) invariance against the grid degree
python reproduction/fig_mc_sampling.py   # appendix: Monte Carlo sampling figure
```

**2. Tables from the published chains.** With the chains in `reproduction/chains/`,

```bash
python reproduction/reproduce_tables.py --chains reproduction/chains setA
python reproduction/reproduce_tables.py --chains reproduction/chains setA-coupling
python reproduction/reproduce_tables.py --chains reproduction/chains setB --jobs 4
python reproduction/fit_histograms.py   --chains reproduction/chains      # m2_crossing columns (minutes)
python reproduction/fig_massless.py     --chains reproduction/chains      # appendix figure
```

recompute `data/*.csv` into `recomputed/` and print the largest deviation from the published values, column by column. Set A takes seconds, Set B about two hours of CPU time, the histogram fits a few minutes. Every number, including the bootstrap errors, is a deterministic function of the chains; the recomputed tables are written with the rounding of the published ones and reproduce them to within that rounding (the scripts report how many values are identical once rounded; a different platform may change the last printed digit of a few values).

**3. Chains from scratch.** The `lattice-ds` command (top-level README) writes files in the format of the published chains.

```bash
# Set A: one-loop bare mass, 18 chains of 25 000 measurements per point
lattice-ds run --n-max 4 6 8 10 --lam 0.01 0.02 0.05 0.1 0.2 0.5 \
    --one-loop-mass --seeds 18 --n-meas 25000 --outdir runs/setA

# Set B: unconstrained scan in the bare mass around the massless point
lattice-ds run --n-max 4 --lam 20 --m2 -11.7 -11.6 -11.5 -11.4 \
    --precondition renorm --eps-auto --seeds 6 --n-meas 30000 --outdir runs/scan_n4_l20

# Set B: constrained chain at fixed zero mode (mean force; curvature at phibar = 0)
lattice-ds run-constrained --tag n4_l20_m0.00_s1 --n-max 4 --lam 20 --m2 -11.55 \
    --phibar 0.0 --n-meas 100000 --outdir runs/constrained

# unconstrained chain that also records the force on the zero mode (files force_*)
lattice-ds run-with-force --tag force_n6_l30 --n-max 6 --lam 30 --m2 -28.8 --n-meas 50000 --outdir runs/force
```

The published files record the seed, step size and number of leapfrog steps of every chain (`lattice-ds run --seed <seed> ...` re-runs a given chain; by default the seed of a chain is derived from its point $(n_{\max}, m^2, \lambda)$ and chain index, so chains at different points are independent). A rerun on a different machine or BLAS library follows a different floating-point path, so agreement with the published chains is statistical, not bit by bit. The data set of the paper is about 4000 chains; one trajectory takes 2 ms to 37 ms between $n_{\max} = 4$ and $10$ on one core of a MacBook Pro (Apple M3 Max); see the top-level README for details.

## Notation

Units, the action and the zero mode $\bar\phi = c_0/\sqrt{\Omega}$ are as in the [top-level README](../README.md#conventions). The tables also use:

- the Starobinsky–Yokoyama (SY) distribution at the massless point, $P(\bar\phi) \propto e^{-\Omega\lambda\bar\phi^4/4}$, with $\langle\bar\phi^2\rangle_{\rm SY} = 0.13176/\sqrt{\lambda}$ and Binder cumulant $U_{\rm SY} = 1 - \langle\bar\phi^4\rangle/3\langle\bar\phi^2\rangle^2 = 0.27052$;
- the one-loop sums $S_k(n_{\max}) = \sum_{n=1}^{n_{\max}} d_n/\lambda_n^k$, with $S_2 = 0.6152,\ 0.7201,\ 0.7986,\ 0.8615$ for $n_{\max} = 4, 6, 8, 10$;
- the couplings $x = 9\lambda S_2/\Omega$ and $y = 9\lambda_R^{\rm (eff)} S_2/\Omega$.

Two simulation sets:

- **Set A** ($n_{\max} = 4, 6, 8, 10$; $\lambda = 0.01$–$0.5$): bare mass fixed to the analytic one-loop counterterm $m^2 = -3\lambda S_1/\Omega$.
- **Set B** ($n_{\max} = 4, 6, 8, 10$; $\lambda = 0.2$–$60$ in the paper; the tables also list $n_{\max} = 4$ at $\lambda = 80, 120$, not used): bare mass tuned nonperturbatively so that the renormalized mass $m_R^2 = V_C''(0)$ vanishes, where $V_C$ is the constraint effective potential of the zero mode, measured as a mean force from fixed-$\bar\phi$ ("constrained") chains.

## data/

`setA_points.csv` — one row per $(n_{\max}, \lambda)$ of Set A (`role` column).

| column | meaning |
|---|---|
| `m2_bare` | bare mass $m^2 = -3\lambda S_1/\Omega$ |
| `n_chains`, `n_meas_per_chain` | independent chains and measurements (HMC trajectories) per chain |
| `phibar2`, `phibar2_err` | $\langle\bar\phi^2\rangle$ in units of $H^2$, inverse-variance combination of the chains |
| `phibar2_SY` | $0.13176/\sqrt{\lambda}$ |
| `U`, `U_err` | Binder cumulant, blocked-jackknife error (100 blocks per chain), combined over chains |

`fig1_zero_mode_samples.npz` — the zero-mode values $c_0$ of all Set A chains at $n_{\max} = 10$ for $\lambda = 0.01, 0.1, 0.2$ (keys `lam0.01`, `lam0.1`, `lam0.2`, concatenated over chains; `vol` $= \Omega$). Used for the histograms of Fig. 1(a). The same values are in `chains/`, chain by chain.

`setA_effective_coupling.csv` — Set A effective quartic coupling $\lambda_R^{\rm (eff)} = V_C'(\sigma)/\sigma^3$ (with $\sigma^2 = \langle\bar\phi^2\rangle$) from three constrained chains at $0.8\sigma$, $\sigma$, $1.2\sigma$, as `x`, `y`, `y_err` (Fig. 3, open symbols).

`setB_points.csv` — one row per Set B point at the massless point $m_*^2$.

| column | meaning |
|---|---|
| `x`, `y_chain` | $9\lambda S_2/\Omega$ and the bubble-chain value $x/(1+x)$ |
| `y`, `y_err_stat`, `y_err_window` | $9\lambda_R^{\rm (eff)} S_2/\Omega$; block-bootstrap error (includes the uncertainty of $m_*^2$) and the fit-window systematic of $m_*^2$ (add in quadrature for a total error) |
| `U`, `U_err` | Binder cumulant at $m_*^2$ by exact reweighting of the unconstrained ensembles |
| `sigma` | $\sigma = \langle\bar\phi^2\rangle^{1/2}$ at $m_*^2$ |
| `m2_star`, `m2_star_err` | massless bare mass from the joint fit $V_C'/\bar\phi = m_R^2 + \lambda_R\bar\phi^2 + a_6\bar\phi^4$ of the reweighted curvature and forces, and its statistical error |
| `fit_window` | upper edge in $\bar\phi$ of the joint fit |
| `local_exponent` | $d\ln V_C'/d\ln\bar\phi$ at $\sigma$ (3 for a pure quartic) |
| `y_taylor`, `y_taylor_err` | $9\lambda_R S_2/\Omega$ with $\lambda_R$ the quartic Taylor coefficient at the origin from the same fit |
| `used_in_paper` | 0 for the $n_{\max} = 4$ points at $\lambda = 80$ and $120$ ($\lambda = 240$ was also simulated and is not tabulated), whose massless point is prescription dependent because the induced mass of the nonzero modes at the typical field approaches the cutoff; they are excluded from the figures and from every number quoted in the paper, which restricts Set B to $\lambda \le 60$ (Appendix D4). Their chains are in `chains/`. |
| `m2_crossing`, `m2_crossing_err` | independent estimate of the massless point from the joint fit of the zero-mode histograms of the mass scans (Appendix D3 of the paper; `fit_histograms.py`, which also writes `data/histogram_crossing.csv` with the selected model and the number of masses and chains per point), at the points where the dense scan exists and the fit passes the calibrated goodness-of-fit test; shown in the appendix figure on the massless point (`fig_massless.py`) |
| `chi2`, `dof` | of the joint fit at $m_*^2$ |
| `n_constrained`, `n_unconstrained_files` | number of constrained chains and of unconstrained files pooled into the ensemble at the point |
| `ess_unconstrained` | effective sample fraction of the unconstrained ensemble after reweighting to $m_*^2$ |
| `m2_star_alt` | massless point from the alternative fit window, which gives `y_err_window` |

`so5_scan.json` — SO(5) violation of the interaction and orthonormality error against the degree of the quadrature grid (appendix figure; `fig_so5.py`).

## chains/ (Zenodo)

`manifest.csv` lists every file with `kind`, `set`, `n_max`, `lam`, `m2_bare`, `phibar_fixed` (constrained chains), `n_meas`, `seed`, `host` (`laptop`: MacBook Pro M3 Max, macOS; `snellius`: AMD Genoa nodes of the Snellius cluster, Linux — 533 of the mass-scan chains), `acceptance`, `exp_negdH` (the HMC identity $\langle e^{-\Delta H}\rangle = 1$; averaged over the measurement trajectories for the constrained chains, and over all trajectories, thermalization included, for the unconstrained ones). Every chain listed passed the health criteria (acceptance $\ge 0.3$, $0.9 < \langle e^{-\Delta H}\rangle < 1.1$); thermalization trajectories are not included. All arrays are per measurement (one HMC trajectory), in order.

`chains/unconstrained/*.npz` (Set A: `setA_*`; Set B mass scans: `setB_*`, at several bare masses per point; unconstrained chains with the force recorded: `force_*`):

| key | meaning |
|---|---|
| `c0` | zero-mode coefficient $c_0 = \sqrt{\Omega}\,\bar\phi$ |
| `phi2` | $\Omega^{-1}\int_{S^4}\phi^2$, the volume average of $\phi^2$ (all modes). Writing $p$ for `phi2`, the observable conjugate to $m^2$ is $\int\phi^2 = \Omega p$, so reweighting from $m_s^2$ to $m_s^2 + \delta$ uses the weight $e^{-\delta\Omega p/2}$ |
| `force` | (force chains only) $\Omega^{-1}\,\partial S/\partial\bar\phi$ per configuration |
| `vol`, `n_max`, `lam4`, `m2`, `seed`, `n_meas`, `acceptance`, `exp_negdH`, `eps`, `n_steps` | metadata ($\Omega$, cutoff, $\lambda$, bare mass, …) |
| `action`, `shell_series`, `shell_power`, `mode_power`, `shell_index`, `n_therm`, `therm_acceptance`, `zero_mode_acceptance`, `exp_negdH_meas`, `dH_max`, `settings`, `minutes` | (`setA_*` and `setB_*` files, written by `lattice-ds run`, see its `--help`) action per configuration; mean of $c_{nI}^2$ per level $n$ per configuration and averaged; mean of $c_A^2$ per mode and the level of each mode; thermalization length and acceptance; zero-mode heat-bath acceptance; $\langle e^{-\Delta H}\rangle$ over the measurement trajectories; largest $|\Delta H|$; the run settings (JSON); wall-clock minutes. Seeds were derived by `lattice-ds run` from the point and the chain index with the default base seed 260910, except for the 32 chains of 25 000 measurements at the one-loop bare mass, which share their bare mass with 8 000-measurement scan chains and were run with base seed 260911 |

`chains/constrained/*.npz` (fixed-$\bar\phi$ chains; tag `..._m0.15_s2` means $\bar\phi = 0.15$, seed 2; `..._f0.8` means $\bar\phi = 0.8\sigma$):

| key | meaning |
|---|---|
| `force` | estimator of $V_C'(\bar\phi)$ per configuration, $\Omega^{-1}\partial S/\partial\bar\phi = m^2\bar\phi + \lambda\bar\phi^3 + (3\lambda\bar\phi/\Omega)\int\psi^2 + (\lambda/\Omega)\int\psi^3$, with $\psi = \phi - \bar\phi$ |
| `A`, `B` | $A = \lambda\Omega^{-1/2}\int\psi^3$, $B = m^2 + 3\lambda\Omega^{-1}\int\psi^2$; at $\bar\phi = 0$ the curvature is $m_R^2 = V_C''(0) = \langle B\rangle - \langle A^2\rangle$; $\int\psi^2 = (B - m^2)\,\Omega/3\lambda$ gives the reweighting weight $\exp(-\delta\int\psi^2/2)$ |
| `phibar`, `m2`, `lam4`, `n_max`, `vol`, `n_meas`, `seed`, `acceptance`, `exp_negdH`, `eps`, `n_steps` | metadata |

With these chains every number in the paper can be recomputed: the mean force and curvature at any nearby bare mass by reweighting, the massless point by the joint fit, $\sigma$ and $U$ by reweighting the unconstrained ensembles, and the block bootstrap of all of it. `reproduce_tables.py` does this.
