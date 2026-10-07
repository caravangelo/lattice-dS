# Sampling the four-sphere

This folder contains the scripts that make the [interactive animation](https://caravangelo.github.io/lattice-dS/) of the Monte Carlo sampling and the banner at the top of the main README. Both show the same Monte Carlo chain: 1500 trajectories at $n_{\max} = 10$ and $\lambda = 0.01$, after 2000 thermalization trajectories, with the sampler settings of Set A of the paper.

- `run_chain.py` runs the chain and stores what the page needs (several minutes).
- `chain_nmax10.npz` is its output for seed 20261003, the chain shown.
- `build_page.py` packs the chain into `site/index.html` using `template.html`. The page is one self-contained file; `.github/workflows/pages.yml` publishes `site/` with GitHub Pages.
- `make_banner.py` draws the README banner (below).
- `bands.py` computes the 68% and 95% bands of the running estimates from 20 000 independent chains drawn from the Starobinsky–Yokoyama distribution.

The Monte Carlo figure in the appendix of the paper is made from the same chain by `reproduction/fig_mc_sampling.py`, which reads `chain_nmax10.npz` and `bands.py` from this folder.

To rebuild everything:

```bash
python animations/run_chain.py --n-max 10 --n-meas 1500 --seed 20261003 --out animations/chain_nmax10.npz
python animations/build_page.py
python animations/make_banner.py
```

## How the field is drawn

Restricted to a 2-sphere at fixed $(\chi, \psi)$, a field with modes $n \le n_{\max}$ is a function on $S^2$ with $l \le n_{\max}$, so each sphere needs only $(n_{\max} + 1)^2 = 121$ numbers per configuration. The page stores those for five 2-spheres ($\chi = 30^\circ, 60^\circ, 90^\circ, 120^\circ, 150^\circ$ at $\psi = \pi/2$) and synthesises the pictures in the browser.

Every frame at a trajectory boundary is a configuration of the chain. Between two stored configurations, at slow tempo, the nonzero modes are moved along the free Hamiltonian flow that joins them (a unit-frequency rotation, exact for the Fourier-accelerated free theory); this interpolation is for display only. A rejected trajectory leaves the field unchanged, and the exact redraw of the zero mode is shown as a uniform shift.

## README banner

`make_banner.py` draws the banner of the top-level README, `banner-light.gif` and `banner-dark.gif` (one for each GitHub colour theme), directly from the same chain: the first trajectories slowly, then the whole chain in a few seconds. The field is drawn in 20 colour steps.
