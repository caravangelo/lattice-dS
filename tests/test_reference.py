"""Analytic reference data of the free theory on S^4."""

import numpy as np
import pytest

from lattice_ds import harmonics as H
from lattice_ds import perturbation as P
from lattice_ds.grid import VOL_S4


def test_volume():
    assert VOL_S4 == pytest.approx(8.0 * np.pi**2 / 3.0)


def test_degeneracy_and_eigenvalue():
    # S^4 scalar harmonics: 1, 5, 14, 30, 55, ... ; eigenvalues n(n+3)
    assert [int(H.degeneracy(n)) for n in range(5)] == [1, 5, 14, 30, 55]
    assert [int(H.eigenvalue(n)) for n in range(5)] == [0, 4, 10, 18, 28]


def test_mode_counts():
    assert H.n_modes(8) == 825
    assert H.n_modes(10) == 1716


def test_gegenbauer_ratio_endpoint():
    assert np.allclose(P.gegenbauer_ratio(12, 1.0), 1.0)


def test_mode_sum_matches_continuum_propagator():
    """Mode sum through n = 400 000 against the hypergeometric closed form."""
    worst = 0.0
    for a, b in [(0.25, 2.0), (0.05, 1.0), (0.5, 2.2), (1.0, 2.0), (0.01, 0.5)]:
        for th in [1.0, np.pi / 2, 2.2, 3.0]:
            ms = P.propagator_mode_difference(th, a, b)
            cf = P.propagator_closed_form(th, a) - P.propagator_closed_form(th, b)
            worst = max(worst, abs(ms - cf) / abs(cf))
    assert worst < 1e-12


def test_loop_sums_and_sy_constants():
    # values quoted in the paper
    assert [round(P.loop_sum(2, n), 4) for n in (4, 6, 8, 10)] == [0.6152, 0.7201, 0.7986, 0.8615]
    assert P.SY_PHIBAR2 == pytest.approx(0.13176, abs=5e-6)
    assert P.SY_BINDER == pytest.approx(0.27052, abs=5e-6)
    assert P.sy_phi2_massless(0.3) == pytest.approx(P.SY_PHIBAR2 / np.sqrt(0.3), rel=1e-12)
    for n_max in (4, 10):
        assert P.coincident_propagator_nonzero(n_max, 0.0) == pytest.approx(P.loop_sum(1, n_max) / VOL_S4, rel=1e-12)
        assert P.one_loop_bare_mass(n_max, 0.2) == pytest.approx(-0.6 * P.loop_sum(1, n_max) / VOL_S4, rel=1e-12)
