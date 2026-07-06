"""Spec tests for the Crawford & Duchon (1999) clearness index.

Regression gate for the pressure unit bug: ``Weather.pressure`` is hPa, and
1 hPa == 1 mb, so the transmission formula must use it unscaled. Applying
classic UMEP's kPa-to-mb x10 to an hPa input drove p to ~10000 mb and
underestimated clear-sky irradiance by ~20-25%.
"""

from __future__ import annotations

import numpy as np
import pytest
from solweig.physics.clearnessindex_2013b import clearnessindex_2013b

LOCATION = {"latitude": 38.0, "longitude": 23.75, "altitude": 0.0}
ZEN_30 = np.radians(30.0)
MIDSUMMER = 180


def _i0(pressure_hpa: float) -> float:
    i0, _, _, _, _ = clearnessindex_2013b(ZEN_30, MIDSUMMER, 25.0, 0.5, 800.0, LOCATION, pressure_hpa)
    return float(i0)


def test_standard_pressure_matches_default_atmosphere():
    """P=1013.25 hPa must reproduce the -999 standard-atmosphere fallback."""
    assert _i0(1013.25) == pytest.approx(_i0(-999.0), rel=1e-3)


def test_clear_sky_irradiance_plausible_midsummer():
    """Clear-sky global irradiance at 30 deg zenith is ~900 W/m2, not ~740.

    With the x10 pressure bug, this case produced I0 = 743 W/m2; the
    Crawford & Duchon transmission chain at p = 1013 mb gives 916 W/m2.
    """
    i0 = _i0(1013.25)
    assert 850.0 < i0 < 1000.0


def test_lower_station_pressure_increases_transmission():
    """Less atmosphere (mountain station) means less Rayleigh scattering."""
    assert _i0(900.0) > _i0(1013.25)


def test_clearness_index_below_one_for_subclear_sky():
    """radG below the clear-sky value must give CI < 1 + correction margin.

    With the pressure bug, I0 was so low that measured radG could exceed it,
    driving corrected CI well above 1 and disabling the Ldown cloud
    correction for genuinely hazy conditions.
    """
    _, ci, _, _, _ = clearnessindex_2013b(ZEN_30, MIDSUMMER, 25.0, 0.5, 700.0, LOCATION, 1013.25)
    assert ci < 1.0
