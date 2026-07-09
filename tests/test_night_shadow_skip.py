"""Night shadow: below the horizon the sun casts no obstruction shadows.

When the sun altitude is negative, the shadow ray-march runs downward and can
never meet an obstruction, so every pixel is "sunlit" (1.0) — identical to
running the full cast. ``calculate_shadows_rust`` therefore skips the march at
night and returns the all-sunlit result directly. Shortwave is zeroed at night,
so this does not change any output; it only saves the wasted ray-march.

This test locks the night-shadow output invariant (a building that would cast a
real shadow by day produces no shadow at night).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
from conftest import make_mock_svf
from solweig import Location, SurfaceData, Weather
from solweig.api import _calculate_single


@pytest.mark.slow
def test_night_shadow_is_all_sunlit_with_finite_tmrt():
    size = 120
    dsm = np.full((size, size), 10.0, dtype=np.float32)
    dsm[40:80, 40:80] = 30.0  # a 20 m building — would cast a long shadow by day
    surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf((size, size)))
    loc = Location(latitude=40.4, longitude=-3.7, utc_offset=1)
    w = Weather(datetime=datetime(2024, 7, 15, 1, 0), ta=20.0, rh=60.0, global_rad=0.0, ws=2.0)
    w.compute_derived(loc)
    assert w.sun_altitude < 0, f"expected a night timestep, got altitude {w.sun_altitude}"

    r = _calculate_single(
        surface=surface, location=loc, weather=w, use_anisotropic_sky=False, max_shadow_distance_m=500
    )

    shadow = np.asarray(r.shadow)
    valid = np.isfinite(shadow)
    assert valid.any()
    # No obstruction shadows at night: every valid pixel is fully sunlit.
    assert np.all(shadow[valid] == 1.0), "night shadow must be all-sunlit (1.0)"

    tmrt = np.asarray(r.tmrt)
    assert np.isfinite(tmrt[valid]).all(), "night Tmrt must be finite where valid"
