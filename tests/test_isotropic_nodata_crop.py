"""Regression: isotropic side-radiation over a partially-nodata tile.

When a tile is only partially valid, ``calculate_core_fused`` crops the heavy
compute to the valid bounding box; the cropped arrays are non-contiguous numpy
views. The isotropic ``kside_veg_isotropic_pure`` path takes the u8 ``valid``
mask by slice, so a non-contiguous valid mask made the Rust side panic
("vegetation.rs invariant: all f32 arrays here are contiguous"). The valid mask
must be made contiguous at the FFI boundary. Anisotropic runs never call that
function, so this only affects isotropic runs over irregular extents.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
from conftest import make_mock_svf, read_timestep_geotiff
from solweig import Location, SurfaceData, Weather
from solweig.timeseries import _calculate_timeseries


@pytest.mark.slow
def test_isotropic_partial_valid_tile_does_not_panic(tmp_path):
    """A partially-nodata tile triggers the valid-bbox crop; isotropic must not panic."""
    size = 100
    dsm = np.full((size, size), np.nan, dtype=np.float32)
    # Valid data in a centred sub-block only -> the valid bbox is a strict
    # sub-rectangle, so use_crop activates and the cropped valid mask is
    # non-contiguous. No DEM, so the nodata border stays NaN.
    dsm[20:80, 20:80] = 10.0
    surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf((size, size)))
    location = Location(latitude=40.4, longitude=-3.7, utc_offset=1)
    weather = [Weather(datetime=datetime(2024, 7, 15, 13, 0), ta=30.0, rh=40.0, global_rad=800.0, ws=2.0)]

    _calculate_timeseries(
        surface=surface,
        weather_series=weather,
        location=location,
        output_dir=tmp_path,
        outputs=["tmrt"],
        use_anisotropic_sky=False,
    )

    tmrt = read_timestep_geotiff(tmp_path, "tmrt", 0)
    assert tmrt is not None
    assert np.isfinite(tmrt[30:70, 30:70]).any(), "valid sub-block must produce finite Tmrt"
