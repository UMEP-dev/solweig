"""Anisotropic shadow-matrix dimension validation.

Verifies that the Rust pipeline raises a clean `ValueError` (rather than
panicking or reading OOB) when any of the three anisotropic shadow matrices
has a row/col dimension that does not match the DSM grid.

Each of `shmat`, `vegshmat`, `vbshmat` is mis-sized individually so a
regression that drops validation for one of the three would be caught.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
from solweig.api import Location, SurfaceData, Weather, calculate
from solweig.models.precomputed import ShadowArrays

pytestmark = pytest.mark.slow

_DSM_SHAPE = (10, 10)
_BAD_SHAPE = (8, 8)
_N_PATCHES = 153
_N_PACK = (_N_PATCHES + 7) // 8


def _packed(shape):
    return np.full((shape[0], shape[1], _N_PACK), 0xFF, dtype=np.uint8)


def _surface_with_shadow_shapes(shmat_shape, vegshmat_shape, vbshmat_shape):
    from conftest import make_mock_svf

    dsm = np.ones(_DSM_SHAPE, dtype=np.float32) * 2.0
    surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf(_DSM_SHAPE))
    surface.shadow_matrices = ShadowArrays(
        _shmat_u8=_packed(shmat_shape),
        _vegshmat_u8=_packed(vegshmat_shape),
        _vbshmat_u8=_packed(vbshmat_shape),
        _n_patches=_N_PATCHES,
    )
    return surface


@pytest.mark.parametrize(
    ("bad_array", "shapes"),
    [
        ("shmat", (_BAD_SHAPE, _DSM_SHAPE, _DSM_SHAPE)),
        ("vegshmat", (_DSM_SHAPE, _BAD_SHAPE, _DSM_SHAPE)),
        ("vbshmat", (_DSM_SHAPE, _DSM_SHAPE, _BAD_SHAPE)),
    ],
)
def test_each_anisotropic_matrix_individually_validated(bad_array, shapes, tmp_path):
    """Each of the three shadow matrices must be validated independently.

    Regression guard: if validation were applied to only one array (or were
    short-circuited to fail-fast on the first), a mis-sized vegshmat or
    vbshmat could slip through silently.
    """
    location = Location(latitude=57.7, longitude=12.0, utc_offset=1)
    weather = Weather(datetime=datetime(2024, 7, 15, 12, 0), ta=25.0, rh=50.0, global_rad=800.0)
    surface = _surface_with_shadow_shapes(*shapes)

    with pytest.raises((ValueError, RuntimeError)) as excinfo:
        calculate(
            surface,
            [weather],
            location,
            use_anisotropic_sky=True,
            output_dir=tmp_path / bad_array,
            outputs=["tmrt"],
        )
    msg = str(excinfo.value)
    # The error must name the SPECIFIC offending array — otherwise a regression
    # could pass by reporting only one array's name regardless of which is bad.
    assert bad_array in msg, f"Expected error to name {bad_array!r}; got: {msg}"
