"""calculate() rejects layers whose shape differs from the DSM before reaching Rust.

The Rust extension is built with ``panic = "abort"``, so a shape mismatch that
reaches the pipeline kills the interpreter (and QGIS with it) instead of
raising. Up to 0.1.0b97 a mis-shaped albedo, emissivity or SVF array did
exactly that, and a mis-shaped CDSM surfaced as a bare numpy broadcast error.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pytest
from conftest import make_mock_svf
from solweig import Location, SurfaceData, Weather, calculate
from solweig.errors import GridShapeMismatch

SHAPE = (20, 20)
BAD = (12, 12)

LOCATION = Location(latitude=57.7, longitude=12.0, utc_offset=1)
WEATHER = Weather(datetime=datetime(2024, 7, 15, 12, 0), ta=25.0, rh=50.0, global_rad=800.0)


def _dsm():
    dsm = np.zeros(SHAPE, dtype=np.float32)
    dsm[8:12, 8:12] = 6.0
    return dsm


def _surface(**overrides: Any) -> SurfaceData:
    kw: dict[str, Any] = {"dsm": _dsm(), "pixel_size": 1.0, "svf": make_mock_svf(SHAPE)}
    kw.update(overrides)
    return SurfaceData(**kw)


@pytest.mark.parametrize(
    ("field", "overrides"),
    [
        ("cdsm", {"cdsm": np.zeros(BAD, dtype=np.float32)}),
        ("dem", {"dem": np.zeros(BAD, dtype=np.float32)}),
        ("wall_height", {"wall_height": np.zeros(BAD, dtype=np.float32)}),
        ("land_cover", {"land_cover": np.zeros(BAD, dtype=np.uint8)}),
        ("albedo", {"albedo": np.full(BAD, 0.2, dtype=np.float32)}),
        ("emissivity", {"emissivity": np.full(BAD, 0.95, dtype=np.float32)}),
        ("svf.svf", {"svf": make_mock_svf(BAD)}),
    ],
)
def test_mismatched_layer_raises_before_the_pipeline(field, overrides, tmp_path):
    surface = _surface(**overrides)
    with pytest.raises(GridShapeMismatch) as excinfo:
        calculate(surface, [WEATHER], LOCATION, output_dir=tmp_path, use_anisotropic_sky=False)
    assert excinfo.value.field == field
    assert excinfo.value.expected == str(SHAPE)
    assert excinfo.value.got == str(BAD)
    # Nothing was started: no output folder content was written.
    assert not any(tmp_path.iterdir())


def test_one_mismatched_directional_svf_is_caught(tmp_path):
    """Every SVF array is checked, not only the total."""
    svf = make_mock_svf(SHAPE)
    svf.svf_veg_west = np.ones(BAD, dtype=np.float32)
    surface = _surface(svf=svf)
    with pytest.raises(GridShapeMismatch) as excinfo:
        calculate(surface, [WEATHER], LOCATION, output_dir=tmp_path, use_anisotropic_sky=False)
    assert excinfo.value.field == "svf.svf_veg_west"


def test_matching_layers_still_run(tmp_path):
    summary = calculate(_surface(), [WEATHER], LOCATION, output_dir=tmp_path, use_anisotropic_sky=False)
    assert summary is not None
