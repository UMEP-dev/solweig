"""Smoke tests for `pipeline.SvfBundle` and `pipeline.StateBundle`.

Cover the FFI argument-bundling refactor: 17 SVF + 9 state args
collapsed into two PyO3 classes. These tests verify each bundle is
constructable, that the version check on StateBundle fires correctly,
and that calculate() survives a round-trip.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
from solweig.api import Location, SurfaceData, Weather, calculate
from solweig.rustalgos import pipeline


def _zero(shape):
    return np.zeros(shape, dtype=np.float32)


def _full(shape, value):
    return np.full(shape, value, dtype=np.float32)


def test_svf_bundle_constructs_from_17_arrays():
    shape = (10, 10)
    bundle = pipeline.SvfBundle(
        _full(shape, 1.0),  # svf
        _full(shape, 0.9),  # svf_n
        _full(shape, 0.9),  # svf_e
        _full(shape, 0.9),  # svf_s
        _full(shape, 0.9),  # svf_w
        _full(shape, 1.0),  # svf_veg
        _full(shape, 0.9),  # svf_veg_n
        _full(shape, 0.9),  # svf_veg_e
        _full(shape, 0.9),  # svf_veg_s
        _full(shape, 0.9),  # svf_veg_w
        _full(shape, 1.0),  # svf_aveg
        _full(shape, 0.9),  # svf_aveg_n
        _full(shape, 0.9),  # svf_aveg_e
        _full(shape, 0.9),  # svf_aveg_s
        _full(shape, 0.9),  # svf_aveg_w
        _full(shape, 1.0),  # svfbuveg
        _zero(shape),  # svfalfa
    )
    # The bundle should construct without error and be a valid Python object.
    assert bundle is not None


def test_svf_bundle_rejects_wrong_argument_count():
    """Missing arguments must produce a TypeError, not a silent default."""
    shape = (5, 5)
    arr = _full(shape, 1.0)
    with pytest.raises(TypeError):
        # Only 5 args, not 17.
        pipeline.SvfBundle(arr, arr, arr, arr, arr)


# ── StateBundle ─────────────────────────────────────────────────────────────


def test_state_bundle_constructs_at_current_version():
    shape = (5, 5)
    bundle = pipeline.StateBundle(
        pipeline.STATE_BUNDLE_VERSION,
        0,  # firstdaytime
        0.0,  # timeadd
        1.0 / 24.0,  # timestep_dec (1 hour)
        _zero(shape),  # tgmap1
        _zero(shape),  # tgmap1_e
        _zero(shape),  # tgmap1_s
        _zero(shape),  # tgmap1_w
        _zero(shape),  # tgmap1_n
        _zero(shape),  # tgout1
    )
    assert bundle.version == pipeline.STATE_BUNDLE_VERSION


def test_state_bundle_rejects_version_mismatch():
    """A version mismatch must raise ValueError, not silently mis-map fields."""
    shape = (5, 5)
    wrong = pipeline.STATE_BUNDLE_VERSION + 99
    with pytest.raises(ValueError, match="StateBundle version mismatch"):
        pipeline.StateBundle(
            wrong,
            0,
            0.0,
            1.0 / 24.0,
            _zero(shape),
            _zero(shape),
            _zero(shape),
            _zero(shape),
            _zero(shape),
            _zero(shape),
        )


# ── End-to-end ──────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_calculate_uses_svf_bundle_end_to_end(tmp_path):
    """A full calculate() call exercises the Python-side bundle construction
    AND the Rust-side bundle unpacking. If either is broken the test fails
    loudly; if both are correct golden tests catch numerical drift."""
    from conftest import make_mock_svf

    dsm = np.ones((20, 20), dtype=np.float32) * 2.0
    surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf(dsm.shape))
    location = Location(latitude=57.7, longitude=12.0, utc_offset=1)
    weather = Weather(datetime=datetime(2024, 7, 15, 12, 0), ta=25.0, rh=50.0, global_rad=800.0)

    calculate(surface, [weather], location, output_dir=tmp_path, outputs=["tmrt"])

    from conftest import read_timestep_geotiff

    tmrt = read_timestep_geotiff(tmp_path, "tmrt", 0)
    assert np.isfinite(tmrt).any()
