"""Empty-tile compute skip.

On an irregular raster (e.g. the Madrid extent), the geometric tiler emits a
full grid of tiles regardless of coverage, so some tiles fall entirely outside
the valid data. Those tiles produce only NaN outputs, yet the pipeline used to
run the whole shadow/GVF/aniso/radiation compute on them anyway.

``calculate_core_fused`` now short-circuits a tile whose valid mask is entirely
zero: it returns NaN directly without the Rust FFI call. The only output that
changes versus the old behaviour is the per-timestep shadow raster, which
becomes NaN over nodata rather than a spurious "fully sunlit" 1.0 (the raw
shadow cast on a NaN-filled flat DSM). Tmrt/UTCI were already NaN there, and the
summary grids are unchanged because the accumulators gate on finite Tmrt.

These tests lock in both halves:
  * the heavy Rust compute is never invoked on a fully-nodata tile (regression),
  * the tiled end-to-end output over nodata is NaN, including the shadow layer.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
from conftest import make_mock_svf, read_timestep_geotiff
from solweig import Location, SurfaceData, Weather
from solweig.api import _calculate_single
from solweig.timeseries import _calculate_timeseries

_LOCATION = Location(latitude=40.4, longitude=-3.7, utc_offset=1)
_WEATHER_DAY = Weather(datetime=datetime(2024, 7, 15, 13, 0), ta=30.0, rh=40.0, global_rad=800.0, ws=2.0)


def test_all_nodata_tile_skips_rust_compute(monkeypatch):
    """A tile with an all-zero valid mask returns NaN without calling the Rust pipeline."""
    import solweig.rustalgos as rustalgos

    size = 64
    dsm = np.full((size, size), 10.0, dtype=np.float32)
    surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf((size, size)))
    # Mark the entire tile as nodata (mirrors a fully-clipped tile). The DSM stays
    # finite so max_height etc. behave; the valid mask is what drives the skip.
    surface._valid_mask = np.zeros((size, size), dtype=bool)

    compute_calls: list[int] = []
    real_compute = rustalgos.pipeline.compute_timestep

    def _spy(*args, **kwargs):
        compute_calls.append(1)
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(rustalgos.pipeline, "compute_timestep", _spy)

    result = _calculate_single(surface=surface, location=_LOCATION, weather=_WEATHER_DAY)

    assert compute_calls == [], "compute_timestep must not run on a fully-nodata tile"
    assert np.isnan(result.tmrt).all(), "Tmrt must be all-NaN over a nodata tile"
    assert result.shadow is not None and np.isnan(result.shadow).all(), (
        "shadow must be NaN over a nodata tile, not a spurious sunlit 1.0"
    )


@pytest.mark.slow
def test_tiled_run_skips_empty_tiles_and_writes_nan(monkeypatch, tmp_path):
    """End-to-end tiled run: empty tiles are skipped and their region is NaN.

    A 576x576 surface has valid data filling only the top-left tile; every other
    tile is entirely nodata. The Rust compute must run fewer times than the number
    of tile/timestep invocations (proving empty tiles were skipped), the nodata
    region must be NaN in both Tmrt and shadow, and the valid tile must compute.
    """
    import solweig.api as api_mod
    import solweig.rustalgos as rustalgos

    size = 576
    tile = 256
    dsm = np.full((size, size), np.nan, dtype=np.float32)
    # Fill exactly the top-left tile with flat, valid data. A constant height gives
    # max_height 0 -> zero overlap buffer, so this tile is fully valid (not cropped)
    # and computes without hitting the separate isotropic-crop contiguity path.
    # Every other tile is entirely nodata and must be skipped.
    dsm[0:tile, 0:tile] = 10.0
    surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf((size, size)))

    single_calls: list[int] = []
    compute_calls: list[int] = []
    real_single = api_mod._calculate_single
    real_compute = rustalgos.pipeline.compute_timestep

    def _spy_single(**kwargs):
        single_calls.append(1)
        return real_single(**kwargs)

    def _spy_compute(*args, **kwargs):
        compute_calls.append(1)
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(api_mod, "_calculate_single", _spy_single)
    monkeypatch.setattr(rustalgos.pipeline, "compute_timestep", _spy_compute)

    _calculate_timeseries(
        surface=surface,
        weather_series=[_WEATHER_DAY],
        location=_LOCATION,
        output_dir=tmp_path,
        outputs=["tmrt", "shadow"],
        tile_size=tile,
    )

    # Every tile/timestep enters _calculate_single, but only non-empty tiles reach
    # the Rust compute. With valid data confined to one corner, most tiles are skipped.
    assert len(single_calls) > 1, "expected a multi-tile run"
    assert len(compute_calls) > 0, "the valid corner tile must actually compute"
    assert len(compute_calls) < len(single_calls), "at least one empty tile must skip the Rust compute"

    tmrt = read_timestep_geotiff(tmp_path, "tmrt", 0)
    shadow = read_timestep_geotiff(tmp_path, "shadow", 0)
    assert tmrt is not None and shadow is not None

    # Deep inside the nodata region: both layers NaN (shadow NaN is the behaviour change).
    assert np.isnan(tmrt[450:560, 450:560]).all()
    assert np.isnan(shadow[450:560, 450:560]).all()

    # The valid corner actually produced finite Tmrt.
    assert np.isfinite(tmrt[10:80, 10:80]).any()
