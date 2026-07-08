"""Byte-identity gate for the block-streaming preprocess path.

The relative->absolute height conversion in ``SurfaceData.preprocess`` is
numerically sensitive and is NOT exercised by the validation sites (they use
absolute DSMs). This test pins the streamed, block-in-place implementation to
the reference whole-array result across the cases that matter: relative DSM/
CDSM/TDSM, a DEM, sub-threshold flattening, canopy-below-DSM clearing, and
NaN handling — at a size large enough to span multiple row-blocks.
"""

from __future__ import annotations

import numpy as np
import pytest
from solweig.models.surface import SurfaceData


def _synthetic_relative_scene(rows: int, cols: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float32)
    # Sloped terrain so DEM + nDSM and base-relative thresholds are exercised.
    dem = (100.0 + 0.02 * yy + 0.01 * xx).astype(np.float32)

    # Relative building heights: mostly 0, plus buildings, sub-threshold
    # residuals, scattered noise, and NaN nodata.
    ndsm = np.zeros((rows, cols), dtype=np.float32)
    bmask = rng.random((rows, cols)) < 0.05
    ndsm[bmask] = rng.uniform(3.0, 30.0, (rows, cols)).astype(np.float32)[bmask]
    ndsm[10:40, 10:40] = 12.0
    ndsm[50:70, :] = 0.4  # sub-threshold residual (< 1.0 default min_object_height)
    ndsm[rng.random((rows, cols)) < 0.02] = np.nan

    # Relative canopy: scattered vegetation, plus canopy inside the building
    # block (falls below DSM -> cleared) and sub-threshold canopy.
    cdsm = np.full((rows, cols), np.nan, dtype=np.float32)
    veg = rng.random((rows, cols)) < 0.15
    cdsm[veg] = rng.uniform(0.2, 20.0, (rows, cols)).astype(np.float32)[veg]
    cdsm[80:120, 80:160] = 8.0
    cdsm[15:35, 15:35] = 2.0  # canopy inside the building block -> below DSM
    return dem, ndsm, cdsm


def _prepare(dem, ndsm, cdsm, *, force_block: int | None):
    surf = SurfaceData(
        dsm=ndsm.copy(),
        cdsm=cdsm.copy(),
        dem=dem.copy(),
        pixel_size=1.0,
        dsm_relative=True,
        cdsm_relative=True,
    )
    if force_block is not None:
        surf._preprocess_block_rows = force_block  # exercise multi-block streaming
    surf.preprocess()
    return surf


def _file_prepare_cleaned(tmp_path, dem, ndsm, cdsm):
    """File-mode prepare of a relative-height scene (DEM at half resolution so
    the resample path is exercised); returns the cleaned layers from the .npy
    sidecars."""
    import pyproj
    from solweig import io

    src = tmp_path / "src"
    src.mkdir(parents=True)
    rows = ndsm.shape[0]
    gt = [300000.0, 1.0, 0.0, 6_400_000.0 + rows, 0.0, -1.0]
    crs = pyproj.CRS.from_epsg(3006).to_wkt()  # projected, metres
    io.save_raster(str(src / "dsm.tif"), ndsm, gt, crs)
    io.save_raster(str(src / "cdsm.tif"), cdsm, gt, crs)
    # DEM at 2 m so prepare must resample it to the 1 m target grid
    dem_2m = dem[::2, ::2].copy()
    gt_dem = [300000.0, 2.0, 0.0, 6_400_000.0 + rows, 0.0, -2.0]
    io.save_raster(str(src / "dem.tif"), dem_2m, gt_dem, crs)

    SurfaceData.prepare(
        dsm=str(src / "dsm.tif"),
        dem=str(src / "dem.tif"),
        cdsm=str(src / "cdsm.tif"),
        working_dir=str(tmp_path / "work"),
        pixel_size=1.0,
        dsm_relative=True,
        cdsm_relative=True,
    )
    cleaned = tmp_path / "work" / "cleaned"
    return {n: np.array(np.load(cleaned / f"{n}.npy")) for n in ("dsm", "cdsm", "dem", "tdsm")}


def test_large_sequential_path_matches_small(tmp_path, monkeypatch):
    """The large-raster path (layer-sequential spill-to-memmap load/align +
    in-place streamed fill_nan/preprocess + plain-GeoTIFF save) produces
    byte-identical cleaned layers to the default whole-array path."""
    import solweig.models.surface as surf_mod

    dem, ndsm, cdsm = _synthetic_relative_scene(180, 260, seed=3)

    small = _file_prepare_cleaned(tmp_path / "a", dem, ndsm, cdsm)
    monkeypatch.setattr(surf_mod, "_PREPROCESS_STREAM_MIN_PIXELS", 0)  # force large path
    large = _file_prepare_cleaned(tmp_path / "b", dem, ndsm, cdsm)

    for name in ("dsm", "cdsm", "dem", "tdsm"):
        a, b = small[name], large[name]
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b), err_msg=f"{name} NaN mask differs")
        np.testing.assert_array_equal(a[~np.isnan(a)], b[~np.isnan(b)], err_msg=f"{name} values differ")


@pytest.mark.parametrize("shape", [(200, 300), (137, 251)])
def test_streamed_preprocess_matches_wholearray(shape):
    rows, cols = shape
    dem, ndsm, cdsm = _synthetic_relative_scene(rows, cols)

    ref = _prepare(dem, ndsm, cdsm, force_block=None)  # whole-array (one block)
    streamed = _prepare(dem, ndsm, cdsm, force_block=32)  # many small blocks

    for name in ("dsm", "cdsm", "tdsm", "dem"):
        a = np.asarray(getattr(ref, name))
        b = np.asarray(getattr(streamed, name))
        # Bit-exact, NaN in the same places.
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b), err_msg=f"{name} NaN mask differs")
        np.testing.assert_array_equal(a[~np.isnan(a)], b[~np.isnan(b)], err_msg=f"{name} values differ")
