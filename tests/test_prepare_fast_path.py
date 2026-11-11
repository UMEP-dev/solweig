"""A repeated ``SurfaceData.prepare()`` on unchanged inputs must take the fast path.

Up to 0.1.0b102 it never did unless the caller passed ``pixel_size``: the
fingerprint saved at the end of a build recorded the resolved pixel size
(1.0) while the check on the next call compared it with the requested one
(None), so every call rebuilt from the source rasters. The existing plugin
test only checked that a second run was faster than the first, which the
wall and SVF caches satisfy on their own.
"""

from __future__ import annotations

import numpy as np
import pytest
from solweig import SurfaceData

from tests.raster_helpers import write_geotiff

HIT = "Fast-path cache hit"
MISS = "Fast-path cache invalidated"


@pytest.fixture()
def dsm_path(tmp_path):
    rows, cols = 48, 64
    dsm = (np.linspace(0.0, 2.0, rows)[:, None] + np.zeros((1, cols))).astype(np.float32)
    dsm[10:22, 20:36] += 12.0
    path = tmp_path / "dsm.tif"
    write_geotiff(path, dsm, origin_x=319000.0, origin_y_top=6400000.0 + rows, pixel_size=1.0, epsg=3006)
    return path


@pytest.mark.parametrize("pixel_size", [None, 1.0], ids=["pixel size from the DSM", "pixel size given"])
def test_second_prepare_takes_the_fast_path(dsm_path, tmp_path, caplog, pixel_size):
    work = tmp_path / "work"
    first = SurfaceData.prepare(dsm=str(dsm_path), working_dir=str(work), pixel_size=pixel_size)

    caplog.clear()
    with caplog.at_level("INFO"):
        second = SurfaceData.prepare(dsm=str(dsm_path), working_dir=str(work), pixel_size=pixel_size)

    assert MISS not in caplog.text, caplog.text
    assert HIT in caplog.text
    np.testing.assert_array_equal(second.dsm, first.dsm)
    assert first.wall_height is not None and second.wall_height is not None
    np.testing.assert_array_equal(second.wall_height, first.wall_height)
    assert first.svf is not None and second.svf is not None
    np.testing.assert_array_equal(second.svf.svf, first.svf.svf)


def test_a_changed_setting_still_invalidates(dsm_path, tmp_path, caplog):
    work = tmp_path / "work"
    SurfaceData.prepare(dsm=str(dsm_path), working_dir=str(work))
    caplog.clear()
    with caplog.at_level("INFO"):
        SurfaceData.prepare(dsm=str(dsm_path), working_dir=str(work), min_object_height=2.5)
    assert MISS in caplog.text
    assert "min_object_height" in caplog.text
