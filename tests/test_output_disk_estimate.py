"""Disk space of per-timestep GeoTIFFs: compression, the up-front estimate, and the refusal to start."""

from __future__ import annotations

from collections import namedtuple
from datetime import datetime

import numpy as np
import pytest
import rasterio
from conftest import make_mock_svf
from solweig import Location, SurfaceData, Weather, calculate, io
from solweig import timeseries as ts
from solweig.errors import ConfigurationError

Usage = namedtuple("Usage", "total used free")
LOCATION = Location(latitude=57.7, longitude=12.0, utc_offset=1)


def _weather(hours=(9, 10, 11)):
    return [Weather(datetime=datetime(2024, 7, 15, h, 0), ta=25.0, rh=50.0, global_rad=500.0) for h in hours]


def _surface(shape=(64, 80)):
    dsm = np.zeros(shape, dtype=np.float32)
    dsm[20:30, 20:30] = 10.0
    return SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf(shape))


# ── estimate ───────────────────────────────────────────────────────────────


def test_estimate_counts_each_output_once_and_shadow_at_its_own_rate(tmp_path, caplog):
    with caplog.at_level("INFO"):
        estimate = ts._check_output_disk_space(tmp_path, ["tmrt", "shadow", "tmrt"], 24, 1000, 500)
    pixels = 1000 * 500
    per_step = ts._OUTPUT_BYTES_PER_PIXEL + ts._SHADOW_BYTES_PER_PIXEL
    assert estimate == int(per_step * pixels * 24) + ts._SUMMARY_BYTES_PER_PIXEL * pixels
    assert "48 GeoTIFFs, up to about 57.2 MB" in caplog.text


def test_no_outputs_still_checks_the_summary_grids(tmp_path, caplog, monkeypatch):
    with caplog.at_level("INFO"):
        assert ts._check_output_disk_space(tmp_path, None, 8760, 1000, 1000) == 60 * 1000 * 1000
    assert "Per-timestep outputs" not in caplog.text

    monkeypatch.setattr("shutil.disk_usage", lambda _p: Usage(10**12, 10**12 - 10**6, 10**6))
    with pytest.raises(ConfigurationError) as excinfo:
        ts._check_output_disk_space(tmp_path, None, 8760, 1000, 1000)
    assert excinfo.value.parameter == "output_dir"


def test_estimate_is_an_upper_bound_for_the_files_a_run_writes(tmp_path):
    surface = _surface()
    outputs = ["tmrt", "shadow", "kdown", "kup", "ldown", "lup"]
    weather = _weather()
    calculate(surface, weather, LOCATION, output_dir=tmp_path, outputs=outputs, use_anisotropic_sky=False)

    pixels = surface.dsm.size
    for name in outputs:
        sizes = [f.stat().st_size for f in (tmp_path / name).glob("*.tif")]
        assert len(sizes) == len(weather), name
        rate = ts._SHADOW_BYTES_PER_PIXEL if name == "shadow" else ts._OUTPUT_BYTES_PER_PIXEL
        assert max(sizes) <= rate * pixels, f"{name}: {max(sizes) / pixels:.2f} B/px exceeds the estimate {rate}"


# ── refusal ────────────────────────────────────────────────────────────────


def test_run_is_refused_before_anything_is_written_when_the_disk_is_too_small(tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setattr("shutil.disk_usage", lambda _p: Usage(10**12, 10**12 - 10**5, 10**5))
    with pytest.raises(ConfigurationError) as excinfo:
        calculate(_surface(), _weather(), LOCATION, output_dir=out, outputs=["tmrt"], use_anisotropic_sky=False)
    assert excinfo.value.parameter == "outputs"
    message = str(excinfo.value)
    assert "97.7 KB free" in message
    assert "3 timesteps" in message
    assert not any(p.is_file() for p in out.rglob("*"))


def test_run_proceeds_when_the_estimate_fits(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.disk_usage", lambda _p: Usage(10**12, 0, 10**12))
    calculate(_surface(), _weather(), LOCATION, output_dir=tmp_path, outputs=["tmrt"], use_anisotropic_sky=False)
    assert len(list((tmp_path / "tmrt").glob("*.tif"))) == 3


# ── compression ────────────────────────────────────────────────────────────


def test_per_timestep_outputs_are_compressed_and_lossless(tmp_path):
    """Compressed files hold exactly the values an uncompressed write holds."""
    rows, cols = 300, 700  # not multiples of the 256 px block
    gt = [0.0, 1.0, 0.0, float(rows), 0.0, -1.0]
    # A field with flat regions and gradients, like a Tmrt raster. (White noise
    # does not compress: such a file comes out slightly larger than plain.)
    yy, xx = np.mgrid[0:rows, 0:cols]
    data = (30.0 + 20.0 * ((xx // 40 + yy // 40) % 2) + 0.01 * xx).astype(np.float32)
    data[5:9, 7:11] = np.nan
    halves = [(slice(0, 130), slice(0, cols)), (slice(130, rows), slice(0, cols))]  # crosses blocks

    paths = {}
    for compress in (True, False):
        path = tmp_path / f"compress_{compress}.tif"
        io.create_empty_raster(path, rows, cols, gt, "", dtype=np.float32, nodata=np.nan, compress=compress)
        for window in halves:
            io.write_raster_window(path, data[window], window=window)
        paths[compress] = path

    with rasterio.open(paths[True]) as packed, rasterio.open(paths[False]) as plain:
        assert packed.profile["compress"] == "deflate"
        assert packed.profile["tiled"] is True
        np.testing.assert_array_equal(packed.read(1), plain.read(1))
        np.testing.assert_array_equal(packed.read(1), data)
    assert paths[True].stat().st_size < 0.9 * paths[False].stat().st_size


@pytest.mark.parametrize(
    ("tile_size", "expected"),
    [
        (None, 256),  # untiled run
        (256, 256),
        (512, 512),
        (1024, 512),
        (384, 128),
        (304, 304),  # multiple of 16, used as the block itself
        (300, None),  # cannot align: written uncompressed
        (1500, None),
        (5771, 256),  # large tile: few blocks straddle an edge
    ],
)
def test_block_size_follows_the_tile_size(tile_size, expected):
    from solweig.output_async import output_block_size

    assert output_block_size(tile_size) == expected


@pytest.mark.parametrize("tile_size", [256, 304, 300])
def test_tiled_run_output_is_no_larger_than_uncompressed(tmp_path, tile_size):
    """Tile-by-tile writes must not leave the file bigger than the plain layout."""
    from solweig.output_async import output_block_size

    rows = cols = 912
    gt = [0.0, 1.0, 0.0, float(rows), 0.0, -1.0]
    yy, xx = np.mgrid[0:rows, 0:cols]
    data = (30.0 + 20.0 * ((xx // 40 + yy // 40) % 2) + 0.01 * xx).astype(np.float32)
    block = output_block_size(tile_size)
    path = tmp_path / "tiled.tif"
    io.create_empty_raster(path, rows, cols, gt, "", nodata=np.nan, compress=block is not None, block_size=block)
    for r in range(0, rows, tile_size):
        for c in range(0, cols, tile_size):
            window = (slice(r, min(r + tile_size, rows)), slice(c, min(c + tile_size, cols)))
            io.write_raster_window(path, data[window], window=window)

    with rasterio.open(path) as src:
        np.testing.assert_array_equal(src.read(1), data)
    assert path.stat().st_size <= 4.05 * data.size
    if block is not None:
        assert path.stat().st_size < 0.5 * 4 * data.size  # aligned blocks: no dead space


def test_unwritten_part_of_a_compressed_output_reads_as_nodata(tmp_path):
    """The file is created sparse; a tile that was never written must read as NaN."""
    rows, cols = 600, 600
    path = tmp_path / "partial.tif"
    io.create_empty_raster(path, rows, cols, [0.0, 1.0, 0.0, 600.0, 0.0, -1.0], "", nodata=np.nan)
    with rasterio.open(path) as src:
        assert np.isnan(src.read(1)).all()

    io.write_raster_window(path, np.full((100, 100), 7.0, dtype=np.float32), window=(slice(50, 150), slice(300, 400)))
    with rasterio.open(path) as src:
        got = src.read(1)
    assert np.all(got[50:150, 300:400] == 7.0)
    got[50:150, 300:400] = np.nan
    assert np.isnan(got).all()
