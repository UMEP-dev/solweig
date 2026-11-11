"""Up-front estimate of the disk space taken by per-timestep GeoTIFFs."""

from __future__ import annotations

from collections import namedtuple
from datetime import datetime

import numpy as np
from conftest import make_mock_svf
from solweig import Location, SurfaceData, Weather, calculate
from solweig import timeseries as ts

Usage = namedtuple("Usage", "total used free")


def test_no_outputs_means_no_estimate(tmp_path, caplog):
    with caplog.at_level("INFO"):
        assert ts._report_output_disk_use(tmp_path, None, 8760, 5000, 5000) == 0
    assert "Per-timestep outputs" not in caplog.text


def test_estimate_is_logged_and_duplicates_count_once(tmp_path, caplog):
    with caplog.at_level("INFO"):
        estimate = ts._report_output_disk_use(tmp_path, ["tmrt", "shadow", "tmrt"], 24, 1000, 500)
    assert estimate == 2 * 24 * 1000 * 500 * 4
    assert "48 GeoTIFFs, about 91.6 MB" in caplog.text
    assert "free" not in caplog.text


def test_warns_when_the_estimate_exceeds_free_space(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr("shutil.disk_usage", lambda _p: Usage(10**12, 10**12 - 10**9, 10**9))
    with caplog.at_level("WARNING"):
        ts._report_output_disk_use(tmp_path, ["tmrt"], 8760, 5000, 5000)
    assert "815.8 GB" in caplog.text
    assert "953.7 MB free" in caplog.text


def test_estimate_matches_the_files_a_run_writes(tmp_path, caplog):
    """The 4 bytes per pixel figure is what the writer produces."""
    shape = (64, 80)
    surface = SurfaceData(dsm=np.zeros(shape, dtype=np.float32), pixel_size=1.0, svf=make_mock_svf(shape))
    weather = [
        Weather(datetime=datetime(2024, 7, 15, hour, 0), ta=25.0, rh=50.0, global_rad=500.0) for hour in (9, 10, 11)
    ]
    location = Location(latitude=57.7, longitude=12.0, utc_offset=1)
    outputs = ["tmrt", "shadow"]
    with caplog.at_level("INFO"):
        calculate(surface, weather, location, output_dir=tmp_path, outputs=outputs, use_anisotropic_sky=False)

    estimate = len(outputs) * len(weather) * shape[0] * shape[1] * 4
    on_disk = sum(f.stat().st_size for name in outputs for f in (tmp_path / name).glob("*.tif"))
    assert f"{len(outputs) * len(weather)} GeoTIFFs" in caplog.text
    # GeoTIFF headers add a few hundred bytes per file.
    assert estimate <= on_disk <= estimate * 1.10
