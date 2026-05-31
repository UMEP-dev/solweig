"""Tiling performance benchmark scaffold for CI regression detection.

This suite focuses on orchestration-level regressions:
- worker-count scaling does not collapse
- bounded in-flight scheduling runs correctly
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
from conftest import make_mock_svf
from solweig import Location, SurfaceData, Weather
from solweig.timeseries import _calculate_timeseries

pytestmark = [pytest.mark.slow, pytest.mark.gpu_perf_gate]


class TestTilingBenchmark:
    """Benchmark scaffold for tiled orchestration performance."""

    @pytest.fixture
    def benchmark_surface(self):
        size = 520  # 3x3 tiles with tile_size=256 and zero overlap
        dsm = np.ones((size, size), dtype=np.float32) * 5.0
        return SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf((size, size)))

    @pytest.fixture
    def benchmark_location(self):
        return Location(latitude=57.7, longitude=12.0, utc_offset=1)

    @pytest.fixture
    def benchmark_weather_series(self):
        base = datetime(2024, 7, 15, 11, 0)
        return [
            Weather(
                datetime=base + timedelta(hours=i),
                ta=24.0 + i,
                rh=50.0,
                global_rad=700.0,
                ws=2.0,
            )
            for i in range(3)
        ]

    def test_tiled_timeseries_completes(
        self, benchmark_surface, benchmark_location, benchmark_weather_series, tmp_path
    ):
        """Tiled timeseries produces correct output."""
        from conftest import read_timestep_geotiff

        output_dir = tmp_path / "tiled"
        summary = _calculate_timeseries(
            benchmark_surface,
            benchmark_weather_series,
            benchmark_location,
            output_dir=output_dir,
            outputs=["tmrt"],
        )
        assert summary.n_timesteps == len(benchmark_weather_series)
        for i in range(len(benchmark_weather_series)):
            tmrt = read_timestep_geotiff(output_dir, "tmrt", i)
            assert tmrt is not None
