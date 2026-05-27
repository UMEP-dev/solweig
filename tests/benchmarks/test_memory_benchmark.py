"""Memory benchmark tests for CI regression detection.

These tests verify that memory usage stays within expected bounds.
They run on small grids to be fast in CI while still detecting regressions.

Memory target: ~377 bytes/pixel (re-measured 2026-05-27 — Feb 2026
baseline was ~370; the b85 architecture-stabilisation refactors moved
the number by under 2%, confirming the typed views / FFI bundling
don't add measurable memory overhead).

Regression threshold: 500 bytes/pixel (~32% headroom for CI variance).
Local runs are deterministic across repeated invocations (5/5 identical
in re-measurement); the headroom exists for OS/Python-version drift
in CI, not local variance.
"""

import tracemalloc
from datetime import datetime

import numpy as np
import pytest
from solweig import Location, SurfaceData, Weather, calculate

pytestmark = pytest.mark.slow


class TestMemoryBenchmark:
    """Memory usage benchmarks for CI."""

    # Target: ~377 bytes/pixel (2026-05-27 re-measurement, stable
    # across 5 repeated runs). Threshold leaves room for CI/OS drift.
    MAX_BYTES_PER_PIXEL = 500

    @pytest.fixture
    def benchmark_surface(self):
        """Create a 150x150 benchmark surface.

        Small enough to be fast in CI, large enough to amortize fixed overhead.
        """
        size = 150
        np.random.seed(42)

        dsm = np.ones((size, size), dtype=np.float32) * 10.0

        # Add a few low buildings (5m above ground to keep buffer small)
        for _ in range(5):
            x, y = np.random.randint(15, size - 15, 2)
            w, h = np.random.randint(5, 10, 2)
            dsm[y : y + h, x : x + w] = 15.0  # 5m above ground

        land_cover = np.ones((size, size), dtype=np.int32) * 5
        land_cover[dsm > 12] = 2

        from conftest import make_mock_svf

        return SurfaceData(
            dsm=dsm,
            land_cover=land_cover,
            pixel_size=1.0,
            svf=make_mock_svf((size, size)),
        )

    @pytest.fixture
    def benchmark_location(self):
        """Athens, Greece - good sun angle for testing."""
        return Location(latitude=37.98, longitude=23.73, utc_offset=2)

    @pytest.fixture
    def benchmark_weather(self):
        """Summer noon conditions."""
        return Weather(
            datetime=datetime(2024, 7, 21, 12, 0),
            ta=30.0,
            rh=50.0,
            global_rad=800.0,
            ws=2.0,
        )

    def test_memory_per_pixel_within_threshold(
        self, benchmark_surface, benchmark_location, benchmark_weather, tmp_path
    ):
        """Verify memory usage stays within acceptable bounds.

        This test catches memory regressions (e.g., accidental float64 usage,
        leaked allocations, or inefficient intermediate arrays).
        """
        from conftest import read_timestep_geotiff

        # Preprocess surface first (one-time cost not counted in per-timestep)
        benchmark_surface.preprocess()

        # Warmup run: triggers all lazy imports (matplotlib, rasterio, etc.)
        # so tracemalloc only measures actual computation memory, not import overhead.
        warmup_dir = tmp_path / "warmup"
        calculate(
            benchmark_surface,
            [benchmark_weather],
            benchmark_location,
            output_dir=warmup_dir,
            outputs=["tmrt"],
        )

        output_dir = tmp_path / "memory"

        # Start memory tracing after warmup
        tracemalloc.start()
        tracemalloc.reset_peak()

        # Run calculation
        calculate(
            benchmark_surface,
            [benchmark_weather],
            benchmark_location,
            output_dir=output_dir,
            outputs=["tmrt"],
        )

        # Get peak memory
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Calculate bytes per pixel
        n_pixels = benchmark_surface.shape[0] * benchmark_surface.shape[1]
        bytes_per_pixel = peak / n_pixels

        # Verify result is valid (sanity check)
        tmrt = read_timestep_geotiff(output_dir, "tmrt", 0)
        assert tmrt is not None
        assert np.isfinite(tmrt).sum() > 0.8 * n_pixels

        # Verify memory within threshold
        assert bytes_per_pixel < self.MAX_BYTES_PER_PIXEL, (
            f"Memory regression detected: {bytes_per_pixel:.1f} bytes/pixel "
            f"exceeds threshold of {self.MAX_BYTES_PER_PIXEL} bytes/pixel. "
            f"Peak memory: {peak / 1024 / 1024:.1f} MB for {n_pixels:,} pixels."
        )

    def test_float32_arrays_used(self, benchmark_surface, benchmark_location, benchmark_weather, tmp_path):
        """Verify output arrays use float32 (not float64)."""
        from conftest import read_timestep_geotiff

        benchmark_surface.preprocess()

        output_dir = tmp_path / "float32"
        calculate(
            benchmark_surface,
            [benchmark_weather],
            benchmark_location,
            output_dir=output_dir,
            outputs=["tmrt", "shadow", "kdown", "kup", "ldown", "lup"],
        )

        # All output arrays should be float32
        tmrt = read_timestep_geotiff(output_dir, "tmrt", 0)
        assert tmrt.dtype == np.float32, f"tmrt dtype is {tmrt.dtype}, expected float32"
        for field in ["shadow", "kdown", "kup", "ldown", "lup"]:
            try:
                arr = read_timestep_geotiff(output_dir, field, 0)
                assert arr.dtype == np.float32, f"{field} dtype is {arr.dtype}"
            except FileNotFoundError:
                pass  # Field not written (e.g., nighttime shadow)

    def test_surface_arrays_float32(self):
        """Verify surface data arrays use float32."""
        size = 100
        np.random.seed(42)

        dsm = np.ones((size, size), dtype=np.float32) * 10.0
        cdsm = np.ones((size, size), dtype=np.float32) * 5.0

        surface = SurfaceData(dsm=dsm, cdsm=cdsm, pixel_size=1.0)
        surface.preprocess()

        assert surface.dsm.dtype == np.float32
        if surface.cdsm is not None:
            assert surface.cdsm.dtype == np.float32
        if surface.dem is not None:
            assert surface.dem.dtype == np.float32
