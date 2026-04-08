"""
Unit tests for the orchestration layer.

Tests internal helper functions in computation.py, timeseries.py, and tiling.py
that aren't exercised by the higher-level integration tests.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
import solweig
from solweig.models.state import ThermalState, TileSpec
from solweig.tiling import (
    MAX_BUFFER_M,
    MIN_TILE_SIZE,
    calculate_buffer_distance,
    compute_max_tile_pixels,
    compute_max_tile_side,
    generate_tiles,
    validate_tile_size,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_shape():
    return (10, 10)


@pytest.fixture()
def state_10x10(small_shape):
    """Fresh ThermalState for a 10×10 grid."""
    return ThermalState.initial(small_shape)


# ---------------------------------------------------------------------------
# _precompute_weather
# ---------------------------------------------------------------------------


class TestPrecomputeWeather:
    """Tests for _precompute_weather() in timeseries.py."""

    def test_empty_list_noop(self):
        """Empty weather series is a no-op."""
        from solweig.timeseries import _precompute_weather

        location = solweig.Location(latitude=57.7, longitude=12.0, utc_offset=2)
        _precompute_weather([], location)  # Should not raise

    def test_assigns_precomputed_altmax(self):
        """Each weather object gets precomputed_altmax assigned."""
        from solweig.timeseries import _precompute_weather

        location = solweig.Location(latitude=57.7, longitude=12.0, utc_offset=2)
        dt_noon = datetime(2024, 7, 15, 12, 0)
        w = solweig.Weather(datetime=dt_noon, ta=25.0, rh=50.0, global_rad=800.0)

        _precompute_weather([w], location)

        assert hasattr(w, "precomputed_altmax")
        assert w.precomputed_altmax is not None
        assert w.precomputed_altmax > 0  # Summer noon in Sweden: ~55°

    def test_same_day_shares_altmax(self):
        """Multiple timesteps on the same day share the same altmax value."""
        from solweig.timeseries import _precompute_weather

        location = solweig.Location(latitude=57.7, longitude=12.0, utc_offset=2)
        base = datetime(2024, 7, 15)

        weather_series = [
            solweig.Weather(datetime=base.replace(hour=h), ta=25.0, rh=50.0, global_rad=800.0) for h in range(6, 18)
        ]

        _precompute_weather(weather_series, location)

        altmaxes = [w.precomputed_altmax for w in weather_series]
        assert all(a == altmaxes[0] for a in altmaxes)

    def test_different_days_may_differ(self):
        """Different days may have different altmax (season effect)."""
        from solweig.timeseries import _precompute_weather

        location = solweig.Location(latitude=57.7, longitude=12.0, utc_offset=2)

        w_summer = solweig.Weather(datetime=datetime(2024, 6, 21, 12, 0), ta=25.0, rh=50.0, global_rad=800.0)
        w_winter = solweig.Weather(datetime=datetime(2024, 12, 21, 12, 0), ta=0.0, rh=70.0, global_rad=200.0)

        _precompute_weather([w_summer, w_winter], location)

        assert w_summer.precomputed_altmax is not None
        assert w_winter.precomputed_altmax is not None
        assert w_summer.precomputed_altmax > w_winter.precomputed_altmax

    def test_marks_derived_computed(self):
        """After precomputation, weather objects have derived values computed."""
        from solweig.timeseries import _precompute_weather

        location = solweig.Location(latitude=57.7, longitude=12.0, utc_offset=2)
        w = solweig.Weather(datetime=datetime(2024, 7, 15, 12, 0), ta=25.0, rh=50.0, global_rad=800.0)

        assert not w._derived_computed
        _precompute_weather([w], location)
        assert w._derived_computed


# ---------------------------------------------------------------------------
# ThermalState
# ---------------------------------------------------------------------------


class TestThermalState:
    """Tests for ThermalState creation and copying."""

    def test_initial_state_shape(self, small_shape):
        """Initial state arrays have the requested shape."""
        state = ThermalState.initial(small_shape)
        assert state.tgmap1.shape == small_shape
        assert state.tgmap1_e.shape == small_shape
        assert state.tgmap1_s.shape == small_shape
        assert state.tgmap1_w.shape == small_shape
        assert state.tgmap1_n.shape == small_shape
        assert state.tgout1.shape == small_shape

    def test_initial_state_zeros(self, small_shape):
        """Initial state arrays are all zeros."""
        state = ThermalState.initial(small_shape)
        np.testing.assert_array_equal(state.tgmap1, 0.0)
        np.testing.assert_array_equal(state.tgout1, 0.0)

    def test_initial_state_flags(self, small_shape):
        """Initial state has firstdaytime=1.0, timeadd=0.0."""
        state = ThermalState.initial(small_shape)
        assert state.firstdaytime == 1.0
        assert state.timeadd == 0.0
        assert state.timestep_dec == 0.0

    def test_copy_is_independent(self, small_shape):
        """Copy creates independent arrays."""
        state = ThermalState.initial(small_shape)
        state.tgmap1[:] = 5.0
        state.firstdaytime = 0.0

        copy = state.copy()
        copy.tgmap1[:] = 99.0
        copy.firstdaytime = 1.0

        assert state.tgmap1[0, 0] == 5.0  # Original unchanged
        assert state.firstdaytime == 0.0  # Original unchanged

    def test_copy_preserves_values(self, small_shape):
        """Copy preserves all field values."""
        state = ThermalState.initial(small_shape)
        state.tgmap1[:] = 3.0
        state.timeadd = 1.5
        state.timestep_dec = 0.042

        copy = state.copy()
        np.testing.assert_array_equal(copy.tgmap1, 3.0)
        assert copy.timeadd == 1.5
        assert copy.timestep_dec == 0.042


# ---------------------------------------------------------------------------
# TileSpec
# ---------------------------------------------------------------------------


class TestTileSpec:
    """Tests for TileSpec properties."""

    def test_core_shape(self):
        tile = TileSpec(
            row_start=0,
            row_end=100,
            col_start=0,
            col_end=200,
            row_start_full=0,
            row_end_full=150,
            col_start_full=0,
            col_end_full=250,
            overlap_top=0,
            overlap_bottom=50,
            overlap_left=0,
            overlap_right=50,
        )
        assert tile.core_shape == (100, 200)

    def test_full_shape(self):
        tile = TileSpec(
            row_start=0,
            row_end=100,
            col_start=0,
            col_end=200,
            row_start_full=0,
            row_end_full=150,
            col_start_full=0,
            col_end_full=250,
            overlap_top=0,
            overlap_bottom=50,
            overlap_left=0,
            overlap_right=50,
        )
        assert tile.full_shape == (150, 250)

    def test_core_slice_no_overlap(self):
        """First tile (no top/left overlap) has core starting at (0, 0)."""
        tile = TileSpec(
            row_start=0,
            row_end=100,
            col_start=0,
            col_end=100,
            row_start_full=0,
            row_end_full=150,
            col_start_full=0,
            col_end_full=150,
            overlap_top=0,
            overlap_bottom=50,
            overlap_left=0,
            overlap_right=50,
        )
        rs, cs = tile.core_slice
        assert rs == slice(0, 100)
        assert cs == slice(0, 100)

    def test_core_slice_with_overlap(self):
        """Middle tile has core offset by overlap."""
        tile = TileSpec(
            row_start=100,
            row_end=200,
            col_start=100,
            col_end=200,
            row_start_full=50,
            row_end_full=250,
            col_start_full=50,
            col_end_full=250,
            overlap_top=50,
            overlap_bottom=50,
            overlap_left=50,
            overlap_right=50,
        )
        rs, cs = tile.core_slice
        assert rs == slice(50, 150)
        assert cs == slice(50, 150)

    def test_write_slice(self):
        tile = TileSpec(
            row_start=100,
            row_end=200,
            col_start=50,
            col_end=150,
            row_start_full=50,
            row_end_full=250,
            col_start_full=0,
            col_end_full=200,
            overlap_top=50,
            overlap_bottom=50,
            overlap_left=50,
            overlap_right=50,
        )
        rs, cs = tile.write_slice
        assert rs == slice(100, 200)
        assert cs == slice(50, 150)

    def test_read_slice(self):
        tile = TileSpec(
            row_start=100,
            row_end=200,
            col_start=50,
            col_end=150,
            row_start_full=50,
            row_end_full=250,
            col_start_full=0,
            col_end_full=200,
            overlap_top=50,
            overlap_bottom=50,
            overlap_left=50,
            overlap_right=50,
        )
        rs, cs = tile.read_slice
        assert rs == slice(50, 250)
        assert cs == slice(0, 200)


# ---------------------------------------------------------------------------
# calculate_buffer_distance
# ---------------------------------------------------------------------------


class TestCalculateBufferDistance:
    """Tests for calculate_buffer_distance() in tiling.py."""

    def test_zero_height(self):
        assert calculate_buffer_distance(0.0) == 0.0

    def test_negative_height(self):
        assert calculate_buffer_distance(-5.0) == 0.0

    def test_short_building(self):
        """10m building: 10 / tan(3°) ≈ 190.8m."""
        result = calculate_buffer_distance(10.0)
        assert 190 < result < 192

    def test_tall_building_capped(self):
        """60m building would need ~1145m, capped at 1000m."""
        result = calculate_buffer_distance(60.0)
        assert result == MAX_BUFFER_M

    def test_custom_min_elevation(self):
        """Higher min elevation reduces buffer distance."""
        buffer_3 = calculate_buffer_distance(10.0, min_sun_elev_deg=3.0)
        buffer_10 = calculate_buffer_distance(10.0, min_sun_elev_deg=10.0)
        assert buffer_10 < buffer_3


# ---------------------------------------------------------------------------
# validate_tile_size
# ---------------------------------------------------------------------------


class TestComputeMaxTilePixels:
    """Tests for compute_max_tile_pixels() in tiling.py."""

    def test_gpu_memory_budget_drives_tile_sizing(self, monkeypatch):
        """When gpu_memory_budget is present, it drives tile sizing directly."""
        budget = 8_000_000_000  # 8 GiB
        headroom = compute_max_tile_pixels.__globals__["_GPU_HEADROOM"]

        monkeypatch.setattr(
            solweig,
            "get_gpu_limits",
            lambda: {"max_buffer_size": 4_503_599_627_370_496, "gpu_memory_budget": budget, "backend": "Dx12"},
        )
        monkeypatch.setitem(compute_max_tile_pixels.__globals__, "_get_available_ram_bytes", lambda: None)
        monkeypatch.setitem(compute_max_tile_pixels.__globals__, "_get_total_ram_bytes", lambda: None)

        solweig_pixels = compute_max_tile_pixels(context="solweig")
        svf_pixels = compute_max_tile_pixels(context="svf")

        assert solweig_pixels == int(budget * headroom) // 360  # _TIMESTEP_GPU_BPP
        assert svf_pixels == int(budget * headroom) // 384  # _SVF_GPU_BPP

    def test_no_gpu_memory_budget_uses_conservative_fallback(self, monkeypatch):
        """When Rust can't determine VRAM, fall back to _GPU_BUDGET_BYTES."""
        gpu_budget = compute_max_tile_pixels.__globals__["_GPU_BUDGET_BYTES"]
        headroom = compute_max_tile_pixels.__globals__["_GPU_HEADROOM"]

        # No gpu_memory_budget key — Rust couldn't detect real VRAM
        monkeypatch.setattr(
            solweig,
            "get_gpu_limits",
            lambda: {"max_buffer_size": 4_503_599_627_370_496, "backend": "Dx12"},
        )
        monkeypatch.setitem(compute_max_tile_pixels.__globals__, "_get_available_ram_bytes", lambda: None)
        monkeypatch.setitem(compute_max_tile_pixels.__globals__, "_get_total_ram_bytes", lambda: None)

        solweig_pixels = compute_max_tile_pixels(context="solweig")
        svf_pixels = compute_max_tile_pixels(context="svf")

        assert solweig_pixels == int(gpu_budget * headroom) // 360
        assert svf_pixels == int(gpu_budget * headroom) // 384

    def test_svf_and_solweig_use_different_bpp(self, monkeypatch):
        """SVF (384 B/px) should produce smaller tiles than solweig (360 B/px)."""
        budget = 4_000_000_000
        headroom = compute_max_tile_pixels.__globals__["_GPU_HEADROOM"]

        monkeypatch.setattr(
            solweig,
            "get_gpu_limits",
            lambda: {"max_buffer_size": budget, "gpu_memory_budget": budget, "backend": "Metal"},
        )
        monkeypatch.setitem(compute_max_tile_pixels.__globals__, "_get_available_ram_bytes", lambda: None)
        monkeypatch.setitem(compute_max_tile_pixels.__globals__, "_get_total_ram_bytes", lambda: None)

        svf_pixels = compute_max_tile_pixels(context="svf")
        solweig_pixels = compute_max_tile_pixels(context="solweig")

        assert svf_pixels == int(budget * headroom) // 384
        assert solweig_pixels == int(budget * headroom) // 360
        assert svf_pixels < solweig_pixels

    def test_available_ram_takes_precedence_over_total(self, monkeypatch):
        """When available RAM < total, tiles should be sized to available."""
        monkeypatch.setattr(solweig, "get_gpu_limits", lambda: None)
        monkeypatch.setitem(compute_max_tile_pixels.__globals__, "_get_available_ram_bytes", lambda: 4_000_000_000)
        monkeypatch.setitem(compute_max_tile_pixels.__globals__, "_get_total_ram_bytes", lambda: 32_000_000_000)

        pixels = compute_max_tile_pixels(context="solweig")
        # 4GB * 0.50 / 400 = 5_000_000
        assert pixels == 5_000_000

    def test_available_ram_without_total(self, monkeypatch):
        """Available RAM alone (total detection failed) should still work."""
        monkeypatch.setattr(solweig, "get_gpu_limits", lambda: None)
        monkeypatch.setitem(compute_max_tile_pixels.__globals__, "_get_available_ram_bytes", lambda: 8_000_000_000)
        monkeypatch.setitem(compute_max_tile_pixels.__globals__, "_get_total_ram_bytes", lambda: None)

        pixels = compute_max_tile_pixels(context="solweig")
        assert pixels == 10_000_000  # 8GB * 0.50 / 400

    def test_falls_back_to_total_when_available_fails(self, monkeypatch):
        """When available detection fails, total × fraction is used."""
        monkeypatch.setattr(solweig, "get_gpu_limits", lambda: None)
        monkeypatch.setitem(compute_max_tile_pixels.__globals__, "_get_available_ram_bytes", lambda: None)
        monkeypatch.setitem(compute_max_tile_pixels.__globals__, "_get_total_ram_bytes", lambda: 16_000_000_000)

        pixels = compute_max_tile_pixels(context="solweig")
        assert pixels == 20_000_000  # 16GB * 0.50 / 400

    def test_tile_side_logging_with_avail_only(self, monkeypatch):
        """compute_max_tile_side must not crash when total RAM is None."""
        from solweig.tiling import _cached_max_tile_side, compute_max_tile_side

        monkeypatch.setattr(solweig, "get_gpu_limits", lambda: None)
        monkeypatch.setitem(compute_max_tile_side.__globals__, "_get_available_ram_bytes", lambda: 8_000_000_000)
        monkeypatch.setitem(compute_max_tile_side.__globals__, "_get_total_ram_bytes", lambda: None)
        _cached_max_tile_side.pop("solweig", None)

        try:
            side = compute_max_tile_side(context="solweig")
            assert side >= MIN_TILE_SIZE
        finally:
            _cached_max_tile_side.pop("solweig", None)


class TestValidateTileSize:
    """Tests for validate_tile_size() — tile_size is core (excluding overlap)."""

    def test_valid_core_passes(self):
        """Core size that fits with buffer within resource limit passes unchanged."""
        adjusted, warning = validate_tile_size(800, buffer_pixels=50, pixel_size=1.0)
        assert adjusted == 800
        assert warning is None

    def test_below_minimum_adjusted(self):
        adjusted, warning = validate_tile_size(100, buffer_pixels=10, pixel_size=1.0)
        assert adjusted == MIN_TILE_SIZE
        assert warning is not None
        assert "below minimum" in warning

    def test_core_plus_buffer_exceeds_limit(self):
        """Core is reduced so core + 2*buffer fits within resource limit."""
        max_full = compute_max_tile_side(context="solweig")
        buffer_pixels = 50
        # Request core that with buffer would exceed limit
        adjusted, warning = validate_tile_size(max_full, buffer_pixels=buffer_pixels, pixel_size=1.0)
        assert adjusted == max_full - 2 * buffer_pixels
        assert warning is not None
        assert "exceeds resource limit" in warning

    def test_large_buffer_allows_subminimum_core_to_respect_limit(self):
        """When overlap is huge, core may drop below MIN_TILE_SIZE to keep full tile valid."""
        max_full = compute_max_tile_side(context="solweig")
        # Buffer so large that max_core < MIN_TILE_SIZE
        huge_buffer = (max_full - MIN_TILE_SIZE) // 2 + 100
        adjusted, warning = validate_tile_size(800, buffer_pixels=huge_buffer, pixel_size=1.0)
        max_core = max_full - 2 * huge_buffer
        assert adjusted == max(1, max_core)
        assert warning is not None

    def test_exact_minimum(self):
        adjusted, warning = validate_tile_size(MIN_TILE_SIZE, buffer_pixels=10, pixel_size=1.0)
        assert adjusted == MIN_TILE_SIZE
        assert warning is None

    def test_small_buffer_allows_large_core(self):
        """With small buffer, core can use nearly all of the resource limit."""
        max_full = compute_max_tile_side(context="solweig")
        max_core = max_full - 2 * 10
        adjusted, warning = validate_tile_size(max_core, buffer_pixels=10, pixel_size=1.0)
        assert adjusted == max_core
        assert warning is None

    def test_context_uses_context_specific_limits(self, monkeypatch):
        """SVF context should not be constrained by SOLWEIG tile limits."""

        def _fake_max_side(*, context: str = "solweig"):
            return 4000 if context == "svf" else 1000

        # Patch directly on the function's globals dict.  QGIS mock imports can
        # cause a double-load of solweig.tiling so the module object in
        # sys.modules may differ from the one validate_tile_size was defined in.
        monkeypatch.setitem(
            validate_tile_size.__globals__,
            "compute_max_tile_side",
            _fake_max_side,
        )

        svf_adjusted, _ = validate_tile_size(950, buffer_pixels=50, pixel_size=1.0, context="svf")
        solweig_adjusted, _ = validate_tile_size(950, buffer_pixels=50, pixel_size=1.0, context="solweig")

        assert svf_adjusted == 950
        assert solweig_adjusted == 900


# ---------------------------------------------------------------------------
# generate_tiles
# ---------------------------------------------------------------------------


class TestGenerateTiles:
    """Tests for generate_tiles() in tiling.py."""

    def test_single_tile(self):
        """Small raster fits in one tile."""
        tiles = generate_tiles(100, 100, tile_size=256, overlap=50)
        assert len(tiles) == 1
        assert tiles[0].row_start == 0
        assert tiles[0].row_end == 100
        assert tiles[0].col_start == 0
        assert tiles[0].col_end == 100

    def test_single_tile_no_overlap(self):
        """Single tile has no overlap (no neighbors)."""
        tiles = generate_tiles(100, 100, tile_size=256, overlap=50)
        assert tiles[0].overlap_top == 0
        assert tiles[0].overlap_bottom == 0
        assert tiles[0].overlap_left == 0
        assert tiles[0].overlap_right == 0

    def test_2x2_tiles(self):
        """500x500 raster with 256 tile size creates 2x2 grid."""
        tiles = generate_tiles(500, 500, tile_size=256, overlap=50)
        assert len(tiles) == 4

    def test_tiles_cover_entire_raster(self):
        """All pixels are covered by at least one tile's core area."""
        rows, cols = 500, 700
        tiles = generate_tiles(rows, cols, tile_size=256, overlap=50)

        covered = np.zeros((rows, cols), dtype=bool)
        for tile in tiles:
            covered[tile.row_start : tile.row_end, tile.col_start : tile.col_end] = True

        assert np.all(covered)

    def test_overlap_only_on_inner_edges(self):
        """Edge tiles don't extend beyond raster bounds."""
        tiles = generate_tiles(500, 500, tile_size=256, overlap=50)

        for tile in tiles:
            assert tile.row_start_full >= 0
            assert tile.row_end_full <= 500
            assert tile.col_start_full >= 0
            assert tile.col_end_full <= 500

    def test_first_tile_no_top_left_overlap(self):
        """Top-left tile has no top or left overlap."""
        tiles = generate_tiles(500, 500, tile_size=256, overlap=50)
        first = tiles[0]
        assert first.overlap_top == 0
        assert first.overlap_left == 0

    def test_last_tile_no_bottom_right_overlap(self):
        """Bottom-right tile has no bottom or right overlap."""
        tiles = generate_tiles(500, 500, tile_size=256, overlap=50)
        last = tiles[-1]
        assert last.overlap_bottom == 0
        assert last.overlap_right == 0

    def test_middle_tile_has_all_overlaps(self):
        """Middle tile in a 3x3 grid has overlap on all sides."""
        tiles = generate_tiles(768, 768, tile_size=256, overlap=50)
        assert len(tiles) == 9  # 3x3

        middle = tiles[4]  # center tile
        assert middle.overlap_top == 50
        assert middle.overlap_bottom == 50
        assert middle.overlap_left == 50
        assert middle.overlap_right == 50

    def test_non_square_raster(self):
        """Non-square raster creates asymmetric tile grid."""
        tiles = generate_tiles(rows=200, cols=600, tile_size=256, overlap=30)
        assert len(tiles) == 3  # 1 row × 3 cols

    def test_exact_tile_size_fit(self):
        """Raster exactly matching tile size creates exactly 1 tile."""
        tiles = generate_tiles(256, 256, tile_size=256, overlap=50)
        assert len(tiles) == 1

    def test_core_slice_within_full_shape(self):
        """core_slice must always fit within the tile's full_shape."""
        # Regression: when overlap > tile_size, clamped bounds and stored
        # overlap diverged, producing core_slice coordinates beyond the tile.
        for rows, cols, tile_size, overlap in [
            (500, 500, 256, 50),  # normal case
            (512, 1144, 256, 1000),  # overlap >> tile_size (tall buildings)
            (300, 300, 100, 200),  # overlap = 2x tile_size
        ]:
            tiles = generate_tiles(rows, cols, tile_size, overlap)
            for tile in tiles:
                fs = tile.full_shape
                cs = tile.core_slice
                assert cs[0].stop <= fs[0], f"core row {cs[0].stop} exceeds tile height {fs[0]}"
                assert cs[1].stop <= fs[1], f"core col {cs[1].stop} exceeds tile width {fs[1]}"
