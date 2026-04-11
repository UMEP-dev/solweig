"""Integration tests for multi-tile processing.

These tests use larger synthetic rasters to actually exercise multi-tile
processing rather than falling back to single-tile mode.
"""

import logging
import tempfile
from datetime import datetime

import numpy as np
import pytest
from conftest import make_mock_svf, read_timestep_geotiff
from solweig import (
    Location,
    ModelConfig,
    PrecomputedData,
    SolweigResult,
    SurfaceData,
    Weather,
)
from solweig.models.state import ThermalState, TileSpec
from solweig.tiling import (
    _calculate_auto_tile_size,
    _extract_tile_surface,
    calculate_buffer_distance,
    compute_max_tile_side,
)
from solweig.timeseries import _calculate_timeseries

pytestmark = pytest.mark.slow


class TestTilingHelpers:
    """Tests for tiling helper functions."""

    def test_auto_tile_size_returns_resource_max(self):
        """Auto tile size returns resource-derived maximum when no buffer."""
        max_side = compute_max_tile_side(context="solweig")
        assert _calculate_auto_tile_size(max_side + 1000, max_side + 1000) == max_side
        assert _calculate_auto_tile_size(100, 100) == max_side

    def test_auto_tile_size_reserves_buffer(self):
        """Auto tile size subtracts 2*buffer_pixels so `core + 2*buffer`
        fits within the resource budget without clamping."""
        from solweig.tiling import MIN_TILE_SIZE, validate_tile_size

        max_side = compute_max_tile_side(context="solweig")
        buffer_px = 200

        core = _calculate_auto_tile_size(10000, 10000, buffer_px)
        # Core must fit core + 2*buffer inside the full budget.
        assert core + 2 * buffer_px <= max_side
        # And it should actually reserve the buffer (not leave room unused),
        # down to the MIN_TILE_SIZE floor.
        expected = max(MIN_TILE_SIZE, max_side - 2 * buffer_px)
        assert core == expected

        # validate_tile_size should now accept the auto-sized core without
        # clamping — the hallmark of the fix for the spurious warning.
        adjusted, warning = validate_tile_size(core, buffer_px, 1.0, context="solweig")
        assert adjusted == core
        assert warning is None

    def test_extract_tile_surface_reuses_svf(self):
        """When surface has precomputed SVF, tile surface should get sliced SVF."""
        size = 100
        dsm = np.ones((size, size), dtype=np.float32) * 5.0
        mock_svf = make_mock_svf((size, size))
        surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=mock_svf)

        # Create a tile covering rows 10-60, cols 10-60 (with 10px overlap)
        tile = TileSpec(
            row_start=20,
            row_end=50,
            col_start=20,
            col_end=50,
            row_start_full=10,
            row_end_full=60,
            col_start_full=10,
            col_end_full=60,
            overlap_top=10,
            overlap_bottom=10,
            overlap_left=10,
            overlap_right=10,
        )

        tile_surface = _extract_tile_surface(surface, tile, pixel_size=1.0)

        # SVF should be set (sliced from global, not recomputed)
        assert tile_surface.svf is not None
        assert tile_surface.svf.svf.shape == (50, 50)  # 60-10 = 50

        # Values should match the sliced region of the global SVF
        np.testing.assert_array_equal(
            tile_surface.svf.svf,
            mock_svf.svf[10:60, 10:60],
        )

    def test_extract_tile_surface_leaves_svf_unset_when_missing(self):
        """When SVF is unavailable globally, tile extraction must not compute it."""
        size = 50
        dsm = np.ones((size, size), dtype=np.float32) * 5.0
        surface = SurfaceData(dsm=dsm, pixel_size=1.0)
        assert surface.svf is None

        tile = TileSpec(
            row_start=10,
            row_end=40,
            col_start=10,
            col_end=40,
            row_start_full=0,
            row_end_full=50,
            col_start_full=0,
            col_end_full=50,
            overlap_top=10,
            overlap_bottom=10,
            overlap_left=10,
            overlap_right=10,
        )

        tile_surface = _extract_tile_surface(surface, tile, pixel_size=1.0)

        # SVF remains unset; callers enforce the SVF precondition.
        assert tile_surface.svf is None

    def test_extract_tile_surface_uses_precomputed_svf_without_recompute(self):
        """When surface.svf is missing, precomputed.svf should be sliced and reused."""
        from unittest.mock import patch

        size = 50
        dsm = np.ones((size, size), dtype=np.float32) * 5.0
        surface = SurfaceData(dsm=dsm, pixel_size=1.0)
        precomputed = PrecomputedData(svf=make_mock_svf((size, size)))

        tile = TileSpec(
            row_start=10,
            row_end=40,
            col_start=10,
            col_end=40,
            row_start_full=0,
            row_end_full=50,
            col_start_full=0,
            col_end_full=50,
            overlap_top=10,
            overlap_bottom=10,
            overlap_left=10,
            overlap_right=10,
        )

        with patch.object(SurfaceData, "compute_svf", side_effect=AssertionError("compute_svf should not be called")):
            tile_surface = _extract_tile_surface(surface, tile, pixel_size=1.0, precomputed=precomputed)

        assert tile_surface.svf is not None
        assert tile_surface.svf.svf.shape == (50, 50)
        assert precomputed.svf is not None
        np.testing.assert_array_equal(
            tile_surface.svf.svf,
            precomputed.svf.svf[0:50, 0:50],
        )


class TestTimeseriesTiledIntegration:
    """Integration tests for tiled timeseries processing."""

    @pytest.fixture(scope="class")
    def multi_tile_surface(self):
        """520x520 surface with a low building so multi-tile runs also exercise overlap stitching."""
        size = 520
        dsm = np.ones((size, size), dtype=np.float32) * 10.0
        dsm[220:300, 220:300] = 15.0  # 5m relative height -> non-zero overlap buffer
        return SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf((size, size)))

    @pytest.fixture(scope="class")
    def location(self):
        return Location(latitude=57.7, longitude=12.0, utc_offset=2)

    @pytest.fixture(scope="class")
    def weather_pair(self):
        """Two consecutive timesteps for minimal timeseries."""
        return [
            Weather(datetime=datetime(2024, 7, 15, 11, 0), ta=26.0, rh=50.0, global_rad=750.0, ws=2.0),
            Weather(datetime=datetime(2024, 7, 15, 12, 0), ta=28.0, rh=45.0, global_rad=850.0, ws=2.0),
        ]

    def test_timeseries_tiled_matches_nontiled(self, multi_tile_surface, location, weather_pair, tmp_path):
        """Tiled timeseries should match non-tiled within numerical precision.

        Both paths use the same mock SVF from the surface (tiled path slices
        the global SVF per tile instead of recomputing).
        """
        ref_dir = tmp_path / "ref"
        ref_dir.mkdir()
        tiled_dir = tmp_path / "tiled"
        tiled_dir.mkdir()

        # Non-tiled (normal path -- uses mock SVF from surface)
        _calculate_timeseries(
            surface=multi_tile_surface,
            weather_series=weather_pair,
            location=location,
            output_dir=ref_dir,
            outputs=["tmrt"],
        )

        # Tiled (forced via tile_size -- slices mock SVF from the global surface)
        _calculate_timeseries(
            surface=multi_tile_surface,
            weather_series=weather_pair,
            location=location,
            output_dir=tiled_dir,
            outputs=["tmrt"],
            tile_size=256,
        )

        for i in range(len(weather_pair)):
            ref_tmrt = read_timestep_geotiff(ref_dir, "tmrt", i)
            tiled_tmrt = read_timestep_geotiff(tiled_dir, "tmrt", i)
            both_valid = np.isfinite(ref_tmrt) & np.isfinite(tiled_tmrt)
            if both_valid.sum() > 0:
                diff = np.abs(ref_tmrt[both_valid] - tiled_tmrt[both_valid])
                # Both paths now use the same mock SVF (tiled path slices from global).
                assert diff.mean() < 0.01, f"Timestep {i}: mean Tmrt diff {diff.mean():.2f}deg C too large"
                assert diff.max() < 0.1, f"Timestep {i}: max Tmrt diff {diff.max():.2f}deg C too large"

    def test_timeseries_tiled_writes_output_with_async_disabled(
        self, multi_tile_surface, location, weather_pair, tmp_path
    ):
        """Tiled timeseries must write per-timestep GeoTIFFs even with SOLWEIG_ASYNC_OUTPUT=0."""
        import os

        old_val = os.environ.get("SOLWEIG_ASYNC_OUTPUT")
        os.environ["SOLWEIG_ASYNC_OUTPUT"] = "0"
        try:
            _calculate_timeseries(
                surface=multi_tile_surface,
                weather_series=weather_pair,
                location=location,
                output_dir=tmp_path,
                outputs=["tmrt"],
                tile_size=256,
            )
        finally:
            if old_val is None:
                os.environ.pop("SOLWEIG_ASYNC_OUTPUT", None)
            else:
                os.environ["SOLWEIG_ASYNC_OUTPUT"] = old_val

        # Verify output files were actually written
        for i in range(len(weather_pair)):
            tmrt = read_timestep_geotiff(tmp_path, "tmrt", i)
            assert tmrt is not None, f"No tmrt GeoTIFF for timestep {i} with SOLWEIG_ASYNC_OUTPUT=0"
            assert np.isfinite(tmrt).sum() > 0

    def test_timeseries_tiled_state_accumulates(self, multi_tile_surface, location, weather_pair, tmp_path):
        """Thermal state should evolve across timesteps in tiled mode."""
        from solweig import tiling as tiling_module

        original_calculate_single = __import__("solweig.api", fromlist=["_calculate_single"])._calculate_single
        state_calls: list[tuple[int, float, float]] = []

        def _spy_calculate_single(**kwargs):
            state = kwargs["state"]
            state_calls.append((id(state), float(state.firstdaytime), float(state.timeadd)))
            return original_calculate_single(**kwargs)

        rows, cols = multi_tile_surface.shape
        pixel_size = multi_tile_surface.pixel_size
        max_height = multi_tile_surface.max_height
        buffer_m = calculate_buffer_distance(max_height)
        buffer_pixels = int(np.ceil(buffer_m / pixel_size))
        adjusted_tile_size, _warning = tiling_module.validate_tile_size(256, buffer_pixels, pixel_size)
        expected_tiles = len(tiling_module.generate_tiles(rows, cols, adjusted_tile_size, buffer_pixels))

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr("solweig.api._calculate_single", _spy_calculate_single)
        try:
            summary = _calculate_timeseries(
                surface=multi_tile_surface,
                weather_series=weather_pair,
                location=location,
                output_dir=tmp_path,
                outputs=["tmrt"],
                tile_size=256,
            )
        finally:
            monkeypatch.undo()

        assert buffer_pixels > 0
        assert len(state_calls) == expected_tiles * len(weather_pair)
        for tile_idx in range(expected_tiles):
            first_call = state_calls[tile_idx * len(weather_pair)]
            second_call = state_calls[tile_idx * len(weather_pair) + 1]
            assert first_call[1:] == (1.0, 0.0)
            assert second_call[0] == first_call[0]
            assert second_call[1] == 0.0
            assert second_call[2] > 0.0

        # Both timesteps should produce valid results
        assert len(summary) == 2
        for i in range(len(weather_pair)):
            tmrt = read_timestep_geotiff(tmp_path, "tmrt", i)
            valid = np.isfinite(tmrt)
            assert valid.sum() > 0, "Expected some valid Tmrt values"

    def test_timeseries_tiled_progress_callback(self, multi_tile_surface, location, weather_pair, tmp_path):
        """Progress callback should be called for tiled timeseries."""
        calls = []

        def track(current, total):
            calls.append((current, total))

        _calculate_timeseries(
            surface=multi_tile_surface,
            weather_series=weather_pair,
            location=location,
            output_dir=tmp_path,
            progress_callback=track,
            tile_size=256,
        )

        # Callbacks report (current, total) where total = n_tiles * n_timesteps
        assert len(calls) >= 2  # at least one per timestep
        assert calls[-1][0] == calls[-1][1]  # final callback: current == total

    def test_timeseries_tiled_summary_only(self, multi_tile_surface, location, weather_pair, tmp_path):
        """Default mode (no outputs) should still produce summary grids."""
        summary = _calculate_timeseries(
            surface=multi_tile_surface,
            weather_series=weather_pair,
            location=location,
            output_dir=tmp_path,
            tile_size=256,
        )

        assert len(summary) == 2
        # Summary grids should exist
        assert summary.tmrt_mean is not None
        assert summary.tmrt_max is not None
        assert summary.tmrt_min is not None

    def test_timeseries_tiled_summary_only_requests_tmrt_and_shadow(
        self, multi_tile_surface, location, weather_pair, tmp_path
    ):
        """Summary-only mode should request tmrt and shadow from tiled per-tile calculations."""
        from solweig import SolweigResult

        captured: list[set[str] | None] = []

        def _fake_calculate(**kwargs):
            captured.append(kwargs.get("_requested_outputs"))
            shape = kwargs["surface"].dsm.shape
            return SolweigResult(
                tmrt=np.zeros(shape, dtype=np.float32),
                shadow=np.zeros(shape, dtype=np.float32),
                kdown=None,
                kup=None,
                ldown=None,
                lup=None,
                utci=None,
                pet=None,
                state=None,
            )

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr("solweig.api._calculate_single", _fake_calculate)
        try:
            _calculate_timeseries(
                surface=multi_tile_surface,
                weather_series=weather_pair,
                location=location,
                output_dir=tmp_path,
                tile_size=256,
            )
        finally:
            monkeypatch.undo()

        assert captured and all(req == {"tmrt", "shadow"} for req in captured)

    def test_timeseries_tiled_precreates_tile_surfaces_once(self, multi_tile_surface, location, weather_pair, tmp_path):
        """Tile surfaces should be extracted once per tile, not once per timestep."""
        from unittest.mock import patch

        from solweig import tiling as tiling_module

        extract_calls = 0
        original_extract = tiling_module._extract_tile_surface

        def spy_extract(*args, **kwargs):
            nonlocal extract_calls
            extract_calls += 1
            return original_extract(*args, **kwargs)

        rows, cols = multi_tile_surface.shape
        pixel_size = multi_tile_surface.pixel_size
        max_height = multi_tile_surface.max_height
        buffer_m = calculate_buffer_distance(max_height)
        buffer_pixels = int(np.ceil(buffer_m / pixel_size))
        tile_size = 256
        adjusted_tile_size, _warning = tiling_module.validate_tile_size(tile_size, buffer_pixels, pixel_size)
        expected_tiles = len(tiling_module.generate_tiles(rows, cols, adjusted_tile_size, buffer_pixels))

        with patch("solweig.tiling._extract_tile_surface", side_effect=spy_extract):
            _calculate_timeseries(
                surface=multi_tile_surface,
                weather_series=weather_pair,
                location=location,
                output_dir=tmp_path,
                tile_size=256,
            )

        assert extract_calls == expected_tiles, (
            f"Expected {expected_tiles} tile surface extractions, got {extract_calls} "
            "(possible per-timestep recomputation regression)"
        )

    def test_invalid_tile_workers_raises(self):
        """Invalid tile_workers in ModelConfig raises ValueError."""
        with pytest.raises(ValueError, match="tile_workers must be >= 1"):
            ModelConfig(tile_workers=0)

    def test_invalid_tile_queue_depth_raises(self):
        """Invalid tile_queue_depth in ModelConfig raises ValueError."""
        with pytest.raises(ValueError, match="tile_queue_depth must be >= 0"):
            ModelConfig(tile_queue_depth=-1)


class TestTiledAccumulatorParity:
    """Verify GridAccumulator.update_tile() produces identical results to update().

    This is the core correctness gate for the memory-efficient tiled path:
    accumulating per-tile must yield the same grids as accumulating the whole
    raster at once.
    """

    @pytest.fixture
    def setup(self):
        """Create a 100x100 raster with known values and 4 tiles."""
        from solweig.models.state import TileSpec
        from solweig.postprocess import compute_utci_grid

        np.random.seed(42)
        rows, cols = 100, 100
        shape = (rows, cols)

        tmrt = np.random.uniform(20.0, 60.0, shape).astype(np.float32)
        # Sprinkle some NaNs to exercise the valid mask
        tmrt[0:5, :] = np.nan
        tmrt[50:55, 30:40] = np.nan

        shadow = np.random.uniform(0.0, 1.0, shape).astype(np.float32)
        shadow[0:5, :] = np.nan

        weather_day = Weather(
            datetime=datetime(2024, 7, 15, 12, 0),
            ta=28.0,
            rh=45.0,
            global_rad=850.0,
            ws=2.0,
        )
        weather_night = Weather(
            datetime=datetime(2024, 7, 15, 22, 0),
            ta=18.0,
            rh=70.0,
            global_rad=0.0,
            ws=1.0,
        )
        location = Location(latitude=57.7, longitude=12.0, utc_offset=2)
        weather_day.compute_derived(location)
        weather_night.compute_derived(location)

        # 4 tiles: 2x2, tile_size=50, no overlap for simplicity
        tiles = [
            TileSpec(
                row_start=r * 50,
                row_end=(r + 1) * 50,
                col_start=c * 50,
                col_end=(c + 1) * 50,
                row_start_full=r * 50,
                row_end_full=(r + 1) * 50,
                col_start_full=c * 50,
                col_end_full=(c + 1) * 50,
                overlap_top=0,
                overlap_bottom=0,
                overlap_left=0,
                overlap_right=0,
            )
            for r in range(2)
            for c in range(2)
        ]

        return {
            "shape": shape,
            "tmrt": tmrt,
            "shadow": shadow,
            "weather_day": weather_day,
            "weather_night": weather_night,
            "tiles": tiles,
            "compute_utci_fn": compute_utci_grid,
        }

    def _run_reference(self, setup, weathers):
        """Run the reference (non-tiled) path."""
        from solweig.summary import GridAccumulator

        acc = GridAccumulator(
            shape=setup["shape"],
            heat_thresholds_day=[32.0, 38.0],
            heat_thresholds_night=[26.0],
            timestep_hours=1.0,
        )
        for w in weathers:
            result = SolweigResult(
                tmrt=setup["tmrt"].copy(),
                shadow=setup["shadow"].copy(),
                kdown=None,
                kup=None,
                ldown=None,
                lup=None,
                utci=None,
                pet=None,
                state=None,
            )
            acc.update(result, w, compute_utci_fn=setup["compute_utci_fn"])
        return acc

    def _run_tiled(self, setup, weathers):
        """Run the tile-aware path."""
        from solweig.summary import GridAccumulator

        acc = GridAccumulator(
            shape=setup["shape"],
            heat_thresholds_day=[32.0, 38.0],
            heat_thresholds_night=[26.0],
            timestep_hours=1.0,
        )
        for w in weathers:
            acc.begin_timestep()
            for tile in setup["tiles"]:
                # Tile-sized arrays (identical to full since overlap=0)
                tile_tmrt = setup["tmrt"][tile.read_slice]
                tile_shadow = setup["shadow"][tile.read_slice]
                acc.update_tile(
                    tile_tmrt,
                    tile_shadow,
                    tile.write_slice,
                    tile.core_slice,
                    w,
                    setup["compute_utci_fn"],
                )
            acc.commit_timestep(w)
        return acc

    def test_grid_accumulators_match_exactly(self, setup):
        """All internal grid accumulators must be identical between paths."""
        weathers = [setup["weather_day"], setup["weather_night"]]
        ref = self._run_reference(setup, weathers)
        tiled = self._run_tiled(setup, weathers)

        # Compare all grid accumulators
        grid_attrs = [
            "_tmrt_sum",
            "_tmrt_count",
            "_tmrt_max",
            "_tmrt_min",
            "_tmrt_day_sum",
            "_tmrt_day_count",
            "_tmrt_night_sum",
            "_tmrt_night_count",
            "_utci_sum",
            "_utci_count",
            "_utci_max",
            "_utci_min",
            "_utci_day_sum",
            "_utci_day_count",
            "_utci_night_sum",
            "_utci_night_count",
            "_sun_hours",
            "_shade_hours",
        ]
        for attr in grid_attrs:
            ref_arr = getattr(ref, attr)
            tiled_arr = getattr(tiled, attr)
            np.testing.assert_array_equal(
                ref_arr,
                tiled_arr,
                err_msg=f"GridAccumulator.{attr} differs between update() and update_tile()",
            )

        # UTCI threshold exceedance
        for threshold in ref._utci_hours_above:
            np.testing.assert_array_equal(
                ref._utci_hours_above[threshold],
                tiled._utci_hours_above[threshold],
                err_msg=f"UTCI hours above {threshold} differs",
            )

    def test_scalar_accumulators_match(self, setup):
        """Per-timestep scalar timeseries must match between paths."""
        weathers = [setup["weather_day"], setup["weather_night"]]
        ref = self._run_reference(setup, weathers)
        tiled = self._run_tiled(setup, weathers)

        assert ref._n_timesteps == tiled._n_timesteps
        assert ref._n_daytime == tiled._n_daytime
        assert ref._n_nighttime == tiled._n_nighttime
        assert ref._ts_datetime == tiled._ts_datetime
        assert ref._ts_ta == tiled._ts_ta
        assert ref._ts_is_daytime == tiled._ts_is_daytime

        # Spatial means may differ in the last few decimal places due to
        # summation order (tile-by-tile vs whole-array).  Use rtol.
        for attr in ["_ts_tmrt_mean", "_ts_utci_mean", "_ts_sun_fraction"]:
            ref_vals = getattr(ref, attr)
            tiled_vals = getattr(tiled, attr)
            for i, (r, t) in enumerate(zip(ref_vals, tiled_vals, strict=True)):
                if np.isnan(r) and np.isnan(t):
                    continue
                assert abs(r - t) < 1e-4, f"{attr}[{i}] differs: ref={r}, tiled={t} (diff={abs(r - t):.2e})"

    def test_finalized_summaries_match(self, setup):
        """Final TimeseriesSummary grids must match."""
        weathers = [setup["weather_day"], setup["weather_night"]]
        ref = self._run_reference(setup, weathers).finalize()
        tiled = self._run_tiled(setup, weathers).finalize()

        for attr in [
            "tmrt_mean",
            "tmrt_max",
            "tmrt_min",
            "utci_mean",
            "utci_max",
            "utci_min",
            "sun_hours",
            "shade_hours",
        ]:
            ref_arr = getattr(ref, attr)
            tiled_arr = getattr(tiled, attr)
            both_valid = np.isfinite(ref_arr) & np.isfinite(tiled_arr)
            if both_valid.sum() > 0:
                np.testing.assert_allclose(
                    tiled_arr[both_valid],
                    ref_arr[both_valid],
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"TimeseriesSummary.{attr} differs",
                )

    def test_memmap_accumulator_matches_inmemory(self, setup, tmp_path):
        """GridAccumulator with memmap backing must produce identical results."""
        from solweig.summary import GridAccumulator

        weathers = [setup["weather_day"]]
        ref = self._run_tiled(setup, weathers)

        # Memmap-backed
        acc = GridAccumulator(
            shape=setup["shape"],
            heat_thresholds_day=[32.0, 38.0],
            heat_thresholds_night=[26.0],
            timestep_hours=1.0,
            memmap_dir=tmp_path,
        )
        for w in weathers:
            acc.begin_timestep()
            for tile in setup["tiles"]:
                acc.update_tile(
                    setup["tmrt"][tile.read_slice],
                    setup["shadow"][tile.read_slice],
                    tile.write_slice,
                    tile.core_slice,
                    w,
                    setup["compute_utci_fn"],
                )
            acc.commit_timestep(w)

        np.testing.assert_array_equal(ref._tmrt_sum, acc._tmrt_sum)
        np.testing.assert_array_equal(ref._utci_sum, acc._utci_sum)
        np.testing.assert_array_equal(ref._sun_hours, acc._sun_hours)


class TestThermalStateMemmapParity:
    """Verify ThermalState.initial_memmap() behaves identically to initial()."""

    def test_memmap_state_shape_and_dtype(self, tmp_path):
        shape = (50, 50)
        state = ThermalState.initial_memmap(shape, tmp_path)
        assert state.tgmap1.shape == shape
        assert state.tgmap1.dtype == np.float32
        assert state.tgout1.shape == shape

    def test_memmap_state_zeros(self, tmp_path):
        shape = (50, 50)
        state = ThermalState.initial_memmap(shape, tmp_path)
        np.testing.assert_array_equal(state.tgmap1, 0.0)
        np.testing.assert_array_equal(state.tgout1, 0.0)

    def test_memmap_state_matches_initial(self, tmp_path):
        shape = (30, 40)
        ref = ThermalState.initial(shape)
        mm = ThermalState.initial_memmap(shape, tmp_path)
        for attr in ["tgmap1", "tgmap1_e", "tgmap1_s", "tgmap1_w", "tgmap1_n", "tgout1"]:
            np.testing.assert_array_equal(getattr(ref, attr), getattr(mm, attr))
        assert mm.firstdaytime == ref.firstdaytime
        assert mm.timeadd == ref.timeadd

    def test_memmap_state_is_writable(self, tmp_path):
        shape = (20, 20)
        state = ThermalState.initial_memmap(shape, tmp_path)
        state.tgmap1[5, 5] = 42.0
        assert state.tgmap1[5, 5] == 42.0


_STUB_LOCATION = Location(latitude=57.7, longitude=12.0, utc_offset=2)
_STUB_WEATHER_PAIR = [
    Weather(datetime=datetime(2024, 7, 15, 11, 0), ta=26.0, rh=50.0, global_rad=750.0, ws=2.0),
    Weather(datetime=datetime(2024, 7, 15, 12, 0), ta=28.0, rh=45.0, global_rad=850.0, ws=2.0),
]


def _stub_calculate(**kwargs):
    shape = kwargs["surface"].dsm.shape
    return SolweigResult(
        tmrt=np.full(shape, 40.0, dtype=np.float32),
        shadow=np.full(shape, 0.5, dtype=np.float32),
        kdown=None,
        kup=None,
        ldown=None,
        lup=None,
        utci=None,
        pet=None,
        state=None,
    )


class TestTiledMemoryRegression:
    """Verify the tiled timeseries path does not allocate full-raster output arrays.

    This is the primary regression gate: the fix for issue #11 eliminates
    per-timestep np.full((rows, cols), ...) allocations in the tiled loop.
    We use tracemalloc to detect any allocation proportional to full-raster
    size within the critical section.
    """

    def test_no_full_raster_output_arrays_in_tiled_timeseries(self, tmp_path):
        """Peak memory during tiled timeseries must NOT grow with raster size.

        Strategy: run the tiled timeseries with a mock _calculate_single that
        returns tile-shaped results.  Track allocations with tracemalloc.
        Compare peak allocation of a 200x200 raster (4 tiles) vs a 400x400
        raster (16 tiles).  Peak should be proportional to tile size, not
        raster size.
        """
        import tracemalloc

        def _run_tiled(size):
            dsm = np.ones((size, size), dtype=np.float32) * 5.0
            dsm[size // 4 : size // 4 + 10, size // 4 : size // 4 + 10] = 10.0
            surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf((size, size)))

            import unittest.mock

            tracemalloc.start()
            snapshot_before = tracemalloc.take_snapshot()

            with unittest.mock.patch("solweig.api._calculate_single", _stub_calculate):
                _calculate_timeseries(
                    surface=surface,
                    weather_series=_STUB_WEATHER_PAIR,
                    location=_STUB_LOCATION,
                    output_dir=tmp_path / f"mem_{size}",
                )

            snapshot_after = tracemalloc.take_snapshot()
            tracemalloc.stop()

            # Sum allocated bytes from tiling.py and summary.py
            stats = snapshot_after.compare_to(snapshot_before, "filename")
            tiling_bytes = sum(
                s.size_diff
                for s in stats
                if ("tiling.py" in str(s.traceback) or "summary.py" in str(s.traceback)) and s.size_diff > 0
            )
            return tiling_bytes

        small_bytes = _run_tiled(200)
        large_bytes = _run_tiled(400)

        # 400x400 is 4x the pixels of 200x200.  If the old code allocated
        # full-raster arrays per timestep, large_bytes would be ~4x small_bytes.
        # With the fix, both should be similar (proportional to tile size).
        # Allow a generous 2.5x ratio to account for accumulator resizing —
        # the key assertion is that it's NOT ~4x.
        ratio = large_bytes / max(small_bytes, 1)
        assert ratio < 2.5, (
            f"Memory scaled {ratio:.1f}x when raster size quadrupled. "
            f"Expected <2.5x (small={small_bytes / 1024:.0f}KB, "
            f"large={large_bytes / 1024:.0f}KB). "
            f"This suggests full-raster arrays are still being allocated."
        )

    def test_memmap_threshold_activates(self, tmp_path):
        """Verify memmap is used when pixel count exceeds the threshold."""
        from unittest.mock import patch

        # Use a small raster but lower the threshold so memmap activates
        size = 100
        dsm = np.ones((size, size), dtype=np.float32) * 5.0
        surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf((size, size)))

        # Set threshold to 1 pixel so memmap always activates
        with (
            patch("solweig.tiling._MEMMAP_PIXEL_THRESHOLD", 1),
            patch("solweig.api._calculate_single", _stub_calculate),
        ):
            summary = _calculate_timeseries(
                surface=surface,
                weather_series=[_STUB_WEATHER_PAIR[0]],
                location=_STUB_LOCATION,
                output_dir=tmp_path,
            )

        # Summary should still be valid
        assert summary is not None
        assert summary.tmrt_mean is not None
        valid = np.isfinite(summary.tmrt_mean)
        assert valid.sum() > 0

    @pytest.mark.parametrize(
        "shape, max_fill_shape",
        [
            ((520, 520), (520, 520)),
            ((300, 1025), (256, 256)),
        ],
        ids=["square-rejects-full-frame", "wide-rejects-oversized-block"],
    )
    def test_tiled_outputs_precreation_bounded_fill_buffer(self, tmp_path, shape, max_fill_shape):
        """Rasterio output precreation must keep fill buffers bounded (not full-raster)."""
        from unittest.mock import patch

        from solweig._compat import GDAL_ENV

        if GDAL_ENV:
            pytest.skip("Rasterio-specific regression guard")

        rows, cols = shape
        dsm = np.ones((rows, cols), dtype=np.float32) * 5.0
        surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf((rows, cols)))

        real_full = np.full

        def _guarded_full(alloc_shape, fill_value, *args, **kwargs):
            s = (alloc_shape,) if isinstance(alloc_shape, int) else tuple(alloc_shape)
            if len(s) == 2 and (s[0] > max_fill_shape[0] or s[1] > max_fill_shape[1]):
                raise AssertionError(f"create_empty_raster allocated an oversized fill buffer: {s}")
            return real_full(alloc_shape, fill_value, *args, **kwargs)

        with (
            patch("solweig.tiling._MEMMAP_PIXEL_THRESHOLD", 1),
            patch("solweig.api._calculate_single", _stub_calculate),
            patch("solweig.io.np.full", side_effect=_guarded_full),
        ):
            _calculate_timeseries(
                surface=surface,
                weather_series=_STUB_WEATHER_PAIR,
                location=_STUB_LOCATION,
                output_dir=tmp_path / "tiled_outputs",
                outputs=["tmrt"],
                tile_size=256,
            )

    def test_memmap_cleanup_warning_does_not_abort_timeseries(self, tmp_path, caplog):
        """Cleanup failures should be logged and not abort successful runs."""
        from unittest.mock import patch

        size = 100
        dsm = np.ones((size, size), dtype=np.float32) * 5.0
        surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf((size, size)))

        caplog.set_level(logging.WARNING, logger="solweig.timeseries")

        with (
            patch("solweig.tiling._MEMMAP_PIXEL_THRESHOLD", 1),
            patch("solweig.api._calculate_single", _stub_calculate),
            patch.object(
                tempfile.TemporaryDirectory,
                "cleanup",
                autospec=True,
                side_effect=OSError("simulated cleanup failure"),
            ),
        ):
            summary = _calculate_timeseries(
                surface=surface,
                weather_series=[_STUB_WEATHER_PAIR[0]],
                location=_STUB_LOCATION,
                output_dir=tmp_path / "cleanup_warning",
            )

        assert summary is not None
        assert summary.tmrt_mean is not None
        assert "Could not remove memmap temp dir" in caplog.text
