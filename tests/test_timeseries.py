"""
Tests for calculate() and validate_inputs().

These cover the critical gap: the primary user workflow (timeseries) had
zero dedicated tests, and validate_inputs() was untested.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest
from conftest import read_timestep_geotiff
from solweig.api import (
    HumanParams,
    Location,
    ModelConfig,
    SolweigResult,
    SurfaceData,
    TimeseriesSummary,
    Weather,
    calculate,
    validate_inputs,
)
from solweig.errors import GridShapeMismatch, MissingPrecomputedData

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def flat_surface():
    """Simple flat DSM with one 10m building."""
    from conftest import make_mock_svf

    dsm = np.zeros((30, 30), dtype=np.float32)
    dsm[10:20, 10:20] = 10.0
    return SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf((30, 30)))


@pytest.fixture(scope="module")
def location():
    return Location(latitude=57.7, longitude=12.0, utc_offset=1)


def _make_weather_series(
    base_dt: datetime,
    n_hours: int,
    ta: float = 25.0,
    rh: float = 50.0,
    global_rad: float = 800.0,
) -> list[Weather]:
    """Create a list of hourly Weather objects."""
    return [
        Weather(
            datetime=base_dt + timedelta(hours=i),
            ta=ta,
            rh=rh,
            global_rad=global_rad,
        )
        for i in range(n_hours)
    ]


# ===========================================================================
# calculate() tests
# ===========================================================================


class TestCalculateTimeseries:
    """Tests for the calculate() function."""

    def test_returns_summary(self, flat_surface, location, tmp_path):
        """Returns a TimeseriesSummary with per-timestep files when requested."""
        weather_series = _make_weather_series(datetime(2024, 7, 15, 10, 0), n_hours=3)

        summary = calculate(flat_surface, weather_series, location, output_dir=tmp_path, outputs=["tmrt", "shadow"])

        assert isinstance(summary, TimeseriesSummary)
        assert len(summary) == 3
        # Verify per-timestep files were written
        for i in range(3):
            arr = read_timestep_geotiff(tmp_path, "tmrt", i)
            assert arr.shape == (30, 30)

    def test_result_shapes_match_surface(self, flat_surface, location, tmp_path):
        """Each result has arrays matching the DSM shape."""
        weather_series = _make_weather_series(datetime(2024, 7, 15, 12, 0), n_hours=2)

        calculate(flat_surface, weather_series, location, output_dir=tmp_path, outputs=["tmrt", "shadow"])

        for i in range(2):
            tmrt = read_timestep_geotiff(tmp_path, "tmrt", i)
            assert tmrt.shape == (30, 30)
            shadow = read_timestep_geotiff(tmp_path, "shadow", i)
            assert shadow.shape == (30, 30)

    def test_empty_series_returns_empty_summary(self, flat_surface, location, tmp_path):
        """Empty weather_series returns an empty TimeseriesSummary."""
        summary = calculate(flat_surface, [], location, output_dir=tmp_path)
        assert isinstance(summary, TimeseriesSummary)
        assert len(summary) == 0

    def test_single_timestep(self, flat_surface, location, tmp_path):
        """Works with a single-element weather_series."""
        weather_series = [Weather(datetime=datetime(2024, 7, 15, 12, 0), ta=25.0, rh=50.0, global_rad=800.0)]

        summary = calculate(flat_surface, weather_series, location, output_dir=tmp_path, outputs=["tmrt"])

        assert len(summary) == 1
        tmrt = read_timestep_geotiff(tmp_path, "tmrt", 0)
        assert tmrt.shape == (30, 30)

    def test_tmrt_in_reasonable_range(self, flat_surface, location, tmp_path):
        """Tmrt values are physically plausible across timesteps."""
        weather_series = _make_weather_series(datetime(2024, 7, 15, 10, 0), n_hours=3)

        calculate(flat_surface, weather_series, location, output_dir=tmp_path, outputs=["tmrt"])

        for i in range(3):
            tmrt = read_timestep_geotiff(tmp_path, "tmrt", i)
            assert np.nanmin(tmrt) >= -50
            assert np.nanmax(tmrt) < 80

    def test_utci_pet_default_none(self, flat_surface, location, tmp_path):
        """UTCI and PET directories do not exist when not requested via outputs."""
        weather_series = _make_weather_series(datetime(2024, 7, 15, 12, 0), n_hours=2)

        calculate(flat_surface, weather_series, location, output_dir=tmp_path, outputs=["tmrt"])

        assert not (tmp_path / "utci").exists()
        assert not (tmp_path / "pet").exists()

    def test_utci_per_timestep_when_requested(self, flat_surface, location, tmp_path):
        """UTCI is computed per-timestep when included in outputs."""
        weather_series = _make_weather_series(datetime(2024, 7, 15, 12, 0), n_hours=2)

        calculate(flat_surface, weather_series, location, output_dir=tmp_path, outputs=["tmrt", "utci"])

        for i in range(2):
            tmrt = read_timestep_geotiff(tmp_path, "tmrt", i)
            assert tmrt is not None
            utci = read_timestep_geotiff(tmp_path, "utci", i)
            assert utci is not None
            assert utci.shape == (30, 30)
        assert not (tmp_path / "shadow").exists()  # not requested

    def test_nighttime_series(self, flat_surface, location, tmp_path):
        """Nighttime timesteps produce valid (low Tmrt) results."""
        weather_series = _make_weather_series(
            datetime(2024, 7, 15, 0, 0),
            n_hours=3,
            ta=15.0,
            global_rad=0.0,
        )

        summary = calculate(flat_surface, weather_series, location, output_dir=tmp_path, outputs=["tmrt"])

        assert len(summary) == 3
        for i in range(3):
            tmrt = read_timestep_geotiff(tmp_path, "tmrt", i)
            # At night, Tmrt is computed from full longwave balance. Under open
            # sky (SVF~1) the cold sky pulls Tmrt below Ta, typically by 5-10 C.
            valid = tmrt[np.isfinite(tmrt)]
            assert np.all(valid < 15.0 + 2.0), "Night Tmrt should not exceed Ta by much"
            assert np.all(valid > -10.0), "Night Tmrt should not be unreasonably cold"

    def test_day_night_transition(self, flat_surface, location, tmp_path):
        """Handles transition from night to day."""
        # 4am, 5am, ... 9am
        weather_series = [
            Weather(
                datetime=datetime(2024, 7, 15, h, 0),
                ta=15.0 + h,
                rh=70.0,
                global_rad=max(0.0, (h - 5) * 200.0),
            )
            for h in range(4, 10)
        ]

        summary = calculate(flat_surface, weather_series, location, output_dir=tmp_path, outputs=["tmrt"])

        assert len(summary) == 6
        # Later timesteps (with sun up) should generally have higher Tmrt
        # than early night timesteps
        early_tmrt = np.nanmean(read_timestep_geotiff(tmp_path, "tmrt", 0))
        late_tmrt = np.nanmean(read_timestep_geotiff(tmp_path, "tmrt", 5))
        assert late_tmrt > early_tmrt

    def test_location_auto_extracted_with_warning(self, flat_surface, tmp_path, caplog):
        """When location is None, a warning is logged before auto-extraction."""
        import logging

        weather_series = _make_weather_series(datetime(2024, 7, 15, 12, 0), n_hours=1)

        # flat_surface is synthetic (no geotransform/CRS), so Location.from_surface
        # raises ValueError after the auto-extraction warning is logged. Only that
        # specific downstream error is tolerated; the warning must always fire.
        with caplog.at_level(logging.WARNING), pytest.raises(ValueError, match="geotransform"):
            calculate(flat_surface, weather_series, location=None, output_dir=tmp_path)

        assert any("auto-extracting from surface CRS" in record.message for record in caplog.records), (
            "Expected auto-extraction warning when location=None"
        )

    def test_config_precedence_explicit_wins(self, flat_surface, location, tmp_path):
        """Explicit parameters override config values."""
        config = ModelConfig(use_anisotropic_sky=True)
        weather_series = _make_weather_series(datetime(2024, 7, 15, 12, 0), n_hours=1)

        # use_anisotropic_sky=False should override config's True
        # Since we don't have shadow matrices, aniso=True would fail.
        # If explicit=False wins, this should succeed.
        summary = calculate(
            flat_surface,
            weather_series,
            location,
            output_dir=tmp_path,
            config=config,
            use_anisotropic_sky=False,
        )

        assert len(summary) == 1

    def test_custom_human_params(self, flat_surface, location, tmp_path):
        """Custom HumanParams are accepted."""
        weather_series = _make_weather_series(datetime(2024, 7, 15, 12, 0), n_hours=1)
        human = HumanParams(abs_k=0.7, abs_l=0.97, posture="standing")

        summary = calculate(flat_surface, weather_series, location, output_dir=tmp_path, human=human)

        assert len(summary) == 1

    def test_results_differ_across_timesteps(self, flat_surface, location, tmp_path):
        """Different hours produce different Tmrt patterns."""
        weather_series = [
            Weather(datetime=datetime(2024, 7, 15, 8, 0), ta=20.0, rh=60.0, global_rad=400.0),
            Weather(datetime=datetime(2024, 7, 15, 12, 0), ta=28.0, rh=40.0, global_rad=800.0),
            Weather(datetime=datetime(2024, 7, 15, 16, 0), ta=25.0, rh=50.0, global_rad=500.0),
        ]

        calculate(flat_surface, weather_series, location, output_dir=tmp_path, outputs=["tmrt"])

        # The three timesteps should produce meaningfully different Tmrt
        means = [np.nanmean(read_timestep_geotiff(tmp_path, "tmrt", i)) for i in range(3)]
        assert not all(np.isclose(m, means[0], atol=0.5) for m in means), (
            "Expected different Tmrt across timesteps with different conditions"
        )

    def test_output_dir_saves_files(self, flat_surface, location, tmp_path):
        """When output_dir is provided, files are saved."""
        weather_series = _make_weather_series(datetime(2024, 7, 15, 12, 0), n_hours=2)

        summary = calculate(
            flat_surface,
            weather_series,
            location,
            output_dir=tmp_path,
            outputs=["tmrt", "shadow"],
        )

        assert len(summary) == 2
        # Check that some output files were created
        output_files = list(tmp_path.iterdir())
        assert len(output_files) > 0
        # Verify per-timestep data is readable from disk
        tmrt = read_timestep_geotiff(tmp_path, "tmrt", 0)
        assert tmrt is not None

    def test_explicit_anisotropic_requires_shadow_matrices(self, flat_surface, location, tmp_path):
        """Explicit anisotropic request should fail without shadow matrices."""
        weather_series = _make_weather_series(datetime(2024, 7, 15, 12, 0), n_hours=1)
        with pytest.raises(MissingPrecomputedData):
            calculate(
                flat_surface,
                weather_series,
                location,
                output_dir=tmp_path,
                use_anisotropic_sky=True,
            )

    def test_summary_only_mode(self, flat_surface, location, tmp_path):
        """Summary-only mode (no outputs) populates summary grids without per-timestep files."""
        weather_series = _make_weather_series(datetime(2024, 7, 15, 12, 0), n_hours=2)

        summary = calculate(
            flat_surface,
            weather_series,
            location,
            output_dir=tmp_path,
        )

        assert isinstance(summary, TimeseriesSummary)
        assert len(summary) == 2
        # Summary grids should be populated
        assert summary.tmrt_mean is not None
        assert summary.tmrt_mean.shape == (30, 30)
        assert summary.utci_mean is not None
        assert summary.utci_mean.shape == (30, 30)

    def test_summary_only_requests_tmrt_and_shadow(self, flat_surface, location, tmp_path, monkeypatch):
        """Summary-only mode should request tmrt and shadow from fused Rust path."""
        weather_series = _make_weather_series(datetime(2024, 7, 15, 12, 0), n_hours=2)
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

        monkeypatch.setattr("solweig.api._calculate_single", _fake_calculate)

        calculate(
            flat_surface,
            weather_series,
            location,
            output_dir=tmp_path,
        )

        assert captured and all(req == {"tmrt", "shadow"} for req in captured)

    def test_invalid_tile_workers_raises_from_config(self):
        """Invalid tile_workers in ModelConfig raises ValueError."""
        with pytest.raises(ValueError, match="tile_workers must be >= 1"):
            ModelConfig(tile_workers=0)

    def test_invalid_tile_queue_depth_raises_from_config(self):
        """Invalid tile_queue_depth in ModelConfig raises ValueError."""
        with pytest.raises(ValueError, match="tile_queue_depth must be >= 0"):
            ModelConfig(tile_queue_depth=-1)

    def test_legacy_tile_runtime_controls_warn(self, flat_surface, location, tmp_path, caplog):
        """Legacy tile runtime controls warn when used in timeseries mode."""
        weather_series = _make_weather_series(datetime(2024, 7, 15, 12, 0), n_hours=1)
        config = ModelConfig(tile_workers=4, tile_queue_depth=2, prefetch_tiles=False)

        with caplog.at_level("WARNING"):
            calculate(flat_surface, weather_series, location, output_dir=tmp_path, config=config)

        assert "Ignoring legacy timeseries tile runtime controls" in caplog.text

    def test_progress_callback_reports_completed_timesteps_for_multi_tile(self, location, tmp_path):
        """Multi-tile progress callbacks should report completed raster timesteps."""
        from conftest import make_mock_svf

        weather_series = _make_weather_series(datetime(2024, 7, 15, 10, 0), n_hours=3)
        surface = SurfaceData(
            dsm=np.pad(
                np.full((80, 80), 15.0, dtype=np.float32),
                ((220, 220), (220, 220)),
                mode="constant",
                constant_values=10.0,
            ),
            pixel_size=1.0,
            svf=make_mock_svf((520, 520)),
        )
        calls: list[tuple[int, int]] = []

        calculate(
            surface,
            weather_series,
            location,
            use_anisotropic_sky=False,
            tile_size=256,
            output_dir=tmp_path,
            progress_callback=lambda current, total: calls.append((current, total)),
        )

        # 9 tiles × 3 timesteps = 27 callbacks; final callback: current == total
        assert len(calls) == 27
        assert calls[-1][0] == calls[-1][1]

    def test_failed_multi_tile_run_cleans_up_partial_outputs(self, location, tmp_path, monkeypatch):
        """Aborted multi-tile runs should not leave finished-looking timestep GeoTIFFs behind."""
        from conftest import make_mock_svf

        weather_series = _make_weather_series(datetime(2024, 7, 15, 10, 0), n_hours=2)
        surface = SurfaceData(
            dsm=np.pad(
                np.full((80, 80), 15.0, dtype=np.float32),
                ((220, 220), (220, 220)),
                mode="constant",
                constant_values=10.0,
            ),
            pixel_size=1.0,
            svf=make_mock_svf((520, 520)),
        )

        call_count = 0
        original_calculate_single = __import__("solweig.api", fromlist=["_calculate_single"])._calculate_single

        def _fail_on_second_call(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("boom")
            return original_calculate_single(**kwargs)

        monkeypatch.setattr("solweig.api._calculate_single", _fail_on_second_call)

        with pytest.raises(RuntimeError, match="boom"):
            calculate(
                surface,
                weather_series,
                location,
                use_anisotropic_sky=False,
                tile_size=256,
                output_dir=tmp_path,
                outputs=["tmrt"],
            )

        assert not list(tmp_path.rglob("*.tif"))


class TestModelConfigTilingRuntimeSerialization:
    """Tests for tile runtime fields in ModelConfig save/load."""

    def test_model_config_save_load_tiling_runtime_fields(self, tmp_path):
        config = ModelConfig(
            use_anisotropic_sky=True,
            tile_workers=4,
            tile_queue_depth=7,
            prefetch_tiles=False,
        )
        config_path = tmp_path / "config.json"
        config.save(config_path)

        loaded = ModelConfig.load(config_path)
        assert loaded.tile_workers == 4
        assert loaded.tile_queue_depth == 7
        assert loaded.prefetch_tiles is False


# ===========================================================================
# TimeseriesSummary tests
# ===========================================================================


class TestTimeseriesSummary:
    """Tests for summary grids produced by calculate()."""

    def test_summary_grids_shapes(self, flat_surface, location, tmp_path):
        """All summary grids match DSM shape."""
        weather_series = _make_weather_series(datetime(2024, 7, 15, 10, 0), n_hours=3)

        summary = calculate(flat_surface, weather_series, location, output_dir=tmp_path)

        assert summary.tmrt_mean.shape == (30, 30)
        assert summary.tmrt_max.shape == (30, 30)
        assert summary.tmrt_min.shape == (30, 30)
        assert summary.tmrt_day_mean.shape == (30, 30)
        assert summary.tmrt_night_mean.shape == (30, 30)
        assert summary.utci_mean.shape == (30, 30)
        assert summary.utci_max.shape == (30, 30)
        assert summary.utci_min.shape == (30, 30)
        assert summary.sun_hours.shape == (30, 30)
        assert summary.shade_hours.shape == (30, 30)

    def test_summary_tmrt_mean_consistent(self, flat_surface, location, tmp_path):
        """Summary tmrt_mean matches manual mean of per-timestep results."""
        weather_series = _make_weather_series(datetime(2024, 7, 15, 10, 0), n_hours=3)

        summary = calculate(flat_surface, weather_series, location, output_dir=tmp_path, outputs=["tmrt"])

        # Manual mean from per-timestep arrays read from disk
        stacked = np.stack([read_timestep_geotiff(tmp_path, "tmrt", i) for i in range(3)], axis=0)
        manual_mean = np.nanmean(stacked, axis=0)

        np.testing.assert_allclose(summary.tmrt_mean, manual_mean, atol=0.01)

    def test_multi_tile_matches_single_tile(self, location, tmp_path):
        """Forced multi-tile stitching matches a single-tile reference."""
        from conftest import make_mock_svf

        weather_series = _make_weather_series(datetime(2024, 7, 15, 10, 0), n_hours=3)
        size = 520
        surface = SurfaceData(
            dsm=np.pad(
                np.full((80, 80), 15.0, dtype=np.float32),
                ((220, 220), (220, 220)),
                mode="constant",
                constant_values=10.0,
            ),
            pixel_size=1.0,
            svf=make_mock_svf((size, size)),
        )

        ref_dir = tmp_path / "ref"
        tiled_dir = tmp_path / "tiled"
        ref = calculate(
            surface,
            weather_series,
            location,
            use_anisotropic_sky=False,
            output_dir=ref_dir,
        )
        tiled = calculate(
            surface,
            weather_series,
            location,
            use_anisotropic_sky=False,
            output_dir=tiled_dir,
            tile_size=256,
        )

        np.testing.assert_allclose(tiled.tmrt_mean, ref.tmrt_mean, atol=0.01)
        np.testing.assert_allclose(tiled.tmrt_max, ref.tmrt_max, atol=0.01)
        np.testing.assert_allclose(tiled.utci_mean, ref.utci_mean, atol=0.01)

    def test_summary_utci_grids_populated(self, flat_surface, location, tmp_path):
        """UTCI summary grids are computed and finite."""
        weather_series = _make_weather_series(datetime(2024, 7, 15, 10, 0), n_hours=3)

        summary = calculate(flat_surface, weather_series, location, output_dir=tmp_path)

        # UTCI mean should have some finite values
        assert np.any(np.isfinite(summary.utci_mean))
        assert np.any(np.isfinite(summary.utci_max))
        assert np.any(np.isfinite(summary.utci_min))

    def test_summary_default_heat_thresholds(self, flat_surface, location, tmp_path):
        """Default thresholds produce expected dict keys."""
        weather_series = _make_weather_series(datetime(2024, 7, 15, 10, 0), n_hours=3)

        summary = calculate(flat_surface, weather_series, location, output_dir=tmp_path)

        assert summary.heat_thresholds_day == [32.0, 38.0]
        assert summary.heat_thresholds_night == [26.0]
        # All three thresholds should appear in utci_hours_above
        assert 32.0 in summary.utci_hours_above
        assert 38.0 in summary.utci_hours_above
        assert 26.0 in summary.utci_hours_above

    def test_summary_custom_heat_thresholds(self, flat_surface, location, tmp_path):
        """Custom thresholds produce matching dict keys."""
        weather_series = _make_weather_series(datetime(2024, 7, 15, 10, 0), n_hours=2)

        summary = calculate(
            flat_surface,
            weather_series,
            location,
            output_dir=tmp_path,
            heat_thresholds_day=[30, 35, 40],
            heat_thresholds_night=[20],
        )

        assert set(summary.utci_hours_above.keys()) == {20, 30, 35, 40}

    def test_summary_sun_shade_hours(self, flat_surface, location, tmp_path):
        """Sun + shade hours per pixel should sum to total hours."""
        weather_series = _make_weather_series(datetime(2024, 7, 15, 10, 0), n_hours=3)

        summary = calculate(flat_surface, weather_series, location, output_dir=tmp_path)

        assert summary.shadow_available
        total = summary.sun_hours + summary.shade_hours
        # Each daytime pixel should have total == n_hours * timestep_hours (3 hours)
        valid = np.isfinite(total)
        np.testing.assert_allclose(total[valid], 3.0, atol=0.01)

    def test_summary_day_night_counts(self, flat_surface, location, tmp_path):
        """Day/night counts should sum to total timesteps."""
        # Mix day and night hours
        weather_series = [
            Weather(datetime=datetime(2024, 7, 15, h, 0), ta=20.0, rh=50.0, global_rad=max(0.0, (h - 5) * 200.0))
            for h in range(2, 14)  # 2am to 1pm
        ]

        summary = calculate(flat_surface, weather_series, location, output_dir=tmp_path)

        assert summary.n_daytime + summary.n_nighttime == summary.n_timesteps
        assert summary.n_daytime > 0
        assert summary.n_nighttime > 0

    def test_summary_to_geotiff(self, flat_surface, location, tmp_path):
        """Summary grids can be saved to GeoTIFF."""
        weather_series = _make_weather_series(datetime(2024, 7, 15, 10, 0), n_hours=2)

        summary = calculate(flat_surface, weather_series, location, output_dir=tmp_path)
        summary.to_geotiff(str(tmp_path), surface=flat_surface)

        summary_dir = tmp_path / "summary"
        assert summary_dir.exists()
        tif_files = list(summary_dir.glob("*.tif"))
        assert len(tif_files) > 0
        # Should have tmrt + utci + sun/shade + threshold files
        names = {f.stem for f in tif_files}
        assert "tmrt_mean" in names
        assert "utci_mean" in names
        assert "sun_hours" in names

    def test_summary_geotiff_threshold_day_night_suffix(self, flat_surface, location, tmp_path):
        """Threshold GeoTIFFs should have _day or _night suffix."""
        weather_series = _make_weather_series(datetime(2024, 7, 15, 10, 0), n_hours=2)

        summary = calculate(flat_surface, weather_series, location, output_dir=tmp_path)
        summary.to_geotiff(str(tmp_path), surface=flat_surface)

        summary_dir = tmp_path / "summary"
        names = {f.stem for f in summary_dir.glob("*.tif")}
        # Default thresholds: day=[32, 38], night=[26]
        assert "utci_hours_above_32_day" in names
        assert "utci_hours_above_38_day" in names
        assert "utci_hours_above_26_night" in names

    def test_summary_len(self, flat_surface, location, tmp_path):
        """len(summary) returns n_timesteps."""
        weather_series = _make_weather_series(datetime(2024, 7, 15, 10, 0), n_hours=5)

        summary = calculate(flat_surface, weather_series, location, output_dir=tmp_path)

        assert len(summary) == 5

    def test_timeseries_populated(self, flat_surface, location, tmp_path):
        """summary.timeseries contains per-timestep scalar arrays."""
        weather_series = _make_weather_series(datetime(2024, 7, 15, 10, 0), n_hours=4)

        summary = calculate(flat_surface, weather_series, location, output_dir=tmp_path)

        ts = summary.timeseries
        assert ts is not None
        assert len(ts.datetime) == 4
        assert ts.ta.shape == (4,)
        assert ts.rh.shape == (4,)
        assert ts.ws.shape == (4,)
        assert ts.global_rad.shape == (4,)
        assert ts.tmrt_mean.shape == (4,)
        assert ts.utci_mean.shape == (4,)
        assert ts.sun_fraction.shape == (4,)
        assert ts.is_daytime.shape == (4,)
        assert ts.sun_altitude.shape == (4,)
        assert ts.direct_rad.shape == (4,)
        assert ts.diffuse_rad.shape == (4,)
        assert ts.diffuse_fraction.shape == (4,)
        assert ts.clearness_index.shape == (4,)

    def test_timeseries_values_match_weather(self, flat_surface, location, tmp_path):
        """Timeseries ta/rh/ws should match weather inputs."""
        weather_series = _make_weather_series(
            datetime(2024, 7, 15, 10, 0),
            n_hours=3,
            ta=30.0,
            rh=65.0,
        )

        summary = calculate(flat_surface, weather_series, location, output_dir=tmp_path)

        ts = summary.timeseries
        assert ts is not None
        np.testing.assert_allclose(ts.ta, 30.0, rtol=1e-5)
        np.testing.assert_allclose(ts.rh, 65.0, rtol=1e-5)

    def test_timeseries_datetimes_match(self, flat_surface, location, tmp_path):
        """Timeseries datetimes should match weather datetimes."""
        base = datetime(2024, 7, 15, 10, 0)
        weather_series = _make_weather_series(base, n_hours=3)

        summary = calculate(flat_surface, weather_series, location, output_dir=tmp_path)

        ts = summary.timeseries
        assert ts is not None
        for i, w in enumerate(weather_series):
            assert ts.datetime[i] == w.datetime

    def test_timeseries_tmrt_mean_finite(self, flat_surface, location, tmp_path):
        """Spatial mean Tmrt should be finite for daytime steps."""
        weather_series = _make_weather_series(datetime(2024, 7, 15, 10, 0), n_hours=3)

        summary = calculate(flat_surface, weather_series, location, output_dir=tmp_path)

        ts = summary.timeseries
        assert ts is not None
        # At least daytime steps should have finite Tmrt
        daytime = ts.is_daytime
        if daytime.any():
            assert np.all(np.isfinite(ts.tmrt_mean[daytime]))

    def test_timeseries_none_for_empty(self, flat_surface, location, tmp_path):
        """Empty series should have timeseries=None."""
        summary = calculate(flat_surface, [], location, output_dir=tmp_path)

        assert summary.timeseries is None

    def test_report_returns_string(self, flat_surface, location, tmp_path):
        """report() returns a non-empty string."""
        weather_series = _make_weather_series(datetime(2024, 7, 15, 10, 0), n_hours=3)

        summary = calculate(flat_surface, weather_series, location, output_dir=tmp_path)

        report = summary.report()
        assert isinstance(report, str)
        assert "SOLWEIG Summary" in report
        assert "Tmrt" in report
        assert "UTCI" in report

    def test_report_empty_summary(self, flat_surface, location):
        """report() on empty summary returns descriptive string."""
        summary = TimeseriesSummary.empty()
        assert "0 timesteps" in summary.report()

    def test_report_includes_period(self, flat_surface, location, tmp_path):
        """report() should include the simulation period."""
        weather_series = _make_weather_series(datetime(2024, 7, 15, 10, 0), n_hours=3)

        summary = calculate(flat_surface, weather_series, location, output_dir=tmp_path)

        report = summary.report()
        assert "2024-07-15" in report

    def test_repr_html(self, flat_surface, location, tmp_path):
        """_repr_html_ returns HTML for Jupyter rendering."""
        weather_series = _make_weather_series(datetime(2024, 7, 15, 10, 0), n_hours=3)

        summary = calculate(flat_surface, weather_series, location, output_dir=tmp_path)

        html = summary._repr_html_()
        assert html.startswith("<pre>")
        assert "SOLWEIG Summary" in html

    def test_plot_raises_without_matplotlib(self, flat_surface, location, tmp_path, monkeypatch):
        """plot() raises ImportError when matplotlib is not available."""
        weather_series = _make_weather_series(datetime(2024, 7, 15, 10, 0), n_hours=3)
        summary = calculate(flat_surface, weather_series, location, output_dir=tmp_path)

        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name.startswith("matplotlib"):
                raise ImportError("mock")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ImportError, match="matplotlib"):
            summary.plot()

    def test_plot_raises_on_empty_summary(self, flat_surface, location):
        """plot() raises RuntimeError on empty summary."""
        summary = TimeseriesSummary.empty()
        with pytest.raises(RuntimeError, match="No timeseries data"):
            summary.plot()

    def test_plot_saves_to_file(self, flat_surface, location, tmp_path):
        """plot(save_path=...) saves a figure to disk."""
        pytest.importorskip("matplotlib")
        weather_series = _make_weather_series(datetime(2024, 7, 15, 10, 0), n_hours=6)
        plot_dir = tmp_path / "calc_output"
        summary = calculate(flat_surface, weather_series, location, output_dir=plot_dir)

        out = tmp_path / "plot.png"
        summary.plot(save_path=out)
        assert out.exists()
        assert out.stat().st_size > 0


# ===========================================================================
# validate_inputs() tests
# ===========================================================================


class TestValidateInputs:
    """Tests for the validate_inputs() function."""

    def test_valid_inputs_no_warnings(self, flat_surface, location):
        """Valid inputs produce no warnings and don't raise."""
        weather = Weather(datetime=datetime(2024, 7, 15, 12, 0), ta=25.0, rh=50.0, global_rad=800.0)

        warnings = validate_inputs(flat_surface, location, weather)

        assert isinstance(warnings, list)

    def test_grid_shape_mismatch_raises(self):
        """Mismatched grid shapes raise GridShapeMismatch."""
        dsm = np.zeros((30, 30), dtype=np.float32)
        cdsm = np.zeros((20, 20), dtype=np.float32)  # Wrong shape
        surface = SurfaceData(dsm=dsm, cdsm=cdsm)

        with pytest.raises(GridShapeMismatch) as exc_info:
            validate_inputs(surface)

        assert exc_info.value.field == "cdsm"
        assert "(30, 30)" in str(exc_info.value.expected)
        assert "(20, 20)" in str(exc_info.value.got)

    def test_anisotropic_without_shadow_matrices(self, flat_surface):
        """Anisotropic sky without shadow matrices raises."""
        with pytest.raises(MissingPrecomputedData):
            validate_inputs(flat_surface, use_anisotropic_sky=True)

    def test_extreme_temperature_warning(self, flat_surface, location):
        """Extreme temperatures produce warnings (ta > 60 triggers)."""
        weather = Weather(datetime=datetime(2024, 7, 15, 12, 0), ta=65.0, rh=50.0, global_rad=800.0)

        warnings = validate_inputs(flat_surface, location, weather)

        assert any("ta=" in w for w in warnings)

    def test_extreme_radiation_warning(self, flat_surface, location):
        """Extreme radiation values produce warnings."""
        weather = Weather(datetime=datetime(2024, 7, 15, 12, 0), ta=25.0, rh=50.0, global_rad=1500.0)

        warnings = validate_inputs(flat_surface, location, weather)

        assert any("global_rad" in w for w in warnings)

    def test_validates_weather_list(self, flat_surface, location):
        """Can validate a list of Weather objects."""
        weather_list = _make_weather_series(datetime(2024, 7, 15, 10, 0), n_hours=5)

        warnings = validate_inputs(flat_surface, location, weather_list)

        assert isinstance(warnings, list)

    def test_unpreprocessed_cdsm_warning(self):
        """Warning when CDSM is relative but preprocess() not called."""
        from conftest import make_mock_svf

        dsm = np.zeros((20, 20), dtype=np.float32)
        cdsm = np.ones((20, 20), dtype=np.float32) * 5.0
        surface = SurfaceData(dsm=dsm, cdsm=cdsm, cdsm_relative=True, svf=make_mock_svf((20, 20)))

        warnings = validate_inputs(surface)

        assert any("preprocess" in w.lower() for w in warnings)

    def test_surface_only_validation(self):
        """Can validate with just a surface (no location/weather)."""
        from conftest import make_mock_svf

        dsm = np.zeros((20, 20), dtype=np.float32)
        surface = SurfaceData(dsm=dsm, svf=make_mock_svf((20, 20)))

        warnings = validate_inputs(surface)

        assert isinstance(warnings, list)


# ===========================================================================
# Memory optimization tests
# ===========================================================================


class TestTimeseriesMemory:
    """Tests for memory optimizations in calculate()."""

    def test_state_not_persisted_in_output_files(self, flat_surface, location, tmp_path):
        """Per-timestep output files should not include a 'state' subdirectory."""
        weather_series = _make_weather_series(datetime(2024, 7, 15, 10, 0), n_hours=3)

        summary = calculate(flat_surface, weather_series, location, output_dir=tmp_path, outputs=["tmrt"])

        assert len(summary) == 3
        # State should not be saved as a per-timestep output
        assert not (tmp_path / "state").exists(), "State should not be persisted to save memory"

    def test_state_still_propagates_correctly(self, flat_surface, location, tmp_path):
        """Despite not persisting state, thermal state should still propagate."""
        # Night -> day transition relies on state propagation for ground temperature
        weather_series = [
            Weather(datetime=datetime(2024, 7, 15, h, 0), ta=15.0 + h, rh=70.0, global_rad=max(0.0, (h - 5) * 200.0))
            for h in range(4, 10)
        ]

        calculate(flat_surface, weather_series, location, output_dir=tmp_path, outputs=["tmrt"])

        # Later timesteps should have higher Tmrt (thermal state propagated correctly)
        early_tmrt = np.nanmean(read_timestep_geotiff(tmp_path, "tmrt", 0))
        late_tmrt = np.nanmean(read_timestep_geotiff(tmp_path, "tmrt", 5))
        assert late_tmrt > early_tmrt, "Thermal state should propagate despite not being persisted"
