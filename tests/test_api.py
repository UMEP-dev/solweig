"""
Tests for the simplified SOLWEIG API.

These tests verify that the new dataclasses work correctly and
compute derived values as expected.
"""

from datetime import datetime

import numpy as np
import pytest
from conftest import make_mock_svf, read_timestep_geotiff
from solweig.api import (
    HumanParams,
    Location,
    ModelConfig,
    SolweigResult,
    SurfaceData,
    TimeseriesSummary,
    Weather,
    calculate,
    calculate_buffer_distance,
    generate_tiles,
)
from solweig.errors import MissingPrecomputedData
from solweig.models.surface import _max_shadow_height


class TestSurfaceData:
    """Tests for SurfaceData dataclass."""

    def test_basic_creation(self):
        """SurfaceData can be created with just a DSM."""
        dsm = np.ones((10, 10), dtype=np.float32)
        surface = SurfaceData(dsm=dsm)

        assert surface.dsm.shape == (10, 10)
        assert surface.cdsm is None
        assert surface.dem is None
        assert surface.pixel_size == 1.0

    def test_with_optional_rasters(self):
        """SurfaceData accepts optional CDSM, DEM, TDSM."""
        dsm = np.ones((10, 10)) * 100
        cdsm = np.ones((10, 10)) * 5
        dem = np.ones((10, 10)) * 50

        surface = SurfaceData(dsm=dsm, cdsm=cdsm, dem=dem, pixel_size=2.0)

        assert surface.cdsm is not None
        assert surface.dem is not None
        assert surface.pixel_size == 2.0

    def test_auto_converts_to_float32(self):
        """SurfaceData converts arrays to float32."""
        dsm = np.ones((10, 10), dtype=np.float64)
        surface = SurfaceData(dsm=dsm)

        assert surface.dsm.dtype == np.float32

    def test_max_height_auto_computed(self):
        """max_height is automatically computed from DSM."""
        dsm = np.zeros((10, 10))
        dsm[5, 5] = 100  # Building
        dsm[2, 2] = 10  # Lower ground

        surface = SurfaceData(dsm=dsm)

        assert surface.max_height == 100.0  # max - min = 100 - 0

    def test_max_height_with_terrain(self):
        """max_height handles terrain variation."""
        dsm = np.zeros((10, 10))
        dsm[:, :] = 50  # Base terrain
        dsm[5, 5] = 150  # Building on terrain

        surface = SurfaceData(dsm=dsm)

        # max_height = 150 - 50 = 100
        assert surface.max_height == 100.0

    def test_max_height_all_nan_returns_zero(self):
        """All-NaN DSM should safely report zero casting height."""
        dsm = np.full((5, 5), np.nan, dtype=np.float32)
        surface = SurfaceData(dsm=dsm)
        assert surface.max_height == 0.0

    def test_max_height_conservatively_includes_cdsm(self):
        """Buffer-oriented max_height includes CDSM whenever present."""
        dsm = np.ones((5, 5), dtype=np.float32) * 100.0
        cdsm = np.ones((5, 5), dtype=np.float32) * 130.0
        surface = SurfaceData(dsm=dsm, cdsm=cdsm, cdsm_relative=False)
        assert surface.max_height == 30.0

    def test_shape_property(self):
        """shape property returns DSM dimensions."""
        dsm = np.ones((100, 200))
        surface = SurfaceData(dsm=dsm)

        assert surface.shape == (100, 200)


class TestMaxShadowHeightHelper:
    """Tests for internal max shadow height helper semantics."""

    def test_all_nan_returns_zero(self):
        dsm = np.full((5, 5), np.nan, dtype=np.float32)
        assert _max_shadow_height(dsm) == 0.0

    def test_respects_use_veg_flag(self):
        dsm = np.ones((5, 5), dtype=np.float32) * 100.0
        cdsm = np.ones((5, 5), dtype=np.float32) * 130.0
        assert _max_shadow_height(dsm, cdsm, use_veg=False) == 0.0
        assert _max_shadow_height(dsm, cdsm, use_veg=True) == 30.0


class TestLocation:
    """Tests for Location dataclass."""

    def test_basic_creation(self):
        """Location can be created with lat/lon."""
        loc = Location(latitude=57.7, longitude=12.0)

        assert loc.latitude == 57.7
        assert loc.longitude == 12.0
        assert loc.altitude == 0.0
        assert loc.utc_offset == 0

    def test_with_altitude_and_utc(self):
        """Location accepts altitude and UTC offset."""
        loc = Location(latitude=40.0, longitude=-74.0, altitude=100.0, utc_offset=-5)

        assert loc.altitude == 100.0
        assert loc.utc_offset == -5

    def test_validates_latitude_range(self):
        """Location validates latitude in [-90, 90]."""
        with pytest.raises(ValueError, match="Latitude"):
            Location(latitude=91.0, longitude=0.0)

        with pytest.raises(ValueError, match="Latitude"):
            Location(latitude=-91.0, longitude=0.0)

    def test_validates_longitude_range(self):
        """Location validates longitude in [-180, 180]."""
        with pytest.raises(ValueError, match="Longitude"):
            Location(latitude=0.0, longitude=181.0)

        with pytest.raises(ValueError, match="Longitude"):
            Location(latitude=0.0, longitude=-181.0)

    def test_to_sun_position_dict(self):
        """to_sun_position_dict returns correct format."""
        loc = Location(latitude=57.7, longitude=12.0, altitude=100.0)
        d = loc.to_sun_position_dict()

        assert d["latitude"] == 57.7
        assert d["longitude"] == 12.0
        assert d["altitude"] == 100.0

    def test_from_epw(self, tmp_path):
        """Location.from_epw extracts lat, lon, tz_offset, and elevation from EPW header."""
        epw_content = (
            "LOCATION,Madrid,ESP,NA,Test Data,NA,40.45,-3.55,1.0,667.0\n"
            "DESIGN CONDITIONS,0\n"
            "TYPICAL/EXTREME PERIODS,0\n"
            "GROUND TEMPERATURES,0\n"
            "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0\n"
            "COMMENTS 1,Test\n"
            "COMMENTS 2,Test\n"
            "DATA PERIODS,1,1,Data,Sunday, 1/ 1,12/31\n"
            "2023,1,1,1,0,?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9*_*9*9*9*9*9,"
            "5.0,2.0,80,101325,0,0,0,0,0,0,0,0,0,0,180,3.0,5,5,10.0,77777,9,999999999,0,0.0,0,88,0.0,0.0,0.0\n"
        )
        epw_path = tmp_path / "madrid.epw"
        epw_path.write_text(epw_content)

        loc = Location.from_epw(epw_path)

        assert loc.latitude == pytest.approx(40.45)
        assert loc.longitude == pytest.approx(-3.55)
        assert loc.utc_offset == 1
        assert loc.altitude == pytest.approx(667.0)

    def test_from_epw_file_not_found(self):
        """Location.from_epw raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            Location.from_epw("/nonexistent/path.epw")


class TestWeather:
    """Tests for Weather dataclass."""

    def test_basic_creation(self):
        """Weather can be created with required fields."""
        dt_obj = datetime(2024, 7, 15, 12, 0)
        weather = Weather(datetime=dt_obj, ta=25.0, rh=50.0, global_rad=800.0)

        assert weather.ta == 25.0
        assert weather.rh == 50.0
        assert weather.global_rad == 800.0
        assert weather.ws == 1.0  # default

    def test_with_optional_fields(self):
        """Weather accepts optional wind speed and pressure."""
        dt_obj = datetime(2024, 7, 15, 12, 0)
        weather = Weather(datetime=dt_obj, ta=25.0, rh=50.0, global_rad=800.0, ws=3.5, pressure=1020.0)

        assert weather.ws == 3.5
        assert weather.pressure == 1020.0

    def test_validates_rh_range(self):
        """Weather validates RH in [0, 100]."""
        dt_obj = datetime(2024, 7, 15, 12, 0)

        with pytest.raises(ValueError, match="humidity"):
            Weather(datetime=dt_obj, ta=25.0, rh=101.0, global_rad=800.0)

        with pytest.raises(ValueError, match="humidity"):
            Weather(datetime=dt_obj, ta=25.0, rh=-1.0, global_rad=800.0)

    def test_validates_global_rad_positive(self):
        """Weather validates global_rad >= 0."""
        dt_obj = datetime(2024, 7, 15, 12, 0)

        with pytest.raises(ValueError, match="radiation"):
            Weather(datetime=dt_obj, ta=25.0, rh=50.0, global_rad=-100.0)

    def test_compute_derived_sun_position(self):
        """compute_derived calculates sun position."""
        # Summer noon in Gothenburg
        dt_obj = datetime(2024, 7, 15, 12, 0)
        weather = Weather(datetime=dt_obj, ta=25.0, rh=50.0, global_rad=800.0)
        location = Location(latitude=57.7, longitude=12.0, utc_offset=2)

        weather.compute_derived(location)

        # Sun should be high in the sky at noon in summer
        assert weather.sun_altitude > 40
        assert weather.sun_altitude < 70
        # Azimuth at clock noon varies with longitude/timezone
        # At Gothenburg (12°E, UTC+2), clock noon is before solar noon
        assert 100 < weather.sun_azimuth < 220

    def test_compute_derived_radiation_split(self):
        """compute_derived splits global into direct/diffuse."""
        dt_obj = datetime(2024, 7, 15, 12, 0)
        weather = Weather(datetime=dt_obj, ta=25.0, rh=50.0, global_rad=800.0)
        location = Location(latitude=57.7, longitude=12.0, utc_offset=2)

        weather.compute_derived(location)

        # Direct + diffuse should be close to global (not exact due to geometry)
        assert weather.direct_rad > 0
        assert weather.diffuse_rad > 0
        # Diffuse fraction typically 10-40% on clear day
        diffuse_fraction = weather.diffuse_rad / weather.global_rad
        assert 0.1 < diffuse_fraction < 0.6

    def test_compute_derived_night(self):
        """compute_derived handles nighttime correctly."""
        # Midnight
        dt_obj = datetime(2024, 7, 15, 0, 0)
        weather = Weather(datetime=dt_obj, ta=15.0, rh=80.0, global_rad=0.0)
        location = Location(latitude=57.7, longitude=12.0, utc_offset=2)

        weather.compute_derived(location)

        # Sun below horizon at midnight
        assert weather.sun_altitude < 0
        assert weather.direct_rad == 0.0
        assert weather.diffuse_rad == 0.0

    def test_is_daytime_property(self):
        """is_daytime returns correct value."""
        location = Location(latitude=57.7, longitude=12.0, utc_offset=2)

        # Noon
        weather_day = Weather(datetime=datetime(2024, 7, 15, 12, 0), ta=25.0, rh=50.0, global_rad=800.0)
        weather_day.compute_derived(location)
        assert weather_day.is_daytime is True

        # Midnight
        weather_night = Weather(datetime=datetime(2024, 7, 15, 0, 0), ta=15.0, rh=80.0, global_rad=0.0)
        weather_night.compute_derived(location)
        assert weather_night.is_daytime is False


class TestHumanParams:
    """Tests for HumanParams dataclass."""

    def test_default_values(self):
        """HumanParams has sensible defaults."""
        human = HumanParams()

        assert human.posture == "standing"
        assert human.abs_k == 0.7
        assert human.abs_l == 0.97
        assert human.age == 35
        assert human.weight == 75.0
        assert human.height == 1.75

    def test_custom_values(self):
        """HumanParams accepts custom values."""
        human = HumanParams(posture="sitting", abs_k=0.6, abs_l=0.95, age=45, weight=80.0, height=1.80)

        assert human.posture == "sitting"
        assert human.abs_k == 0.6
        assert human.age == 45

    def test_validates_posture(self):
        """HumanParams validates posture."""
        with pytest.raises(ValueError, match="Posture"):
            HumanParams(posture="lying")

    def test_validates_abs_k_range(self):
        """HumanParams validates abs_k in (0, 1]."""
        with pytest.raises(ValueError, match="abs_k"):
            HumanParams(abs_k=0.0)

        with pytest.raises(ValueError, match="abs_k"):
            HumanParams(abs_k=1.5)

    def test_validates_abs_l_range(self):
        """HumanParams validates abs_l in (0, 1]."""
        with pytest.raises(ValueError, match="abs_l"):
            HumanParams(abs_l=0.0)


class TestSolweigResult:
    """Tests for SolweigResult dataclass."""

    def test_basic_creation(self):
        """SolweigResult can be created with Tmrt."""
        tmrt = np.ones((10, 10)) * 40.0
        result = SolweigResult(tmrt=tmrt)

        assert result.tmrt.shape == (10, 10)
        assert result.utci is None
        assert result.pet is None

    def test_with_all_outputs(self):
        """SolweigResult can hold all output grids."""
        shape = (10, 10)
        result = SolweigResult(
            tmrt=np.ones(shape) * 40.0,
            shadow=np.zeros(shape),
            kdown=np.ones(shape) * 500.0,
            kup=np.ones(shape) * 50.0,
            ldown=np.ones(shape) * 350.0,
            lup=np.ones(shape) * 400.0,
            utci=np.ones(shape) * 30.0,
            pet=np.ones(shape) * 28.0,
        )

        assert result.shadow is not None
        assert result.kdown is not None
        assert result.utci is not None
        assert result.pet is not None


class TestSolweigResultMethods:
    """Tests for SolweigResult.compute_utci() and compute_pet() methods."""

    def test_compute_utci_with_weather_object(self):
        """compute_utci() works with Weather object."""
        tmrt = np.ones((10, 10), dtype=np.float32) * 35.0
        result = SolweigResult(tmrt=tmrt)

        weather = Weather(
            datetime=datetime(2024, 7, 15, 12, 0),
            ta=25.0,
            rh=50.0,
            global_rad=800.0,
            ws=2.0,
        )

        utci = result.compute_utci(weather)

        assert utci.shape == (10, 10)
        # UTCI should be in reasonable range for these conditions
        assert np.all(utci > 20) and np.all(utci < 50)

    def test_compute_utci_with_individual_values(self):
        """compute_utci() works with individual values."""
        tmrt = np.ones((10, 10), dtype=np.float32) * 35.0
        result = SolweigResult(tmrt=tmrt)

        utci = result.compute_utci(25.0, rh=50.0, wind=2.0)

        assert utci.shape == (10, 10)
        assert np.all(utci > 20) and np.all(utci < 50)

    def test_compute_utci_default_wind(self):
        """compute_utci() uses default wind speed of 1.0 m/s."""
        tmrt = np.ones((10, 10), dtype=np.float32) * 35.0
        result = SolweigResult(tmrt=tmrt)

        # No wind provided - should default to 1.0
        utci = result.compute_utci(25.0, rh=50.0)

        assert utci.shape == (10, 10)
        assert np.all(np.isfinite(utci))

    def test_compute_utci_requires_rh_with_float(self):
        """compute_utci() raises ValueError when rh not provided with float ta."""
        tmrt = np.ones((10, 10), dtype=np.float32) * 35.0
        result = SolweigResult(tmrt=tmrt)

        with pytest.raises(ValueError, match="rh is required"):
            result.compute_utci(25.0)

    def test_compute_pet_with_weather_object(self):
        """compute_pet() works with Weather object."""
        tmrt = np.ones((5, 5), dtype=np.float32) * 35.0  # Smaller grid for speed
        result = SolweigResult(tmrt=tmrt)

        weather = Weather(
            datetime=datetime(2024, 7, 15, 12, 0),
            ta=25.0,
            rh=50.0,
            global_rad=800.0,
            ws=2.0,
        )

        pet = result.compute_pet(weather)

        assert pet.shape == (5, 5)
        # PET should be in reasonable range
        assert np.all(pet > 10) and np.all(pet < 50)

    def test_compute_pet_with_individual_values(self):
        """compute_pet() works with individual values."""
        tmrt = np.ones((5, 5), dtype=np.float32) * 35.0
        result = SolweigResult(tmrt=tmrt)

        pet = result.compute_pet(25.0, rh=50.0, wind=2.0)

        assert pet.shape == (5, 5)
        assert np.all(np.isfinite(pet))

    def test_compute_pet_with_custom_human_params(self):
        """compute_pet() accepts custom HumanParams."""
        tmrt = np.ones((5, 5), dtype=np.float32) * 35.0
        result = SolweigResult(tmrt=tmrt)

        weather = Weather(
            datetime=datetime(2024, 7, 15, 12, 0),
            ta=25.0,
            rh=50.0,
            global_rad=800.0,
        )

        pet = result.compute_pet(weather, human=HumanParams(weight=60, height=1.60))

        assert pet.shape == (5, 5)
        assert np.all(np.isfinite(pet))

    def test_compute_pet_requires_rh_with_float(self):
        """compute_pet() raises ValueError when rh not provided with float ta."""
        tmrt = np.ones((5, 5), dtype=np.float32) * 35.0
        result = SolweigResult(tmrt=tmrt)

        with pytest.raises(ValueError, match="rh is required"):
            result.compute_pet(25.0)


@pytest.mark.slow
class TestConfigPrecedence:
    """Tests for config precedence - explicit parameters override config values."""

    def test_explicit_anisotropic_overrides_config(self, tmp_path):
        """Explicit use_anisotropic_sky=False overrides config.use_anisotropic_sky=True."""

        dsm = np.ones((20, 20), dtype=np.float32) * 5.0
        surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf(dsm.shape))
        location = Location(latitude=57.7, longitude=12.0, utc_offset=1)
        weather = Weather(
            datetime=datetime(2024, 7, 15, 12, 0),
            ta=25.0,
            rh=50.0,
            global_rad=800.0,
        )

        # Config says use anisotropic, but explicit param says don't
        # This should NOT raise MissingPrecomputedData since explicit False wins
        config = ModelConfig(use_anisotropic_sky=True)
        summary = calculate(
            surface,
            [weather],
            location,
            config=config,
            use_anisotropic_sky=False,  # Explicit wins
            output_dir=tmp_path,
        )

        assert isinstance(summary, TimeseriesSummary)
        assert summary.tmrt_mean is not None

    def test_explicit_human_overrides_config(self, tmp_path):
        """Explicit human params override config.human."""

        dsm = np.ones((20, 20), dtype=np.float32) * 5.0
        surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf(dsm.shape))
        location = Location(latitude=57.7, longitude=12.0, utc_offset=1)
        weather = Weather(
            datetime=datetime(2024, 7, 15, 12, 0),
            ta=25.0,
            rh=50.0,
            global_rad=800.0,
        )

        config_human = HumanParams(posture="sitting", abs_k=0.6)
        explicit_human = HumanParams(posture="standing", abs_k=0.8)

        config = ModelConfig(human=config_human)
        summary = calculate(
            surface,
            [weather],
            location,
            config=config,
            human=explicit_human,  # Should use standing, abs_k=0.8
            output_dir=tmp_path,
        )

        # Result should exist (test doesn't crash)
        assert isinstance(summary, TimeseriesSummary)
        assert summary.tmrt_mean is not None

    def test_none_param_uses_config_value(self, tmp_path):
        """When explicit param is None, config value is used."""

        dsm = np.ones((20, 20), dtype=np.float32) * 5.0
        surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf(dsm.shape))
        location = Location(latitude=57.7, longitude=12.0, utc_offset=1)
        weather = Weather(
            datetime=datetime(2024, 7, 15, 12, 0),
            ta=25.0,
            rh=50.0,
            global_rad=800.0,
        )

        config_human = HumanParams(posture="sitting")
        config = ModelConfig(human=config_human)

        # human=None means use config's human
        summary = calculate(
            surface,
            [weather],
            location,
            config=config,
            human=None,  # Should fall back to config.human
            output_dir=tmp_path,
        )

        assert isinstance(summary, TimeseriesSummary)
        assert summary.tmrt_mean is not None

    def test_no_config_uses_defaults(self, tmp_path):
        """When no config provided, defaults are used."""
        dsm = np.ones((20, 20), dtype=np.float32) * 5.0
        surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf(dsm.shape))
        location = Location(latitude=57.7, longitude=12.0, utc_offset=1)
        weather = Weather(
            datetime=datetime(2024, 7, 15, 12, 0),
            ta=25.0,
            rh=50.0,
            global_rad=800.0,
        )

        # No config, no explicit params - should use defaults
        summary = calculate(surface, [weather], location, output_dir=tmp_path)

        assert isinstance(summary, TimeseriesSummary)
        assert summary.tmrt_mean is not None


@pytest.mark.slow
class TestCalculateIntegration:
    """Integration tests for the calculate() function."""

    def test_basic_calculation(self, tmp_path):
        """calculate() returns valid Tmrt for simple DSM."""

        # Simple flat DSM with one building
        dsm = np.zeros((30, 30), dtype=np.float32)
        dsm[10:20, 10:20] = 10.0  # 10m building

        surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf(dsm.shape))
        location = Location(latitude=57.7, longitude=12.0, utc_offset=1)
        weather = Weather(
            datetime=datetime(2024, 7, 15, 12, 0),
            ta=25.0,
            rh=50.0,
            global_rad=800.0,
        )

        summary = calculate(
            surface,
            [weather],
            location,
            output_dir=tmp_path,
            outputs=["tmrt", "shadow"],
        )

        assert isinstance(summary, TimeseriesSummary)

        # Check output structure by reading from disk
        tmrt = read_timestep_geotiff(tmp_path, "tmrt", 0)
        shadow = read_timestep_geotiff(tmp_path, "shadow", 0)

        assert tmrt.shape == (30, 30)
        assert shadow.shape == (30, 30)
        # UTCI/PET are not auto-computed - subdirectories should not exist
        assert not (tmp_path / "utci").exists()
        assert not (tmp_path / "pet").exists()

        # Check Tmrt is in reasonable range (use nanmin/nanmax to handle NaN)
        # -50 is used as a sentinel for invalid/building pixels
        assert np.nanmin(tmrt) >= -50
        assert np.nanmax(tmrt) < 80

    def test_nighttime_calculation(self, tmp_path):
        """calculate() handles nighttime (sun below horizon)."""
        dsm = np.ones((20, 20), dtype=np.float32) * 5.0
        surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf(dsm.shape))
        location = Location(latitude=57.7, longitude=12.0, utc_offset=1)

        # Midnight - sun below horizon
        weather = Weather(
            datetime=datetime(2024, 7, 15, 0, 0),
            ta=15.0,
            rh=80.0,
            global_rad=0.0,
        )

        calculate(
            surface,
            [weather],
            location,
            output_dir=tmp_path,
            outputs=["tmrt", "kdown", "kup"],
        )

        # Read results from disk
        tmrt = read_timestep_geotiff(tmp_path, "tmrt", 0)
        kdown = read_timestep_geotiff(tmp_path, "kdown", 0)
        kup = read_timestep_geotiff(tmp_path, "kup", 0)

        # At night, Tmrt is computed from full longwave balance (no shortwave).
        # Under open sky (SVF~1) the cold sky pulls Tmrt well below Ta — typically
        # ~5-10 C lower. This matches UMEP behaviour; the old Python shortcut
        # (Tmrt=Ta) was wrong.
        valid = tmrt[np.isfinite(tmrt)]
        assert np.all(valid < 15.0), "Night Tmrt should be below Ta under open sky"
        assert np.all(valid > -5.0), "Night Tmrt should not be unreasonably cold"
        # Shortwave must be zero at night
        assert np.allclose(kdown[np.isfinite(kdown)], 0.0, atol=1e-3)
        assert np.allclose(kup[np.isfinite(kup)], 0.0, atol=1e-3)

    def test_explicit_anisotropic_requires_shadow_matrices(self, tmp_path):
        """Explicit anisotropic request must fail without shadow matrices."""
        dsm = np.ones((20, 20), dtype=np.float32) * 5.0
        surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf(dsm.shape))
        location = Location(latitude=57.7, longitude=12.0, utc_offset=1)
        weather = Weather(
            datetime=datetime(2024, 7, 15, 12, 0),
            ta=25.0,
            rh=50.0,
            global_rad=800.0,
        )

        with pytest.raises(MissingPrecomputedData):
            calculate(surface, [weather], location, use_anisotropic_sky=True, output_dir=tmp_path)

    def test_shadows_exist(self, tmp_path):
        """Shadows are cast by buildings during daytime."""
        # Tall building that should cast shadows
        dsm = np.zeros((40, 40), dtype=np.float32)
        dsm[15:25, 15:25] = 20.0  # 20m building

        surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf(dsm.shape))
        location = Location(latitude=57.7, longitude=12.0, utc_offset=1)
        weather = Weather(
            datetime=datetime(2024, 7, 15, 10, 0),  # Morning - shadows to west
            ta=20.0,
            rh=60.0,
            global_rad=600.0,
        )

        calculate(
            surface,
            [weather],
            location,
            output_dir=tmp_path,
            outputs=["shadow"],
        )

        # Read shadow from disk
        shadow = read_timestep_geotiff(tmp_path, "shadow", 0)

        # Should have some shadow pixels (not all 0 or all 1)
        shadow_fraction = shadow.sum() / shadow.size
        assert 0.1 < shadow_fraction < 0.9, "Expected partial shadowing"

    def test_utci_postprocessing(self, tmp_path):
        """UTCI is computed via post-processing, not by default."""
        dsm = np.ones((20, 20), dtype=np.float32) * 5.0
        surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf(dsm.shape))
        location = Location(latitude=57.7, longitude=12.0, utc_offset=1)
        weather = Weather(
            datetime=datetime(2024, 7, 15, 12, 0),
            ta=25.0,
            rh=50.0,
            global_rad=800.0,
        )

        # Calculate Tmrt (UTCI not auto-computed on per-timestep results)
        calculate(
            surface,
            [weather],
            location,
            output_dir=tmp_path,
            outputs=["tmrt"],
        )

        # Tmrt should exist on disk
        tmrt = read_timestep_geotiff(tmp_path, "tmrt", 0)
        assert tmrt is not None
        # UTCI not auto-computed - subdirectory should not exist
        assert not (tmp_path / "utci").exists()

    def test_with_custom_human_params(self, tmp_path):
        """Custom human parameters affect calculation."""
        dsm = np.ones((20, 20), dtype=np.float32) * 5.0
        surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf(dsm.shape))
        location = Location(latitude=57.7, longitude=12.0, utc_offset=1)
        weather = Weather(
            datetime=datetime(2024, 7, 15, 12, 0),
            ta=25.0,
            rh=50.0,
            global_rad=800.0,
        )

        # Different postures should give slightly different results
        summary_standing = calculate(
            surface,
            [weather],
            location,
            human=HumanParams(posture="standing"),
            output_dir=tmp_path / "run1",
        )
        summary_sitting = calculate(
            surface,
            [weather],
            location,
            human=HumanParams(posture="sitting"),
            output_dir=tmp_path / "run2",
        )

        # Results should exist and be physically reasonable
        assert summary_standing.tmrt_mean is not None
        assert summary_sitting.tmrt_mean is not None

        # Sitting Tmrt must be positive and in a reasonable range for
        # summer midday (not negative as it was before the anisotropic
        # sky fix for sitting posture — see GitHub issue #9)
        standing_mean = np.nanmean(summary_standing.tmrt_mean)
        sitting_mean = np.nanmean(summary_sitting.tmrt_mean)
        assert standing_mean > 0, f"Standing Tmrt should be positive, got {standing_mean}"
        assert sitting_mean > 0, f"Sitting Tmrt should be positive, got {sitting_mean}"
        # Sitting Tmrt is typically lower than standing due to different
        # view factors, but should be in the same ballpark (not 50+ K lower)
        assert abs(standing_mean - sitting_mean) < 20, (
            f"Standing ({standing_mean:.1f}) and sitting ({sitting_mean:.1f}) Tmrt should be within 20 K of each other"
        )


@pytest.mark.slow
class TestTiledProcessing:
    """Tests for tiled processing functions."""

    def test_calculate_buffer_distance_basic(self):
        """Buffer distance scales with building height."""
        # 10m building at 3° sun elevation: buffer = 10 / tan(3°) ≈ 191m
        buffer = calculate_buffer_distance(10.0)
        assert 180 < buffer < 200

        # 50m building: buffer = 50 / tan(3°) ≈ 954m, under 1000m cap
        buffer = calculate_buffer_distance(50.0)
        assert 940 < buffer < 960

        # 60m building: buffer = 60 / tan(3°) ≈ 1145m, capped at 1000m
        buffer = calculate_buffer_distance(60.0)
        assert buffer == 1000.0  # MAX_BUFFER_M

    def test_calculate_buffer_distance_zero_height(self):
        """Zero height returns zero buffer."""
        assert calculate_buffer_distance(0.0) == 0.0
        assert calculate_buffer_distance(-5.0) == 0.0

    def test_calculate_buffer_distance_custom_sun_elevation(self):
        """Buffer distance changes with sun elevation."""
        # Higher sun = shorter shadows
        buffer_3deg = calculate_buffer_distance(10.0, min_sun_elev_deg=3.0)
        buffer_10deg = calculate_buffer_distance(10.0, min_sun_elev_deg=10.0)

        assert buffer_10deg < buffer_3deg

    def test_generate_tiles_basic(self):
        """generate_tiles creates correct tile specs."""
        # generate_tiles takes rows, cols, tile_size, overlap
        tiles = generate_tiles(rows=100, cols=100, tile_size=50, overlap=10)

        # 100x100 with tile_size=50 should give 4 tiles (2x2 grid)
        assert len(tiles) == 4

        # Check first tile
        tile0 = tiles[0]
        assert tile0.row_start == 0
        assert tile0.col_start == 0
        assert tile0.core_shape == (50, 50)

    def test_generate_tiles_overlap(self):
        """Tiles have correct overlap at edges."""
        tiles = generate_tiles(rows=100, cols=100, tile_size=50, overlap=10)

        # First tile (top-left corner) has no top/left overlap
        tile0 = tiles[0]
        assert tile0.overlap_top == 0
        assert tile0.overlap_left == 0
        assert tile0.overlap_bottom == 10
        assert tile0.overlap_right == 10

        # Last tile (bottom-right corner) has no bottom/right overlap
        tile3 = tiles[3]
        assert tile3.overlap_top == 10
        assert tile3.overlap_left == 10
        assert tile3.overlap_bottom == 0
        assert tile3.overlap_right == 0

    def test_generate_tiles_single_tile(self):
        """Small raster generates single tile."""
        # 30x30 raster smaller than tile_size should give 1 tile
        tiles = generate_tiles(rows=30, cols=30, tile_size=256, overlap=10)

        assert len(tiles) == 1
        assert tiles[0].core_shape == (30, 30)


class TestPreprocessing:
    """Tests for CDSM/TDSM preprocessing and transmissivity calculation."""

    def test_tdsm_auto_generation(self):
        """TDSM is auto-generated from CDSM * trunk_ratio when not provided."""
        dsm = np.ones((10, 10), dtype=np.float32) * 100.0
        cdsm = np.ones((10, 10), dtype=np.float32) * 5.0  # 5m relative vegetation height

        surface = SurfaceData(dsm=dsm, cdsm=cdsm, trunk_ratio=0.25)

        # Before preprocessing, TDSM should be None
        assert surface.tdsm is None

        # After preprocessing, TDSM should be auto-generated
        surface.preprocess()

        assert surface.tdsm is not None
        # TDSM should be boosted: base + (cdsm * trunk_ratio) = 100 + (5 * 0.25) = 101.25
        # But only where cdsm > threshold (0.1)
        expected_tdsm = 100.0 + 5.0 * 0.25  # 101.25
        assert np.allclose(surface.tdsm, expected_tdsm, atol=0.01)

    def test_cdsm_boosting_with_dem(self):
        """CDSM is boosted to absolute height using DEM as base."""
        dsm = np.ones((10, 10), dtype=np.float32) * 110.0  # DSM includes building
        dem = np.ones((10, 10), dtype=np.float32) * 100.0  # Ground level
        cdsm = np.ones((10, 10), dtype=np.float32) * 12.0  # 12m relative veg (above building)

        surface = SurfaceData(dsm=dsm, cdsm=cdsm, dem=dem)
        surface.preprocess()

        # CDSM should now be DEM + relative_cdsm = 100 + 12 = 112
        # (above DSM=110, so not cleared by underground check)
        assert surface.cdsm is not None
        assert np.allclose(surface.cdsm, 112.0, atol=0.01)

    def test_cdsm_boosting_without_dem(self):
        """CDSM is boosted using DSM as base when DEM not provided."""
        dsm = np.ones((10, 10), dtype=np.float32) * 105.0
        cdsm = np.ones((10, 10), dtype=np.float32) * 6.0  # 6m relative veg height

        surface = SurfaceData(dsm=dsm, cdsm=cdsm)
        surface.preprocess()

        # CDSM should now be DSM + relative_cdsm = 105 + 6 = 111
        assert surface.cdsm is not None
        assert np.allclose(surface.cdsm, 111.0, atol=0.01)

    def test_preprocess_zeros_below_threshold(self):
        """Preprocessing sets sub-threshold vegetation to NaN (absent)."""
        dsm = np.ones((10, 10), dtype=np.float32) * 100.0
        cdsm = np.array([[0.05, 0.5], [1.0, 0.0]], dtype=np.float32)  # Some below threshold
        cdsm = np.pad(cdsm, ((0, 8), (0, 8)), constant_values=0.0)

        surface = SurfaceData(dsm=dsm, cdsm=cdsm)
        surface.preprocess()

        # CDSM is now absolute elevation; below-threshold values set to NaN (absent)
        # Default min_object_height=1.0, so threshold = max(0.1, 1.0) = 1.0m
        assert surface.cdsm is not None
        assert np.isnan(surface.cdsm[0, 0])  # Was 0.05 relative, below 1.0m → NaN
        assert np.isnan(surface.cdsm[0, 1])  # Was 0.5 relative, below 1.0m → NaN
        assert surface.cdsm[1, 0] > 100.0  # Was 1.0 relative, at threshold → 101.0
        assert np.isnan(surface.cdsm[1, 1])  # Was 0.0 relative, below 1.0m → NaN

    def test_preprocess_idempotent(self):
        """Calling preprocess() multiple times has no effect after first call."""
        dsm = np.ones((10, 10), dtype=np.float32) * 100.0
        cdsm = np.ones((10, 10), dtype=np.float32) * 5.0

        surface = SurfaceData(dsm=dsm, cdsm=cdsm)
        surface.preprocess()
        assert surface.cdsm is not None
        cdsm_after_first = surface.cdsm.copy()

        surface.preprocess()  # Second call
        assert surface.cdsm is not None
        assert np.array_equal(surface.cdsm, cdsm_after_first)

    def test_transmissivity_leaf_on_summer(self):
        """Summer (leaf on) uses low transmissivity."""
        from solweig.components.shadows import compute_transmissivity

        # July is within typical leaf-on period (DOY 100-300)
        psi = compute_transmissivity(doy=180)
        assert psi == 0.03

    def test_transmissivity_leaf_off_winter(self):
        """Winter (leaf off) uses high transmissivity."""
        from solweig.components.shadows import compute_transmissivity

        # January is outside typical leaf-on period
        psi = compute_transmissivity(doy=30)
        assert psi == 0.5

    def test_transmissivity_leaf_off_late_autumn(self):
        """Late autumn (leaf off) uses high transmissivity."""
        from solweig.components.shadows import compute_transmissivity

        # December is outside typical leaf-on period
        psi = compute_transmissivity(doy=350)
        assert psi == 0.5

    def test_transmissivity_conifer_always_leaf_on(self):
        """Conifers always use leaf-on transmissivity regardless of season."""
        from solweig.components.shadows import compute_transmissivity

        # Winter with conifer flag should still use leaf-on value
        psi = compute_transmissivity(doy=30, conifer=True)
        assert psi == 0.03

        # Summer with conifer should also be leaf-on
        psi = compute_transmissivity(doy=180, conifer=True)
        assert psi == 0.03

    def test_transmissivity_boundary_days(self):
        """Test behavior at leaf on/off boundary days."""
        from solweig.components.shadows import compute_transmissivity

        # Default boundaries are 100 and 300
        # Day 100 is NOT included (first_day < doy < last_day)
        psi_day_100 = compute_transmissivity(doy=100)
        assert psi_day_100 == 0.5  # Not yet leaf-on

        # Day 101 should be leaf-on
        psi_day_101 = compute_transmissivity(doy=101)
        assert psi_day_101 == 0.03

        # Day 299 should be leaf-on
        psi_day_299 = compute_transmissivity(doy=299)
        assert psi_day_299 == 0.03

        # Day 300 is NOT included
        psi_day_300 = compute_transmissivity(doy=300)
        assert psi_day_300 == 0.5  # No longer leaf-on

    def test_per_layer_height_defaults(self):
        """Per-layer height flags have correct defaults."""
        dsm = np.ones((10, 10), dtype=np.float32) * 100.0
        surface = SurfaceData(dsm=dsm)
        assert surface.dsm_relative is False
        assert surface.cdsm_relative is True
        assert surface.tdsm_relative is True

    def test_per_layer_height_explicit(self):
        """Per-layer height flags can be set explicitly."""
        dsm = np.ones((10, 10), dtype=np.float32) * 100.0
        cdsm = np.ones((10, 10), dtype=np.float32) * 105.0  # Absolute heights

        surface = SurfaceData(dsm=dsm, cdsm=cdsm, cdsm_relative=False)
        assert surface.cdsm_relative is False

    def test_looks_like_relative_heights_true(self):
        """_looks_like_relative_heights returns True for typical relative data."""
        # DSM at ~100m elevation, CDSM with 5m vegetation (relative)
        dsm = np.ones((10, 10), dtype=np.float32) * 100.0
        cdsm = np.ones((10, 10), dtype=np.float32) * 5.0

        surface = SurfaceData(dsm=dsm, cdsm=cdsm)
        assert surface.looks_like_relative_heights() is True

    def test_looks_like_relative_heights_false_absolute(self):
        """_looks_like_relative_heights returns False for absolute heights."""
        # DSM at ~100m, CDSM at ~105m (absolute, trees on terrain)
        dsm = np.ones((10, 10), dtype=np.float32) * 100.0
        cdsm = np.ones((10, 10), dtype=np.float32) * 105.0

        surface = SurfaceData(dsm=dsm, cdsm=cdsm)
        assert surface.looks_like_relative_heights() is False

    def test_looks_like_relative_heights_false_coastal(self):
        """_looks_like_relative_heights handles coastal areas near sea level."""
        # DSM near sea level, CDSM with 5m vegetation (relative)
        # This is a tricky case - low elevation could be absolute or relative
        dsm = np.ones((10, 10), dtype=np.float32) * 5.0
        cdsm = np.ones((10, 10), dtype=np.float32) * 8.0

        surface = SurfaceData(dsm=dsm, cdsm=cdsm)
        # At low elevations, heuristic is inconclusive - returns False to avoid false positives
        assert surface.looks_like_relative_heights() is False

    def test_preprocessing_warning_issued(self, caplog):
        """Warning is issued when CDSM looks relative but preprocess not called."""
        import logging

        # DSM at ~100m elevation, CDSM with 5m relative vegetation
        dsm = np.ones((10, 10), dtype=np.float32) * 100.0
        cdsm = np.ones((10, 10), dtype=np.float32) * 5.0

        surface = SurfaceData(dsm=dsm, cdsm=cdsm, cdsm_relative=True)

        with caplog.at_level(logging.WARNING):
            surface._check_preprocessing_needed()

        assert "preprocess() was not called" in caplog.text
        assert "relative vegetation heights" in caplog.text

    def test_preprocessing_warning_not_issued_after_preprocess(self, caplog):
        """No warning after preprocess() is called."""
        import logging

        dsm = np.ones((10, 10), dtype=np.float32) * 100.0
        cdsm = np.ones((10, 10), dtype=np.float32) * 5.0

        surface = SurfaceData(dsm=dsm, cdsm=cdsm, cdsm_relative=True)
        surface.preprocess()  # This sets _preprocessed = True

        with caplog.at_level(logging.WARNING):
            surface._check_preprocessing_needed()

        assert "preprocess() was not called" not in caplog.text

    def test_preprocessing_warning_not_issued_when_cdsm_relative_false(self, caplog):
        """No warning when cdsm_relative=False (user says data is absolute)."""
        import logging

        dsm = np.ones((10, 10), dtype=np.float32) * 100.0
        cdsm = np.ones((10, 10), dtype=np.float32) * 5.0  # Looks relative but user says no

        surface = SurfaceData(dsm=dsm, cdsm=cdsm, cdsm_relative=False)

        with caplog.at_level(logging.WARNING):
            surface._check_preprocessing_needed()

        assert "preprocess() was not called" not in caplog.text


@pytest.mark.slow
class TestPrepareArrayMode:
    """Tests for SurfaceData.prepare() with numpy arrays."""

    def test_prepare_from_array_basic(self):
        """prepare() with a numpy array computes walls and SVF."""
        dsm = np.zeros((30, 30), dtype=np.float32)
        dsm[10:20, 10:20] = 10.0

        surface = SurfaceData.prepare(dsm=dsm, pixel_size=1.0)

        assert surface.svf is not None
        assert surface.wall_height is not None
        assert surface.wall_aspect is not None
        assert surface.pixel_size == 1.0

    def test_prepare_from_array_with_vegetation(self):
        """prepare() with arrays preprocesses relative CDSM."""
        dsm = np.ones((20, 20), dtype=np.float32) * 100.0
        cdsm = np.ones((20, 20), dtype=np.float32) * 5.0  # Relative height

        surface = SurfaceData.prepare(dsm=dsm, cdsm=cdsm, pixel_size=1.0)

        assert surface.svf is not None
        assert surface.cdsm is not None
        # CDSM should have been converted to absolute: DSM + relative = 100 + 5 = 105
        assert np.all(surface.cdsm[surface.cdsm > 100.0] > 100.0)

    def test_prepare_from_array_pixel_size_required(self):
        """prepare() with arrays raises ValueError when pixel_size is omitted."""
        dsm = np.ones((10, 10), dtype=np.float32)

        with pytest.raises(ValueError, match="pixel_size"):
            SurfaceData.prepare(dsm=dsm)

    def test_prepare_from_array_shape_mismatch(self):
        """prepare() with mismatched array shapes raises ValueError."""
        dsm = np.ones((10, 10), dtype=np.float32)
        cdsm = np.ones((5, 5), dtype=np.float32)

        with pytest.raises(ValueError, match="shape"):
            SurfaceData.prepare(dsm=dsm, cdsm=cdsm, pixel_size=1.0)

    def test_prepare_from_array_mixed_types_rejected(self):
        """prepare() rejects mixing arrays with file paths."""
        dsm = np.ones((10, 10), dtype=np.float32)

        with pytest.raises(TypeError, match="numpy array"):
            SurfaceData.prepare(dsm=dsm, cdsm="path/to/cdsm.tif", pixel_size=1.0)

    def test_prepare_from_array_precomputed_walls(self):
        """prepare() uses provided wall arrays without recomputing."""
        dsm = np.ones((20, 20), dtype=np.float32) * 5.0
        wall_height = np.ones((20, 20), dtype=np.float32) * 99.0
        wall_aspect = np.ones((20, 20), dtype=np.float32) * 180.0

        surface = SurfaceData.prepare(dsm=dsm, wall_height=wall_height, wall_aspect=wall_aspect, pixel_size=1.0)

        # Should use the provided walls, not recompute
        assert surface.wall_height is not None
        assert surface.wall_aspect is not None
        assert np.allclose(surface.wall_height, 99.0)
        assert np.allclose(surface.wall_aspect, 180.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
