"""
Tests for QGIS plugin converter functions.

Mocks QGIS and GDAL dependencies so these tests run without a QGIS installation.
Tests the pure logic: parameter dict -> solweig dataclass conversion.
"""

from __future__ import annotations

import contextlib
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import solweig

from tests.qgis_mocks import QgsProcessingException, install, install_osgeo, preserve_solweig_modules, uninstall_osgeo

install()  # Must be called before any qgis_plugin imports
install_osgeo()  # Temporarily needed for osgeo imports in converters.py

with preserve_solweig_modules():
    from solweig.geospatial import looks_like_relative as _looks_like_relative_heights  # noqa: E402
    from qgis_plugin.solweig_qgis.utils.converters import (  # noqa: E402
        create_human_params_from_parameters,
        create_location_from_parameters,
        create_surface_from_parameters,
        create_weather_from_parameters,
        load_prepared_surface,
        load_raster_from_layer,
        load_weather_from_epw,
    )

uninstall_osgeo()  # Clean up immediately after imports to avoid polluting other tests


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def feedback():
    """Create a mock QgsProcessingFeedback."""
    fb = MagicMock()
    fb.pushInfo = MagicMock()
    fb.reportError = MagicMock()
    return fb


# ---------------------------------------------------------------------------
# create_human_params_from_parameters
# ---------------------------------------------------------------------------


class TestCreateHumanParams:
    """Tests for create_human_params_from_parameters."""

    def test_defaults(self):
        """Default parameters produce standing posture with abs_k=0.7."""
        human = create_human_params_from_parameters({})
        assert human.posture == "standing"
        assert human.abs_k == 0.7

    def test_posture_standing(self):
        """Posture enum 0 -> 'standing'."""
        human = create_human_params_from_parameters({"POSTURE": 0})
        assert human.posture == "standing"

    def test_posture_sitting(self):
        """Posture enum 1 -> 'sitting'."""
        human = create_human_params_from_parameters({"POSTURE": 1})
        assert human.posture == "sitting"

    def test_posture_unknown_defaults_standing(self):
        """Unknown posture enum falls back to 'standing'."""
        human = create_human_params_from_parameters({"POSTURE": 99})
        assert human.posture == "standing"

    def test_custom_abs_k(self):
        """Custom absorption coefficient."""
        human = create_human_params_from_parameters({"ABS_K": 0.5})
        assert human.abs_k == 0.5

    def test_pet_body_params(self):
        """PET body parameters are set when provided."""
        params = {
            "WEIGHT": 80.0,
            "HEIGHT": 1.80,
            "AGE": 40,
            "ACTIVITY": 100.0,
            "CLOTHING": 1.2,
        }
        human = create_human_params_from_parameters(params)
        assert human.weight == 80.0
        assert human.height == 1.80
        assert human.age == 40
        assert human.activity == 100.0
        assert human.clothing == 1.2

    def test_sex_mapping_male(self):
        """QGIS enum 0 (Male) -> solweig sex 1."""
        human = create_human_params_from_parameters({"SEX": 0})
        assert human.sex == 1

    def test_sex_mapping_female(self):
        """QGIS enum 1 (Female) -> solweig sex 2."""
        human = create_human_params_from_parameters({"SEX": 1})
        assert human.sex == 2

    def test_sex_unknown_defaults_male(self):
        """Unknown sex enum defaults to male (1)."""
        human = create_human_params_from_parameters({"SEX": 99})
        assert human.sex == 1

    def test_returns_human_params_instance(self):
        """Result is a solweig.HumanParams dataclass."""
        human = create_human_params_from_parameters({})
        assert isinstance(human, solweig.HumanParams)

    def test_partial_pet_params(self):
        """Only provided PET params are set, others keep defaults."""
        human = create_human_params_from_parameters({"WEIGHT": 90.0})
        assert human.weight == 90.0
        assert human.height == 1.75  # default
        assert human.age == 35  # default


# ---------------------------------------------------------------------------
# create_weather_from_parameters
# ---------------------------------------------------------------------------


def _make_qdt(dt_obj: datetime):
    """Create a mock QDateTime that returns the given datetime."""
    qdt = MagicMock()
    qdt.toPyDateTime.return_value = dt_obj
    return qdt


class TestCreateWeather:
    """Tests for create_weather_from_parameters."""

    def test_basic_weather(self, feedback):
        """Basic weather parameters produce correct Weather object."""
        dt_obj = datetime(2024, 7, 15, 12, 0)
        params = {
            "DATETIME": _make_qdt(dt_obj),
            "TEMPERATURE": 30.0,
            "HUMIDITY": 60.0,
            "GLOBAL_RADIATION": 900.0,
            "WIND_SPEED": 2.0,
            "PRESSURE": 1010.0,
        }
        weather = create_weather_from_parameters(params, feedback)
        assert weather.datetime == dt_obj
        assert weather.ta == 30.0
        assert weather.rh == 60.0
        assert weather.global_rad == 900.0
        assert weather.ws == 2.0
        assert weather.pressure == 1010.0

    def test_default_values(self, feedback):
        """Missing parameters use default values."""
        dt_obj = datetime(2024, 1, 1, 12, 0)
        params = {"DATETIME": _make_qdt(dt_obj)}
        weather = create_weather_from_parameters(params, feedback)
        assert weather.ta == 25.0
        assert weather.rh == 50.0
        assert weather.global_rad == 800.0
        assert weather.ws == 1.0
        assert weather.pressure == 1013.25

    def test_feedback_message(self, feedback):
        """Feedback receives info message about weather."""
        dt_obj = datetime(2024, 7, 15, 12, 0)
        params = {
            "DATETIME": _make_qdt(dt_obj),
            "TEMPERATURE": 30.0,
            "HUMIDITY": 60.0,
            "GLOBAL_RADIATION": 900.0,
        }
        create_weather_from_parameters(params, feedback)
        feedback.pushInfo.assert_called_once()
        msg = feedback.pushInfo.call_args[0][0]
        assert "30.0" in msg
        assert "60" in msg
        assert "900" in msg

    def test_returns_weather_instance(self, feedback):
        """Result is a solweig.Weather dataclass."""
        dt_obj = datetime(2024, 7, 15, 12, 0)
        weather = create_weather_from_parameters({"DATETIME": _make_qdt(dt_obj)}, feedback)
        assert isinstance(weather, solweig.Weather)


class TestRelativeHeightHeuristic:
    """Tests for detecting relative canopy/trunk rasters."""

    def test_detects_bilbao_style_relative_canopy(self):
        """Low canopy heights over high terrain should be flagged as relative."""
        reference = np.array([[85.0, 90.0], [95.0, 100.0]], dtype=np.float32)
        cdsm = np.array([[5.0, 12.0], [18.0, 25.0]], dtype=np.float32)
        assert _looks_like_relative_heights(cdsm, reference) is True

    def test_does_not_flag_absolute_canopy(self):
        """Absolute canopy elevations should not be flagged."""
        reference = np.array([[85.0, 90.0], [95.0, 100.0]], dtype=np.float32)
        cdsm = np.array([[90.0, 102.0], [111.0, 120.0]], dtype=np.float32)
        assert _looks_like_relative_heights(cdsm, reference) is False


# ---------------------------------------------------------------------------
# create_location_from_parameters
# ---------------------------------------------------------------------------


class TestCreateLocation:
    """Tests for create_location_from_parameters."""

    def test_manual_location(self, feedback):
        """Manual lat/lon input creates correct Location."""
        surface = MagicMock()
        params = {
            "AUTO_EXTRACT_LOCATION": False,
            "LATITUDE": 37.97,
            "LONGITUDE": 23.73,
            "UTC_OFFSET": 2,
        }
        location = create_location_from_parameters(params, surface, feedback)
        assert location.latitude == 37.97
        assert location.longitude == 23.73
        assert location.utc_offset == 2

    def test_manual_location_defaults_utc0(self, feedback):
        """UTC offset defaults to 0."""
        surface = MagicMock()
        params = {
            "AUTO_EXTRACT_LOCATION": False,
            "LATITUDE": 51.5,
            "LONGITUDE": -0.1,
        }
        location = create_location_from_parameters(params, surface, feedback)
        assert location.utc_offset == 0

    def test_manual_missing_coords_raises(self, feedback):
        """Missing lat/lon raises QgsProcessingException."""
        surface = MagicMock()
        params = {"AUTO_EXTRACT_LOCATION": False}
        with pytest.raises(QgsProcessingException, match="Latitude and longitude are required"):
            create_location_from_parameters(params, surface, feedback)

    def test_auto_extract_no_crs_raises(self, feedback):
        """Auto-extract with missing CRS raises QgsProcessingException."""
        surface = MagicMock()
        surface._crs_wkt = None
        params = {"AUTO_EXTRACT_LOCATION": True, "UTC_OFFSET": 0}
        with pytest.raises(QgsProcessingException, match="Cannot auto-extract"):
            create_location_from_parameters(params, surface, feedback)

    def test_returns_location_instance(self, feedback):
        """Result is a solweig.Location dataclass."""
        surface = MagicMock()
        params = {"LATITUDE": 57.7, "LONGITUDE": 12.0}
        location = create_location_from_parameters(params, surface, feedback)
        assert isinstance(location, solweig.Location)

    def test_feedback_for_manual_location(self, feedback):
        """Feedback reports manual coordinates."""
        surface = MagicMock()
        params = {
            "AUTO_EXTRACT_LOCATION": False,
            "LATITUDE": 57.7,
            "LONGITUDE": 12.0,
        }
        create_location_from_parameters(params, surface, feedback)
        feedback.pushInfo.assert_called()
        msg = feedback.pushInfo.call_args[0][0]
        assert "57.7" in msg
        assert "12.0" in msg

    def test_auto_extract_delegates_to_location_from_surface(self, feedback, monkeypatch):
        """Auto-extract should use the core Location.from_surface helper."""
        surface = MagicMock()
        expected = solweig.Location(latitude=57.7, longitude=12.0, utc_offset=1)

        def fake_from_surface(surface_arg, utc_offset=None, altitude=0.0):
            assert surface_arg is surface
            assert utc_offset == 1
            return expected

        monkeypatch.setattr(solweig.Location, "from_surface", fake_from_surface)

        location = create_location_from_parameters({"AUTO_EXTRACT_LOCATION": True, "UTC_OFFSET": 1}, surface, feedback)

        assert location is expected

    def test_epw_location_delegates_to_location_from_epw(self, feedback, monkeypatch):
        """EPW mode should use the core Location.from_epw helper."""
        expected = solweig.Location(latitude=43.2, longitude=-2.9, utc_offset=1, altitude=12.0)

        def fake_from_epw(path):
            assert path == "/tmp/bilbao.epw"
            return expected

        monkeypatch.setattr(solweig.Location, "from_epw", fake_from_epw)

        location = create_location_from_parameters({}, MagicMock(), feedback, epw_path="/tmp/bilbao.epw")

        assert location is expected


# ---------------------------------------------------------------------------
# load_raster_from_layer
# ---------------------------------------------------------------------------


class TestLoadRasterFromLayer:
    """Tests for QGIS raster loading helpers."""

    def test_positive_nodata_is_mapped_to_nan(self):
        """Positive nodata sentinels should be treated the same as local API loads."""
        layer = MagicMock()
        layer.source.return_value = "/tmp/fake.tif"

        mock_band = MagicMock()
        mock_band.ReadAsArray.return_value = np.array([[1.0, 9999.0], [2.0, 3.0]], dtype=np.float32)
        mock_band.GetNoDataValue.return_value = 9999.0

        mock_ds = MagicMock()
        mock_ds.GetRasterBand.return_value = mock_band
        mock_ds.GetGeoTransform.return_value = [0.0, 1.0, 0.0, 2.0, 0.0, -1.0]
        mock_ds.GetProjection.return_value = "WKT"

        with patch("qgis_plugin.solweig_qgis.utils.converters.gdal.Open", return_value=mock_ds):
            array, geotransform, crs_wkt = load_raster_from_layer(layer)

        assert np.isnan(array[0, 1])
        assert array[0, 0] == 1.0
        assert geotransform == [0.0, 1.0, 0.0, 2.0, 0.0, -1.0]
        assert crs_wkt == "WKT"


class TestLoadPreparedSurface:
    """Tests for prepared surface loading safeguards."""

    def test_rejects_prepared_relative_cdsm(self, tmp_path, feedback):
        """Malformed prepared surfaces with relative canopy heights should fail loudly.

        SurfaceData.load() raises ValueError when CDSM looks relative;
        load_prepared_surface wraps it as QgsProcessingException.
        """
        with (
            patch(
                "solweig.SurfaceData.load",
                side_effect=ValueError("Loaded CDSM appears to contain relative heights"),
            ),
            pytest.raises(QgsProcessingException, match="relative heights"),
        ):
            load_prepared_surface(str(tmp_path), feedback)


# ---------------------------------------------------------------------------
# load_weather_from_epw
# ---------------------------------------------------------------------------


class TestLoadWeatherFromEpw:
    """Tests for load_weather_from_epw."""

    def test_invalid_epw_path_raises(self, feedback):
        """Non-existent EPW file raises QgsProcessingException."""
        with pytest.raises(QgsProcessingException, match="EPW file not found"):
            load_weather_from_epw(
                "/nonexistent/file.epw",
                start_dt=datetime(2024, 1, 1),
                end_dt=datetime(2024, 12, 31),
                hours_filter=None,
                feedback=feedback,
            )

    def test_qdatetime_conversion(self, feedback):
        """QDateTime objects are converted to Python datetime."""
        qdt_start = MagicMock()
        qdt_start.toPyDateTime.return_value = datetime(2024, 1, 1)
        qdt_end = MagicMock()
        qdt_end.toPyDateTime.return_value = datetime(2024, 12, 31)

        with pytest.raises(QgsProcessingException):
            load_weather_from_epw("/nonexistent.epw", qdt_start, qdt_end, None, feedback)

        # Verify toPyDateTime was called (conversion happened)
        qdt_start.toPyDateTime.assert_called_once()
        qdt_end.toPyDateTime.assert_called_once()

    def test_hours_filter_parsing(self, feedback):
        """Valid hours filter string is parsed and reported to feedback."""
        with contextlib.suppress(Exception):
            load_weather_from_epw(
                "/nonexistent.epw",
                start_dt=datetime(2024, 1, 1),
                end_dt=datetime(2024, 12, 31),
                hours_filter="9,10,11,12",
                feedback=feedback,
            )

        found_hour_msg = any("9" in str(call) and "10" in str(call) for call in feedback.pushInfo.call_args_list)
        assert found_hour_msg, "Expected hour filter info message"

    def test_invalid_hours_filter_warns(self, feedback):
        """Invalid hours filter reports error via feedback."""
        with contextlib.suppress(Exception):
            load_weather_from_epw(
                "/nonexistent.epw",
                start_dt=datetime(2024, 1, 1),
                end_dt=datetime(2024, 12, 31),
                hours_filter="abc",
                feedback=feedback,
            )

        feedback.reportError.assert_called_once()
        msg = feedback.reportError.call_args[0][0]
        assert "Invalid hours filter" in msg

    def test_delegates_to_weather_from_epw(self, feedback, monkeypatch):
        """Weather loading should delegate to the core Weather.from_epw API."""
        expected = [solweig.Weather(datetime=datetime(2024, 7, 15, 12, 0), ta=25.0, rh=50.0, global_rad=800.0)]

        def fake_from_epw(path, start=None, end=None, hours=None, year=None):
            assert path == "/tmp/test.epw"
            assert start == datetime(2024, 7, 15)
            assert end == datetime(2024, 7, 16)
            assert hours == [9, 10, 11]
            assert year is None
            return expected

        monkeypatch.setattr(solweig.Weather, "from_epw", fake_from_epw)

        result = load_weather_from_epw(
            "/tmp/test.epw",
            start_dt=datetime(2024, 7, 15),
            end_dt=datetime(2024, 7, 16),
            hours_filter="9,10,11",
            feedback=feedback,
        )

        assert result == expected


# ---------------------------------------------------------------------------
# create_surface_from_parameters
# ---------------------------------------------------------------------------


class _DummyParamHandler:
    def __init__(self, layers: dict[str, object]):
        self._layers = layers

    def parameterAsRasterLayer(self, parameters, param_name, context):
        return self._layers.get(param_name)

    def parameterAsDouble(self, parameters, param_name, context):
        return float(parameters.get(param_name, 1.0))


class TestCreateSurfaceFromParameters:
    def test_walls_are_computed_before_mask_crop(self, monkeypatch, feedback):
        from solweig.physics import wallalgorithms as wa

        import qgis_plugin.solweig_qgis.utils.converters as converters

        dsm_layer = MagicMock(name="dsm_layer")
        dem_layer = MagicMock(name="dem_layer")
        layers = {"DSM": dsm_layer, "DEM": dem_layer}
        handler = _DummyParamHandler(layers)

        dsm = np.array(
            [
                [np.nan, np.nan, np.nan],
                [np.nan, 10.0, 10.0],
                [np.nan, 10.0, 10.0],
            ],
            dtype=np.float32,
        )
        dem = np.zeros_like(dsm, dtype=np.float32)
        geotransform = [0.0, 1.0, 0.0, 3.0, 0.0, -1.0]

        def fake_load_raster_from_layer(layer):
            if layer is dsm_layer:
                return dsm.copy(), geotransform, "WKT"
            if layer is dem_layer:
                return dem.copy(), geotransform, "WKT"
            raise AssertionError(f"Unexpected layer {layer!r}")

        monkeypatch.setattr(converters, "load_raster_from_layer", fake_load_raster_from_layer)

        events: list[str] = []

        orig_preprocess = solweig.SurfaceData.preprocess
        orig_compute_valid_mask = solweig.SurfaceData.compute_valid_mask
        orig_apply_valid_mask = solweig.SurfaceData.apply_valid_mask
        orig_crop_to_valid_bbox = solweig.SurfaceData.crop_to_valid_bbox

        def wrapped_preprocess(self):
            events.append("preprocess")
            return orig_preprocess(self)

        def wrapped_compute_valid_mask(self):
            events.append("compute_valid_mask")
            return orig_compute_valid_mask(self)

        def wrapped_apply_valid_mask(self):
            events.append("apply_valid_mask")
            return orig_apply_valid_mask(self)

        def wrapped_crop_to_valid_bbox(self):
            events.append("crop_to_valid_bbox")
            return orig_crop_to_valid_bbox(self)

        def wrapped_findwalls(arr, walllimit):
            events.append("findwalls")
            return np.zeros_like(arr, dtype=np.float32)

        def wrapped_filter1(walls, dsm_scale, dsm_arr, feedback=None):
            events.append("filter1Goodwin_as_aspect_v3")
            return np.zeros_like(dsm_arr, dtype=np.float32)

        monkeypatch.setattr(solweig.SurfaceData, "preprocess", wrapped_preprocess)
        monkeypatch.setattr(solweig.SurfaceData, "compute_valid_mask", wrapped_compute_valid_mask)
        monkeypatch.setattr(solweig.SurfaceData, "apply_valid_mask", wrapped_apply_valid_mask)
        monkeypatch.setattr(solweig.SurfaceData, "crop_to_valid_bbox", wrapped_crop_to_valid_bbox)
        monkeypatch.setattr(wa, "findwalls", wrapped_findwalls)
        monkeypatch.setattr(wa, "filter1Goodwin_as_aspect_v3", wrapped_filter1)

        params = {
            "DSM": True,
            "DEM": True,
            "DSM_HEIGHT_MODE": 1,
            "MIN_OBJECT_HEIGHT": 1.0,
        }

        create_surface_from_parameters(params, context=MagicMock(), param_handler=handler, feedback=feedback)

        assert events == [
            "preprocess",
            "findwalls",
            "filter1Goodwin_as_aspect_v3",
            "compute_valid_mask",
            "apply_valid_mask",
            "crop_to_valid_bbox",
        ]
