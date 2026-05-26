"""
Tests for QGIS plugin base algorithm class.

Mocks QGIS and GDAL dependencies so these tests run without a QGIS installation.
Tests grid validation, output path logic, and georeferenced output saving.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.qgis_mocks import QgsProcessingException, install, install_osgeo, preserve_solweig_modules, uninstall_osgeo

install()  # Must be called before any qgis_plugin imports
install_osgeo()  # Temporarily needed for osgeo imports in base.py + utils.converters

with preserve_solweig_modules():
    from qgis_plugin.solweig_qgis.algorithms.base import SolweigAlgorithmBase  # noqa: E402

    # Pre-import the converters module too, so the lazy `from ..utils.converters
    # import load_raster_from_layer` call inside the base-class delegating
    # method sees the osgeo mock when the test triggers it. Otherwise the
    # test environment (which uninstalls osgeo immediately after this block)
    # would hit ModuleNotFoundError when load_raster_from_layer is called.
    from qgis_plugin.solweig_qgis.utils import converters as _converters  # noqa: E402, F401

uninstall_osgeo()  # Clean up immediately after imports to avoid polluting other tests


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def algo():
    """Create a SolweigAlgorithmBase instance for testing."""
    return SolweigAlgorithmBase()


@pytest.fixture()
def feedback():
    """Create a mock QgsProcessingFeedback."""
    fb = MagicMock()
    fb.pushInfo = MagicMock()
    return fb


@pytest.fixture()
def gdal_mocks():
    """Create fresh GDAL mock chain for save operations (patched on the base module)."""
    mock_band = MagicMock()
    mock_ds = MagicMock()
    mock_ds.GetRasterBand.return_value = mock_band
    mock_driver = MagicMock()
    mock_driver.Create.return_value = mock_ds

    mock_gdal = MagicMock()
    mock_gdal.GetDriverByName.return_value = mock_driver
    mock_gdal.GDT_Float32 = 6

    with patch("qgis_plugin.solweig_qgis.algorithms.base.gdal", mock_gdal):
        yield mock_driver, mock_ds, mock_band


# ---------------------------------------------------------------------------
# check_grid_shapes_match
# ---------------------------------------------------------------------------


class TestCheckGridShapesMatch:
    """Tests for SolweigAlgorithmBase.check_grid_shapes_match."""

    def test_matching_shapes_pass(self, algo, feedback):
        """No exception when all arrays match reference shape."""
        ref = (100, 200)
        arrays = {
            "CDSM": np.zeros((100, 200)),
            "DEM": np.ones((100, 200)),
        }
        algo.check_grid_shapes_match(ref, arrays, feedback)

    def test_none_arrays_skipped(self, algo, feedback):
        """None values are silently skipped."""
        ref = (100, 200)
        arrays = {
            "CDSM": None,
            "DEM": np.zeros((100, 200)),
            "TDSM": None,
        }
        algo.check_grid_shapes_match(ref, arrays, feedback)

    def test_mismatched_shape_raises(self, algo, feedback):
        """Mismatched array shape raises QgsProcessingException."""
        ref = (100, 200)
        arrays = {"CDSM": np.zeros((50, 200))}
        with pytest.raises(QgsProcessingException, match="Grid shape mismatch"):
            algo.check_grid_shapes_match(ref, arrays, feedback)

    def test_error_message_includes_name(self, algo, feedback):
        """Error message includes the array name."""
        ref = (100, 200)
        arrays = {"DEM": np.zeros((100, 100))}
        with pytest.raises(QgsProcessingException, match="DEM"):
            algo.check_grid_shapes_match(ref, arrays, feedback)

    def test_empty_arrays_pass(self, algo, feedback):
        """Empty arrays dict doesn't raise."""
        algo.check_grid_shapes_match((100, 200), {}, feedback)


# ---------------------------------------------------------------------------
# get_output_path
# ---------------------------------------------------------------------------


class TestGetOutputPath:
    """Tests for SolweigAlgorithmBase.get_output_path."""

    def test_temp_file_when_no_param(self, algo):
        """Returns temp path when output parameter is empty."""
        context = MagicMock()
        result = algo.get_output_path({}, "OUTPUT_TMRT", "tmrt.tif", context)
        assert result.endswith("tmrt.tif")
        assert "solweig_qgis_output" in result

    def test_temp_file_when_param_empty(self, algo):
        """Returns temp path when parameter is empty string."""
        context = MagicMock()
        result = algo.get_output_path({"OUTPUT_TMRT": ""}, "OUTPUT_TMRT", "tmrt.tif", context)
        assert result.endswith("tmrt.tif")


# ---------------------------------------------------------------------------
# save_georeferenced_output
# ---------------------------------------------------------------------------


class TestSaveGeoreferencedOutput:
    """Tests for SolweigAlgorithmBase.save_georeferenced_output."""

    def test_creates_output_dir(self, algo, tmp_path, gdal_mocks):
        """Output directory is created if it doesn't exist."""
        mock_driver, mock_ds, mock_band = gdal_mocks
        output_path = tmp_path / "subdir" / "output.tif"
        geotransform = [0.0, 1.0, 0.0, 10.0, 0.0, -1.0]

        algo.save_georeferenced_output(np.ones((10, 10)), output_path, geotransform, "WKT")

        assert (tmp_path / "subdir").exists()
        mock_driver.Create.assert_called_once()
        mock_ds.SetGeoTransform.assert_called_once_with(geotransform)
        mock_ds.SetProjection.assert_called_once_with("WKT")
        mock_band.WriteArray.assert_called_once()

    def test_nan_replaced_with_nodata(self, algo, tmp_path, gdal_mocks):
        """NaN values are replaced with nodata value."""
        _, _, mock_band = gdal_mocks
        array = np.array([[1.0, np.nan], [np.nan, 2.0]])

        algo.save_georeferenced_output(array, tmp_path / "out.tif", [0, 1, 0, 2, 0, -1], "")

        written = mock_band.WriteArray.call_args[0][0]
        assert not np.any(np.isnan(written))
        assert written[0, 1] == -9999.0
        assert written[1, 0] == -9999.0

    def test_custom_nodata(self, algo, tmp_path, gdal_mocks):
        """Custom nodata value is used."""
        _, _, mock_band = gdal_mocks
        array = np.array([[1.0, np.nan]])

        algo.save_georeferenced_output(array, tmp_path / "out.tif", [0, 1, 0, 1, 0, -1], "", nodata=-999.0)

        mock_band.SetNoDataValue.assert_called_once_with(-999.0)

    def test_feedback_message(self, algo, tmp_path, feedback, gdal_mocks):
        """Feedback reports saved file path."""
        algo.save_georeferenced_output(
            np.ones((2, 2)), tmp_path / "output.tif", [0, 1, 0, 2, 0, -1], "", feedback=feedback
        )

        feedback.pushInfo.assert_called_once()
        assert "output.tif" in feedback.pushInfo.call_args[0][0]

    def test_driver_create_failure_raises(self, algo, tmp_path):
        """Raises QgsProcessingException when GDAL cannot create output."""
        mock_driver = MagicMock()
        mock_driver.Create.return_value = None
        mock_gdal = MagicMock()
        mock_gdal.GetDriverByName.return_value = mock_driver
        mock_gdal.GDT_Float32 = 6

        with (
            patch("qgis_plugin.solweig_qgis.algorithms.base.gdal", mock_gdal),
            pytest.raises(QgsProcessingException, match="Cannot create output"),
        ):
            algo.save_georeferenced_output(np.ones((2, 2)), tmp_path / "out.tif", [0, 1, 0, 2, 0, -1], "")


# ---------------------------------------------------------------------------
# load_raster_from_layer
# ---------------------------------------------------------------------------


class TestLoadRasterFromLayer:
    """Tests for SolweigAlgorithmBase.load_raster_from_layer."""

    def test_positive_nodata_is_mapped_to_nan(self, algo):
        """Positive nodata sentinels should not survive as valid terrain values."""
        layer = MagicMock()
        layer.source.return_value = "/tmp/fake.tif"

        mock_band = MagicMock()
        mock_band.ReadAsArray.return_value = np.array([[1.0, 9999.0], [2.0, 3.0]], dtype=np.float32)
        mock_band.GetNoDataValue.return_value = 9999.0

        mock_ds = MagicMock()
        mock_ds.GetRasterBand.return_value = mock_band
        mock_ds.GetGeoTransform.return_value = [0.0, 1.0, 0.0, 2.0, 0.0, -1.0]
        mock_ds.GetProjection.return_value = "WKT"

        mock_gdal = MagicMock()
        mock_gdal.Open.return_value = mock_ds
        mock_gdal.GA_ReadOnly = 0

        # The implementation lives in utils.converters now; base.load_raster_from_layer
        # is a thin delegating wrapper. Patch the canonical home.
        with patch("qgis_plugin.solweig_qgis.utils.converters.gdal", mock_gdal):
            array, geotransform, crs_wkt = algo.load_raster_from_layer(layer)

        assert np.isnan(array[0, 1])
        assert array[0, 0] == 1.0
        assert geotransform == [0.0, 1.0, 0.0, 2.0, 0.0, -1.0]
        assert crs_wkt == "WKT"


# ---------------------------------------------------------------------------
# createInstance / group / groupId
# ---------------------------------------------------------------------------


class TestAlgorithmMeta:
    """Tests for algorithm metadata methods."""

    def test_create_instance_returns_same_class(self, algo):
        """createInstance returns a new instance of the same class."""
        new = algo.createInstance()
        assert type(new) is SolweigAlgorithmBase

    def test_group_id(self, algo):
        """Group ID is empty (algorithms appear directly under provider)."""
        assert algo.groupId() == ""

    def test_help_url(self, algo):
        """Help URL points to UMEP docs."""
        assert "umep" in algo.helpUrl().lower()
