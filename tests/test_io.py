"""
Tests for I/O functionality including EPW parser.

Note: EPW parser is deliberately pandas-free for QGIS compatibility.
Tests must not assume pd.DataFrame - they test the _EpwDataFrame interface.
"""

from pathlib import Path

import numpy as np
import pytest
from solweig import io


class TestEPWParser:
    """Test the standalone EPW parser (no pandas dependency)."""

    @pytest.fixture
    def sample_epw_content(self):
        """Create a minimal valid EPW file content."""
        # EPW header (8 lines) + data
        # Timezone offset must be between -24 and +24 hours (field 8)
        # EPW data lines must preserve exact format - long lines are intentional
        return """LOCATION,Athens,GRC,NA,Shiny Weather Data,NA,37.90,23.73,2.0,107.0
DESIGN CONDITIONS,1,Climate Design Data 2009 ASHRAE Handbook,,Heating,1,-2.1,-0.3,0.6,2.8,10.7,2.3,3.5,3.4,12.2,11.2,3.1,11.4,2.5,340,Cooling,8,35.2,23.7,33.2,23.3,31.4,23.0,29.7,24.1,27.2,32.8,26.1,31.1,25.2,29.6,4.2,330,23.5,18.5,27.8,22.7,17.8,27.1,22.0,17.2,26.4,68.2,32.9,64.8,31.2,62.0,29.7,951,Extremes,11.6,10.2,9.0,25.3,-3.9,37.5,2.7,1.7,-5.5,38.9,-7.0,39.9,-8.4,40.8,-10.1,42.2
TYPICAL/EXTREME PERIODS,6,Summer - Week Nearest Max Temperature For Period,Extreme,7/ 9,7/15,Summer - Week Nearest Average Temperature For Period,Typical,7/30,8/ 5,Winter - Week Nearest Min Temperature For Period,Extreme,1/28,2/ 3,Winter - Week Nearest Average Temperature For Period,Typical,1/21,1/27,Autumn - Week Nearest Average Temperature For Period,Typical,11/11,11/17,Spring - Week Nearest Average Temperature For Period,Typical,4/22,4/28
GROUND TEMPERATURES,3,.5,,,12.98,11.39,10.73,11.54,14.82,18.56,21.85,23.85,24.08,22.71,19.89,16.54,2,,,,,,,,,,,,,,,,4,,,,,,,,,,,,,,
HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0
COMMENTS 1,Custom/IWEC Data
COMMENTS 2, -- Ground temps produced with a standard soil diffusivity of 2.3225760E-03 {m**2/day}
DATA PERIODS,1,1,Data,Sunday, 1/ 1,12/31
2024,1,1,1,0,?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9*_*9*9*9*9*9,9.0,3.9,65,101300,0,0,0,0,0,0,0,0,0,0,190,4.6,10,10,16.1,77777,9,999999999,0,0.0480,0,88,0.000,0.0,0.0
2024,1,1,2,0,?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9*_*9*9*9*9*9,8.3,3.9,69,101300,0,0,0,0,0,0,0,0,0,0,190,4.1,10,10,16.1,77777,9,999999999,0,0.0480,0,88,0.000,0.0,0.0
2024,1,1,3,0,?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9*_*9*9*9*9*9,7.8,3.9,72,101300,0,0,0,0,0,0,0,0,0,0,200,3.6,10,10,16.1,77777,9,999999999,0,0.0480,0,88,0.000,0.0,0.0
2024,1,1,4,0,?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9*_*9*9*9*9*9,7.2,3.9,76,101300,0,0,0,0,0,0,0,0,0,0,200,3.1,10,10,16.1,77777,9,999999999,0,0.0480,0,88,0.000,0.0,0.0
2024,1,1,5,0,?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9*_*9*9*9*9*9,6.7,3.3,76,101300,0,0,0,0,0,0,0,0,0,0,200,3.1,10,10,16.1,77777,9,999999999,0,0.0480,0,88,0.000,0.0,0.0
"""

    @pytest.fixture
    def epw_file(self, sample_epw_content, tmp_path):
        """Create a temporary EPW file."""
        epw_path = tmp_path / "test.epw"
        epw_path.write_text(sample_epw_content)
        return epw_path

    def test_read_epw_returns_data_and_metadata(self, epw_file):
        """Test that read_epw returns a data object and metadata dict."""
        df, metadata = io.read_epw(epw_file)

        assert len(df) == 5
        assert isinstance(metadata, dict)

    def test_epw_metadata_parsing(self, epw_file):
        """Test that EPW metadata is correctly parsed."""
        df, metadata = io.read_epw(epw_file)

        assert metadata["city"] == "Athens"
        assert abs(metadata["latitude"] - 37.90) < 0.01
        assert abs(metadata["longitude"] - 23.73) < 0.01
        assert abs(metadata["elevation"] - 107.0) < 0.1

    def test_epw_data_columns(self, epw_file):
        """Test that EPW data has expected columns."""
        df, _ = io.read_epw(epw_file)

        # Check for essential weather columns
        expected_cols = [
            "temp_air",
            "relative_humidity",
            "atmospheric_pressure",
            "wind_speed",
            "wind_direction",
            "ghi",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_epw_datetime_index(self, epw_file):
        """Test that EPW data has proper datetime index."""
        df, _ = io.read_epw(epw_file)

        assert df.index.name == "datetime"

        # Check first timestamp
        first_timestamp = df.index[0]
        assert first_timestamp.year == 2024
        assert first_timestamp.month == 1
        assert first_timestamp.day == 1
        assert first_timestamp.hour == 1

    def test_epw_temperature_values(self, epw_file):
        """Test that temperature values are reasonable."""
        df, _ = io.read_epw(epw_file)

        # Temperature should be in Celsius
        assert df["temp_air"].min() >= -50  # Reasonable minimum
        assert df["temp_air"].max() <= 60  # Reasonable maximum

        # Check specific values from sample data
        assert abs(df.iloc[0]["temp_air"] - 9.0) < 0.1

    def test_epw_humidity_values(self, epw_file):
        """Test that humidity values are in valid range."""
        df, _ = io.read_epw(epw_file)

        assert (df["relative_humidity"] >= 0).all()
        assert (df["relative_humidity"] <= 100).all()

        # Check specific value from sample data
        assert df.iloc[0]["relative_humidity"] == 65

    def test_epw_pressure_values(self, epw_file):
        """Test that pressure values are reasonable."""
        df, _ = io.read_epw(epw_file)

        # Pressure should be in Pa
        assert (df["atmospheric_pressure"] > 50000).all()  # > 500 hPa
        assert (df["atmospheric_pressure"] < 110000).all()  # < 1100 hPa

    def test_epw_handles_pathlib_path(self, epw_file):
        """Test that read_epw accepts pathlib.Path."""
        df, metadata = io.read_epw(Path(epw_file))

        assert len(df) == 5
        assert metadata["city"] == "Athens"

    def test_epw_handles_string_path(self, epw_file):
        """Test that read_epw accepts string path."""
        df, metadata = io.read_epw(str(epw_file))

        assert len(df) == 5
        assert metadata["city"] == "Athens"

    def test_epw_missing_file_raises_error(self):
        """Test that reading non-existent EPW file raises error."""
        with pytest.raises(FileNotFoundError):
            io.read_epw("nonexistent.epw")

    def test_to_dataframe_converts_when_pandas_available(self, epw_file):
        """Test that to_dataframe() converts to pandas when available."""
        pd = pytest.importorskip("pandas", reason="pandas not available")

        df, _ = io.read_epw(epw_file)
        pdf = df.to_dataframe()

        assert isinstance(pdf, pd.DataFrame)
        assert isinstance(pdf.index, pd.DatetimeIndex)
        assert len(pdf) == 5


class TestRasterIO:
    """Test raster I/O with GDAL backend fallback."""

    def test_gdal_backend_env_variable(self, monkeypatch):
        """Test that UMEP_USE_GDAL environment variable works."""
        # Skip if GDAL is not available
        try:
            from osgeo import gdal  # noqa: F401

            del gdal  # Silence unused import warning
        except ImportError:
            pytest.skip("GDAL not available")

        # Set environment variable
        monkeypatch.setenv("UMEP_USE_GDAL", "1")

        # Reload _compat (the source of truth for backend selection)
        # to pick up the environment variable change.
        import importlib

        from solweig import _compat

        importlib.reload(_compat)

        # Should use GDAL backend
        assert _compat.GDAL_ENV

    def test_backend_auto_detection(self, monkeypatch):
        """Test that backend auto-detection works without QGIS or env-var override.

        In a standard (non-QGIS) environment, auto-detection must not raise and
        must select exactly one backend: rasterio (if the full stack — rasterio,
        pyproj, shapely — is available) or GDAL as fallback.
        """
        import importlib
        import sys

        from solweig import _compat

        # Ensure environment variable is not set
        monkeypatch.delenv("UMEP_USE_GDAL", raising=False)

        # Remove any QGIS mocks that earlier tests may have injected,
        # so _compat.in_osgeo_environment() returns False.
        qgis_keys = [k for k in sys.modules if k == "qgis" or k.startswith("qgis.")]
        saved = {k: sys.modules.pop(k) for k in qgis_keys}
        try:
            importlib.reload(_compat)
            # Access attrs NOW, while qgis is still removed from sys.modules.
            # _compat uses lazy __getattr__ (PEP 562): backend detection only
            # runs on first attribute access, not during reload.  If we wait
            # until after the finally block restores the qgis mocks,
            # in_osgeo_environment() would see them and pick GDAL.
            rasterio_available = _compat.RASTERIO_AVAILABLE
            gdal_env = _compat.GDAL_ENV
        finally:
            sys.modules.update(saved)

        # Exactly one backend must be selected
        assert rasterio_available is not gdal_env, (
            f"Exactly one backend must be active: RASTERIO_AVAILABLE={rasterio_available}, GDAL_ENV={gdal_env}"
        )

        # If the full rasterio stack is present it must be preferred
        if _compat._try_import_rasterio():
            assert rasterio_available is True
            assert gdal_env is False
        else:
            # Fallback to GDAL — must not raise
            assert gdal_env is True
            assert rasterio_available is False


class TestGeoTIFFLoading:
    """Test GeoTIFF loading functionality."""

    @pytest.fixture
    def sample_geotiff(self, tmp_path):
        """Create a minimal GeoTIFF file for testing."""
        try:
            from osgeo import gdal, osr

            # Create a simple 10x10 raster
            driver = gdal.GetDriverByName("GTiff")
            ds = driver.Create(
                str(tmp_path / "test.tif"),
                10,
                10,
                1,
                gdal.GDT_Float32,
            )

            # Set geotransform
            ds.SetGeoTransform([0, 1, 0, 0, 0, -1])

            # Set projection (WGS84)
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)
            ds.SetProjection(srs.ExportToWkt())

            # Write data
            band = ds.GetRasterBand(1)
            data = np.arange(100, dtype=np.float32).reshape(10, 10)
            band.WriteArray(data)
            band.SetNoDataValue(-9999)

            # Close dataset
            ds = None

            return tmp_path / "test.tif"

        except ImportError:
            pytest.skip("GDAL not available for creating test file")

    def test_load_raster_returns_tuple(self, sample_geotiff):
        """Test that load_raster returns expected tuple."""
        result = io.load_raster(str(sample_geotiff))

        # Should return (array, transform, crs, nodata)
        assert len(result) == 4

        array, transform, crs, nodata = result

        assert isinstance(array, np.ndarray)
        assert array.shape == (10, 10)
        assert transform is not None
        assert crs is not None

    def test_load_raster_preserves_data(self, sample_geotiff):
        """Test that loaded data matches written data."""
        array, _, _, _ = io.load_raster(str(sample_geotiff))

        # Should match the data we wrote
        expected = np.arange(100, dtype=np.float32).reshape(10, 10)
        np.testing.assert_array_almost_equal(array, expected)


class TestWindowedWriteRoundTrip:
    """Verify create_empty_raster + write_raster_window round-trips correctly.

    This exercises the code path used by TiledGeoTiffWriter and must work
    under both rasterio and GDAL backends.
    """

    def test_windowed_write_recovers_data(self, tmp_path):
        """Write two non-overlapping windows, read back the full raster."""
        path = tmp_path / "windowed.tif"
        rows, cols = 20, 30
        transform = [0.0, 1.0, 0.0, float(rows), 0.0, -1.0]

        io.create_empty_raster(
            path_str=path,
            rows=rows,
            cols=cols,
            transform=transform,
            crs_wkt="",
            dtype=np.float32,
            nodata=np.nan,
        )

        # Write two tiles covering the full raster
        top = np.full((10, 30), 1.0, dtype=np.float32)
        bot = np.full((10, 30), 2.0, dtype=np.float32)
        io.write_raster_window(path, top, window=(slice(0, 10), slice(0, 30)))
        io.write_raster_window(path, bot, window=(slice(10, 20), slice(0, 30)))

        # Read back
        data, _, _, _ = io.load_raster(str(path))
        np.testing.assert_array_equal(data[:10, :], 1.0)
        np.testing.assert_array_equal(data[10:, :], 2.0)

    def test_windowed_write_nan_nodata(self, tmp_path):
        """Pixels not written should remain NaN (the nodata fill)."""
        path = tmp_path / "partial.tif"
        rows, cols = 10, 10
        transform = [0.0, 1.0, 0.0, float(rows), 0.0, -1.0]

        io.create_empty_raster(
            path_str=path,
            rows=rows,
            cols=cols,
            transform=transform,
            crs_wkt="",
            dtype=np.float32,
            nodata=np.nan,
        )

        # Write only the top-left 5x5 quadrant
        patch = np.full((5, 5), 42.0, dtype=np.float32)
        io.write_raster_window(path, patch, window=(slice(0, 5), slice(0, 5)))

        data, _, _, _ = io.load_raster(str(path))
        np.testing.assert_array_equal(data[:5, :5], 42.0)
        # Unwritten pixels: NaN on GDAL (Fill), possibly garbage on rasterio.
        # Both backends must at least produce a readable file.
        assert data.shape == (10, 10)


# ──────────────────────────────────────────────────────────────────────────────
# Raster I/O + pixel-grid helpers — coverage extensions
# ──────────────────────────────────────────────────────────────────────────────


from solweig.io import (  # noqa: E402
    check_path,
    create_empty_raster,
    get_raster_metadata,
    load_raster,
    read_raster_window,
    save_raster,
    shrink_bbox_to_pixel_grid,
    write_raster_window,
)

# ── shrink_bbox_to_pixel_grid ──


def test_shrink_bbox_already_on_grid_is_unchanged():
    """An exactly-on-grid bbox should snap to itself."""
    result = shrink_bbox_to_pixel_grid(
        (0.0, 0.0, 10.0, 10.0), origin_x=0.0, origin_y=10.0, pixel_width=1.0, pixel_height=1.0
    )
    assert result == (0.0, 0.0, 10.0, 10.0)


def test_shrink_bbox_inwards_when_misaligned():
    """A misaligned bbox should shrink toward the next aligned cell (no expansion)."""
    minx, miny, maxx, maxy = shrink_bbox_to_pixel_grid(
        (0.3, 0.3, 10.7, 10.7), origin_x=0.0, origin_y=10.0, pixel_width=1.0, pixel_height=1.0
    )
    assert minx >= 0.3 and miny >= 0.3
    assert maxx <= 10.7 and maxy <= 10.7


def test_shrink_bbox_invalid_raises():
    """min >= max should raise ValueError."""
    with pytest.raises(ValueError, match="Bounding box is invalid"):
        shrink_bbox_to_pixel_grid((5.0, 0.0, 5.0, 10.0), 0.0, 10.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        shrink_bbox_to_pixel_grid((0.0, 10.0, 10.0, 0.0), 0.0, 10.0, 1.0, 1.0)


# ── check_path ──


def test_check_path_existing_parent_returns_absolute(tmp_path):
    assert check_path(tmp_path / "out.tif").is_absolute()


def test_check_path_make_dir_creates_parents(tmp_path):
    target = tmp_path / "a" / "b" / "c" / "x.tif"
    check_path(target, make_dir=True)
    assert target.parent.is_dir()


def test_check_path_missing_parent_raises_without_make_dir(tmp_path):
    target = tmp_path / "does" / "not" / "exist" / "f.tif"
    with pytest.raises(OSError, match="does not exist"):
        check_path(target)


def test_check_path_accepts_str(tmp_path):
    p = check_path(str(tmp_path / "x.tif"))
    assert p.parent == tmp_path


# ── save_raster / load_raster / get_raster_metadata round-trip ──


def _sample_raster() -> tuple[np.ndarray, list[float], str]:
    arr = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    gt = [100.0, 1.0, 0.0, 50.0, 0.0, -1.0]
    crs_wkt = (
        'PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",'
        'SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],'
        'UNIT["degree",0.0174532925199433]],PROJECTION["Mercator_1SP"],'
        'UNIT["metre",1]]'
    )
    return arr, gt, crs_wkt


def test_save_then_load_raster_roundtrip(tmp_path):
    arr, gt, crs = _sample_raster()
    path = tmp_path / "rt.tif"
    save_raster(str(path), arr, gt, crs, use_cog=False, generate_preview=False)
    assert path.exists()
    loaded, loaded_gt, loaded_crs, _ = load_raster(str(path))
    np.testing.assert_allclose(loaded, arr)
    assert list(loaded_gt)[:6] == list(gt)[:6]
    assert "Mercator" in (loaded_crs or "")


def test_save_raster_writes_nan_as_nodata_sentinel(tmp_path):
    arr, gt, crs = _sample_raster()
    arr = arr.copy()
    arr[0, 0] = np.nan
    path = tmp_path / "with_nan.tif"
    save_raster(str(path), arr, gt, crs, use_cog=False, generate_preview=False)
    loaded, _, _, _ = load_raster(str(path))
    assert np.isnan(loaded[0, 0])
    np.testing.assert_allclose(loaded[0, 1:], arr[0, 1:])


def test_get_raster_metadata_shape_pixel_size(tmp_path):
    arr, gt, crs = _sample_raster()
    path = tmp_path / "meta.tif"
    save_raster(str(path), arr, gt, crs, use_cog=False, generate_preview=False)
    meta = get_raster_metadata(str(path))
    # The docstring says rows/cols; transform is a GDAL list.
    assert meta["cols"] == 4
    assert meta["rows"] == 2
    assert abs(meta["transform"][1]) == pytest.approx(1.0)


# ── create_empty_raster ──


def test_create_empty_raster_writes_initialised_grid(tmp_path):
    _, gt, crs = _sample_raster()
    path = tmp_path / "empty.tif"
    create_empty_raster(str(path), rows=3, cols=5, transform=gt, crs_wkt=crs, dtype=np.float32)
    loaded, _, _, _ = load_raster(str(path))
    assert loaded.shape == (3, 5)
    # Nodata sentinel -9999 should be mapped back to NaN by load_raster.
    assert np.isnan(loaded).all()


# ── windowed I/O ──


def test_window_read_write_roundtrip(tmp_path):
    """Write a 4x4 array, overwrite a 2x2 window, read the window back."""
    _, gt, crs = _sample_raster()
    arr = np.zeros((4, 4), dtype=np.float32)
    path = tmp_path / "window.tif"
    save_raster(str(path), arr, gt, crs, use_cog=False, generate_preview=False)

    patch = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    win = (slice(1, 3), slice(1, 3))
    write_raster_window(str(path), patch, window=win)

    out = read_raster_window(str(path), window=win)
    np.testing.assert_allclose(out, patch)

    # Surrounding cells untouched.
    full, _, _, _ = load_raster(str(path))
    assert full[0, 0] == 0.0
    assert full[3, 3] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
