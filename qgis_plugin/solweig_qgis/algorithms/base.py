"""
Base algorithm class for SOLWEIG processing algorithms.

Provides shared utilities for loading rasters, saving outputs,
and integrating with QGIS.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from osgeo import gdal
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProject,
    QgsRasterLayer,
)
from qgis.PyQt.QtCore import QCoreApplication

if TYPE_CHECKING:
    from numpy.typing import NDArray


def format_solweig_error(prefix: str, error: Exception) -> str:
    """Build a QgsProcessingException message that exposes the structured
    attributes of a ``solweig.SolweigError`` subclass.

    Core error types (``GridShapeMismatch``, ``InvalidSurfaceData``,
    ``MissingPrecomputedData``, ``WeatherDataError``, ``ConfigurationError``)
    carry extra fields (``field``, ``expected``, ``got``, ``what``,
    ``suggestion``, ``parameter``, ``reason``) that carry the actionable
    content the user needs to fix the input. The plain ``str(error)`` usually
    covers them, but we also surface the error type name and any structured
    attributes that are set, so the QGIS error dialog is useful on its own.
    """
    parts = [f"{prefix}: {type(error).__name__}: {error}"]
    for attr in ("field", "expected", "got", "what", "suggestion", "parameter", "reason", "value"):
        val = getattr(error, attr, None)
        if val is not None and val != "":
            parts.append(f"  {attr}: {val}")
    return "\n".join(parts)


class SolweigAlgorithmBase(QgsProcessingAlgorithm):
    """
    Base class for all SOLWEIG processing algorithms.

    Provides common functionality:
    - Raster loading via GDAL
    - Georeferenced output saving
    - Thermal comfort styling for outputs
    - Translation support
    """

    def tr(self, string: str) -> str:
        """Translate string to current locale."""
        return QCoreApplication.translate("SolweigProcessing", string)

    def createInstance(self):
        """Return new instance of algorithm."""
        return self.__class__()

    def group(self) -> str:
        """Return algorithm group name (empty = directly under provider)."""
        return ""

    def groupId(self) -> str:
        """Return algorithm group ID (empty = directly under provider)."""
        return ""

    def helpUrl(self) -> str:
        """Return URL to algorithm documentation."""
        return "https://umep-dev.github.io/solweig/guide/qgis-plugin/"

    # -------------------------------------------------------------------------
    # SOLWEIG Import Helper
    # -------------------------------------------------------------------------

    def import_solweig(self):
        """
        Import the solweig library.

        Returns:
            The imported solweig module.

        Raises:
            QgsProcessingException: If solweig cannot be imported.
        """
        try:
            from .. import check_dependencies

            success, message = check_dependencies()
            if not success:
                raise QgsProcessingException(message)

            import solweig

            return solweig
        except QgsProcessingException:
            raise
        except Exception as e:
            raise QgsProcessingException("SOLWEIG library not found. Install it with:  pip install solweig") from e

    # -------------------------------------------------------------------------
    # Raster Loading
    # -------------------------------------------------------------------------

    def load_raster_from_layer(self, layer: QgsRasterLayer) -> tuple[NDArray[np.floating], list[float], str]:
        """
        Load QGIS raster layer to numpy array using GDAL.

        Args:
            layer: QGIS raster layer to load.

        Returns:
            tuple of (array, geotransform, crs_wkt):
                - array: 2D numpy float32 array
                - geotransform: GDAL 6-tuple [x_origin, x_res, 0, y_origin, 0, -y_res]
                - crs_wkt: Coordinate reference system as WKT string

        Raises:
            QgsProcessingException: If raster cannot be opened.
        """
        source = layer.source()
        ds = gdal.Open(source, gdal.GA_ReadOnly)
        if ds is None:
            raise QgsProcessingException(f"Cannot open raster: {source}")

        try:
            band = ds.GetRasterBand(1)
            array = band.ReadAsArray().astype(np.float32)

            # Match solweig.io.load_raster(): honor any explicit non-NaN nodata
            # sentinel so QGIS and local API runs see the same valid mask.
            nodata = band.GetNoDataValue()
            if nodata is not None and not np.isnan(nodata):
                array = np.where(array == nodata, np.nan, array)

            geotransform = list(ds.GetGeoTransform())
            crs_wkt = ds.GetProjection()

            return array, geotransform, crs_wkt
        finally:
            ds = None  # Close dataset

    def load_optional_raster(
        self,
        parameters: dict[str, Any],
        param_name: str,
        context: QgsProcessingContext,
    ) -> NDArray[np.floating] | None:
        """
        Load optional raster parameter, return None if not provided.

        Args:
            parameters: Algorithm parameters dict.
            param_name: Name of the raster parameter.
            context: Processing context.

        Returns:
            Numpy array if parameter provided, None otherwise.
        """
        if param_name not in parameters or not parameters[param_name]:
            return None

        layer = self.parameterAsRasterLayer(parameters, param_name, context)
        if layer is None:
            return None

        array, _, _ = self.load_raster_from_layer(layer)
        return array

    def get_pixel_size_from_layer(self, layer: QgsRasterLayer) -> float:
        """
        Extract pixel size from raster layer.

        Args:
            layer: QGIS raster layer.

        Returns:
            Pixel size in meters (assumes square pixels).
        """
        source = layer.source()
        ds = gdal.Open(source, gdal.GA_ReadOnly)
        if ds is None:
            raise QgsProcessingException(f"Cannot open raster: {source}")

        try:
            gt = ds.GetGeoTransform()
            # gt[1] is x pixel size, gt[5] is y pixel size (negative)
            pixel_size = abs(gt[1])
            return pixel_size
        finally:
            ds = None

    # -------------------------------------------------------------------------
    # Output Saving
    # -------------------------------------------------------------------------

    def save_georeferenced_output(
        self,
        array: NDArray[np.floating],
        output_path: str | Path,
        geotransform: list[float],
        crs_wkt: str,
        nodata: float = -9999.0,
        feedback: QgsProcessingFeedback | None = None,
    ) -> str:
        """
        Save numpy array to GeoTIFF with proper georeferencing.

        Uses Cloud-Optimized GeoTIFF (COG) format with LZW compression.

        Args:
            array: 2D numpy array to save.
            output_path: Path for output GeoTIFF.
            geotransform: GDAL geotransform [x_origin, x_res, 0, y_origin, 0, -y_res].
            crs_wkt: Coordinate reference system as WKT string.
            nodata: NoData value to use. Default -9999.
            feedback: Optional feedback for progress reporting.

        Returns:
            Path to saved file.
        """
        output_path = str(output_path)

        # Replace NaN with nodata
        array_out = np.where(np.isnan(array), nodata, array).astype(np.float32)

        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Create GeoTIFF
        driver = gdal.GetDriverByName("GTiff")
        rows, cols = array_out.shape
        ds = driver.Create(
            output_path,
            cols,
            rows,
            1,  # bands
            gdal.GDT_Float32,
            options=["COMPRESS=LZW", "TILED=YES"],
        )

        if ds is None:
            raise QgsProcessingException(f"Cannot create output raster: {output_path}")

        try:
            ds.SetGeoTransform(geotransform)
            ds.SetProjection(crs_wkt)

            band = ds.GetRasterBand(1)
            band.WriteArray(array_out)
            band.SetNoDataValue(nodata)
            band.FlushCache()
        finally:
            ds = None  # Close and flush

        if feedback:
            feedback.pushInfo(f"Saved: {output_path}")

        return output_path

    def get_output_path(
        self,
        parameters: dict[str, Any],
        param_name: str,
        default_name: str,
        context: QgsProcessingContext,
    ) -> str:
        """
        Get output path from parameter or create temp file.

        Args:
            parameters: Algorithm parameters.
            param_name: Output parameter name.
            default_name: Default filename if not specified.
            context: Processing context.

        Returns:
            Path for output file.
        """
        if param_name in parameters and parameters[param_name]:
            output_dest = self.parameterAsOutputLayer(parameters, param_name, context)
            if output_dest:
                return output_dest

        # Create temp file
        temp_dir = Path(tempfile.gettempdir()) / "solweig_qgis_output"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return str(temp_dir / default_name)

    # -------------------------------------------------------------------------
    # Canvas Integration
    # -------------------------------------------------------------------------

    def add_raster_to_canvas(
        self,
        path: str,
        layer_name: str,
        style: str | None = None,
        context: QgsProcessingContext | None = None,
    ) -> QgsRasterLayer:
        """
        Add raster layer to QGIS canvas with optional styling.

        Args:
            path: Path to raster file.
            layer_name: Display name in layer panel.
            style: Style preset ('tmrt', 'utci', 'pet', 'shadow', or None).
            context: Processing context.

        Returns:
            The created QgsRasterLayer.

        Raises:
            QgsProcessingException: If layer cannot be loaded.
        """
        layer = QgsRasterLayer(path, layer_name)
        if not layer.isValid():
            raise QgsProcessingException(f"Cannot load output layer: {path}")

        # Apply thermal comfort color ramp if requested
        if style in ("tmrt", "utci", "pet"):
            self.apply_thermal_comfort_style(layer, style)
        elif style == "shadow":
            self.apply_shadow_style(layer)

        # Add to project
        QgsProject.instance().addMapLayer(layer)

        return layer

    def apply_thermal_comfort_style(self, layer: QgsRasterLayer, style_type: str) -> None:
        """
        Apply thermal comfort color ramp for visualization.

        Args:
            layer: QgsRasterLayer to style.
            style_type: 'tmrt', 'utci', or 'pet'.
        """
        from qgis.core import (
            QgsColorRampShader,
            QgsRasterShader,
            QgsSingleBandPseudoColorRenderer,
        )
        from qgis.PyQt.QtGui import QColor

        # Define color ramps based on style type
        if style_type == "utci":
            # UTCI thermal stress categories (ISO 7730 / Jendritzky et al. 2012)
            color_points = [
                (-40, QColor(0, 0, 128), "Extreme cold stress"),
                (-27, QColor(0, 100, 200), "Very strong cold stress"),
                (-13, QColor(51, 153, 255), "Strong cold stress"),
                (0, QColor(153, 204, 255), "Moderate cold stress"),
                (9, QColor(204, 255, 204), "Slight cold stress"),
                (26, QColor(255, 255, 102), "No thermal stress"),
                (32, QColor(255, 204, 51), "Moderate heat stress"),
                (38, QColor(255, 128, 0), "Strong heat stress"),
                (46, QColor(255, 51, 51), "Very strong heat stress"),
                (60, QColor(128, 0, 0), "Extreme heat stress"),
            ]
        else:  # tmrt, pet - use generic thermal ramp
            color_points = [
                (0, QColor(0, 0, 200), "Cold"),
                (15, QColor(51, 153, 255), "Cool"),
                (25, QColor(153, 255, 153), "Comfortable"),
                (35, QColor(255, 255, 102), "Warm"),
                (45, QColor(255, 153, 51), "Hot"),
                (55, QColor(255, 51, 51), "Very hot"),
                (70, QColor(128, 0, 0), "Extreme"),
            ]

        # Create shader
        shader = QgsRasterShader()
        ramp_shader = QgsColorRampShader()
        ramp_shader.setColorRampType(QgsColorRampShader.Type.Interpolated)

        items = []
        for value, color, label in color_points:
            items.append(QgsColorRampShader.ColorRampItem(value, color, label))

        ramp_shader.setColorRampItemList(items)
        shader.setRasterShaderFunction(ramp_shader)

        # Apply renderer
        renderer = QgsSingleBandPseudoColorRenderer(
            layer.dataProvider(),
            1,  # band
            shader,
        )
        layer.setRenderer(renderer)
        layer.triggerRepaint()

    def apply_shadow_style(self, layer: QgsRasterLayer) -> None:
        """
        Apply shadow mask styling (binary: sunlit/shadow).

        Args:
            layer: QgsRasterLayer to style.
        """
        from qgis.core import (
            QgsColorRampShader,
            QgsRasterShader,
            QgsSingleBandPseudoColorRenderer,
        )
        from qgis.PyQt.QtGui import QColor

        shader = QgsRasterShader()
        ramp_shader = QgsColorRampShader()
        ramp_shader.setColorRampType(QgsColorRampShader.Type.Interpolated)

        items = [
            QgsColorRampShader.ColorRampItem(0, QColor(255, 255, 153), "Sunlit"),
            QgsColorRampShader.ColorRampItem(1, QColor(102, 102, 102), "Shadow"),
        ]

        ramp_shader.setColorRampItemList(items)
        shader.setRasterShaderFunction(ramp_shader)

        renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
        layer.setRenderer(renderer)
        layer.triggerRepaint()

    # -------------------------------------------------------------------------
    # Validation Helpers
    # -------------------------------------------------------------------------

    def check_grid_shapes_match(
        self,
        reference_shape: tuple[int, int],
        arrays: dict[str, NDArray | None],
        feedback: QgsProcessingFeedback,
    ) -> None:
        """
        Verify all provided arrays match reference shape.

        Args:
            reference_shape: Expected (rows, cols) shape.
            arrays: Dict of {name: array} to check (None values skipped).
            feedback: For reporting errors.

        Raises:
            QgsProcessingException: If shapes don't match.
        """
        for name, arr in arrays.items():
            if arr is not None and arr.shape != reference_shape:
                raise QgsProcessingException(
                    f"Grid shape mismatch: {name} has shape {arr.shape}, expected {reference_shape} (matching DSM)"
                )
