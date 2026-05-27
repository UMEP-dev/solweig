"""Geographic location dataclass for sun-position calculations.

Extracted from `models/weather.py` so the weather module stays under
the 700-line hot-file threshold. The :class:`Location` symbol is
re-exported from :mod:`solweig.models.weather` and
:mod:`solweig.models` for backwards compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .._compat import GDAL_ENV
from ..solweig_logging import get_logger

if TYPE_CHECKING:
    from .surface import SurfaceData

logger = get_logger(__name__)


def _transform_to_wgs84(crs_wkt: str, x: float, y: float) -> tuple[float, float]:
    """Convert projected coordinates to WGS84 (lon, lat).

    Uses the same backend (pyproj or GDAL osr) that the rest of the
    package uses for raster I/O, so QGIS environments never touch pyproj.
    """
    if GDAL_ENV:
        from osgeo import osr

        src = osr.SpatialReference()
        src.ImportFromWkt(crs_wkt)
        dst = osr.SpatialReference()
        dst.ImportFromEPSG(4326)
        # osr may return (lat, lon) depending on axis order —
        # force traditional GIS (lon, lat) order.
        src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ct = osr.CoordinateTransformation(src, dst)
        lon, lat, _ = ct.TransformPoint(x, y)
    else:
        from pyproj import Transformer

        transformer = Transformer.from_crs(crs_wkt, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x, y)
    return lon, lat


@dataclass
class Location:
    """
    Geographic location for sun position calculations.

    Attributes:
        latitude: Latitude in degrees (north positive).
        longitude: Longitude in degrees (east positive).
        altitude: Altitude above sea level in meters. Default 0.
        utc_offset: UTC offset in hours. Default 0.
    """

    latitude: float
    longitude: float
    altitude: float = 0.0
    utc_offset: float = 0

    def __post_init__(self):
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"Latitude must be in [-90, 90], got {self.latitude}")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"Longitude must be in [-180, 180], got {self.longitude}")

    @classmethod
    def from_dsm_crs(cls, dsm_path: str | Path, utc_offset: float = 0, altitude: float = 0.0) -> Location:
        """
        Extract location from DSM raster's CRS by converting center point to WGS84.

        Args:
            dsm_path: Path to DSM GeoTIFF file with valid CRS.
            utc_offset: UTC offset in hours. Must be provided by user.
            altitude: Altitude above sea level in meters. Default 0.

        Returns:
            Location object with lat/lon from DSM center point.

        Raises:
            ValueError: If DSM has no CRS or CRS conversion fails.

        Example:
            location = Location.from_dsm_crs("dsm.tif", utc_offset=2)
        """
        from .. import io

        # Load DSM to get CRS and bounds
        _, transform, crs_wkt, _ = io.load_raster(str(dsm_path))

        if not crs_wkt:
            raise ValueError(
                f"DSM has no CRS metadata: {dsm_path}\n"
                f"Either:\n"
                f"  1. Add CRS to GeoTIFF: gdal_edit.py -a_srs EPSG:XXXXX {dsm_path}\n"
                f"  2. Provide location manually: Location(latitude=X, longitude=Y, utc_offset={utc_offset})"
            )

        # Get center point from geotransform
        # Transform is [x_origin, x_pixel_size, x_rotation, y_origin, y_rotation, y_pixel_size]
        # We need the raster dimensions to find center - load again to get shape
        dsm_array, _, _, _ = io.load_raster(str(dsm_path))
        rows, cols = dsm_array.shape

        center_x = transform[0] + (cols / 2) * transform[1]
        center_y = transform[3] + (rows / 2) * transform[5]

        # Convert to WGS84
        lon, lat = _transform_to_wgs84(crs_wkt, center_x, center_y)

        logger.info(f"Extracted location from DSM CRS: {lat:.4f}°N, {lon:.4f}°E (UTC{utc_offset:+g})")
        return cls(latitude=lat, longitude=lon, altitude=altitude, utc_offset=utc_offset)

    @classmethod
    def from_surface(cls, surface: SurfaceData, utc_offset: float | None = None, altitude: float = 0.0) -> Location:
        """
        Extract location from SurfaceData's CRS by converting center point to WGS84.

        This avoids reloading the DSM raster when you already have loaded SurfaceData.

        Args:
            surface: SurfaceData instance loaded from GeoTIFF.
            utc_offset: UTC offset in hours. If not provided, defaults to 0 with a warning.
                Always provide this explicitly for correct sun position calculations.
            altitude: Altitude above sea level in meters. Default 0.

        Returns:
            Location object with lat/lon from DSM center point.

        Raises:
            ValueError: If surface has no CRS metadata.
            ImportError: If pyproj is not installed.

        Example:
            surface = SurfaceData.from_geotiff("dsm.tif")
            location = Location.from_surface(surface, utc_offset=2)  # Athens: UTC+2
        """
        import warnings

        # Check if geotransform and CRS are available
        if surface.geotransform is None:
            raise ValueError(
                "Surface data has no geotransform metadata.\n"
                "Load surface with SurfaceData.from_geotiff() or provide location manually."
            )
        if surface.crs is None:
            raise ValueError(
                "Surface data has no CRS metadata.\n"
                "Provide location manually: Location(latitude=X, longitude=Y, utc_offset=0)"
            )

        transform = surface.geotransform
        crs_wkt = surface.crs
        rows, cols = surface.dsm.shape

        # Get center point from geotransform
        # Transform is [x_origin, x_pixel_size, x_rotation, y_origin, y_rotation, y_pixel_size]
        center_x = transform[0] + (cols / 2) * transform[1]
        center_y = transform[3] + (rows / 2) * transform[5]

        # Convert to WGS84
        lon, lat = _transform_to_wgs84(crs_wkt, center_x, center_y)

        # Warn if utc_offset not explicitly provided
        if utc_offset is None:
            warnings.warn(
                f"UTC offset not specified for auto-extracted location ({lat:.4f}°N, {lon:.4f}°E).\n"
                f"Defaulting to UTC+0, which may cause incorrect sun positions.\n"
                f"Fix: Location.from_surface(surface, utc_offset=YOUR_OFFSET) or\n"
                f"     Location(latitude={lat:.4f}, longitude={lon:.4f}, utc_offset=YOUR_OFFSET)",
                UserWarning,
                stacklevel=2,
            )
            utc_offset = 0

        logger.debug(f"Auto-extracted location: {lat:.4f}°N, {lon:.4f}°E (UTC{utc_offset:+g})")
        return cls(latitude=lat, longitude=lon, altitude=altitude, utc_offset=utc_offset)

    @classmethod
    def from_epw(cls, path: str | Path) -> Location:
        """
        Extract location from an EPW weather file header.

        The EPW LOCATION line contains latitude, longitude, timezone offset,
        and elevation — everything needed for a complete Location.

        Args:
            path: Path to the EPW file.

        Returns:
            Location with lat, lon, utc_offset, and altitude from the EPW header.

        Raises:
            FileNotFoundError: If the EPW file doesn't exist.
            ValueError: If the EPW header is malformed.

        Example:
            location = Location.from_epw("madrid.epw")
            # Location(latitude=40.45, longitude=-3.55, altitude=667.0, utc_offset=1)
        """
        from .. import io as common

        metadata = common._parse_epw_metadata(Path(path))
        utc_offset = float(metadata["tz_offset"])

        logger.info(
            f"Location from EPW: {metadata['city']} — "
            f"{metadata['latitude']:.4f}°N, {metadata['longitude']:.4f}°E "
            f"(UTC{utc_offset:+g}, {metadata['elevation']:.0f}m)"
        )
        return cls(
            latitude=metadata["latitude"],
            longitude=metadata["longitude"],
            altitude=metadata["elevation"],
            utc_offset=utc_offset,
        )

    def to_sun_position_dict(self) -> dict:
        """Convert to dict format expected by sun_position module."""
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
        }
