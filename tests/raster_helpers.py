"""Raster reads and writes for tests that work on either backend.

Tests must not import rasterio directly: on the GDAL backend it is fenced off
(see ``tests.backend_fence``). ``solweig.io.load_raster`` covers most needs;
these helpers cover what it does not: raw band values without the nodata to
NaN conversion, the TIFF layout, and writing a georeferenced input raster.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.backend_fence import gdal_backend_requested


def read_band(path: str | Path) -> np.ndarray:
    """Band 1 exactly as stored (nodata values are not replaced)."""
    if gdal_backend_requested():
        from osgeo import gdal

        ds = gdal.Open(str(path))
        if ds is None:
            raise FileNotFoundError(path)
        arr = ds.GetRasterBand(1).ReadAsArray()
        ds = None
        return arr
    import rasterio

    with rasterio.open(path) as src:
        return src.read(1)


def tiff_structure(path: str | Path) -> dict:
    """``{"compress": "deflate" | None, "block": (rows, cols), "tiled": bool}`` of a GeoTIFF."""
    if gdal_backend_requested():
        from osgeo import gdal

        ds = gdal.Open(str(path))
        if ds is None:
            raise FileNotFoundError(path)
        compress = ds.GetMetadataItem("COMPRESSION", "IMAGE_STRUCTURE")
        block_x, block_y = ds.GetRasterBand(1).GetBlockSize()
        width = ds.RasterXSize
        ds = None
        return {
            "compress": compress.lower() if compress else None,
            "block": (block_y, block_x),
            "tiled": block_x != width,
        }
    import rasterio

    with rasterio.open(path) as src:
        compress = src.profile.get("compress")
        block_y, block_x = src.block_shapes[0]
        return {
            "compress": str(compress).lower() if compress else None,
            "block": (block_y, block_x),
            "tiled": bool(src.profile.get("tiled", False)),
        }


def write_geotiff(
    path: str | Path,
    array: np.ndarray,
    *,
    origin_x: float,
    origin_y_top: float,
    pixel_size: float,
    epsg: int,
    nodata: float = -9999.0,
) -> None:
    """Write a north-up single-band float32 GeoTIFF."""
    array = np.asarray(array, dtype=np.float32)
    rows, cols = array.shape
    if gdal_backend_requested():
        from osgeo import gdal, osr

        ds = gdal.GetDriverByName("GTiff").Create(str(path), cols, rows, 1, gdal.GDT_Float32)
        ds.SetGeoTransform([origin_x, pixel_size, 0.0, origin_y_top, 0.0, -pixel_size])
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)
        ds.SetProjection(srs.ExportToWkt())
        band = ds.GetRasterBand(1)
        band.SetNoDataValue(nodata)
        band.WriteArray(array)
        ds = None
        return
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype="float32",
        transform=from_origin(origin_x, origin_y_top, pixel_size, pixel_size),
        crs=f"EPSG:{epsg}",
        nodata=nodata,
    ) as dst:
        dst.write(array, 1)
