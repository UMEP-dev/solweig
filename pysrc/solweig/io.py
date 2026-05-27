"""Raster I/O and EPW weather file handling.

Provides functions for reading and writing GeoTIFF rasters, parsing
EnergyPlus Weather (EPW) files, and downloading TMY data from PVGIS.
Automatically selects between rasterio and GDAL backends (see ``_compat``).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ._compat import GDAL_ENV
from .buffers import as_float32

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Conditional imports based on the backend chosen in _compat
if GDAL_ENV:
    from osgeo import gdal
else:
    import pyproj
    import rasterio
    from rasterio.features import rasterize
    from rasterio.mask import mask
    from rasterio.transform import Affine, from_origin
    from rasterio.windows import Window
    from shapely import geometry


FLOAT_TOLERANCE = 1e-9


def _assert_north_up(transform) -> None:
    """Ensure the raster transform describes a north-up raster."""
    if hasattr(transform, "b") and hasattr(transform, "d"):
        if not math.isclose(transform.b, 0.0, abs_tol=FLOAT_TOLERANCE) or not math.isclose(
            transform.d, 0.0, abs_tol=FLOAT_TOLERANCE
        ):
            raise ValueError("Only north-up rasters (no rotation) are supported.")
    else:
        # GDAL-style tuple (c, a, b, f, d, e)
        if len(transform) < 6:
            raise ValueError("Transform must contain 6 elements.")
        if not math.isclose(transform[2], 0.0, abs_tol=FLOAT_TOLERANCE) or not math.isclose(
            transform[4], 0.0, abs_tol=FLOAT_TOLERANCE
        ):
            raise ValueError("Only north-up rasters (no rotation) are supported.")


def _shrink_axis_to_grid(min_val: float, max_val: float, origin: float, pixel_size: float) -> tuple[float, float]:
    if pixel_size == 0:
        raise ValueError("Pixel size must be non-zero to shrink bbox to pixel grid.")
    step = abs(pixel_size)
    start_idx = math.ceil(((min_val - origin) / step) - FLOAT_TOLERANCE)
    end_idx = math.floor(((max_val - origin) / step) + FLOAT_TOLERANCE)
    new_min = origin + start_idx * step
    new_max = origin + end_idx * step
    if not new_max > new_min:
        raise ValueError("Bounding box collapsed after snapping to the pixel grid.")
    return new_min, new_max


def shrink_bbox_to_pixel_grid(
    bbox: tuple[float, float, float, float],
    origin_x: float,
    origin_y: float,
    pixel_width: float,
    pixel_height: float,
) -> tuple[float, float, float, float]:
    """Shrink bbox so its edges land on the pixel grid defined by the raster origin."""

    minx, miny, maxx, maxy = bbox
    if minx >= maxx or miny >= maxy:
        raise ValueError("Bounding box is invalid (min must be < max for both axes).")
    snapped_minx, snapped_maxx = _shrink_axis_to_grid(minx, maxx, origin_x, pixel_width)
    snapped_miny, snapped_maxy = _shrink_axis_to_grid(miny, maxy, origin_y, pixel_height)
    return snapped_minx, snapped_miny, snapped_maxx, snapped_maxy


def _bounds_to_tuple(bounds) -> tuple[float, float, float, float]:
    if hasattr(bounds, "left"):
        return bounds.left, bounds.bottom, bounds.right, bounds.top
    return tuple(bounds)


def _validate_bbox_within_bounds(
    bbox: tuple[float, float, float, float], bounds, *, tol: float = FLOAT_TOLERANCE
) -> None:
    minx, miny, maxx, maxy = bbox
    left, bottom, right, top = _bounds_to_tuple(bounds)
    if minx < left - tol or maxx > right + tol or miny < bottom - tol or maxy > top + tol:
        raise ValueError("Bounding box is not fully contained within the raster dataset bounds")


def _compute_bounds_from_transform(transform, width: int, height: int) -> tuple[float, float, float, float]:
    """Return raster bounds for a GDAL-style transform tuple."""
    left = transform[0]
    top = transform[3]
    right = transform[0] + width * transform[1]
    bottom = transform[3] + height * transform[5]
    minx = min(left, right)
    maxx = max(left, right)
    miny = min(top, bottom)
    maxy = max(top, bottom)
    return minx, miny, maxx, maxy


def _normalise_bbox(bbox_sequence) -> tuple[float, float, float, float]:
    try:
        minx, miny, maxx, maxy = bbox_sequence
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Bounding box must contain exactly four numeric values") from exc
    return float(minx), float(miny), float(maxx), float(maxy)


def rasterise_gdf(
    gdf: Any,
    geom_col: str,
    ht_col: str,
    bbox: tuple[float, float, float, float] | list[float] | None = None,
    pixel_size: float = 1.0,
) -> tuple[NDArray[np.float32], Any]:
    """Burn vector geometries into a raster, keyed by an attribute column.

    Args:
        gdf: GeoDataFrame containing polygon geometries.
        geom_col: Column name holding the geometry.
        ht_col: Column name holding the per-feature value (e.g. building height).
        bbox: Optional (minx, miny, maxx, maxy) bounding box. Defaults to
            the GeoDataFrame's total bounds.
        pixel_size: Output pixel size in CRS units. Default 1.0.

    Returns:
        ``(raster, transform)`` — ``raster`` is a ``float32`` numpy array shaped
        ``(height, width)``; ``transform`` is the corresponding rasterio
        :class:`Affine`.

    Raises:
        ValueError: If ``pixel_size`` is non-positive or the bounding box
            collapses after pixel-grid snapping.
    """
    # Define raster parameters
    if bbox is not None:
        # Unpack bbox values
        minx, miny, maxx, maxy = _normalise_bbox(bbox)
    else:
        # Use the total bounds of the GeoDataFrame
        minx, miny, maxx, maxy = map(float, gdf.total_bounds)
    if pixel_size <= 0:
        raise ValueError("Pixel size must be a positive number.")
    minx, miny, maxx, maxy = shrink_bbox_to_pixel_grid(
        (minx, miny, maxx, maxy),
        origin_x=minx,
        origin_y=maxy,
        pixel_width=pixel_size,
        pixel_height=pixel_size,
    )
    width = int(round((maxx - minx) / pixel_size))
    height = int(round((maxy - miny) / pixel_size))
    if width <= 0 or height <= 0:
        raise ValueError("Bounding box collapsed after snapping to pixel grid.")
    transform = from_origin(minx, maxy, pixel_size, pixel_size)
    # Create a blank array for the raster
    raster = np.zeros((height, width), dtype=np.float32)
    # Burn geometries into the raster
    shapes = ((geom, value) for geom, value in zip(gdf[geom_col], gdf[ht_col], strict=False))
    raster = rasterize(shapes, out_shape=raster.shape, transform=transform, fill=0, dtype=np.float32)

    return raster, transform


def check_path(path_str: str | Path, make_dir: bool = False) -> Path:
    """Resolve ``path_str`` to an absolute :class:`Path`, optionally creating its parent.

    Args:
        path_str: Path-like input.
        make_dir: If True, create the parent directory chain if missing.

    Returns:
        Absolute :class:`Path`.

    Raises:
        OSError: If the parent doesn't exist and ``make_dir`` is False.
    """
    # Ensure path exists
    path = Path(path_str).absolute()
    if not path.parent.exists():
        if make_dir:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            raise OSError(
                f"Parent directory {path.parent} does not exist for path {path}. Set make_dir=True to create it."
            )
    if path.is_dir():
        # Already an existing directory; nothing to do.
        pass
    elif not path.exists() and not path.suffix:
        # Looks like a directory path (no file extension); create or raise.
        if make_dir:
            path.mkdir(parents=True, exist_ok=True)
        else:
            raise OSError(f"Path {path} does not exist. Set make_dir=True to create it.")
    return path


# Preview PNG helpers were extracted to io_preview.py to keep this module
# under the 700-line hot-file threshold. Re-export for internal callers
# (save_raster references _generate_preview_png by its short name).
from .io_preview import _generate_preview_png, _get_preview_range  # noqa: E402, F401


def save_raster(
    out_path_str: str,
    data_arr: np.ndarray,
    trf_arr: list[float],
    crs_wkt: str | None,
    no_data_val: float = -9999,
    ensure_float32: bool = True,
    use_cog: bool = True,
    generate_preview: bool = True,
):
    """
    Save raster to GeoTIFF (Cloud-Optimized by default).

    Args:
        out_path_str: Output file path
        data_arr: 2D numpy array to save
        trf_arr: GDAL-style geotransform [top_left_x, pixel_width, rotation, top_left_y, rotation, pixel_height]
        crs_wkt: CRS in WKT format
        no_data_val: No-data value to use
        ensure_float32: If True, ensure array is float32 before saving
                        (default: True — converts any non-float32 dtype)
        use_cog: If True, save as Cloud-Optimized GeoTIFF with built-in overviews
                 (default: True for better OS thumbnail support)
        generate_preview: If True, generate a sidecar .preview.png file for OS thumbnails
                         (default: True for float data that can't be previewed directly)
    """
    if ensure_float32:
        data_arr = as_float32(data_arr)

    attempts = 2
    while attempts > 0:
        attempts -= 1
        try:
            out_path = check_path(out_path_str, make_dir=True)
            height, width = data_arr.shape

            if GDAL_ENV is False:
                trf = Affine.from_gdal(*trf_arr)
                crs = None
                if crs_wkt:
                    crs = pyproj.CRS(crs_wkt)

                if use_cog:
                    # Write as Cloud-Optimized GeoTIFF
                    # COG driver creates overviews automatically
                    from rasterio.io import MemoryFile

                    # Create in memory first, then write as COG
                    memfile = MemoryFile()
                    with memfile.open(
                        driver="GTiff",
                        height=height,
                        width=width,
                        count=1,
                        dtype=data_arr.dtype,
                        crs=crs,
                        transform=trf,
                        nodata=no_data_val,
                    ) as mem:
                        mem.write(data_arr, 1)

                    # Now copy to COG format
                    from rasterio.shutil import copy

                    with memfile.open() as src:
                        copy(
                            src,
                            out_path,
                            driver="COG",
                            overview_resampling="average",
                        )
                    memfile.close()
                    logger.debug(f"Saved COG: {out_path}")
                else:
                    # Standard GeoTIFF
                    with rasterio.open(
                        out_path,
                        "w",
                        driver="GTiff",
                        height=height,
                        width=width,
                        count=1,
                        dtype=data_arr.dtype,
                        crs=crs,
                        transform=trf,
                        nodata=no_data_val,
                    ) as dst:
                        dst.write(data_arr, 1)
            else:
                # GDAL backend
                if use_cog:
                    # Use COG driver (GDAL 3.1+)
                    driver = gdal.GetDriverByName("COG")
                    if driver is None:
                        # Fallback to GTiff with overviews if COG driver not available
                        logger.warning("COG driver not available, using GTiff with overviews")
                        driver = gdal.GetDriverByName("GTiff")
                        options = ["TILED=YES"]
                        ds = driver.Create(str(out_path), width, height, 1, gdal.GDT_Float32, options)
                        ds.SetGeoTransform(trf_arr)
                        if crs_wkt:
                            ds.SetProjection(crs_wkt)
                        band = ds.GetRasterBand(1)
                        band.SetNoDataValue(no_data_val)
                        band.WriteArray(data_arr)
                        # Build overviews
                        if min(height, width) > 256:
                            overview_levels = []
                            size = min(height, width)
                            level = 2
                            while size // level > 128:
                                overview_levels.append(level)
                                level *= 2
                            if overview_levels:
                                ds.BuildOverviews("AVERAGE", overview_levels)
                        ds = None
                    else:
                        # COG driver requires creating via CreateCopy from memory dataset
                        mem_driver = gdal.GetDriverByName("MEM")
                        mem_ds = mem_driver.Create("", width, height, 1, gdal.GDT_Float32)
                        mem_ds.SetGeoTransform(trf_arr)
                        if crs_wkt:
                            mem_ds.SetProjection(crs_wkt)
                        band = mem_ds.GetRasterBand(1)
                        band.SetNoDataValue(no_data_val)
                        band.WriteArray(data_arr)

                        # Copy to COG
                        cog_options = ["OVERVIEW_RESAMPLING=AVERAGE"]
                        driver.CreateCopy(str(out_path), mem_ds, options=cog_options)
                        mem_ds = None
                        logger.debug(f"Saved COG: {out_path}")
                else:
                    # Standard GeoTIFF
                    driver = gdal.GetDriverByName("GTiff")
                    ds = driver.Create(str(out_path), width, height, 1, gdal.GDT_Float32)
                    ds.SetGeoTransform(trf_arr)
                    if crs_wkt:
                        ds.SetProjection(crs_wkt)
                    band = ds.GetRasterBand(1)
                    band.SetNoDataValue(no_data_val)
                    band.WriteArray(data_arr)
                    ds = None

            # Generate sidecar preview PNG for float data (OS can't render float GeoTIFFs)
            if generate_preview and np.issubdtype(data_arr.dtype, np.floating):
                _generate_preview_png(data_arr, out_path)

            return
        except Exception as e:
            if attempts == 0:
                raise e
            logger.warning(f"Failed to save raster to {out_path_str}: {e}. Retrying...")


def get_raster_metadata(path_str: str | Path) -> dict:
    """
    Get raster metadata without loading the whole file.
    Returns dict with keys: rows, cols, transform, crs, nodata, res.
    Transform is always a list [c, a, b, f, d, e] (GDAL-style).
    CRS is always a WKT string (or None).
    """
    path = check_path(path_str)
    if GDAL_ENV is False:
        with rasterio.open(path) as src:
            # Convert Affine to GDAL-style list
            trf = src.transform
            transform_list = [trf.c, trf.a, trf.b, trf.f, trf.d, trf.e]
            # Convert CRS to WKT string
            crs_wkt = src.crs.to_wkt() if src.crs is not None else None
            return {
                "rows": src.height,
                "cols": src.width,
                "transform": transform_list,
                "crs": crs_wkt,
                "nodata": src.nodata,
                "res": src.res,  # (xres, yres)
                "bounds": src.bounds,
            }
    else:
        ds = gdal.Open(str(path))
        if ds is None:
            raise OSError(f"Could not open {path}")
        gt = ds.GetGeoTransform()
        return {
            "rows": ds.RasterYSize,
            "cols": ds.RasterXSize,
            "transform": gt,
            "crs": ds.GetProjection() or None,
            "nodata": ds.GetRasterBand(1).GetNoDataValue(),
            "res": (gt[1], abs(gt[5])),  # Approximate resolution
        }


def read_raster_window(path_str: str | Path, window: tuple[slice, slice], band: int = 1) -> np.ndarray:
    """
    Read a window from a raster file.
    window is (row_slice, col_slice).
    """
    path = check_path(path_str)
    row_slice, col_slice = window

    # Handle None slices (read full dimension)
    # This is tricky without knowing full shape, so we assume caller provides valid slices
    # or we'd need to open file to check shape first.
    # For now, assume valid integer slices.

    if GDAL_ENV is False:
        with rasterio.open(path) as src:
            # rasterio Window(col_off, row_off, width, height)
            # Slices are start:stop
            r_start = row_slice.start if row_slice.start is not None else 0
            r_stop = row_slice.stop if row_slice.stop is not None else src.height
            c_start = col_slice.start if col_slice.start is not None else 0
            c_stop = col_slice.stop if col_slice.stop is not None else src.width

            win = Window(c_start, r_start, c_stop - c_start, r_stop - r_start)  # type: ignore[too-many-positional-arguments]
            return src.read(band, window=win)
    else:
        ds = gdal.Open(str(path))
        if ds is None:
            raise OSError(f"Could not open {path}")

        r_start = row_slice.start if row_slice.start is not None else 0
        r_stop = row_slice.stop if row_slice.stop is not None else ds.RasterYSize
        c_start = col_slice.start if col_slice.start is not None else 0
        c_stop = col_slice.stop if col_slice.stop is not None else ds.RasterXSize

        xoff = c_start
        yoff = r_start
        xsize = c_stop - c_start
        ysize = r_stop - r_start

        return ds.GetRasterBand(band).ReadAsArray(xoff, yoff, xsize, ysize)


def load_raster(
    path_str: str, bbox: list[float] | None = None, band: int = 0, ensure_float32: bool = True
) -> tuple[np.ndarray, list[float], str | None, float | None]:
    """
    Load raster, optionally crop to bbox.

    Args:
        path_str: Path to raster file
        bbox: Optional bounding box [minx, miny, maxx, maxy]
        band: Band index to read (0-based)
        ensure_float32: If True, ensure output array is float32
                        (default: True — converts any non-float32 dtype including integers)

    Returns:
        Tuple of (array, transform, crs_wkt, no_data_value)
    """
    # Load raster, optionally crop to bbox
    path = check_path(path_str, make_dir=False)
    if not path.exists():
        raise FileNotFoundError(f"Raster file {path} does not exist.")
    if GDAL_ENV is False:
        with rasterio.open(path) as dataset:
            _assert_north_up(dataset.transform)
            crs_wkt = dataset.crs.to_wkt() if dataset.crs is not None else None
            no_data_val = dataset.nodata
            transform = dataset.transform
            if bbox is not None:
                bbox_tuple = _normalise_bbox(bbox)
                snapped_bbox = shrink_bbox_to_pixel_grid(
                    bbox_tuple,
                    origin_x=transform.c,
                    origin_y=transform.f,
                    pixel_width=transform.a,
                    pixel_height=transform.e,
                )
                _validate_bbox_within_bounds(snapped_bbox, dataset.bounds)
                bbox_geom = geometry.box(*snapped_bbox)
                rast, trf = mask(dataset, [bbox_geom], crop=True)
            else:
                rast = dataset.read()
                trf = transform
            # Convert rasterio Affine to GDAL-style list
            trf_arr = [trf.c, trf.a, trf.b, trf.f, trf.d, trf.e]
            # rast shape: (bands, rows, cols)
            if rast.ndim == 3:
                if band < 0 or band >= rast.shape[0]:
                    raise IndexError(f"Requested band {band} out of range; raster has {rast.shape[0]} band(s)")
                rast_arr = rast[band]
            else:
                rast_arr = rast
    else:
        dataset = gdal.Open(str(path))
        if dataset is None:
            raise FileNotFoundError(f"Could not open {path}")
        trf = dataset.GetGeoTransform()
        _assert_north_up(trf)
        # GetProjection returns WKT string (or empty string)
        crs_wkt = dataset.GetProjection() or None
        rb = dataset.GetRasterBand(band + 1)
        if rb is None:
            dataset = None
            raise IndexError(f"Requested band {band} out of range in GDAL dataset")
        rast_arr = rb.ReadAsArray()
        no_data_val = rb.GetNoDataValue()
        if bbox is not None:
            bbox_tuple = _normalise_bbox(bbox)
            snapped_bbox = shrink_bbox_to_pixel_grid(
                bbox_tuple,
                origin_x=trf[0],
                origin_y=trf[3],
                pixel_width=trf[1],
                pixel_height=trf[5],
            )
            bounds = _compute_bounds_from_transform(trf, dataset.RasterXSize, dataset.RasterYSize)
            _validate_bbox_within_bounds(snapped_bbox, bounds)
            min_x, min_y, max_x, max_y = snapped_bbox
            pixel_width = trf[1]
            pixel_height = abs(trf[5])
            xoff = int(round((min_x - trf[0]) / pixel_width))
            yoff = int(round((trf[3] - max_y) / pixel_height))
            xsize = int(round((max_x - min_x) / pixel_width))
            ysize = int(round((max_y - min_y) / pixel_height))
            # guard offsets/sizes
            if xoff < 0 or yoff < 0 or xsize <= 0 or ysize <= 0:
                dataset = None
                raise ValueError("Computed window from bbox is out of raster bounds or invalid")
            rast_arr = rast_arr[yoff : yoff + ysize, xoff : xoff + xsize]
            trf_arr = [min_x, trf[1], 0, max_y, 0, trf[5]]
        else:
            trf_arr = [trf[0], trf[1], 0, trf[3], 0, trf[5]]
        dataset = None  # ensure dataset closed
    if ensure_float32:
        rast_arr = as_float32(rast_arr)
    # Handle no-data (support NaN)
    if no_data_val is not None and not np.isnan(no_data_val):
        logger.info(f"No-data value is {no_data_val}, replacing with NaN")
        if not np.issubdtype(rast_arr.dtype, np.floating):
            rast_arr = rast_arr.astype(np.float32)
        rast_arr[rast_arr == no_data_val] = np.nan
    if rast_arr.size == 0:
        raise ValueError("Raster array is empty after loading/cropping")
    return rast_arr, trf_arr, crs_wkt, no_data_val


def create_empty_raster(
    path_str: str | Path,
    rows: int,
    cols: int,
    transform: list[float],
    crs_wkt: str,
    dtype=np.float32,
    nodata: float = -9999,
    bands: int = 1,
):
    """
    Create an empty GeoTIFF file initialized with nodata.
    """
    path = check_path(path_str, make_dir=True)

    if GDAL_ENV is False:
        trf = Affine.from_gdal(*transform)
        crs = None
        if crs_wkt:
            crs = pyproj.CRS(crs_wkt)

        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=rows,
            width=cols,
            count=bands,
            dtype=dtype,
            crs=crs,
            transform=trf,
            nodata=nodata,
        ) as dst:
            chunk_h = min(256, rows)
            chunk_w = min(256, cols)
            fill_block = np.full((chunk_h, chunk_w), nodata, dtype=dtype)
            for b in range(1, bands + 1):
                for row_off in range(0, rows, chunk_h):
                    h = min(chunk_h, rows - row_off)
                    for col_off in range(0, cols, chunk_w):
                        w = min(chunk_w, cols - col_off)
                        win = Window(col_off, row_off, w, h)  # type: ignore[too-many-positional-arguments]
                        dst.write(fill_block[:h, :w], b, window=win)
    else:
        driver = gdal.GetDriverByName("GTiff")
        # Map numpy dtype to GDAL type
        gdal_type = gdal.GDT_Float32  # Default
        if dtype == np.float64:
            gdal_type = gdal.GDT_Float64
        elif dtype == np.int32:
            gdal_type = gdal.GDT_Int32
        elif dtype == np.int16:
            gdal_type = gdal.GDT_Int16
        elif dtype == np.uint8:
            gdal_type = gdal.GDT_Byte

        ds = driver.Create(str(path), cols, rows, bands, gdal_type)
        ds.SetGeoTransform(transform)
        if crs_wkt:
            ds.SetProjection(crs_wkt)
        for b in range(1, bands + 1):
            band = ds.GetRasterBand(b)
            band.SetNoDataValue(nodata)
            band.Fill(nodata)
        ds = None


def write_raster_window(path_str: str | Path, data: np.ndarray, window: tuple[slice, slice], band: int = 1):
    """
    Write a data array to a specific window in an existing raster.
    window is (row_slice, col_slice).
    """
    path = check_path(path_str)
    row_slice, col_slice = window

    if GDAL_ENV is False:
        from rasterio.windows import Window

        with rasterio.open(path, "r+") as dst:
            win = Window(
                col_slice.start,  # type: ignore[too-many-positional-arguments]
                row_slice.start,
                col_slice.stop - col_slice.start,
                row_slice.stop - row_slice.start,
            )
            dst.write(data, band, window=win)
    else:
        ds = gdal.Open(str(path), gdal.GA_Update)
        if ds is None:
            raise OSError(f"Could not open {path} for update")

        xoff = col_slice.start
        yoff = row_slice.start

        ds.GetRasterBand(band).WriteArray(data, xoff, yoff)
        ds = None


# EPW (EnergyPlus Weather) parsing and PVGIS TMY download were extracted to
# io_epw.py to keep this module under the 700-line hot-file threshold. The
# names below are re-exported so existing callers (from solweig.io import
# read_epw, download_epw) keep working unchanged. The leading-underscore
# names are kept available for the few internal callers (e.g. weather.py and
# the QGIS plugin's epw_import helper) that reach into them.
from .io_epw import (  # noqa: E402, F401
    _BooleanArray,
    _EpwColumn,
    _EpwDataFrame,
    _EpwDataIndex,
    _EpwIloc,
    _EpwRow,
    _EpwTimestamp,
    _IndexAccessor,
    _parse_epw_metadata,
    _read_epw_pure_python,
    download_epw,
    read_epw,
)
