"""Raster I/O and EPW weather file handling.

Provides functions for reading and writing GeoTIFF rasters, parsing
EnergyPlus Weather (EPW) files, and downloading TMY data from PVGIS.
Automatically selects between rasterio and GDAL backends (see ``_compat``).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from ._compat import GDAL_ENV
from .buffers import as_float32

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


def rasterise_gdf(gdf, geom_col, ht_col, bbox=None, pixel_size: float = 1.0):
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


# Default color scale ranges for preview images (ensures consistency across timesteps)
# Format: prefix -> (vmin, vmax)
_PREVIEW_RANGES: dict[str, tuple[float, float]] = {
    "tmrt": (0, 80),  # Mean radiant temperature (°C)
    "utci": (-40, 50),  # Universal Thermal Climate Index (°C)
    "pet": (-40, 50),  # Physiological Equivalent Temperature (°C)
    "shadow": (0, 1),  # Shadow fraction (1=sunlit, 0=shaded)
    "kdown": (0, 1200),  # Downwelling shortwave radiation (W/m²)
    "kup": (0, 800),  # Upwelling shortwave radiation (W/m²)
    "ldown": (150, 550),  # Downwelling longwave radiation (W/m²)
    "lup": (250, 650),  # Upwelling longwave radiation (W/m²)
    "svf": (0, 1),  # Sky view factor
    "gvf": (0, 1),  # Ground view factor
}


def _get_preview_range(filename: str) -> tuple[float, float] | None:
    """Get the color scale range for a variable based on filename prefix."""
    name = filename.lower()
    for prefix, range_vals in _PREVIEW_RANGES.items():
        if name.startswith(prefix):
            return range_vals
    return None


def _generate_preview_png(data_arr: np.ndarray, out_path: Path, max_size: int = 512, colormap: str = "turbo") -> None:
    """
    Generate a color PNG preview image from raster data.

    Uses consistent color scales for known variable types (tmrt, utci, shadow, etc.)
    to enable visual comparison across timesteps. Falls back to percentile-based
    scaling for unknown variables.

    Args:
        data_arr: 2D numpy array to visualize
        out_path: Output file path (preview will be saved as .preview.png)
        max_size: Maximum dimension for the preview image (maintains aspect ratio)
        colormap: Matplotlib colormap name (default: 'turbo'). Falls back to grayscale if unavailable.
                  Common options: 'turbo', 'viridis', 'plasma', 'inferno', 'magma', 'coolwarm'
    """
    try:
        from PIL import Image

        # Handle NaN values
        valid_mask = ~np.isnan(data_arr)
        if not np.any(valid_mask):
            return  # All NaN, skip preview

        # Use variable-specific range if available, otherwise fall back to percentiles
        preset_range = _get_preview_range(out_path.stem)
        if preset_range is not None:
            vmin, vmax = preset_range
        else:
            # Fallback: use percentiles for unknown variables
            valid_data = data_arr[valid_mask]
            vmin, vmax = np.nanpercentile(valid_data, [2, 98])

        if vmax <= vmin:
            vmax = vmin + 1  # Avoid division by zero

        # Normalize to 0-1
        normalized = np.clip((data_arr - vmin) / (vmax - vmin), 0, 1)
        normalized = np.nan_to_num(normalized, nan=0)

        # Try to apply matplotlib colormap for color output
        try:
            import matplotlib.pyplot as plt

            # Get colormap and apply
            cmap = plt.get_cmap(colormap)
            colored = cmap(normalized)  # Returns RGBA in [0, 1]

            # Convert to RGB uint8 (drop alpha channel)
            rgb = (colored[:, :, :3] * 255).astype(np.uint8)
            img = Image.fromarray(rgb, mode="RGB")
        except (ImportError, ValueError):
            # Fallback to grayscale if matplotlib not available or colormap invalid
            grayscale = (normalized * 255).astype(np.uint8)
            img = Image.fromarray(grayscale, mode="L")

        # Resize to max_size while maintaining aspect ratio
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Save preview
        preview_path = out_path.with_suffix(".preview.png")
        img.save(preview_path, "PNG")
        logger.debug(f"Saved preview: {preview_path}")
    except ImportError:
        logger.debug("PIL not available, skipping preview generation")
    except Exception as e:
        logger.warning(f"Failed to generate preview: {e}")


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


class _EpwDataIndex:
    """Lightweight index class mimicking pandas DatetimeIndex for EPW data."""

    def __init__(self, timestamps: list):
        self._timestamps = timestamps
        self.tz = None
        self.name = "datetime"

    def __len__(self):
        return len(self._timestamps)

    def __getitem__(self, idx):
        return self._timestamps[idx]

    def __iter__(self):
        return iter(self._timestamps)

    def __ge__(self, other):
        """Greater than or equal comparison, returns boolean array."""
        return _BooleanArray([t >= other for t in self._timestamps])

    def __le__(self, other):
        """Less than or equal comparison, returns boolean array."""
        return _BooleanArray([t <= other for t in self._timestamps])

    def __gt__(self, other):
        """Greater than comparison, returns boolean array."""
        return _BooleanArray([t > other for t in self._timestamps])

    def __lt__(self, other):
        """Less than comparison, returns boolean array."""
        return _BooleanArray([t < other for t in self._timestamps])

    @property
    def empty(self):
        return len(self._timestamps) == 0

    @property
    def year(self):
        return [t.year for t in self._timestamps]

    @property
    def month(self):
        return _IndexAccessor([t.month for t in self._timestamps])

    @property
    def day(self):
        return _IndexAccessor([t.day for t in self._timestamps])

    @property
    def hour(self):
        return _IndexAccessor([t.hour for t in self._timestamps])

    def min(self):
        return min(self._timestamps) if self._timestamps else None

    def max(self):
        return max(self._timestamps) if self._timestamps else None

    def tz_localize(self, tz):
        # Return self since we don't handle timezones in the fallback
        return self


class _IndexAccessor:
    """Helper for index property access like df.index.hour."""

    def __init__(self, values: list):
        self._values = values

    def __iter__(self):
        return iter(self._values)

    def __gt__(self, other):
        return _BooleanArray([v > other for v in self._values])

    def __ge__(self, other):
        return _BooleanArray([v >= other for v in self._values])

    def __lt__(self, other):
        return _BooleanArray([v < other for v in self._values])

    def __le__(self, other):
        return _BooleanArray([v <= other for v in self._values])

    def __eq__(self, other):
        return _BooleanArray([v == other for v in self._values])

    def isin(self, values_set):
        return [v in values_set for v in self._values]


class _BooleanArray:
    """Helper for boolean array operations (& and |)."""

    def __init__(self, values: list):
        self._values = values

    def __and__(self, other):
        if isinstance(other, _BooleanArray):
            return _BooleanArray([a and b for a, b in zip(self._values, other._values, strict=False)])
        return _BooleanArray([a and b for a, b in zip(self._values, other, strict=False)])

    def __or__(self, other):
        if isinstance(other, _BooleanArray):
            return _BooleanArray([a or b for a, b in zip(self._values, other._values, strict=False)])
        return _BooleanArray([a or b for a, b in zip(self._values, other, strict=False)])

    def __iter__(self):
        return iter(self._values)

    def __getitem__(self, idx):
        return self._values[idx]

    def __len__(self):
        return len(self._values)

    def all(self):
        return all(self._values)

    def any(self):
        return any(self._values)

    def tolist(self):
        return self._values


class _EpwRow:
    """Lightweight row class mimicking pandas Series for EPW data."""

    def __init__(self, data: dict):
        self._data = data

    def __getitem__(self, key):
        return self._data.get(key, float("nan"))

    def get(self, key, default=None):
        """Get value with default, like dict.get()."""
        val = self._data.get(key, default)
        if val is None or (isinstance(val, float) and val != val):  # NaN check
            return default
        return val


class _EpwColumn:
    """Lightweight column accessor mimicking a pandas Series for a single column."""

    def __init__(self, values: list):
        self._values = values

    def __getitem__(self, idx):
        return self._values[idx]

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def min(self):
        return min(v for v in self._values if v == v)  # skip NaN

    def max(self):
        return max(v for v in self._values if v == v)  # skip NaN

    def __ge__(self, other):
        return _BooleanArray([v >= other for v in self._values])

    def __le__(self, other):
        return _BooleanArray([v <= other for v in self._values])

    def __gt__(self, other):
        return _BooleanArray([v > other for v in self._values])

    def __lt__(self, other):
        return _BooleanArray([v < other for v in self._values])

    def all(self):
        return all(self._values)


class _EpwIloc:
    """Positional indexing for _EpwDataFrame."""

    def __init__(self, rows: list[dict]):
        self._rows = rows

    def __getitem__(self, idx):
        return _EpwRow(self._rows[idx])


class _EpwDataFrame:
    """Lightweight DataFrame-like class for EPW data without pandas dependency."""

    def __init__(self, rows: list[dict], timestamps: list):
        self._rows = rows
        self._timestamps = timestamps
        self.index = _EpwDataIndex(timestamps)

    def __len__(self):
        return len(self._rows)

    @property
    def columns(self):
        """Column names from the first row."""
        if self._rows:
            return list(self._rows[0].keys())
        return []

    @property
    def iloc(self):
        """Positional indexing (returns _EpwRow objects)."""
        return _EpwIloc(self._rows)

    def __getitem__(self, key):
        """Access by column name (str) or filter by boolean mask."""
        if isinstance(key, str):
            return _EpwColumn([row.get(key, float("nan")) for row in self._rows])
        if isinstance(key, _BooleanArray):
            key = key._values
        if isinstance(key, list):
            filtered_rows = [r for r, m in zip(self._rows, key, strict=False) if m]
            filtered_ts = [t for t, m in zip(self._timestamps, key, strict=False) if m]
            return _EpwDataFrame(filtered_rows, filtered_ts)
        raise TypeError(f"Unsupported indexing type: {type(key)}")

    @property
    def empty(self):
        return len(self._rows) == 0

    def iterrows(self):
        """Iterate over (timestamp, row) pairs."""
        for ts, row_data in zip(self._timestamps, self._rows, strict=False):
            yield _EpwTimestamp(ts), _EpwRow(row_data)

    def to_dataframe(self):
        """Convert to pandas DataFrame if pandas is available.

        Returns:
            pd.DataFrame with DatetimeIndex, or self if pandas unavailable.
        """
        try:
            import pandas as pd

            df = pd.DataFrame(self._rows)
            df.index = pd.DatetimeIndex(self._timestamps, name="datetime")
            return df
        except ImportError:
            return self


class _EpwTimestamp:
    """Wrapper for datetime to provide pandas-like interface."""

    def __init__(self, dt_obj):
        self._dt = dt_obj

    def __getattr__(self, name):
        return getattr(self._dt, name)

    def to_pydatetime(self):
        return self._dt

    def replace(self, **kwargs):
        return self._dt.replace(**kwargs)


def _parse_epw_metadata(path: Path) -> dict:
    """Parse EPW header to extract metadata."""
    metadata = {}
    with open(path, encoding="utf-8") as f:
        location_line = f.readline().strip()
        if not location_line.startswith("LOCATION"):
            raise ValueError("Invalid EPW file: first line must start with 'LOCATION'")

        location_parts = location_line.split(",")
        if len(location_parts) < 10:
            raise ValueError(f"Invalid LOCATION line: expected at least 10 fields, got {len(location_parts)}")

        metadata["city"] = location_parts[1].strip()
        metadata["state"] = location_parts[2].strip()
        metadata["country"] = location_parts[3].strip()
        metadata["latitude"] = float(location_parts[6])
        metadata["longitude"] = float(location_parts[7])
        metadata["tz_offset"] = float(location_parts[8])
        metadata["elevation"] = float(location_parts[9])

    return metadata


def _read_epw_pure_python(path: Path) -> tuple:
    """Pure Python EPW parser without pandas dependency."""
    import csv
    from datetime import datetime as dt_class
    from datetime import timedelta

    metadata = _parse_epw_metadata(path)

    # Column indices for the fields we need
    # EPW format has 35 fields per line
    col_indices = {
        "year": 0,
        "month": 1,
        "day": 2,
        "hour": 3,
        "minute": 4,
        "temp_air": 6,
        "relative_humidity": 8,
        "atmospheric_pressure": 9,
        "ghi": 13,
        "dni": 14,
        "dhi": 15,
        "wind_direction": 20,
        "wind_speed": 21,
    }

    na_values = {"99", "999", "9999", "99999", "999999999", ""}

    rows = []
    timestamps = []

    with open(path, encoding="utf-8") as f:
        # Skip 8 header lines
        for _ in range(8):
            f.readline()

        reader = csv.reader(f)
        for line in reader:
            if len(line) < 22:
                continue

            try:
                year = int(line[col_indices["year"]])
                month = int(line[col_indices["month"]])
                day = int(line[col_indices["day"]])
                hour = int(line[col_indices["hour"]])
                minute = int(line[col_indices["minute"]])

                # EPW uses 1-24 hour format; hour 24 means midnight of next day
                if hour == 24:
                    timestamp = dt_class(year, month, day, 0, minute) + timedelta(days=1)
                else:
                    timestamp = dt_class(year, month, day, hour, minute)
                timestamps.append(timestamp)

                def parse_float(idx, row_data=line):
                    val = row_data[idx].strip()
                    if val in na_values:
                        return float("nan")
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return float("nan")

                row = {
                    "temp_air": parse_float(col_indices["temp_air"]),
                    "relative_humidity": parse_float(col_indices["relative_humidity"]),
                    "atmospheric_pressure": parse_float(col_indices["atmospheric_pressure"]),
                    "ghi": parse_float(col_indices["ghi"]),
                    "dni": parse_float(col_indices["dni"]),
                    "dhi": parse_float(col_indices["dhi"]),
                    "wind_speed": parse_float(col_indices["wind_speed"]),
                    "wind_direction": parse_float(col_indices["wind_direction"]),
                }
                rows.append(row)
            except (ValueError, IndexError):
                continue

    if not rows:
        raise ValueError("EPW file contains no valid data rows")

    df = _EpwDataFrame(rows, timestamps)
    logger.info(f"Loaded EPW file: {metadata['city']}, {len(df)} timesteps (pure Python parser)")

    return df, metadata


def download_epw(
    latitude: float,
    longitude: float,
    output_path: str | Path,
    *,
    timeout: int = 60,
) -> Path:
    """
    Download a Typical Meteorological Year (TMY) EPW file from PVGIS.

    Uses the EU Joint Research Centre's PVGIS API (v5.3, no API key required).
    Coverage is near-global (all continents except polar regions),
    using ERA5 reanalysis data.

    **Important:** TMY files are *not* observations for a specific year.
    A TMY is a statistical composite — each calendar month is selected from
    the most "typical" month across a multi-year reference period (2005–2023
    for PVGIS v5.3).  The resulting file represents long-term average climate
    conditions, not a recent or continuously updated dataset.  Because each
    month is taken from a real historical year, the row timestamps in the
    returned file will legitimately span multiple years within the reference
    window (e.g. a January from 2022 alongside a February from 2014). This is
    the defining property of a TMY and not an artefact.  The reference period
    is fixed per PVGIS release; data freshness depends on the upstream PVGIS
    version, not on SOLWEIG.

    See the PVGIS TMY documentation for full methodology and data sources:
    https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/pvgis-tools/pvgis-typical-meteorological-year-tmy-generator_en

    Ref: UMEP-dev/solweig#8

    The downloaded data contains modified Copernicus Climate Change Service
    information. Neither the European Commission nor ECMWF is responsible
    for any use that may be made of the Copernicus information or data it
    contains. See https://cds.climate.copernicus.eu/disclaimer for the full
    licence terms.

    Args:
        latitude: Latitude in decimal degrees (-90 to 90).
        longitude: Longitude in decimal degrees (-180 to 180).
        output_path: Path where the EPW file will be saved.
        timeout: HTTP request timeout in seconds (default 60).

    Returns:
        Path to the saved EPW file.

    Raises:
        ValueError: If coordinates are out of range.
        ConnectionError: If the PVGIS server is unreachable.
        RuntimeError: If the download fails (e.g. location over ocean).

    Example:
        >>> from solweig.io import download_epw
        >>> path = download_epw(37.98, 23.73, "athens.epw")
        >>> data, metadata = read_epw(path)
    """
    import urllib.error
    import urllib.request

    if not -90 <= latitude <= 90:
        raise ValueError(f"Latitude must be between -90 and 90, got {latitude}")
    if not -180 <= longitude <= 180:
        raise ValueError(f"Longitude must be between -180 and 180, got {longitude}")

    output_path = Path(output_path)

    url = f"https://re.jrc.ec.europa.eu/api/v5_3/tmy?lat={latitude}&lon={longitude}&outputformat=epw"

    logger.info(f"Downloading EPW from PVGIS for ({latitude:.4f}, {longitude:.4f})...")

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 400:
            raise RuntimeError(
                f"PVGIS has no data for ({latitude}, {longitude}). The location may be over ocean or outside coverage."
            ) from e
        raise RuntimeError(f"PVGIS download failed (HTTP {e.code}): {e.reason}") from e
    except urllib.error.URLError as e:
        raise ConnectionError(f"Cannot reach PVGIS server: {e.reason}") from e

    if len(data) < 1000:
        # PVGIS returns a short error message for invalid locations
        text = data.decode("utf-8", errors="replace")
        raise RuntimeError(f"PVGIS returned an error: {text.strip()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)

    lines = data.decode("utf-8", errors="replace").split("\n")
    n_data_lines = len(lines) - 8  # subtract header lines
    logger.info(f"Saved EPW file: {output_path} ({n_data_lines} hourly records)")

    return output_path


def read_epw(path: str | Path) -> tuple:
    """
    Read EnergyPlus Weather (EPW) file and return weather data with metadata.

    EPW files have 8 header lines followed by hourly weather data.
    Uses pure Python parser (no pandas/scipy dependencies).

    Args:
        path: Path to EPW file (string or Path)

    Returns:
        Tuple of (data, metadata_dict):
        - data: DataFrame-like object with datetime index and weather columns:
            - temp_air: Dry bulb temperature (°C)
            - relative_humidity: Relative humidity (%)
            - atmospheric_pressure: Atmospheric pressure (Pa)
            - ghi: Global horizontal irradiance (W/m²)
            - dni: Direct normal irradiance (W/m²)
            - dhi: Diffuse horizontal irradiance (W/m²)
            - wind_speed: Wind speed (m/s)
            - wind_direction: Wind direction (degrees)
        - metadata_dict: Dictionary with keys:
            - city: Location city name
            - latitude: Latitude (degrees)
            - longitude: Longitude (degrees)
            - elevation: Elevation (m)
            - tz_offset: Timezone offset (hours)

    Raises:
        FileNotFoundError: If EPW file doesn't exist
        ValueError: If EPW file is malformed
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"EPW file not found: {path}")

    return _read_epw_pure_python(path)
