"""Raster loading + alignment helpers for :meth:`SurfaceData.prepare`.

Extracted from ``models/surface.py`` to keep that file focused on the
:class:`SurfaceData` data semantics. These helpers were previously
``@staticmethod`` / ``@classmethod`` on the class but had no dependency
on instance state — they're plain functions here.

Pipeline (called from :meth:`SurfaceData.prepare` in this order):

1. :func:`load_and_validate_dsm` — load the DSM raster, derive the
   target pixel size, validate the CRS is projected.
2. :func:`load_terrain_rasters` — load optional CDSM / DEM / TDSM /
   land-cover rasters (each may be absent).
3. :func:`load_preprocessing_data` — discover / load existing wall and
   SVF caches under the working directory.
4. :func:`align_rasters` — compute the target bounding box and
   resample any layer that doesn't already match the target grid.
5. :func:`create_surface_instance` — assemble the aligned arrays into
   a :class:`SurfaceData`.

None of these touch :class:`SurfaceData` state (they only read paths
and arrays). The final ``create_surface_instance`` step does a deferred
import of :class:`SurfaceData` to construct it; that import is inside
the function body to keep this module importable without a circular
dependency on ``surface.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .._compat import GDAL_ENV
from ..cache import pixel_size_tag
from ..solweig_logging import get_logger
from ..utils import extract_bounds, intersect_bounds, resample_to_grid

if TYPE_CHECKING:
    from .precomputed import ShadowArrays, SvfArrays
    from .surface import SurfaceData

logger = get_logger(__name__)


# ── 1. DSM load + CRS validation ───────────────────────────────────────────


def load_and_validate_dsm(dsm: str | Path, pixel_size: float | None) -> tuple:
    """Load DSM raster and validate its CRS.

    Args:
        dsm: Path to DSM GeoTIFF file.
        pixel_size: Optional pixel size in metres. If ``None``, extracted
            from the DSM's geotransform.

    Returns:
        Tuple of ``(dsm_array, dsm_transform, dsm_crs, pixel_size)``.

    Raises:
        ValueError: If the DSM has no CRS, is not projected, or if
            ``pixel_size`` is finer than the DSM's native resolution.
    """
    from .. import io

    dsm_arr, dsm_transform, dsm_crs, _ = io.load_raster(str(dsm))
    logger.info(f"  DSM: {dsm_arr.shape[1]}×{dsm_arr.shape[0]} pixels")

    # Compute pixel size from geotransform if not provided
    native_pixel_size = abs(dsm_transform[1])
    if pixel_size is None:
        pixel_size = native_pixel_size
        logger.info(f"  Extracted pixel size from DSM: {pixel_size:.2f} m")
    else:
        # Validate against native resolution
        if pixel_size < native_pixel_size - 0.01:
            raise ValueError(
                f"Specified pixel_size ({pixel_size:.2f} m) is finer than the DSM native "
                f"resolution ({native_pixel_size:.2f} m). Upsampling creates false precision. "
                f"Use pixel_size >= {native_pixel_size:.2f} or omit to use native resolution."
            )
        if abs(pixel_size - native_pixel_size) > 0.01:
            logger.warning(
                f"  ⚠ Specified pixel_size ({pixel_size:.2f} m) differs from DSM native "
                f"resolution ({native_pixel_size:.2f} m) — all rasters will be resampled"
            )
        logger.info(f"  Using specified pixel size: {pixel_size:.2f} m")

    if pixel_size < 1.0:
        logger.warning(
            f"  ⚠ Pixel size ({pixel_size:.2f} m) is less than 1 meter - calculations may be slow for large areas"
        )

    # Validate CRS is projected (required for distance calculations)
    if dsm_crs is None:
        raise ValueError("DSM file has no CRS information. SOLWEIG requires a projected coordinate system.")

    # Parse failures (exotic WKT the backend can't read) downgrade to a
    # warning, but a successfully parsed *geographic* CRS is a hard reject —
    # the raise must sit outside the try so the except cannot swallow it.
    is_projected: bool | None = None
    crs_name = "unknown"
    epsg: str | int = "custom"
    try:
        if GDAL_ENV:
            from osgeo import osr

            srs = osr.SpatialReference()
            srs.ImportFromWkt(dsm_crs)
            is_projected = bool(srs.IsProjected())
            crs_name = srs.GetName() or "unknown"
            epsg = srs.GetAuthorityCode(None) or "custom"
        else:
            from pyproj import CRS as pyproj_CRS

            crs_obj = pyproj_CRS.from_wkt(dsm_crs)
            is_projected = crs_obj.is_projected
            crs_name = crs_obj.name
            epsg = crs_obj.to_epsg() or "custom"
    except Exception as e:
        logger.warning(f"  ⚠ Could not validate CRS: {e}")

    if is_projected is False:
        raise ValueError(
            f"DSM CRS is geographic (lat/lon): {crs_name}. "
            f"SOLWEIG requires a projected coordinate system (e.g., UTM, State Plane) "
            f"for accurate distance and area calculations. Please reproject your data."
        )
    if is_projected:
        logger.info(f"  CRS validated: {crs_name} (EPSG:{epsg})")

    return dsm_arr, dsm_transform, dsm_crs, pixel_size


# ── 2. Optional terrain raster loading ─────────────────────────────────────


def load_terrain_rasters(
    cdsm: str | Path | None,
    dem: str | Path | None,
    tdsm: str | Path | None,
    land_cover: str | Path | None,
    trunk_ratio: float,
) -> dict:
    """Load optional terrain rasters (CDSM, DEM, TDSM, land_cover).

    Each input may be ``None``; missing rasters become ``None`` entries
    in the returned dict. ``trunk_ratio`` is only used for logging
    (the TDSM is auto-generated from CDSM downstream).
    """
    from .. import io

    result: dict = {}

    if cdsm is not None:
        result["cdsm_arr"], result["cdsm_transform"], _, _ = io.load_raster(str(cdsm))
        logger.info("  ✓ Canopy DSM (CDSM) provided")
    else:
        result["cdsm_arr"], result["cdsm_transform"] = None, None
        logger.info("  → No vegetation data - simulation without trees/vegetation")

    if dem is not None:
        result["dem_arr"], result["dem_transform"], _, _ = io.load_raster(str(dem))
        logger.info("  ✓ Ground elevation (DEM) provided")
    else:
        result["dem_arr"], result["dem_transform"] = None, None

    if tdsm is not None:
        result["tdsm_arr"], result["tdsm_transform"], _, _ = io.load_raster(str(tdsm))
        logger.info("  ✓ Trunk DSM (TDSM) provided")
    elif result["cdsm_arr"] is not None:
        result["tdsm_arr"], result["tdsm_transform"] = None, None
        logger.info(f"  → No TDSM provided - will auto-generate from CDSM (ratio={trunk_ratio})")
    else:
        result["tdsm_arr"], result["tdsm_transform"] = None, None

    if land_cover is not None:
        result["land_cover_arr"], result["land_cover_transform"], _, _ = io.load_raster(str(land_cover))
        logger.info("  ✓ Land cover provided (albedo/emissivity derived from classification)")
    else:
        result["land_cover_arr"], result["land_cover_transform"] = None, None

    return result


# ── 3. Wall/SVF cache discovery ────────────────────────────────────────────


def _load_svf_from_dir(
    svf_path: Path,
    SvfArraysCls,
) -> tuple[SvfArrays | None, str]:
    """Load SVF data from a directory, preferring memmap over zip.

    Returns ``(SvfArrays | None, source)`` where ``source`` is one of
    ``"memmap"``, ``"zip"``, ``"none"``.
    """
    memmap_dir = svf_path / "memmap"
    svf_zip_path = svf_path / "svfs.zip"

    if memmap_dir.exists() and (memmap_dir / "svf.npy").exists():
        svf_data = SvfArraysCls.from_memmap(memmap_dir)
        logger.info("  ✓ SVF loaded from memmap (memory-efficient)")
        return svf_data, "memmap"
    elif svf_zip_path.exists():
        svf_data = SvfArraysCls.from_zip(str(svf_zip_path))
        logger.info("  ✓ SVF loaded from zip")
        return svf_data, "zip"
    return None, "none"


def _load_shadow_from_dir(
    base_path: Path,
    ShadowArraysCls,
) -> tuple[ShadowArrays | None, str]:
    """Load shadow matrices from a directory, preferring NPZ over memmap.

    Returns ``(ShadowArrays | None, source)`` where ``source`` is one of
    ``"npz"``, ``"memmap"``, ``"none"``.
    """
    shadow_npz_path = base_path / "shadowmats.npz"
    if shadow_npz_path.exists():
        shadow_data = ShadowArraysCls.from_npz(str(shadow_npz_path))
        logger.info("  ✓ Shadow matrices loaded from npz")
        return shadow_data, "npz"

    shadow_mm_dir = base_path / "shadow_memmaps"
    if shadow_mm_dir.exists() and (shadow_mm_dir / "metadata.json").exists():
        shadow_data = ShadowArraysCls.from_memmap(shadow_mm_dir)
        logger.info("  ✓ Shadow matrices loaded from memmap cache")
        return shadow_data, "memmap"

    return None, "none"


def load_preprocessing_data(
    wall_height: str | Path | None,
    wall_aspect: str | Path | None,
    svf_dir: str | Path | None,
    working_path: Path,
    force_recompute: bool,
    pixel_size: float = 1.0,
) -> dict:
    """Discover and load existing wall + SVF caches from ``working_path``.

    Auto-discovers pixel-size-keyed cache subdirectories under
    ``working_path/walls/`` and ``working_path/svf/``, falling back to
    legacy flat directories if the keyed ones don't exist.

    Args:
        wall_height, wall_aspect: Optional explicit paths to wall files.
            If both are provided, they're used directly; if neither is
            provided, the working directory is searched. If only one is
            provided, a warning is emitted and walls are recomputed.
        svf_dir: Optional explicit path to an SVF directory.
        working_path: Project working directory (cache root).
        force_recompute: If ``True``, skip cache discovery and signal
            "compute walls/SVF from scratch."
        pixel_size: Used for pixel-size-keyed cache paths.

    Returns:
        Dict with the loaded arrays + flags indicating whether walls
        and SVF still need to be computed. See the function body for
        the exact key set.
    """
    from .. import io
    from .precomputed import ShadowArrays, SvfArrays

    logger.info("Checking for preprocessing data...")
    px_tag = pixel_size_tag(pixel_size)

    result: dict = {
        "wall_height_arr": None,
        "wall_height_transform": None,
        "wall_aspect_arr": None,
        "wall_aspect_transform": None,
        "svf_data": None,
        "svf_source": "none",
        "shadow_data": None,
        "compute_walls": False,
        "compute_svf": False,
    }

    # Load walls with auto-discovery
    if wall_height is not None and wall_aspect is not None:
        result["wall_height_arr"], result["wall_height_transform"], _, _ = io.load_raster(str(wall_height))
        result["wall_aspect_arr"], result["wall_aspect_transform"], _, _ = io.load_raster(str(wall_aspect))
        logger.info("  ✓ Existing walls found (will use precomputed)")
    elif wall_height is not None or wall_aspect is not None:
        logger.warning("  ⚠ Only one wall file provided - both wall_height and wall_aspect required")
        logger.info("  → Walls will be computed from DSM and cached")
        result["compute_walls"] = True
    else:
        if force_recompute:
            logger.info("  → force_recompute=True - will recompute walls from DSM and cache")
            result["compute_walls"] = True
        else:
            walls_cache_dir = working_path / "walls" / px_tag
            wall_hts_path = walls_cache_dir / "wall_hts.tif"
            wall_aspects_path = walls_cache_dir / "wall_aspects.tif"

            # Legacy fallback: try flat working_dir/walls/ if keyed dir absent
            if not wall_hts_path.exists():
                legacy_dir = working_path / "walls"
                legacy_hts = legacy_dir / "wall_hts.tif"
                legacy_asp = legacy_dir / "wall_aspects.tif"
                if legacy_hts.exists() and legacy_asp.exists():
                    logger.info(f"  ⚠ Legacy wall cache at {legacy_dir} — future runs will use pixel-size-keyed path")
                    walls_cache_dir = legacy_dir
                    wall_hts_path = legacy_hts
                    wall_aspects_path = legacy_asp

            if wall_hts_path.exists() and wall_aspects_path.exists():
                result["wall_height_arr"], result["wall_height_transform"], _, _ = io.load_raster(str(wall_hts_path))
                result["wall_aspect_arr"], result["wall_aspect_transform"], _, _ = io.load_raster(
                    str(wall_aspects_path)
                )
                logger.info(f"  ✓ Walls found in working_dir: {walls_cache_dir}")
            else:
                logger.info("  → No walls found in working_dir - will compute from DSM and cache")
                result["compute_walls"] = True

    # Load SVF with auto-discovery
    if svf_dir is not None:
        svf_path = Path(svf_dir)

        svf_data, svf_source = _load_svf_from_dir(svf_path, SvfArrays)
        if svf_data is not None:
            result["svf_data"] = svf_data
            result["svf_source"] = svf_source
            logger.info("  ✓ Existing SVF found (will use precomputed)")

            shadow_data, _ = _load_shadow_from_dir(svf_path, ShadowArrays)
            if shadow_data is None:
                tagged_cache = svf_path / "svf" / px_tag
                shadow_data, _ = _load_shadow_from_dir(tagged_cache, ShadowArrays)
            if shadow_data is not None:
                result["shadow_data"] = shadow_data
                logger.info("  ✓ Existing shadow matrices found (anisotropic sky enabled)")
            else:
                logger.info("  → Shadow matrices not found alongside SVFs — will recompute to generate them")
                result["svf_data"] = None
                result["compute_svf"] = True
        else:
            logger.info(f"  → SVF directory provided but no SVF files found: {svf_path}")
            logger.info("  → SVF will be computed and cached")
            result["compute_svf"] = True
    else:
        if force_recompute:
            logger.info("  → force_recompute=True - will recompute SVF and cache")
            result["compute_svf"] = True
        else:
            svf_cache_dir = working_path / "svf" / px_tag

            if not svf_cache_dir.exists():
                legacy_svf_dir = working_path / "svf"
                if (legacy_svf_dir / "memmap" / "svf.npy").exists() or (legacy_svf_dir / "svfs.zip").exists():
                    logger.info(
                        f"  ⚠ Legacy SVF cache at {legacy_svf_dir} — future runs will use pixel-size-keyed path"
                    )
                    svf_cache_dir = legacy_svf_dir

            svf_data, svf_source = _load_svf_from_dir(svf_cache_dir, SvfArrays)
            if svf_data is not None:
                result["svf_data"] = svf_data
                result["svf_source"] = svf_source
                logger.info(f"  ✓ SVF found in working_dir: {svf_cache_dir}")

                shadow_data, _ = _load_shadow_from_dir(svf_cache_dir, ShadowArrays)
                if shadow_data is not None:
                    result["shadow_data"] = shadow_data
                    logger.info("  ✓ Shadow matrices found (anisotropic sky enabled)")
                else:
                    logger.info("  → Shadow matrices not found in working_dir cache — will recompute to generate them")
                    result["svf_data"] = None
                    result["compute_svf"] = True
            else:
                logger.info("  → No SVF found in working_dir - will compute and cache")
                result["compute_svf"] = True

    return result


# ── 3b. Header-only raster metadata (for layer-sequential alignment) ───────


def _read_raster_header(path: str | Path) -> tuple[list[float], tuple[int, int]]:
    """Read a raster's geotransform and shape without loading its data.

    Returns ``(gdal_style_transform, (rows, cols))``. Backend-agnostic:
    rasterio in the standard environment, GDAL under QGIS.
    """
    if GDAL_ENV:
        from osgeo import gdal

        ds = gdal.Open(str(path))
        if ds is None:
            raise FileNotFoundError(f"Could not open raster: {path}")
        transform = list(ds.GetGeoTransform())
        shape = (ds.RasterYSize, ds.RasterXSize)
        ds = None
        return transform, shape
    else:
        import rasterio

        with rasterio.open(path) as src:
            return list(src.transform.to_gdal()), (src.height, src.width)


# ── 4. Grid alignment + resampling ─────────────────────────────────────────


def align_rasters(
    dsm_arr,
    dsm_transform,
    dsm_crs,
    pixel_size: float,
    terrain_rasters: dict,
    preprocess_data: dict,
    bbox: list[float] | None,
) -> dict:
    """Compute target extent, validate bbox, resample all rasters to a common grid.

    If ``bbox`` is provided, it must be inside the intersection of input
    rasters. Each layer is independently checked: it's only resampled if
    its bounds, pixel size, or shape don't already match the target grid.

    Returns a dict containing every aligned array + the resolved
    ``(dsm_transform, dsm_crs, pixel_size)`` triple.
    """
    logger.info("Computing spatial extent and resolution...")

    # Collect bounds from every loaded raster
    bounds_list = [extract_bounds(dsm_transform, dsm_arr.shape)]

    if terrain_rasters["cdsm_arr"] is not None and terrain_rasters["cdsm_transform"] is not None:
        bounds_list.append(extract_bounds(terrain_rasters["cdsm_transform"], terrain_rasters["cdsm_arr"].shape))
    if terrain_rasters["dem_arr"] is not None and terrain_rasters["dem_transform"] is not None:
        bounds_list.append(extract_bounds(terrain_rasters["dem_transform"], terrain_rasters["dem_arr"].shape))
    if terrain_rasters["tdsm_arr"] is not None and terrain_rasters["tdsm_transform"] is not None:
        bounds_list.append(extract_bounds(terrain_rasters["tdsm_transform"], terrain_rasters["tdsm_arr"].shape))
    if terrain_rasters["land_cover_arr"] is not None and terrain_rasters["land_cover_transform"] is not None:
        bounds_list.append(
            extract_bounds(terrain_rasters["land_cover_transform"], terrain_rasters["land_cover_arr"].shape)
        )
    if preprocess_data["wall_height_arr"] is not None and preprocess_data["wall_height_transform"] is not None:
        bounds_list.append(
            extract_bounds(preprocess_data["wall_height_transform"], preprocess_data["wall_height_arr"].shape)
        )
    if preprocess_data["wall_aspect_arr"] is not None and preprocess_data["wall_aspect_transform"] is not None:
        bounds_list.append(
            extract_bounds(preprocess_data["wall_aspect_transform"], preprocess_data["wall_aspect_arr"].shape)
        )

    # Determine target bounding box
    if bbox is not None:
        computed_intersection = intersect_bounds(bounds_list)
        user_minx, user_miny, user_maxx, user_maxy = bbox
        int_minx, int_miny, int_maxx, int_maxy = computed_intersection

        if (
            user_minx < int_minx - 1e-6
            or user_maxx > int_maxx + 1e-6
            or user_miny < int_miny - 1e-6
            or user_maxy > int_maxy + 1e-6
        ):
            raise ValueError(
                f"Specified bbox {bbox} extends beyond the intersection of input rasters "
                f"{computed_intersection}. Bbox must be within or equal to the intersection."
            )

        target_bbox = bbox
        logger.info(f"  Using user-specified extent: {target_bbox}")
    else:
        target_bbox = intersect_bounds(bounds_list)
        logger.info(f"  Auto-computed extent from raster intersection: {target_bbox}")

    expected_h = int(np.round((target_bbox[3] - target_bbox[1]) / pixel_size))
    expected_w = int(np.round((target_bbox[2] - target_bbox[0]) / pixel_size))
    expected_shape = (expected_h, expected_w)

    def _layer_needs_resample(arr, transform):
        """Check if a single layer needs resampling (bounds, pixel size, or shape mismatch)."""
        layer_bounds = extract_bounds(transform, arr.shape)
        layer_px = abs(transform[1]) if isinstance(transform, list) else abs(transform.a)
        return (
            abs(layer_bounds[0] - target_bbox[0]) > 1e-6
            or abs(layer_bounds[1] - target_bbox[1]) > 1e-6
            or abs(layer_bounds[2] - target_bbox[2]) > 1e-6
            or abs(layer_bounds[3] - target_bbox[3]) > 1e-6
            or abs(layer_px - pixel_size) > 1e-6
            or arr.shape != expected_shape
        )

    resampled_any = False

    # Resample DSM if needed
    if _layer_needs_resample(dsm_arr, dsm_transform):
        dsm_arr, dsm_transform = resample_to_grid(
            dsm_arr, dsm_transform, target_bbox, pixel_size, method="bilinear", src_crs=dsm_crs
        )
        resampled_any = True

    # Resample optional terrain rasters independently
    for key, method in [("cdsm", "bilinear"), ("dem", "bilinear"), ("tdsm", "bilinear"), ("land_cover", "nearest")]:
        arr_key, tf_key = f"{key}_arr", f"{key}_transform"
        if (
            terrain_rasters[arr_key] is not None
            and terrain_rasters[tf_key] is not None
            and _layer_needs_resample(terrain_rasters[arr_key], terrain_rasters[tf_key])
        ):
            terrain_rasters[arr_key], _ = resample_to_grid(
                terrain_rasters[arr_key],
                terrain_rasters[tf_key],
                target_bbox,
                pixel_size,
                method=method,
                src_crs=dsm_crs,
            )
            resampled_any = True

    # Resample preprocessing data independently
    for key in ["wall_height", "wall_aspect"]:
        arr_key, tf_key = f"{key}_arr", f"{key}_transform"
        if (
            preprocess_data[arr_key] is not None
            and preprocess_data[tf_key] is not None
            and _layer_needs_resample(preprocess_data[arr_key], preprocess_data[tf_key])
        ):
            preprocess_data[arr_key], _ = resample_to_grid(
                preprocess_data[arr_key],
                preprocess_data[tf_key],
                target_bbox,
                pixel_size,
                method="bilinear",
                src_crs=dsm_crs,
            )
            resampled_any = True

    # SVF resampling is more complex (multiple arrays) - handled separately if needed
    if preprocess_data["svf_data"] is not None and preprocess_data["svf_data"].svf.shape != dsm_arr.shape:
        logger.warning(
            f"  ⚠ SVF shape {preprocess_data['svf_data'].svf.shape} doesn't match target shape "
            f"{dsm_arr.shape} - SVF resampling not yet implemented. "
            f"SVF cache will be dropped; recompute via SurfaceData.prepare() or compute_svf()."
        )
        preprocess_data["svf_data"] = None
        preprocess_data["shadow_data"] = None

    if resampled_any:
        logger.info(f"  ✓ Resampled to {dsm_arr.shape[1]}×{dsm_arr.shape[0]} pixels")
    else:
        logger.info("  ✓ No resampling needed - all rasters match target grid")

    return {
        "dsm_arr": dsm_arr,
        "dsm_transform": dsm_transform,
        "dsm_crs": dsm_crs,
        "pixel_size": pixel_size,
        "cdsm_arr": terrain_rasters["cdsm_arr"],
        "dem_arr": terrain_rasters["dem_arr"],
        "tdsm_arr": terrain_rasters["tdsm_arr"],
        "land_cover_arr": terrain_rasters["land_cover_arr"],
        "wall_height_arr": preprocess_data["wall_height_arr"],
        "wall_aspect_arr": preprocess_data["wall_aspect_arr"],
        "svf_data": preprocess_data["svf_data"],
        "shadow_data": preprocess_data["shadow_data"],
    }


def load_align_layers_sequential(
    dsm_arr,
    dsm_transform,
    dsm_crs,
    pixel_size: float,
    terrain_paths: dict,
    preprocess_data: dict,
    bbox: list[float] | None,
    spill_dir: str | Path,
) -> dict:
    """Layer-sequential variant of :func:`load_terrain_rasters` + :func:`align_rasters`.

    Memory-bounding path for very large rasters. The whole-array pipeline
    loads every layer into RAM and keeps the full stack alive through
    alignment, preprocessing and save (~2 GB per float32 layer at 500 Mpx,
    six-plus layers at once). Here each layer is handled one at a time:
    load → resample to the target grid (same :func:`resample_to_grid`
    call and parameters as :func:`align_rasters`, so the maths is
    identical) → spill to ``spill_dir/<name>.npy`` → replace the RAM array
    with an ``r+`` memmap over that file. Peak residency is ~2 layer
    arrays (the one being processed plus its resample destination)
    regardless of how many layers the surface has.

    The target bounding box is computed up front from raster *headers*
    (no data read), using the same intersection/validation logic as
    :func:`align_rasters`. Returns the same dict contract as
    :func:`align_rasters`. Downstream, ``preprocess()`` streams its
    conversions in place through these memmaps and ``save_cleaned()``
    reuses the spilled ``.npy`` files directly.
    """
    from .. import io

    spill = Path(spill_dir)
    spill.mkdir(parents=True, exist_ok=True)
    logger.info("Computing spatial extent and resolution (layer-sequential, memory-bounded)...")

    # ── Target bbox from headers (bounds only, no pixel data) ──
    bounds_list = [extract_bounds(dsm_transform, dsm_arr.shape)]
    headers: dict[str, tuple[list[float], tuple[int, int]] | None] = {}
    for name in ("cdsm", "dem", "tdsm", "land_cover"):
        p = terrain_paths.get(name)
        if p is not None:
            hdr = _read_raster_header(p)
            headers[name] = hdr
            bounds_list.append(extract_bounds(hdr[0], hdr[1]))
        else:
            headers[name] = None
    for name in ("wall_height", "wall_aspect"):
        arr = preprocess_data[f"{name}_arr"]
        tf = preprocess_data[f"{name}_transform"]
        if arr is not None and tf is not None:
            bounds_list.append(extract_bounds(tf, arr.shape))

    if bbox is not None:
        computed_intersection = intersect_bounds(bounds_list)
        user_minx, user_miny, user_maxx, user_maxy = bbox
        int_minx, int_miny, int_maxx, int_maxy = computed_intersection
        if (
            user_minx < int_minx - 1e-6
            or user_maxx > int_maxx + 1e-6
            or user_miny < int_miny - 1e-6
            or user_maxy > int_maxy + 1e-6
        ):
            raise ValueError(
                f"Specified bbox {bbox} extends beyond the intersection of input rasters "
                f"{computed_intersection}. Bbox must be within or equal to the intersection."
            )
        target_bbox = bbox
        logger.info(f"  Using user-specified extent: {target_bbox}")
    else:
        target_bbox = intersect_bounds(bounds_list)
        logger.info(f"  Auto-computed extent from raster intersection: {target_bbox}")

    expected_h = int(np.round((target_bbox[3] - target_bbox[1]) / pixel_size))
    expected_w = int(np.round((target_bbox[2] - target_bbox[0]) / pixel_size))
    expected_shape = (expected_h, expected_w)

    def _needs_resample(shape, transform) -> bool:
        layer_bounds = extract_bounds(transform, shape)
        layer_px = abs(transform[1]) if isinstance(transform, list) else abs(transform.a)
        return (
            abs(layer_bounds[0] - target_bbox[0]) > 1e-6
            or abs(layer_bounds[1] - target_bbox[1]) > 1e-6
            or abs(layer_bounds[2] - target_bbox[2]) > 1e-6
            or abs(layer_bounds[3] - target_bbox[3]) > 1e-6
            or abs(layer_px - pixel_size) > 1e-6
            or shape != expected_shape
        )

    def _spill(name: str, arr) -> np.ndarray:
        path = spill / f"{name}.npy"
        np.save(path, np.ascontiguousarray(arr, dtype=np.float32))
        return np.load(path, mmap_mode="r+")

    resampled_any = False

    # ── DSM (already in RAM from load_and_validate_dsm) ──
    if _needs_resample(dsm_arr.shape, dsm_transform):
        dsm_arr, dsm_transform = resample_to_grid(
            dsm_arr, dsm_transform, target_bbox, pixel_size, method="bilinear", src_crs=dsm_crs
        )
        resampled_any = True
    dsm_arr = _spill("dsm", dsm_arr)

    # ── Terrain layers, one at a time ──
    out: dict = {"cdsm_arr": None, "dem_arr": None, "tdsm_arr": None, "land_cover_arr": None}
    method_by_name = {"cdsm": "bilinear", "dem": "bilinear", "tdsm": "bilinear", "land_cover": "nearest"}
    log_by_name = {
        "cdsm": "  ✓ Canopy DSM (CDSM) provided",
        "dem": "  ✓ Ground elevation (DEM) provided",
        "tdsm": "  ✓ Trunk DSM (TDSM) provided",
        "land_cover": "  ✓ Land cover provided (albedo/emissivity derived from classification)",
    }
    for name in ("cdsm", "dem", "tdsm", "land_cover"):
        p = terrain_paths.get(name)
        if p is None:
            continue
        arr, transform, _, _ = io.load_raster(str(p))
        logger.info(log_by_name[name])
        if _needs_resample(arr.shape, transform):
            arr, _ = resample_to_grid(
                arr, transform, target_bbox, pixel_size, method=method_by_name[name], src_crs=dsm_crs
            )
            resampled_any = True
        out[f"{name}_arr"] = _spill(name, arr)
        del arr
    if terrain_paths.get("cdsm") is None:
        logger.info("  → No vegetation data - simulation without trees/vegetation")
    if terrain_paths.get("tdsm") is None and terrain_paths.get("cdsm") is not None:
        logger.info("  → No TDSM provided - will auto-generate from CDSM")

    # ── Walls (loaded by load_preprocessing_data; spill to release RAM) ──
    for name in ("wall_height", "wall_aspect"):
        arr = preprocess_data[f"{name}_arr"]
        tf = preprocess_data[f"{name}_transform"]
        if arr is None:
            out[f"{name}_arr"] = None
            continue
        if tf is not None and _needs_resample(arr.shape, tf):
            arr, _ = resample_to_grid(arr, tf, target_bbox, pixel_size, method="bilinear", src_crs=dsm_crs)
            resampled_any = True
        out[f"{name}_arr"] = _spill(name, arr)
        preprocess_data[f"{name}_arr"] = None  # release the RAM copy

    # ── SVF shape check (same semantics as align_rasters) ──
    if preprocess_data["svf_data"] is not None and preprocess_data["svf_data"].svf.shape != dsm_arr.shape:
        logger.warning(
            f"  ⚠ SVF shape {preprocess_data['svf_data'].svf.shape} doesn't match target shape "
            f"{dsm_arr.shape} - SVF resampling not yet implemented. "
            f"SVF cache will be dropped; recompute via SurfaceData.prepare() or compute_svf()."
        )
        preprocess_data["svf_data"] = None
        preprocess_data["shadow_data"] = None

    if resampled_any:
        logger.info(f"  ✓ Resampled to {dsm_arr.shape[1]}×{dsm_arr.shape[0]} pixels")
    else:
        logger.info("  ✓ No resampling needed - all rasters match target grid")
    logger.info(f"  Layers spilled to memmaps under {spill}")

    return {
        "dsm_arr": dsm_arr,
        "dsm_transform": dsm_transform,
        "dsm_crs": dsm_crs,
        "pixel_size": pixel_size,
        "cdsm_arr": out["cdsm_arr"],
        "dem_arr": out["dem_arr"],
        "tdsm_arr": out["tdsm_arr"],
        "land_cover_arr": out["land_cover_arr"],
        "wall_height_arr": out["wall_height_arr"],
        "wall_aspect_arr": out["wall_aspect_arr"],
        "svf_data": preprocess_data["svf_data"],
        "shadow_data": preprocess_data["shadow_data"],
    }


# ── 5. SurfaceData factory ─────────────────────────────────────────────────


def create_surface_instance(
    aligned_rasters: dict,
    pixel_size: float,
    trunk_ratio: float,
    *,
    dsm_relative: bool = False,
    cdsm_relative: bool = True,
    tdsm_relative: bool = True,
    min_object_height: float = 1.0,
    smooth_quantized_dem: bool = True,
    dem_smooth_sigma: float = 3.0,
) -> SurfaceData:
    """Assemble aligned rasters into a :class:`SurfaceData` instance.

    Deferred import of :class:`SurfaceData` to avoid a circular import
    between this module and ``surface.py``.
    """
    from .surface import SurfaceData

    surface_data = SurfaceData(
        dsm=aligned_rasters["dsm_arr"],
        cdsm=aligned_rasters["cdsm_arr"],
        dem=aligned_rasters["dem_arr"],
        tdsm=aligned_rasters["tdsm_arr"],
        land_cover=aligned_rasters["land_cover_arr"],
        wall_height=aligned_rasters["wall_height_arr"],
        wall_aspect=aligned_rasters["wall_aspect_arr"],
        svf=aligned_rasters["svf_data"],
        shadow_matrices=aligned_rasters["shadow_data"],
        pixel_size=pixel_size,
        trunk_ratio=trunk_ratio,
        dsm_relative=dsm_relative,
        cdsm_relative=cdsm_relative,
        tdsm_relative=tdsm_relative,
        min_object_height=min_object_height,
        smooth_quantized_dem=smooth_quantized_dem,
        dem_smooth_sigma=dem_smooth_sigma,
    )

    # Store geotransform and CRS for later export
    dsm_transform = aligned_rasters["dsm_transform"]
    if hasattr(dsm_transform, "to_gdal"):
        surface_data._geotransform = list(dsm_transform.to_gdal())
    else:
        surface_data._geotransform = dsm_transform
    surface_data._crs_wkt = aligned_rasters["dsm_crs"]

    layers_loaded = ["DSM"]
    if aligned_rasters["cdsm_arr"] is not None:
        layers_loaded.append("CDSM")
    if aligned_rasters["dem_arr"] is not None:
        layers_loaded.append("DEM")
    if aligned_rasters["tdsm_arr"] is not None:
        layers_loaded.append("TDSM")
    if aligned_rasters["land_cover_arr"] is not None:
        layers_loaded.append("land_cover")
    logger.info(f"  Layers loaded: {', '.join(layers_loaded)}")

    return surface_data
