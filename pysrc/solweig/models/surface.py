"""Surface and terrain data model.

Defines :class:`SurfaceData`, the primary input container for SOLWEIG
calculations.  Holds the DSM and optional rasters (CDSM, DEM, TDSM,
land cover, walls, SVF).  The :meth:`SurfaceData.prepare` class method
loads GeoTIFFs from disk, aligns extents, and computes or caches
walls and sky view factors automatically.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .. import io
from ..buffers import BufferPool
from ..cache import clear_stale_cache, pixel_size_tag, validate_cache
from ..loaders import get_lc_properties_from_params
from ..rustalgos import skyview
from ..solweig_logging import get_logger
from . import surface_loading
from .precomputed import ShadowArrays, SvfArrays
from .surface_views import OpticalPropertiesView, PreprocessedAuxiliaryView, SurfaceGeometryView

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)


class _ComputationCache:
    """Per-timestep computation caches for ``calculate_core_fused``.

    Separates transient cache state from the persistent ``SurfaceData``
    fields, keeping the dataclass focused on surface/terrain data.

    All attributes default to ``None`` and are lazily populated by
    ``computation.calculate_core_fused()`` on first access.
    """

    __slots__ = (
        "valid_mask_u8_cache",
        "valid_bbox_cache",
        "land_cover_props_cache",
        "buildings_mask_cache",
        "lc_grid_f32_cache",
        "gvf_geometry_cache",
        "gvf_geometry_cache_crop",
        "aniso_shadow_crop_cache",
    )

    def __init__(self) -> None:
        self.valid_mask_u8_cache: tuple[Any, Any] | None = None
        self.valid_bbox_cache: tuple[Any, Any] | None = None
        self.land_cover_props_cache: tuple[Any, Any] | None = None
        self.buildings_mask_cache: tuple[Any, Any] | None = None
        self.lc_grid_f32_cache: tuple[Any, Any] | None = None
        self.gvf_geometry_cache: tuple[Any, Any] | None = None
        self.gvf_geometry_cache_crop: tuple[Any, Any] | None = None
        self.aniso_shadow_crop_cache: tuple[Any, Any] | None = None

    def clear(self) -> None:
        """Reset all cached values."""
        for attr in self.__slots__:
            setattr(self, attr, None)

    def get_or_compute(self, slot: str, key: Any, compute: Callable[[], Any]) -> Any:
        """Return the cached value for ``slot`` when its key matches, else
        recompute via ``compute()``, store ``(key, value)`` in the slot, and
        return the new value.

        Centralises the 6 ad-hoc ``(key, value)`` tuple caches previously
        duplicated across ``computation.calculate_core_fused``. The slot must
        be one of the declared ``__slots__``.
        """
        existing = getattr(self, slot)
        if existing is not None and existing[0] == key:
            return existing[1]
        value = compute()
        setattr(self, slot, (key, value))
        return value


# Serialization helpers moved to models/surface_serialization.py. Aliased
# here under their historical underscored names for in-file callers.


def _detect_dem_quantization(dem: NDArray[np.floating], sample_size: int = 20000) -> float:
    """
    Detect the vertical quantization step of a DEM, in metres.

    Returns 0.0 if the DEM appears smooth (e.g. genuine sub-metre float data),
    or 1.0 if the DEM is effectively integer-quantized at 1 m. Integer-stored
    DEMs (int16 with 1 m precision) are a common source of visible stair-step
    contour artifacts in SVF over gently sloped open terrain, because each 1 m
    terrain step casts a discrete shadow at low-altitude sky patches.

    Detection strategy:

    - Integer dtype → unambiguous 1 m quantization.
    - Float dtype → sample residuals after 1 m rounding. Pixels that came
      directly from an int16 source (without bilinear mixing) have residuals
      of exactly 0, while pixels produced by bilinear resampling have
      fractional residuals. If a large fraction of samples have near-zero
      residuals, the underlying source was integer-quantized.

    A truly smooth float DEM has residuals uniformly distributed in [0, 0.5],
    so roughly ~2% of pixels fall within 0.01 m of an integer. An int16-sourced
    DEM with bilinear resampling has ~50%+ of pixels landing exactly on integer
    values (the pixels where the resample weights happened to pick out a single
    source cell). The 30 % threshold below cleanly separates the two regimes.

    Args:
        dem: DEM array (any dtype, interpreted as metres).
        sample_size: Number of random finite values to inspect.

    Returns:
        Quantization step in metres. 0.0 means "no quantization detected".
    """
    if dem is None or dem.size == 0:
        return 0.0

    # Integer dtype → unambiguous 1 m quantization (units are metres).
    if np.issubdtype(dem.dtype, np.integer):
        return 1.0

    finite = dem[np.isfinite(dem)]
    if finite.size < 100:
        return 0.0
    n = min(int(sample_size), finite.size)
    rng = np.random.default_rng(seed=0)  # deterministic → cache-stable
    # integers(replace=True implicit) is O(n) vs choice(replace=False) which
    # allocates an O(N) scratch for Fisher-Yates — at Madrid scale (500 M
    # finite pixels) choice() costs hundreds of MB. Duplicates among a
    # 20 000 / 500 M sample are statistically irrelevant for a residual
    # fraction estimate.
    idx = rng.integers(0, finite.size, size=n)
    sample = finite[idx]

    # Residual test: for int-sourced DEMs, a large fraction of samples have
    # residuals of exactly 0 (source pixels untouched by bilinear weights).
    residuals = np.abs(sample - np.round(sample))
    near_integer_fraction = float((residuals < 0.01).mean())
    if near_integer_fraction >= 0.30:
        return 1.0

    return 0.0


def _gaussian_smooth_2d(arr: NDArray[np.floating], sigma: float) -> NDArray[np.floating]:
    """
    Separable 2D Gaussian smoothing with edge-replicated boundaries, pure numpy.

    Avoids ``scipy.ndimage`` so the QGIS plugin runtime (which excludes
    scipy) can use this. Works in float32 throughout — matches codebase
    convention, halves peak memory vs a float64 scratch, and the resulting
    ~0.2 m error on 700 m DEM heights at sigma=3 is far below the 1 m
    quantization this smoother is meant to remove. Not suitable as a
    general-purpose scipy replacement outside the anti-quantization path.

    Edge replication (``mode='edge'``) is the correct boundary for terrain:
    extends the DEM as constant at the last known value, avoiding the
    phantom-copy artifacts that reflect-style padding would inject.

    Args:
        arr: 2D float array.
        sigma: Gaussian standard deviation in pixel units. Returns the
            input unchanged if ``sigma <= 0``.

    Returns:
        Smoothed array with the same shape and dtype as ``arr``.
    """
    if sigma <= 0.0 or arr.ndim != 2:
        return arr

    # Kernel radius = ceil(3σ) captures ~99.7% of the Gaussian mass.
    radius = int(np.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma)).astype(np.float32)
    kernel /= kernel.sum()

    orig_dtype = arr.dtype
    work: NDArray[np.float32] = np.asarray(arr, dtype=np.float32)
    rows, cols = work.shape

    # Horizontal pass.
    padded_h = np.pad(work, ((0, 0), (radius, radius)), mode="edge")
    h = np.zeros_like(work)
    for k_idx, w in enumerate(kernel):
        h += w * padded_h[:, k_idx : k_idx + cols]

    # Vertical pass.
    padded_v = np.pad(h, ((radius, radius), (0, 0)), mode="edge")
    out = np.zeros_like(work)
    for k_idx, w in enumerate(kernel):
        out += w * padded_v[k_idx : k_idx + rows, :]

    return out.astype(orig_dtype, copy=False)


def _max_shadow_height(dsm: np.ndarray, cdsm: np.ndarray | None = None, use_veg: bool = False) -> float:
    """
    Estimate maximum casting height above local ground for shadow reach logic.

    Uses local relief (max - min) instead of absolute elevation so tiled buffer
    sizing and SVF ray reach do not explode on high-elevation terrain.

    Vegetation is only considered when ``use_veg=True``. This differs from
    ``SurfaceData.max_height``, which is intentionally conservative for buffer
    sizing and always considers CDSM when present.
    """
    if dsm.size == 0 or not np.isfinite(dsm).any():
        return 0.0

    dsm_max = float(np.nanmax(dsm))
    dsm_min = float(np.nanmin(dsm))
    max_elevation = dsm_max
    if use_veg and cdsm is not None and cdsm.size > 0 and np.isfinite(cdsm).any():
        cdsm_max = float(np.nanmax(cdsm))
        if np.isfinite(cdsm_max):
            max_elevation = max(max_elevation, cdsm_max)
    height = max_elevation - dsm_min
    if not np.isfinite(height) or height <= 0:
        return 0.0
    return height


def looks_like_relative(
    layer: np.ndarray | None,
    reference: np.ndarray | None,
) -> bool:
    """Heuristically detect relative-height rasters passed as absolute.

    Prepared surfaces store CDSM/TDSM as absolute elevations.  If a
    layer tops out far below the reference surface, it very likely still
    contains height-above-ground values.
    """
    if layer is None or reference is None:
        return False
    if layer.size == 0 or reference.size == 0:
        return False
    if not np.isfinite(layer).any() or not np.isfinite(reference).any():
        return False
    layer_max = float(np.nanmax(layer))
    ref_min = float(np.nanmin(reference))
    if ref_min > 10 and layer_max < ref_min * 0.5:
        return True
    return bool(layer_max < 60 and ref_min > layer_max + 20)


@dataclass
class SurfaceData:
    """
    Surface/terrain data for SOLWEIG calculations.

    Only `dsm` is required. Other rasters are optional and will be
    treated as absent if not provided.

    Attributes:
        dsm: Digital Surface Model (elevation in meters). Required.
        cdsm: Canopy Digital Surface Model (vegetation heights). Optional.
        dem: Digital Elevation Model (ground elevation). Optional.
        tdsm: Trunk Digital Surface Model (trunk zone heights). Optional.
        land_cover: Land cover classification grid (UMEP standard IDs). Optional.
            IDs: 0=paved, 1=asphalt, 2=buildings, 5=grass, 6=bare_soil, 7=water.
            When provided, albedo and emissivity are derived from land cover.
        wall_height: Preprocessed wall heights (meters). Optional.
            If not provided, computed during preparation from DSM.
        wall_aspect: Preprocessed wall aspects (degrees, 0=N). Optional.
            If not provided, computed during preparation from DSM.
        svf: Preprocessed Sky View Factor arrays. Optional.
            If not provided, must be prepared explicitly before calculate()
            (e.g. via SurfaceData.prepare() or compute_svf()).
        shadow_matrices: Preprocessed shadow matrices for anisotropic sky. Optional.
        pixel_size: Pixel size in meters. Default 1.0.
        trunk_ratio: Ratio for auto-generating TDSM from CDSM. Default 0.25.
        dsm_relative: Whether DSM contains relative heights (above ground)
            rather than absolute elevations. Default False. If True, DEM is
            required and preprocess() converts DSM to absolute via DSM + DEM.
        cdsm_relative: Whether CDSM contains relative heights. Default True.
            If True and preprocess() is not called, a warning is issued.
        tdsm_relative: Whether TDSM contains relative heights. Default True.
            If True and preprocess() is not called, a warning is issued.

    Note:
        Albedo and emissivity are derived internally from land_cover using
        standard UMEP parameters. They cannot be directly specified.

    Note:
        max_height is auto-computed from dsm as: np.nanmax(dsm) - np.nanmin(dsm)

    Height Conventions:
        Each raster layer can independently use relative or absolute heights.
        The per-layer flags (``dsm_relative``, ``cdsm_relative``,
        ``tdsm_relative``) control the convention for each layer.

        **Relative Heights** (height above ground):
            - CDSM/TDSM: vegetation height above ground (e.g., 6m tree)
            - DSM: building/surface height above ground (requires DEM)
            - Typical range: 0-40m for CDSM, 0-10m for TDSM
            - Must call ``preprocess()`` before calculations

        **Absolute Heights** (elevation above sea level):
            - Values in the same vertical reference system
            - Example: DSM=127m, CDSM=133m means 6m vegetation
            - No preprocessing needed

        The internal algorithms (Rust) always use **absolute heights**. The
        ``preprocess()`` method converts relative → absolute using:
            dsm_absolute = dem + dsm_relative  (requires DEM)
            cdsm_absolute = base + cdsm_relative
            tdsm_absolute = base + tdsm_relative
        where ``base = DEM`` if available, else ``base = DSM``.

    Example:
        # Relative CDSM (common case):
        surface = SurfaceData(dsm=dsm, cdsm=cdsm_rel)
        surface.preprocess()  # Converts CDSM to absolute

        # Absolute CDSM:
        surface = SurfaceData(dsm=dsm, cdsm=cdsm_abs, cdsm_relative=False)

        # Mixed: absolute DSM, relative CDSM, absolute TDSM:
        surface = SurfaceData(
            dsm=dsm, cdsm=cdsm, tdsm=tdsm,
            cdsm_relative=True, tdsm_relative=False,
        )
        surface.preprocess()  # Only converts CDSM

        # Relative DSM (requires DEM):
        surface = SurfaceData(dsm=ndsm, dem=dem, dsm_relative=True)
        surface.preprocess()  # Converts DSM to absolute via DEM + nDSM
    """

    # Surface rasters
    dsm: NDArray[np.floating]
    cdsm: NDArray[np.floating] | None = None
    dem: NDArray[np.floating] | None = None
    tdsm: NDArray[np.floating] | None = None
    albedo: NDArray[np.floating] | None = None
    emissivity: NDArray[np.floating] | None = None
    land_cover: NDArray[np.integer] | None = None

    # Preprocessing data (walls, SVF, shadows)
    wall_height: NDArray[np.floating] | None = None
    wall_aspect: NDArray[np.floating] | None = None
    svf: SvfArrays | None = None
    shadow_matrices: ShadowArrays | None = None

    # Grid properties
    pixel_size: float = 1.0
    trunk_ratio: float = 0.25  # Trunk zone ratio for auto-generating TDSM from CDSM
    dsm_relative: bool = False  # Whether DSM contains relative heights (requires DEM)
    cdsm_relative: bool = True  # Whether CDSM contains relative heights
    tdsm_relative: bool = True  # Whether TDSM contains relative heights
    min_object_height: float = 1.0  # Min nDSM height (m) to cast shadows; below this, DSM is flattened to DEM
    # DEM integer-quantization smoothing: int16 DEMs (e.g. PNOA-LiDAR at 1m precision)
    # produce visible stair-step contour artifacts in SVF over gently sloped open
    # terrain. When enabled and an integer-quantized DEM is detected in preprocess(),
    # a small Gaussian smooth is applied (sigma in pixel units at target resolution)
    # to break the quantization without touching nDSM/buildings. Set to False to
    # preserve bit-exact legacy behaviour or when the DEM has genuine 1m-step terrain.
    smooth_quantized_dem: bool = True
    dem_smooth_sigma: float = 3.0

    # Internal state
    _dem_quantization_m: float = field(default=0.0, init=False, repr=False)  # detected in preprocess()
    _nan_filled: bool = field(default=False, init=False, repr=False)
    _preprocessed: bool = field(default=False, init=False, repr=False)
    _geotransform: list[float] | None = field(default=None, init=False, repr=False)  # GDAL geotransform
    _crs_wkt: str | None = field(default=None, init=False, repr=False)  # CRS as WKT string
    _buffer_pool: BufferPool | None = field(default=None, init=False, repr=False)  # Reusable array pool
    _gvf_geometry_cache: object = field(default=None, init=False, repr=False)  # Rust GVF geometry cache
    _valid_mask: NDArray[np.bool_] | None = field(default=None, init=False, repr=False)  # Combined valid mask
    # Per-timestep computation caches (grouped in _ComputationCache)
    _cache: _ComputationCache = field(default_factory=_ComputationCache, init=False, repr=False)

    def __post_init__(self):
        # Ensure dsm is float32 for memory efficiency
        self.dsm = np.asarray(self.dsm, dtype=np.float32)

        # Convert optional surface arrays if provided
        if self.cdsm is not None:
            self.cdsm = np.asarray(self.cdsm, dtype=np.float32)
        if self.dem is not None:
            self.dem = np.asarray(self.dem, dtype=np.float32)
        if self.tdsm is not None:
            self.tdsm = np.asarray(self.tdsm, dtype=np.float32)
        if self.albedo is not None:
            self.albedo = np.asarray(self.albedo, dtype=np.float32)
        if self.emissivity is not None:
            self.emissivity = np.asarray(self.emissivity, dtype=np.float32)
        if self.land_cover is not None:
            self.land_cover = np.asarray(self.land_cover, dtype=np.uint8)

        # Convert optional preprocessing arrays if provided
        if self.wall_height is not None:
            self.wall_height = np.asarray(self.wall_height, dtype=np.float32)
        if self.wall_aspect is not None:
            self.wall_aspect = np.asarray(self.wall_aspect, dtype=np.float32)

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        load_svf: bool = True,
    ) -> SurfaceData:
        """
        Load a pre-prepared surface directory produced by :meth:`prepare`.

        This is the counterpart to :meth:`prepare`: it reads the cached
        rasters, walls, SVF, and shadow matrices from ``directory`` and
        returns a ready-to-use :class:`SurfaceData`.  The loaded data is
        marked as fully preprocessed so that :meth:`fill_nan` and
        :meth:`preprocess` are no-ops — this prevents NaN "no vegetation"
        markers in CDSM/TDSM from being overwritten with DEM values.

        Args:
            directory: Path to the prepared surface directory (the
                ``working_dir`` passed to :meth:`prepare`).
            load_svf: Whether to load SVF arrays and shadow matrices.
                Default True.  Set to False if only the raster data is
                needed (avoids loading large SVF files).

        Returns:
            SurfaceData ready for :func:`calculate`.

        Raises:
            FileNotFoundError: If ``metadata.json`` or the DSM raster
                is missing.
            ValueError: If the loaded CDSM/TDSM appears to contain
                relative heights instead of absolute elevations.

        Example::

            surface = SurfaceData.load("prepared_surface/")
            result = calculate(surface, weather, location)
        """
        from .precomputed import PrecomputedData

        directory = Path(directory)

        # Read metadata (written by prepare → save_cleaned, or QGIS plugin)
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Not a valid prepared surface directory: {directory}\n"
                "Missing metadata.json.  Run SurfaceData.prepare() first."
            )

        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        pixel_size = metadata.get("pixel_size", 1.0)

        # Determine raster source directory: prefer cleaned/, fall back to root
        cleaned_dir = directory / "cleaned"
        if (cleaned_dir / "dsm.tif").exists():
            raster_dir = cleaned_dir
            logger.info(f"Loading prepared surface from {cleaned_dir}")
        elif (directory / "dsm.tif").exists():
            raster_dir = directory
            logger.info(f"Loading prepared surface from {directory} (legacy layout)")
        else:
            raise FileNotFoundError(
                f"DSM raster not found in {cleaned_dir} or {directory}.\nRun SurfaceData.prepare() first."
            )

        # Load DSM (required)
        dsm_arr, gt, crs_wkt, _ = io.load_raster(str(raster_dir / "dsm.tif"))
        logger.info(f"  DSM: {dsm_arr.shape[1]}×{dsm_arr.shape[0]} pixels")

        # Load optional layers based on metadata flags
        def _load_if_present(name: str, flag: str) -> np.ndarray | None:
            if not metadata.get(flag, False):
                return None
            try:
                arr, _, _, _ = io.load_raster(str(raster_dir / f"{name}.tif"))
                return arr
            except FileNotFoundError:
                logger.warning(f"  {name}.tif flagged in metadata but not found")
                return None

        cdsm = _load_if_present("cdsm", "has_cdsm")
        dem = _load_if_present("dem", "has_dem")
        tdsm = _load_if_present("tdsm", "has_tdsm")
        lc = _load_if_present("land_cover", "has_land_cover")
        land_cover = lc.astype(np.uint8) if lc is not None else None

        # Load walls (always present after prepare)
        wall_height = None
        wall_aspect = None
        try:
            wall_height, _, _, _ = io.load_raster(str(raster_dir / "wall_height.tif"))
            wall_aspect, _, _, _ = io.load_raster(str(raster_dir / "wall_aspect.tif"))
        except FileNotFoundError:
            pass

        # Validate before expensive SVF loading
        base_surface = dem if dem is not None else dsm_arr
        if cdsm is not None and looks_like_relative(cdsm, base_surface):
            raise ValueError(
                "Loaded CDSM appears to contain relative heights (height above ground) "
                "instead of absolute elevations.  The prepared surface may be corrupt — "
                "re-run SurfaceData.prepare() with the correct cdsm_relative flag."
            )
        if tdsm is not None and looks_like_relative(tdsm, base_surface):
            raise ValueError(
                "Loaded TDSM appears to contain relative heights (height above ground) "
                "instead of absolute elevations.  The prepared surface may be corrupt — "
                "re-run SurfaceData.prepare() with the correct tdsm_relative flag."
            )

        # Construct SurfaceData
        surface = cls(
            dsm=dsm_arr,
            cdsm=cdsm,
            dem=dem,
            tdsm=tdsm,
            land_cover=land_cover,
            wall_height=wall_height,
            wall_aspect=wall_aspect,
            pixel_size=pixel_size,
            dsm_relative=False,
            cdsm_relative=False,
            tdsm_relative=False,
        )
        surface._geotransform = gt
        surface._crs_wkt = crs_wkt
        # Mark as fully preprocessed — the saved rasters are already
        # absolute with NaN meaning "absent" for CDSM/TDSM.
        surface._preprocessed = True
        surface._nan_filled = True

        # Load SVF and shadow matrices (expensive — done after validation)
        if load_svf:
            precomputed = PrecomputedData.prepare(svf_dir=str(directory))
            if precomputed.svf is not None:
                surface.svf = precomputed.svf
            if precomputed.shadow_matrices is not None:
                surface.shadow_matrices = precomputed.shadow_matrices

        layers = ["DSM"]
        for name, arr in [
            ("CDSM", cdsm),
            ("DEM", dem),
            ("TDSM", tdsm),
            ("land_cover", land_cover),
            ("walls", wall_height),
            ("SVF", surface.svf),
            ("shadows", surface.shadow_matrices),
        ]:
            if arr is not None:
                layers.append(name)
        logger.info(f"  Loaded: {', '.join(layers)}")

        return surface

    @classmethod
    def prepare(
        cls,
        dsm: str | Path | NDArray[np.floating],
        working_dir: str | Path | None = None,
        cdsm: str | Path | NDArray[np.floating] | None = None,
        dem: str | Path | NDArray[np.floating] | None = None,
        tdsm: str | Path | NDArray[np.floating] | None = None,
        land_cover: str | Path | NDArray[np.integer] | None = None,
        wall_height: str | Path | NDArray[np.floating] | None = None,
        wall_aspect: str | Path | NDArray[np.floating] | None = None,
        svf_dir: str | Path | None = None,
        bbox: list[float] | None = None,
        pixel_size: float | None = None,
        trunk_ratio: float = 0.25,
        dsm_relative: bool = False,
        cdsm_relative: bool = True,
        tdsm_relative: bool = True,
        min_object_height: float = 1.0,
        smooth_quantized_dem: bool = True,
        dem_smooth_sigma: float = 3.0,
        force_recompute: bool = False,
        tile_size: int | None = None,
        feedback: Any = None,
    ) -> SurfaceData:
        """
        Prepare surface data for SOLWEIG calculations.

        Loads inputs, computes walls, SVF, and shadow matrices, and returns
        a ready-to-use :class:`SurfaceData`. This is the only setup step
        needed before calling :func:`calculate`.

        Accepts either **file paths** (GeoTIFF) or **numpy arrays**:

        - *File mode* (dsm is a path): loads and aligns rasters, caches
          results in ``working_dir`` for fast reuse.
        - *Array mode* (dsm is an ndarray): works in memory.
          ``pixel_size`` is required; ``working_dir`` is not needed.

        Args:
            dsm: DSM as a GeoTIFF path or numpy array (required).
            working_dir: Cache directory (required for file mode).
            cdsm: Canopy height model (tree tops). Optional.
            dem: Ground elevation model. Optional.
            tdsm: Trunk height model. Optional; auto-generated from CDSM
                if not provided (using ``trunk_ratio``).
            land_cover: Land cover classification grid. Optional.
            wall_height: Pre-computed wall heights. Optional; computed
                from DSM if not provided.
            wall_aspect: Pre-computed wall aspects. Optional; computed
                from DSM if not provided.
            svf_dir: Directory with existing SVF files (file mode only).
            bbox: Bounding box [minx, miny, maxx, maxy] (file mode only).
            pixel_size: Pixel size in meters. Required for array mode;
                extracted from GeoTIFF in file mode.
            trunk_ratio: Trunk-to-canopy height ratio for auto TDSM. Default 0.25.
            dsm_relative: DSM values are height above ground (not elevation). Default False.
            cdsm_relative: CDSM values are height above ground. Default True.
            tdsm_relative: TDSM values are height above ground. Default True.
            min_object_height: Minimum nDSM height (m) for shadow casting.
                DSM pixels below this height above DEM are flattened to remove
                kerbs, street furniture, and LiDAR noise. Default 1.0. Set to
                0 to disable. Requires DEM.
            smooth_quantized_dem: Apply Gaussian smoothing to the DEM when it
                is detected to be integer-quantized (e.g. int16 with 1 m
                precision). Default True. Integer-stored DEMs produce visible
                stair-step contour artifacts in SVF over gently sloped open
                terrain because each 1 m terrain step casts a discrete shadow
                at low-altitude sky patches; smoothing recovers the sub-metre
                variation that was lost at storage time. Disable to preserve
                bit-exact legacy behaviour or when your terrain has genuine
                1 m steps (e.g. agricultural terraces).
            dem_smooth_sigma: Gaussian standard deviation in pixel units at
                target resolution. Default 3.0 — FWHM ≈ 7 px gives bulletproof
                elimination of the stair-step artifact across all sky patches
                including the lowest (3°). Softens real terrain features
                smaller than ~15 m horizontal scale; below that scale, a
                1 m-quantized DEM has no physically meaningful information
                anyway (the sharp transitions are storage truncation, not
                real features). Override to 1.5–2.0 for high-resolution
                non-quantized DEMs or when 10 m-scale terrain features must
                be preserved. Only used when ``smooth_quantized_dem=True``
                and a quantized DEM is detected.
            force_recompute: Recompute walls/SVF even if cached (file mode only).
            tile_size: Core tile side length in pixels for SVF tiling.
                If None (default), auto-calculated from available resources.
                Minimum 256.
            feedback: QGIS QgsProcessingFeedback for progress/cancellation.

        Returns:
            SurfaceData ready for :func:`calculate`.

        Example::

            # From GeoTIFF files
            surface = SurfaceData.prepare(dsm="dsm.tif", working_dir="cache/")

            # From numpy arrays
            surface = SurfaceData.prepare(dsm=dsm_array, pixel_size=1.0)
        """
        # Array mode: delegate to in-memory preparation
        if isinstance(dsm, np.ndarray):
            # Runtime validation in _prepare_from_arrays rejects non-array args
            return cls._prepare_from_arrays(
                dsm=cast("NDArray[np.floating]", dsm),
                cdsm=cast("NDArray[np.floating] | None", cdsm),
                dem=cast("NDArray[np.floating] | None", dem),
                tdsm=cast("NDArray[np.floating] | None", tdsm),
                land_cover=cast("NDArray[np.integer] | None", land_cover),
                wall_height=cast("NDArray[np.floating] | None", wall_height),
                wall_aspect=cast("NDArray[np.floating] | None", wall_aspect),
                pixel_size=pixel_size,
                trunk_ratio=trunk_ratio,
                dsm_relative=dsm_relative,
                cdsm_relative=cdsm_relative,
                tdsm_relative=tdsm_relative,
                min_object_height=min_object_height,
                smooth_quantized_dem=smooth_quantized_dem,
                dem_smooth_sigma=dem_smooth_sigma,
            )

        # File mode: working_dir is required
        if working_dir is None:
            raise ValueError("working_dir is required when dsm is a file path")

        working_path = Path(working_dir)
        dsm_path = cast("str | Path", dsm)

        # Capture every input that determines the output of a prepare() call
        # into a single fingerprint. Used by the fast-path check below and by
        # save_cleaned() at the end of a cold build — a single source of
        # truth prevents check/save drift if new kwargs are added later.
        from ..cache import compare_prepare_fingerprints, compute_prepare_fingerprint

        def _build_prepare_fingerprint() -> dict:
            return compute_prepare_fingerprint(
                sources={
                    "dsm": dsm_path,
                    "dem": cast("str | Path | None", dem),
                    "cdsm": cast("str | Path | None", cdsm),
                    "tdsm": cast("str | Path | None", tdsm),
                    "land_cover": cast("str | Path | None", land_cover),
                    "wall_height": cast("str | Path | None", wall_height),
                    "wall_aspect": cast("str | Path | None", wall_aspect),
                },
                kwargs={
                    "pixel_size": pixel_size,
                    "bbox": list(bbox) if bbox is not None else None,
                    "trunk_ratio": float(trunk_ratio),
                    "dsm_relative": bool(dsm_relative),
                    "cdsm_relative": bool(cdsm_relative),
                    "tdsm_relative": bool(tdsm_relative),
                    "min_object_height": float(min_object_height),
                    "smooth_quantized_dem": bool(smooth_quantized_dem),
                    "dem_smooth_sigma": float(dem_smooth_sigma),
                },
            )

        # ── Fast-path: reuse cleaned/ output from a previous prepare() run ──
        # When working_dir already holds a matching fingerprint we can skip
        # the entire load → resample → preprocess → walls → SVF pipeline and
        # jump straight to SurfaceData.load(). The fingerprint covers every
        # source file's mtime/size and every preprocessing kwarg that
        # affects the output. Skipped when ``svf_dir`` is supplied because
        # that's an explicit "load SVF from elsewhere" request.
        if not force_recompute and svf_dir is None:
            fast_path_metadata = working_path / "metadata.json"
            fast_path_cleaned = working_path / "cleaned" / "dsm.tif"
            if fast_path_metadata.exists() and fast_path_cleaned.exists():
                try:
                    with open(fast_path_metadata, encoding="utf-8") as fp:
                        stored_metadata = json.load(fp)
                except (OSError, json.JSONDecodeError) as e:
                    logger.debug(f"Could not read fast-path metadata: {e}")
                    stored_metadata = None

                if stored_metadata is not None:
                    stored_fp = stored_metadata.get("prepare_fingerprint")
                    if stored_fp is None:
                        logger.info(
                            "Fast-path: metadata.json has no prepare_fingerprint "
                            "(legacy cache), falling through to full prepare"
                        )
                    else:
                        mismatches = compare_prepare_fingerprints(stored_fp, _build_prepare_fingerprint())
                        if not mismatches:
                            logger.info(f"Fast-path cache hit — loading prepared surface from {working_path}")
                            if feedback is not None and hasattr(feedback, "setProgressText"):
                                feedback.setProgressText("Loading cached surface (walls + SVF from previous run)...")
                            # Corrupted cache → fall back to full rebuild.
                            try:
                                return cls.load(working_path)
                            except (FileNotFoundError, OSError) as e:
                                logger.warning(
                                    f"Fast-path load failed ({type(e).__name__}: {e}); falling through to full prepare"
                                )
                        else:
                            logger.info(
                                f"Fast-path cache invalidated ({len(mismatches)} change"
                                f"{'s' if len(mismatches) != 1 else ''}):"
                            )
                            for reason in mismatches:
                                logger.info(f"  - {reason}")
                            logger.info("Rebuilding from source rasters…")

        logger.info("Preparing surface data from GeoTIFF files...")
        if feedback is not None and hasattr(feedback, "setProgressText"):
            feedback.setProgressText("Loading and aligning rasters...")

        # Load and validate DSM — dsm is str | Path after the isinstance guard above
        dsm_arr, dsm_transform, dsm_crs, pixel_size = surface_loading.load_and_validate_dsm(dsm_path, pixel_size)

        # Load optional terrain rasters — these are str | Path | None after array branch
        terrain_rasters = surface_loading.load_terrain_rasters(
            cast("str | Path | None", cdsm),
            cast("str | Path | None", dem),
            cast("str | Path | None", tdsm),
            cast("str | Path | None", land_cover),
            trunk_ratio,
        )

        # Load preprocessing data (walls, SVF). ``working_path`` was
        # already constructed above for the fast-path check — reuse it.
        preprocess_data = surface_loading.load_preprocessing_data(
            cast("str | Path | None", wall_height),
            cast("str | Path | None", wall_aspect),
            svf_dir,
            working_path,
            force_recompute,
            pixel_size=pixel_size,
        )

        # Compute extent, validate bbox, and resample all rasters
        aligned_rasters = surface_loading.align_rasters(
            dsm_arr,
            dsm_transform,
            dsm_crs,
            pixel_size,
            terrain_rasters,
            preprocess_data,
            bbox,
        )

        # Create SurfaceData instance
        surface_data = surface_loading.create_surface_instance(
            aligned_rasters,
            pixel_size,
            trunk_ratio,
            dsm_relative=dsm_relative,
            cdsm_relative=cdsm_relative,
            tdsm_relative=tdsm_relative,
            min_object_height=min_object_height,
            smooth_quantized_dem=smooth_quantized_dem,
            dem_smooth_sigma=dem_smooth_sigma,
        )

        # Preprocess layers: convert relative heights to absolute and
        # enforce DSM >= DEM (terrain is the minimum surface elevation).
        # This must happen BEFORE cache validation and walls/SVF so they
        # see absolute heights and hashes match the post-processed arrays.
        needs_preprocess = (
            dsm_relative
            or (cdsm_relative and surface_data.cdsm is not None)
            or (tdsm_relative and surface_data.tdsm is not None)
            or surface_data.dem is not None
        )
        if needs_preprocess:
            logger.debug("  Preprocessing heights")
            surface_data.preprocess()
            # Sync aligned_rasters so cache helpers see absolute heights
            aligned_rasters["dsm_arr"] = surface_data.dsm
            if surface_data.cdsm is not None:
                aligned_rasters["cdsm_arr"] = surface_data.cdsm
            if surface_data.tdsm is not None:
                aligned_rasters["tdsm_arr"] = surface_data.tdsm

        # Validate cached walls against current (post-processed) inputs.
        # Cache metadata is written by _compute_and_cache_walls using the
        # post-processed DSM, so we must validate after preprocessing.
        if (
            preprocess_data["wall_height_arr"] is not None
            and not preprocess_data["compute_walls"]
            and not force_recompute
        ):
            walls_cache_dir = working_path / "walls" / pixel_size_tag(pixel_size)
            if not walls_cache_dir.exists():
                walls_cache_dir = working_path / "walls"  # legacy fallback
            dsm_arr = aligned_rasters["dsm_arr"]
            cdsm_arr = aligned_rasters.get("cdsm_arr")
            if not validate_cache(walls_cache_dir, dsm_arr, pixel_size, cdsm_arr):
                logger.info("  → Wall cache stale, clearing and recomputing walls...")
                clear_stale_cache(walls_cache_dir)
                preprocess_data["wall_height_arr"] = None
                preprocess_data["wall_aspect_arr"] = None
                preprocess_data["compute_walls"] = True
                surface_data.wall_height = None
                surface_data.wall_aspect = None

        # Validate cached SVF against current (post-processed) inputs.
        # Hashing after preprocessing ensures that relative-height runs
        # don't spuriously invalidate the cache.
        if preprocess_data["svf_data"] is not None and not force_recompute:
            dsm_arr = aligned_rasters["dsm_arr"]
            cdsm_arr = aligned_rasters.get("cdsm_arr")
            svf_source = preprocess_data.get("svf_source", "none")

            # Resolve the SVF cache directory (pixel-size-keyed or legacy)
            svf_base = working_path / "svf" / pixel_size_tag(pixel_size)
            if not svf_base.exists():
                svf_base = working_path / "svf"  # legacy fallback

            cache_valid = False
            if svf_source == "memmap":
                # Memmap has cache_meta.json — use hash-based validation
                cache_valid = validate_cache(svf_base / "memmap", dsm_arr, pixel_size, cdsm_arr)
            elif svf_source == "zip":
                # Try metadata first, fall back to shape check
                zip_meta_dir = svf_base
                cache_valid = validate_cache(zip_meta_dir, dsm_arr, pixel_size, cdsm_arr)
                if not cache_valid:
                    # Legacy zip without metadata — validate by shape only
                    svf_shape = preprocess_data["svf_data"].svf.shape
                    cache_valid = svf_shape == dsm_arr.shape
                    if not cache_valid:
                        logger.info(f"  SVF shape {svf_shape} doesn't match DSM {dsm_arr.shape}")

            if not cache_valid:
                logger.info("  → Cache stale, clearing and recomputing SVF...")
                clear_stale_cache(svf_base / "memmap")
                # Also remove zip/npz/memmaps so stale data doesn't persist
                for stale_file in ("svfs.zip", "shadowmats.npz"):
                    stale_path = svf_base / stale_file
                    if stale_path.exists():
                        stale_path.unlink()
                stale_shadow_memmaps = svf_base / "shadow_memmaps"
                if stale_shadow_memmaps.exists():
                    shutil.rmtree(stale_shadow_memmaps, ignore_errors=True)
                preprocess_data["svf_data"] = None
                preprocess_data["compute_svf"] = True
                surface_data.svf = None

        # Compute and cache walls if needed
        compute_walls = preprocess_data["compute_walls"]
        compute_svf = preprocess_data["compute_svf"]

        if compute_walls:
            if feedback is not None and hasattr(feedback, "setProgressText"):
                feedback.setProgressText("Computing wall heights and aspects...")
            # Walls use 10-30% when SVF follows, or 10-75% when walls-only
            walls_range = (10, 30 if compute_svf else 75) if feedback is not None else None
            cls._compute_and_cache_walls(
                surface_data,
                aligned_rasters,
                working_path,
                pixel_size=pixel_size,
                feedback=feedback,
                progress_range=walls_range,
            )

        if compute_svf:
            if feedback is not None and hasattr(feedback, "setProgressText"):
                feedback.setProgressText("Computing Sky View Factor...")
            # SVF uses ~30-75% of QGIS progress bar
            svf_range = (30, 75) if feedback is not None else None
            cls._compute_and_cache_svf(
                surface_data,
                aligned_rasters,
                working_path,
                trunk_ratio,
                tile_size=tile_size,
                feedback=feedback,
                progress_range=svf_range,
            )

        # Compute unified valid mask, apply across all layers, crop to valid bbox
        surface_data.compute_valid_mask()
        surface_data.apply_valid_mask()
        surface_data.crop_to_valid_bbox()

        # Persist the fingerprint so the next prepare() call can short-circuit.
        # Reuses the same builder the fast-path check used so the stored and
        # compared fingerprints cannot drift.
        surface_data.save_cleaned(working_path, prepare_fingerprint=_build_prepare_fingerprint())

        logger.info("✓ Surface data prepared successfully")
        return surface_data

    @classmethod
    def _prepare_from_arrays(
        cls,
        dsm: NDArray[np.floating],
        *,
        cdsm: NDArray[np.floating] | None = None,
        dem: NDArray[np.floating] | None = None,
        tdsm: NDArray[np.floating] | None = None,
        land_cover: NDArray[np.integer] | None = None,
        wall_height: NDArray[np.floating] | None = None,
        wall_aspect: NDArray[np.floating] | None = None,
        pixel_size: float | None = None,
        trunk_ratio: float = 0.25,
        dsm_relative: bool = False,
        cdsm_relative: bool = True,
        tdsm_relative: bool = True,
        min_object_height: float = 1.0,
        smooth_quantized_dem: bool = True,
        dem_smooth_sigma: float = 3.0,
    ) -> SurfaceData:
        """Prepare surface data from in-memory numpy arrays."""
        from ..physics import wallalgorithms as wa

        # Validate pixel_size
        if pixel_size is None:
            raise ValueError("pixel_size is required when dsm is a numpy array")

        # Validate no mixing of arrays and file paths
        raster_args = {
            "cdsm": cdsm,
            "dem": dem,
            "tdsm": tdsm,
            "land_cover": land_cover,
            "wall_height": wall_height,
            "wall_aspect": wall_aspect,
        }
        for name, val in raster_args.items():
            if val is not None and not isinstance(val, np.ndarray):
                raise TypeError(f"{name} must be a numpy array when dsm is a numpy array, got {type(val).__name__}")

        # Validate shapes match
        for name, val in raster_args.items():
            if val is not None and val.shape != dsm.shape:
                raise ValueError(f"{name} shape {val.shape} does not match dsm shape {dsm.shape}")

        logger.info("Preparing surface data from arrays...")

        # Construct SurfaceData
        surface_data = cls(
            dsm=dsm,
            cdsm=cdsm,
            dem=dem,
            tdsm=tdsm,
            land_cover=land_cover,
            wall_height=wall_height,
            wall_aspect=wall_aspect,
            pixel_size=pixel_size,
            trunk_ratio=trunk_ratio,
            dsm_relative=dsm_relative,
            cdsm_relative=cdsm_relative,
            tdsm_relative=tdsm_relative,
            min_object_height=min_object_height,
            smooth_quantized_dem=smooth_quantized_dem,
            dem_smooth_sigma=dem_smooth_sigma,
        )

        # Preprocess: convert relative heights to absolute and
        # enforce DSM >= DEM (terrain is the minimum surface elevation).
        needs_preprocess = (
            dsm_relative
            or (cdsm_relative and surface_data.cdsm is not None)
            or (tdsm_relative and surface_data.tdsm is not None)
            or surface_data.dem is not None
        )
        if needs_preprocess:
            logger.debug("  Preprocessing heights")
            surface_data.preprocess()

        # Compute walls if not provided
        if surface_data.wall_height is None or surface_data.wall_aspect is None:
            logger.info("  Computing walls from DSM...")
            dsm_f32 = surface_data.dsm.astype(np.float32)
            walls = wa.findwalls(dsm_f32, 1.0)
            dsm_scale = 1.0 / pixel_size
            dirwalls = wa.filter1Goodwin_as_aspect_v3(walls, dsm_scale, dsm_f32)
            surface_data.wall_height = walls.astype(np.float32)
            surface_data.wall_aspect = dirwalls.astype(np.float32)

        # Compute SVF
        surface_data.compute_svf()

        logger.info("✓ Surface data prepared successfully (array mode)")
        return surface_data

    # Static raster-loading helpers extracted to surface_loading.py.
    # See `surface_loading.load_and_validate_dsm` etc.

    # Wall + SVF computation extracted to surface_compute.py.
    # Back-compat wrappers below preserve the previous static-method API
    # so external callers (QGIS plugin: SD._compute_and_cache_svf, etc.)
    # keep working unchanged.
    @staticmethod
    def _compute_and_cache_walls(*args, **kwargs):
        """Backwards-compat wrapper — see :func:`surface_compute.compute_and_cache_walls`."""
        from . import surface_compute as _sc

        return _sc.compute_and_cache_walls(*args, **kwargs)

    @staticmethod
    def _compute_and_cache_svf(*args, **kwargs):
        """Backwards-compat wrapper — see :func:`surface_compute.compute_and_cache_svf`."""
        from . import surface_compute as _sc

        return _sc.compute_and_cache_svf(*args, **kwargs)

    def preprocess(self) -> None:
        """
        Convert layers from relative to absolute heights based on per-layer flags.

        Converts each layer that is flagged as relative (``dsm_relative``,
        ``cdsm_relative``, ``tdsm_relative``) to absolute heights. Layers
        already flagged as absolute are left unchanged.

        This method:
        1. (Optional) Detects integer-quantized DEMs (e.g. int16 with 1 m
           precision) and Gaussian-smooths the DEM to remove stair-step
           contour artifacts in downstream SVF. Controlled by
           ``smooth_quantized_dem`` and ``dem_smooth_sigma`` on the dataclass.
        2. Converts DSM from relative to absolute if ``dsm_relative=True``
           (requires DEM: ``dsm_absolute = dem + dsm_relative``)
        3. Auto-generates TDSM from CDSM * trunk_ratio if TDSM is not provided
        4. Converts CDSM from relative to absolute if ``cdsm_relative=True``
        5. Converts TDSM from relative to absolute if ``tdsm_relative=True``
        6. NaN's canopy pixels that sit below the DSM (inside buildings)

        Note:
            This method modifies arrays in-place and clears the per-layer
            relative flags once conversion is done. ``self.dem`` may also be
            overwritten with a smoothed copy (step 1) — the original values
            are discarded.
        """
        if self._preprocessed:
            return

        # Fill NaN in surface layers before any height conversion
        self.fill_nan()

        # Detect integer-quantized DEM and smooth it. Must happen BEFORE the DSM
        # relative→absolute conversion so that the smoothed DEM feeds into
        # `DSM = DEM + nDSM` and all downstream base-relative threshold checks
        # (CDSM/TDSM sub-threshold NaN, canopy-below-DSM) use the smoothed base
        # consistently. Buildings are unaffected: nDSM is integer but preserved,
        # and DSM = smoothed_DEM + nDSM keeps building geometry intact.
        if self.smooth_quantized_dem and self.dem is not None:
            q = _detect_dem_quantization(self.dem)
            self._dem_quantization_m = q
            if q > 0.0 and self.dem_smooth_sigma > 0.0:
                logger.info(
                    f"Smoothing quantized DEM (Q={q:.2f}m, sigma={self.dem_smooth_sigma}px) "
                    f"to suppress stair-step SVF artifacts over gently sloped terrain"
                )
                # self.dem is already float32 (coerced in __post_init__), and
                # _gaussian_smooth_2d also runs in float32.
                self.dem = _gaussian_smooth_2d(self.dem, self.dem_smooth_sigma)

        threshold = np.float32(max(0.1, self.min_object_height))
        zero32 = np.float32(0.0)
        nan32 = np.float32(np.nan)

        # Step 1: Convert DSM from relative to absolute (requires DEM)
        if self.dsm_relative:
            if self.dem is None:
                raise ValueError(
                    "DSM is flagged as relative (dsm_relative=True) but no DEM "
                    "is provided. A DEM is required to convert relative DSM "
                    "(height above ground) to absolute elevations."
                )
            logger.info("Converting relative DSM to absolute: DSM = DEM + nDSM")
            self.dsm = np.asarray(self.dem + self.dsm, dtype=np.float32)
            self.dsm_relative = False

        # Step 1b: Ensure DSM is never below DEM (terrain is the minimum surface)
        # This handles cases where an absolute DSM has gaps or zero-valued
        # pixels (e.g. a building-only nDSM passed without dsm_relative=True)
        # that would otherwise sit below the terrain, producing incorrect
        # shadows.
        if self.dem is not None:
            below = self.dsm < self.dem
            if np.any(below):
                n = int(below.sum())
                logger.info(f"Raising {n} DSM pixels to DEM (DSM was below terrain)")
                self.dsm = np.asarray(np.maximum(self.dsm, self.dem), dtype=np.float32)

        # Step 1c: Flatten sub-threshold DSM features to DEM
        # Small nDSM residuals (kerbs, street furniture, LiDAR noise) cast
        # spurious shadows at low sun angles.  Flatten to DEM where the DSM
        # protrudes less than min_object_height above the terrain.
        if self.dem is not None and self.min_object_height > 0:
            ndsm = self.dsm - self.dem
            small = (ndsm > 0) & (ndsm < self.min_object_height)
            n_flat = int(small.sum())
            if n_flat:
                self.dsm[small] = self.dem[small]
                logger.info(
                    f"Flattened {n_flat} DSM pixels below {self.min_object_height}m "
                    f"nDSM to DEM (removing sub-threshold features)"
                )

        # Step 2: Auto-generate TDSM from trunk ratio if CDSM provided but not TDSM
        if self.cdsm is not None and self.tdsm is None:
            logger.info(f"Auto-generating TDSM from CDSM using trunk_ratio={self.trunk_ratio}")
            self.tdsm = np.asarray(self.cdsm * self.trunk_ratio, dtype=np.float32)
            self.tdsm_relative = self.cdsm_relative

        # Use DEM as base if available, otherwise DSM (now absolute after step 1)
        base = self.dem if self.dem is not None else self.dsm

        # Step 3: Convert CDSM from relative to absolute
        # Sub-threshold vegetation is set to NaN (absent), NOT to DEM height.
        # Setting it to DEM would make the shadow caster treat bare ground as
        # "vegetation at ground level," producing false vegetation shadows on
        # every steep slope.
        if self.cdsm_relative and self.cdsm is not None:
            cdsm_rel = np.where(np.isnan(self.cdsm), zero32, self.cdsm)
            cdsm_abs = np.where(~np.isnan(base), base + cdsm_rel, nan32)
            cdsm_abs = np.where(cdsm_abs - base < threshold, nan32, cdsm_abs)
            self.cdsm = np.asarray(cdsm_abs, dtype=np.float32)
            self.cdsm_relative = False
            logger.info(f"Converted relative CDSM to absolute (base: {'DEM' if self.dem is not None else 'DSM'})")

        # Step 4: Convert TDSM from relative to absolute
        if self.tdsm_relative and self.tdsm is not None:
            tdsm_rel = np.where(np.isnan(self.tdsm), zero32, self.tdsm)
            tdsm_abs = np.where(~np.isnan(base), base + tdsm_rel, nan32)
            tdsm_abs = np.where(tdsm_abs - base < threshold, nan32, tdsm_abs)
            self.tdsm = np.asarray(tdsm_abs, dtype=np.float32)
            self.tdsm_relative = False
            logger.info(f"Converted relative TDSM to absolute (base: {'DEM' if self.dem is not None else 'DSM'})")

        # Step 5: NaN out CDSM where canopy is below the DSM surface
        # Canopy below the DSM is physically impossible — it means the
        # vegetation layer sits inside a building or underground.  Mark as
        # absent (NaN) so the shadow caster skips these pixels entirely.
        # NOTE: Only CDSM is checked against DSM.  TDSM (trunk height) is
        # naturally below the canopy top, and the DSM already includes the
        # canopy, so trunk < DSM is expected at every tree pixel.  Clearing
        # TDSM here would destroy all trunk data and break pergola shadows.
        if self.cdsm is not None:
            below = self.cdsm < self.dsm
            if np.any(below):
                n = int(below.sum())
                self.cdsm[below] = np.float32(np.nan)
                # Also clear TDSM at the same pixels — if canopy is inside
                # a building, the trunk is too.
                if self.tdsm is not None:
                    self.tdsm[below] = np.float32(np.nan)
                logger.info(f"Cleared {n} vegetation pixels below DSM (canopy was underground)")

        self._preprocessed = True

    def compute_svf(self) -> None:
        """
        Compute SVF and shadow matrices, storing them on this instance.

        Only needed when constructing SurfaceData manually.
        :meth:`prepare` calls this automatically.

        Example::

            surface = SurfaceData(dsm=dsm, pixel_size=1.0)
            surface.preprocess()
            surface.compute_svf()
            result = calculate(surface, weather)
        """
        if self.svf is not None:
            return  # Already computed

        use_veg = self.cdsm is not None
        dsm_f32 = np.asarray(self.dsm, dtype=np.float32)

        if use_veg:
            assert self.cdsm is not None  # Type narrowing for type checker
            cdsm_f32 = np.asarray(self.cdsm, dtype=np.float32)
            if self.tdsm is not None:
                tdsm_f32 = np.asarray(self.tdsm, dtype=np.float32)
            else:
                tdsm_f32 = np.asarray(self.cdsm * self.trunk_ratio, dtype=np.float32)
        else:
            cdsm_f32 = np.zeros_like(dsm_f32)
            tdsm_f32 = np.zeros_like(dsm_f32)

        max_height = _max_shadow_height(dsm_f32, cdsm_f32 if use_veg else None, use_veg=use_veg)

        logger.info("Computing Sky View Factor...")
        from .. import _ensure_gpu_initialized

        _ensure_gpu_initialized()
        svf_result = skyview.calculate_svf(
            dsm_f32,
            cdsm_f32,
            tdsm_f32,
            self.pixel_size,
            use_veg,
            max_height,
            2,  # patch_option (153 patches)
            3.0,  # min_sun_elev_deg
            None,  # progress callback
        )

        self.svf = SvfArrays.from_rust_result(svf_result, use_veg=use_veg)

        # Store shadow matrices for anisotropic sky model
        # Shadow matrices are bitpacked uint8 from Rust
        self.shadow_matrices = ShadowArrays(
            _shmat_u8=np.array(svf_result.bldg_sh_matrix),
            _vegshmat_u8=np.array(svf_result.veg_sh_matrix),
            _vbshmat_u8=np.array(svf_result.veg_blocks_bldg_sh_matrix),
            _n_patches=153,  # patch_option=2
        )

        logger.info("  SVF computed successfully")

    @property
    def max_height(self) -> float:
        """Auto-compute maximum height difference for shadow buffer calculation.

        Considers both DSM (buildings) and CDSM (vegetation) since both cast shadows.
        Returns max elevation minus ground level.

        This property is conservative by design for shadow buffer sizing:
        CDSM is included whenever present, independent of current per-call
        vegetation switches.
        """
        if self.dsm.size == 0 or not np.isfinite(self.dsm).any():
            return 0.0

        dsm_max = float(np.nanmax(self.dsm))
        ground_min = float(np.nanmin(self.dsm))

        # Also consider vegetation if present (CDSM may be taller than buildings)
        if self.cdsm is not None and self.cdsm.size > 0 and np.isfinite(self.cdsm).any():
            cdsm_max = float(np.nanmax(self.cdsm))
            # After preprocessing, CDSM contains absolute elevations
            # Use the higher of DSM or CDSM
            max_elevation = max(dsm_max, cdsm_max)
        else:
            max_elevation = dsm_max

        height = max_elevation - ground_min
        if not np.isfinite(height) or height <= 0:
            return 0.0
        return height

    @property
    def shape(self) -> tuple[int, int]:
        """Return DSM shape (rows, cols)."""
        rows, cols = self.dsm.shape
        return (rows, cols)

    @property
    def geotransform(self) -> list[float] | None:
        """Return the raster geotransform, or None if not set."""
        return self._geotransform

    @property
    def crs(self) -> str | None:
        """Return CRS as WKT string, or None if not set."""
        return self._crs_wkt

    @property
    def valid_mask(self) -> NDArray[np.bool_] | None:
        """Return computed valid mask, or None if not yet computed."""
        return self._valid_mask

    def fill_nan(self, tolerance: float = 0.1) -> None:
        """Fill NaN in surface layers using DEM as ground reference.

        NaN in DSM/CDSM/TDSM means "no data, assume ground level."
        After filling, values within *tolerance* of ground are clamped
        to exactly the ground value to avoid shadow/SVF noise from
        resampling jitter.

        Fill rules:
            - DSM NaN  → DEM value  (if DEM provided, else left as NaN)
            - CDSM NaN → base value (DEM if available, else DSM)
            - TDSM NaN → base value (DEM if available, else DSM)
            - DEM NaN  → not filled (DEM is the ground-truth baseline)

        Works identically for relative and absolute height conventions.

        Args:
            tolerance: Height difference (m) below which a surface pixel
                is considered "at ground" and clamped. Default 0.1 m.
        """
        if self._nan_filled:
            return

        tol = np.float32(tolerance)

        # DSM: fill with DEM where available
        if self.dem is not None:
            dsm_nan = np.isnan(self.dsm)
            if np.any(dsm_nan):
                n = int(dsm_nan.sum())
                self.dsm = np.asarray(np.where(dsm_nan, self.dem, self.dsm), dtype=np.float32)
                logger.info(f"  Filled {n} NaN DSM pixels with DEM")

        base = self.dem if self.dem is not None else self.dsm
        base_label = "DEM" if self.dem is not None else "DSM"

        # CDSM: fill NaN with base, clamp near-ground noise
        if self.cdsm is not None:
            cdsm_nan = np.isnan(self.cdsm)
            if np.any(cdsm_nan):
                n = int(cdsm_nan.sum())
                self.cdsm = np.asarray(np.where(cdsm_nan, base, self.cdsm), dtype=np.float32)
                logger.info(f"  Filled {n} NaN CDSM pixels with {base_label}")
            near_ground = np.abs(self.cdsm - base) < tol
            if np.any(near_ground):
                self.cdsm = np.asarray(np.where(near_ground, base, self.cdsm), dtype=np.float32)

        # TDSM: same treatment as CDSM
        if self.tdsm is not None:
            tdsm_nan = np.isnan(self.tdsm)
            if np.any(tdsm_nan):
                n = int(tdsm_nan.sum())
                self.tdsm = np.asarray(np.where(tdsm_nan, base, self.tdsm), dtype=np.float32)
                logger.info(f"  Filled {n} NaN TDSM pixels with {base_label}")
            near_ground = np.abs(self.tdsm - base) < tol
            if np.any(near_ground):
                self.tdsm = np.asarray(np.where(near_ground, base, self.tdsm), dtype=np.float32)

        self._nan_filled = True

    def compute_valid_mask(self) -> NDArray[np.bool_]:
        """Compute combined valid mask: True where ALL ground-reference layers have finite data.

        A pixel is valid only if DSM (and DEM/walls if provided) have finite values.
        CDSM/TDSM are excluded — NaN vegetation means "at ground", not "invalid pixel".
        Call fill_nan() before this to fill vegetation NaN with ground values.

        Returns:
            Boolean array with same shape as DSM. True = valid pixel.
        """
        valid = np.isfinite(self.dsm)
        for arr in [self.dem, self.wall_height, self.wall_aspect]:
            if arr is not None:
                valid &= np.isfinite(arr)
        if self.land_cover is not None:
            valid &= self.land_cover != 255
        self._valid_mask = valid
        n_invalid = int(np.sum(~valid))
        if n_invalid > 0:
            pct = 100.0 * n_invalid / valid.size
            logger.info(f"  Valid mask: {n_invalid} invalid pixels ({pct:.1f}%)")
        else:
            logger.info("  Valid mask: all pixels valid")
        return valid

    def apply_valid_mask(self) -> None:
        """Set NaN in ALL layers where ANY layer has nodata.

        Ensures consistent nodata across all surface arrays.
        Must call compute_valid_mask() first (or it will be called automatically).
        """
        if self._valid_mask is None:
            self.compute_valid_mask()
        assert self._valid_mask is not None  # set by compute_valid_mask
        invalid = ~self._valid_mask
        if not np.any(invalid):
            return
        self.dsm[invalid] = np.nan
        for attr in ("cdsm", "dem", "tdsm", "wall_height", "wall_aspect", "albedo", "emissivity"):
            arr = getattr(self, attr)
            if arr is not None:
                arr[invalid] = np.nan
        if self.land_cover is not None:
            self.land_cover[invalid] = 255

    def crop_to_valid_bbox(self) -> tuple[int, int, int, int]:
        """Crop all arrays to minimum bounding box of valid pixels.

        Eliminates edge NaN bands to reduce wasted computation.
        Updates geotransform to reflect the new origin.

        Returns:
            (row_start, row_end, col_start, col_end) of the crop window.
        """
        if self._valid_mask is None:
            self.compute_valid_mask()
        assert self._valid_mask is not None  # set by compute_valid_mask
        rows_any = np.any(self._valid_mask, axis=1)
        cols_any = np.any(self._valid_mask, axis=0)
        if not np.any(rows_any):
            logger.warning("  No valid pixels found — cannot crop")
            return (0, self.dsm.shape[0], 0, self.dsm.shape[1])
        r0 = int(np.argmax(rows_any))
        r1 = len(rows_any) - int(np.argmax(rows_any[::-1]))
        c0 = int(np.argmax(cols_any))
        c1 = len(cols_any) - int(np.argmax(cols_any[::-1]))

        if r0 == 0 and r1 == self.dsm.shape[0] and c0 == 0 and c1 == self.dsm.shape[1]:
            logger.info("  Crop: no trimming needed (valid bbox = full extent)")
            return (r0, r1, c0, c1)

        old_shape = self.dsm.shape
        self.dsm = self.dsm[r0:r1, c0:c1].copy()
        self._valid_mask = self._valid_mask[r0:r1, c0:c1].copy()
        for attr in ("cdsm", "dem", "tdsm", "wall_height", "wall_aspect", "albedo", "emissivity", "land_cover"):
            arr = getattr(self, attr)
            if arr is not None:
                setattr(self, attr, arr[r0:r1, c0:c1].copy())

        # Update geotransform to reflect new origin
        if self._geotransform is not None:
            gt = self._geotransform
            self._geotransform = [
                gt[0] + c0 * gt[1] + r0 * gt[2],  # new origin X
                gt[1],
                gt[2],
                gt[3] + c0 * gt[4] + r0 * gt[5],  # new origin Y
                gt[4],
                gt[5],
            ]

        # Crop SVF arrays if present
        if self.svf is not None:
            self.svf = self.svf.crop(r0, r1, c0, c1)
        if self.shadow_matrices is not None:
            self.shadow_matrices = self.shadow_matrices.crop(r0, r1, c0, c1)

        # Clear buffer pool (shape changed)
        self.clear_buffers()

        logger.info(f"  Cropped: {old_shape[1]}x{old_shape[0]} → {c1 - c0}x{r1 - r0} pixels")
        return (r0, r1, c0, c1)

    def save_cleaned(
        self,
        output_dir: str | Path,
        *,
        prepare_fingerprint: dict | None = None,
    ) -> None:
        """Save cleaned, aligned rasters to disk and write ``metadata.json``.

        Writes all present layers to ``output_dir/cleaned/`` as GeoTIFFs and
        a top-level ``output_dir/metadata.json`` so future calls to
        :meth:`load` and the :meth:`prepare` fast-path can find them.

        Args:
            output_dir: Parent directory. Rasters are saved under
                ``output_dir/cleaned/``; metadata lives one level up at
                ``output_dir/metadata.json``.
            prepare_fingerprint: Optional fingerprint dict from
                :func:`cache.compute_prepare_fingerprint`. When provided,
                embedded under the ``prepare_fingerprint`` key so the
                :meth:`prepare` fast-path can short-circuit on warm runs
                where inputs and kwargs are unchanged.
        """
        out = Path(output_dir) / "cleaned"
        out.mkdir(parents=True, exist_ok=True)
        gt = self._geotransform or [0, self.pixel_size, 0, 0, 0, -self.pixel_size]
        crs = self._crs_wkt or ""
        io.save_raster(str(out / "dsm.tif"), self.dsm, gt, crs)
        for name, arr in [
            ("cdsm", self.cdsm),
            ("dem", self.dem),
            ("tdsm", self.tdsm),
            ("wall_height", self.wall_height),
            ("wall_aspect", self.wall_aspect),
        ]:
            if arr is not None:
                io.save_raster(str(out / f"{name}.tif"), arr, gt, crs)
        if self.land_cover is not None:
            io.save_raster(str(out / "land_cover.tif"), self.land_cover.astype(np.float32), gt, crs)
        if self._valid_mask is not None:
            io.save_raster(str(out / "valid_mask.tif"), self._valid_mask.astype(np.float32), gt, crs)

        # Write top-level metadata.json used by SurfaceData.load() and the
        # prepare() fast-path. Keys mirror the QGIS plugin's own
        # metadata.json schema (see qgis_plugin/.../surface_preprocessing.py).
        metadata: dict = {
            "pixel_size": float(self.pixel_size),
            "geotransform": list(gt),
            "crs_wkt": crs,
            "shape": [int(self.dsm.shape[0]), int(self.dsm.shape[1])],
            "dsm_relative": False,  # always absolute after preprocess()
            "cdsm_relative": False,
            "tdsm_relative": False,
            "has_cdsm": self.cdsm is not None,
            "has_dem": self.dem is not None,
            "has_tdsm": self.tdsm is not None,
            "has_land_cover": self.land_cover is not None,
            "has_walls": self.wall_height is not None and self.wall_aspect is not None,
            "has_svf": self.svf is not None,
            "dem_quantization_m": float(self._dem_quantization_m),
            "smooth_quantized_dem": bool(self.smooth_quantized_dem),
            "dem_smooth_sigma": float(self.dem_smooth_sigma),
            "min_object_height": float(self.min_object_height),
            "trunk_ratio": float(self.trunk_ratio),
        }
        if prepare_fingerprint is not None:
            metadata["prepare_fingerprint"] = prepare_fingerprint

        meta_path = Path(output_dir) / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"  Cleaned rasters saved to {out}")

    def get_buffer_pool(self) -> BufferPool:
        """Get or create a buffer pool for this surface.

        The buffer pool provides pre-allocated numpy arrays that can be
        reused across timesteps during timeseries calculations. This
        reduces memory allocation overhead and GC pressure.

        Returns:
            BufferPool sized to this surface's grid dimensions.

        Example:
            pool = surface.get_buffer_pool()
            temp = pool.get_zeros("ani_lum")  # First call allocates
            temp = pool.get_zeros("ani_lum")  # Second call reuses same memory
        """
        if self._buffer_pool is None:
            self._buffer_pool = BufferPool(self.shape)
        return self._buffer_pool

    def clear_buffers(self) -> None:
        """Clear the buffer pool to free memory.

        Call this after completing a timeseries calculation to release
        the pre-allocated arrays.
        """
        if self._buffer_pool is not None:
            self._buffer_pool.clear()
            self._buffer_pool = None
        # Clear runtime compute caches tied to this surface.
        # These are lazily rebuilt on demand in computation.calculate_core_fused().
        self._cache.clear()
        self._gvf_geometry_cache = None

    # ── Typed views (see models/surface_views.py) ────────────────────────────
    # Group fields by concern without changing the underlying field layout.
    # Internal callers can opt in to these views for clarity; existing
    # ``surface.dsm`` / ``surface.svf`` / … access keeps working unchanged.

    @property
    def geometry(self) -> SurfaceGeometryView:
        """Read-only view of the user-provided geometry inputs."""
        return SurfaceGeometryView(_surface=self)

    @property
    def optical(self) -> OpticalPropertiesView:
        """Read-only view of the per-pixel optical inputs."""
        return OpticalPropertiesView(_surface=self)

    @property
    def auxiliary(self) -> PreprocessedAuxiliaryView:
        """Read-only view of library-derived auxiliary data (walls, SVF, …)."""
        return PreprocessedAuxiliaryView(_surface=self)

    def looks_like_relative_heights(self) -> bool:
        """
        Heuristic check if CDSM appears to contain relative heights.

        Returns True if max(CDSM) is much smaller than min(DSM), suggesting
        CDSM contains height-above-ground values rather than absolute elevations.

        This is used to warn users who may have forgotten to call preprocess().
        """
        if self.cdsm is None:
            return False

        cdsm_max = np.nanmax(self.cdsm)
        dsm_min = np.nanmin(self.dsm)

        # If CDSM max is much smaller than DSM min, it's likely relative heights
        # Typical case: DSM min ~100m elevation, CDSM max ~30m tree height
        # Exception: coastal areas where DSM min could be near 0
        if dsm_min > 10 and cdsm_max < dsm_min * 0.5:
            return True

        # Also check if CDSM values are typical vegetation heights (0-50m range)
        # while DSM has larger values
        return bool(cdsm_max < 60 and dsm_min > cdsm_max + 20)

    def _check_preprocessing_needed(self) -> None:
        """
        Warn if CDSM appears to need preprocessing but wasn't preprocessed.

        Called internally before calculations to alert users.
        """
        if self.cdsm is None:
            return

        if self.cdsm_relative and not self._preprocessed and self.looks_like_relative_heights():
            logger.warning(
                f"CDSM appears to contain relative vegetation heights "
                f"(max CDSM={np.nanmax(self.cdsm):.1f}m < min DSM={np.nanmin(self.dsm):.1f}m), "
                f"but preprocess() was not called. "
                f"Call surface.preprocess() to convert to absolute heights, "
                f"or set cdsm_relative=False if CDSM already contains absolute elevations."
            )

    def get_land_cover_properties(
        self,
        params: SimpleNamespace | None = None,
    ) -> tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """
        Derive surface properties from land cover grid.

        Args:
            params: Optional loaded parameters from JSON file (via load_params()).
                When provided, land cover properties are read from the params.
                When None, uses built-in defaults matching parametersforsolweig.json.

        Returns:
            Tuple of (albedo_grid, emissivity_grid, tgk_grid, tstart_grid, tmaxlst_grid).
            If land_cover is None, returns defaults.

        Land cover parameters from Lindberg et al. 2008, 2016 (parametersforsolweig.json):
            - TgK (Ts_deg): Temperature coefficient for surface heating
            - Tstart: Temperature offset at sunrise
            - TmaxLST: Hour of maximum local surface temperature
        """
        if self.land_cover is None:
            # Use provided grids or defaults
            alb = self.albedo if self.albedo is not None else np.full_like(self.dsm, 0.15)
            emis = self.emissivity if self.emissivity is not None else np.full_like(self.dsm, 0.95)
            tgk = np.full_like(self.dsm, 0.37)  # Default TgK (cobblestone)
            tstart = np.full_like(self.dsm, -3.41)  # Default Tstart (cobblestone)
            tmaxlst = np.full_like(self.dsm, 15.0)  # Default TmaxLST (cobblestone)
            return alb, emis, tgk, tstart, tmaxlst

        # If params provided, use the helper function to extract from JSON
        if params is not None:
            return get_lc_properties_from_params(self.land_cover, params, self.shape)

        # UMEP standard land cover properties from parametersforsolweig.json
        # ID: (albedo, emissivity, TgK, Tstart, TmaxLST)
        # Values must match the JSON parameters file for parity with runner
        lc_properties = {
            0: (0.20, 0.95, 0.37, -3.41, 15.0),  # Paved/cobblestone (Cobble_stone_2014a)
            1: (0.18, 0.95, 0.58, -9.78, 15.0),  # Dark asphalt (albedo from JSON)
            2: (0.18, 0.95, 0.58, -9.78, 15.0),  # Buildings/roofs (emissivity=0.95, albedo=0.18)
            3: (0.20, 0.95, 0.37, -3.41, 15.0),  # Undefined (use paved defaults)
            4: (0.20, 0.95, 0.37, -3.41, 15.0),  # Undefined (use paved defaults)
            5: (0.16, 0.94, 0.21, -3.38, 14.0),  # Grass (Grass_unmanaged) - albedo=0.16, emis=0.94
            6: (0.25, 0.94, 0.33, -3.01, 14.0),  # Bare soil - emis=0.94
            7: (0.05, 0.98, 0.00, 0.00, 12.0),  # Water - albedo=0.05
        }

        rows, cols = self.shape
        alb_grid = np.full((rows, cols), 0.15, dtype=np.float32)
        emis_grid = np.full((rows, cols), 0.95, dtype=np.float32)
        tgk_grid = np.full((rows, cols), 0.37, dtype=np.float32)
        tstart_grid = np.full((rows, cols), -3.41, dtype=np.float32)
        tmaxlst_grid = np.full((rows, cols), 15.0, dtype=np.float32)

        lc = self.land_cover
        for lc_id, (alb, emis, tgk, tstart, tmaxlst) in lc_properties.items():
            mask = lc == lc_id
            if np.any(mask):
                alb_grid[mask] = alb
                emis_grid[mask] = emis
                tgk_grid[mask] = tgk
                tstart_grid[mask] = tstart
                tmaxlst_grid[mask] = tmaxlst

        return alb_grid, emis_grid, tgk_grid, tstart_grid, tmaxlst_grid
