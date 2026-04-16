"""
Tiled processing for large rasters.

This module provides automatic and manual tiling for SOLWEIG calculations,
supporting both single-timestep and timeseries modes. Large rasters are
automatically divided into overlapping tiles with buffers sized to capture
shadows from the tallest buildings at low sun angles.

Timeseries mode preserves thermal state accumulation across tiles and timesteps,
ensuring physically accurate ground temperature modeling with thermal inertia.
"""

from __future__ import annotations

import math
import os

import numpy as np

from .constants import MIN_SUN_ELEVATION_DEG
from .models import PrecomputedData, SurfaceData, TileSpec
from .solweig_logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Constants
# =============================================================================

MIN_TILE_SIZE = 256  # Minimum tile size in pixels
_FALLBACK_MAX_TILE_SIZE = 2500  # Used when GPU + RAM detection both fail
MAX_BUFFER_M = 1000.0  # Default maximum buffer / shadow distance in meters

# Backward-compat alias (imported by tests)
MAX_TILE_SIZE = _FALLBACK_MAX_TILE_SIZE


# Resource estimation constants — overridable via environment variables so
# users with unusual hardware can tune without patching source code.
#
#   SOLWEIG_RAM_FRACTION    — fraction of physical RAM for tile arrays (default 0.50)
#   SOLWEIG_GPU_HEADROOM    — fraction of GPU limit/budget to use (default 0.80)
#   SOLWEIG_GPU_BUDGET_BYTES — conservative GPU budget fallback when the Rust
#                              layer cannot determine real VRAM (default 3 GiB)
#   SOLWEIG_MAX_TILE_SIDE   — hard cap on tile side in pixels (default: unlimited)
#   SOLWEIG_MEMMAP_THRESHOLD — pixel count above which memmap is used (default 50M)
#
def _parse_env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        import warnings

        warnings.warn(f"Ignoring invalid {key}={raw!r}, using default {default}", stacklevel=2)
        return default


def _parse_env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        import warnings

        warnings.warn(f"Ignoring invalid {key}={raw!r}, using default {default}", stacklevel=2)
        return default


_RAM_FRACTION = _parse_env_float("SOLWEIG_RAM_FRACTION", 0.50)

# GPU memory constants — bytes per pixel for ALL wgpu buffers alive
# simultaneously during a given processing stage.  The Rust layer
# provides a resolved ``gpu_memory_budget`` (real VRAM when detectable,
# else a sane heuristic); Python divides that budget by these BPP values
# to determine tile sizes.
#
# Timestep context ("solweig"): shadow + GVF run together
#   Shadow (shadow_gpu.rs allocate_buffers):
#     17 f32 storage + 10× staging = 27 × 4 = 108, ~120 with overhead
#   GVF (gvf_gpu.rs ensure_buffers_locked, 18 azimuths):
#     lup/albshadow/sunwall (12) + blocking_distance 18×4 (72)
#     + facesh 18×4 (72) + outputs 10×4 (40) + staging 10×4 (40) = 236
#   Combined: 120 + 236 = 356, rounded to 360
_TIMESTEP_GPU_BPP = 360
#
# SVF context: shadow + SVF accumulation + bitpack
#   shadow (108) + svf_data 15× (60) + svf_staging (60)
#   + bitpack 3×20 output (60) + bitpack staging (60) = ~348, rounded to 384
_SVF_GPU_BPP = 384
#
# RAM bytes per pixel (Python-side peak allocation)
_TIMESTEP_RAM_BPP = 400  # benchmarked ~370 for full timestep pipeline
_SVF_RAM_BPP = 150  # Rust SvfIntermediate + bitpack + memmap overhead

_GPU_HEADROOM = _parse_env_float("SOLWEIG_GPU_HEADROOM", 0.80)
_GPU_BUDGET_BYTES = _parse_env_int("SOLWEIG_GPU_BUDGET_BYTES", 3 * 1024**3)
_MAX_AUTO_TILE_WORKERS = min(os.cpu_count() or 4, 16)  # Scale with CPU count, cap at 16

# When total pixels exceed this threshold, use memory-mapped files for
# GridAccumulator and ThermalState arrays so the OS can page to disk.
# 50M pixels ≈ 200 MB per float32 array, ~4 GB for the full accumulator set.
_MEMMAP_PIXEL_THRESHOLD = _parse_env_int("SOLWEIG_MEMMAP_THRESHOLD", 50_000_000)

# Cache for computed tile limits (populated once per context on first call)
_cached_max_tile_side: dict[str, int] = {}


# =============================================================================
# Resource detection
# =============================================================================


def _get_total_ram_bytes() -> int | None:
    """
    Detect total physical RAM in bytes.

    Uses ``os.sysconf`` on POSIX (macOS/Linux) and ``ctypes`` on Windows.
    Returns ``None`` if detection fails.  No external dependencies.
    """
    import sys

    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys
        else:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                return pages * page_size
    except (OSError, ValueError, AttributeError):
        pass
    return None


def _get_available_ram_bytes() -> int | None:
    """Detect available (free + reclaimable) RAM in bytes.

    Unlike :func:`_get_total_ram_bytes`, this accounts for memory already
    consumed by the OS, loaded raster data, SVF caches, etc.  Using
    available RAM for tile sizing means tiles automatically shrink when the
    process has a large memory footprint.
    """
    import subprocess
    import sys

    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullAvailPhys
        elif sys.platform == "darwin":
            # macOS: vm_stat reports free + inactive (reclaimable) pages
            result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                free = inactive = 0
                for line in result.stdout.splitlines():
                    if "Pages free" in line:
                        free = int(line.split()[-1].rstrip("."))
                    elif "Pages inactive" in line:
                        inactive = int(line.split()[-1].rstrip("."))
                page_size = os.sysconf("SC_PAGE_SIZE")
                return (free + inactive) * page_size
        else:
            # Linux: /proc/meminfo MemAvailable is the best estimate
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024  # kB -> bytes
    except (OSError, ValueError, AttributeError, subprocess.TimeoutExpired):
        pass
    return None


def _get_gpu_memory_budget() -> int | None:
    """
    Resolved GPU memory budget in bytes, or ``None``.

    Prefers the platform-specific ``gpu_memory_budget`` from the Rust layer
    (DXGI on Windows, sysfs on Linux, Metal passthrough).  When the Rust
    layer cannot determine real VRAM it returns no ``gpu_memory_budget``
    key, and we fall back to the conservative ``_GPU_BUDGET_BYTES``.
    """
    from . import get_gpu_limits

    limits = get_gpu_limits()
    if limits is None:
        return None
    budget = limits.get("gpu_memory_budget")
    if budget is not None:
        return int(budget)
    # Rust couldn't determine real VRAM — use conservative default
    return _GPU_BUDGET_BYTES


def _get_usable_ram() -> int | None:
    """Usable RAM in bytes for tile arrays (available or total × fraction)."""
    ram = _get_available_ram_bytes()
    if ram is None:
        ram = _get_total_ram_bytes()
    if ram is not None:
        return int(ram * _RAM_FRACTION)
    return None


def compute_max_tile_pixels(*, context: str = "solweig") -> int:
    """
    Compute the maximum number of pixels that fit in a single tile,
    based on GPU memory budget and system RAM.

    Args:
        context: ``"solweig"`` for timestep tiling (shadow + GVF), or
            ``"svf"`` for SVF preprocessing. Affects the bytes-per-pixel
            estimate.

    Returns:
        Maximum pixel count for a tile (rows * cols).
    """
    if context == "svf":
        gpu_bpp = _SVF_GPU_BPP
        ram_bpp = _SVF_RAM_BPP
    else:
        gpu_bpp = _TIMESTEP_GPU_BPP
        ram_bpp = _TIMESTEP_RAM_BPP

    # GPU constraint: budget / bpp
    gpu_max_pixels = None
    budget = _get_gpu_memory_budget()
    if budget is not None:
        gpu_max_pixels = int(budget * _GPU_HEADROOM) // gpu_bpp

    # RAM constraint
    ram_max_pixels = None
    usable_ram = _get_usable_ram()
    if usable_ram is not None:
        ram_max_pixels = usable_ram // ram_bpp

    # Tightest constraint wins
    candidates = [c for c in [gpu_max_pixels, ram_max_pixels] if c is not None]
    if candidates:
        return max(MIN_TILE_SIZE**2, min(candidates))

    # Fallback: no detection succeeded
    return _FALLBACK_MAX_TILE_SIZE**2


def compute_max_tile_side(*, context: str = "solweig") -> int:
    """
    Compute the maximum tile side length (square tiles) from resource limits.

    The result is cached per *context* for the lifetime of the process.

    Returns:
        Maximum tile side in pixels (at least ``MIN_TILE_SIZE``).
    """
    if context in _cached_max_tile_side:
        return _cached_max_tile_side[context]

    max_pixels = compute_max_tile_pixels(context=context)
    side = max(MIN_TILE_SIZE, int(math.isqrt(max_pixels)))

    # User-configurable hard cap — allows limiting tile size without
    # patching source code (addresses issue #11 usability feedback).
    env_cap = os.environ.get("SOLWEIG_MAX_TILE_SIDE")
    if env_cap is not None:
        parsed = _parse_env_int("SOLWEIG_MAX_TILE_SIDE", side)
        cap = max(MIN_TILE_SIZE, parsed)
        if cap < side:
            side = cap

    # Log once per context
    budget = _get_gpu_memory_budget()
    if budget is not None:
        from . import get_gpu_limits

        limits = get_gpu_limits()
        assert limits is not None  # budget came from limits
        raw_buf = int(limits.get("max_buffer_size", 0))
        extra = f" (raw max_buffer_size={raw_buf:,})" if raw_buf != budget else ""
        gpu_str = f"{budget:,} bytes{extra}"
    else:
        gpu_str = "N/A"
    avail_ram = _get_available_ram_bytes()
    total_ram = _get_total_ram_bytes()
    if avail_ram is not None and total_ram is not None:
        ram_str = f"{avail_ram:,} available of {total_ram:,} total"
    elif avail_ram is not None:
        ram_str = f"{avail_ram:,} bytes (available)"
    elif total_ram is not None:
        ram_str = f"{total_ram:,} bytes (total; available detection failed)"
    else:
        ram_str = "N/A"
    cap_str = f", SOLWEIG_MAX_TILE_SIDE={env_cap}" if env_cap else ""
    logger.info(
        f"Resource-aware tile sizing (context={context}): "
        f"GPU budget={gpu_str}, RAM={ram_str}, "
        f"max_tile_side={side} px{cap_str}"
    )

    _cached_max_tile_side[context] = side
    return side


# =============================================================================
# Helper Functions
# =============================================================================


def _calculate_auto_tile_size(rows: int, cols: int, buffer_pixels: int = 0) -> int:
    """
    Calculate optimal core tile size based on raster dimensions and resources.

    The resource-derived budget from :func:`compute_max_tile_side` is the
    *full* tile side (core plus both sides of the overlap buffer). The core
    must leave room for ``2 × buffer_pixels`` of overlap inside that budget,
    so we subtract them here. ``buffer_pixels`` defaults to 0 for callers
    that don't use overlap (e.g. SVF-less paths); the standard timeseries
    caller should pass the real buffer so the returned core fits
    ``validate_tile_size()`` without clamping.

    Args:
        rows: Total raster rows (currently unused; reserved for future
            aspect-ratio-aware sizing).
        cols: Total raster cols (currently unused).
        buffer_pixels: Overlap buffer that will wrap each core tile side.

    Returns:
        Core tile size in pixels, at least ``MIN_TILE_SIZE``.
    """
    max_full = compute_max_tile_side(context="solweig")
    return max(MIN_TILE_SIZE, max_full - 2 * buffer_pixels)


def _extract_tile_surface(
    surface: SurfaceData,
    tile: TileSpec,
    pixel_size: float,
    precomputed: PrecomputedData | None = None,
) -> SurfaceData:
    """
    Extract tile slice from full surface, reusing precomputed SVF when available.

    Creates a new SurfaceData with sliced arrays (DSM, CDSM, etc.).
    If the global surface has precomputed SVF (via prepare() or compute_svf()),
    the SVF is sliced to the tile bounds — avoiding expensive per-tile
    recomputation. If surface.svf is absent but precomputed.svf is provided,
    that precomputed SVF is sliced instead. When neither source exists, SVF
    remains unset and callers must fail fast before computation.

    Args:
        surface: Full raster surface data.
        tile: Tile specification with slice bounds.
        pixel_size: Pixel size in meters.
        precomputed: Optional precomputed data containing SVF.

    Returns:
        SurfaceData for this tile.
    """
    read_slice = tile.read_slice

    # Use views (no .copy()) for read-only surface arrays. These are never mutated
    # during computation — Rust receives PyReadonlyArray2 and Python never writes.
    # Views avoid redundant allocation, especially for overlapping tile buffers.
    tile_dsm = surface.dsm[read_slice]
    tile_cdsm = surface.cdsm[read_slice] if surface.cdsm is not None else None
    tile_tdsm = surface.tdsm[read_slice] if surface.tdsm is not None else None
    tile_dem = surface.dem[read_slice] if surface.dem is not None else None
    tile_lc = surface.land_cover[read_slice] if surface.land_cover is not None else None
    tile_albedo = surface.albedo[read_slice] if surface.albedo is not None else None
    tile_emis = surface.emissivity[read_slice] if surface.emissivity is not None else None
    tile_wall_ht = surface.wall_height[read_slice] if surface.wall_height is not None else None
    tile_wall_asp = surface.wall_aspect[read_slice] if surface.wall_aspect is not None else None

    # Slice precomputed SVF if available (avoids per-tile recomputation)
    tile_svf = None
    if surface.svf is not None:
        tile_svf = surface.svf.crop(
            tile.row_start_full,
            tile.row_end_full,
            tile.col_start_full,
            tile.col_end_full,
        )
    elif precomputed is not None and precomputed.svf is not None:
        tile_svf = precomputed.svf.crop(
            tile.row_start_full,
            tile.row_end_full,
            tile.col_start_full,
            tile.col_end_full,
        )

    # Slice shadow matrices if available (required for anisotropic sky in tiled mode)
    tile_shadow_matrices = None
    if surface.shadow_matrices is not None:
        tile_shadow_matrices = surface.shadow_matrices.crop(
            tile.row_start_full,
            tile.row_end_full,
            tile.col_start_full,
            tile.col_end_full,
        )

    tile_surface = SurfaceData(
        dsm=tile_dsm,
        cdsm=tile_cdsm,
        tdsm=tile_tdsm,
        dem=tile_dem,
        land_cover=tile_lc,
        albedo=tile_albedo,
        emissivity=tile_emis,
        wall_height=tile_wall_ht,
        wall_aspect=tile_wall_asp,
        pixel_size=pixel_size,
        svf=tile_svf,
        shadow_matrices=tile_shadow_matrices,
    )
    # Inherit NaN-filled state from parent — prevents fill_nan() from
    # re-filling intentional CDSM/TDSM NaN markers (no-vegetation pixels)
    # with DEM values, which would create phantom vegetation.
    if surface._nan_filled:
        tile_surface._nan_filled = True
    return tile_surface


def _slice_tile_precomputed(
    precomputed: PrecomputedData | None,
    tile: TileSpec,
) -> PrecomputedData | None:
    """
    Slice walls and shadow matrices from precomputed data for a tile.

    SVF is handled via surface.svf (sliced in _extract_tile_surface).
    Shadow matrices are spatially cropped to the tile bounds for
    anisotropic sky support in tiled mode.

    Args:
        precomputed: Full raster precomputed data (or None).
        tile: Tile specification with slice bounds.

    Returns:
        PrecomputedData with sliced walls and shadow matrices, or None.
    """
    if precomputed is None:
        return None

    read_slice = tile.read_slice

    tile_wall_ht = None
    tile_wall_asp = None
    tile_shadow_matrices = None

    if precomputed.wall_height is not None:
        tile_wall_ht = precomputed.wall_height[read_slice].copy()
    if precomputed.wall_aspect is not None:
        tile_wall_asp = precomputed.wall_aspect[read_slice].copy()
    if precomputed.shadow_matrices is not None:
        tile_shadow_matrices = precomputed.shadow_matrices.crop(
            tile.row_start_full,
            tile.row_end_full,
            tile.col_start_full,
            tile.col_end_full,
        )

    if tile_wall_ht is None and tile_wall_asp is None and tile_shadow_matrices is None:
        return None

    return PrecomputedData(
        wall_height=tile_wall_ht,
        wall_aspect=tile_wall_asp,
        svf=None,
        shadow_matrices=tile_shadow_matrices,
    )


# =============================================================================
# Public Functions
# =============================================================================


def calculate_buffer_distance(
    max_height: float,
    min_sun_elev_deg: float = MIN_SUN_ELEVATION_DEG,
    max_shadow_distance_m: float = MAX_BUFFER_M,
) -> float:
    """
    Calculate required buffer distance for tiled processing based on max building height.

    The buffer must be large enough to capture shadows cast by the tallest buildings
    at the lowest sun elevation angle.

    Formula: buffer = min(max_height / tan(min_sun_elevation), max_shadow_distance_m)

    Args:
        max_height: Maximum building/DSM height in meters.
        min_sun_elev_deg: Minimum sun elevation angle in degrees. Default 3.0.
        max_shadow_distance_m: Maximum buffer distance in meters. Default 1000.0.

    Returns:
        Buffer distance in meters, capped at max_shadow_distance_m.

    Example:
        >>> calculate_buffer_distance(30.0)  # 30m building
        572.9  # 30m / tan(3deg)
        >>> calculate_buffer_distance(10.0)  # 10m building
        190.8  # 10m / tan(3deg)
    """
    if max_height <= 0:
        return 0.0

    tan_elev = np.tan(np.radians(min_sun_elev_deg))
    if tan_elev <= 0:
        return max_shadow_distance_m

    buffer = max_height / tan_elev
    return min(buffer, max_shadow_distance_m)


def validate_tile_size(
    tile_size: int,
    buffer_pixels: int,
    pixel_size: float,
    context: str = "solweig",
) -> tuple[int, str | None]:
    """
    Validate and adjust core tile size for tiled processing.

    ``tile_size`` is the **core** tile side (the region whose results are
    kept). The actual tile in memory is ``core + 2 * buffer_pixels``.
    This function ensures the full tile fits within resource-derived limits.

    Args:
        tile_size: Requested core tile size in pixels.
        buffer_pixels: Overlap buffer size in pixels.
        pixel_size: Pixel size in meters.
        context: Resource context for limit detection. Use ``"svf"`` for
            SVF preprocessing tiles and ``"solweig"`` for timestep tiles.

    Returns:
        Tuple of (adjusted_core_size, warning_message or None).

    Constraints:
        - core >= MIN_TILE_SIZE (preferred)
        - core >= 1 when large overlap leaves less than MIN_TILE_SIZE
        - core + 2 * buffer_pixels <= resource-derived maximum
    """
    max_full = compute_max_tile_side(context=context)
    warning = None
    core = tile_size

    # Enforce maximum: full tile (core + 2*buffer) must fit resource limit
    max_core = max_full - 2 * buffer_pixels
    if max_core < 1:
        warning = f"Buffer {buffer_pixels}px too large for resource limit ({max_full}px). Using minimum feasible core=1"
        return 1, warning

    # Enforce minimum core size (prefer MIN_TILE_SIZE, but allow smaller when overlap is large)
    min_core = MIN_TILE_SIZE if max_core >= MIN_TILE_SIZE else 1
    if core < min_core:
        warning = f"Tile core size {tile_size} below minimum, using {min_core}"
        core = min_core

    if core > max_core:
        core = max_core
        warning = (
            f"Tile core {tile_size} + 2x{buffer_pixels}px buffer exceeds resource limit "
            f"({max_full}px). Using core={core}"
        )

    return core, warning


def generate_tiles(
    rows: int,
    cols: int,
    tile_size: int,
    overlap: int,
) -> list[TileSpec]:
    """
    Generate tile specifications with overlaps for tiled processing.

    Args:
        rows: Total number of rows in raster.
        cols: Total number of columns in raster.
        tile_size: Core tile size in pixels (without overlap).
        overlap: Overlap size in pixels.

    Returns:
        List of TileSpec objects covering the entire raster.
    """
    tiles = []
    n_tiles_row = int(np.ceil(rows / tile_size))
    n_tiles_col = int(np.ceil(cols / tile_size))

    for i in range(n_tiles_row):
        for j in range(n_tiles_col):
            # Core tile bounds
            row_start = i * tile_size
            row_end = min((i + 1) * tile_size, rows)
            col_start = j * tile_size
            col_end = min((j + 1) * tile_size, cols)

            # Full tile bounds with overlap (clamped to raster edges)
            row_start_full = max(0, row_start - overlap) if i > 0 else row_start
            row_end_full = min(rows, row_end + overlap) if row_end < rows else row_end
            col_start_full = max(0, col_start - overlap) if j > 0 else col_start
            col_end_full = min(cols, col_end + overlap) if col_end < cols else col_end

            # Actual overlaps after clamping to raster edges
            overlap_top = row_start - row_start_full
            overlap_bottom = row_end_full - row_end
            overlap_left = col_start - col_start_full
            overlap_right = col_end_full - col_end

            tiles.append(
                TileSpec(
                    row_start=row_start,
                    row_end=row_end,
                    col_start=col_start,
                    col_end=col_end,
                    row_start_full=row_start_full,
                    row_end_full=row_end_full,
                    col_start_full=col_start_full,
                    col_end_full=col_end_full,
                    overlap_top=overlap_top,
                    overlap_bottom=overlap_bottom,
                    overlap_left=overlap_left,
                    overlap_right=overlap_right,
                )
            )

    return tiles
