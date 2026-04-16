"""SOLWEIG - High-performance urban microclimate model.

A Python package with Rust-accelerated algorithms for computing mean radiant
temperature (Tmrt) and thermal comfort indices (UTCI, PET) in complex urban
environments.

Quick start::

    import solweig
    from datetime import datetime

    summary = solweig.calculate(
        surface=solweig.SurfaceData(dsm=my_dsm_array),
        weather=[solweig.Weather(datetime=datetime(2025, 7, 15, 12, 0), ta=25, rh=50, global_rad=800)],
        location=solweig.Location(latitude=57.7, longitude=12.0),
    )
    print(f"Tmrt: {summary.tmrt_mean.mean():.1f} C")

I/O helpers::

    # Load raster data
    dsm, transform, crs, nodata = solweig.io.load_raster("dsm.tif")

    # Generate wall heights and aspects
    solweig.walls.generate_wall_hts(dsm_path, bbox, out_dir)
"""

import contextlib
import logging
import os
from importlib.metadata import PackageNotFoundError, version

logger = logging.getLogger(__name__)

# Version: single source of truth is pyproject.toml
try:
    __version__ = version("solweig")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"  # Fallback for editable/source installs without metadata

# Import simplified API
# Import utility modules
from . import io, progress, walls  # noqa: E402
from .api import (  # noqa: E402
    HumanParams,
    Location,
    ModelConfig,
    PrecomputedData,
    SolweigResult,
    SurfaceData,
    TileSpec,
    Timeseries,
    TimeseriesSummary,
    Weather,
    calculate,
    # Tiling utilities
    calculate_buffer_distance,
    # Run metadata/provenance
    create_run_metadata,
    # I/O
    download_epw,
    generate_tiles,
    load_materials,
    load_params,
    load_physics,
    load_run_metadata,
    save_run_metadata,
    # Validation
    validate_inputs,
)
from .cache import pixel_size_tag  # noqa: E402
from .errors import SolweigError  # noqa: E402
from .models.surface import looks_like_relative  # noqa: E402
from .physics import wallalgorithms  # noqa: E402
from .tiling import compute_max_tile_pixels  # noqa: E402
from .utils import extract_bounds, intersect_bounds, namespace_to_dict, resample_to_grid  # noqa: E402

# Try to import Rust algorithms
try:
    from .rustalgos import GPU_ENABLED, RELEASE_BUILD, gvf, pet, shadowing, sky, skyview, utci, vegetation

    # Defer GPU initialization until first use (avoids import-time failures
    # on headless systems). Set SOLWEIG_NO_GPU=1 to disable entirely.
    _gpu_initialized = False

    if GPU_ENABLED and not os.environ.get("SOLWEIG_NO_GPU"):
        logger.debug("GPU support compiled in; will enable on first use")
    elif not GPU_ENABLED:
        logger.debug("GPU support not compiled in this build")
    else:
        logger.debug("GPU disabled via SOLWEIG_NO_GPU environment variable")

except ImportError as e:
    logger.warning(f"Failed to import Rust algorithms: {e}")
    GPU_ENABLED = False
    RELEASE_BUILD = False
    _gpu_initialized = False
    shadowing = None
    skyview = None
    gvf = None
    sky = None
    vegetation = None
    utci = None
    pet = None


def _ensure_gpu_initialized() -> None:
    """Lazily initialize GPU on first use."""
    global _gpu_initialized
    if _gpu_initialized:
        return
    _gpu_initialized = True
    if not GPU_ENABLED or shadowing is None:
        return
    if os.environ.get("SOLWEIG_NO_GPU"):
        return
    try:
        shadowing.enable_gpu()
        logger.info("GPU acceleration enabled")
    except Exception:
        logger.warning("GPU initialization failed, falling back to CPU", exc_info=True)


def is_gpu_available() -> bool:
    """
    Check if GPU acceleration is available at runtime.

    Returns True if:
    - GPU support was compiled into the Rust extension
    - A GPU device was successfully detected and initialized

    Use this to check GPU status before running compute-intensive operations.

    Returns:
        True if GPU acceleration is available, False otherwise.
    """
    _ensure_gpu_initialized()
    if not GPU_ENABLED:
        return False
    if shadowing is None:
        return False
    try:
        return shadowing.is_gpu_enabled()
    except (AttributeError, RuntimeError):
        return False


def get_compute_backend() -> str:
    """
    Get the current compute backend.

    Returns:
        "gpu" if GPU acceleration is available and enabled, "cpu" otherwise.
    """
    return "gpu" if is_gpu_available() else "cpu"


def disable_gpu() -> None:
    """
    Disable GPU acceleration, falling back to CPU.

    This can be useful for debugging or if GPU results differ from expected.
    The change takes effect immediately for subsequent calculations.
    """
    if shadowing is not None:
        with contextlib.suppress(AttributeError):
            shadowing.disable_gpu()


def get_gpu_limits() -> dict[str, int | str] | None:
    """
    Query GPU limits from the wgpu adapter.

    Returns a dict with keys:
      - ``max_buffer_size``: int — raw adapter-reported maximum buffer size in bytes
      - ``backend``: str — GPU backend name (``"Metal"``, ``"Vulkan"``, ``"Dx12"``, ``"Gl"``, etc.)
      - ``gpu_memory_budget``: int — resolved GPU memory budget in bytes
        (only present when real VRAM is detectable via DXGI/sysfs/Metal)

    Returns ``None`` if GPU is not available or not compiled in.
    Lazily initialises the GPU context on first call.
    """
    if not GPU_ENABLED or shadowing is None:
        return None
    try:
        return shadowing.gpu_limits()
    except (AttributeError, RuntimeError):
        return None


__all__ = [
    # Version
    "__version__",
    # Core API
    "SurfaceData",
    "PrecomputedData",
    "Location",
    "Weather",
    "HumanParams",
    "ModelConfig",
    "SolweigResult",
    "Timeseries",
    "TimeseriesSummary",
    "SolweigError",
    "calculate",
    "validate_inputs",
    "load_params",
    "load_physics",
    "load_materials",
    # Tiling utilities
    "calculate_buffer_distance",
    "TileSpec",
    "generate_tiles",
    # Run metadata/provenance
    "create_run_metadata",
    "save_run_metadata",
    "load_run_metadata",
    # I/O
    "download_epw",
    # Utility modules
    "io",
    "walls",
    "progress",
    "wallalgorithms",
    # Preprocessing / grid helpers (used by the QGIS plugin and external tools)
    "extract_bounds",
    "intersect_bounds",
    "resample_to_grid",
    "namespace_to_dict",
    "pixel_size_tag",
    "compute_max_tile_pixels",
    "looks_like_relative",
    # GPU utilities
    "is_gpu_available",
    "get_compute_backend",
    "get_gpu_limits",
    "disable_gpu",
    "GPU_ENABLED",
    "RELEASE_BUILD",
]
