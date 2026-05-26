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

import contextlib as _contextlib
import logging as _logging
import os as _os
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

_logger = _logging.getLogger(__name__)

# Version: single source of truth is pyproject.toml
try:
    __version__ = _version("solweig")
except _PackageNotFoundError:
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
from .errors import SolweigError  # noqa: E402

# ── Deprecated top-level geospatial helpers ──────────────────────────────────
# These were promoted to the top level in 0.1.0b84 so the QGIS plugin could
# stop reaching into internals. The cleaner home is `solweig.geospatial`
# (created 0.1.0b85). Top-level access is preserved here behind a
# DeprecationWarning so downstream code keeps working but is nudged toward
# the structured import path.
#
# Removal target: 0.1.0b88 (or first 0.2.x). When removing, delete the
# `_DEPRECATED_REEXPORTS` map, the `__getattr__` block below, and update
# the matching entries in `__all__`.
_DEPRECATED_REEXPORTS = {
    "extract_bounds": "solweig.geospatial",
    "intersect_bounds": "solweig.geospatial",
    "resample_to_grid": "solweig.geospatial",
    "namespace_to_dict": "solweig.geospatial",
    "pixel_size_tag": "solweig.geospatial",
    "compute_max_tile_pixels": "solweig.geospatial",
    "looks_like_relative": "solweig.geospatial",
    "wallalgorithms": "solweig.geospatial",
}


def __getattr__(name: str):  # noqa: N807 — PEP 562 module-level hook
    if name in _DEPRECATED_REEXPORTS:
        import warnings as _warnings

        target = _DEPRECATED_REEXPORTS[name]
        _warnings.warn(
            f"`solweig.{name}` is deprecated and will be removed in a future "
            f"release. Import from `{target}` instead: "
            f"`from {target} import {name}`.",
            DeprecationWarning,
            stacklevel=2,
        )
        from . import geospatial as _geo

        return getattr(_geo, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Try to import Rust algorithms. The submodules are imported with underscore
# aliases so they don't leak into the top-level public surface; user code that
# really needs a specific Rust submodule should import from `solweig.rustalgos`
# directly.
try:
    from .rustalgos import GPU_ENABLED, RELEASE_BUILD
    from .rustalgos import gvf as _gvf  # noqa: F401
    from .rustalgos import pet as _pet  # noqa: F401
    from .rustalgos import shadowing as _shadowing
    from .rustalgos import sky as _sky  # noqa: F401
    from .rustalgos import skyview as _skyview  # noqa: F401
    from .rustalgos import utci as _utci  # noqa: F401
    from .rustalgos import vegetation as _vegetation  # noqa: F401

    # Defer GPU initialization until first use (avoids import-time failures
    # on headless systems). Set SOLWEIG_NO_GPU=1 to disable entirely.
    _gpu_initialized = False

    if GPU_ENABLED and not _os.environ.get("SOLWEIG_NO_GPU"):
        _logger.debug("GPU support compiled in; will enable on first use")
    elif not GPU_ENABLED:
        _logger.debug("GPU support not compiled in this build")
    else:
        _logger.debug("GPU disabled via SOLWEIG_NO_GPU environment variable")

except ImportError as e:
    _logger.warning(f"Failed to import Rust algorithms: {e}")
    GPU_ENABLED = False
    RELEASE_BUILD = False
    _gpu_initialized = False
    _shadowing = None
    _skyview = None
    _gvf = None
    _sky = None
    _vegetation = None
    _utci = None
    _pet = None


def _ensure_gpu_initialized() -> None:
    """Lazily initialize GPU on first use."""
    global _gpu_initialized
    if _gpu_initialized:
        return
    _gpu_initialized = True
    if not GPU_ENABLED or _shadowing is None:
        return
    if _os.environ.get("SOLWEIG_NO_GPU"):
        return
    try:
        _shadowing.enable_gpu()
        _logger.info("GPU acceleration enabled")
    except Exception:
        _logger.warning("GPU initialization failed, falling back to CPU", exc_info=True)


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
    if _shadowing is None:
        return False
    try:
        return _shadowing.is_gpu_enabled()
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
    if _shadowing is not None:
        with _contextlib.suppress(AttributeError):
            _shadowing.disable_gpu()


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
    if not GPU_ENABLED or _shadowing is None:
        return None
    try:
        return _shadowing.gpu_limits()
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
    # Tiling utilities (calculate_buffer_distance + generate_tiles + TileSpec
    # are part of the documented public API; compute_max_tile_pixels is plugin
    # plumbing and lives in solweig.geospatial)
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
    # GPU utilities
    "is_gpu_available",
    "get_compute_backend",
    "get_gpu_limits",
    "disable_gpu",
    "GPU_ENABLED",
    "RELEASE_BUILD",
]
# NOTE: The following names are still reachable as `solweig.<name>` for
# backwards-compatibility but are NOT in __all__: extract_bounds,
# intersect_bounds, resample_to_grid, namespace_to_dict, pixel_size_tag,
# compute_max_tile_pixels, looks_like_relative, wallalgorithms. They emit a
# DeprecationWarning on access (see the `__getattr__` hook above). Import
# from `solweig.geospatial` instead.
