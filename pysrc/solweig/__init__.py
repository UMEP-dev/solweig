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
    Settings,
    SolweigResult,
    SurfaceData,
    ThermalState,
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
from .errors import (  # noqa: E402
    ComputationCancelled,
    ConfigurationError,
    GridShapeMismatch,
    InvalidSurfaceData,
    MissingPrecomputedData,
    SolweigError,
    WeatherDataError,
)

# Try to import Rust algorithms. The submodules are imported with underscore
# aliases so they don't leak into the top-level public surface; user code that
# really needs a specific Rust submodule should import from `solweig.rustalgos`
# directly.
try:
    from .rustalgos import GPU_ENABLED, RELEASE_BUILD
    from .rustalgos import gvf as _gvf  # noqa: F401
    from .rustalgos import pet as _pet  # noqa: F401
    from .rustalgos import pipeline as _pipeline
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
    _pipeline = None
    _shadowing = None
    _skyview = None
    _gvf = None
    _sky = None
    _vegetation = None
    _utci = None
    _pet = None


def _ensure_gpu_initialized() -> None:
    """Honour SOLWEIG_NO_GPU on first call. Idempotent.

    The Rust-side toggles (shadow / aniso / GVF) all default to enabled at
    process start, so no explicit enable is needed here. We only act when
    the user requested CPU-only via `SOLWEIG_NO_GPU=1`, in which case we
    flip every GPU path off exactly once. Subsequent calls are no-ops so a
    prior user-level `disable_gpu()` / `enable_gpu()` is preserved.

    Prior to b87 this also emitted `_shadowing.enable_gpu()` unconditionally,
    which silently re-enabled the GPU after a user-level `disable_gpu()`
    call when followed by any `is_gpu_available()` lookup. That re-enable
    has been removed.
    """
    global _gpu_initialized
    if _gpu_initialized:
        return
    _gpu_initialized = True
    if not GPU_ENABLED or _shadowing is None:
        return
    if _os.environ.get("SOLWEIG_NO_GPU"):
        disable_gpu()
        _logger.info("GPU disabled via SOLWEIG_NO_GPU environment variable")


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
    Disable GPU acceleration on every Rust path (shadows, anisotropic sky,
    GVF), falling back to CPU.

    Useful for debugging, CPU-only benchmarks, or when GPU results need to
    be compared against the CPU reference. The change takes effect
    immediately for subsequent calculations.

    Note: Prior to b87 this function only flipped the shadow path; the
    anisotropic sky and GVF paths kept running on GPU. It now toggles all
    three so a single call genuinely produces a CPU-only run.
    """
    if _shadowing is not None:
        with _contextlib.suppress(AttributeError):
            _shadowing.disable_gpu()
    if _pipeline is not None:
        with _contextlib.suppress(AttributeError):
            _pipeline.disable_aniso_gpu()
        with _contextlib.suppress(AttributeError):
            _pipeline.disable_gvf_gpu()


def enable_gpu() -> None:
    """
    Enable GPU acceleration on every Rust path (shadows, anisotropic sky,
    GVF).

    The counterpart to :func:`disable_gpu`. Each Rust pipeline starts
    enabled by default at import time, so the common use of this function
    is re-enabling after an explicit :func:`disable_gpu` call (for
    instance, between benchmark scenarios).

    Has no effect if GPU support wasn't compiled in or no GPU device is
    available — both :func:`is_gpu_available` and the per-path fallbacks
    continue to govern actual execution.
    """
    if _shadowing is not None:
        with _contextlib.suppress(AttributeError):
            _shadowing.enable_gpu()
    if _pipeline is not None:
        with _contextlib.suppress(AttributeError):
            _pipeline.enable_aniso_gpu()
        with _contextlib.suppress(AttributeError):
            _pipeline.enable_gvf_gpu()


def gpu_dispatch_count() -> int:
    """
    Cumulative count of successful GPU dispatches since process start (or
    since the last :func:`reset_gpu_metrics` call).

    Reads a thread-safe atomic counter incremented on the success branch
    of every GPU kernel call across the four GPU paths (shadows, SVF,
    anisotropic sky, GVF). Pair with :func:`gpu_fallback_count` to
    distinguish "GPU is disabled" (both zero after a run) from "GPU tried
    and fell back" (dispatch=0, fallback>0).

    Use in tests to assert "the GPU path actually ran":

    >>> import solweig
    >>> solweig.reset_gpu_metrics()
    >>> # ... run a calculation ...
    >>> assert solweig.gpu_dispatch_count() > 0, "GPU path never executed"
    """
    if _shadowing is None:
        return 0
    return int(_shadowing.gpu_dispatch_count())


def gpu_fallback_count() -> int:
    """
    Cumulative count of GPU→CPU fall-backs since process start (or since
    the last :func:`reset_gpu_metrics` call).

    A non-zero value during a "should be GPU" run indicates either a GPU
    device problem (driver crash, OOM, unsupported shape) or a per-dispatch
    failure mode worth investigating. Zero after a CPU-only run just means
    no fall-back happened because the GPU branch was never entered.

    See also: :func:`gpu_dispatch_count`, :func:`reset_gpu_metrics`.
    """
    if _shadowing is None:
        return 0
    return int(_shadowing.gpu_fallback_count())


def reset_gpu_metrics() -> None:
    """Reset :func:`gpu_dispatch_count` and :func:`gpu_fallback_count` to 0.

    Typical use in tests:

    >>> solweig.reset_gpu_metrics()
    >>> # ... run a calculation ...
    >>> dispatched = solweig.gpu_dispatch_count()
    >>> fellback = solweig.gpu_fallback_count()
    """
    if _shadowing is not None:
        with _contextlib.suppress(AttributeError):
            _shadowing.reset_gpu_metrics()


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
    "InvalidSurfaceData",
    "GridShapeMismatch",
    "MissingPrecomputedData",
    "WeatherDataError",
    "ConfigurationError",
    "ComputationCancelled",
    "calculate",
    "validate_inputs",
    "load_params",
    "load_physics",
    "load_materials",
    # Tiling utilities (calculate_buffer_distance + generate_tiles + TileSpec
    # are part of the documented public API; compute_max_tile_pixels is plugin
    # plumbing and lives in solweig.geospatial)
    "calculate_buffer_distance",
    "Settings",
    "TileSpec",
    "ThermalState",
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
    "enable_gpu",
    "gpu_dispatch_count",
    "gpu_fallback_count",
    "reset_gpu_metrics",
    "GPU_ENABLED",
    "RELEASE_BUILD",
]
# Geospatial helpers (extract_bounds, intersect_bounds, resample_to_grid,
# namespace_to_dict, pixel_size_tag, compute_max_tile_pixels, looks_like_relative,
# wallalgorithms) live in `solweig.geospatial` — there is no top-level
# re-export. The b85→b86 DeprecationWarning shim was removed in b87.
