"""Public SOLWEIG API.

This module re-exports all user-facing symbols and provides the top-level
:func:`calculate` and :func:`validate_inputs` entry points. Internal
machinery (sun position, radiation split, etc.) is handled automatically,
while SVF must be prepared ahead of time (for example via
``SurfaceData.prepare()`` or ``surface.compute_svf()``).

Example::

    import solweig
    from datetime import datetime

    summary = solweig.calculate(
        surface=solweig.SurfaceData(dsm=my_dsm_array),
        weather=[solweig.Weather(datetime=datetime(2025, 7, 15, 12, 0), ta=25.0, rh=50.0, global_rad=800.0)],
        location=solweig.Location(latitude=57.7, longitude=12.0),
        output_dir="output/",
    )
    print(f"Tmrt: {summary.tmrt_mean.mean():.1f} C")
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from .computation import calculate_core_fused
from .errors import (
    ConfigurationError,
    GridShapeMismatch,
    InvalidSurfaceData,
    MissingPrecomputedData,
    SolweigError,
    WeatherDataError,
)
from .io import download_epw

# Import from extracted modules
from .loaders import load_materials, load_params, load_physics, resolve_wall_params
from .metadata import create_run_metadata, load_run_metadata, save_run_metadata
from .models import (
    HumanParams,
    Location,
    ModelConfig,
    PrecomputedData,
    ShadowArrays,
    SolweigResult,
    SurfaceData,
    SvfArrays,
    ThermalState,
    TileSpec,
    Weather,
)
from .models.settings import Settings
from .solweig_logging import get_logger
from .summary import Timeseries, TimeseriesSummary
from .tiling import (
    calculate_buffer_distance,
    generate_tiles,
    validate_tile_size,
)
from .utils import dict_to_namespace, extract_bounds, intersect_bounds, namespace_to_dict, resample_to_grid

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


def validate_inputs(
    surface: SurfaceData,
    location: Location | None = None,
    weather: Weather | list[Weather] | None = None,
    use_anisotropic_sky: bool = False,
    precomputed: PrecomputedData | None = None,
) -> list[str]:
    """
    Validate inputs before calculation (preflight check).

    Call this before expensive operations to catch errors early.
    Raises exceptions for fatal errors, returns warnings for potential issues.

    Args:
        surface: Surface data to validate.
        location: Location to validate (optional).
        weather: Weather data to validate (optional, can be single or list).
        use_anisotropic_sky: Whether anisotropic sky will be used.
        precomputed: Precomputed data to validate.

    Returns:
        List of warning messages (empty if all valid).

    Raises:
        GridShapeMismatch: If surface grid shapes don't match DSM.
        MissingPrecomputedData: If required precomputed data is missing.
        WeatherDataError: If weather data is invalid.

    Example:
        try:
            warnings = solweig.validate_inputs(surface, location, weather)
            for w in warnings:
                print(f"Warning: {w}")
            summary = solweig.calculate(surface, weather, location, output_dir="out/")
        except solweig.GridShapeMismatch as e:
            print(f"Grid mismatch: {e.field} expected {e.expected}, got {e.got}")
        except solweig.MissingPrecomputedData as e:
            print(f"Missing data: {e}")
    """
    warnings = []
    geom = surface.geometry
    optical = surface.optical
    aux = surface.auxiliary
    dsm_shape = geom.shape

    # Check grid shapes match DSM
    grids_to_check = [
        ("cdsm", geom.cdsm),
        ("dem", geom.dem),
        ("tdsm", geom.tdsm),
        ("wall_height", aux.wall_height),
        ("wall_aspect", aux.wall_aspect),
        ("land_cover", optical.land_cover),
        ("albedo", optical.albedo),
        ("emissivity", optical.emissivity),
    ]
    for name, grid in grids_to_check:
        if grid is not None and grid.shape != dsm_shape:
            raise GridShapeMismatch(name, dsm_shape, grid.shape)

    # Check SVF arrays if present
    if aux.svf is not None:
        svf_grids = [
            ("svf.svf", aux.svf.svf),
            ("svf.svf_north", aux.svf.svf_north),
            ("svf.svf_east", aux.svf.svf_east),
            ("svf.svf_south", aux.svf.svf_south),
            ("svf.svf_west", aux.svf.svf_west),
        ]
        for name, grid in svf_grids:
            if grid is not None and grid.shape != dsm_shape:
                raise GridShapeMismatch(name, dsm_shape, grid.shape)

    # Check SVF is available (required for all calculations)
    if not aux.has_svf and (precomputed is None or precomputed.svf is None):
        raise MissingPrecomputedData(
            "Sky View Factor (SVF) data is required but not available.",
            "Call surface.compute_svf() before calculate(), or use SurfaceData.prepare() "
            "which computes SVF automatically.",
        )

    # Check anisotropic sky requirements
    if use_anisotropic_sky:
        has_shadow_matrices = (precomputed is not None and precomputed.shadow_matrices is not None) or (
            aux.shadow_matrices is not None
        )
        if not has_shadow_matrices:
            raise MissingPrecomputedData(
                "shadow_matrices required for anisotropic sky model",
                "Either set use_anisotropic_sky=False, or provide shadow matrices via "
                "precomputed=PrecomputedData(shadow_matrices=...) or surface.shadow_matrices",
            )

    # Check for potential issues (warnings, not errors)
    if geom.cdsm is not None and not surface._preprocessed and geom.cdsm_relative:
        warnings.append(
            "CDSM provided with cdsm_relative=True but preprocess() not called. "
            "Vegetation heights may be incorrect. Call surface.preprocess() first."
        )
    if geom.tdsm is not None and not surface._preprocessed and geom.tdsm_relative:
        warnings.append(
            "TDSM provided with tdsm_relative=True but preprocess() not called. "
            "Trunk heights may be incorrect. Call surface.preprocess() first."
        )

    # DSM height sanity checks
    dsm_max = float(np.nanmax(geom.dsm))
    dsm_min = float(np.nanmin(geom.dsm))
    height_range = dsm_max - dsm_min

    if height_range > 500:
        warnings.append(
            f"DSM height range is {height_range:.0f}m (max={dsm_max:.0f}m, min={dsm_min:.0f}m). "
            "This is typical for mountainous terrain. Shadow rays will use full terrain "
            "relief; consider increasing max_shadow_distance_m for wide valleys."
        )

    if geom.dem is None and dsm_min > 100:
        warnings.append(
            f"DSM minimum value is {dsm_min:.0f}m with no DEM provided. "
            "If this is above-sea-level elevation, provide a DEM so SOLWEIG can "
            "compute building heights correctly."
        )

    # Per-layer relative height mismatch detection
    for grid_name, grid, is_relative in [
        ("CDSM", geom.cdsm, geom.cdsm_relative),
        ("TDSM", geom.tdsm, geom.tdsm_relative),
    ]:
        if grid is not None and is_relative:
            nonzero = grid[grid > 0]
            if nonzero.size > 0:
                grid_min_nz = float(np.nanmin(nonzero))
                if grid_min_nz > 50:
                    flag = f"{grid_name.lower()}_relative"
                    warnings.append(
                        f"{grid_name} minimum non-zero value is {grid_min_nz:.0f}m with "
                        f"{flag}=True. Relative vegetation heights are typically "
                        f"0-50m. If it contains absolute elevations, set {flag}=False."
                    )

    if geom.cdsm is not None and not geom.cdsm_relative and surface.looks_like_relative_heights():
        cdsm_max = float(np.nanmax(geom.cdsm))
        warnings.append(
            f"CDSM values (max={cdsm_max:.1f}m) are much smaller than DSM "
            f"(min={dsm_min:.1f}m) with cdsm_relative=False. "
            "If CDSM contains height-above-ground, set cdsm_relative=True "
            "and call surface.preprocess()."
        )

    # Validate weather if provided
    if weather is not None:
        weather_list = weather if isinstance(weather, list) else [weather]
        for i, w in enumerate(weather_list):
            # Basic range checks (Weather.__post_init__ catches some, but we add more)
            if w.ta < -100 or w.ta > 60:
                warnings.append(
                    f"Weather[{i}].ta={w.ta}°C is outside typical range [-100, 60]. Verify this is correct."
                )
            if w.global_rad > 1400:
                warnings.append(
                    f"Weather[{i}].global_rad={w.global_rad} W/m² exceeds solar constant. Verify this is correct."
                )
            if w.ws == 0:
                warnings.append(
                    f"Weather[{i}].ws=0 m/s. UTCI is sensitive to wind speed near zero; "
                    "consider using a small positive value (e.g. 0.5 m/s) if calm conditions are not intended."
                )

    return warnings


def _calculate_single(
    surface: SurfaceData,
    location: Location,
    weather: Weather,
    config: ModelConfig | None = None,
    human: HumanParams | None = None,
    precomputed: PrecomputedData | None = None,
    use_anisotropic_sky: bool | None = None,
    conifer: bool = False,
    state: ThermalState | None = None,
    physics: SimpleNamespace | None = None,
    materials: SimpleNamespace | None = None,
    wall_material: str | None = None,
    max_shadow_distance_m: float | None = None,
    return_state_copy: bool = True,
    _requested_outputs: set[str] | None = None,
) -> SolweigResult:
    """Single-timestep Rust FFI call. Internal building block for calculate()."""
    from .models.settings import Settings

    # Track whether anisotropic mode was explicitly requested by direct API arg.
    # Config/default fallbacks intentionally do not trigger strict precondition
    # failures because we cannot distinguish "config default" from a deliberate
    # explicit request at runtime.
    anisotropic_requested_explicitly = use_anisotropic_sky is True

    # Resolve effective settings: per-call kwargs > ModelConfig > defaults.
    # See `pysrc/solweig/models/settings.py` for merge semantics.
    settings = Settings.resolve(
        config=config,
        use_anisotropic_sky=use_anisotropic_sky,
        conifer=conifer,
        wall_material=wall_material,
        max_shadow_distance_m=max_shadow_distance_m,
        human=human,
        physics=physics,
        materials=materials,
    ).with_loaded_defaults()

    # Debug log: which fields explicitly overrode the config?
    if config is not None:
        overrides: list[str] = []
        if use_anisotropic_sky is not None and use_anisotropic_sky != config.use_anisotropic_sky:
            overrides.append(f"use_anisotropic_sky={use_anisotropic_sky}")
        if human is not None and config.human is not None:
            overrides.append("human")
        if physics is not None and config.physics is not None:
            overrides.append("physics")
        if materials is not None and config.materials is not None:
            overrides.append("materials")
        if overrides:
            logger.debug(f"Explicit params override config: {', '.join(overrides)}")

    # Compute derived weather values (sun position, radiation split)
    if not weather._derived_computed:
        weather.compute_derived(location)

    # Fill NaN in surface layers (idempotent — skipped if already done)
    surface.fill_nan()

    # Explicit anisotropic requests must have shadow matrices available.
    if anisotropic_requested_explicitly and settings.use_anisotropic_sky:
        has_shadow_matrices = (precomputed is not None and precomputed.shadow_matrices is not None) or (
            surface.auxiliary.shadow_matrices is not None
        )
        if not has_shadow_matrices:
            raise MissingPrecomputedData(
                "shadow_matrices required for anisotropic sky model",
                "Either set use_anisotropic_sky=False, or provide shadow matrices via "
                "precomputed=PrecomputedData(shadow_matrices=...) or surface.shadow_matrices",
            )

    # Fused Rust pipeline — single FFI call per daytime timestep.
    # Both isotropic and anisotropic sky models are supported.
    return calculate_core_fused(
        surface=surface,
        location=location,
        weather=weather,
        human=settings.human,
        precomputed=precomputed,
        state=state,
        physics=settings.physics,
        materials=settings.materials,
        conifer=settings.conifer,
        wall_material=settings.wall_material,
        use_anisotropic_sky=settings.use_anisotropic_sky,
        max_shadow_distance_m=settings.max_shadow_distance_m,
        return_state_copy=return_state_copy,
        requested_outputs=_requested_outputs,
    )


def calculate(
    surface: SurfaceData,
    weather: Weather | list[Weather],
    location: Location | None = None,
    *,
    output_dir: str | Path,
    config: ModelConfig | None = None,
    human: HumanParams | None = None,
    precomputed: PrecomputedData | None = None,
    use_anisotropic_sky: bool | None = None,
    conifer: bool = False,
    physics: SimpleNamespace | None = None,
    materials: SimpleNamespace | None = None,
    wall_material: str | None = None,
    max_shadow_distance_m: float | None = None,
    tile_size: int | None = None,
    outputs: list[str] | None = None,
    heat_thresholds_day: list[float] | None = None,
    heat_thresholds_night: list[float] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> TimeseriesSummary:
    """
    Calculate mean radiant temperature (Tmrt).

    Single entry point for all SOLWEIG calculations. SVF and shadow matrices
    must already be on ``surface`` (via :meth:`SurfaceData.prepare`) or in
    ``precomputed``. The anisotropic sky model is on by default.

    Args:
        surface: Prepared surface data from :meth:`SurfaceData.prepare`.
        weather: One or more Weather objects. A single Weather is auto-wrapped.
        location: Geographic location. If None, extracted from surface CRS.
        config: Model settings. Explicit keyword args override config values.
        human: Human body parameters (posture, absorption, etc.).
        precomputed: Alternative source for SVF/shadow matrices (advanced).
        use_anisotropic_sky: Anisotropic (True) or uniform (False) sky
            diffuse radiation. Default True.
        conifer: Treat vegetation as evergreen (always leaf-on).
        physics: Physics parameters from load_physics().
        materials: Material properties from load_materials().
        wall_material: Wall material type ("brick", "concrete", "wood", "cobblestone").
        max_shadow_distance_m: Maximum shadow reach in metres (default 1000.0).
        tile_size: Core tile side in pixels for tiled processing. If None
            (default), auto-calculated from available resources. Minimum 256.
        output_dir: Working directory for all output. Summary grids are always
            saved to ``output_dir/summary/``. Per-timestep GeoTIFFs are saved
            when ``outputs`` is specified.
        outputs: Which per-timestep outputs to save as GeoTIFFs
            (e.g., ``["tmrt", "shadow"]``). If None, only summary grids are saved.
        heat_thresholds_day: Daytime UTCI thresholds for exceedance grids.
        heat_thresholds_night: Nighttime UTCI thresholds for exceedance grids.
        progress_callback: Called as progress_callback(current, total) per tile-timestep.

    Returns:
        TimeseriesSummary with per-pixel grids (mean/max/min Tmrt and UTCI,
        sun/shade hours, heat-stress exceedance).

    Notes:
        Modelled downwelling longwave (L↓) carries a known +18 to +55 W/m²
        positive bias inherited from the UMEP Jonsson et al. (2006) formulation.
        The bias is steady across hours and grows with reduced sky-view; see
        ``VALIDATION.md § Ldown overestimation``. K↓ at any single pixel is
        shadow-edge-sensitive and may spike on a single mis-aligned hour;
        spatially-averaged K↓ has much lower error.

    Example::

        surface = solweig.SurfaceData.prepare(dsm="dsm.tif", working_dir="cache/")
        weather = solweig.Weather.from_epw("weather.epw")

        summary = solweig.calculate(
            surface=surface,
            weather=weather,
            output_dir="output/",
        )
    """
    from .timeseries import _calculate_timeseries

    # Accept a single Weather or a list
    weather_series = [weather] if isinstance(weather, Weather) else weather

    # Validate chronological order and positive timestep
    if len(weather_series) >= 2:
        for i in range(1, len(weather_series)):
            if weather_series[i].datetime <= weather_series[i - 1].datetime:
                raise WeatherDataError(
                    field="datetime",
                    value=str(weather_series[i].datetime),
                    reason=(
                        f"Weather series must be in strict chronological order. "
                        f"Entry {i} ({weather_series[i].datetime}) is not after "
                        f"entry {i - 1} ({weather_series[i - 1].datetime})."
                    ),
                )

    return _calculate_timeseries(
        surface=surface,
        weather_series=weather_series,
        location=location,
        config=config,
        human=human,
        precomputed=precomputed,
        use_anisotropic_sky=use_anisotropic_sky,
        conifer=conifer,
        physics=physics,
        materials=materials,
        wall_material=wall_material,
        max_shadow_distance_m=max_shadow_distance_m,
        tile_size=tile_size,
        output_dir=output_dir,
        outputs=outputs,
        heat_thresholds_day=heat_thresholds_day,
        heat_thresholds_night=heat_thresholds_night,
        progress_callback=progress_callback,
    )


# =============================================================================
# Public API - All exports
# =============================================================================

__all__ = [
    # Main calculation function
    "calculate",
    "validate_inputs",
    # Dataclasses - Core inputs
    "SurfaceData",
    "Location",
    "Weather",
    "HumanParams",
    # Dataclasses - Configuration
    "ModelConfig",
    "PrecomputedData",
    "Settings",
    "ThermalState",
    "TileSpec",
    # Dataclasses - Internal (for advanced use)
    "SvfArrays",
    "ShadowArrays",
    # Results
    "SolweigResult",
    "Timeseries",
    "TimeseriesSummary",
    # Errors
    "SolweigError",
    "InvalidSurfaceData",
    "GridShapeMismatch",
    "MissingPrecomputedData",
    "WeatherDataError",
    "ConfigurationError",
    # Configuration loading
    "load_params",
    "load_physics",
    "load_materials",
    "resolve_wall_params",
    # Metadata
    "create_run_metadata",
    "save_run_metadata",
    "load_run_metadata",
    # Tiling utilities
    "calculate_buffer_distance",
    "validate_tile_size",
    "generate_tiles",
    # I/O
    "download_epw",
    # Utilities
    "dict_to_namespace",
    "namespace_to_dict",
    "extract_bounds",
    "intersect_bounds",
    "resample_to_grid",
]
