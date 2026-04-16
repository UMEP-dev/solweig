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
            result = solweig.calculate(surface, location, weather)
        except solweig.GridShapeMismatch as e:
            print(f"Grid mismatch: {e.field} expected {e.expected}, got {e.got}")
        except solweig.MissingPrecomputedData as e:
            print(f"Missing data: {e}")
    """
    warnings = []
    dsm_shape = surface.dsm.shape

    # Check grid shapes match DSM
    grids_to_check = [
        ("cdsm", surface.cdsm),
        ("dem", surface.dem),
        ("tdsm", surface.tdsm),
        ("wall_height", surface.wall_height),
        ("wall_aspect", surface.wall_aspect),
        ("land_cover", surface.land_cover),
        ("albedo", surface.albedo),
        ("emissivity", surface.emissivity),
    ]
    for name, grid in grids_to_check:
        if grid is not None and grid.shape != dsm_shape:
            raise GridShapeMismatch(name, dsm_shape, grid.shape)

    # Check SVF arrays if present
    if surface.svf is not None:
        svf_grids = [
            ("svf.svf", surface.svf.svf),
            ("svf.svf_north", surface.svf.svf_north),
            ("svf.svf_east", surface.svf.svf_east),
            ("svf.svf_south", surface.svf.svf_south),
            ("svf.svf_west", surface.svf.svf_west),
        ]
        for name, grid in svf_grids:
            if grid is not None and grid.shape != dsm_shape:
                raise GridShapeMismatch(name, dsm_shape, grid.shape)

    # Check SVF is available (required for all calculations)
    if surface.svf is None and (precomputed is None or precomputed.svf is None):
        raise MissingPrecomputedData(
            "Sky View Factor (SVF) data is required but not available.",
            "Call surface.compute_svf() before calculate(), or use SurfaceData.prepare() "
            "which computes SVF automatically.",
        )

    # Check anisotropic sky requirements
    if use_anisotropic_sky:
        has_shadow_matrices = (precomputed is not None and precomputed.shadow_matrices is not None) or (
            surface.shadow_matrices is not None
        )
        if not has_shadow_matrices:
            raise MissingPrecomputedData(
                "shadow_matrices required for anisotropic sky model",
                "Either set use_anisotropic_sky=False, or provide shadow matrices via "
                "precomputed=PrecomputedData(shadow_matrices=...) or surface.shadow_matrices",
            )

    # Check for potential issues (warnings, not errors)
    if surface.cdsm is not None and not surface._preprocessed and surface.cdsm_relative:
        warnings.append(
            "CDSM provided with cdsm_relative=True but preprocess() not called. "
            "Vegetation heights may be incorrect. Call surface.preprocess() first."
        )
    if surface.tdsm is not None and not surface._preprocessed and surface.tdsm_relative:
        warnings.append(
            "TDSM provided with tdsm_relative=True but preprocess() not called. "
            "Trunk heights may be incorrect. Call surface.preprocess() first."
        )

    # DSM height sanity checks
    dsm_max = float(np.nanmax(surface.dsm))
    dsm_min = float(np.nanmin(surface.dsm))
    height_range = dsm_max - dsm_min

    if height_range > 500:
        warnings.append(
            f"DSM height range is {height_range:.0f}m (max={dsm_max:.0f}m, min={dsm_min:.0f}m). "
            "This is typical for mountainous terrain. Shadow rays will use full terrain "
            "relief; consider increasing max_shadow_distance_m for wide valleys."
        )

    if surface.dem is None and dsm_min > 100:
        warnings.append(
            f"DSM minimum value is {dsm_min:.0f}m with no DEM provided. "
            "If this is above-sea-level elevation, provide a DEM so SOLWEIG can "
            "compute building heights correctly."
        )

    # Per-layer relative height mismatch detection
    for grid_name, grid, is_relative in [
        ("CDSM", surface.cdsm, surface.cdsm_relative),
        ("TDSM", surface.tdsm, surface.tdsm_relative),
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

    if surface.cdsm is not None and not surface.cdsm_relative and surface.looks_like_relative_heights():
        cdsm_max = float(np.nanmax(surface.cdsm))
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
    # Track whether anisotropic mode was explicitly requested by direct API arg.
    # Config/default fallbacks intentionally do not trigger strict precondition
    # failures because we cannot distinguish "config default" from a deliberate
    # explicit request at runtime.
    anisotropic_requested_explicitly = use_anisotropic_sky is True

    # Build effective configuration: explicit params override config
    # Config provides base values, explicit params take precedence
    effective_aniso = use_anisotropic_sky
    effective_human = human
    effective_physics = physics
    effective_materials = materials
    effective_max_shadow = max_shadow_distance_m

    if config is not None:
        # Use config values as fallback for None parameters
        if effective_aniso is None:
            effective_aniso = config.use_anisotropic_sky
        if effective_human is None:
            effective_human = config.human
        if effective_physics is None:
            effective_physics = config.physics
        if effective_materials is None:
            effective_materials = config.materials
        if effective_max_shadow is None:
            effective_max_shadow = config.max_shadow_distance_m

        # Debug log when explicit params override config
        overrides = []
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

    # Apply defaults for anything still None
    if effective_aniso is None:
        effective_aniso = True
    if effective_human is None:
        effective_human = HumanParams()
    # Auto-load bundled UMEP JSON as default materials (single source of truth)
    if effective_materials is None:
        effective_materials = load_params()

    # Assign back to use in the rest of the function
    use_anisotropic_sky = effective_aniso
    human = effective_human
    physics = effective_physics
    materials = effective_materials

    # Load default physics if not provided
    if physics is None:
        physics = load_physics()

    # Compute derived weather values (sun position, radiation split)
    if not weather._derived_computed:
        weather.compute_derived(location)

    # Fill NaN in surface layers (idempotent — skipped if already done)
    surface.fill_nan()

    # Explicit anisotropic requests must have shadow matrices available.
    if anisotropic_requested_explicitly and use_anisotropic_sky:
        has_shadow_matrices = (precomputed is not None and precomputed.shadow_matrices is not None) or (
            surface.shadow_matrices is not None
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
        human=human,
        precomputed=precomputed,
        state=state,
        physics=physics,
        materials=materials,
        conifer=conifer,
        wall_material=wall_material,
        use_anisotropic_sky=use_anisotropic_sky,
        max_shadow_distance_m=effective_max_shadow,
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
