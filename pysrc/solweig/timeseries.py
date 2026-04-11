"""Time-series SOLWEIG calculation with thermal state management.

Provides :func:`_calculate_timeseries`, the internal implementation that
iterates over a list of :class:`~solweig.Weather` objects, carrying
thermal state (ground and wall temperatures) forward between timesteps.
Large rasters are transparently routed to the tiled processing path.

Users should call :func:`solweig._calculate_single` (the public entry point).
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from .models import HumanParams, Location, ThermalState
from .postprocess import compute_utci_grid
from .progress import ProgressReporter
from .solweig_logging import get_logger
from .summary import GridAccumulator, TimeseriesSummary

logger = get_logger(__name__)


def _precompute_weather(weather_series: list, location: Location) -> None:
    """
    Pre-compute derived weather values for all timesteps efficiently.

    Optimizations:
    1. Compute max sun altitude (altmax) only once per unique day
    2. Pre-assign altmax to Weather objects to skip the 96-iteration loop

    This reduces compute_derived() from O(96) iterations to O(1) per timestep
    when multiple timesteps share the same day.

    Args:
        weather_series: List of Weather objects to process
        location: Geographic location for sun position calculations
    """
    if not weather_series:
        return

    from datetime import timedelta

    from .physics import sun_position as sp

    location_dict = location.to_sun_position_dict()

    # Step 1: Compute altmax once per unique day
    altmax_cache = {}  # date -> altmax

    for weather in weather_series:
        day = weather.datetime.date()
        if day not in altmax_cache:
            # Compute max sun altitude for this day (iterate in 15-min intervals)
            ymd = weather.datetime.replace(hour=0, minute=0, second=0, microsecond=0)
            sunmaximum = -90.0
            fifteen_min = 15.0 / 1440.0  # 15 minutes as fraction of day

            for step in range(96):  # 24 hours * 4 (15-min intervals)
                step_time = ymd + timedelta(days=step * fifteen_min)
                time_dict_step = {
                    "year": step_time.year,
                    "month": step_time.month,
                    "day": step_time.day,
                    "hour": step_time.hour,
                    "min": step_time.minute,
                    "sec": 0,
                    "UTC": location.utc_offset,
                }
                sun_step = sp.sun_position(time_dict_step, location_dict)
                zenith_step = sun_step["zenith"]
                zenith_val = (
                    float(np.asarray(zenith_step).flat[0]) if hasattr(zenith_step, "__iter__") else float(zenith_step)
                )
                altitude_step = 90.0 - zenith_val
                if altitude_step > sunmaximum:
                    sunmaximum = altitude_step

            altmax_cache[day] = max(sunmaximum, 0.0)

    # Step 2: Pre-assign altmax to each weather object
    for weather in weather_series:
        day = weather.datetime.date()
        weather.precomputed_altmax = altmax_cache[day]

    # Step 3: Compute derived values (now fast since altmax is cached)
    for weather in weather_series:
        if not weather._derived_computed:
            weather.compute_derived(location)

    # Step 4: Carry forward clearness index at night.
    # UMEP Python does not recompute CI when the sun is below the horizon;
    # instead the last daytime CI value persists into the nighttime hours.
    # This matters for the Ldown cloud-correction: if the last daytime CI
    # was < 0.95, the correction raises nighttime Ldown (more cloud → more
    # downwelling longwave).  Without carry-forward, CI defaults to 1.0
    # at night and the correction is never triggered.
    last_daytime_ci = 1.0
    for weather in weather_series:
        if weather.sun_altitude > 0 and weather.global_rad > 0:
            last_daytime_ci = weather.clearness_index
        else:
            weather.clearness_index = last_daytime_ci


if TYPE_CHECKING:
    from .models import (
        ModelConfig,
        PrecomputedData,
        SurfaceData,
        Weather,
    )


def _calculate_timeseries(
    surface: SurfaceData,
    weather_series: list[Weather],
    location: Location | None = None,
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
    *,
    output_dir: str | Path,
    outputs: list[str] | None = None,
    heat_thresholds_day: list[float] | None = None,
    heat_thresholds_night: list[float] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> TimeseriesSummary:
    """
    Calculate Tmrt for a time series of weather data.

    Returns a :class:`TimeseriesSummary` with aggregated per-pixel grids
    (mean/max/min Tmrt and UTCI, sun/shade hours, heat-stress exceedance).
    Per-timestep arrays are written to disk and freed after each step.

    Maintains thermal state across timesteps for accurate surface temperature
    modeling with thermal inertia (TsWaveDelay_2015a).

    Large rasters are automatically processed using overlapping tiles to
    manage memory. The tile size and buffer distance are computed dynamically
    from available GPU/RAM resources and the maximum building height in the DSM,
    ensuring accurate shadows at tile boundaries without wasting overlap on
    short buildings.

    This is a convenience function that manages state automatically. For custom
    control over state, use calculate() directly with the state parameter.

    Args:
        surface: Surface/terrain data (DSM required, CDSM/DEM optional).
        weather_series: List of Weather objects in chronological order.
            The datetime of each Weather object determines the timestep size.
        location: Geographic location (lat, lon, UTC offset). If None, automatically
            extracted from surface's CRS metadata.
        config: Model configuration object providing base settings.
            Explicit parameters override config values when provided.
        human: Human body parameters (absorption, posture, weight, height, etc.).
            If None, uses config.human or HumanParams defaults.
        precomputed: Pre-computed SVF and/or shadow matrices. Optional.
        use_anisotropic_sky: Use anisotropic sky model.
            If None, uses config.use_anisotropic_sky (default True).
        conifer: Treat vegetation as evergreen conifers (always leaf-on). Default False.
        physics: Physics parameters (Tree_settings, Posture geometry) from load_physics().
            Site-independent scientific constants. If None, uses config.physics or bundled defaults.
        materials: Material properties (albedo, emissivity per landcover class) from load_materials().
            Site-specific landcover parameters. Only needed if surface has land_cover grid.
            If None, uses config.materials.
        wall_material: Wall material type for temperature model.
            One of "brick", "concrete", "wood", "cobblestone" (case-insensitive).
            If None (default), uses generic wall params from materials JSON.
        max_shadow_distance_m: Maximum shadow reach in metres (default 1000.0).
            Caps horizontal shadow ray distance and serves as the tile overlap
            buffer for automatic tiled processing of large rasters. If None,
            uses config.max_shadow_distance_m or 1000.0.
        tile_size: Core tile side in pixels for tiled processing. If None
            (default), auto-calculated from available GPU/RAM resources.
            Minimum 256. Small rasters that fit in a single tile are
            processed without tiling overhead.
        output_dir: Working directory for all output. Summary grids are always
            saved to ``output_dir/summary/``. Per-timestep GeoTIFFs are saved
            when ``outputs`` is specified.
        outputs: Which per-timestep outputs to save as GeoTIFFs (e.g., ["tmrt", "shadow"]).
            If None, only summary grids are saved.
        heat_thresholds_day: UTCI thresholds (°C) for daytime exceedance hours.
            Default ``[32, 38]`` (strong / very strong heat stress).
        heat_thresholds_night: UTCI thresholds (°C) for nighttime exceedance hours.
            Default ``[26]`` (tropical night threshold).
        progress_callback: Optional callback(current_step, total_steps) called after
            each tile-timestep. For single-tile runs, total_steps equals the
            number of weather timesteps. For multi-tile runs, total_steps is
            n_tiles * n_timesteps. If None, a tqdm progress bar is shown
            automatically (one bar per tile).

    Returns:
        :class:`TimeseriesSummary` with aggregated grids and metadata.

    Example:
        # Summary grids only
        summary = _calculate_timeseries(
            surface=surface,
            weather_series=weather_list,
            output_dir="output/",
        )
        print(summary.tmrt_mean, summary.utci_hours_above[32])

        # With per-timestep GeoTIFFs
        summary = _calculate_timeseries(
            surface=surface,
            weather_series=weather_list,
            output_dir="output/",
            outputs=["tmrt", "shadow"],
        )
    """
    if not weather_series:
        return TimeseriesSummary.empty()

    anisotropic_requested_explicitly = use_anisotropic_sky is True

    # Auto-extract location from surface if not provided
    if location is None:
        logger.warning(
            "Location not provided - auto-extracting from surface CRS.\n"
            "⚠️  UTC offset will default to 0 if not specified, which may cause incorrect sun positions.\n"
            "   Recommend: provide location explicitly with correct UTC offset."
        )
        location = Location.from_surface(surface)

    # Build effective configuration: explicit params override config
    from ._orchestration import resolve_config_params

    _resolved = resolve_config_params(
        config=config,
        use_anisotropic_sky=use_anisotropic_sky,
        human=human,
        physics=physics,
        materials=materials,
        outputs=outputs,
        max_shadow_distance_m=max_shadow_distance_m,
        tile_size=tile_size,
    )
    use_anisotropic_sky = _resolved["use_anisotropic_sky"]
    human = _resolved["human"]
    physics = _resolved["physics"]
    materials = _resolved["materials"]
    outputs = _resolved["outputs"]
    effective_max_shadow = _resolved["max_shadow_distance_m"]
    tile_size = _resolved["tile_size"]
    ignored_tile_runtime = {
        "tile_workers": _resolved["tile_workers"],
        "tile_queue_depth": _resolved["tile_queue_depth"],
        "prefetch_tiles": _resolved["prefetch_tiles"],
    }
    anisotropic_arg = (
        use_anisotropic_sky if (anisotropic_requested_explicitly or use_anisotropic_sky is False) else None
    )

    ignored_runtime_names = [name for name, value in ignored_tile_runtime.items() if value is not None]
    if ignored_runtime_names:
        logger.warning(
            "Ignoring legacy timeseries tile runtime controls: "
            f"{', '.join(ignored_runtime_names)}. "
            "The unified tile-outer timeseries path only honors tile_size."
        )

    # Fill NaN in surface layers (idempotent — skipped if already done)
    surface.fill_nan()

    # ── Tile layout ──────────────────────────────────────────────────────
    # Always generate tiles. For small rasters this produces a single tile
    # covering the whole raster (no overlap). For large rasters it produces
    # multiple overlapping tiles sized to fit GPU/RAM.
    from .tiling import (
        MAX_BUFFER_M,
        _calculate_auto_tile_size,
        _extract_tile_surface,
        _slice_tile_precomputed,
        calculate_buffer_distance,
        compute_max_tile_side,
        generate_tiles,
        validate_tile_size,
    )

    rows, cols = surface.shape
    pixel_size = surface.pixel_size
    max_height = surface.max_height
    eff_max_shadow = effective_max_shadow if effective_max_shadow is not None else MAX_BUFFER_M
    buffer_m = calculate_buffer_distance(max_height, max_shadow_distance_m=eff_max_shadow)
    buffer_pixels = int(np.ceil(buffer_m / pixel_size))

    # If the entire raster fits within resource limits, process as a single
    # tile with no buffer — shadow casting already sees the full extent.
    max_side = compute_max_tile_side(context="solweig")
    if tile_size is None and max(rows, cols) <= max_side:
        buffer_pixels = 0
        adjusted_tile_size = max(rows, cols)
    else:
        if tile_size is not None:
            core_tile_size = tile_size
            logger.info(f"Using explicit tile_size={tile_size}")
        else:
            core_tile_size = _calculate_auto_tile_size(rows, cols, buffer_pixels)
        adjusted_tile_size, tile_warning = validate_tile_size(core_tile_size, buffer_pixels, pixel_size)
        if tile_warning:
            logger.warning(tile_warning)
        if adjusted_tile_size != core_tile_size:
            logger.info(f"Tile size adjusted from {core_tile_size} to {adjusted_tile_size} (buffer constraints)")

    tiles = generate_tiles(rows, cols, adjusted_tile_size, buffer_pixels)
    n_tiles = len(tiles)
    n_steps = len(weather_series)

    # ── Logging ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Starting SOLWEIG timeseries calculation")
    logger.info(f"  Grid size: {cols}x{rows} pixels")
    logger.info(f"  Timesteps: {n_steps}")
    start_str = weather_series[0].datetime.strftime("%Y-%m-%d %H:%M")
    end_str = weather_series[-1].datetime.strftime("%Y-%m-%d %H:%M")
    logger.info(f"  Period: {start_str} -> {end_str}")
    logger.info(f"  Location: {location.latitude:.2f}N, {location.longitude:.2f}E")
    if n_tiles > 1:
        logger.info(
            f"  Tiles: {n_tiles} (size={adjusted_tile_size}, buffer={buffer_m:.0f}m from max height {max_height:.1f}m)"
        )
    logger.info("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine which arrays the Rust compute needs to return.
    requested_outputs: set[str] = {"tmrt", "shadow"}
    if outputs:
        requested_outputs |= set(outputs)

    from .api import _calculate_single

    # Pre-compute weather (sun positions, radiation splits)
    logger.info("Pre-computing sun positions and radiation splits...")
    precompute_start = time.time()
    _precompute_weather(weather_series, location)
    precompute_time = time.time() - precompute_start
    logger.info(f"  Pre-computed {n_steps} timesteps in {precompute_time:.1f}s")

    # Timestep size
    if len(weather_series) >= 2:
        dt0 = weather_series[0].datetime
        dt1 = weather_series[1].datetime
        timestep_dec = (dt1 - dt0).total_seconds() / 86400.0
        _timestep_hours = (dt1 - dt0).total_seconds() / 3600.0
    else:
        timestep_dec = 1.0 / 24.0
        _timestep_hours = 1.0

    # ── Pre-allocate full-raster output grids ────────────────────────────
    # Use memmap backing for large rasters to avoid OOM.
    from .tiling import _MEMMAP_PIXEL_THRESHOLD

    shape = (rows, cols)
    total_pixels = rows * cols
    heat_thresh_day = heat_thresholds_day if heat_thresholds_day is not None else [32.0, 38.0]
    heat_thresh_night = heat_thresholds_night if heat_thresholds_night is not None else [26.0]

    _memmap_tmpdir = None
    if total_pixels > _MEMMAP_PIXEL_THRESHOLD:
        import tempfile

        _memmap_tmpdir = tempfile.TemporaryDirectory(prefix="solweig_ts_")
        _mm_dir = Path(_memmap_tmpdir.name)
        logger.info(f"Large raster ({total_pixels / 1e6:.1f}M pixels) — using memmap in {_mm_dir}")

        def _mm_nan(name: str) -> np.ndarray:
            arr = np.memmap(_mm_dir / f"{name}.dat", dtype=np.float32, mode="w+", shape=shape)
            arr[:] = np.nan
            return arr

        def _mm_zero(name: str) -> np.ndarray:
            arr = np.memmap(_mm_dir / f"{name}.dat", dtype=np.float32, mode="w+", shape=shape)
            arr[:] = 0
            return arr
    else:

        def _mm_nan(name: str) -> np.ndarray:
            return np.full(shape, np.nan, dtype=np.float32)

        def _mm_zero(name: str) -> np.ndarray:
            return np.zeros(shape, dtype=np.float32)

    _SUMMARY_GRID_FIELDS = [
        "tmrt_mean",
        "tmrt_max",
        "tmrt_min",
        "tmrt_day_mean",
        "tmrt_night_mean",
        "utci_mean",
        "utci_max",
        "utci_min",
        "utci_day_mean",
        "utci_night_mean",
        "sun_hours",
        "shade_hours",
    ]
    full_grids = {name: _mm_nan(name) for name in _SUMMARY_GRID_FIELDS}
    full_utci_hours = {t: _mm_zero(f"utci_hours_{t}") for t in heat_thresh_day + heat_thresh_night}

    # Per-timestep GeoTIFF writer (windowed writes for tiled output)
    from .output_async import TiledGeoTiffWriter

    _writer_output_names: list[str] = list(outputs) if outputs else []
    _tiled_writer: TiledGeoTiffWriter | None = None
    if _writer_output_names:
        _writer_transform = surface._geotransform if surface._geotransform is not None else None
        _writer_crs = surface._crs_wkt if surface._crs_wkt is not None else ""
        _tiled_writer = TiledGeoTiffWriter(
            output_dir=output_path,
            rows=rows,
            cols=cols,
            transform=_writer_transform,
            crs_wkt=_writer_crs,
        )
        _tiled_writer.precreate_timesteps([w.datetime for w in weather_series], _writer_output_names)

    # Per-timestep scalar accumulators (partial sums across tiles)
    ts_tmrt_sum = np.zeros(n_steps)
    ts_tmrt_count = np.zeros(n_steps, dtype=np.int64)
    ts_utci_sum = np.zeros(n_steps)
    ts_utci_count = np.zeros(n_steps, dtype=np.int64)
    ts_shadow_sunlit_sum = np.zeros(n_steps)
    ts_shadow_valid_count = np.zeros(n_steps, dtype=np.int64)
    start_time = time.time()
    tile_loop_completed = False

    # ── Main loop: process each tile's full timeseries ───────────────────
    try:
        for tile_idx, tile in enumerate(tiles):
            tile_desc = f"Tile {tile_idx + 1}/{n_tiles}" if n_tiles > 1 else "SOLWEIG timeseries"
            _tile_progress = None if progress_callback is not None else ProgressReporter(total=n_steps, desc=tile_desc)

            # 1. Cut tile (with overlap buffer)
            tile_surface = _extract_tile_surface(surface, tile, pixel_size, precomputed=precomputed)
            tile_precomputed = _slice_tile_precomputed(precomputed, tile)
            tile_state = ThermalState.initial(tile.full_shape)
            tile_state.timestep_dec = timestep_dec
            tile_accum = GridAccumulator(
                shape=tile.full_shape,
                heat_thresholds_day=heat_thresh_day,
                heat_thresholds_night=heat_thresh_night,
                timestep_hours=_timestep_hours,
                track_scalars=False,
            )

            # 2. Run exactly like the non-tiled path
            for t_idx, weather in enumerate(weather_series):
                result = _calculate_single(
                    surface=tile_surface,
                    location=location,
                    weather=weather,
                    human=human,
                    precomputed=tile_precomputed,
                    use_anisotropic_sky=anisotropic_arg,
                    conifer=conifer,
                    state=tile_state,
                    physics=physics,
                    materials=materials,
                    wall_material=wall_material,
                    max_shadow_distance_m=eff_max_shadow,
                    return_state_copy=False,
                    _requested_outputs=requested_outputs,
                )

                # Carry forward thermal state
                if result.state is not None:
                    tile_state = result.state
                    result.state = None

                # Accumulate summary (returns the UTCI grid for reuse)
                utci_full = tile_accum.update(result, weather, compute_utci_fn=compute_utci_grid)
                cs = tile.core_slice

                # Per-timestep GeoTIFF output (windowed write)
                if _tiled_writer is not None:
                    core = cs
                    tile_arrays: dict[str, np.ndarray] = {}
                    if "tmrt" in _writer_output_names and result.tmrt is not None:
                        tile_arrays["tmrt"] = result.tmrt[core]
                    if "shadow" in _writer_output_names and result.shadow is not None:
                        tile_arrays["shadow"] = result.shadow[core]
                    if "kdown" in _writer_output_names and result.kdown is not None:
                        tile_arrays["kdown"] = result.kdown[core]
                    if "kup" in _writer_output_names and result.kup is not None:
                        tile_arrays["kup"] = result.kup[core]
                    if "ldown" in _writer_output_names and result.ldown is not None:
                        tile_arrays["ldown"] = result.ldown[core]
                    if "lup" in _writer_output_names and result.lup is not None:
                        tile_arrays["lup"] = result.lup[core]
                    if "utci" in _writer_output_names and utci_full is not None:
                        tile_arrays["utci"] = utci_full[core]
                    if "pet" in _writer_output_names and result.tmrt is not None:
                        from .postprocess import compute_pet_grid

                        tile_arrays["pet"] = compute_pet_grid(
                            result.tmrt[core], weather.ta, weather.rh, weather.ws, human
                        )
                    if tile_arrays:
                        _tiled_writer.write_tile_at(weather.datetime, tile.write_slice, tile_arrays)

                # Per-timestep scalar accumulation (core region only, reuse UTCI)
                if result.tmrt is not None:
                    tmrt_cs = result.tmrt[cs]
                    valid = np.isfinite(tmrt_cs)
                    n_v = int(valid.sum())
                    if n_v > 0:
                        ts_tmrt_sum[t_idx] += float(tmrt_cs[valid].sum())
                        ts_tmrt_count[t_idx] += n_v
                        utci_cs = utci_full[cs]
                        uv = np.isfinite(utci_cs) & valid
                        n_u = int(uv.sum())
                        if n_u > 0:
                            ts_utci_sum[t_idx] += float(utci_cs[uv].sum())
                            ts_utci_count[t_idx] += n_u
                if result.shadow is not None and weather.is_daytime:
                    shadow_cs = result.shadow[cs]
                    sv = np.isfinite(shadow_cs)
                    n_s = int(sv.sum())
                    if n_s > 0:
                        ts_shadow_sunlit_sum[t_idx] += float(shadow_cs[sv].sum())
                        ts_shadow_valid_count[t_idx] += n_s

                # Free result arrays
                result.tmrt = None  # type: ignore[assignment]
                result.shadow = None
                result.kdown = None
                result.kup = None
                result.ldown = None
                result.lup = None

                # Progress: callback reports tile-level progress
                if progress_callback is not None:
                    progress_callback(tile_idx * n_steps + t_idx + 1, n_tiles * n_steps)
                elif _tile_progress is not None:
                    _tile_progress.update(1)

            # 3. Finalize tile and stitch core to full-raster output
            tile_summary = tile_accum.finalize()
            cs, ws = tile.core_slice, tile.write_slice
            for name in _SUMMARY_GRID_FIELDS:
                full_grids[name][ws] = getattr(tile_summary, name)[cs]
            for t_val, grid in tile_summary.utci_hours_above.items():
                full_utci_hours[t_val][ws] = grid[cs]

            if n_tiles > 1:
                logger.debug(f"  Tile {tile_idx + 1}/{n_tiles} complete")

            if _tile_progress is not None:
                _tile_progress.close()
            del tile_surface, tile_precomputed, tile_state, tile_accum, tile_summary
        tile_loop_completed = True
    finally:
        if _tiled_writer is not None and not tile_loop_completed:
            _tiled_writer.close(success=False)
            _tiled_writer = None

    # ── Build final TimeseriesSummary ────────────────────────────────────
    # Summary construction and GeoTIFF export must happen while memmaps
    # still exist. The finally block ensures cleanup even on error.
    from .summary import Timeseries

    try:
        timeseries = Timeseries(
            datetime=[w.datetime for w in weather_series],
            ta=np.array([w.ta for w in weather_series], dtype=np.float32),
            rh=np.array([w.rh for w in weather_series], dtype=np.float32),
            ws=np.array([w.ws for w in weather_series], dtype=np.float32),
            global_rad=np.array([w.global_rad for w in weather_series], dtype=np.float32),
            direct_rad=np.array([w.direct_rad for w in weather_series], dtype=np.float32),
            diffuse_rad=np.array([w.diffuse_rad for w in weather_series], dtype=np.float32),
            sun_altitude=np.array([w.sun_altitude for w in weather_series], dtype=np.float32),
            tmrt_mean=np.divide(
                ts_tmrt_sum, ts_tmrt_count, out=np.full(n_steps, np.nan), where=ts_tmrt_count > 0
            ).astype(np.float32),
            utci_mean=np.divide(
                ts_utci_sum, ts_utci_count, out=np.full(n_steps, np.nan), where=ts_utci_count > 0
            ).astype(np.float32),
            sun_fraction=np.divide(
                ts_shadow_sunlit_sum,
                ts_shadow_valid_count,
                out=np.where([w.is_daytime for w in weather_series], np.nan, 0.0),
                where=ts_shadow_valid_count > 0,
            ).astype(np.float32),
            diffuse_fraction=np.array(
                [w.diffuse_rad / w.global_rad if w.global_rad > 0 else np.nan for w in weather_series],
                dtype=np.float32,
            ),
            clearness_index=np.array([w.clearness_index for w in weather_series], dtype=np.float32),
            is_daytime=np.array([w.is_daytime for w in weather_series], dtype=np.bool_),
        )

        n_daytime = sum(1 for w in weather_series if w.is_daytime)

        total_time = time.time() - start_time
        logger.info("=" * 60)
        label = " (tiled)" if n_tiles > 1 else ""
        logger.info(f"Calculation complete: {n_steps} timesteps processed{label}")
        logger.info(f"  Total time: {total_time:.1f}s ({n_steps / total_time:.2f} steps/s)" if total_time > 0 else "")
        logger.info("=" * 60)

        summary = TimeseriesSummary(
            tmrt_mean=full_grids["tmrt_mean"],
            tmrt_max=full_grids["tmrt_max"],
            tmrt_min=full_grids["tmrt_min"],
            tmrt_day_mean=full_grids["tmrt_day_mean"],
            tmrt_night_mean=full_grids["tmrt_night_mean"],
            utci_mean=full_grids["utci_mean"],
            utci_max=full_grids["utci_max"],
            utci_min=full_grids["utci_min"],
            utci_day_mean=full_grids["utci_day_mean"],
            utci_night_mean=full_grids["utci_night_mean"],
            sun_hours=full_grids["sun_hours"],
            shade_hours=full_grids["shade_hours"],
            utci_hours_above=full_utci_hours,
            n_timesteps=n_steps,
            n_daytime=n_daytime,
            n_nighttime=n_steps - n_daytime,
            shadow_available=True,
            heat_thresholds_day=heat_thresh_day,
            heat_thresholds_night=heat_thresh_night,
            timeseries=timeseries,
        )
        summary._surface = surface

        # Save summary GeoTIFFs (must happen while memmaps still exist)
        summary.to_geotiff(output_dir, surface=surface)
        summary._output_dir = output_path

        from .metadata import create_run_metadata, save_run_metadata

        metadata = create_run_metadata(
            surface=surface,
            location=location,
            weather_series=weather_series,
            human=human,
            physics=physics,
            materials=materials,
            use_anisotropic_sky=use_anisotropic_sky,
            conifer=conifer,
            output_dir=output_dir,
            outputs=outputs,
        )
        save_run_metadata(metadata, output_dir)

        # Convert memmap arrays to heap before deleting backing files
        if _memmap_tmpdir is not None:
            for name in _SUMMARY_GRID_FIELDS:
                setattr(summary, name, np.array(getattr(summary, name)))
            summary.utci_hours_above = {t: np.array(g) for t, g in summary.utci_hours_above.items()}

        if _tiled_writer is not None:
            _tiled_writer.close(success=True)
            _tiled_writer = None

        return summary
    finally:
        if _tiled_writer is not None:
            _tiled_writer.close(success=False)
        if _memmap_tmpdir is not None:
            # Release memmap file handles before deleting backing files.
            # On Windows, open handles prevent file deletion (PermissionError).
            full_grids.clear()
            full_utci_hours.clear()
            gc.collect()
            try:
                _memmap_tmpdir.cleanup()
            except OSError:
                logger.warning(f"Could not remove memmap temp dir {_memmap_tmpdir.name} (will be cleaned by OS)")
