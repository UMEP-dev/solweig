"""Orchestration layer for a single SOLWEIG timestep.

The public entry point is :func:`calculate_core_fused`, which hands off
the full pipeline (shadows, ground temperature, GVF, thermal delay,
radiation, Tmrt) to a single fused Rust FFI call.

Pipeline::

    SVF resolution → Shadows → Ground temp → GVF
        → Thermal delay → Radiation → Tmrt
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

_OUT_SHADOW = 1 << 0
_OUT_KDOWN = 1 << 1
_OUT_KUP = 1 << 2
_OUT_LDOWN = 1 << 3
_OUT_LUP = 1 << 4
_OUT_ALL = _OUT_SHADOW | _OUT_KDOWN | _OUT_KUP | _OUT_LDOWN | _OUT_LUP


def _arr_key(arr):
    """Cache key for `arr` that catches in-place mutations cheaply.

    Composition: data pointer + shape + dtype + bit-pattern of three
    witness elements (first / middle / last). Pointer + shape catches
    array replacement; witness bytes catch most in-place edits without
    paying for a full content hash.

    NOT caught: a mutation that touches only elements *other* than
    positions 0, n//2, n-1. The documented invariant (`INVARIANTS.md`
    invariant 3) is "don't mutate surface arrays after passing them to
    calculate()", and this key is a defense-in-depth check, not a full
    guarantee.

    O(1) regardless of array size. NaN-safe via uint8 bit-pattern view.
    """
    if arr is None:
        return None
    n = arr.size
    if n == 0:
        witness = b""
    else:
        flat = arr.ravel()
        mid = n // 2
        # 1-element slices keep dimensionality so .view(uint8) is legal.
        # Bit-pattern bytes avoid NaN equality issues.
        witness = (
            flat[0:1].view(np.uint8).tobytes(),
            flat[mid : mid + 1].view(np.uint8).tobytes(),
            flat[n - 1 : n].view(np.uint8).tobytes(),
        )
    return (arr.ctypes.data, arr.shape, arr.dtype.str, witness)


if TYPE_CHECKING:
    from .api import HumanParams, Location, PrecomputedData, SolweigResult, SurfaceData, ThermalState, Weather
    from .components.ground_scheme import GroundSchemeState


def calculate_core_fused(
    surface: SurfaceData,
    location: Location,
    weather: Weather,
    human: HumanParams,
    precomputed: PrecomputedData | None,
    state: ThermalState | None,
    physics: SimpleNamespace | None,
    materials: SimpleNamespace | None,
    conifer: bool = False,
    wall_material: str | None = None,
    use_anisotropic_sky: bool = False,
    max_shadow_distance_m: float | None = None,
    ground_scheme_state: GroundSchemeState | None = None,
    return_state_copy: bool = True,
    requested_outputs: set[str] | None = None,
) -> SolweigResult:
    """
    Fused SOLWEIG calculation — single Rust FFI call per daytime timestep.

    Orchestrates shadows, ground temperature, GVF, thermal delay, radiation,
    and Tmrt entirely within Rust, eliminating intermediate numpy allocations
    and FFI round-trips.

    This is the primary compute path used by ``calculate()``.
    Supports both isotropic and anisotropic (Perez) sky models.

    Args:
        surface: Surface/terrain data (DSM, vegetation, walls, land cover).
        location: Geographic location (latitude, longitude).
        weather: Weather conditions with derived sun position.
        human: Human parameters (height, posture, absorptivities).
        precomputed: Optional pre-computed data (SVF, shadow matrices).
        state: Optional thermal state for time-series (carries forward temperatures).
        physics: Optional physics parameters (vegetation transmissivity, etc.).
        materials: Optional material properties (albedo, emissivity by land cover).
        conifer: Treat vegetation as evergreen conifers (always leaf-on).
        wall_material: Wall material type ("brick", "concrete", "wood", "cobblestone").
        use_anisotropic_sky: Use anisotropic (Perez) diffuse sky model.
        max_shadow_distance_m: Maximum shadow reach in metres.
        ground_scheme_state: UMEP 2026a ground-surface scheme state (force-restore
            surface temperature + solid-angle outgoing longwave). When provided,
            the scheme replaces the classic sinusoidal Tg / GVF path; its carried
            arrays (tg, rn, rn_past, g, shadow_past) are mutated in place for the
            next timestep. None (default) runs the byte-identical baseline.
        return_state_copy: If True, return a deep-copied thermal state.
        requested_outputs: Set of output names to materialize (None = all).

    Returns:
        SolweigResult with Tmrt, shadow, radiation components, and updated state.
    """
    from .api import SolweigResult
    from .buffers import as_float32
    from .components.gvf import detect_building_mask
    from .components.shadows import compute_transmissivity
    from .components.svf_resolution import resolve_svf
    from .models.state import ThermalState
    from .physics.clearnessindex_2013b import clearnessindex_2013b
    from .physics.daylen import daylen
    from .physics.diffusefraction import diffusefraction
    from .rustalgos import pipeline

    # Ensure derived weather fields are computed (sun position, radiation split)
    if not weather._derived_computed:
        weather.compute_derived(location)

    # === Precompute (stays in Python) ===

    # Access surface state through the typed views (geometry/optical/auxiliary).
    # The views proxy to the same fields as direct attribute access; using them
    # makes the per-concern grouping explicit at the call site.
    geom = surface.geometry
    optical = surface.optical
    aux = surface.auxiliary

    rows, cols = geom.shape
    pixel_size = geom.pixel_size

    # Valid pixel mask (True where all layers have finite data)
    # Computed once by SurfaceData.prepare(), or derived from DSM if missing
    valid_mask = aux.valid_mask
    valid_source = valid_mask if valid_mask is not None else geom.dsm
    valid_mask_key = _arr_key(valid_source)
    cache = surface._cache

    def _compute_valid_mask_u8():
        vm = valid_mask if valid_mask is not None else np.isfinite(geom.dsm)
        return np.ascontiguousarray(vm, dtype=np.uint8)

    valid_mask_u8 = cache.get_or_compute("valid_mask_u8_cache", valid_mask_key, _compute_valid_mask_u8)

    # Valid-bounds crop: trim heavy per-timestep compute to the minimal bounding
    # rectangle of valid pixels.
    def _compute_valid_bbox():
        rows_any = np.any(valid_mask_u8 != 0, axis=1)
        cols_any = np.any(valid_mask_u8 != 0, axis=0)
        if not rows_any.any() or not cols_any.any():
            return 0, rows, 0, cols
        r_idx = np.flatnonzero(rows_any)
        c_idx = np.flatnonzero(cols_any)
        return int(r_idx[0]), int(r_idx[-1]) + 1, int(c_idx[0]), int(c_idx[-1]) + 1

    r0, r1, c0, c1 = cache.get_or_compute("valid_bbox_cache", valid_mask_key, _compute_valid_bbox)

    full_area = rows * cols
    crop_area = (r1 - r0) * (c1 - c0)
    use_crop = (r0, r1, c0, c1) != (0, rows, 0, cols) and crop_area < int(full_area * 0.98)
    # The ground scheme carries per-pixel state (Tg, fluxes, shadow_past) and
    # marches ~11 m across the grid; run it on the full raster so the carried
    # state and the march are not truncated to the valid bounding box.
    if ground_scheme_state is not None:
        use_crop = False
    crop_slice = (slice(r0, r1), slice(c0, c1))

    # Select which non-Tmrt outputs to materialize from Rust.
    if requested_outputs is None:
        output_mask = _OUT_ALL
    else:
        output_mask = 0
        if "shadow" in requested_outputs:
            output_mask |= _OUT_SHADOW
        if "kdown" in requested_outputs:
            output_mask |= _OUT_KDOWN
        if "kup" in requested_outputs:
            output_mask |= _OUT_KUP
        if "ldown" in requested_outputs:
            output_mask |= _OUT_LDOWN
        if "lup" in requested_outputs:
            output_mask |= _OUT_LUP

    # The ground scheme carries shadow_past forward from the shadow the Rust
    # side returns, so shadow must always be materialised when it is active.
    if ground_scheme_state is not None:
        output_mask |= _OUT_SHADOW

    # Fast path — fully-nodata tile. The geometric tiler (generate_tiles) emits a
    # grid of tiles regardless of coverage, so on an irregular raster (e.g. Madrid)
    # some tiles fall entirely outside the valid data. Such a tile produces only
    # NaN outputs, so skip the whole shadow/GVF/aniso/radiation compute and return
    # NaN directly. The summary accumulator gates on finite Tmrt, so this yields the
    # identical 0.0 sun/shade-hours and NaN means as the full compute; the only
    # change is the per-timestep shadow raster, which becomes NaN over nodata rather
    # than a spurious "sunlit" 1.0 (invalid pixels are not tagged nodata otherwise).
    # The 2026a ground scheme carries per-pixel state and marches across the grid, so
    # it is excluded here (it also runs untiled on the full raster).
    if ground_scheme_state is None and not bool(np.any(valid_mask_u8)):
        if state is None:
            state = ThermalState.initial((rows, cols))

        def _nan_grid() -> np.ndarray:
            return np.full((rows, cols), np.nan, dtype=np.float32)

        return SolweigResult(
            tmrt=_nan_grid(),
            shadow=_nan_grid() if output_mask & _OUT_SHADOW else None,
            kdown=_nan_grid() if output_mask & _OUT_KDOWN else None,
            kup=_nan_grid() if output_mask & _OUT_KUP else None,
            ldown=_nan_grid() if output_mask & _OUT_LDOWN else None,
            lup=_nan_grid() if output_mask & _OUT_LUP else None,
            utci=None,
            pet=None,
            state=state.copy() if return_state_copy else state,
        )

    # Land cover properties
    lc_props_key = (_arr_key(optical.land_cover), _arr_key(optical.albedo), _arr_key(optical.emissivity), id(materials))
    alb_grid, emis_grid, tgk_grid, tstart_grid, tmaxlst_grid = cache.get_or_compute(
        "land_cover_props_cache",
        lc_props_key,
        lambda: surface.get_land_cover_properties(materials),
    )

    # Vegetation inputs
    use_veg = geom.cdsm is not None
    cdsm = geom.cdsm if use_veg else None
    tdsm = geom.tdsm if use_veg else None
    if use_veg:
        pool = surface.get_buffer_pool()
        bush = pool.get_zeros("bush")
    else:
        bush = None

    # Wall inputs (via the auxiliary view which exposes has_walls explicitly)
    has_walls = aux.has_walls
    wall_ht = aux.wall_height if has_walls else None
    wall_asp = aux.wall_aspect if has_walls else None

    # Use full terrain relief for shadow ray termination so that mountain
    # ridges can correctly shadow valleys.  The horizontal reach is still
    # bounded by max_shadow_distance_m via max_index in Rust, so rays
    # don't run forever — they just won't terminate prematurely on the
    # vertical axis when terrain relief exceeds building heights.
    max_height = geom.max_height

    # SVF resolution (cached between timesteps)
    svf_bundle = resolve_svf(
        surface=surface,
        precomputed=precomputed,
    )

    # Vegetation transmissivity
    doy = weather.datetime.timetuple().tm_yday
    psi = compute_transmissivity(doy, physics, conifer)

    # Adjust svfbuveg for vegetation transmissivity (shortwave sees through canopy)
    # Without this, isotropic diffuse (drad), Kup, and Kdown treat vegetation as
    # fully opaque.  The anisotropic path already applies psi per sky patch via
    # diffsh(psi), and kside_veg applies psi per direction, but the scalar svfbuveg
    # used for isotropic diffuse and wall reflection was unadjusted.
    from .components.svf_resolution import adjust_svfbuveg_with_psi

    svf_bundle.svfbuveg = adjust_svfbuveg_with_psi(svf_bundle.svf, svf_bundle.svf_veg, psi, use_veg)

    # Wall material resolution. Defaults match historical UMEP behaviour;
    # `wall_material` takes precedence, otherwise fall back to per-property
    # overrides under `materials.{Ts_deg,Tstart,TmaxLST}.Value.Walls`.
    tgk_wall = 0.37
    tstart_wall = -3.41
    tmaxlst_wall = 15.0
    albedo_wall = 0.20
    emis_wall = 0.90
    if wall_material is not None:
        from .loaders import resolve_wall_params

        tgk_wall, tstart_wall, tmaxlst_wall = resolve_wall_params(wall_material, materials)
    else:
        from .models.materials import WallMaterialDefaults

        tgk_wall, tstart_wall, tmaxlst_wall = WallMaterialDefaults.from_namespace(materials).apply(
            tgk_wall, tstart_wall, tmaxlst_wall
        )

    # Weather-derived scalars for ground temperature model
    _, _, _, snup = daylen(doy, location.latitude)
    dectime = (weather.datetime.hour + weather.datetime.minute / 60.0) / 24.0
    zen_deg = 90.0 - weather.sun_altitude

    # Clear-sky radiation for ground temperature CI correction
    zen_rad = zen_deg * (np.pi / 180.0)
    location_dict = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "altitude": 0.0,
    }
    i0, _, _, _, _ = clearnessindex_2013b(
        zen_rad,
        doy,
        weather.ta,
        weather.rh / 100.0,
        weather.global_rad,
        location_dict,
        -999.0,
    )
    if i0 > 0 and weather.sun_altitude > 0:
        rad_i0, rad_d0 = diffusefraction(i0, weather.sun_altitude, 1.0, weather.ta, weather.rh)
        rad_g0 = rad_i0 * np.sin(weather.sun_altitude * np.pi / 180.0) + rad_d0
    else:
        rad_g0 = 0.0

    # === Build Rust input structs ===

    ws = pipeline.WeatherScalars(
        sun_azimuth=float(weather.sun_azimuth),
        sun_altitude=float(weather.sun_altitude),
        sun_zenith=float(weather.sun_zenith),
        ta=float(weather.ta),
        rh=float(weather.rh),
        global_rad=float(weather.global_rad),
        direct_rad=float(weather.direct_rad),
        diffuse_rad=float(weather.diffuse_rad),
        altmax=float(weather.altmax),
        clearness_index=float(weather.clearness_index),
        dectime=float(dectime),
        snup=float(snup),
        rad_g0=float(rad_g0),
        zen_deg=float(zen_deg),
        psi=float(psi),
        is_daytime=weather.sun_altitude > 0,
        jday=int(weather.datetime.timetuple().tm_yday) if weather.datetime is not None else 180,
        patch_option=0,  # Set below if anisotropic
    )

    hs = pipeline.HumanScalars(
        height=float(human.height),
        abs_k=float(human.abs_k),
        abs_l=float(human.abs_l),
        is_standing=human.posture == "standing",
    )

    cs = pipeline.ConfigScalars(
        pixel_size=float(pixel_size),
        max_height=float(max_height),
        albedo_wall=float(albedo_wall),
        emis_wall=float(emis_wall),
        tgk_wall=float(tgk_wall),
        tstart_wall=float(tstart_wall),
        tmaxlst_wall=float(tmaxlst_wall),
        use_veg=use_veg,
        has_walls=has_walls,
        conifer=conifer,
        use_anisotropic=use_anisotropic_sky,
        max_shadow_distance_m=float(max_shadow_distance_m or 1000.0),
    )

    # Buildings mask for GVF (computed from DSM/land_cover/walls)
    buildings_key = (_arr_key(geom.dsm), _arr_key(optical.land_cover), _arr_key(wall_ht), float(pixel_size))
    buildings = cache.get_or_compute(
        "buildings_mask_cache",
        buildings_key,
        lambda: detect_building_mask(geom.dsm, optical.land_cover, wall_ht, pixel_size),
    )

    if optical.has_land_cover:
        land_cover = optical.land_cover  # narrow for the lambda closure
        assert land_cover is not None  # has_land_cover implies non-None
        lc_grid = cache.get_or_compute(
            "lc_grid_f32_cache",
            _arr_key(land_cover),
            lambda: land_cover.astype(np.float32),
        )
    else:
        lc_grid = None

    # GVF geometry cache: precompute on first daytime call, reuse on subsequent.
    # Keep separate caches for full-grid and cropped-grid execution.
    # The 2026a ground scheme replaces the GVF step with the solid-angle march,
    # so the geometry cache is never consulted there — skip the precompute.
    gvf_cache = None
    if has_walls and ground_scheme_state is None:
        assert wall_asp is not None  # guaranteed by has_walls
        assert wall_ht is not None
        if use_crop:
            gvf_crop_key = (
                _arr_key(buildings),
                _arr_key(wall_asp),
                _arr_key(wall_ht),
                _arr_key(alb_grid),
                r0,
                r1,
                c0,
                c1,
                float(pixel_size),
                float(human.height),
                float(albedo_wall),
            )
            gvf_cache = cache.get_or_compute(
                "gvf_geometry_cache_crop",
                gvf_crop_key,
                lambda: pipeline.precompute_gvf_cache(
                    as_float32(buildings[crop_slice]),
                    as_float32(wall_asp[crop_slice]),
                    as_float32(wall_ht[crop_slice]),
                    as_float32(alb_grid[crop_slice]),
                    float(pixel_size),
                    float(human.height),
                    float(albedo_wall),
                ),
            )
        else:
            # Full-grid case: cache slot value is the gvf_cache directly (no
            # key needed — it's invariant across timesteps for the full grid).
            gvf_cache = cache.gvf_geometry_cache
            if gvf_cache is None:
                gvf_cache = pipeline.precompute_gvf_cache(
                    as_float32(buildings),
                    as_float32(wall_asp),
                    as_float32(wall_ht),
                    as_float32(alb_grid),
                    float(pixel_size),
                    float(human.height),
                    float(albedo_wall),
                )
                cache.gvf_geometry_cache = gvf_cache

    # Anisotropic sky: Perez luminance, steradians, ASVF, and esky are now
    # computed inside the Rust pipeline (no Python round-trip). We only need
    # the shadow matrices and the patch_option.
    aniso_shmat = None
    aniso_vegshmat = None
    aniso_vbshmat = None

    if use_anisotropic_sky:
        shadow_mats = None
        if precomputed is not None and precomputed.shadow_matrices is not None:
            shadow_mats = precomputed.shadow_matrices
        elif aux.shadow_matrices is not None:
            shadow_mats = aux.shadow_matrices

        if shadow_mats is not None:
            ws.patch_option = shadow_mats.patch_option
            if use_crop:
                aniso_crop_key = (
                    _arr_key(shadow_mats._shmat_u8),
                    _arr_key(shadow_mats._vegshmat_u8),
                    _arr_key(shadow_mats._vbshmat_u8),
                    r0,
                    r1,
                    c0,
                    c1,
                )
                aniso_shmat, aniso_vegshmat, aniso_vbshmat = cache.get_or_compute(
                    "aniso_shadow_crop_cache",
                    aniso_crop_key,
                    lambda: (
                        np.ascontiguousarray(shadow_mats._shmat_u8[crop_slice]),
                        np.ascontiguousarray(shadow_mats._vegshmat_u8[crop_slice]),
                        np.ascontiguousarray(shadow_mats._vbshmat_u8[crop_slice]),
                    ),
                )
            else:
                # Keep original arrays to preserve stable pointers across timesteps.
                aniso_shmat = shadow_mats._shmat_u8
                aniso_vegshmat = shadow_mats._vegshmat_u8
                aniso_vbshmat = shadow_mats._vbshmat_u8

    # Thermal state (create initial if None)
    if state is None:
        state = ThermalState.initial((rows, cols))

    firstdaytime_int = int(state.firstdaytime)

    def _sel(arr):
        if arr is None:
            return None
        return arr[crop_slice] if use_crop else arr

    dsm_call = _sel(geom.dsm)
    cdsm_call = _sel(cdsm)
    tdsm_call = _sel(tdsm)
    bush_call = _sel(bush)
    wall_ht_call = _sel(wall_ht)
    wall_asp_call = _sel(wall_asp)
    # Build the Rust SvfBundle (17 contiguous f32 arrays) once per timestep.
    # The bundle takes ownership of the numpy array refs; the per-array
    # `.bind(py).readonly()` happens inside Rust. Eliminates 17 positional
    # args from the FFI signature.
    rust_svf_bundle = pipeline.SvfBundle(
        as_float32(_sel(svf_bundle.svf)),
        as_float32(_sel(svf_bundle.svf_directional.north)),
        as_float32(_sel(svf_bundle.svf_directional.east)),
        as_float32(_sel(svf_bundle.svf_directional.south)),
        as_float32(_sel(svf_bundle.svf_directional.west)),
        as_float32(_sel(svf_bundle.svf_veg)),
        as_float32(_sel(svf_bundle.svf_veg_directional.north)),
        as_float32(_sel(svf_bundle.svf_veg_directional.east)),
        as_float32(_sel(svf_bundle.svf_veg_directional.south)),
        as_float32(_sel(svf_bundle.svf_veg_directional.west)),
        as_float32(_sel(svf_bundle.svf_aveg)),
        as_float32(_sel(svf_bundle.svf_aveg_directional.north)),
        as_float32(_sel(svf_bundle.svf_aveg_directional.east)),
        as_float32(_sel(svf_bundle.svf_aveg_directional.south)),
        as_float32(_sel(svf_bundle.svf_aveg_directional.west)),
        as_float32(_sel(svf_bundle.svfbuveg)),
        as_float32(_sel(svf_bundle.svfalfa)),
    )
    alb_call = _sel(alb_grid)
    emis_call = _sel(emis_grid)
    tgk_call = _sel(tgk_grid)
    tstart_call = _sel(tstart_grid)
    tmaxlst_call = _sel(tmaxlst_grid)
    buildings_call = _sel(buildings)
    lc_grid_call = _sel(lc_grid)
    valid_mask_call = _sel(valid_mask_u8)
    tgmap1_call = _sel(state.tgmap1)
    tgmap1_e_call = _sel(state.tgmap1_e)
    tgmap1_s_call = _sel(state.tgmap1_s)
    tgmap1_w_call = _sel(state.tgmap1_w)
    tgmap1_n_call = _sel(state.tgmap1_n)
    tgout1_call = _sel(state.tgout1)

    # === Call fused Rust pipeline ===

    # Surface bundle (DSM required, 5 optional auxiliaries).
    rust_surface_bundle = pipeline.SurfaceBundle(
        as_float32(dsm_call),
        as_float32(cdsm_call) if cdsm_call is not None else None,
        as_float32(tdsm_call) if tdsm_call is not None else None,
        as_float32(bush_call) if bush_call is not None else None,
        as_float32(wall_ht_call) if wall_ht_call is not None else None,
        as_float32(wall_asp_call) if wall_asp_call is not None else None,
    )

    # Land-cover property bundle (5 rasters).
    rust_properties_bundle = pipeline.PropertiesBundle(
        as_float32(alb_call),
        as_float32(emis_call),
        as_float32(tgk_call),
        as_float32(tstart_call),
        as_float32(tmaxlst_call),
    )

    # Thermal state bundle (6 arrays + 3 scalars + version).
    rust_state_bundle = pipeline.StateBundle(
        pipeline.STATE_BUNDLE_VERSION,
        firstdaytime_int,
        float(state.timeadd),
        float(state.timestep_dec),
        as_float32(tgmap1_call),
        as_float32(tgmap1_e_call),
        as_float32(tgmap1_s_call),
        as_float32(tgmap1_w_call),
        as_float32(tgmap1_n_call),
        as_float32(tgout1_call),
    )

    # Ground-scheme bundle (UMEP 2026a, opt-in). timestep_s comes from the
    # thermal state's timestep_dec (fraction of a day → seconds).
    gss = ground_scheme_state
    rust_ground_scheme = None
    if gss is not None:
        rust_ground_scheme = pipeline.GroundSchemeBundle(
            pipeline.GROUND_SCHEME_BUNDLE_VERSION,
            float(state.timestep_dec) * 86400.0,
            as_float32(gss.tg),
            as_float32(gss.tm),
            as_float32(gss.rn),
            as_float32(gss.rn_past),
            as_float32(gss.g),
            as_float32(gss.cap),
            as_float32(gss.diff),
            as_float32(gss.a1),
            as_float32(gss.a2),
            as_float32(gss.a3),
            as_float32(gss.lc_grid),
            as_float32(gss.shadow_past),
        )

    result = pipeline.compute_timestep(
        # Scalar structs
        ws,
        hs,
        cs,
        # GVF geometry cache (None on first call triggers full GVF, then cached)
        gvf_cache,
        # Surface arrays (6 rasters bundled into one SurfaceBundle)
        rust_surface_bundle,
        # SVF arrays (17 rasters bundled into one PyO3 SvfBundle)
        rust_svf_bundle,
        # Land cover property grids (5 rasters bundled into one PropertiesBundle)
        rust_properties_bundle,
        # Buildings mask + land cover
        as_float32(buildings_call),
        as_float32(lc_grid_call) if lc_grid_call is not None else None,
        # Anisotropic sky inputs (None for isotropic; Perez computed in Rust)
        aniso_shmat,
        aniso_vegshmat,
        aniso_vbshmat,
        # Thermal state (9 fields bundled with explicit FFI version check)
        rust_state_bundle,
        # UMEP 2026a ground scheme (None = classic byte-identical path)
        rust_ground_scheme,
        # Valid pixel mask for early NaN exit
        valid_mask_call,
        output_mask,
    )

    # === Unpack result and update thermal state ===

    state.timeadd = result.timeadd
    if use_crop:
        state.tgmap1[crop_slice] = np.asarray(result.tgmap1)
        state.tgmap1_e[crop_slice] = np.asarray(result.tgmap1_e)
        state.tgmap1_s[crop_slice] = np.asarray(result.tgmap1_s)
        state.tgmap1_w[crop_slice] = np.asarray(result.tgmap1_w)
        state.tgmap1_n[crop_slice] = np.asarray(result.tgmap1_n)
        state.tgout1[crop_slice] = np.asarray(result.tgout1)
    else:
        state.tgmap1 = np.asarray(result.tgmap1)
        state.tgmap1_e = np.asarray(result.tgmap1_e)
        state.tgmap1_s = np.asarray(result.tgmap1_s)
        state.tgmap1_w = np.asarray(result.tgmap1_w)
        state.tgmap1_n = np.asarray(result.tgmap1_n)
        state.tgout1 = np.asarray(result.tgout1)

    if weather.is_daytime:
        state.firstdaytime = 0.0
    else:
        state.firstdaytime = 1.0
        state.timeadd = 0.0

    # Carry the ground-scheme state forward (scheme active → Rust returned
    # the updated force-restore state; shadow_past becomes this shadow).
    if gss is not None:
        assert result.tg is not None, "scheme active but Rust returned no tg"
        gss.tg = np.asarray(result.tg)
        gss.rn = np.asarray(result.rn)
        gss.rn_past = np.asarray(result.rn_past)
        gss.g = np.asarray(result.g)
        gss.shadow_past = np.asarray(result.shadow)

    output_state = state.copy() if return_state_copy else state

    tmrt = np.asarray(result.tmrt)
    shadow = np.asarray(result.shadow) if result.shadow is not None else None
    kdown = np.asarray(result.kdown) if result.kdown is not None else None
    kup = np.asarray(result.kup) if result.kup is not None else None
    ldown = np.asarray(result.ldown) if result.ldown is not None else None
    lup = np.asarray(result.lup) if result.lup is not None else None

    if use_crop:

        def _uncrop(arr: np.ndarray | None) -> np.ndarray | None:
            if arr is None:
                return None
            full = np.full((rows, cols), np.nan, dtype=np.float32)
            full[crop_slice] = arr
            return full

        tmrt = _uncrop(tmrt)
        shadow = _uncrop(shadow)
        kdown = _uncrop(kdown)
        kup = _uncrop(kup)
        ldown = _uncrop(ldown)
        lup = _uncrop(lup)

    assert tmrt is not None  # tmrt is always computed
    return SolweigResult(
        tmrt=tmrt,
        shadow=shadow,
        kdown=kdown,
        kup=kup,
        ldown=ldown,
        lup=lup,
        utci=None,
        pet=None,
        state=output_state,
    )
