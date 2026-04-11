use ndarray::{par_azip, s, Array2, ArrayView2, Zip};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

#[cfg(feature = "gpu")]
use crate::gpu::ShadowGpuContext;
#[cfg(feature = "gpu")]
use std::sync::OnceLock;

// Constants
const PI_OVER_4: f32 = std::f32::consts::FRAC_PI_4;
const THREE_PI_OVER_4: f32 = 3.0 * PI_OVER_4;
const FIVE_PI_OVER_4: f32 = 5.0 * PI_OVER_4;
const SEVEN_PI_OVER_4: f32 = 7.0 * PI_OVER_4;
const TAU: f32 = std::f32::consts::TAU; // 2 * PI
const EPSILON: f32 = 1e-8; // Small value for float comparisons

#[cfg(feature = "gpu")]
static GPU_CONTEXT: OnceLock<Option<ShadowGpuContext>> = OnceLock::new();

#[cfg(feature = "gpu")]
static GPU_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

#[cfg(feature = "gpu")]
pub(crate) fn get_gpu_context() -> Option<&'static ShadowGpuContext> {
    // Check if GPU is enabled
    if !GPU_ENABLED.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }

    GPU_CONTEXT
        .get_or_init(|| match crate::gpu::create_shadow_gpu_context() {
            Ok(ctx) => {
                eprintln!("[GPU] Shadow GPU context initialized successfully");
                Some(ctx)
            }
            Err(e) => {
                eprintln!(
                    "[GPU] Failed to initialize GPU context: {}. Falling back to CPU.",
                    e
                );
                None
            }
        })
        .as_ref()
}

#[cfg(feature = "gpu")]
#[pyfunction]
/// Enable GPU acceleration for shadow calculations
pub fn enable_gpu() {
    GPU_ENABLED.store(true, std::sync::atomic::Ordering::Relaxed);
    eprintln!("[GPU] GPU acceleration enabled");
}

#[cfg(feature = "gpu")]
#[pyfunction]
/// Disable GPU acceleration for shadow calculations (use CPU only)
pub fn disable_gpu() {
    GPU_ENABLED.store(false, std::sync::atomic::Ordering::Relaxed);
    eprintln!("[GPU] GPU acceleration disabled - using CPU");
}

#[cfg(feature = "gpu")]
#[pyfunction]
/// Check if GPU acceleration is currently enabled
pub fn is_gpu_enabled() -> bool {
    GPU_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(feature = "gpu")]
#[pyfunction]
/// Return GPU limits as a dict, or None if GPU is unavailable.
///
/// Keys:
///   - "max_buffer_size": u64 — raw adapter-reported maximum buffer size
///   - "backend": str — GPU backend name ("Metal", "Vulkan", "Dx12", "Gl", etc.)
///   - "gpu_memory_budget": u64 — resolved GPU memory budget (only present when detectable)
///
/// Initialises the GPU context lazily on first call.
pub fn gpu_limits(py: Python<'_>) -> PyResult<Option<PyObject>> {
    let ctx = match get_gpu_context() {
        Some(c) => c,
        None => return Ok(None),
    };
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("max_buffer_size", ctx.max_buffer_size)?;
    dict.set_item("backend", format!("{:?}", ctx.backend))?;
    if let Some(budget) = ctx.gpu_memory_budget {
        dict.set_item("gpu_memory_budget", budget)?;
    }
    Ok(Some(dict.into()))
}

/// Rust-native result struct for internal shadow calculations.
pub(crate) struct ShadowingResultRust {
    pub bldg_sh: Array2<f32>,
    pub veg_sh: Array2<f32>,
    pub veg_blocks_bldg_sh: Array2<f32>,
    pub wall_sh: Option<Array2<f32>>,
    pub wall_sun: Option<Array2<f32>>,
    pub wall_sh_veg: Option<Array2<f32>>,
    pub face_sh: Option<Array2<f32>>,
    pub face_sun: Option<Array2<f32>>,
    pub sh_on_wall: Option<Array2<f32>>,
}

#[pyclass]
/// Result of the shadowing function, containing all output shadow maps (Python version).
pub struct ShadowingResult {
    #[pyo3(get)]
    pub bldg_sh: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub veg_sh: Py<PyArray2<f32>>,
    #[pyo3(get)]
    /// Vegetation Blocks Building Shadow: Indicates where vegetation prevents building shadow.
    pub veg_blocks_bldg_sh: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub wall_sh: Option<Py<PyArray2<f32>>>, // Shadowed wall height (by buildings) - Optional
    #[pyo3(get)]
    pub wall_sun: Option<Py<PyArray2<f32>>>, // Sunlit wall height - Optional
    #[pyo3(get)]
    pub wall_sh_veg: Option<Py<PyArray2<f32>>>, // Wall height shadowed by vegetation - Optional
    #[pyo3(get)]
    pub face_sh: Option<Py<PyArray2<f32>>>, // Wall face shadow mask (1 if face away from sun) - Optional
    #[pyo3(get)]
    pub face_sun: Option<Py<PyArray2<f32>>>, // Sunlit wall face mask (1 if face towards sun and not obstructed) - Optional
    #[pyo3(get)]
    /// Combined building and vegetation shadow height on walls (optional scheme).
    pub sh_on_wall: Option<Py<PyArray2<f32>>>,
}

/// Internal Rust function for shadow calculations.
/// Operates purely on ndarray types.
#[allow(clippy::too_many_arguments)]
pub(crate) fn calculate_shadows_rust(
    azimuth_deg: f32,
    altitude_deg: f32,
    scale: f32,
    max_local_dsm_ht: f32,
    dsm_view: ArrayView2<f32>,
    veg_canopy_dsm_view_opt: Option<ArrayView2<f32>>,
    veg_trunk_dsm_view_opt: Option<ArrayView2<f32>>,
    bush_view_opt: Option<ArrayView2<f32>>,
    walls_view_opt: Option<ArrayView2<f32>>,
    aspect_view_opt: Option<ArrayView2<f32>>,
    walls_scheme_view_opt: Option<ArrayView2<f32>>,
    aspect_scheme_view_opt: Option<ArrayView2<f32>>,
    need_full_wall_outputs: bool,
    min_sun_elev_deg: f32,
    max_shadow_distance_m: f32,
) -> ShadowingResultRust {
    let shape = dsm_view.shape();
    let num_rows = shape[0];
    let num_cols = shape[1];
    let dim = (num_rows, num_cols);

    // Handle zenith case (altitude >= 89.5°): no shadows cast from directly overhead.
    // This avoids tan(90°) = infinity which breaks the shadow propagation loop.
    // For SVF calculations, zenith patches represent looking straight up - all points
    // can see the sky in this direction (no obstruction).
    if altitude_deg >= 89.5 {
        return ShadowingResultRust {
            bldg_sh: Array2::<f32>::ones(dim),
            veg_sh: Array2::<f32>::ones(dim),
            veg_blocks_bldg_sh: Array2::<f32>::ones(dim),
            wall_sh: if need_full_wall_outputs {
                walls_view_opt.map(|_| Array2::<f32>::zeros(dim))
            } else {
                None
            },
            wall_sun: walls_view_opt.map(|w| w.to_owned()),
            wall_sh_veg: if need_full_wall_outputs {
                walls_view_opt.map(|_| Array2::<f32>::zeros(dim))
            } else {
                None
            },
            face_sh: if need_full_wall_outputs {
                walls_view_opt.map(|_| Array2::<f32>::zeros(dim))
            } else {
                None
            },
            face_sun: if need_full_wall_outputs {
                walls_view_opt.map(|w| w.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }))
            } else {
                None
            },
            sh_on_wall: walls_scheme_view_opt.map(|_| Array2::<f32>::zeros(dim)),
        };
    }

    // GPU acceleration path: use GPU if available for all shadow types
    #[cfg(feature = "gpu")]
    {
        if let Some(gpu_ctx) = get_gpu_context() {
            // Pass ArrayView directly - GPU will handle conversion efficiently
            match gpu_ctx.compute_all_shadows_view(
                dsm_view,
                veg_canopy_dsm_view_opt,
                veg_trunk_dsm_view_opt,
                bush_view_opt,
                walls_view_opt,
                aspect_view_opt,
                walls_scheme_view_opt.is_some() && aspect_scheme_view_opt.is_some(),
                need_full_wall_outputs,
                azimuth_deg,
                altitude_deg,
                scale,
                max_local_dsm_ht,
                min_sun_elev_deg,
                max_shadow_distance_m,
            ) {
                Ok(gpu_result) => {
                    // Handle sh_on_wall if wall scheme is present
                    let sh_on_wall = if let (Some(walls_scheme_view), Some(aspect_scheme_view)) =
                        (walls_scheme_view_opt, aspect_scheme_view_opt)
                    {
                        // Need to compute scheme-based wall shadows on CPU for now
                        // since it requires a second set of walls/aspect inputs
                        if let (Some(ref _bldg_sh), Some(ref _veg_sh)) =
                            (&Some(gpu_result.bldg_sh.clone()), &gpu_result.veg_sh)
                        {
                            // Create propagated heights from GPU results
                            let mut prop_bldg_h = Array2::<f32>::zeros(dim);
                            prop_bldg_h.assign(&dsm_view);

                            let prop_veg_h = gpu_result
                                .propagated_veg_height
                                .clone()
                                .unwrap_or_else(|| Array2::<f32>::zeros(dim));

                            let (scheme_wall_sh, _, scheme_wall_sh_veg, _, _) = shade_on_walls(
                                azimuth_deg.to_radians(),
                                aspect_scheme_view,
                                walls_scheme_view,
                                dsm_view,
                                prop_bldg_h.view(),
                                prop_veg_h.view(),
                            );

                            let mut sh_on_wall_combined =
                                Array2::<f32>::zeros(scheme_wall_sh.dim());
                            Zip::from(&mut sh_on_wall_combined)
                                .and(&scheme_wall_sh)
                                .and(&scheme_wall_sh_veg)
                                .par_for_each(|sow, &wsh, &wsv| *sow = f32::max(wsh, wsv));
                            Some(sh_on_wall_combined)
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    return ShadowingResultRust {
                        bldg_sh: gpu_result.bldg_sh,
                        veg_sh: gpu_result
                            .veg_sh
                            .unwrap_or_else(|| Array2::<f32>::ones(dim)),
                        veg_blocks_bldg_sh: gpu_result
                            .veg_blocks_bldg_sh
                            .unwrap_or_else(|| Array2::<f32>::ones(dim)),
                        wall_sh: gpu_result.wall_sh,
                        wall_sun: gpu_result.wall_sun,
                        wall_sh_veg: gpu_result.wall_sh_veg,
                        face_sh: gpu_result.face_sh,
                        face_sun: gpu_result.face_sun,
                        sh_on_wall,
                    };
                }
                Err(e) => {
                    eprintln!(
                        "[GPU] GPU shadow calculation failed: {}. Falling back to CPU.",
                        e
                    );
                    // Fall through to CPU path
                }
            }
        }
    }

    // Determine if all vegetation inputs are present
    let veg_inputs_present = veg_canopy_dsm_view_opt.is_some()
        && veg_trunk_dsm_view_opt.is_some()
        && bush_view_opt.is_some();

    // Allocate arrays for vegetation only if all inputs are present
    let (mut veg_sh, mut veg_blocks_bldg_sh, mut propagated_veg_sh_height) =
        if let (Some(bush_view), Some(veg_canopy_dsm_view)) =
            (bush_view_opt, veg_canopy_dsm_view_opt)
        {
            (
                bush_view.mapv(|v| if v > 1.0 { 1.0 } else { 0.0 }),
                Array2::<f32>::zeros(dim),
                {
                    let mut arr = Array2::<f32>::zeros(dim);
                    arr.assign(&veg_canopy_dsm_view);
                    arr
                },
            )
        } else {
            (
                Array2::<f32>::zeros(dim),
                Array2::<f32>::zeros(dim),
                Array2::<f32>::zeros(dim),
            )
        };

    let mut bldg_sh = Array2::<f32>::zeros(dim);
    let mut propagated_bldg_sh_height = Array2::<f32>::zeros(dim);
    propagated_bldg_sh_height.assign(&dsm_view);

    let azimuth_rad = azimuth_deg.to_radians();
    let altitude_rad = altitude_deg.to_radians();
    let sin_azimuth = azimuth_rad.sin();
    let cos_azimuth = azimuth_rad.cos();
    let tan_azimuth = azimuth_rad.tan();
    let sign_sin_azimuth = sin_azimuth.signum();
    let sign_cos_azimuth = cos_azimuth.signum();
    let ds_sin = (1.0 / sin_azimuth).abs();
    let ds_cos = (1.0 / cos_azimuth).abs();
    // NOTE on `scale`: this convention has been "pixel size in metres" since
    // the first Rust port — i.e. solweig passes `scale = pixel_size_m`. This
    // is the *opposite* of UMEP's upstream convention (`scale = 1/px` per
    // `umep.util.SEBESOLWEIGCommonFiles.shadowingfunction_wallheight_23:53`).
    // We keep solweig's convention because all spec tests, Python callers,
    // and the `max_radius_pixels = max_reach_m / scale` formula below depend
    // on it; only the vertical-drop math needs a compensating `* scale` here
    // so that `dz = ds*index*tan(α)*pixel_size` is in metres.
    let tan_altitude_by_scale = altitude_rad.tan() * scale;
    let mut dx: f32 = 0.0;
    let mut dy: f32 = 0.0;
    let mut dz: f32 = 0.0;
    let mut prev_dz: f32 = 0.0;
    let mut ds: f32;
    let mut index = 0.0;

    // Horizontal reach: derived from height, optionally capped by max_shadow_distance_m.
    // On mountainous terrain the height-derived reach can be huge, so the distance
    // cap prevents tracing rays for kilometres while the dz guard (below) still
    // uses the full terrain relief for correct vertical cutoff.
    let min_sun_elev_rad = min_sun_elev_deg.to_radians();
    let height_reach_m = max_local_dsm_ht / min_sun_elev_rad.tan();
    let max_reach_m = if max_shadow_distance_m > 0.0 {
        height_reach_m.min(max_shadow_distance_m)
    } else {
        height_reach_m
    };
    let max_radius_pixels = (max_reach_m / scale).ceil() as usize;
    let max_index = max_radius_pixels as f32; // index uses f32

    // while condition:
    while index <= max_index
        && max_local_dsm_ht >= dz
        && dx.abs() < num_rows as f32
        && dy.abs() < num_cols as f32
    {
        if (PI_OVER_4..THREE_PI_OVER_4).contains(&azimuth_rad)
            || (FIVE_PI_OVER_4..SEVEN_PI_OVER_4).contains(&azimuth_rad)
        {
            dy = sign_sin_azimuth * index;
            dx = -1.0 * sign_cos_azimuth * (index / tan_azimuth).round().abs();
            ds = ds_sin;
        } else {
            dy = sign_sin_azimuth * (index * tan_azimuth).round().abs();
            dx = -1.0 * sign_cos_azimuth * index;
            ds = ds_cos;
        }
        dz = (ds * index) * tan_altitude_by_scale;

        // --- Slicing logic to operate only on overlapping regions ---
        let absdx = dx.abs();
        let absdy = dy.abs();
        let xc1 = ((dx + absdx) / 2.0) as isize;
        let xc2 = (num_rows as f32 + (dx - absdx) / 2.0) as isize;
        let yc1 = ((dy + absdy) / 2.0) as isize;
        let yc2 = (num_cols as f32 + (dy - absdy) / 2.0) as isize;
        let xp1 = -((dx - absdx) / 2.0) as isize;
        let xp2 = (num_rows as f32 - (dx + absdx) / 2.0) as isize;
        let yp1 = -((dy - absdy) / 2.0) as isize;
        let yp2 = (num_cols as f32 - (dy + absdy) / 2.0) as isize;

        // Clamp indices to valid ranges
        let xc1c = xc1.max(0).min(num_rows as isize) as usize;
        let xc2c = xc2.max(0).min(num_rows as isize) as usize;
        let yc1c = yc1.max(0).min(num_cols as isize) as usize;
        let yc2c = yc2.max(0).min(num_cols as isize) as usize;
        let xp1c = xp1.max(0).min(num_rows as isize) as usize;
        let xp2c = xp2.max(0).min(num_rows as isize) as usize;
        let yp1c = yp1.max(0).min(num_cols as isize) as usize;
        let yp2c = yp2.max(0).min(num_cols as isize) as usize;

        if xc2c > xc1c && yc2c > yc1c && xp2c > xp1c && yp2c > yp1c {
            let xlen = xc2c - xc1c;
            let ylen = yc2c - yc1c;
            let xplen = xp2c - xp1c;
            let yplen = yp2c - yp1c;
            let minx = xlen.min(xplen);
            let miny = ylen.min(yplen);

            // Building shadow calculation on the slice
            let dsm_src_slice = dsm_view.slice(s![xc1c..xc1c + minx, yc1c..yc1c + miny]);
            let mut prop_bldg_h_dst_slice =
                propagated_bldg_sh_height.slice_mut(s![xp1c..xp1c + minx, yp1c..yp1c + miny]);
            let dsm_dst_slice = dsm_view.slice(s![xp1c..xp1c + minx, yp1c..yp1c + miny]);
            let mut bldg_sh_dst_slice = bldg_sh.slice_mut(s![xp1c..xp1c + minx, yp1c..yp1c + miny]);

            par_azip!((prop_h in &mut prop_bldg_h_dst_slice, &dsm_src in &dsm_src_slice) {
                if dsm_src.is_finite() {
                    let shifted_dsm = dsm_src - dz;
                    *prop_h = prop_h.max(shifted_dsm);
                }
            });

            par_azip!((bldg_sh_flag in &mut bldg_sh_dst_slice, &prop_h in &prop_bldg_h_dst_slice, &dsm_target in &dsm_dst_slice) {
                if dsm_target.is_finite() {
                    *bldg_sh_flag = if prop_h > dsm_target { 1.0 } else { 0.0 };
                } else {
                    *bldg_sh_flag = f32::NAN;
                }
            });

            // Vegetation shadow calculation on the slice
            if let (Some(veg_canopy_dsm_view), Some(veg_trunk_dsm_view)) =
                (veg_canopy_dsm_view_opt, veg_trunk_dsm_view_opt)
            {
                let veg_canopy_src_slice =
                    veg_canopy_dsm_view.slice(s![xc1c..xc1c + minx, yc1c..yc1c + miny]);
                let mut prop_veg_h_dst_slice =
                    propagated_veg_sh_height.slice_mut(s![xp1c..xp1c + minx, yp1c..yp1c + miny]);

                par_azip!((prop_veg_h in &mut prop_veg_h_dst_slice, &source_veg_canopy in &veg_canopy_src_slice) {
                    if source_veg_canopy.is_finite() {
                        let shifted_veg_canopy = source_veg_canopy - dz;
                        *prop_veg_h = prop_veg_h.max(shifted_veg_canopy);
                    }
                });

                let veg_trunk_src_slice =
                    veg_trunk_dsm_view.slice(s![xc1c..xc1c + minx, yc1c..yc1c + miny]);
                let mut veg_sh_dst_slice =
                    veg_sh.slice_mut(s![xp1c..xp1c + minx, yp1c..yp1c + miny]);

                par_azip!((
                    veg_sh_flag in &mut veg_sh_dst_slice,
                    &dsm_h_target in &dsm_dst_slice,
                    &source_veg_canopy in &veg_canopy_src_slice,
                    &source_veg_trunk in &veg_trunk_src_slice
                ) {
                    let shifted_veg_canopy = source_veg_canopy - dz;
                    let shifted_veg_trunk = source_veg_trunk - dz;
                    let prev_shifted_veg_canopy = source_veg_canopy - prev_dz;
                    let prev_shifted_veg_trunk = source_veg_trunk - prev_dz;

                    let cond1 = if shifted_veg_canopy > dsm_h_target { 1.0 } else { 0.0 };
                    let cond2 = if shifted_veg_trunk > dsm_h_target { 1.0 } else { 0.0 };
                    let cond3 = if prev_shifted_veg_canopy > dsm_h_target { 1.0 } else { 0.0 };
                    let cond4 = if prev_shifted_veg_trunk > dsm_h_target { 1.0 } else { 0.0 };
                    let conditions_sum = cond1 + cond2 + cond3 + cond4;
                    let pergola_shadow = if conditions_sum > 0.0 && conditions_sum < 4.0 { 1.0 } else { 0.0 };
                    *veg_sh_flag = f32::max(*veg_sh_flag, pergola_shadow);
                });

                // Clear building shadow from veg_sh BEFORE accumulating veg_blocks_bldg_sh
                let bldg_sh_dst_slice_ro = bldg_sh.slice(s![xp1c..xp1c + minx, yp1c..yp1c + miny]);
                let mut veg_sh_dst_slice_rw =
                    veg_sh.slice_mut(s![xp1c..xp1c + minx, yp1c..yp1c + miny]);

                par_azip!((veg_sh_flag in &mut veg_sh_dst_slice_rw, &bldg_sh_flag in &bldg_sh_dst_slice_ro) {
                    if *veg_sh_flag > 0.0 && bldg_sh_flag > 0.0 {
                        *veg_sh_flag = 0.0;
                    }
                });

                // Now accumulate veg_blocks_bldg_sh using the cleared veg_sh (matches Python order)
                let veg_sh_dst_slice_ro = veg_sh.slice(s![xp1c..xp1c + minx, yp1c..yp1c + miny]);
                let mut veg_blocks_bldg_sh_dst_slice =
                    veg_blocks_bldg_sh.slice_mut(s![xp1c..xp1c + minx, yp1c..yp1c + miny]);

                par_azip!((vbs_acc in &mut veg_blocks_bldg_sh_dst_slice, &veg_sh_flag in &veg_sh_dst_slice_ro) {
                    *vbs_acc += veg_sh_flag;
                });
            }
        }

        prev_dz = dz;
        index += 1.0;
    }

    bldg_sh.par_mapv_inplace(|v| 1.0 - v);

    if veg_inputs_present {
        // Process veg_blocks_bldg_sh to match Python logic exactly:
        // Python: vbshvegsh[vbshvegsh > 0] = 1  (binarize)
        //         vbshvegsh = vbshvegsh - vegsh  (subtract, vegsh is 0 or 1 from fmax)
        //         vbshvegsh = 1 - vbshvegsh      (invert)

        // 1. Binarize accumulated veg_blocks_bldg_sh
        veg_blocks_bldg_sh.par_mapv_inplace(|v| if v > 0.0 { 1.0 } else { 0.0 });

        // 2. Subtract veg_sh (NO clamping - Python doesn't clamp here)
        veg_blocks_bldg_sh = &veg_blocks_bldg_sh - &veg_sh;

        // 3. Invert (Python: vbshvegsh = 1 - vbshvegsh)
        // If subtraction was negative, result will be > 1, which gets clamped implicitly in final usage
        veg_blocks_bldg_sh.par_mapv_inplace(|v| 1.0 - v); // Now process veg_sh: binarize and invert
        veg_sh.par_mapv_inplace(|v| if v > 0.0 { 1.0 } else { 0.0 });
        veg_sh.par_mapv_inplace(|v| 1.0 - v);
        let final_veg_sh_mask = veg_sh.mapv(|v| 1.0 - v);
        Zip::from(&mut propagated_veg_sh_height)
            .and(&dsm_view)
            .and(&final_veg_sh_mask)
            .par_for_each(|prop_veg_h, &dsm_h, &mask| {
                *prop_veg_h = (*prop_veg_h - dsm_h).max(0.0) * mask;
            });
    }

    let mut wall_sh: Option<Array2<f32>> = None;
    let mut wall_sun: Option<Array2<f32>> = None;
    let mut wall_sh_veg: Option<Array2<f32>> = None;
    let mut face_sh: Option<Array2<f32>> = None;
    let mut face_sun: Option<Array2<f32>> = None;
    let mut sh_on_wall: Option<Array2<f32>> = None;

    if let (Some(walls_view), Some(aspect_view)) = (walls_view_opt, aspect_view_opt) {
        let (wall_sh_calc, wall_sun_calc, wall_sh_veg_calc, face_sh_calc, face_sun_calc) =
            shade_on_walls(
                azimuth_rad,
                aspect_view,
                walls_view,
                dsm_view,
                propagated_bldg_sh_height.view(),
                propagated_veg_sh_height.view(),
            );
        wall_sh = Some(wall_sh_calc);
        wall_sun = Some(wall_sun_calc);
        wall_sh_veg = Some(wall_sh_veg_calc);
        face_sh = Some(face_sh_calc);
        face_sun = Some(face_sun_calc);

        if let (Some(walls_scheme_view), Some(aspect_scheme_view)) =
            (walls_scheme_view_opt, aspect_scheme_view_opt)
        {
            let (
                scheme_wall_sh,
                _scheme_wall_sun,
                scheme_wall_sh_veg,
                _scheme_face_sh,
                _scheme_face_sun,
            ) = shade_on_walls(
                azimuth_rad,
                aspect_scheme_view,
                walls_scheme_view,
                dsm_view,
                propagated_bldg_sh_height.view(),
                propagated_veg_sh_height.view(),
            );
            let mut sh_on_wall_combined = Array2::<f32>::zeros(scheme_wall_sh.dim());
            Zip::from(&mut sh_on_wall_combined)
                .and(&scheme_wall_sh)
                .and(&scheme_wall_sh_veg)
                .par_for_each(|sow, &wsh, &wsv| *sow = f32::max(wsh, wsv));
            sh_on_wall = Some(sh_on_wall_combined);
        }
    }

    ShadowingResultRust {
        veg_sh,
        bldg_sh,
        veg_blocks_bldg_sh,
        wall_sh,
        wall_sun,
        wall_sh_veg,
        face_sh,
        face_sun,
        sh_on_wall,
    }
}

fn shade_on_walls(
    azimuth: f32,
    aspect: ArrayView2<f32>,
    walls: ArrayView2<f32>,
    dsm: ArrayView2<f32>,
    propagated_bldg_sh_height: ArrayView2<f32>,
    propagated_veg_sh_height: ArrayView2<f32>,
) -> (
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
) {
    let shape = walls.dim();
    let mut wall_mask = Array2::<f32>::zeros(shape);
    Zip::from(&mut wall_mask)
        .and(&walls)
        .par_for_each(|mask_val, &wall_h| *mask_val = if wall_h > 0.0 { 1.0 } else { 0.0 });

    let azimuth_low = azimuth - std::f32::consts::FRAC_PI_2;
    let azimuth_high = azimuth + std::f32::consts::FRAC_PI_2;
    let mut face_sh = Array2::<f32>::zeros(shape);
    if azimuth_low >= 0.0 && azimuth_high < TAU {
        Zip::from(&mut face_sh)
            .and(aspect)
            .and(&wall_mask)
            .par_for_each(|f_sh, &asp, &w_mask| {
                *f_sh = if asp < azimuth_low || asp >= azimuth_high {
                    1.0
                } else {
                    0.0
                } - w_mask
                    + 1.0;
            });
    } else if azimuth_low < 0.0 && azimuth_high <= TAU {
        let azimuth_low_wrapped = azimuth_low + TAU;
        Zip::from(&mut face_sh)
            .and(aspect)
            .par_for_each(|f_sh, &asp| {
                *f_sh = if asp > azimuth_low_wrapped || asp <= azimuth_high {
                    -1.0
                } else {
                    0.0
                } + 1.0;
            });
    } else {
        let azimuth_high_wrapped = azimuth_high - TAU;
        Zip::from(&mut face_sh)
            .and(aspect)
            .par_for_each(|f_sh, &asp| {
                *f_sh = if asp > azimuth_low || asp <= azimuth_high_wrapped {
                    -1.0
                } else {
                    0.0
                } + 1.0;
            });
    }

    let mut bldg_sh_vol_height = Array2::<f32>::zeros(shape);
    Zip::from(&mut bldg_sh_vol_height)
        .and(&propagated_bldg_sh_height)
        .and(&dsm)
        .par_for_each(|sh_vol, &prop_h, &dsm_h| *sh_vol = prop_h - dsm_h);

    let mut face_sun = Array2::<f32>::zeros(shape);
    Zip::from(&mut face_sun)
        .and(&face_sh)
        .and(walls)
        .par_for_each(|sf_mask, &f_sh_mask, &w_h| {
            let wall_exists = w_h > 0.0;
            let wall_exists_flag = if wall_exists { 1.0 } else { 0.0 };
            *sf_mask = if (f_sh_mask + wall_exists_flag) == 1.0 && wall_exists {
                1.0
            } else {
                0.0
            };
        });

    let mut wall_sun = Array2::<f32>::zeros(shape);
    Zip::from(&mut wall_sun)
        .and(&walls)
        .and(&bldg_sh_vol_height)
        .par_for_each(|sun_h, &wall_h, &sh_vol_h| *sun_h = wall_h - sh_vol_h);
    wall_sun.par_mapv_inplace(|v| v.max(0.0));
    Zip::from(&mut wall_sun)
        .and(&face_sh)
        .par_for_each(|sun_h, &f_sh_mask| {
            if (f_sh_mask - 1.0).abs() < EPSILON {
                *sun_h = 0.0
            }
        });

    let mut wall_sh = Array2::<f32>::zeros(shape);
    Zip::from(&mut wall_sh)
        .and(&walls)
        .and(&wall_sun)
        .par_for_each(|sh_h, &wall_h, &sun_h| *sh_h = wall_h - sun_h);

    let mut wall_sh_veg = Array2::<f32>::zeros(shape);
    Zip::from(&mut wall_sh_veg)
        .and(&propagated_veg_sh_height)
        .and(&wall_mask)
        .par_for_each(|veg_sh_h, &prop_veg_h, &w_mask| *veg_sh_h = prop_veg_h * w_mask);
    Zip::from(&mut wall_sh_veg)
        .and(&wall_sh)
        .par_for_each(|veg_sh_h, &bldg_sh_h| *veg_sh_h -= bldg_sh_h);
    wall_sh_veg.par_mapv_inplace(|v| v.max(0.0));
    Zip::from(&mut wall_sh_veg)
        .and(walls)
        .par_for_each(|veg_sh_h, &wall_h| {
            if *veg_sh_h > wall_h {
                *veg_sh_h = wall_h
            }
        });
    Zip::from(&mut wall_sun)
        .and(&wall_sh_veg)
        .par_for_each(|sun_h, &veg_sh_h| *sun_h -= veg_sh_h);
    Zip::from(&mut wall_sh_veg)
        .and(&wall_sun)
        .par_for_each(|veg_sh_h, &sun_h| {
            if sun_h < 0.0 {
                *veg_sh_h = 0.0
            }
        });
    wall_sun.par_mapv_inplace(|v| v.max(0.0));

    (wall_sh, wall_sun, wall_sh_veg, face_sh, face_sun)
}

#[pyfunction]
#[pyo3(signature = (azimuth_deg, altitude_deg, scale, max_local_dsm_ht, dsm, veg_canopy_dsm=None, veg_trunk_dsm=None, bush=None, walls=None, aspect=None, walls_scheme=None, aspect_scheme=None, min_sun_elev_deg=None, max_shadow_distance_m=None))]
/// Calculates shadow maps for buildings, vegetation, and walls given DSM and sun position (Python wrapper).
///
/// This function handles Python type conversions and calls the internal Rust shadow calculation logic.
/// See `calculate_shadows_rust` for core algorithm details.
///
/// Vegetation arguments are optional and only processed if all three are provided.
///
/// # Arguments
/// * `dsm` - Digital Surface Model (buildings, ground)
/// * `veg_canopy_dsm` - Optional: Vegetation canopy height DSM
/// * `veg_trunk_dsm` - Optional: Vegetation trunk height DSM (defines bottom of canopy)
/// * `azimuth_deg` - Sun azimuth in degrees (0=N, 90=E, 180=S, 270=W)
/// * `altitude_deg` - Sun altitude/elevation in degrees (0=horizon, 90=zenith)
/// * `scale` - Pixel size (meters)
/// * `max_local_dsm_ht` - Maximum local DSM height (optimization hint)
/// * `bush` - Optional: Bush/low vegetation layer (binary or height)
/// * `walls` - Optional wall height layer. If None, wall calculations are skipped.
/// * `aspect` - Optional wall aspect/orientation layer (radians or degrees). Required if `walls` is provided.
/// * `walls_scheme` - Optional alternative wall height layer for specific calculations
/// * `aspect_scheme` - Optional alternative wall aspect layer
/// * `max_shadow_distance_m` - Optional: Maximum horizontal shadow distance in metres (0 = no cap)
///
/// # Returns
/// * `ShadowingResult` struct containing various shadow maps (ground, vegetation, walls) as PyArrays.
pub fn calculate_shadows_wall_ht_25(
    py: Python,
    azimuth_deg: f32,
    altitude_deg: f32,
    scale: f32,
    max_local_dsm_ht: f32,
    dsm: PyReadonlyArray2<f32>,
    veg_canopy_dsm: Option<PyReadonlyArray2<f32>>,
    veg_trunk_dsm: Option<PyReadonlyArray2<f32>>,
    bush: Option<PyReadonlyArray2<f32>>,
    walls: Option<PyReadonlyArray2<f32>>,
    aspect: Option<PyReadonlyArray2<f32>>,
    walls_scheme: Option<PyReadonlyArray2<f32>>,
    aspect_scheme: Option<PyReadonlyArray2<f32>>,
    min_sun_elev_deg: Option<f32>,
    max_shadow_distance_m: Option<f32>,
) -> PyResult<PyObject> {
    let dsm_view = dsm.as_array();
    let shape = dsm_view.shape();

    // --- Vegetation Input Validation ---
    let veg_inputs_provided = [
        veg_canopy_dsm.is_some(),
        veg_trunk_dsm.is_some(),
        bush.is_some(),
    ];
    let num_veg_inputs = veg_inputs_provided.iter().filter(|&&x| x).count();

    let (veg_canopy_dsm_view_opt, veg_trunk_dsm_view_opt, bush_view_opt) = if num_veg_inputs == 3 {
        let veg_canopy_view = veg_canopy_dsm
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("veg_canopy_dsm is missing"))?
            .as_array();
        let veg_trunk_view = veg_trunk_dsm
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("veg_trunk_dsm is missing"))?
            .as_array();
        let bush_view = bush
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("bush is missing"))?
            .as_array();
        if veg_canopy_view.shape() != shape {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "veg_canopy_dsm must have the same shape as dsm.",
            ));
        }
        if veg_trunk_view.shape() != shape {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "veg_trunk_dsm must have the same shape as dsm.",
            ));
        }
        if bush_view.shape() != shape {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "bush must have the same shape as dsm.",
            ));
        }
        (Some(veg_canopy_view), Some(veg_trunk_view), Some(bush_view))
    } else if num_veg_inputs == 0 {
        (None, None, None)
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
                "Either all vegetation inputs (veg_canopy_dsm, veg_trunk_dsm, bush) must be provided, or none of them.",
            ));
    };

    // --- Wall Input Validation ---
    let walls_view_opt = walls.as_ref().map(|w| w.as_array());
    let aspect_view_opt = aspect.as_ref().map(|a| a.as_array());
    if walls_view_opt.is_some() != aspect_view_opt.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Both 'walls' and 'aspect' must be provided together, or both must be None.",
        ));
    }
    if let Some(walls_view) = walls_view_opt {
        if walls_view.shape() != shape {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "walls must have the same shape as dsm.",
            ));
        }
    }
    if let Some(aspect_view) = aspect_view_opt {
        if aspect_view.shape() != shape {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "aspect must have the same shape as dsm.",
            ));
        }
    }
    let walls_scheme_view_opt = walls_scheme.as_ref().map(|w| w.as_array());
    let aspect_scheme_view_opt = aspect_scheme.as_ref().map(|a| a.as_array());
    if walls_scheme_view_opt.is_some() != aspect_scheme_view_opt.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Both 'walls_scheme' and 'aspect_scheme' must be provided together, or both must be None.",
        ));
    }
    if let Some(walls_scheme_view) = walls_scheme_view_opt {
        if walls_scheme_view.shape() != shape {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "walls_scheme must have the same shape as dsm.",
            ));
        }
    }
    if let Some(aspect_scheme_view) = aspect_scheme_view_opt {
        if aspect_scheme_view.shape() != shape {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "aspect_scheme must have the same shape as dsm.",
            ));
        }
    }

    let rust_result = calculate_shadows_rust(
        azimuth_deg,
        altitude_deg,
        scale,
        max_local_dsm_ht,
        dsm_view,
        veg_canopy_dsm_view_opt,
        veg_trunk_dsm_view_opt,
        bush_view_opt,
        walls_view_opt,
        aspect_view_opt,
        walls_scheme_view_opt,
        aspect_scheme_view_opt,
        true,
        min_sun_elev_deg.unwrap_or(3.0_f32),
        max_shadow_distance_m.unwrap_or(0.0_f32),
    );

    let py_result = ShadowingResult {
        veg_sh: rust_result.veg_sh.into_pyarray(py).to_owned().into(),
        bldg_sh: rust_result.bldg_sh.into_pyarray(py).to_owned().into(),
        veg_blocks_bldg_sh: rust_result
            .veg_blocks_bldg_sh
            .into_pyarray(py)
            .to_owned()
            .into(),
        wall_sh: rust_result
            .wall_sh
            .map(|arr| arr.into_pyarray(py).to_owned().into()),
        wall_sun: rust_result
            .wall_sun
            .map(|arr| arr.into_pyarray(py).to_owned().into()),
        wall_sh_veg: rust_result
            .wall_sh_veg
            .map(|arr| arr.into_pyarray(py).to_owned().into()),
        face_sh: rust_result
            .face_sh
            .map(|arr| arr.into_pyarray(py).to_owned().into()),
        face_sun: rust_result
            .face_sun
            .map(|arr| arr.into_pyarray(py).to_owned().into()),
        sh_on_wall: rust_result
            .sh_on_wall
            .map(|arr| arr.into_pyarray(py).to_owned().into()),
    };

    py_result
        .into_pyobject(py)
        .map(|bound| bound.unbind().into())
}
