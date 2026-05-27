use core::f32;
use ndarray::{s, Array2, Array3, ArrayView2, Zip};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2};
use pyo3::prelude::*;
#[cfg(feature = "gpu")]
use std::collections::VecDeque;
use std::f32::consts::PI;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

// Import the correct result struct from shadowing
use crate::shadowing::{calculate_shadows_rust, ShadowingResultRust};

/// Compute shadows for SVF: uses GPU-optimized path when available, CPU fallback otherwise.
///
/// The GPU path skips the wall shader and copies only 3 arrays (instead of 10),
/// saving ~70% staging bandwidth per patch.
fn compute_svf_shadows(
    dsm: ArrayView2<f32>,
    veg_canopy: Option<ArrayView2<f32>>,
    veg_trunk: Option<ArrayView2<f32>>,
    bush: Option<ArrayView2<f32>>,
    azimuth: f32,
    altitude: f32,
    scale: f32,
    max_dsm_ht: f32,
    min_sun_elev: f32,
    max_shadow_distance_m: f32,
) -> ShadowingResultRust {
    #[cfg(feature = "gpu")]
    {
        if let Some(gpu_ctx) = crate::shadowing::get_gpu_context() {
            match gpu_ctx.compute_shadows_for_svf(
                dsm,
                veg_canopy,
                veg_trunk,
                bush,
                azimuth,
                altitude,
                scale,
                max_dsm_ht,
                min_sun_elev,
                max_shadow_distance_m,
            ) {
                Ok(r) => {
                    crate::shadowing::record_gpu_dispatch();
                    let dim = dsm.dim();
                    return ShadowingResultRust {
                        bldg_sh: r.bldg_sh,
                        veg_sh: r.veg_sh.unwrap_or_else(|| Array2::ones(dim)),
                        veg_blocks_bldg_sh: r
                            .veg_blocks_bldg_sh
                            .unwrap_or_else(|| Array2::ones(dim)),
                        wall_sh: None,
                        wall_sun: None,
                        wall_sh_veg: None,
                        face_sh: None,
                        face_sun: None,
                        sh_on_wall: None,
                    };
                }
                Err(e) => {
                    eprintln!("[GPU] SVF shadow failed: {}. Falling back to CPU.", e);
                    crate::shadowing::record_gpu_fallback();
                }
            }
        }
    }

    // CPU fallback
    calculate_shadows_rust(
        azimuth,
        altitude,
        scale,
        max_dsm_ht,
        dsm,
        veg_canopy,
        veg_trunk,
        bush,
        None,
        None,
        None,
        None,
        false,
        min_sun_elev,
        max_shadow_distance_m,
    )
}

// Correction factor applied in finalize step
const LAST_ANNULUS_CORRECTION: f32 = 3.0459e-4;

/// Pre-computed total weights for a single sky patch.
///
/// Since the shadow array is identical across all annuli within a patch, the
/// accumulation `Σ(wᵢ × sh) = (Σwᵢ) × sh` allows collapsing the inner annulus
/// loop from ~10 iterations to a single weighted accumulation.
struct PatchWeights {
    weight_iso: f32,
    weight_n: f32,
    weight_e: f32,
    weight_s: f32,
    weight_w: f32,
}

/// Sum annulus weights for a patch, collapsing ~10 annuli to scalar totals.
fn precompute_patch_weights(patch: &PatchInfo) -> PatchWeights {
    let n = 90.0_f32;
    let common_w_factor = (1.0 / (2.0 * PI)) * (PI / (2.0 * n)).sin();
    let steprad_iso = (360.0 / patch.azimuth_patches) * (PI / 180.0);
    let steprad_aniso = (360.0 / patch.azimuth_patches_aniso) * (PI / 180.0);

    let mut sin_term_sum = 0.0_f32;
    for annulus_idx in patch.annulino_start..=patch.annulino_end {
        let annulus = 91.0 - annulus_idx as f32;
        sin_term_sum += ((PI * (2.0 * annulus - 1.0)) / (2.0 * n)).sin();
    }

    let total_iso = steprad_iso * common_w_factor * sin_term_sum;
    let total_aniso = steprad_aniso * common_w_factor * sin_term_sum;

    PatchWeights {
        weight_iso: total_iso,
        weight_n: if patch.azimuth >= 270.0 || patch.azimuth < 90.0 {
            total_aniso
        } else {
            0.0
        },
        weight_e: if patch.azimuth >= 0.0 && patch.azimuth < 180.0 {
            total_aniso
        } else {
            0.0
        },
        weight_s: if patch.azimuth >= 90.0 && patch.azimuth < 270.0 {
            total_aniso
        } else {
            0.0
        },
        weight_w: if patch.azimuth >= 180.0 && patch.azimuth < 360.0 {
            total_aniso
        } else {
            0.0
        },
    }
}

// Struct to hold patch configurations

pub struct PatchInfo {
    pub altitude: f32,
    pub azimuth: f32,
    pub azimuth_patches: f32,
    pub azimuth_patches_aniso: f32,
    pub annulino_start: i32,
    pub annulino_end: i32,
}

fn create_patches(option: u8) -> PyResult<Vec<PatchInfo>> {
    let (annulino, altitudes, azi_starts, azimuth_patches) = match option {
        1 => (
            vec![0, 12, 24, 36, 48, 60, 72, 84, 90],
            vec![6, 18, 30, 42, 54, 66, 78, 90],
            vec![0, 4, 2, 5, 8, 0, 10, 0],
            vec![30, 30, 24, 24, 18, 12, 6, 1],
        ),
        2 => (
            vec![0, 12, 24, 36, 48, 60, 72, 84, 90],
            vec![6, 18, 30, 42, 54, 66, 78, 90],
            vec![0, 4, 2, 5, 8, 0, 10, 0],
            vec![31, 30, 28, 24, 19, 13, 7, 1],
        ),
        3 => (
            vec![0, 12, 24, 36, 48, 60, 72, 84, 90],
            vec![6, 18, 30, 42, 54, 66, 78, 90],
            vec![0, 4, 2, 5, 8, 0, 10, 0],
            vec![62, 60, 56, 48, 38, 26, 14, 1],
        ),
        4 => (
            vec![0, 4, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 90],
            vec![3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 90],
            vec![0, 0, 4, 4, 2, 2, 5, 5, 8, 8, 0, 0, 10, 10, 0],
            vec![62, 62, 60, 60, 56, 56, 48, 48, 38, 38, 26, 26, 14, 14, 1],
        ),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported patch option: {} (valid: 1, 2, 3, 4)",
                option
            )));
        }
    };

    // Iterate over the patch configurations and create PatchInfo instances
    let mut patches: Vec<PatchInfo> = Vec::new();
    for i in 0..altitudes.len() {
        let azimuth_interval = 360.0 / azimuth_patches[i] as f32;
        for j in 0..azimuth_patches[i] as usize {
            // Calculate azimuth based on the start and interval
            // Use rem_euclid to ensure azimuth is within [0, 360)
            let azimuth = (azi_starts[i] as f32 + j as f32 * azimuth_interval).rem_euclid(360.0);
            patches.push(PatchInfo {
                altitude: altitudes[i] as f32,
                azimuth,
                azimuth_patches: azimuth_patches[i] as f32,
                // Calculate anisotropic azimuth patches (ceil(interval/2))
                azimuth_patches_aniso: (azimuth_patches[i] as f32 / 2.0).ceil(),
                annulino_start: annulino[i] + 1, // Start from the next annulino degree to avoid overlap
                annulino_end: annulino[i + 1],
            });
        }
    }
    Ok(patches)
}

// Structure to hold SVF results for Python
#[pyclass]
pub struct SvfResult {
    #[pyo3(get)]
    pub svf: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_north: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_east: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_south: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_west: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_veg: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_veg_north: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_veg_east: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_veg_south: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_veg_west: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_veg_blocks_bldg_sh: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_veg_blocks_bldg_sh_north: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_veg_blocks_bldg_sh_east: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_veg_blocks_bldg_sh_south: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_veg_blocks_bldg_sh_west: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub bldg_sh_matrix: Py<PyArray3<u8>>,
    #[pyo3(get)]
    pub veg_sh_matrix: Py<PyArray3<u8>>,
    #[pyo3(get)]
    pub veg_blocks_bldg_sh_matrix: Py<PyArray3<u8>>,
}

// Intermediate (pure Rust) SVF result used to avoid holding the GIL during compute
pub struct SvfIntermediate {
    pub svf: Array2<f32>,
    pub svf_n: Array2<f32>,
    pub svf_e: Array2<f32>,
    pub svf_s: Array2<f32>,
    pub svf_w: Array2<f32>,
    pub svf_veg: Array2<f32>,
    pub svf_veg_n: Array2<f32>,
    pub svf_veg_e: Array2<f32>,
    pub svf_veg_s: Array2<f32>,
    pub svf_veg_w: Array2<f32>,
    pub svf_veg_blocks_bldg_sh: Array2<f32>,
    pub svf_veg_blocks_bldg_sh_n: Array2<f32>,
    pub svf_veg_blocks_bldg_sh_e: Array2<f32>,
    pub svf_veg_blocks_bldg_sh_s: Array2<f32>,
    pub svf_veg_blocks_bldg_sh_w: Array2<f32>,
    pub bldg_sh_matrix: Array3<u8>,
    pub veg_sh_matrix: Array3<u8>,
    pub veg_blocks_bldg_sh_matrix: Array3<u8>,
}

impl SvfIntermediate {
    /// Create a zero-initialized SvfIntermediate with the given dimensions.
    /// Shadow matrices use bitpacked format: shape (rows, cols, ceil(patches/8)).
    pub fn zeros(num_rows: usize, num_cols: usize, total_patches: usize) -> Self {
        let shape2 = (num_rows, num_cols);
        let n_pack = pack_bytes(total_patches);
        let shape3_packed = (num_rows, num_cols, n_pack);

        SvfIntermediate {
            svf: Array2::<f32>::zeros(shape2),
            svf_n: Array2::<f32>::zeros(shape2),
            svf_e: Array2::<f32>::zeros(shape2),
            svf_s: Array2::<f32>::zeros(shape2),
            svf_w: Array2::<f32>::zeros(shape2),
            svf_veg: Array2::<f32>::zeros(shape2),
            svf_veg_n: Array2::<f32>::zeros(shape2),
            svf_veg_e: Array2::<f32>::zeros(shape2),
            svf_veg_s: Array2::<f32>::zeros(shape2),
            svf_veg_w: Array2::<f32>::zeros(shape2),
            svf_veg_blocks_bldg_sh: Array2::<f32>::zeros(shape2),
            svf_veg_blocks_bldg_sh_n: Array2::<f32>::zeros(shape2),
            svf_veg_blocks_bldg_sh_e: Array2::<f32>::zeros(shape2),
            svf_veg_blocks_bldg_sh_s: Array2::<f32>::zeros(shape2),
            svf_veg_blocks_bldg_sh_w: Array2::<f32>::zeros(shape2),
            bldg_sh_matrix: Array3::<u8>::zeros(shape3_packed),
            veg_sh_matrix: Array3::<u8>::zeros(shape3_packed),
            veg_blocks_bldg_sh_matrix: Array3::<u8>::zeros(shape3_packed),
        }
    }
}

/// Number of packed bytes needed for n_patches: ceil(n / 8).
#[inline(always)]
fn pack_bytes(n_patches: usize) -> usize {
    (n_patches + 7) / 8
}

fn prepare_bushes(vegdem: ArrayView2<f32>, vegdem2: ArrayView2<f32>) -> Array2<f32> {
    // Allocate output array with same shape as input
    let mut bush_areas = Array2::<f32>::zeros(vegdem.raw_dim());
    // Fill bush_areas in place, no unnecessary clones
    Zip::from(&mut bush_areas)
        .and(&vegdem)
        .and(&vegdem2)
        .for_each(|bush, &v1, &v2| {
            *bush = if v2 > 0.0 { 0.0 } else { v1 };
        });
    bush_areas
}

// --- Main Calculation Function ---
// Calculate SVF with 153 patches (equivalent to Python's svfForProcessing153)
// Internal implementation that supports an optional progress counter
fn calculate_svf_inner(
    dsm_owned: Array2<f32>,
    vegdem_owned: Array2<f32>,
    vegdem2_owned: Array2<f32>,
    scale: f32,
    usevegdem: bool,
    max_local_dsm_ht: f32,
    patch_option: u8,
    min_sun_elev_deg: Option<f32>,
    progress_counter: Option<Arc<AtomicUsize>>,
    cancel_flag: Option<Arc<AtomicBool>>,
    max_shadow_distance_m: f32,
) -> PyResult<SvfIntermediate> {
    // Convert owned arrays to views for internal processing
    let dsm_f32 = dsm_owned.view();
    let vegdem_f32 = vegdem_owned.view();
    let vegdem2_f32 = vegdem2_owned.view(); // Keep f32 version for finalize step

    let num_rows = dsm_f32.nrows();
    let num_cols = dsm_f32.ncols();

    // Prepare bushes
    let bush_f32 = prepare_bushes(vegdem_f32.view(), vegdem2_f32.view());

    // Create sky patches (use patch_option argument)
    let patches = create_patches(patch_option)?;
    let total_patches = patches.len(); // Needed for 3D array dimensions

    // Create a single intermediate result and allocate all arrays there
    let mut inter = SvfIntermediate::zeros(num_rows, num_cols, total_patches);

    // Try GPU SVF accumulation path: shadow + accumulate in one GPU submission per patch,
    // SVF values stay on GPU (no per-patch readback), read once at end.
    #[cfg(feature = "gpu")]
    let use_gpu_svf = if let Some(gpu_ctx) = crate::shadowing::get_gpu_context() {
        let (vc, vt, b) = if usevegdem {
            (
                Some(vegdem_f32.view()),
                Some(vegdem2_f32.view()),
                Some(bush_f32.view()),
            )
        } else {
            (None, None, None)
        };
        match gpu_ctx.init_svf_accumulation(
            num_rows,
            num_cols,
            usevegdem,
            total_patches,
            dsm_f32.view(),
            vc,
            vt,
            b,
        ) {
            Ok(()) => {
                crate::shadowing::record_gpu_dispatch();
                true
            }
            Err(e) => {
                eprintln!("[GPU] SVF accumulation init failed: {}. CPU fallback.", e);
                crate::shadowing::record_gpu_fallback();
                false
            }
        }
    } else {
        false
    };
    #[cfg(not(feature = "gpu"))]
    let use_gpu_svf = false;

    // Process patches: GPU pipelined path or CPU fallback
    if use_gpu_svf {
        #[cfg(feature = "gpu")]
        {
            let gpu_ctx = crate::shadowing::get_gpu_context().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "GPU context became unavailable during SVF execution",
                )
            })?;
            let min_elev = min_sun_elev_deg.unwrap_or(3.0_f32);
            const MAX_INFLIGHT_SVF_SUBMITS: usize = 16;
            let progress_cap = patches.len().saturating_sub(1);
            let mut inflight_submissions: VecDeque<wgpu::SubmissionIndex> = VecDeque::new();
            let mut completed_patches: usize = 0;

            for (patch_idx, patch) in patches.iter().enumerate() {
                let pw = precompute_patch_weights(patch);
                let submission_index = gpu_ctx
                    .dispatch_shadow_and_accumulate_svf(
                        patch_idx,
                        patch.azimuth,
                        patch.altitude,
                        scale,
                        max_local_dsm_ht,
                        min_elev,
                        max_shadow_distance_m,
                        pw.weight_iso,
                        pw.weight_n,
                        pw.weight_e,
                        pw.weight_s,
                        pw.weight_w,
                    )
                    .map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "GPU SVF dispatch failed at patch {}: {}",
                            patch_idx, e
                        ))
                    })?;
                inflight_submissions.push_back(submission_index);

                // Dispatch is non-blocking. Update progress only after submitted
                // GPU work is observed complete to avoid a progress bar that races ahead.
                if inflight_submissions.len() >= MAX_INFLIGHT_SVF_SUBMITS {
                    if let Some(done_idx) = inflight_submissions.pop_front() {
                        gpu_ctx.wait_for_submission(done_idx).map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(format!(
                                "GPU SVF synchronization failed at patch {}: {}",
                                patch_idx, e
                            ))
                        })?;
                        completed_patches += 1;
                        if let Some(ref counter) = progress_counter {
                            counter.store(completed_patches.min(progress_cap), Ordering::SeqCst);
                        }
                    }
                }

                // Check cancellation flag between patches
                if let Some(ref flag) = cancel_flag {
                    if flag.load(Ordering::SeqCst) {
                        return Err(pyo3::exceptions::PyInterruptedError::new_err(
                            "SVF computation cancelled",
                        ));
                    }
                }
            }

            // Drain remaining submissions so progress advances smoothly during GPU work,
            // not only at the final readback barrier.
            while let Some(done_idx) = inflight_submissions.pop_front() {
                gpu_ctx.wait_for_submission(done_idx).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "GPU SVF synchronization failed while draining submissions: {}",
                        e
                    ))
                })?;
                completed_patches += 1;
                if let Some(ref counter) = progress_counter {
                    counter.store(completed_patches.min(progress_cap), Ordering::SeqCst);
                }
                if let Some(ref flag) = cancel_flag {
                    if flag.load(Ordering::SeqCst) {
                        return Err(pyo3::exceptions::PyInterruptedError::new_err(
                            "SVF computation cancelled",
                        ));
                    }
                }
            }

            // Read back final bitpacked shadow matrices once.
            let bitpacked = gpu_ctx.read_svf_bitpacked_shadows().map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "GPU SVF bitpacked shadow readback failed: {}",
                    e
                ))
            })?;
            inter.bldg_sh_matrix = bitpacked.bldg_sh_matrix;
            inter.veg_sh_matrix = bitpacked.veg_sh_matrix;
            inter.veg_blocks_bldg_sh_matrix = bitpacked.veg_blocks_bldg_sh_matrix;

            // Read back accumulated SVF values from GPU
            let svf = gpu_ctx.read_svf_results().map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("GPU SVF readback failed: {}", e))
            })?;

            inter.svf = svf.svf;
            inter.svf_n = svf.svf_n;
            inter.svf_e = svf.svf_e;
            inter.svf_s = svf.svf_s;
            inter.svf_w = svf.svf_w;

            if usevegdem {
                if let Some(v) = svf.svf_veg {
                    inter.svf_veg = v;
                }
                if let Some(v) = svf.svf_veg_n {
                    inter.svf_veg_n = v;
                }
                if let Some(v) = svf.svf_veg_e {
                    inter.svf_veg_e = v;
                }
                if let Some(v) = svf.svf_veg_s {
                    inter.svf_veg_s = v;
                }
                if let Some(v) = svf.svf_veg_w {
                    inter.svf_veg_w = v;
                }
                if let Some(v) = svf.svf_aveg {
                    inter.svf_veg_blocks_bldg_sh = v;
                }
                if let Some(v) = svf.svf_aveg_n {
                    inter.svf_veg_blocks_bldg_sh_n = v;
                }
                if let Some(v) = svf.svf_aveg_e {
                    inter.svf_veg_blocks_bldg_sh_e = v;
                }
                if let Some(v) = svf.svf_aveg_s {
                    inter.svf_veg_blocks_bldg_sh_s = v;
                }
                if let Some(v) = svf.svf_aveg_w {
                    inter.svf_veg_blocks_bldg_sh_w = v;
                }
            }

            // Keep one progress step for finalize/packing so UI does not show
            // 100% while CPU post-processing is still running.
            if let Some(ref counter) = progress_counter {
                counter.store(progress_cap, Ordering::SeqCst);
            }
        }
    } else {
        for (patch_idx, patch) in patches.iter().enumerate() {
            // CPU fallback path
            let dsm_view = dsm_f32.view();
            let (vegdem_view, vegdem2_view, bush_view) = if usevegdem {
                (
                    Some(vegdem_f32.view()),
                    Some(vegdem2_f32.view()),
                    Some(bush_f32.view()),
                )
            } else {
                (None, None, None)
            };

            let shadow_result = compute_svf_shadows(
                dsm_view,
                vegdem_view,
                vegdem2_view,
                bush_view,
                patch.azimuth,
                patch.altitude,
                scale,
                max_local_dsm_ht,
                min_sun_elev_deg.unwrap_or(3.0_f32),
                max_shadow_distance_m,
            );

            // Bitpack f32 shadows into matrices (bit=1 means shadow value >= 0.5)
            {
                let byte_idx = patch_idx >> 3;
                let bit_mask = 1u8 << (patch_idx & 7);
                let mut bldg_plane = inter.bldg_sh_matrix.slice_mut(s![.., .., byte_idx]);
                Zip::from(&shadow_result.bldg_sh)
                    .and(&mut bldg_plane)
                    .par_for_each(|&b, bp| {
                        if b >= 0.5 {
                            *bp |= bit_mask;
                        }
                    });
                if usevegdem {
                    let mut veg_plane = inter.veg_sh_matrix.slice_mut(s![.., .., byte_idx]);
                    let mut vb_plane =
                        inter.veg_blocks_bldg_sh_matrix.slice_mut(s![.., .., byte_idx]);
                    Zip::from(&shadow_result.veg_sh)
                        .and(&shadow_result.veg_blocks_bldg_sh)
                        .and(&mut veg_plane)
                        .and(&mut vb_plane)
                        .par_for_each(|&v, &vb, vp, vbp| {
                            if v >= 0.5 {
                                *vp |= bit_mask;
                            }
                            if vb >= 0.5 {
                                *vbp |= bit_mask;
                            }
                        });
                }
            }

            let pw = precompute_patch_weights(patch);

            Zip::from(&shadow_result.bldg_sh)
                .and(&mut inter.svf)
                .and(&mut inter.svf_e)
                .and(&mut inter.svf_s)
                .and(&mut inter.svf_w)
                .and(&mut inter.svf_n)
                .par_for_each(|&b, svf, svf_e, svf_s, svf_w, svf_n| {
                    *svf += pw.weight_iso * b;
                    *svf_e += pw.weight_e * b;
                    *svf_s += pw.weight_s * b;
                    *svf_w += pw.weight_w * b;
                    *svf_n += pw.weight_n * b;
                });

            if usevegdem {
                Zip::from(&shadow_result.veg_sh)
                    .and(&mut inter.svf_veg)
                    .and(&mut inter.svf_veg_e)
                    .and(&mut inter.svf_veg_s)
                    .and(&mut inter.svf_veg_w)
                    .and(&mut inter.svf_veg_n)
                    .par_for_each(|&veg, svf_v, svf_v_e, svf_v_s, svf_v_w, svf_v_n| {
                        *svf_v += pw.weight_iso * veg;
                        *svf_v_e += pw.weight_e * veg;
                        *svf_v_s += pw.weight_s * veg;
                        *svf_v_w += pw.weight_w * veg;
                        *svf_v_n += pw.weight_n * veg;
                    });

                Zip::from(&shadow_result.veg_blocks_bldg_sh)
                    .and(&mut inter.svf_veg_blocks_bldg_sh)
                    .and(&mut inter.svf_veg_blocks_bldg_sh_e)
                    .and(&mut inter.svf_veg_blocks_bldg_sh_s)
                    .and(&mut inter.svf_veg_blocks_bldg_sh_w)
                    .and(&mut inter.svf_veg_blocks_bldg_sh_n)
                    .par_for_each(
                        |&veg_bldg, svf_v_b, svf_v_be, svf_v_bs, svf_v_bw, svf_v_bn| {
                            *svf_v_b += pw.weight_iso * veg_bldg;
                            *svf_v_be += pw.weight_e * veg_bldg;
                            *svf_v_bs += pw.weight_s * veg_bldg;
                            *svf_v_bw += pw.weight_w * veg_bldg;
                            *svf_v_bn += pw.weight_n * veg_bldg;
                        },
                    );
            }

            // Update progress counter
            if let Some(ref counter) = progress_counter {
                counter.fetch_add(1, Ordering::SeqCst);
            }

            // Check cancellation flag
            if let Some(ref flag) = cancel_flag {
                if flag.load(Ordering::SeqCst) {
                    return Err(pyo3::exceptions::PyInterruptedError::new_err(
                        "SVF computation cancelled",
                    ));
                }
            }
        }
    }

    // Finalize: apply last-annulus correction and clamp values, same semantics as the previous finalize
    let has_nan = dsm_f32.iter().any(|v| v.is_nan());

    inter.svf_s += LAST_ANNULUS_CORRECTION;
    inter.svf_w += LAST_ANNULUS_CORRECTION;

    Zip::from(&mut inter.svf).par_for_each(|x| *x = x.min(1.0));
    Zip::from(&mut inter.svf_n).par_for_each(|x| *x = x.min(1.0));
    Zip::from(&mut inter.svf_e).par_for_each(|x| *x = x.min(1.0));
    Zip::from(&mut inter.svf_s).par_for_each(|x| *x = x.min(1.0));
    Zip::from(&mut inter.svf_w).par_for_each(|x| *x = x.min(1.0));

    // Set NaN in outputs only when needed.
    if has_nan {
        Zip::from(&mut inter.svf)
            .and(&mut inter.svf_n)
            .and(&mut inter.svf_e)
            .and(&mut inter.svf_s)
            .and(&mut inter.svf_w)
            .and(&dsm_f32)
            .par_for_each(|svf, svf_n, svf_e, svf_s, svf_w, &dsm_val| {
                if dsm_val.is_nan() {
                    *svf = f32::NAN;
                    *svf_n = f32::NAN;
                    *svf_e = f32::NAN;
                    *svf_s = f32::NAN;
                    *svf_w = f32::NAN;
                }
            });
    }

    if usevegdem {
        // Apply directional correction in-place without allocating a temporary full-grid array.
        Zip::from(&mut inter.svf_veg_s)
            .and(&vegdem2_f32)
            .par_for_each(|svf_veg_s, &veg2| {
                if veg2 == 0.0 {
                    *svf_veg_s += LAST_ANNULUS_CORRECTION;
                }
            });
        Zip::from(&mut inter.svf_veg_w)
            .and(&vegdem2_f32)
            .par_for_each(|svf_veg_w, &veg2| {
                if veg2 == 0.0 {
                    *svf_veg_w += LAST_ANNULUS_CORRECTION;
                }
            });
        Zip::from(&mut inter.svf_veg_blocks_bldg_sh_s)
            .and(&vegdem2_f32)
            .par_for_each(|svf_vb_s, &veg2| {
                if veg2 == 0.0 {
                    *svf_vb_s += LAST_ANNULUS_CORRECTION;
                }
            });
        Zip::from(&mut inter.svf_veg_blocks_bldg_sh_w)
            .and(&vegdem2_f32)
            .par_for_each(|svf_vb_w, &veg2| {
                if veg2 == 0.0 {
                    *svf_vb_w += LAST_ANNULUS_CORRECTION;
                }
            });

        Zip::from(&mut inter.svf_veg).par_for_each(|x| *x = x.min(1.0));
        Zip::from(&mut inter.svf_veg_n).par_for_each(|x| *x = x.min(1.0));
        Zip::from(&mut inter.svf_veg_e).par_for_each(|x| *x = x.min(1.0));
        Zip::from(&mut inter.svf_veg_s).par_for_each(|x| *x = x.min(1.0));
        Zip::from(&mut inter.svf_veg_w).par_for_each(|x| *x = x.min(1.0));
        Zip::from(&mut inter.svf_veg_blocks_bldg_sh).par_for_each(|x| *x = x.min(1.0));
        Zip::from(&mut inter.svf_veg_blocks_bldg_sh_n).par_for_each(|x| *x = x.min(1.0));
        Zip::from(&mut inter.svf_veg_blocks_bldg_sh_e).par_for_each(|x| *x = x.min(1.0));
        Zip::from(&mut inter.svf_veg_blocks_bldg_sh_s).par_for_each(|x| *x = x.min(1.0));
        Zip::from(&mut inter.svf_veg_blocks_bldg_sh_w).par_for_each(|x| *x = x.min(1.0));

        // Set NaN in veg outputs only when needed (split into two operations due to Zip limit)
        if has_nan {
            Zip::from(&mut inter.svf_veg)
                .and(&mut inter.svf_veg_n)
                .and(&mut inter.svf_veg_e)
                .and(&mut inter.svf_veg_s)
                .and(&mut inter.svf_veg_w)
                .and(&dsm_f32)
                .par_for_each(
                    |svf_veg, svf_veg_n, svf_veg_e, svf_veg_s, svf_veg_w, &dsm_val| {
                        if dsm_val.is_nan() {
                            *svf_veg = f32::NAN;
                            *svf_veg_n = f32::NAN;
                            *svf_veg_e = f32::NAN;
                            *svf_veg_s = f32::NAN;
                            *svf_veg_w = f32::NAN;
                        }
                    },
                );

            Zip::from(&mut inter.svf_veg_blocks_bldg_sh)
                .and(&mut inter.svf_veg_blocks_bldg_sh_n)
                .and(&mut inter.svf_veg_blocks_bldg_sh_e)
                .and(&mut inter.svf_veg_blocks_bldg_sh_s)
                .and(&mut inter.svf_veg_blocks_bldg_sh_w)
                .and(&dsm_f32)
                .par_for_each(|svf_vb, svf_vb_n, svf_vb_e, svf_vb_s, svf_vb_w, &dsm_val| {
                    if dsm_val.is_nan() {
                        *svf_vb = f32::NAN;
                        *svf_vb_n = f32::NAN;
                        *svf_vb_e = f32::NAN;
                        *svf_vb_s = f32::NAN;
                        *svf_vb_w = f32::NAN;
                    }
                });
        }
    }

    // When no vegetation, veg shadow matrices must indicate "no blocking":
    //   veg_sh_matrix: all bits = 1 (sky visible through vegetation at every patch)
    //   veg_blocks_bldg_sh_matrix: copy of bldg_sh_matrix (only buildings matter)
    let n_pack = pack_bytes(total_patches);
    if !usevegdem {
        inter.veg_sh_matrix.fill(0xFF);
        inter
            .veg_blocks_bldg_sh_matrix
            .assign(&inter.bldg_sh_matrix);
    }

    // Zero out bitpacked shadow matrices for NaN pixels only when needed.
    // Run per-byte in parallel over pixels to avoid a long single-core tail.
    if has_nan {
        for bi in 0..n_pack {
            let mut bldg_plane = inter.bldg_sh_matrix.slice_mut(s![.., .., bi]);
            let mut veg_plane = inter.veg_sh_matrix.slice_mut(s![.., .., bi]);
            let mut vb_plane = inter.veg_blocks_bldg_sh_matrix.slice_mut(s![.., .., bi]);
            Zip::from(&dsm_f32)
                .and(&mut bldg_plane)
                .and(&mut veg_plane)
                .and(&mut vb_plane)
                .par_for_each(|&dsm_val, bldg, veg, vb| {
                    if dsm_val.is_nan() {
                        *bldg = 0;
                        *veg = 0;
                        *vb = 0;
                    }
                });
        }
    }

    if let Some(ref counter) = progress_counter {
        counter.store(patches.len(), Ordering::SeqCst);
    }

    Ok(inter)
}

// Convert SvfIntermediate into Python SvfResult under the GIL
fn svf_intermediate_to_py(py: Python, inter: SvfIntermediate) -> PyResult<Py<SvfResult>> {
    Py::new(
        py,
        SvfResult {
            svf: inter.svf.into_pyarray(py).unbind(),
            svf_north: inter.svf_n.into_pyarray(py).unbind(),
            svf_east: inter.svf_e.into_pyarray(py).unbind(),
            svf_south: inter.svf_s.into_pyarray(py).unbind(),
            svf_west: inter.svf_w.into_pyarray(py).unbind(),
            svf_veg: inter.svf_veg.into_pyarray(py).unbind(),
            svf_veg_north: inter.svf_veg_n.into_pyarray(py).unbind(),
            svf_veg_east: inter.svf_veg_e.into_pyarray(py).unbind(),
            svf_veg_south: inter.svf_veg_s.into_pyarray(py).unbind(),
            svf_veg_west: inter.svf_veg_w.into_pyarray(py).unbind(),
            svf_veg_blocks_bldg_sh: inter.svf_veg_blocks_bldg_sh.into_pyarray(py).unbind(),
            svf_veg_blocks_bldg_sh_north: inter.svf_veg_blocks_bldg_sh_n.into_pyarray(py).unbind(),
            svf_veg_blocks_bldg_sh_east: inter.svf_veg_blocks_bldg_sh_e.into_pyarray(py).unbind(),
            svf_veg_blocks_bldg_sh_south: inter.svf_veg_blocks_bldg_sh_s.into_pyarray(py).unbind(),
            svf_veg_blocks_bldg_sh_west: inter.svf_veg_blocks_bldg_sh_w.into_pyarray(py).unbind(),
            bldg_sh_matrix: inter.bldg_sh_matrix.into_pyarray(py).unbind(),
            veg_sh_matrix: inter.veg_sh_matrix.into_pyarray(py).unbind(),
            veg_blocks_bldg_sh_matrix: inter.veg_blocks_bldg_sh_matrix.into_pyarray(py).unbind(),
        },
    )
}

fn crop_svf_intermediate(
    inter: SvfIntermediate,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
) -> PyResult<SvfIntermediate> {
    let rows = inter.svf.nrows();
    let cols = inter.svf.ncols();
    if row_start >= row_end || col_start >= col_end || row_end > rows || col_end > cols {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid core window: rows [{}, {}), cols [{}, {}) for tile {}x{}",
            row_start, row_end, col_start, col_end, rows, cols
        )));
    }

    Ok(SvfIntermediate {
        svf: inter
            .svf
            .slice(s![row_start..row_end, col_start..col_end])
            .to_owned(),
        svf_n: inter
            .svf_n
            .slice(s![row_start..row_end, col_start..col_end])
            .to_owned(),
        svf_e: inter
            .svf_e
            .slice(s![row_start..row_end, col_start..col_end])
            .to_owned(),
        svf_s: inter
            .svf_s
            .slice(s![row_start..row_end, col_start..col_end])
            .to_owned(),
        svf_w: inter
            .svf_w
            .slice(s![row_start..row_end, col_start..col_end])
            .to_owned(),
        svf_veg: inter
            .svf_veg
            .slice(s![row_start..row_end, col_start..col_end])
            .to_owned(),
        svf_veg_n: inter
            .svf_veg_n
            .slice(s![row_start..row_end, col_start..col_end])
            .to_owned(),
        svf_veg_e: inter
            .svf_veg_e
            .slice(s![row_start..row_end, col_start..col_end])
            .to_owned(),
        svf_veg_s: inter
            .svf_veg_s
            .slice(s![row_start..row_end, col_start..col_end])
            .to_owned(),
        svf_veg_w: inter
            .svf_veg_w
            .slice(s![row_start..row_end, col_start..col_end])
            .to_owned(),
        svf_veg_blocks_bldg_sh: inter
            .svf_veg_blocks_bldg_sh
            .slice(s![row_start..row_end, col_start..col_end])
            .to_owned(),
        svf_veg_blocks_bldg_sh_n: inter
            .svf_veg_blocks_bldg_sh_n
            .slice(s![row_start..row_end, col_start..col_end])
            .to_owned(),
        svf_veg_blocks_bldg_sh_e: inter
            .svf_veg_blocks_bldg_sh_e
            .slice(s![row_start..row_end, col_start..col_end])
            .to_owned(),
        svf_veg_blocks_bldg_sh_s: inter
            .svf_veg_blocks_bldg_sh_s
            .slice(s![row_start..row_end, col_start..col_end])
            .to_owned(),
        svf_veg_blocks_bldg_sh_w: inter
            .svf_veg_blocks_bldg_sh_w
            .slice(s![row_start..row_end, col_start..col_end])
            .to_owned(),
        bldg_sh_matrix: inter
            .bldg_sh_matrix
            .slice(s![row_start..row_end, col_start..col_end, ..])
            .to_owned(),
        veg_sh_matrix: inter
            .veg_sh_matrix
            .slice(s![row_start..row_end, col_start..col_end, ..])
            .to_owned(),
        veg_blocks_bldg_sh_matrix: inter
            .veg_blocks_bldg_sh_matrix
            .slice(s![row_start..row_end, col_start..col_end, ..])
            .to_owned(),
    })
}

// Keep existing pyfunction wrapper for backward compatibility (ignores progress)
#[pyfunction]
#[pyo3(signature = (dsm_py, vegdem_py, vegdem2_py, scale, usevegdem, max_local_dsm_ht, patch_option=None, min_sun_elev_deg=None, _progress_callback=None, max_shadow_distance_m=None))]
pub fn calculate_svf(
    py: Python,
    dsm_py: PyReadonlyArray2<f32>,
    vegdem_py: PyReadonlyArray2<f32>,
    vegdem2_py: PyReadonlyArray2<f32>,
    scale: f32,
    usevegdem: bool,
    max_local_dsm_ht: f32,
    patch_option: Option<u8>, // New argument for patch option
    min_sun_elev_deg: Option<f32>,
    _progress_callback: Option<PyObject>,
    max_shadow_distance_m: Option<f32>,
) -> PyResult<Py<SvfResult>> {
    let patch_option = patch_option.unwrap_or(2);
    let max_shadow_dist = max_shadow_distance_m.unwrap_or(0.0);
    // Copy Python arrays into owned Rust arrays so computation can run without the GIL
    let dsm_owned = dsm_py.as_array().to_owned();
    let vegdem_owned = vegdem_py.as_array().to_owned();
    let vegdem2_owned = vegdem2_py.as_array().to_owned();
    let inter = py.allow_threads(|| {
        calculate_svf_inner(
            dsm_owned,
            vegdem_owned,
            vegdem2_owned,
            scale,
            usevegdem,
            max_local_dsm_ht,
            patch_option,
            min_sun_elev_deg,
            None,
            None,
            max_shadow_dist,
        )
    })?;
    svf_intermediate_to_py(py, inter)
}

// New pyclass runner that exposes progress() and cancel() methods
#[pyclass]
pub struct SkyviewRunner {
    progress: Arc<AtomicUsize>,
    cancelled: Arc<AtomicBool>,
}

impl Default for SkyviewRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl SkyviewRunner {
    #[new]
    pub fn new() -> Self {
        Self {
            progress: Arc::new(AtomicUsize::new(0)),
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn progress(&self) -> usize {
        self.progress.load(Ordering::SeqCst)
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    #[pyo3(signature = (dsm_py, vegdem_py, vegdem2_py, scale, usevegdem, max_local_dsm_ht, patch_option=None, min_sun_elev_deg=None, max_shadow_distance_m=None))]
    pub fn calculate_svf(
        &self,
        py: Python,
        dsm_py: PyReadonlyArray2<f32>,
        vegdem_py: PyReadonlyArray2<f32>,
        vegdem2_py: PyReadonlyArray2<f32>,
        scale: f32,
        usevegdem: bool,
        max_local_dsm_ht: f32,
        patch_option: Option<u8>,
        min_sun_elev_deg: Option<f32>,
        max_shadow_distance_m: Option<f32>,
    ) -> PyResult<Py<SvfResult>> {
        let patch_option = patch_option.unwrap_or(2);
        let max_shadow_dist = max_shadow_distance_m.unwrap_or(0.0);
        // reset progress and cancel flag
        self.progress.store(0, Ordering::SeqCst);
        self.cancelled.store(false, Ordering::SeqCst);
        // Copy arrays to owned buffers and run without the GIL so progress can be polled
        let dsm_owned = dsm_py.as_array().to_owned();
        let vegdem_owned = vegdem_py.as_array().to_owned();
        let vegdem2_owned = vegdem2_py.as_array().to_owned();
        let inter = py.allow_threads(|| {
            calculate_svf_inner(
                dsm_owned,
                vegdem_owned,
                vegdem2_owned,
                scale,
                usevegdem,
                max_local_dsm_ht,
                patch_option,
                min_sun_elev_deg,
                Some(self.progress.clone()),
                Some(self.cancelled.clone()),
                max_shadow_dist,
            )
        })?;
        svf_intermediate_to_py(py, inter)
    }

    #[pyo3(signature = (dsm_py, vegdem_py, vegdem2_py, scale, usevegdem, max_local_dsm_ht, patch_option, min_sun_elev_deg, core_row_start, core_row_end, core_col_start, core_col_end, max_shadow_distance_m=None))]
    pub fn calculate_svf_core(
        &self,
        py: Python,
        dsm_py: PyReadonlyArray2<f32>,
        vegdem_py: PyReadonlyArray2<f32>,
        vegdem2_py: PyReadonlyArray2<f32>,
        scale: f32,
        usevegdem: bool,
        max_local_dsm_ht: f32,
        patch_option: Option<u8>,
        min_sun_elev_deg: Option<f32>,
        core_row_start: usize,
        core_row_end: usize,
        core_col_start: usize,
        core_col_end: usize,
        max_shadow_distance_m: Option<f32>,
    ) -> PyResult<Py<SvfResult>> {
        let patch_option = patch_option.unwrap_or(2);
        let max_shadow_dist = max_shadow_distance_m.unwrap_or(0.0);
        self.progress.store(0, Ordering::SeqCst);
        self.cancelled.store(false, Ordering::SeqCst);
        let dsm_owned = dsm_py.as_array().to_owned();
        let vegdem_owned = vegdem_py.as_array().to_owned();
        let vegdem2_owned = vegdem2_py.as_array().to_owned();
        let inter = py.allow_threads(|| {
            calculate_svf_inner(
                dsm_owned,
                vegdem_owned,
                vegdem2_owned,
                scale,
                usevegdem,
                max_local_dsm_ht,
                patch_option,
                min_sun_elev_deg,
                Some(self.progress.clone()),
                Some(self.cancelled.clone()),
                max_shadow_dist,
            )
        })?;
        let core_inter = crop_svf_intermediate(
            inter,
            core_row_start,
            core_row_end,
            core_col_start,
            core_col_end,
        )?;
        svf_intermediate_to_py(py, core_inter)
    }
}
