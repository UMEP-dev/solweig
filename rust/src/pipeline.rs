//! Fused timestep pipeline — single FFI entrance/exit per timestep.
//!
//! Orchestrates: shadows → ground_temp → GVF → thermal_delay → radiation → Tmrt
//! All intermediate arrays stay as ndarray::Array2<f32> — never cross FFI boundary.
//!
//! Supports both isotropic and anisotropic (Perez) sky models.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip};
use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use std::sync::OnceLock;

use crate::ground::{compute_ground_temperature_pure, ts_wave_delay_batch_pure, GroundTempResult};
use crate::ground_surface::{outgoing_longwave_calc_pure, surface_temperature_calc_pure};
use crate::gvf::{gvf_calc_pure, gvf_calc_with_cache, GvfResultPure};
#[cfg(feature = "gpu")]
use crate::gvf::gvf_calc_with_cache_gpu;
use crate::gvf_geometry::{precompute_gvf_geometry, GvfGeometryCache};
use crate::shadowing::{calculate_shadows_rust, ShadowingResultRust};
use crate::sky::{anisotropic_sky_pure, cylindric_wedge_pure_masked};
use crate::tmrt::compute_tmrt_from_dir_sums_pure;
use crate::vegetation::{kside_veg_isotropic_pure, lside_veg_pure, lside_veg_variant_pure, LsideVariant};

#[cfg(feature = "gpu")]
use crate::gpu::AnisoGpuContext;
#[cfg(feature = "gpu")]
use crate::gpu::GvfGpuContext;

use std::time::Instant;

const PI: f32 = std::f32::consts::PI;
const SBC: f32 = 5.67051e-8;
const KELVIN_OFFSET: f32 = 273.15;

/// Check once per process whether timing output is enabled (``SOLWEIG_TIMING=1``).
fn timing_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("SOLWEIG_TIMING")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

// ── GPU anisotropic sky context (lazy-initialized, shares device with shadows) ──

#[cfg(feature = "gpu")]
static ANISO_GPU_CONTEXT: OnceLock<Option<AnisoGpuContext>> = OnceLock::new();

#[cfg(feature = "gpu")]
static ANISO_GPU_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

#[cfg(feature = "gpu")]
fn get_aniso_gpu_context() -> Option<&'static AnisoGpuContext> {
    if !ANISO_GPU_ENABLED.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    ANISO_GPU_CONTEXT
        .get_or_init(|| {
            // Share device/queue from the shadow GPU context
            let shadow_ctx = crate::shadowing::get_gpu_context()?;
            let device = shadow_ctx.device.clone();
            let queue = shadow_ctx.queue.clone();
            let ctx = AnisoGpuContext::new(device, queue);
            eprintln!("[GPU] Anisotropic sky GPU context initialized");
            Some(ctx)
        })
        .as_ref()
}

#[cfg(feature = "gpu")]
#[pyfunction]
/// Enable GPU acceleration for anisotropic sky computation
pub fn enable_aniso_gpu() {
    ANISO_GPU_ENABLED.store(true, std::sync::atomic::Ordering::Relaxed);
    eprintln!("[GPU] Anisotropic sky GPU acceleration enabled");
}

#[cfg(feature = "gpu")]
#[pyfunction]
/// Disable GPU acceleration for anisotropic sky computation (CPU fallback)
pub fn disable_aniso_gpu() {
    ANISO_GPU_ENABLED.store(false, std::sync::atomic::Ordering::Relaxed);
    eprintln!("[GPU] Anisotropic sky GPU acceleration disabled");
}

#[cfg(feature = "gpu")]
#[pyfunction]
/// Check if GPU acceleration is enabled for anisotropic sky
pub fn is_aniso_gpu_enabled() -> bool {
    ANISO_GPU_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

// ── GPU GVF context (lazy-initialized, shares device with shadows) ──────

#[cfg(feature = "gpu")]
static GVF_GPU_CONTEXT: OnceLock<Option<GvfGpuContext>> = OnceLock::new();

#[cfg(feature = "gpu")]
static GVF_GPU_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

#[cfg(feature = "gpu")]
fn get_gvf_gpu_context() -> Option<&'static GvfGpuContext> {
    if !GVF_GPU_ENABLED.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    GVF_GPU_CONTEXT
        .get_or_init(|| {
            let shadow_ctx = crate::shadowing::get_gpu_context()?;
            let device = shadow_ctx.device.clone();
            let queue = shadow_ctx.queue.clone();
            let ctx = GvfGpuContext::new(device, queue);
            eprintln!("[GPU] GVF GPU context initialized");
            Some(ctx)
        })
        .as_ref()
}

#[cfg(feature = "gpu")]
#[pyfunction]
/// Enable GPU acceleration for GVF computation
pub fn enable_gvf_gpu() {
    GVF_GPU_ENABLED.store(true, std::sync::atomic::Ordering::Relaxed);
    eprintln!("[GPU] GVF GPU acceleration enabled");
}

#[cfg(feature = "gpu")]
#[pyfunction]
/// Disable GPU acceleration for GVF computation (CPU fallback)
pub fn disable_gvf_gpu() {
    GVF_GPU_ENABLED.store(false, std::sync::atomic::Ordering::Relaxed);
    eprintln!("[GPU] GVF GPU acceleration disabled");
}

#[cfg(feature = "gpu")]
#[pyfunction]
/// Check if GPU acceleration is enabled for GVF
pub fn is_gvf_gpu_enabled() -> bool {
    GVF_GPU_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

// Scalar input structs (WeatherScalars / HumanScalars / ConfigScalars)
// extracted to pipeline_scalars.rs. Re-exported here so the lib.rs
// registration path `pipeline::WeatherScalars` keeps working.
pub use crate::pipeline_scalars::{ConfigScalars, HumanScalars, WeatherScalars};

// ── Output struct ──────────────────────────────────────────────────────────

/// Result from a single fused timestep.
#[pyclass]
pub struct TimestepResult {
    #[pyo3(get)]
    pub tmrt: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub shadow: Option<Py<PyArray2<f32>>>,
    #[pyo3(get)]
    pub kdown: Option<Py<PyArray2<f32>>>,
    #[pyo3(get)]
    pub kup: Option<Py<PyArray2<f32>>>,
    #[pyo3(get)]
    pub ldown: Option<Py<PyArray2<f32>>>,
    #[pyo3(get)]
    pub lup: Option<Py<PyArray2<f32>>>,
    // Updated thermal state arrays (Python extracts and passes back next timestep)
    #[pyo3(get)]
    pub timeadd: f32,
    #[pyo3(get)]
    pub tgmap1: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub tgmap1_e: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub tgmap1_s: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub tgmap1_w: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub tgmap1_n: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub tgout1: Py<PyArray2<f32>>,
    // Updated ground-scheme state (Some only when the 2026a ground scheme ran;
    // Python carries these forward like the thermal state above).
    #[pyo3(get)]
    pub tg: Option<Py<PyArray2<f32>>>,
    #[pyo3(get)]
    pub rn: Option<Py<PyArray2<f32>>>,
    #[pyo3(get)]
    pub rn_past: Option<Py<PyArray2<f32>>>,
    #[pyo3(get)]
    pub g: Option<Py<PyArray2<f32>>>,
}

/// Raw result struct with owned arrays (no Python types — Send-safe).
struct TimestepResultRaw {
    tmrt: Array2<f32>,
    shadow: Option<Array2<f32>>,
    kdown: Option<Array2<f32>>,
    kup: Option<Array2<f32>>,
    ldown: Option<Array2<f32>>,
    lup: Option<Array2<f32>>,
    timeadd: f32,
    tgmap1: Array2<f32>,
    tgmap1_e: Array2<f32>,
    tgmap1_s: Array2<f32>,
    tgmap1_w: Array2<f32>,
    tgmap1_n: Array2<f32>,
    tgout1: Array2<f32>,
    tg: Option<Array2<f32>>,
    rn: Option<Array2<f32>>,
    rn_past: Option<Array2<f32>>,
    g: Option<Array2<f32>>,
}

const OUT_SHADOW: u32 = 1 << 0;
const OUT_KDOWN: u32 = 1 << 1;
const OUT_KUP: u32 = 1 << 2;
const OUT_LDOWN: u32 = 1 << 3;
const OUT_LUP: u32 = 1 << 4;
const OUT_ALL: u32 = OUT_SHADOW | OUT_KDOWN | OUT_KUP | OUT_LDOWN | OUT_LUP;

/// Release the GIL for a closure whose captured state may not be `Send`.
///
/// # Safety
/// Caller must guarantee that all borrowed data remains alive for the duration
/// of the closure (i.e. the Python objects backing any `ArrayView` are not
/// deallocated while the GIL is released).
unsafe fn allow_threads_unchecked<T: Send, F: FnOnce() -> T>(py: Python, f: F) -> T {
    // Move f to the heap and erase through usize so the auto-Send derivation
    // for the closure sees only Send types (usize), not the non-Send F.
    let raw = Box::into_raw(Box::new(f));
    let addr = raw as usize;
    py.allow_threads(move || unsafe {
        let f = *Box::from_raw(addr as *mut F);
        f()
    })
}

// Radiation helpers extracted to pipeline_radiation.rs. Re-import only
// the symbols used inside compute_timestep below.
use crate::pipeline_radiation::{
    asvf_for_svf_cached, compute_ani_lum_from_packed, compute_esky, compute_kdown, compute_kup,
    compute_ldown, kside_dirs_sum_aniso_from_kup, lside_dirs_sum_aniso_from_lup,
    patch_lut_for_option_cached, side_sum_from_directional,
};


// ── GVF Geometry Cache (opaque handle for Python) ─────────────────────────

/// Opaque handle to a precomputed GVF geometry cache.
///
/// Created once per DSM via `precompute_gvf_cache()`, then passed to
/// `compute_timestep()` on subsequent calls to skip building ray-tracing.
#[pyclass]
pub struct PyGvfGeometryCache {
    pub(crate) inner: GvfGeometryCache,
}

/// Precompute GVF geometry cache for a given set of surface arrays.
///
/// This runs the building ray-trace once (18 azimuths, parallelized).
/// The returned cache is passed to `compute_timestep()` to skip geometry
/// on subsequent timesteps with the same DSM.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn precompute_gvf_cache(
    buildings: PyReadonlyArray2<f32>,
    wall_asp: PyReadonlyArray2<f32>,
    wall_ht: PyReadonlyArray2<f32>,
    alb_grid: PyReadonlyArray2<f32>,
    pixel_size: f32,
    human_height: f32,
    wall_albedo: f32,
) -> PyResult<PyGvfGeometryCache> {
    let first_ht = human_height.round().max(1.0);
    let second_ht = human_height * 20.0;

    let cache = precompute_gvf_geometry(
        buildings.as_array(),
        wall_asp.as_array(),
        wall_ht.as_array(),
        alb_grid.as_array(),
        pixel_size,
        first_ht,
        second_ht,
        wall_albedo,
    );

    // Upload geometry to GPU if available
    #[cfg(feature = "gpu")]
    {
        if let Some(ctx) = get_gvf_gpu_context() {
            match ctx.upload_geometry(&cache) {
                Ok(()) => {
                    crate::shadowing::record_gpu_dispatch();
                }
                Err(e) => {
                    eprintln!("[GPU] GVF geometry upload failed, falling back to CPU: {}", e);
                    crate::shadowing::record_gpu_fallback();
                    GVF_GPU_ENABLED.store(false, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
    }

    Ok(PyGvfGeometryCache { inner: cache })
}

// Argument bundles for compute_timestep live in their own module to keep this
// file focused on the orchestration logic. Re-exported here so consumers of
// `crate::pipeline::{SvfBundle, ...}` keep working unchanged.
pub use crate::pipeline_bundles::{
    GROUND_SCHEME_BUNDLE_VERSION, GroundSchemeBundle, PropertiesBundle, STATE_BUNDLE_VERSION,
    StateBundle, SurfaceBundle, SvfBundle,
};

// ── Main fused timestep function ───────────────────────────────────────────

/// Compute a single daytime timestep entirely in Rust.
///
/// All intermediate arrays stay as ndarray::Array2<f32> — only the final
/// results cross back to Python.
///
/// Parameters are grouped into structs to keep the signature manageable:
/// - weather: Per-timestep scalars (sun position, temperature, radiation)
/// - human: Body parameters (height, posture, absorptivities)
/// - config: Constants (pixel_size, wall materials)
/// - Surface/SVF arrays: Borrowed from Python (zero-copy on input)
/// - Thermal state: Carried forward between timesteps
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn compute_timestep(
    py: Python,
    // Scalar parameter structs
    weather: &WeatherScalars,
    human: &HumanScalars,
    config: &ConfigScalars,
    // Optional GVF geometry cache (skip building ray-tracing if provided)
    gvf_cache: Option<&PyGvfGeometryCache>,
    // Surface arrays (6 rasters bundled into one SurfaceBundle).
    surface_bundle: &SurfaceBundle,
    // SVF arrays (constant across timesteps; 17 rasters bundled into one
    // pyclass to keep the FFI surface manageable).
    svf_bundle: &SvfBundle,
    // Land cover property grids (5 rasters bundled into one PyO3 class).
    properties_bundle: &PropertiesBundle,
    // Buildings mask for GVF
    buildings: PyReadonlyArray2<f32>,
    lc_grid: Option<PyReadonlyArray2<f32>>,
    // Anisotropic sky inputs (None for isotropic)
    shmat: Option<PyReadonlyArray3<u8>>,
    vegshmat: Option<PyReadonlyArray3<u8>>,
    vbshmat: Option<PyReadonlyArray3<u8>>,
    // Thermal state (6 arrays + 3 scalars + version) bundled into one pyclass.
    state_bundle: &StateBundle,
    // UMEP 2026a ground-surface scheme inputs + carried state (None = classic
    // Lindberg et al. ground temperature + GVF path, byte-identical baseline).
    ground_scheme: Option<&GroundSchemeBundle>,
    // Valid pixel mask (1=valid, 0=NaN/nodata — skip computation for invalid pixels)
    valid_mask: PyReadonlyArray2<u8>,
    // Optional output selection bitmask for Python conversion (tmrt always returned)
    output_mask: Option<u32>,
) -> PyResult<TimestepResult> {
    // Borrow all arrays (zero-copy from numpy)
    let valid_v = valid_mask.as_array();
    // Bind surface arrays from the bundle. The *_ro readonly bindings stay
    // alive for the function body so the *_v views are valid throughout.
    let dsm_ro = surface_bundle.dsm.bind(py).readonly();
    let cdsm_ro = surface_bundle.cdsm.as_ref().map(|a| a.bind(py).readonly());
    let tdsm_ro = surface_bundle.tdsm.as_ref().map(|a| a.bind(py).readonly());
    let bush_ro = surface_bundle.bush.as_ref().map(|a| a.bind(py).readonly());
    let wall_ht_ro = surface_bundle.wall_ht.as_ref().map(|a| a.bind(py).readonly());
    let wall_asp_ro = surface_bundle.wall_asp.as_ref().map(|a| a.bind(py).readonly());
    let dsm_v = dsm_ro.as_array();
    let cdsm_v = cdsm_ro.as_ref().map(|a| a.as_array());
    let tdsm_v = tdsm_ro.as_ref().map(|a| a.as_array());
    let bush_v = bush_ro.as_ref().map(|a| a.as_array());
    let wall_ht_v = wall_ht_ro.as_ref().map(|a| a.as_array());
    let wall_asp_v = wall_asp_ro.as_ref().map(|a| a.as_array());
    // Bind each SVF raster from the bundle. The PyReadonlyArray2 bindings
    // (svf_ro etc.) must outlive the ArrayView2s (svf_v etc.); both are
    // declared at this scope so they live for the whole function body.
    let svf_ro = svf_bundle.svf.bind(py).readonly();
    let svf_n_ro = svf_bundle.svf_n.bind(py).readonly();
    let svf_e_ro = svf_bundle.svf_e.bind(py).readonly();
    let svf_s_ro = svf_bundle.svf_s.bind(py).readonly();
    let svf_w_ro = svf_bundle.svf_w.bind(py).readonly();
    let svf_veg_ro = svf_bundle.svf_veg.bind(py).readonly();
    let svf_veg_n_ro = svf_bundle.svf_veg_n.bind(py).readonly();
    let svf_veg_e_ro = svf_bundle.svf_veg_e.bind(py).readonly();
    let svf_veg_s_ro = svf_bundle.svf_veg_s.bind(py).readonly();
    let svf_veg_w_ro = svf_bundle.svf_veg_w.bind(py).readonly();
    let svf_aveg_ro = svf_bundle.svf_aveg.bind(py).readonly();
    let svf_aveg_n_ro = svf_bundle.svf_aveg_n.bind(py).readonly();
    let svf_aveg_e_ro = svf_bundle.svf_aveg_e.bind(py).readonly();
    let svf_aveg_s_ro = svf_bundle.svf_aveg_s.bind(py).readonly();
    let svf_aveg_w_ro = svf_bundle.svf_aveg_w.bind(py).readonly();
    let svfbuveg_ro = svf_bundle.svfbuveg.bind(py).readonly();
    let svfalfa_ro = svf_bundle.svfalfa.bind(py).readonly();
    let svf_v = svf_ro.as_array();
    let svf_n_v = svf_n_ro.as_array();
    let svf_e_v = svf_e_ro.as_array();
    let svf_s_v = svf_s_ro.as_array();
    let svf_w_v = svf_w_ro.as_array();
    let svf_veg_v = svf_veg_ro.as_array();
    let svf_veg_n_v = svf_veg_n_ro.as_array();
    let svf_veg_e_v = svf_veg_e_ro.as_array();
    let svf_veg_s_v = svf_veg_s_ro.as_array();
    let svf_veg_w_v = svf_veg_w_ro.as_array();
    let svf_aveg_v = svf_aveg_ro.as_array();
    let svf_aveg_n_v = svf_aveg_n_ro.as_array();
    let svf_aveg_e_v = svf_aveg_e_ro.as_array();
    let svf_aveg_s_v = svf_aveg_s_ro.as_array();
    let svf_aveg_w_v = svf_aveg_w_ro.as_array();
    let svfbuveg_v = svfbuveg_ro.as_array();
    let svfalfa_v = svfalfa_ro.as_array();
    // Bind land-cover property rasters from the bundle.
    let alb_grid_ro = properties_bundle.alb_grid.bind(py).readonly();
    let emis_grid_ro = properties_bundle.emis_grid.bind(py).readonly();
    let tgk_grid_ro = properties_bundle.tgk_grid.bind(py).readonly();
    let tstart_grid_ro = properties_bundle.tstart_grid.bind(py).readonly();
    let tmaxlst_grid_ro = properties_bundle.tmaxlst_grid.bind(py).readonly();
    let alb_grid_v = alb_grid_ro.as_array();
    let emis_grid_v = emis_grid_ro.as_array();
    let tgk_grid_v = tgk_grid_ro.as_array();
    let tstart_grid_v = tstart_grid_ro.as_array();
    let tmaxlst_grid_v = tmaxlst_grid_ro.as_array();
    let buildings_v = buildings.as_array();
    let lc_grid_v = lc_grid.as_ref().map(|a| a.as_array());
    // Bind thermal state arrays from the bundle. Same lifetime pattern as the
    // SvfBundle: the *_ro binders outlive the *_v ArrayView2s.
    let tgmap1_ro = state_bundle.tgmap1.bind(py).readonly();
    let tgmap1_e_ro = state_bundle.tgmap1_e.bind(py).readonly();
    let tgmap1_s_ro = state_bundle.tgmap1_s.bind(py).readonly();
    let tgmap1_w_ro = state_bundle.tgmap1_w.bind(py).readonly();
    let tgmap1_n_ro = state_bundle.tgmap1_n.bind(py).readonly();
    let tgout1_ro = state_bundle.tgout1.bind(py).readonly();
    let tgmap1_v = tgmap1_ro.as_array();
    let tgmap1_e_v = tgmap1_e_ro.as_array();
    let tgmap1_s_v = tgmap1_s_ro.as_array();
    let tgmap1_w_v = tgmap1_w_ro.as_array();
    let tgmap1_n_v = tgmap1_n_ro.as_array();
    let tgout1_v = tgout1_ro.as_array();
    let firstdaytime = state_bundle.firstdaytime;
    let timeadd = state_bundle.timeadd;
    let timestep_dec = state_bundle.timestep_dec;

    // Bind ground-scheme arrays from the bundle (if the 2026a scheme is active).
    // Same lifetime pattern as the other bundles: the *_ro binders outlive the
    // *_v ArrayView2s. Absent bundle → all None → classic path runs.
    let sch = ground_scheme;
    let sch_timestep_s = sch.map(|s| s.timestep_s);
    let sch_tg_ro = sch.map(|s| s.tg.bind(py).readonly());
    let sch_tm_ro = sch.map(|s| s.tm.bind(py).readonly());
    let sch_rn_ro = sch.map(|s| s.rn.bind(py).readonly());
    let sch_rn_past_ro = sch.map(|s| s.rn_past.bind(py).readonly());
    let sch_g_ro = sch.map(|s| s.g.bind(py).readonly());
    let sch_cap_ro = sch.map(|s| s.cap.bind(py).readonly());
    let sch_diff_ro = sch.map(|s| s.diff.bind(py).readonly());
    let sch_a1_ro = sch.map(|s| s.a1.bind(py).readonly());
    let sch_a2_ro = sch.map(|s| s.a2.bind(py).readonly());
    let sch_a3_ro = sch.map(|s| s.a3.bind(py).readonly());
    let sch_lc_ro = sch.map(|s| s.lc_grid.bind(py).readonly());
    let sch_shadow_past_ro = sch.map(|s| s.shadow_past.bind(py).readonly());
    let sch_tg_v = sch_tg_ro.as_ref().map(|a| a.as_array());
    let sch_tm_v = sch_tm_ro.as_ref().map(|a| a.as_array());
    let sch_rn_v = sch_rn_ro.as_ref().map(|a| a.as_array());
    let sch_rn_past_v = sch_rn_past_ro.as_ref().map(|a| a.as_array());
    let sch_g_v = sch_g_ro.as_ref().map(|a| a.as_array());
    let sch_cap_v = sch_cap_ro.as_ref().map(|a| a.as_array());
    let sch_diff_v = sch_diff_ro.as_ref().map(|a| a.as_array());
    let sch_a1_v = sch_a1_ro.as_ref().map(|a| a.as_array());
    let sch_a2_v = sch_a2_ro.as_ref().map(|a| a.as_array());
    let sch_a3_v = sch_a3_ro.as_ref().map(|a| a.as_array());
    let sch_lc_v = sch_lc_ro.as_ref().map(|a| a.as_array());
    let sch_shadow_past_v = sch_shadow_past_ro.as_ref().map(|a| a.as_array());
    let use_scheme = sch.is_some();

    // Borrow anisotropic arrays (if provided)
    let shmat_v = shmat.as_ref().map(|a| a.as_array());
    let vegshmat_v = vegshmat.as_ref().map(|a| a.as_array());
    let vbshmat_v = vbshmat.as_ref().map(|a| a.as_array());
    let output_mask_bits = output_mask.unwrap_or(OUT_ALL);
    let want_shadow = (output_mask_bits & OUT_SHADOW) != 0;
    let want_kdown = (output_mask_bits & OUT_KDOWN) != 0;
    let want_kup = (output_mask_bits & OUT_KUP) != 0;
    let want_ldown = (output_mask_bits & OUT_LDOWN) != 0;
    let want_lup = (output_mask_bits & OUT_LUP) != 0;

    if config.has_walls && (wall_ht_v.is_none() || wall_asp_v.is_none()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "config.has_walls=true requires both wall_ht and wall_asp inputs",
        ));
    }

    // Validate anisotropic shadow-matrix dimensions match the DSM grid before
    // indexing them. Without this check a mis-sized array silently reads OOB
    // garbage (or aborts via wgpu) inside the hot loop.
    let dsm_shape = dsm_v.dim();
    let n_patches = patch_lut_for_option_cached(weather.patch_option)
        .altitudes
        .len();
    let expected_pack = n_patches.div_ceil(8);
    for (name, opt) in [
        ("shmat", shmat_v.as_ref()),
        ("vegshmat", vegshmat_v.as_ref()),
        ("vbshmat", vbshmat_v.as_ref()),
    ] {
        if let Some(arr) = opt {
            let (rows, cols, depth) = arr.dim();
            if (rows, cols) != dsm_shape {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "{name} dim ({}, {}, _) does not match DSM dim {:?}",
                    rows, cols, dsm_shape
                )));
            }
            // A stale precomputed shadow-matrix file generated with a different
            // patch option would otherwise index out of range deep in the
            // radiation loop and abort the process (panic = "abort").
            if depth != expected_pack {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "{name} packed patch depth {depth} does not match patch_option {} \
                     ({n_patches} patches -> {expected_pack} packed bytes); the precomputed \
                     shadow matrices were generated with a different patch option — \
                     recompute the SVF/shadow cache (force_recompute=True)",
                    weather.patch_option
                )));
            }
        }
    }
    // Extract GVF cache reference (pure Rust data) before releasing the GIL
    let gvf_cache_inner = gvf_cache.map(|c| &c.inner);

    // SAFETY: All array views borrow from PyReadonlyArray parameters that are alive
    // for the entire function call. Releasing the GIL only allows other Python
    // threads to run — it does not invalidate our borrows or trigger GC of the
    // backing numpy arrays.
    let raw = unsafe {
        allow_threads_unchecked(py, || {
            let shape = dsm_v.dim();

            // Wall aspect in radians for shadows
            let wall_asp_rad: Option<Array2<f32>> = wall_asp_v.map(|a| a.mapv(|d| d * PI / 180.0));
            let wall_asp_rad_view = wall_asp_rad.as_ref().map(|a| a.view());

            // ── Step 1: Shadows ──────────────────────────────────────────────────
            let t_shadow = Instant::now();
            let shadow_result: ShadowingResultRust = calculate_shadows_rust(
                weather.sun_azimuth,
                weather.sun_altitude,
                config.pixel_size,
                config.max_height,
                dsm_v,
                if config.use_veg { cdsm_v } else { None },
                if config.use_veg { tdsm_v } else { None },
                if config.use_veg { bush_v } else { None },
                if config.has_walls { wall_ht_v } else { None },
                if config.has_walls {
                    wall_asp_rad_view
                } else {
                    None
                },
                None,  // walls_scheme
                None,  // aspect_scheme
                false, // need_full_wall_outputs (pipeline only needs wall_sun)
                3.0,   // min_sun_altitude
                config.max_shadow_distance_m,
            );

            // Combine shadows with vegetation transmissivity
            let bldg_sh = &shadow_result.bldg_sh;
            let shadow = if config.use_veg {
                let veg_sh = &shadow_result.veg_sh;
                bldg_sh - &((1.0 - veg_sh) * (1.0 - weather.psi))
            } else {
                bldg_sh.clone()
            };
            let shadow_f32 = shadow;

            let wallsun = shadow_result
                .wall_sun
                .unwrap_or_else(|| Array2::zeros(shape));

            let shadow_dur = t_shadow.elapsed();

            // ── Step 2: Ground Temperature ───────────────────────────────────────
            let t_ground = Instant::now();
            let ground: GroundTempResult = compute_ground_temperature_pure(
                weather.sun_altitude,
                weather.altmax,
                weather.dectime,
                weather.snup,
                weather.global_rad,
                weather.rad_g0,
                weather.zen_deg,
                tgk_grid_v,
                tstart_grid_v,
                tmaxlst_grid_v,
                config.tgk_wall,
                config.tstart_wall,
                config.tmaxlst_wall,
            );

            let ground_dur = t_ground.elapsed();

            // At night (sun below horizon), UMEP Python zeros Tgwall and Tg.
            // The sinusoidal model keeps producing non-zero values after sunrise
            // (since dectime > snup_frac even after sunset), but UMEP's runner
            // explicitly overrides them to zero when altitude <= 0.
            let ground = if !weather.is_daytime {
                GroundTempResult {
                    tg: Array2::<f32>::zeros(ground.tg.dim()),
                    tg_wall: 0.0,
                    ci_tg: ground.ci_tg,
                }
            } else {
                ground
            };

            // ── UMEP 2026a ground-surface scheme (opt-in) ────────────────────────
            // When active, the force-restore/OHM surface temperature replaces the
            // classic sinusoidal Tg, and the solid-angle outgoing-longwave march
            // supplies Lup / gvfalb* / directional side longwave (replacing the
            // GVF + thermal-delay steps). The classic wall temperature
            // (ground.tg_wall) is retained. Ordering follows Solweig_2026a_calc.
            if use_scheme {
                let timestep_s = sch_timestep_s.expect("scheme: timestep_s");
                let tg_state = sch_tg_v.expect("scheme: tg");
                let tm_state = sch_tm_v.expect("scheme: tm");
                let rn_state = sch_rn_v.expect("scheme: rn");
                let rn_past_state = sch_rn_past_v.expect("scheme: rn_past");
                let g_state = sch_g_v.expect("scheme: g");
                let cap_v = sch_cap_v.expect("scheme: cap");
                let diff_v = sch_diff_v.expect("scheme: diff");
                let a1_v = sch_a1_v.expect("scheme: a1");
                let a2_v = sch_a2_v.expect("scheme: a2");
                let a3_v = sch_a3_v.expect("scheme: a3");
                let lc_scheme_v = sch_lc_v.expect("scheme: lc_grid");
                let shadow_past_v = sch_shadow_past_v.expect("scheme: shadow_past");

                // Upstream zeros the shadow grid at night (altitude <= 0); both
                // the surface-temperature damping mask and the march consume it.
                let scheme_shadow = if weather.is_daytime {
                    shadow_f32.clone()
                } else {
                    Array2::<f32>::zeros(shape)
                };

                let esky = compute_esky(weather.ta, weather.rh);
                let zen_rad = weather.sun_zenith * PI / 180.0;
                let f_sh = cylindric_wedge_pure_masked(zen_rad, svfalfa_v, Some(valid_v));
                let sin_alt = (weather.sun_altitude * PI / 180.0).sin();
                let rad_i = weather.direct_rad;
                let rad_d = weather.diffuse_rad;
                let rad_g = weather.global_rad;
                let psi = weather.psi;
                let cyl = human.is_standing;

                let use_aniso = config.use_anisotropic
                    && shmat_v.is_some()
                    && vegshmat_v.is_some()
                    && vbshmat_v.is_some();

                // Anisotropic setup (Perez luminance + CI-corrected esky). The
                // scheme uses the CPU sky solver only; GPU aniso for the scheme
                // is future work and produces identical results.
                let (lv_arr, esky_a) = if use_aniso {
                    let lv = crate::perez::perez_v3(
                        weather.zen_deg,
                        weather.sun_azimuth,
                        weather.diffuse_rad,
                        weather.direct_rad,
                        weather.jday,
                        weather.patch_option,
                    );
                    let ci = weather.clearness_index;
                    let ea = if ci < 0.95 { ci * esky + (1.0 - ci) } else { esky };
                    (Some(lv), ea)
                } else {
                    (None, esky)
                };

                // Diffuse shortwave into each cell (drad)
                let drad = if use_aniso {
                    let lv = lv_arr.as_ref().expect("lv gated by use_aniso");
                    let shmat_a = shmat_v.expect("shmat: gated by use_aniso");
                    let vegshmat_a = vegshmat_v.expect("vegshmat: gated by use_aniso");
                    let ani_lum = compute_ani_lum_from_packed(
                        shmat_a,
                        vegshmat_a,
                        lv.column(2),
                        psi,
                        valid_v,
                    );
                    ani_lum.mapv(|v| v * rad_d)
                } else {
                    svfbuveg_v.mapv(|sv| rad_d * sv)
                };

                // Kdown (identical formula to the classic path)
                let kdown = compute_kdown(
                    rad_i,
                    rad_d,
                    rad_g,
                    scheme_shadow.view(),
                    sin_alt,
                    svfbuveg_v,
                    config.albedo_wall,
                    f_sh.view(),
                    drad.view(),
                    valid_v,
                );

                // Isotropic Ldown — the value fed to the surface-temperature ODE,
                // the march, and Lside_veg_v2026 (the anisotropic sky reassigns
                // the Tmrt Ldown below, matching upstream).
                let ldown_iso = compute_ldown(
                    esky,
                    weather.ta,
                    ground.tg_wall,
                    svf_v,
                    svf_veg_v,
                    svf_aveg_v,
                    config.emis_wall,
                    weather.clearness_index,
                    valid_v,
                );

                // Force-restore/OHM surface temperature step → new Tg + fluxes.
                let st = surface_temperature_calc_pure(
                    kdown.view(),
                    ldown_iso.view(),
                    rn_state,
                    rn_past_state,
                    g_state,
                    tg_state,
                    tm_state,
                    alb_grid_v,
                    emis_grid_v,
                    cap_v,
                    diff_v,
                    lc_scheme_v,
                    a1_v,
                    a2_v,
                    a3_v,
                    timestep_s,
                    weather.rh,
                    scheme_shadow.view(),
                    shadow_past_v,
                );

                // Outgoing-longwave solid-angle march. Walls default to zeros
                // when the scene has none (no wall longwave contribution).
                let walls_owned: Array2<f32> = if config.has_walls {
                    wall_ht_v
                        .expect("wall_ht: gated by config.has_walls=true")
                        .to_owned()
                } else {
                    Array2::<f32>::zeros(shape)
                };
                let march = outgoing_longwave_calc_pure(
                    st.tg.view(),
                    ground.tg_wall,
                    weather.ta,
                    ldown_iso.view(),
                    emis_grid_v,
                    alb_grid_v,
                    buildings_v,
                    scheme_shadow.view(),
                    wallsun.view(),
                    walls_owned.view(),
                    config.pixel_size,
                );

                // Kup from the march's albedo view factors (gvfalbsun→gvfalb,
                // gvfalbtot→gvfalbnosh).
                let (kup, kup_e, kup_s, kup_w, kup_n) = compute_kup(
                    rad_i,
                    rad_d,
                    rad_g,
                    weather.sun_altitude,
                    svfbuveg_v,
                    config.albedo_wall,
                    f_sh.view(),
                    march.gvfalbsun.view(),
                    march.gvfalbsun_e.view(),
                    march.gvfalbsun_s.view(),
                    march.gvfalbsun_w.view(),
                    march.gvfalbsun_n.view(),
                    march.gvfalbtot.view(),
                    march.gvfalbtot_e.view(),
                    march.gvfalbtot_s.view(),
                    march.gvfalbtot_w.view(),
                    march.gvfalbtot_n.view(),
                    valid_v,
                );

                // Lside_veg_v2026: reflection loses the Lup term; the ground
                // longwave leaves Lside entirely (supplied by the march's
                // gvfLside*). The anisotropic branch returns zeros. The Lup*
                // slots are unused by V2026 — Ldown stands in.
                let lside = lside_veg_variant_pure(
                    LsideVariant::V2026,
                    svf_s_v,
                    svf_w_v,
                    svf_n_v,
                    svf_e_v,
                    svf_veg_e_v,
                    svf_veg_s_v,
                    svf_veg_w_v,
                    svf_veg_n_v,
                    svf_aveg_e_v,
                    svf_aveg_s_v,
                    svf_aveg_w_v,
                    svf_aveg_n_v,
                    weather.sun_azimuth,
                    weather.sun_altitude,
                    weather.ta,
                    ground.tg_wall,
                    SBC,
                    config.emis_wall,
                    ldown_iso.view(),
                    esky,
                    0.0,
                    f_sh.view(),
                    weather.clearness_index,
                    ldown_iso.view(),
                    ldown_iso.view(),
                    ldown_iso.view(),
                    ldown_iso.view(),
                    use_aniso,
                    Some(valid_v),
                );

                let (kdown_out, ldown_out, kside_dirs_sum, lside_dirs_sum, kside_total, lside_total) =
                    if use_aniso {
                        let shmat_a = shmat_v.expect("shmat: gated by use_aniso");
                        let vegshmat_a = vegshmat_v.expect("vegshmat: gated by use_aniso");
                        let vbshmat_a = vbshmat_v.expect("vbshmat: gated by use_aniso");
                        let lv = lv_arr.as_ref().expect("lv gated by use_aniso");
                        let patch_lut = patch_lut_for_option_cached(weather.patch_option);
                        let steradians_arr = ArrayView1::from(patch_lut.steradians.as_slice());
                        let asvf_cache = asvf_for_svf_cached(svf_v);
                        let asvf_arr = ArrayView2::from_shape(shape, asvf_cache.as_slice())
                            .expect("ASVF cache shape matches DSM");

                        let ani = anisotropic_sky_pure(
                            shmat_a,
                            vegshmat_a,
                            vbshmat_a,
                            weather.sun_altitude,
                            weather.sun_azimuth,
                            esky_a,
                            weather.ta,
                            cyl,
                            false,
                            config.albedo_wall,
                            ground.tg_wall,
                            config.emis_wall,
                            rad_i,
                            rad_d,
                            asvf_arr,
                            lv.view(),
                            steradians_arr,
                            march.gvf_lup.view(),
                            lv.view(),
                            scheme_shadow.view(),
                            kup_e.view(),
                            kup_s.view(),
                            kup_w.view(),
                            kup_n.view(),
                            None,
                            None,
                            Some(valid_v),
                        );
                        let ani_kside_dirs_sum = side_sum_from_directional(
                            ani.knorth.view(),
                            ani.keast.view(),
                            ani.ksouth.view(),
                            ani.kwest.view(),
                            valid_v,
                        );
                        // Directional Lside comes entirely from the march
                        // (Lside_veg_v2026 anisotropic returns zeros).
                        let lside_dirs_sum = side_sum_from_directional(
                            march.gvf_lside_n.view(),
                            march.gvf_lside_e.view(),
                            march.gvf_lside_s.view(),
                            march.gvf_lside_w.view(),
                            valid_v,
                        );
                        // 2026a Tmrt composition: the Fcyl longwave term is
                        // Lside = mean(directional) + anisotropic-sky Lside
                        // (Solweig_2026a_calc L790 + L882).
                        let lside_total = &ani.lside + &lside_dirs_sum.mapv(|v| v * 0.25);
                        (
                            kdown,
                            ani.ldown,
                            ani_kside_dirs_sum,
                            lside_dirs_sum,
                            ani.kside,
                            lside_total,
                        )
                    } else {
                        let kside = kside_veg_isotropic_pure(
                            rad_i,
                            rad_d,
                            rad_g,
                            scheme_shadow.view(),
                            svf_s_v,
                            svf_w_v,
                            svf_n_v,
                            svf_e_v,
                            svf_veg_e_v,
                            svf_veg_s_v,
                            svf_veg_w_v,
                            svf_veg_n_v,
                            weather.sun_azimuth,
                            weather.sun_altitude,
                            psi,
                            0.0,
                            config.albedo_wall,
                            f_sh.view(),
                            kup_e.view(),
                            kup_s.view(),
                            kup_w.view(),
                            kup_n.view(),
                            cyl,
                            Some(valid_v),
                        );
                        let kside_dirs_sum = side_sum_from_directional(
                            kside.knorth.view(),
                            kside.keast.view(),
                            kside.ksouth.view(),
                            kside.kwest.view(),
                            valid_v,
                        );
                        // Directional Lside = march ground/wall side longwave +
                        // Lside_veg_v2026 sky/wall/veg terms.
                        let least_tot = &march.gvf_lside_e + &lside.least;
                        let lsouth_tot = &march.gvf_lside_s + &lside.lsouth;
                        let lwest_tot = &march.gvf_lside_w + &lside.lwest;
                        let lnorth_tot = &march.gvf_lside_n + &lside.lnorth;
                        let lside_dirs_sum = side_sum_from_directional(
                            lnorth_tot.view(),
                            least_tot.view(),
                            lsouth_tot.view(),
                            lwest_tot.view(),
                            valid_v,
                        );
                        (
                            kdown,
                            ldown_iso,
                            kside_dirs_sum,
                            lside_dirs_sum,
                            kside.kside_i,
                            Array2::<f32>::zeros(shape),
                        )
                    };

                // Lup (Fup upwelling) is the march output — no thermal delay.
                let lup = march.gvf_lup;

                let tmrt = compute_tmrt_from_dir_sums_pure(
                    kdown_out.view(),
                    kup.view(),
                    ldown_out.view(),
                    lup.view(),
                    kside_dirs_sum.view(),
                    lside_dirs_sum.view(),
                    kside_total.view(),
                    lside_total.view(),
                    human.abs_k,
                    human.abs_l,
                    human.is_standing,
                    use_aniso,
                );

                return TimestepResultRaw {
                    tmrt,
                    // The scheme carries shadow_past forward from this shadow;
                    // return it whenever the caller asked for shadow output.
                    shadow: if want_shadow { Some(scheme_shadow) } else { None },
                    kdown: if want_kdown { Some(kdown_out) } else { None },
                    kup: if want_kup { Some(kup) } else { None },
                    ldown: if want_ldown { Some(ldown_out) } else { None },
                    lup: if want_lup { Some(lup) } else { None },
                    // Thermal-delay state is untouched by the scheme (the
                    // force-restore ODE carries the inertia); pass it through.
                    timeadd,
                    tgmap1: tgmap1_v.to_owned(),
                    tgmap1_e: tgmap1_e_v.to_owned(),
                    tgmap1_s: tgmap1_s_v.to_owned(),
                    tgmap1_w: tgmap1_w_v.to_owned(),
                    tgmap1_n: tgmap1_n_v.to_owned(),
                    tgout1: tgout1_v.to_owned(),
                    tg: Some(st.tg),
                    rn: Some(st.rn),
                    rn_past: Some(st.rn_past),
                    g: Some(st.g),
                };
            }

            // ── Step 3: GVF ─────────────────────────────────────────────────────
            let t_gvf = Instant::now();
            let first = {
                let h = human.height.round();
                if h == 0.0 {
                    1.0
                } else {
                    h
                }
            };
            let second = (human.height * 20.0).round();

            let gvf: GvfResultPure = if config.has_walls {
                if let Some(cache) = gvf_cache_inner {
                    // Gated by config.has_walls=true (validated at entry).
                    let wh = wall_ht_v.expect("wall_ht: gated by config.has_walls=true");
                    // Try GPU first, fall back to CPU
                    #[cfg(feature = "gpu")]
                    {
                        if let Some(ctx) = get_gvf_gpu_context() {
                            match gvf_calc_with_cache_gpu(
                                ctx,
                                cache,
                                wallsun.view(),
                                wh,
                                buildings_v,
                                shadow_f32.view(),
                                ground.tg.view(),
                                ground.tg_wall,
                                weather.ta,
                                emis_grid_v,
                                config.emis_wall,
                                alb_grid_v,
                                SBC,
                                config.albedo_wall,
                                weather.ta,
                                lc_grid_v,
                                lc_grid_v.is_some(),
                            ) {
                                Ok(result) => {
                                    crate::shadowing::record_gpu_dispatch();
                                    result
                                }
                                Err(e) => {
                                    eprintln!("[GPU] GVF GPU dispatch failed, falling back to CPU: {}", e);
                                    crate::shadowing::record_gpu_fallback();
                                    GVF_GPU_ENABLED.store(false, std::sync::atomic::Ordering::Relaxed);
                                    gvf_calc_with_cache(
                                        cache,
                                        wallsun.view(),
                                        wh,
                                        buildings_v,
                                        shadow_f32.view(),
                                        ground.tg.view(),
                                        ground.tg_wall,
                                        weather.ta,
                                        emis_grid_v,
                                        config.emis_wall,
                                        alb_grid_v,
                                        SBC,
                                        config.albedo_wall,
                                        weather.ta,
                                        lc_grid_v,
                                        lc_grid_v.is_some(),
                                    )
                                }
                            }
                        } else {
                            gvf_calc_with_cache(
                                cache,
                                wallsun.view(),
                                wh,
                                buildings_v,
                                shadow_f32.view(),
                                ground.tg.view(),
                                ground.tg_wall,
                                weather.ta,
                                emis_grid_v,
                                config.emis_wall,
                                alb_grid_v,
                                SBC,
                                config.albedo_wall,
                                weather.ta,
                                lc_grid_v,
                                lc_grid_v.is_some(),
                            )
                        }
                    }
                    #[cfg(not(feature = "gpu"))]
                    gvf_calc_with_cache(
                        cache,
                        wallsun.view(),
                        wh,
                        buildings_v,
                        shadow_f32.view(),
                        ground.tg.view(),
                        ground.tg_wall,
                        weather.ta,
                        emis_grid_v,
                        config.emis_wall,
                        alb_grid_v,
                        SBC,
                        config.albedo_wall,
                        weather.ta, // twater = ta
                        lc_grid_v,
                        lc_grid_v.is_some(),
                    )
                } else {
                    // Full GVF (first timestep or no cache).
                    // wall_ht_v and wall_asp_v are guaranteed Some here: this
                    // branch is gated by config.has_walls=true, which we
                    // validate at function entry (line ~990) rejecting calls
                    // without both arrays.
                    let wh = wall_ht_v.expect("wall_ht: gated by config.has_walls=true");
                    gvf_calc_pure(
                        wallsun.view(),
                        wh,
                        buildings_v,
                        config.pixel_size,
                        shadow_f32.view(),
                        first,
                        second,
                        wall_asp_v.expect("wall_asp: gated by config.has_walls=true"),
                        ground.tg.view(),
                        ground.tg_wall,
                        weather.ta,
                        emis_grid_v,
                        config.emis_wall,
                        alb_grid_v,
                        SBC,
                        config.albedo_wall,
                        weather.ta, // twater = ta
                        lc_grid_v,
                        lc_grid_v.is_some(),
                    )
                }
            } else {
                // Simplified GVF (no walls) - compute inline
                let gvf_simple = 1.0 - &svf_v;
                let tg_with_shadow = &ground.tg * &shadow_f32;
                // Lup = emis × SBC × (Ta + Tg_shadow + 273.15)^4
                let lup_simple = {
                    let mut arr = Array2::<f32>::zeros(shape);
                    Zip::indexed(&mut arr).par_for_each(|(r, c), out| {
                        if valid_v[[r, c]] == 0 {
                            *out = f32::NAN;
                            return;
                        }
                        let t = weather.ta + tg_with_shadow[[r, c]] + KELVIN_OFFSET;
                        *out = emis_grid_v[[r, c]] * SBC * t.powi(4);
                    });
                    arr
                };
                let gvfalb_simple = &alb_grid_v * &gvf_simple;

                GvfResultPure {
                    gvf_lup: lup_simple.clone(),
                    gvfalb: gvfalb_simple.clone(),
                    gvfalbnosh: Some(alb_grid_v.to_owned()),
                    gvf_lup_e: lup_simple.clone(),
                    gvfalb_e: gvfalb_simple.clone(),
                    gvfalbnosh_e: Some(alb_grid_v.to_owned()),
                    gvf_lup_s: lup_simple.clone(),
                    gvfalb_s: gvfalb_simple.clone(),
                    gvfalbnosh_s: Some(alb_grid_v.to_owned()),
                    gvf_lup_w: lup_simple.clone(),
                    gvfalb_w: gvfalb_simple.clone(),
                    gvfalbnosh_w: Some(alb_grid_v.to_owned()),
                    gvf_lup_n: lup_simple.clone(),
                    gvfalb_n: gvfalb_simple,
                    gvfalbnosh_n: Some(alb_grid_v.to_owned()),
                    gvf_sum: Some(Array2::zeros(shape)),
                    gvf_norm: Some(Array2::ones(shape)),
                }
            };

            let gvf_dur = t_gvf.elapsed();

            // ── Step 4: Thermal Delay ────────────────────────────────────────────
            let t_delay = Instant::now();
            let tg_temp = &ground.tg * &shadow_f32 + weather.ta;

            let delay = ts_wave_delay_batch_pure(
                gvf.gvf_lup.view(),
                gvf.gvf_lup_e.view(),
                gvf.gvf_lup_s.view(),
                gvf.gvf_lup_w.view(),
                gvf.gvf_lup_n.view(),
                tg_temp.view(),
                firstdaytime,
                timeadd,
                timestep_dec,
                tgmap1_v,
                tgmap1_e_v,
                tgmap1_s_v,
                tgmap1_w_v,
                tgmap1_n_v,
                tgout1_v,
            );

            let delay_dur = t_delay.elapsed();

            // ── Step 5: Radiation ─────────────────────────────────────────────────
            let t_radiation = Instant::now();
            let esky = compute_esky(weather.ta, weather.rh);
            let sin_alt = (weather.sun_altitude * PI / 180.0).sin();
            let rad_i = weather.direct_rad;
            let rad_d = weather.diffuse_rad;
            let rad_g = weather.global_rad;
            let psi = weather.psi;
            let cyl = human.is_standing;

            // F_sh (cylindric wedge shadow fraction) — shared by both paths
            let zen_rad = weather.sun_zenith * PI / 180.0;
            let f_sh = cylindric_wedge_pure_masked(zen_rad, svfalfa_v, Some(valid_v));

            // Kup helper used in both isotropic and anisotropic paths.
            // In cached-GVF mode, read gvfalbnosh* directly from geometry cache to
            // avoid per-timestep cloning of those static arrays.
            let compute_kup_with =
                |gvfalbnosh: ArrayView2<f32>,
                 gvfalbnosh_e: ArrayView2<f32>,
                 gvfalbnosh_s: ArrayView2<f32>,
                 gvfalbnosh_w: ArrayView2<f32>,
                 gvfalbnosh_n: ArrayView2<f32>| {
                    compute_kup(
                        rad_i,
                        rad_d,
                        rad_g,
                        weather.sun_altitude,
                        svfbuveg_v,
                        config.albedo_wall,
                        f_sh.view(),
                        gvf.gvfalb.view(),
                        gvf.gvfalb_e.view(),
                        gvf.gvfalb_s.view(),
                        gvf.gvfalb_w.view(),
                        gvf.gvfalb_n.view(),
                        gvfalbnosh,
                        gvfalbnosh_e,
                        gvfalbnosh_s,
                        gvfalbnosh_w,
                        gvfalbnosh_n,
                        valid_v,
                    )
                };
            let compute_kup_all = || {
                if let Some(cache) = gvf_cache_inner {
                    compute_kup_with(
                        cache.cached_albnosh.view(),
                        cache.cached_albnosh_e.view(),
                        cache.cached_albnosh_s.view(),
                        cache.cached_albnosh_w.view(),
                        cache.cached_albnosh_n.view(),
                    )
                } else {
                    let gvfalbnosh = gvf
                        .gvfalbnosh
                        .as_ref()
                        .expect("gvfalbnosh missing without cache");
                    let gvfalbnosh_e = gvf
                        .gvfalbnosh_e
                        .as_ref()
                        .expect("gvfalbnosh_e missing without cache");
                    let gvfalbnosh_s = gvf
                        .gvfalbnosh_s
                        .as_ref()
                        .expect("gvfalbnosh_s missing without cache");
                    let gvfalbnosh_w = gvf
                        .gvfalbnosh_w
                        .as_ref()
                        .expect("gvfalbnosh_w missing without cache");
                    let gvfalbnosh_n = gvf
                        .gvfalbnosh_n
                        .as_ref()
                        .expect("gvfalbnosh_n missing without cache");
                    compute_kup_with(
                        gvfalbnosh.view(),
                        gvfalbnosh_e.view(),
                        gvfalbnosh_s.view(),
                        gvfalbnosh_w.view(),
                        gvfalbnosh_n.view(),
                    )
                }
            };

            // Branch: anisotropic vs isotropic
            let use_aniso = config.use_anisotropic
                && shmat_v.is_some()
                && vegshmat_v.is_some()
                && vbshmat_v.is_some();

            let (kup, kdown, ldown, kside_dirs_sum, lside_dirs_sum, kside_total, lside_total) =
                if use_aniso {
                    // === Anisotropic sky ===
                    // The three shadow matrices are guaranteed Some by use_aniso
                    // (lines just above check .is_some() on all three).
                    let shmat_a = shmat_v.expect("shmat: gated by use_aniso");
                    let vegshmat_a = vegshmat_v.expect("vegshmat: gated by use_aniso");
                    let vbshmat_a = vbshmat_v.expect("vbshmat: gated by use_aniso");

                    // Perez sky luminance distribution (computed in Rust — no Python round-trip)
                    let lv_arr = crate::perez::perez_v3(
                        weather.zen_deg,
                        weather.sun_azimuth,
                        weather.diffuse_rad,
                        weather.direct_rad,
                        weather.jday,
                        weather.patch_option,
                    );
                    let patch_lut = patch_lut_for_option_cached(weather.patch_option);
                    let patch_altitude_arr = ArrayView1::from(patch_lut.altitudes.as_slice());
                    let patch_azimuth_arr = ArrayView1::from(patch_lut.azimuths.as_slice());
                    let steradians_arr = ArrayView1::from(patch_lut.steradians.as_slice());
                    let patch_altitude_sin_arr =
                        ArrayView1::from(patch_lut.altitude_sin.as_slice());

                    // ASVF from SVF (arccos(sqrt(clip(svf, 0, 1)))) cached by SVF buffer.
                    // The cache key already includes (nrows, ncols, hash), so the cached
                    // buffer length equals shape.0 * shape.1 by construction. The expect
                    // is kept as a defensive guard with a precise diagnostic.
                    let asvf_cache = asvf_for_svf_cached(svf_v);
                    let asvf_arr =
                        ArrayView2::from_shape(shape, asvf_cache.as_slice()).unwrap_or_else(|e| {
                            panic!(
                                "ASVF cache shape mismatch: requested {:?}, buffer len {} ({})",
                                shape,
                                asvf_cache.len(),
                                e
                            )
                        });

                    // Esky anisotropic (Jonsson + CI correction)
                    let esky_a = {
                        let ci = weather.clearness_index;
                        if ci < 0.95 {
                            ci * esky + (1.0 - ci)
                        } else {
                            esky
                        }
                    };

                    // Full anisotropic sky calculation (ldown, kside, lside totals)
                    // Try GPU path first; fall back to CPU if unavailable.
                    let deg2rad = PI / 180.0;
                    #[allow(unused_variables)]
                    let (lum_chi, rad_tot) = if weather.sun_altitude > 0.0 {
                        let patch_luminance = lv_arr.column(2);
                        let mut rad_tot = 0.0f32;
                        let n_patches = patch_luminance.len();
                        for i in 0..n_patches {
                            rad_tot +=
                                patch_luminance[i] * steradians_arr[i] * patch_altitude_sin_arr[i];
                        }
                        if rad_tot > 0.0 {
                            (patch_luminance.mapv(|lum| (lum * rad_d) / rad_tot), rad_tot)
                        } else {
                            (Array1::<f32>::zeros(n_patches), 0.0)
                        }
                    } else {
                        (Array1::<f32>::zeros(lv_arr.shape()[0]), 0.0)
                    };

                    #[allow(unused_variables)]
                    let (_, esky_band) =
                        crate::emissivity_models::model2(&lv_arr, esky_a, weather.ta);

                    // Launch anisotropic GPU dispatch early, then compute Kup/lside_dirs
                    // while GPU work is in flight.
                    #[cfg(feature = "gpu")]
                    let mut gpu_ctx = None;
                    #[cfg(feature = "gpu")]
                    let mut gpu_pending = None;

                    #[cfg(feature = "gpu")]
                    if let Some(ctx) = get_aniso_gpu_context() {
                        match ctx.dispatch_begin(
                            shmat_a,
                            vegshmat_a,
                            vbshmat_a,
                            asvf_arr,
                            delay.lup.view(),
                            valid_v,
                            patch_altitude_arr,
                            patch_azimuth_arr,
                            steradians_arr,
                            esky_band.view(),
                            lum_chi.view(),
                            weather.sun_altitude,
                            weather.sun_azimuth,
                            weather.ta,
                            cyl,
                            config.albedo_wall,
                            ground.tg_wall,
                            config.emis_wall,
                            rad_i,
                            rad_d,
                            psi,
                            rad_tot,
                        ) {
                            Ok(pending) => {
                                gpu_ctx = Some(ctx);
                                gpu_pending = Some(pending);
                            }
                            Err(e) => {
                                eprintln!(
                                    "[GPU] Anisotropic dispatch begin failed: {}. CPU fallback.",
                                    e
                                );
                                crate::shadowing::record_gpu_fallback();
                            }
                        }
                    }

                    // Shared thermal side inputs (always needed in anisotropic mode).
                    let (kup, kup_e, kup_s, kup_w, kup_n) = compute_kup_all();

                    let lside_dirs_sum = lside_dirs_sum_aniso_from_lup(
                        delay.lup_e.view(),
                        delay.lup_s.view(),
                        delay.lup_w.view(),
                        delay.lup_n.view(),
                        valid_v,
                    );

                    #[cfg(feature = "gpu")]
                    let gpu_result = if let (Some(ctx), Some(pending)) = (gpu_ctx, gpu_pending) {
                        match ctx.dispatch_end(pending) {
                            Ok(gpu) => {
                                crate::shadowing::record_gpu_dispatch();
                                Some(gpu)
                            }
                            Err(e) => {
                                eprintln!(
                                    "[GPU] Anisotropic dispatch end failed: {}. CPU fallback.",
                                    e
                                );
                                crate::shadowing::record_gpu_fallback();
                                None
                            }
                        }
                    } else {
                        None
                    };

                    // Compute anisotropic sky: GPU path + CPU fallback
                    let mut used_gpu = false;
                    #[allow(unused_mut)]
                    let mut ani_ldown = Array2::<f32>::zeros(shape);
                    #[allow(unused_mut)]
                    let mut ani_lside = Array2::<f32>::zeros(shape);
                    #[allow(unused_mut)]
                    let mut ani_kside = Array2::<f32>::zeros(shape);
                    #[allow(unused_mut)]
                    let mut drad = Array2::<f32>::zeros(shape);
                    #[allow(unused_mut)]
                    let mut ani_kside_dirs_sum = Array2::<f32>::zeros(shape);

                    #[cfg(feature = "gpu")]
                    if let Some(gpu) = gpu_result {
                        // GPU path: derive kside and k-directional from GPU partial outputs
                        let kside_i = if cyl {
                            &shadow_f32 * rad_i * (weather.sun_altitude * deg2rad).cos()
                        } else {
                            Array2::<f32>::zeros(shape)
                        };
                        if weather.sun_altitude > 0.0 {
                            ani_kside = kside_i + &gpu.kside_partial;
                            // Ground-reflected shortwave to directional (both postures)
                            ani_kside_dirs_sum = kside_dirs_sum_aniso_from_kup(
                                kup_e.view(),
                                kup_s.view(),
                                kup_w.view(),
                                kup_n.view(),
                                valid_v,
                            );
                            // Box posture: add direct beam to directional faces
                            if !cyl {
                                let cos_alt =
                                    (weather.sun_altitude * deg2rad).cos();
                                let azi = weather.sun_azimuth;
                                let mut kdir_box = Array2::<f32>::zeros(shape);
                                // East face
                                if azi > 360.0 || azi <= 180.0 {
                                    kdir_box += &(&shadow_f32
                                        * rad_i
                                        * cos_alt
                                        * (azi * deg2rad).sin());
                                }
                                // South face
                                if azi > 90.0 && azi <= 270.0 {
                                    kdir_box += &(&shadow_f32
                                        * rad_i
                                        * cos_alt
                                        * ((azi - 90.0) * deg2rad).sin());
                                }
                                // West face
                                if azi > 180.0 && azi <= 360.0 {
                                    kdir_box += &(&shadow_f32
                                        * rad_i
                                        * cos_alt
                                        * ((azi - 180.0) * deg2rad).sin());
                                }
                                // North face
                                if azi <= 90.0 || azi > 270.0 {
                                    kdir_box += &(&shadow_f32
                                        * rad_i
                                        * cos_alt
                                        * ((azi - 270.0) * deg2rad).sin());
                                }
                                ani_kside_dirs_sum += &kdir_box;
                            }
                        }
                        ani_ldown = gpu.ldown;
                        ani_lside = gpu.lside;
                        drad = gpu.drad;
                        used_gpu = true;
                    }

                    if !used_gpu {
                        // drad via direct accumulation from packed shadow matrices.
                        // Shadow matrices are bitpacked: 1 bit per patch, 8 patches per byte.
                        // ani_lum = sum_i((sh_i - (1 - veg_i) * (1 - psi)) * lv_i)
                        let lv_col2 = lv_arr.column(2);
                        let ani_lum =
                            compute_ani_lum_from_packed(shmat_a, vegshmat_a, lv_col2, psi, valid_v);
                        drad = ani_lum.mapv(|v| v * rad_d);

                        let ani = anisotropic_sky_pure(
                            shmat_a,
                            vegshmat_a,
                            vbshmat_a,
                            weather.sun_altitude,
                            weather.sun_azimuth,
                            esky_a,
                            weather.ta,
                            cyl,
                            false, // wall_scheme
                            config.albedo_wall,
                            ground.tg_wall,
                            config.emis_wall,
                            rad_i,
                            rad_d,
                            asvf_arr,
                            lv_arr.view(),
                            steradians_arr,
                            delay.lup.view(),
                            lv_arr.view(),
                            shadow_f32.view(),
                            kup_e.view(),
                            kup_s.view(),
                            kup_w.view(),
                            kup_n.view(),
                            None, // voxel_table
                            None, // voxel_maps
                            Some(valid_v),
                        );
                        ani_ldown = ani.ldown;
                        ani_lside = ani.lside;
                        ani_kside = ani.kside;
                        ani_kside_dirs_sum = side_sum_from_directional(
                            ani.knorth.view(),
                            ani.keast.view(),
                            ani.ksouth.view(),
                            ani.kwest.view(),
                            valid_v,
                        );
                    }

                    // Kdown (shared formula, but with anisotropic drad)
                    let kdown = compute_kdown(
                        rad_i,
                        rad_d,
                        rad_g,
                        shadow_f32.view(),
                        sin_alt,
                        svfbuveg_v,
                        config.albedo_wall,
                        f_sh.view(),
                        drad.view(),
                        valid_v,
                    );

                    // From anisotropic: ldown from ani_sky, lside from lside_veg, kside from ani_sky
                    (
                        kup,
                        kdown,
                        ani_ldown,
                        ani_kside_dirs_sum,
                        lside_dirs_sum,
                        ani_kside,
                        ani_lside,
                    )
                } else {
                    // === Isotropic sky ===
                    let (kup, kup_e, kup_s, kup_w, kup_n) = compute_kup_all();

                    // drad (isotropic diffuse)
                    let drad = svfbuveg_v.mapv(|sv| rad_d * sv);

                    // Ldown
                    let ldown = compute_ldown(
                        esky,
                        weather.ta,
                        ground.tg_wall,
                        svf_v,
                        svf_veg_v,
                        svf_aveg_v,
                        config.emis_wall,
                        weather.clearness_index,
                        valid_v,
                    );

                    // kside_veg (isotropic)
                    let kside = kside_veg_isotropic_pure(
                        rad_i,
                        rad_d,
                        rad_g,
                        shadow_f32.view(),
                        svf_s_v,
                        svf_w_v,
                        svf_n_v,
                        svf_e_v,
                        svf_veg_e_v,
                        svf_veg_s_v,
                        svf_veg_w_v,
                        svf_veg_n_v,
                        weather.sun_azimuth,
                        weather.sun_altitude,
                        psi,
                        0.0, // t (instrument offset)
                        config.albedo_wall,
                        f_sh.view(),
                        kup_e.view(),
                        kup_s.view(),
                        kup_w.view(),
                        kup_n.view(),
                        cyl,
                        Some(valid_v),
                    );

                    // lside_veg (isotropic)
                    let lside = lside_veg_pure(
                        svf_s_v,
                        svf_w_v,
                        svf_n_v,
                        svf_e_v,
                        svf_veg_e_v,
                        svf_veg_s_v,
                        svf_veg_w_v,
                        svf_veg_n_v,
                        svf_aveg_e_v,
                        svf_aveg_s_v,
                        svf_aveg_w_v,
                        svf_aveg_n_v,
                        weather.sun_azimuth,
                        weather.sun_altitude,
                        weather.ta,
                        ground.tg_wall,
                        SBC,
                        config.emis_wall,
                        ldown.view(),
                        esky,
                        0.0, // t
                        f_sh.view(),
                        weather.clearness_index,
                        delay.lup_e.view(),
                        delay.lup_s.view(),
                        delay.lup_w.view(),
                        delay.lup_n.view(),
                        false, // isotropic
                        Some(valid_v),
                    );

                    // Kdown
                    let kdown = compute_kdown(
                        rad_i,
                        rad_d,
                        rad_g,
                        shadow_f32.view(),
                        sin_alt,
                        svfbuveg_v,
                        config.albedo_wall,
                        f_sh.view(),
                        drad.view(),
                        valid_v,
                    );

                    // Isotropic: kside_total = kside_i, lside_total = zeros
                    let kside_dirs_sum = side_sum_from_directional(
                        kside.knorth.view(),
                        kside.keast.view(),
                        kside.ksouth.view(),
                        kside.kwest.view(),
                        valid_v,
                    );
                    let lside_dirs_sum = side_sum_from_directional(
                        lside.lnorth.view(),
                        lside.least.view(),
                        lside.lsouth.view(),
                        lside.lwest.view(),
                        valid_v,
                    );
                    (
                        kup,
                        kdown,
                        ldown,
                        kside_dirs_sum,
                        lside_dirs_sum,
                        kside.kside_i,
                        Array2::<f32>::zeros(shape),
                    )
                };

            let radiation_dur = t_radiation.elapsed();

            // ── Step 6: Tmrt ─────────────────────────────────────────────────────
            let t_tmrt = Instant::now();
            let tmrt = compute_tmrt_from_dir_sums_pure(
                kdown.view(),
                kup.view(),
                ldown.view(),
                delay.lup.view(),
                kside_dirs_sum.view(),
                lside_dirs_sum.view(),
                kside_total.view(),
                lside_total.view(),
                human.abs_k,
                human.abs_l,
                human.is_standing,
                use_aniso,
            );
            let tmrt_dur = t_tmrt.elapsed();

            if timing_enabled() {
                let total =
                    shadow_dur + ground_dur + gvf_dur + delay_dur + radiation_dur + tmrt_dur;
                let total_ms = total.as_secs_f64() * 1000.0;
                let shadow_ms = shadow_dur.as_secs_f64() * 1000.0;
                let ground_ms = ground_dur.as_secs_f64() * 1000.0;
                let gvf_ms = gvf_dur.as_secs_f64() * 1000.0;
                let delay_ms = delay_dur.as_secs_f64() * 1000.0;
                let rad_ms = radiation_dur.as_secs_f64() * 1000.0;
                let tmrt_ms = tmrt_dur.as_secs_f64() * 1000.0;
                // GPU duty cycle: shadow always uses GPU (when available);
                // radiation includes GPU aniso dispatch when anisotropic is active.
                let gpu_ms = shadow_ms + if use_aniso { rad_ms } else { 0.0 };
                let duty = if total_ms > 0.0 {
                    gpu_ms / total_ms * 100.0
                } else {
                    0.0
                };
                eprintln!(
                    "[TIMING] shadow={:.1}ms ground={:.1}ms gvf={:.1}ms delay={:.1}ms \
             radiation={:.1}ms tmrt={:.1}ms | total={:.1}ms gpu_duty={:.0}%",
                    shadow_ms, ground_ms, gvf_ms, delay_ms, rad_ms, tmrt_ms, total_ms, duty,
                );
            }

            TimestepResultRaw {
                tmrt,
                shadow: if want_shadow { Some(shadow_f32) } else { None },
                kdown: if want_kdown { Some(kdown) } else { None },
                kup: if want_kup { Some(kup) } else { None },
                ldown: if want_ldown { Some(ldown) } else { None },
                lup: if want_lup { Some(delay.lup) } else { None },
                timeadd: delay.timeadd,
                tgmap1: delay.tgmap1,
                tgmap1_e: delay.tgmap1_e,
                tgmap1_s: delay.tgmap1_s,
                tgmap1_w: delay.tgmap1_w,
                tgmap1_n: delay.tgmap1_n,
                tgout1: delay.tgout1,
                // Ground scheme inactive on the classic path.
                tg: None,
                rn: None,
                rn_past: None,
                g: None,
            }
        })
    }; // end allow_threads_unchecked

    // ── Convert final outputs to PyArrays (needs GIL) ────────────────────
    Ok(TimestepResult {
        tmrt: raw.tmrt.into_pyarray(py).unbind(),
        shadow: if want_shadow {
            Some(
                raw.shadow
                    .expect("shadow missing despite output mask")
                    .into_pyarray(py)
                    .unbind(),
            )
        } else {
            None
        },
        kdown: if want_kdown {
            Some(
                raw.kdown
                    .expect("kdown missing despite output mask")
                    .into_pyarray(py)
                    .unbind(),
            )
        } else {
            None
        },
        kup: if want_kup {
            Some(
                raw.kup
                    .expect("kup missing despite output mask")
                    .into_pyarray(py)
                    .unbind(),
            )
        } else {
            None
        },
        ldown: if want_ldown {
            Some(
                raw.ldown
                    .expect("ldown missing despite output mask")
                    .into_pyarray(py)
                    .unbind(),
            )
        } else {
            None
        },
        lup: if want_lup {
            Some(
                raw.lup
                    .expect("lup missing despite output mask")
                    .into_pyarray(py)
                    .unbind(),
            )
        } else {
            None
        },
        timeadd: raw.timeadd,
        tgmap1: raw.tgmap1.into_pyarray(py).unbind(),
        tgmap1_e: raw.tgmap1_e.into_pyarray(py).unbind(),
        tgmap1_s: raw.tgmap1_s.into_pyarray(py).unbind(),
        tgmap1_w: raw.tgmap1_w.into_pyarray(py).unbind(),
        tgmap1_n: raw.tgmap1_n.into_pyarray(py).unbind(),
        tgout1: raw.tgout1.into_pyarray(py).unbind(),
        tg: raw.tg.map(|a| a.into_pyarray(py).unbind()),
        rn: raw.rn.map(|a| a.into_pyarray(py).unbind()),
        rn_past: raw.rn_past.map(|a| a.into_pyarray(py).unbind()),
        g: raw.g.map(|a| a.into_pyarray(py).unbind()),
    })
}
