//! UMEP 2026a ground surface temperature scheme (Bridoux, University of
//! Gothenburg): force-restore surface temperature with the ground heat flux
//! from the Objective Hysteresis Model (Grimmond et al. 1991), integrated
//! with a 2nd-order Runge-Kutta step, plus a slab model for water bodies.
//!
//! Ported from `ground_surface.py::surfaceTemperature_calc` in
//! UMEP-dev/UMEP-processing (vendored verbatim at
//! `tests/reference/umep_2026a/ground_surface.py`); parity is gated by
//! `tests/spec/test_parity_2026a.py`.
//!
//! Faithfulness notes (deliberate reproductions of upstream semantics):
//! - `timestep` is in SECONDS (the upstream docstring says minutes, but the
//!   force-restore constant 2π/86400 s⁻¹ and the calc-level caller both use
//!   seconds).
//! - Upstream's `Tg_stored = Tg` is an alias, not a copy, and `Tg` is updated
//!   in place; the water branch therefore adds its slab increment on top of
//!   the already-RK2-updated temperature, and computes the increment from
//!   the updated temperature. Reproduced exactly; flagged for upstream
//!   review in the 2026a port notes.
//! - `np.sign` returns 0 at 0 (Rust `signum` returns ±1); ported explicitly.

use ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

const SBC: f32 = 5.67e-8;
const PI: f32 = std::f32::consts::PI;
/// Angular frequency of the daily temperature wave (rad s^-1 numerator).
const DAY_SECONDS: f32 = 86400.0;

#[inline]
fn np_sign(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// August-Roche-Magnus saturated vapour pressure (Pa), as upstream
/// `saturated_vp` (only the pressure is used by the water branch).
#[inline]
fn saturated_vp_pa(t_c: f32) -> f32 {
    6109.4 * (17.625 * t_c / (t_c + 243.04)).exp()
}

/// Pure result of one surface-temperature step.
pub(crate) struct SurfaceTemperatureResult {
    pub tg: Array2<f32>,
    pub rn: Array2<f32>,
    pub rn_past: Array2<f32>,
    pub g: Array2<f32>,
}

/// One force-restore/OHM/RK2 step of the ground surface temperature.
///
/// All grids share the DSM shape. `rn`, `rn_past`, `g`, `tg`, `tm` are the
/// carried state from the previous timestep (see `initiate_groundScheme`
/// upstream for the initial values). Returns the updated `(tg, rn, rn_past,
/// g)` state.
#[allow(clippy::too_many_arguments)]
pub(crate) fn surface_temperature_calc_pure(
    kdown: ArrayView2<f32>,
    ldown: ArrayView2<f32>,
    rn_in: ArrayView2<f32>,
    rn_past_in: ArrayView2<f32>,
    g_in: ArrayView2<f32>,
    tg_in: ArrayView2<f32>,
    tm: ArrayView2<f32>,
    alb: ArrayView2<f32>,
    emis: ArrayView2<f32>,
    cap: ArrayView2<f32>,
    diff: ArrayView2<f32>,
    lc_grid: ArrayView2<f32>,
    a1: ArrayView2<f32>,
    a2: ArrayView2<f32>,
    a3: ArrayView2<f32>,
    timestep_s: f32,
    rh_percent: f32,
    shadow: ArrayView2<f32>,
    shadow_past: ArrayView2<f32>,
) -> SurfaceTemperatureResult {
    let omega = 2.0 * PI / DAY_SECONDS;

    // ndarray's Zip arity caps out below the 17 grids this step touches, so
    // outputs are built as flat vecs in one indexed rayon pass (the same
    // pattern as vegetation::lside_veg_variant_pure).
    let (rows, cols) = (tg_in.nrows(), tg_in.ncols());
    let npix = rows * cols;
    let mut tg_vec = vec![0.0f32; npix];
    let mut rn_vec = vec![0.0f32; npix];
    let mut rn_past_vec = vec![0.0f32; npix];
    let mut g_vec = vec![0.0f32; npix];

    use rayon::prelude::*;
    tg_vec
        .par_iter_mut()
        .zip(rn_vec.par_iter_mut())
        .zip(rn_past_vec.par_iter_mut())
        .zip(g_vec.par_iter_mut())
        .enumerate()
        .for_each(|(idx, (((tg_cell, rn_cell), rn_past_cell), g_cell))| {
            let r = idx / cols;
            let c = idx % cols;

            let kdown_v = kdown[(r, c)];
            let ldown_v = ldown[(r, c)];
            let rn_v = rn_in[(r, c)];
            let rn_past_v = rn_past_in[(r, c)];
            let g_v = g_in[(r, c)];
            let tg_v = tg_in[(r, c)];
            let tm_v = tm[(r, c)];
            let alb_v = alb[(r, c)];
            let emis_v = emis[(r, c)];
            let cap_v = cap[(r, c)];
            let diff_v = diff[(r, c)];
            let a1_v = a1[(r, c)];
            let a2_v = a2[(r, c)];
            let a3_v = a3[(r, c)];
            let shadow_v = shadow[(r, c)];
            let shadow_past_v = shadow_past[(r, c)];
            let lc_v = lc_grid[(r, c)];

            // Damping depth of the daily surface temperature wave
            let d = ((2.0 * diff_v) / omega).sqrt();

            // RK2: first slope from the carried ground heat flux
            let k1 = 2.0 * g_v / cap_v / d - omega * (tg_v - tm_v);
            let tg_temp = tg_v + k1 * timestep_s;

            // Second slope from fluxes re-evaluated at the estimate
            let lup_temp = SBC * emis_v * (tg_temp + 273.15).powi(4) + ldown_v * (1.0 - emis_v);
            let rn_temp = kdown_v * (1.0 - alb_v) + ldown_v - lup_temp;
            let rn_star_temp = rn_temp - rn_v;
            let mut g_temp = a1_v * rn_temp + a2_v * rn_star_temp + a3_v;

            // Damp ground-heat-flux spikes at shadow transitions
            let delta_g = (g_temp - g_v).abs();
            let rad_criterion = (a1_v * (rn_temp - rn_past_v)).abs();
            if delta_g > rad_criterion && (shadow_v - shadow_past_v).abs() > 0.5 {
                g_temp = g_v + np_sign(g_temp - g_v) * rad_criterion;
            }

            let k2 = 2.0 * g_temp / cap_v / d - omega * (tg_temp - tm_v);
            let mut tg_new = tg_v + (k1 + k2) / 2.0 * timestep_s;

            // Updated fluxes from the accepted temperature
            let rn_past_new = rn_v;
            let g_past = g_v;
            let lup = SBC * emis_v * (tg_new + 273.15).powi(4) + (1.0 - emis_v) * ldown_v;
            let rn_new = (1.0 - alb_v) * kdown_v + ldown_v - lup;
            let rn_star = rn_new - rn_past_new;
            let mut g_new = a1_v * rn_new + a2_v * rn_star + a3_v;

            let delta_g2 = (g_new - g_past).abs();
            let rad_criterion2 = (a1_v * (rn_new - rn_past_new)).abs();
            if delta_g2 > rad_criterion2 && (shadow_v - shadow_past_v).abs() > 0.5 {
                g_new = g_past + np_sign(g_new - g_past) * rad_criterion2;
            }

            // Water bodies: slab energy balance replaces the OHM step.
            // Upstream computes the increment from (and adds it to) the
            // RK2-updated temperature via the Tg_stored alias — reproduced.
            if lc_v == 7.0 {
                const BETA: f32 = 0.45; // shortwave absorbed in the top layer
                const THICKNESS: f32 = 1.0; // slab depth (m)
                const LAMB: f32 = 2.260e6; // latent heat of vaporisation
                const RHO: f32 = 1000.0; // water density (kg m-3)
                let rn_water = kdown_v * (1.0 - alb_v) * (BETA + (1.0 - BETA) * (1.0 - (-1.0f32).exp()))
                    + ldown_v
                    - lup;
                let es = saturated_vp_pa(tg_new);
                let e = 0.0858 * (es / 1000.0) * (1.0 - rh_percent / 100.0) / 3600.0 / 1000.0
                    * RHO
                    * LAMB;
                let delta_tg = timestep_s / cap_v / THICKNESS
                    * (rn_water - e - diff_v * cap_v / THICKNESS * (tg_new - tm_v));
                tg_new += delta_tg;
            }

            *tg_cell = tg_new;
            *rn_cell = rn_new;
            *rn_past_cell = rn_past_new;
            *g_cell = g_new;
        });

    debug_assert_eq!(tg_vec.len(), npix);
    SurfaceTemperatureResult {
        tg: Array2::from_shape_vec((rows, cols), tg_vec).expect("tg length matches (rows, cols)"),
        rn: Array2::from_shape_vec((rows, cols), rn_vec).expect("rn length matches (rows, cols)"),
        rn_past: Array2::from_shape_vec((rows, cols), rn_past_vec)
            .expect("rn_past length matches (rows, cols)"),
        g: Array2::from_shape_vec((rows, cols), g_vec).expect("g length matches (rows, cols)"),
    }
}

/// Result container for the surface-temperature step.
#[pyclass]
pub struct SurfaceTemperatureStep {
    #[pyo3(get)]
    pub tg: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub rn: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub rn_past: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub g: Py<PyArray2<f32>>,
}

/// One step of the UMEP 2026a force-restore/OHM ground surface temperature
/// scheme. `timestep_s` is in seconds; `rh_percent` in 0-100.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn surface_temperature_calc(
    py: Python,
    kdown: PyReadonlyArray2<f32>,
    ldown: PyReadonlyArray2<f32>,
    rn: PyReadonlyArray2<f32>,
    rn_past: PyReadonlyArray2<f32>,
    g: PyReadonlyArray2<f32>,
    tg: PyReadonlyArray2<f32>,
    tm: PyReadonlyArray2<f32>,
    alb: PyReadonlyArray2<f32>,
    emis: PyReadonlyArray2<f32>,
    cap: PyReadonlyArray2<f32>,
    diff: PyReadonlyArray2<f32>,
    lc_grid: PyReadonlyArray2<f32>,
    a1: PyReadonlyArray2<f32>,
    a2: PyReadonlyArray2<f32>,
    a3: PyReadonlyArray2<f32>,
    timestep_s: f32,
    rh_percent: f32,
    shadow: PyReadonlyArray2<f32>,
    shadow_past: PyReadonlyArray2<f32>,
) -> PyResult<Py<SurfaceTemperatureStep>> {
    let result = surface_temperature_calc_pure(
        kdown.as_array(),
        ldown.as_array(),
        rn.as_array(),
        rn_past.as_array(),
        g.as_array(),
        tg.as_array(),
        tm.as_array(),
        alb.as_array(),
        emis.as_array(),
        cap.as_array(),
        diff.as_array(),
        lc_grid.as_array(),
        a1.as_array(),
        a2.as_array(),
        a3.as_array(),
        timestep_s,
        rh_percent,
        shadow.as_array(),
        shadow_past.as_array(),
    );
    Py::new(
        py,
        SurfaceTemperatureStep {
            tg: result.tg.into_pyarray(py).unbind(),
            rn: result.rn.into_pyarray(py).unbind(),
            rn_past: result.rn_past.into_pyarray(py).unbind(),
            g: result.g.into_pyarray(py).unbind(),
        },
    )
}
