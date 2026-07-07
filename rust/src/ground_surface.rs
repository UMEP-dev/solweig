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

// ── Outgoing longwave: solid-angle ground/wall view march ──────────────────

/// Pure result of `outgoing_longwave_calc` (19 grids, upstream return order).
pub(crate) struct OutgoingLongwaveResult {
    pub gvf_lup: Array2<f32>,
    pub gvfalbsun: Array2<f32>,
    pub gvfalbtot: Array2<f32>,
    pub gvf_lup_e: Array2<f32>,
    pub gvfalbsun_e: Array2<f32>,
    pub gvfalbtot_e: Array2<f32>,
    pub gvf_lup_s: Array2<f32>,
    pub gvfalbsun_s: Array2<f32>,
    pub gvfalbtot_s: Array2<f32>,
    pub gvf_lup_w: Array2<f32>,
    pub gvfalbsun_w: Array2<f32>,
    pub gvfalbtot_w: Array2<f32>,
    pub gvf_lup_n: Array2<f32>,
    pub gvfalbsun_n: Array2<f32>,
    pub gvfalbtot_n: Array2<f32>,
    pub gvf_lside_w: Array2<f32>,
    pub gvf_lside_s: Array2<f32>,
    pub gvf_lside_e: Array2<f32>,
    pub gvf_lside_n: Array2<f32>,
}

/// Replicate the upstream numpy translated-raster assignment:
/// `dst[transl] = src[select]` with `int()` floors and `math.ceil` on the
/// float bounds (all bounds non-negative). `dx` shifts rows, `dy` columns.
fn translate_into(
    dst: &mut Array2<f32>,
    src: &Array2<f32>,
    dx: f64,
    dy: f64,
    rows: usize,
    cols: usize,
) {
    let (xs0, xs1, xt0, xt1) = if dx > 0.0 {
        (
            dx as usize,
            rows,
            0usize,
            (rows as f64 - dx).ceil() as usize,
        )
    } else {
        (
            0usize,
            (rows as f64 + dx).ceil() as usize,
            (-dx) as usize,
            rows,
        )
    };
    let (ys0, ys1, yt0, yt1) = if dy > 0.0 {
        (
            dy as usize,
            cols,
            0usize,
            (cols as f64 - dy).ceil() as usize,
        )
    } else {
        (
            0usize,
            (cols as f64 + dy).ceil() as usize,
            (-dy) as usize,
            cols,
        )
    };
    let src_slice = src.slice(ndarray::s![xs0..xs1, ys0..ys1]);
    dst.slice_mut(ndarray::s![xt0..xt1, yt0..yt1]).assign(&src_slice);
}

/// Per-azimuth partial sums produced by the march.
struct AzimuthPartial {
    azimuth: f64,
    lup_sum: Array2<f32>,
    albsun_sum: Array2<f32>,
    albtot_sum: Array2<f32>,
    lside_e: Array2<f32>,
    lside_s: Array2<f32>,
    lside_w: Array2<f32>,
    lside_n: Array2<f32>,
}

/// Port of `ground_surface.py::outgoingLongwave_calc` (UMEP 2026a):
/// directional upwelling/side longwave and albedo view factors from a
/// 20-azimuth translated-raster march out to ~11 m at person height
/// (zs = 1.1 m, 99% of the Lambert view factor).
///
/// Faithfulness notes:
/// - Upstream binarizes the caller's `sunwall` grid in place; this port
///   copies instead (no caller mutation) — outputs are identical.
/// - The translated scratch rasters persist between radius steps (margins
///   keep the previous step's values, with Lup/albedo scratch initialized
///   from the source grids and wall scratch from zeros), reproduced exactly.
/// - `tgwall` and `ta` are scalars, as at the 2026a calc call site.
#[allow(clippy::too_many_arguments)]
pub(crate) fn outgoing_longwave_calc_pure(
    tg: ArrayView2<f32>,
    tgwall: f32,
    ta: f32,
    ldown: ArrayView2<f32>,
    emis: ArrayView2<f32>,
    alb: ArrayView2<f32>,
    buildings: ArrayView2<f32>,
    shadow: ArrayView2<f32>,
    sunwall: ArrayView2<f32>,
    walls: ArrayView2<f32>,
    sizepx: f32,
) -> OutgoingLongwaveResult {
    use rayon::prelude::*;

    let (rows, cols) = (tg.nrows(), tg.ncols());
    let dim = tg.raw_dim();
    let sizepx_f = sizepx as f64;

    const FACTOR: f64 = 0.99; // fraction of the Lambert view factor covered
    const ZS: f64 = 1.1; // receiver height (m)
    let r_max = ZS * (FACTOR / (1.0 - FACTOR)).sqrt();
    const EMIS_WALL: f32 = 0.9;
    const ALB_WALL: f32 = 0.2;
    const STEP: f64 = 1.0;
    const N_AZI: f64 = 20.0;

    // Binarized sunlit-wall grid (upstream mutates the caller's grid here)
    let sunlitwall = sunwall.mapv(|v| if v > 0.0 { 1.0 } else { v });
    let wallbol = walls.mapv(|v| if v > 0.0 { 1.0f32 } else { 0.0 });
    let albsunlit = &alb.to_owned() * &shadow;

    // Ground upwelling longwave, masked to non-building pixels
    let mut lup = Array2::<f32>::zeros(dim.clone());
    ndarray::Zip::from(&mut lup)
        .and(emis)
        .and(tg)
        .and(ldown)
        .and(buildings)
        .for_each(|l, &e, &t, &ld, &b| {
            *l = (SBC * e * (t + 273.15).powi(4) + ld * (1.0 - e)) * b;
        });

    // Wall longwave (scalar wall temperature deviation)
    let lwall_val = |wb: f32| SBC * EMIS_WALL * (tgwall + ta + 273.15).powi(4) * wb;
    let lwall = wallbol.mapv(lwall_val);

    let azimuths: Vec<f64> = (1..=20)
        .map(|i| (18.0 * i as f64) * std::f64::consts::PI / 180.0)
        .collect();

    let alb_owned = alb.to_owned();
    let buildings_owned = buildings.to_owned();

    let partials: Vec<AzimuthPartial> = azimuths
        .par_iter()
        .map(|&azimuth| {
            let mut building_copy = buildings_owned.clone();
            let mut pastwalls = wallbol.clone();

            // Scratch rasters persist across radius steps (see note above)
            let mut building_temp = buildings_owned.clone();
            let mut lup_temp = lup.clone();
            let mut lwall_temp = lwall.clone();
            let mut albsun_temp = albsunlit.clone();
            let mut albtot_temp = alb_owned.clone();
            let mut walls_temp = Array2::<f32>::zeros(dim.clone());
            let mut sunlitwall_temp = Array2::<f32>::zeros(dim.clone());

            let mut lup_sum = Array2::<f32>::zeros(dim.clone());
            let mut albsun_sum = Array2::<f32>::zeros(dim.clone());
            let mut albtot_sum = Array2::<f32>::zeros(dim.clone());
            let mut lside_e = Array2::<f32>::zeros(dim.clone());
            let mut lside_s = Array2::<f32>::zeros(dim.clone());
            let mut lside_w = Array2::<f32>::zeros(dim.clone());
            let mut lside_n = Array2::<f32>::zeros(dim.clone());

            let mut r = sizepx_f / 2.0;
            while r < r_max {
                // Raster translation step for this radius (float bounds,
                // scaled so the grid moves at least one pixel)
                let (dx, dy) = {
                    let dx0 = -azimuth.cos();
                    let dy0 = -azimuth.sin();
                    if dx0.abs() > dy0.abs() {
                        (
                            -r * azimuth.cos().signum(),
                            -r * azimuth.tan().abs() * azimuth.sin().signum(),
                        )
                    } else {
                        (
                            -r / azimuth.tan().abs() * azimuth.cos().signum(),
                            -r * azimuth.sin().signum(),
                        )
                    }
                };

                translate_into(&mut building_temp, &buildings_owned, dx, dy, rows, cols);
                translate_into(&mut lup_temp, &lup, dx, dy, rows, cols);
                translate_into(&mut lwall_temp, &lwall, dx, dy, rows, cols);
                translate_into(&mut albsun_temp, &albsunlit, dx, dy, rows, cols);
                translate_into(&mut albtot_temp, &alb_owned, dx, dy, rows, cols);
                translate_into(&mut sunlitwall_temp, &sunlitwall, dx, dy, rows, cols);
                translate_into(&mut walls_temp, &wallbol, dx, dy, rows, cols);

                // Cumulative occlusion: once a building, always a building
                ndarray::Zip::from(&mut building_copy)
                    .and(&building_temp)
                    .for_each(|bc, &bt| *bc = bc.min(bt));

                // Annulus view factor for the ground contribution
                let view_factor = ((r + STEP).powi(2) / (ZS * ZS + (r + STEP).powi(2))
                    - r * r / (ZS * ZS + r * r)) as f32;

                // Wall-surface view factor at this translation distance
                let rp = r + STEP;
                let rh = r + sizepx_f / 2.0;
                let viewfactor_wall = ((1.0 / 2.0f64.sqrt() / 3.0)
                    * (1.0 + rp / (rp * rp + ZS * ZS).sqrt()).sqrt()
                    * (2.0 + rh / (rh * rh + ZS * ZS).sqrt())
                    * (1.0 - rh / (rh * rh + ZS * ZS).sqrt())
                    / ZS
                    * (rh * rh + ZS * ZS).sqrt()) as f32;

                // Side solid angle for this annulus
                let dphi = (rp / ZS).atan() - (r / ZS).atan();
                let dtrigo = ZS / (r * r + ZS * ZS).sqrt() * r / (r * r + ZS * ZS).sqrt()
                    - ZS / (rp * rp + ZS * ZS).sqrt() * rp / (rp * rp + ZS * ZS).sqrt();
                let dtheta = 2.0 * std::f64::consts::PI / N_AZI;
                let steradians = (dtheta * (dphi + dtrigo) / 2.0) as f32;

                let in_w = azimuth < std::f64::consts::PI;
                let in_s = azimuth >= std::f64::consts::PI / 2.0
                    && azimuth < 3.0 * std::f64::consts::PI / 2.0;
                let in_e =
                    azimuth >= std::f64::consts::PI && azimuth < 2.0 * std::f64::consts::PI;
                let in_n = azimuth >= 3.0 * std::f64::consts::PI / 2.0
                    || azimuth < std::f64::consts::PI / 2.0;

                // Single slice pass for all per-annulus accumulators (Zip's
                // producer arity caps below what this step touches).
                {
                    let wt_s = walls_temp.as_slice().expect("contiguous");
                    let pw_s = pastwalls.as_slice().expect("contiguous");
                    let sw_s = sunlitwall_temp.as_slice().expect("contiguous");
                    let bc_s = building_copy.as_slice().expect("contiguous");
                    let lw_s = lwall_temp.as_slice().expect("contiguous");
                    let lt_s = lup_temp.as_slice().expect("contiguous");
                    let at_s = albsun_temp.as_slice().expect("contiguous");
                    let att_s = albtot_temp.as_slice().expect("contiguous");
                    let ls_s = lup_sum.as_slice_mut().expect("contiguous");
                    let asn_s = albsun_sum.as_slice_mut().expect("contiguous");
                    let ast_s = albtot_sum.as_slice_mut().expect("contiguous");
                    let le_s = lside_e.as_slice_mut().expect("contiguous");
                    let lss_s = lside_s.as_slice_mut().expect("contiguous");
                    let lw2_s = lside_w.as_slice_mut().expect("contiguous");
                    let ln_s = lside_n.as_slice_mut().expect("contiguous");
                    for idx in 0..ls_s.len() {
                        let bc = bc_s[idx];
                        // Ground contribution through the annulus view factor
                        ls_s[idx] += lt_s[idx] * view_factor * bc / 20.0;
                        asn_s[idx] += at_s[idx] * view_factor * bc / 20.0;
                        ast_s[idx] += att_s[idx] * view_factor * bc / 20.0;
                        // Walls newly entering the field of view
                        let onlywall = if wt_s[idx] > 0.0 && pw_s[idx] == 0.0 { bc } else { 0.0 };
                        let onlysunwall = sw_s[idx] * bc;
                        ls_s[idx] += onlywall * lw_s[idx] * viewfactor_wall * bc / 20.0;
                        asn_s[idx] += onlysunwall * ALB_WALL * viewfactor_wall * bc / 20.0;
                        ast_s[idx] += onlywall * ALB_WALL * viewfactor_wall * bc / 20.0;
                        // Directional side longwave: ground solid angle + wall
                        let ground_side = lt_s[idx] / std::f32::consts::PI * steradians * bc;
                        let wall_side = onlywall * lw_s[idx] * viewfactor_wall / 10.0;
                        if in_w {
                            lw2_s[idx] += wall_side + ground_side;
                        }
                        if in_s {
                            lss_s[idx] += wall_side + ground_side;
                        }
                        if in_e {
                            le_s[idx] += wall_side + ground_side;
                        }
                        if in_n {
                            ln_s[idx] += wall_side + ground_side;
                        }
                    }
                }

                // pastwalls |= walls_temp (after onlywall used it)
                ndarray::Zip::from(&mut pastwalls)
                    .and(&walls_temp)
                    .for_each(|pw, &wt| {
                        if *pw == 0.0 && wt > 0.0 {
                            *pw = 1.0;
                        }
                    });

                r += STEP;
            }

            AzimuthPartial {
                azimuth,
                lup_sum,
                albsun_sum,
                albtot_sum,
                lside_e,
                lside_s,
                lside_w,
                lside_n,
            }
        })
        .collect();

    // ── Assemble outputs (deterministic sequential reduction) ──
    let mut out = OutgoingLongwaveResult {
        gvf_lup: Array2::zeros(dim.clone()),
        gvfalbsun: Array2::zeros(dim.clone()),
        gvfalbtot: Array2::zeros(dim.clone()),
        gvf_lup_e: Array2::zeros(dim.clone()),
        gvfalbsun_e: Array2::zeros(dim.clone()),
        gvfalbtot_e: Array2::zeros(dim.clone()),
        gvf_lup_s: Array2::zeros(dim.clone()),
        gvfalbsun_s: Array2::zeros(dim.clone()),
        gvfalbtot_s: Array2::zeros(dim.clone()),
        gvf_lup_w: Array2::zeros(dim.clone()),
        gvfalbsun_w: Array2::zeros(dim.clone()),
        gvfalbtot_w: Array2::zeros(dim.clone()),
        gvf_lup_n: Array2::zeros(dim.clone()),
        gvfalbsun_n: Array2::zeros(dim.clone()),
        gvfalbtot_n: Array2::zeros(dim.clone()),
        gvf_lside_w: Array2::zeros(dim.clone()),
        gvf_lside_s: Array2::zeros(dim.clone()),
        gvf_lside_e: Array2::zeros(dim.clone()),
        gvf_lside_n: Array2::zeros(dim),
    };

    // Contribution of the pixel directly below the receiver
    let vf_below = ((sizepx_f / 2.0).powi(2) / ((sizepx_f / 2.0).powi(2) + ZS * ZS)) as f32;
    ndarray::Zip::from(&mut out.gvf_lup)
        .and(&lup)
        .for_each(|gl, &l| *gl += l * vf_below);
    ndarray::Zip::from(&mut out.gvfalbsun)
        .and(&mut out.gvfalbtot)
        .and(&albsunlit)
        .and(&alb_owned)
        .and(&buildings_owned)
        .for_each(|gas, gat, &asl, &a, &b| {
            *gas += asl * vf_below * b;
            *gat += a * vf_below * b;
        });

    for p in &partials {
        let az = p.azimuth;
        out.gvf_lup += &p.lup_sum;
        out.gvfalbsun += &p.albsun_sum;
        out.gvfalbtot += &p.albtot_sum;
        if az < std::f64::consts::PI {
            out.gvf_lup_w += &p.lup_sum;
            out.gvfalbsun_w += &p.albsun_sum;
            out.gvfalbtot_w += &p.albtot_sum;
            out.gvf_lside_w += &p.lside_w;
        }
        if az >= std::f64::consts::PI / 2.0 && az < 3.0 * std::f64::consts::PI / 2.0 {
            out.gvf_lup_s += &p.lup_sum;
            out.gvfalbsun_s += &p.albsun_sum;
            out.gvfalbtot_s += &p.albtot_sum;
            out.gvf_lside_s += &p.lside_s;
        }
        if az >= std::f64::consts::PI && az < 2.0 * std::f64::consts::PI {
            out.gvf_lup_e += &p.lup_sum;
            out.gvfalbsun_e += &p.albsun_sum;
            out.gvfalbtot_e += &p.albtot_sum;
            out.gvf_lside_e += &p.lside_e;
        }
        if az >= 3.0 * std::f64::consts::PI / 2.0 || az < std::f64::consts::PI / 2.0 {
            out.gvf_lup_n += &p.lup_sum;
            out.gvfalbsun_n += &p.albsun_sum;
            out.gvfalbtot_n += &p.albtot_sum;
            out.gvf_lside_n += &p.lside_n;
        }
    }

    // Roof pixels: allocate their own emission (buildings==0 there)
    let mut roof_emit = Array2::<f32>::zeros(out.gvf_lup.raw_dim());
    ndarray::Zip::from(&mut roof_emit)
        .and(emis)
        .and(tg)
        .and(&buildings_owned)
        .for_each(|re, &e, &t, &b| {
            *re = SBC * e * (t + 273.15).powi(4) * (1.0 - b);
        });
    out.gvf_lup += &roof_emit;
    let roof_emit_half = roof_emit.mapv(|v| v * 0.5);
    out.gvf_lside_e += &roof_emit_half;
    out.gvf_lside_n += &roof_emit_half;
    out.gvf_lside_w += &roof_emit_half;
    out.gvf_lside_s += &roof_emit_half;
    ndarray::Zip::from(&mut out.gvfalbsun)
        .and(&mut out.gvfalbtot)
        .and(&albsunlit)
        .and(&alb_owned)
        .and(&buildings_owned)
        .for_each(|gas, gat, &asl, &a, &b| {
            let roof = 1.0 - b;
            *gas += asl * roof;
            *gat += a * roof;
        });
    for (dir_sun, dir_tot) in [
        (&mut out.gvfalbsun_e, &mut out.gvfalbtot_e),
        (&mut out.gvfalbsun_n, &mut out.gvfalbtot_n),
        (&mut out.gvfalbsun_w, &mut out.gvfalbtot_w),
        (&mut out.gvfalbsun_s, &mut out.gvfalbtot_s),
    ] {
        ndarray::Zip::from(dir_sun)
            .and(dir_tot)
            .and(&albsunlit)
            .and(&alb_owned)
            .and(&buildings_owned)
            .for_each(|gas, gat, &asl, &a, &b| {
                let roof = 1.0 - b;
                *gas += asl * 0.5 * roof;
                *gat += a * 0.5 * roof;
            });
    }

    out
}

/// Result container for the outgoing-longwave march (upstream return order).
#[pyclass]
pub struct OutgoingLongwaveStep {
    #[pyo3(get)]
    pub gvf_lup: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalbsun: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalbtot: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvf_lup_e: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalbsun_e: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalbtot_e: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvf_lup_s: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalbsun_s: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalbtot_s: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvf_lup_w: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalbsun_w: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalbtot_w: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvf_lup_n: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalbsun_n: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalbtot_n: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvf_lside_w: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvf_lside_s: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvf_lside_e: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvf_lside_n: Py<PyArray2<f32>>,
}

/// UMEP 2026a outgoing longwave and albedo view factors via the solid-angle
/// march. `tgwall` is the wall temperature deviation from air temperature
/// (K), `ta` the air temperature (C), `sizepx` the pixel size in metres.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn outgoing_longwave_calc(
    py: Python,
    tg: PyReadonlyArray2<f32>,
    tgwall: f32,
    ta: f32,
    ldown: PyReadonlyArray2<f32>,
    emis: PyReadonlyArray2<f32>,
    alb: PyReadonlyArray2<f32>,
    buildings: PyReadonlyArray2<f32>,
    shadow: PyReadonlyArray2<f32>,
    sunwall: PyReadonlyArray2<f32>,
    walls: PyReadonlyArray2<f32>,
    sizepx: f32,
) -> PyResult<Py<OutgoingLongwaveStep>> {
    let result = outgoing_longwave_calc_pure(
        tg.as_array(),
        tgwall,
        ta,
        ldown.as_array(),
        emis.as_array(),
        alb.as_array(),
        buildings.as_array(),
        shadow.as_array(),
        sunwall.as_array(),
        walls.as_array(),
        sizepx,
    );
    Py::new(
        py,
        OutgoingLongwaveStep {
            gvf_lup: result.gvf_lup.into_pyarray(py).unbind(),
            gvfalbsun: result.gvfalbsun.into_pyarray(py).unbind(),
            gvfalbtot: result.gvfalbtot.into_pyarray(py).unbind(),
            gvf_lup_e: result.gvf_lup_e.into_pyarray(py).unbind(),
            gvfalbsun_e: result.gvfalbsun_e.into_pyarray(py).unbind(),
            gvfalbtot_e: result.gvfalbtot_e.into_pyarray(py).unbind(),
            gvf_lup_s: result.gvf_lup_s.into_pyarray(py).unbind(),
            gvfalbsun_s: result.gvfalbsun_s.into_pyarray(py).unbind(),
            gvfalbtot_s: result.gvfalbtot_s.into_pyarray(py).unbind(),
            gvf_lup_w: result.gvf_lup_w.into_pyarray(py).unbind(),
            gvfalbsun_w: result.gvfalbsun_w.into_pyarray(py).unbind(),
            gvfalbtot_w: result.gvfalbtot_w.into_pyarray(py).unbind(),
            gvf_lup_n: result.gvf_lup_n.into_pyarray(py).unbind(),
            gvfalbsun_n: result.gvfalbsun_n.into_pyarray(py).unbind(),
            gvfalbtot_n: result.gvfalbtot_n.into_pyarray(py).unbind(),
            gvf_lside_w: result.gvf_lside_w.into_pyarray(py).unbind(),
            gvf_lside_s: result.gvf_lside_s.into_pyarray(py).unbind(),
            gvf_lside_e: result.gvf_lside_e.into_pyarray(py).unbind(),
            gvf_lside_n: result.gvf_lside_n.into_pyarray(py).unbind(),
        },
    )
}
