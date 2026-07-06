use ndarray::{Array1, Array2, ArrayView2, Zip};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

const PI: f32 = std::f32::consts::PI;

/// Scalar parameters for GVF calculation.
///
/// Groups all scalar (non-array) parameters to reduce function signature complexity.
#[pyclass]
#[derive(Clone)]
pub struct GvfScalarParams {
    /// Pixel scale (meters per pixel)
    #[pyo3(get, set)]
    pub scale: f32,
    /// First threshold for wall/building ratio
    #[pyo3(get, set)]
    pub first: f32,
    /// Second threshold for wall/building ratio
    #[pyo3(get, set)]
    pub second: f32,
    /// Wall temperature deviation from air temperature (K)
    #[pyo3(get, set)]
    pub tgwall: f32,
    /// Air temperature (°C)
    #[pyo3(get, set)]
    pub ta: f32,
    /// Wall emissivity
    #[pyo3(get, set)]
    pub ewall: f32,
    /// Stefan-Boltzmann constant (W/m²/K⁴)
    #[pyo3(get, set)]
    pub sbc: f32,
    /// Building albedo
    #[pyo3(get, set)]
    pub albedo_b: f32,
    /// Water temperature (°C)
    #[pyo3(get, set)]
    pub twater: f32,
    /// Whether land cover data is available
    #[pyo3(get, set)]
    pub landcover: bool,
}

#[pymethods]
impl GvfScalarParams {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        scale: f32,
        first: f32,
        second: f32,
        tgwall: f32,
        ta: f32,
        ewall: f32,
        sbc: f32,
        albedo_b: f32,
        twater: f32,
        landcover: bool,
    ) -> Self {
        Self {
            scale,
            first,
            second,
            tgwall,
            ta,
            ewall,
            sbc,
            albedo_b,
            twater,
            landcover,
        }
    }
}

#[pyclass]
pub struct GvfResult {
    #[pyo3(get)]
    pub gvf_lup: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalb: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalbnosh: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvf_lup_e: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalb_e: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalbnosh_e: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvf_lup_s: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalb_s: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalbnosh_s: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvf_lup_w: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalb_w: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalbnosh_w: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvf_lup_n: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalb_n: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalbnosh_n: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvf_sum: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvf_norm: Py<PyArray2<f32>>,
}

/// Pure result type for GVF calculation (no PyO3 dependency).
pub(crate) struct GvfResultPure {
    pub gvf_lup: Array2<f32>,
    pub gvfalb: Array2<f32>,
    /// Purely geometric albedo term. `None` when read from `GvfGeometryCache`.
    pub gvfalbnosh: Option<Array2<f32>>,
    pub gvf_lup_e: Array2<f32>,
    pub gvfalb_e: Array2<f32>,
    /// Purely geometric albedo term. `None` when read from `GvfGeometryCache`.
    pub gvfalbnosh_e: Option<Array2<f32>>,
    pub gvf_lup_s: Array2<f32>,
    pub gvfalb_s: Array2<f32>,
    /// Purely geometric albedo term. `None` when read from `GvfGeometryCache`.
    pub gvfalbnosh_s: Option<Array2<f32>>,
    pub gvf_lup_w: Array2<f32>,
    pub gvfalb_w: Array2<f32>,
    /// Purely geometric albedo term. `None` when read from `GvfGeometryCache`.
    pub gvfalbnosh_w: Option<Array2<f32>>,
    pub gvf_lup_n: Array2<f32>,
    pub gvfalb_n: Array2<f32>,
    /// Purely geometric albedo term. `None` when read from `GvfGeometryCache`.
    pub gvfalbnosh_n: Option<Array2<f32>>,
    /// Purely geometric normalization term. `None` when read from `GvfGeometryCache`.
    pub gvf_sum: Option<Array2<f32>>,
    /// Purely geometric normalization term. `None` when read from `GvfGeometryCache`.
    pub gvf_norm: Option<Array2<f32>>,
}

/// Pure-ndarray implementation of GVF calculation.
/// Callable from pipeline.rs (fused path) or from the PyO3 wrapper (modular path).
const SUNWALL_TOL: f32 = 1e-6;

/// UMEP sunwall mask: a wall pixel counts as sunlit only when fully sunlit,
/// i.e. `(wallsun / walls) * buildings == 1` (gvf_2018a, "new as from 2015a").
/// All GVF paths (full, cached, GPU-cached) must use this same construction —
/// a looser `wallsun > 0` mask counts partially sunlit walls and diverges from
/// the full path for the same timestep.
pub(crate) fn compute_sunwall_mask(
    wallsun: ArrayView2<f32>,
    walls: ArrayView2<f32>,
    buildings: ArrayView2<f32>,
) -> Array2<f32> {
    let mut mask = Array2::<f32>::zeros(wallsun.raw_dim());
    Zip::from(&mut mask)
        .and(&wallsun)
        .and(&walls)
        .and(&buildings)
        .par_for_each(|m, &wsun, &wall, &bldg| {
            if wall > 0.0 && bldg > 0.0 {
                let ratio = (wsun / wall) * bldg;
                if (ratio - 1.0).abs() < SUNWALL_TOL {
                    *m = 1.0;
                }
            }
        });
    mask
}

#[allow(clippy::too_many_arguments)]
#[allow(non_snake_case)]
pub(crate) fn gvf_calc_pure(
    wallsun: ArrayView2<f32>,
    walls: ArrayView2<f32>,
    buildings: ArrayView2<f32>,
    scale: f32,
    shadow: ArrayView2<f32>,
    first: f32,
    second: f32,
    dirwalls: ArrayView2<f32>,
    tg: ArrayView2<f32>,
    tgwall: f32,
    ta: f32,
    emis_grid: ArrayView2<f32>,
    ewall: f32,
    alb_grid: ArrayView2<f32>,
    sbc: f32,
    albedo_b: f32,
    twater: f32,
    lc_grid: Option<ArrayView2<f32>>,
    landcover: bool,
) -> GvfResultPure {
    let (rows, cols) = (buildings.shape()[0], buildings.shape()[1]);

    let azimuth_a: Array1<f32> = Array1::range(5.0, 359.0, 20.0);
    let num_azimuths = azimuth_a.len() as f32;
    let num_azimuths_half = num_azimuths / 2.0;

    let sunwall_mask = compute_sunwall_mask(wallsun, walls, buildings);
    let dirwalls_rad = dirwalls.mapv(|d| d * PI / 180.0);

    struct Accum {
        lup: Array2<f32>,
        alb: Array2<f32>,
        albnosh: Array2<f32>,
        sum: Array2<f32>,
        lup_e: Array2<f32>,
        alb_e: Array2<f32>,
        albnosh_e: Array2<f32>,
        lup_s: Array2<f32>,
        alb_s: Array2<f32>,
        albnosh_s: Array2<f32>,
        lup_w: Array2<f32>,
        alb_w: Array2<f32>,
        albnosh_w: Array2<f32>,
        lup_n: Array2<f32>,
        alb_n: Array2<f32>,
        albnosh_n: Array2<f32>,
    }
    let init_accum = || Accum {
        lup: Array2::zeros((rows, cols)),
        alb: Array2::zeros((rows, cols)),
        albnosh: Array2::zeros((rows, cols)),
        sum: Array2::zeros((rows, cols)),
        lup_e: Array2::zeros((rows, cols)),
        alb_e: Array2::zeros((rows, cols)),
        albnosh_e: Array2::zeros((rows, cols)),
        lup_s: Array2::zeros((rows, cols)),
        alb_s: Array2::zeros((rows, cols)),
        albnosh_s: Array2::zeros((rows, cols)),
        lup_w: Array2::zeros((rows, cols)),
        alb_w: Array2::zeros((rows, cols)),
        albnosh_w: Array2::zeros((rows, cols)),
        lup_n: Array2::zeros((rows, cols)),
        alb_n: Array2::zeros((rows, cols)),
        albnosh_n: Array2::zeros((rows, cols)),
    };

    let accum = azimuth_a
        .par_iter()
        .fold(init_accum, |mut a, &azimuth| {
            let (_gvf, gvf_lup_i, gvfalb_i, gvfalbnosh_i, gvf2_i) = crate::sun::sun_on_surface(
                azimuth,
                scale,
                buildings,
                shadow,
                sunwall_mask.view(),
                first,
                second,
                dirwalls_rad.view(),
                walls,
                tg,
                tgwall,
                ta,
                emis_grid,
                ewall,
                alb_grid,
                sbc,
                albedo_b,
                twater,
                lc_grid,
                landcover,
            );
            a.lup.zip_mut_with(&gvf_lup_i, |x, &y| *x += y);
            a.alb.zip_mut_with(&gvfalb_i, |x, &y| *x += y);
            a.albnosh.zip_mut_with(&gvfalbnosh_i, |x, &y| *x += y);
            a.sum.zip_mut_with(&gvf2_i, |x, &y| *x += y);
            if (0.0..180.0).contains(&azimuth) {
                a.lup_e.zip_mut_with(&gvf_lup_i, |x, &y| *x += y);
                a.alb_e.zip_mut_with(&gvfalb_i, |x, &y| *x += y);
                a.albnosh_e.zip_mut_with(&gvfalbnosh_i, |x, &y| *x += y);
            }
            if (90.0..270.0).contains(&azimuth) {
                a.lup_s.zip_mut_with(&gvf_lup_i, |x, &y| *x += y);
                a.alb_s.zip_mut_with(&gvfalb_i, |x, &y| *x += y);
                a.albnosh_s.zip_mut_with(&gvfalbnosh_i, |x, &y| *x += y);
            }
            if (180.0..360.0).contains(&azimuth) {
                a.lup_w.zip_mut_with(&gvf_lup_i, |x, &y| *x += y);
                a.alb_w.zip_mut_with(&gvfalb_i, |x, &y| *x += y);
                a.albnosh_w.zip_mut_with(&gvfalbnosh_i, |x, &y| *x += y);
            }
            if !(90.0..270.0).contains(&azimuth) {
                a.lup_n.zip_mut_with(&gvf_lup_i, |x, &y| *x += y);
                a.alb_n.zip_mut_with(&gvfalb_i, |x, &y| *x += y);
                a.albnosh_n.zip_mut_with(&gvfalbnosh_i, |x, &y| *x += y);
            }
            a
        })
        .reduce(init_accum, |mut a, b| {
            a.lup.zip_mut_with(&b.lup, |x, &y| *x += y);
            a.alb.zip_mut_with(&b.alb, |x, &y| *x += y);
            a.albnosh.zip_mut_with(&b.albnosh, |x, &y| *x += y);
            a.sum.zip_mut_with(&b.sum, |x, &y| *x += y);
            a.lup_e.zip_mut_with(&b.lup_e, |x, &y| *x += y);
            a.alb_e.zip_mut_with(&b.alb_e, |x, &y| *x += y);
            a.albnosh_e.zip_mut_with(&b.albnosh_e, |x, &y| *x += y);
            a.lup_s.zip_mut_with(&b.lup_s, |x, &y| *x += y);
            a.alb_s.zip_mut_with(&b.alb_s, |x, &y| *x += y);
            a.albnosh_s.zip_mut_with(&b.albnosh_s, |x, &y| *x += y);
            a.lup_w.zip_mut_with(&b.lup_w, |x, &y| *x += y);
            a.alb_w.zip_mut_with(&b.alb_w, |x, &y| *x += y);
            a.albnosh_w.zip_mut_with(&b.albnosh_w, |x, &y| *x += y);
            a.lup_n.zip_mut_with(&b.lup_n, |x, &y| *x += y);
            a.alb_n.zip_mut_with(&b.alb_n, |x, &y| *x += y);
            a.albnosh_n.zip_mut_with(&b.albnosh_n, |x, &y| *x += y);
            a
        });

    // Extract totals
    let ta_kelvin_pow4 = (ta + 273.15).powi(4);
    let emis_add = emis_grid.mapv(|e| e * sbc * ta_kelvin_pow4);
    let scale_all = 1.0 / num_azimuths;
    let scale_half = 1.0 / num_azimuths_half;
    let gvf_lup = accum.lup.mapv(|v| v * scale_all) + &emis_add;
    let gvfalb = accum.alb.mapv(|v| v * scale_all);
    let gvfalbnosh = accum.albnosh.mapv(|v| v * scale_all);
    let gvf_lup_e = accum.lup_e.mapv(|v| v * scale_half) + &emis_add;
    let gvfalb_e = accum.alb_e.mapv(|v| v * scale_half);
    let gvfalbnosh_e = accum.albnosh_e.mapv(|v| v * scale_half);
    let gvf_lup_s = accum.lup_s.mapv(|v| v * scale_half) + &emis_add;
    let gvfalb_s = accum.alb_s.mapv(|v| v * scale_half);
    let gvfalbnosh_s = accum.albnosh_s.mapv(|v| v * scale_half);
    let gvf_lup_w = accum.lup_w.mapv(|v| v * scale_half) + &emis_add;
    let gvfalb_w = accum.alb_w.mapv(|v| v * scale_half);
    let gvfalbnosh_w = accum.albnosh_w.mapv(|v| v * scale_half);
    let gvf_lup_n = accum.lup_n.mapv(|v| v * scale_half) + &emis_add;
    let gvfalb_n = accum.alb_n.mapv(|v| v * scale_half);
    let gvfalbnosh_n = accum.albnosh_n.mapv(|v| v * scale_half);
    let gvf_sum = accum.sum;
    let mut gvf_norm = gvf_sum.mapv(|v| v * scale_all);
    Zip::from(&mut gvf_norm)
        .and(&buildings)
        .for_each(|norm, &b| {
            if b == 0.0 {
                *norm = 1.0;
            }
        });

    GvfResultPure {
        gvf_lup,
        gvfalb,
        gvfalbnosh: Some(gvfalbnosh),
        gvf_lup_e,
        gvfalb_e,
        gvfalbnosh_e: Some(gvfalbnosh_e),
        gvf_lup_s,
        gvfalb_s,
        gvfalbnosh_s: Some(gvfalbnosh_s),
        gvf_lup_w,
        gvfalb_w,
        gvfalbnosh_w: Some(gvfalbnosh_w),
        gvf_lup_n,
        gvfalb_n,
        gvfalbnosh_n: Some(gvfalbnosh_n),
        gvf_sum: Some(gvf_sum),
        gvf_norm: Some(gvf_norm),
    }
}

/// GVF calculation using precomputed geometry cache (thermal-only pass).
///
/// Skips all building ray-tracing. Uses cached blocking distances and geometric outputs.
/// Returns identical results to `gvf_calc_pure` but faster on subsequent timesteps.
#[allow(clippy::too_many_arguments)]
#[allow(non_snake_case)]
pub(crate) fn gvf_calc_with_cache(
    cache: &crate::gvf_geometry::GvfGeometryCache,
    wallsun: ArrayView2<f32>,
    walls: ArrayView2<f32>,
    buildings: ArrayView2<f32>,
    shadow: ArrayView2<f32>,
    tg: ArrayView2<f32>,
    tgwall: f32,
    ta: f32,
    emis_grid: ArrayView2<f32>,
    ewall: f32,
    alb_grid: ArrayView2<f32>,
    sbc: f32,
    albedo_b: f32,
    twater: f32,
    lc_grid: Option<ArrayView2<f32>>,
    landcover: bool,
) -> GvfResultPure {
    let (rows, cols) = (buildings.shape()[0], buildings.shape()[1]);

    let num_azimuths = cache.azimuths.len() as f32;
    let num_azimuths_half = num_azimuths / 2.0;
    let ta_k = ta + 273.15;
    let ta_k_pow4 = ta_k.powi(4);

    let sunwall_mask = compute_sunwall_mask(wallsun, walls, buildings);

    let lup = Zip::from(emis_grid)
        .and(tg)
        .and(shadow)
        .map_collect(|&emis, &tg, &sh| {
            sbc * emis * (tg * sh + ta_k).powi(4) - sbc * emis * ta_k_pow4
        });
    let albshadow = &alb_grid * &shadow;

    let mut tg_for_lup_final = tg.to_owned();
    if landcover {
        if let Some(lc_grid) = lc_grid {
            Zip::from(&mut tg_for_lup_final)
                .and(lc_grid)
                .for_each(|tg, &lc| {
                    if lc == 3. {
                        *tg = twater - ta;
                    }
                });
        }
    }
    let lup_final = Zip::from(emis_grid)
        .and(&tg_for_lup_final)
        .and(shadow)
        .map_collect(|&emis, &tg, &sh| {
            sbc * emis * (tg * sh + ta_k).powi(4) - sbc * emis * ta_k_pow4
        });
    let buildings_inv = buildings.mapv(|x| 1.0 - x);
    let lup_base = &lup_final * &buildings_inv;
    let alb_base = &alb_grid * &buildings_inv * &shadow;
    let lwall = sbc * ewall * (tgwall + ta_k).powi(4) - sbc * ewall * ta_k_pow4;

    let first = cache.first;
    let second = cache.second;

    struct ThermalAccum {
        lup: Array2<f32>,
        alb: Array2<f32>,
        lup_e: Array2<f32>,
        alb_e: Array2<f32>,
        lup_s: Array2<f32>,
        alb_s: Array2<f32>,
        lup_w: Array2<f32>,
        alb_w: Array2<f32>,
        lup_n: Array2<f32>,
        alb_n: Array2<f32>,
    }
    let init_accum = || ThermalAccum {
        lup: Array2::zeros((rows, cols)),
        alb: Array2::zeros((rows, cols)),
        lup_e: Array2::zeros((rows, cols)),
        alb_e: Array2::zeros((rows, cols)),
        lup_s: Array2::zeros((rows, cols)),
        alb_s: Array2::zeros((rows, cols)),
        lup_w: Array2::zeros((rows, cols)),
        alb_w: Array2::zeros((rows, cols)),
        lup_n: Array2::zeros((rows, cols)),
        alb_n: Array2::zeros((rows, cols)),
    };

    let accum = cache
        .azimuths
        .par_iter()
        .fold(init_accum, |mut a, geom| {
            let (gvf_lup_i, gvfalb_i) = crate::sun::sun_on_surface_cached(
                geom,
                sunwall_mask.view(),
                lup.view(),
                albshadow.view(),
                lwall,
                albedo_b,
                first,
                second,
            );

            a.lup.zip_mut_with(&gvf_lup_i, |x, &y| *x += y);
            a.alb.zip_mut_with(&gvfalb_i, |x, &y| *x += y);
            let azimuth = geom.azimuth_deg;
            if (0.0..180.0).contains(&azimuth) {
                a.lup_e.zip_mut_with(&gvf_lup_i, |x, &y| *x += y);
                a.alb_e.zip_mut_with(&gvfalb_i, |x, &y| *x += y);
            }
            if (90.0..270.0).contains(&azimuth) {
                a.lup_s.zip_mut_with(&gvf_lup_i, |x, &y| *x += y);
                a.alb_s.zip_mut_with(&gvfalb_i, |x, &y| *x += y);
            }
            if (180.0..360.0).contains(&azimuth) {
                a.lup_w.zip_mut_with(&gvf_lup_i, |x, &y| *x += y);
                a.alb_w.zip_mut_with(&gvfalb_i, |x, &y| *x += y);
            }
            if !(90.0..270.0).contains(&azimuth) {
                a.lup_n.zip_mut_with(&gvf_lup_i, |x, &y| *x += y);
                a.alb_n.zip_mut_with(&gvfalb_i, |x, &y| *x += y);
            }
            a
        })
        .reduce(init_accum, |mut a, b| {
            a.lup.zip_mut_with(&b.lup, |x, &y| *x += y);
            a.alb.zip_mut_with(&b.alb, |x, &y| *x += y);
            a.lup_e.zip_mut_with(&b.lup_e, |x, &y| *x += y);
            a.alb_e.zip_mut_with(&b.alb_e, |x, &y| *x += y);
            a.lup_s.zip_mut_with(&b.lup_s, |x, &y| *x += y);
            a.alb_s.zip_mut_with(&b.alb_s, |x, &y| *x += y);
            a.lup_w.zip_mut_with(&b.lup_w, |x, &y| *x += y);
            a.alb_w.zip_mut_with(&b.alb_w, |x, &y| *x += y);
            a.lup_n.zip_mut_with(&b.lup_n, |x, &y| *x += y);
            a.alb_n.zip_mut_with(&b.alb_n, |x, &y| *x += y);
            a
        });

    let scale_all = 1.0 / num_azimuths;
    let scale_half = 1.0 / num_azimuths_half;

    // Ambient baseline: SBC × emis × Ta_K^4  (UMEP gvf_2018a.py line 64)
    // The per-azimuth accumulation uses *differential* Lup (excess above ambient).
    // After averaging, the ambient baseline must be added back to produce total Lup.
    let ambient_lup = emis_grid.mapv(|emis| sbc * emis * ta_k_pow4);

    let gvf_lup = accum.lup.mapv(|v| v * scale_all) + &lup_base + &ambient_lup;
    let gvfalb = accum.alb.mapv(|v| v * scale_all) + &alb_base;
    let gvf_lup_e = accum.lup_e.mapv(|v| v * scale_half) + &lup_base + &ambient_lup;
    let gvfalb_e = accum.alb_e.mapv(|v| v * scale_half) + &alb_base;
    let gvf_lup_s = accum.lup_s.mapv(|v| v * scale_half) + &lup_base + &ambient_lup;
    let gvfalb_s = accum.alb_s.mapv(|v| v * scale_half) + &alb_base;
    let gvf_lup_w = accum.lup_w.mapv(|v| v * scale_half) + &lup_base + &ambient_lup;
    let gvfalb_w = accum.alb_w.mapv(|v| v * scale_half) + &alb_base;
    let gvf_lup_n = accum.lup_n.mapv(|v| v * scale_half) + &lup_base + &ambient_lup;
    let gvfalb_n = accum.alb_n.mapv(|v| v * scale_half) + &alb_base;

    // `gvfalbnosh*` and GVF normalization terms are purely geometric and already
    // cached in `GvfGeometryCache`. Return `None` for those fields so accidental
    // access is explicit instead of silently carrying invalid sentinel arrays.
    GvfResultPure {
        gvf_lup,
        gvfalb,
        gvfalbnosh: None,
        gvf_lup_e,
        gvfalb_e,
        gvfalbnosh_e: None,
        gvf_lup_s,
        gvfalb_s,
        gvfalbnosh_s: None,
        gvf_lup_w,
        gvfalb_w,
        gvfalbnosh_w: None,
        gvf_lup_n,
        gvfalb_n,
        gvfalbnosh_n: None,
        gvf_sum: None,
        gvf_norm: None,
    }
}

/// GPU-accelerated cached GVF thermal accumulation.
///
/// Prepares the same input arrays as `gvf_calc_with_cache`, dispatches to GPU,
/// and applies scaling/baselines to the raw GPU output.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
#[allow(non_snake_case)]
pub(crate) fn gvf_calc_with_cache_gpu(
    gpu_ctx: &crate::gpu::GvfGpuContext,
    cache: &crate::gvf_geometry::GvfGeometryCache,
    wallsun: ArrayView2<f32>,
    walls: ArrayView2<f32>,
    buildings: ArrayView2<f32>,
    shadow: ArrayView2<f32>,
    tg: ArrayView2<f32>,
    tgwall: f32,
    ta: f32,
    emis_grid: ArrayView2<f32>,
    ewall: f32,
    alb_grid: ArrayView2<f32>,
    sbc: f32,
    albedo_b: f32,
    twater: f32,
    lc_grid: Option<ArrayView2<f32>>,
    landcover: bool,
) -> Result<GvfResultPure, String> {
    let num_azimuths = cache.azimuths.len() as f32;
    let num_azimuths_half = num_azimuths / 2.0;
    let ta_k = ta + 273.15;
    let ta_k_pow4 = ta_k.powi(4);

    // Prepare inputs (same as CPU path)
    let sunwall_mask = compute_sunwall_mask(wallsun, walls, buildings);

    let lup = Zip::from(emis_grid)
        .and(tg)
        .and(shadow)
        .map_collect(|&emis, &tg, &sh| {
            sbc * emis * (tg * sh + ta_k).powi(4) - sbc * emis * ta_k_pow4
        });
    let albshadow = &alb_grid * &shadow;

    let mut tg_for_lup_final = tg.to_owned();
    if landcover {
        if let Some(lc_grid) = lc_grid {
            Zip::from(&mut tg_for_lup_final)
                .and(lc_grid)
                .for_each(|tg, &lc| {
                    if lc == 3. {
                        *tg = twater - ta;
                    }
                });
        }
    }
    let lup_final = Zip::from(emis_grid)
        .and(&tg_for_lup_final)
        .and(shadow)
        .map_collect(|&emis, &tg, &sh| {
            sbc * emis * (tg * sh + ta_k).powi(4) - sbc * emis * ta_k_pow4
        });
    let buildings_inv = buildings.mapv(|x| 1.0 - x);
    let lup_base = &lup_final * &buildings_inv;
    let alb_base = &alb_grid * &buildings_inv * &shadow;
    let lwall = sbc * ewall * (tgwall + ta_k).powi(4) - sbc * ewall * ta_k_pow4;

    let first = cache.first;
    let second = cache.second;

    // GPU dispatch
    let pending = gpu_ctx.dispatch_begin(
        lup.view(),
        albshadow.view(),
        sunwall_mask.view(),
        first,
        second,
        lwall,
        albedo_b,
    )?;
    let gpu_result = gpu_ctx.dispatch_end(pending)?;

    // Apply scaling and baselines (same as CPU post-loop)
    let scale_all = 1.0 / num_azimuths;
    let scale_half = 1.0 / num_azimuths_half;
    let ambient_lup = emis_grid.mapv(|emis| sbc * emis * ta_k_pow4);

    let gvf_lup = gpu_result.lup.mapv(|v| v * scale_all) + &lup_base + &ambient_lup;
    let gvfalb = gpu_result.alb.mapv(|v| v * scale_all) + &alb_base;
    let gvf_lup_e = gpu_result.lup_e.mapv(|v| v * scale_half) + &lup_base + &ambient_lup;
    let gvfalb_e = gpu_result.alb_e.mapv(|v| v * scale_half) + &alb_base;
    let gvf_lup_s = gpu_result.lup_s.mapv(|v| v * scale_half) + &lup_base + &ambient_lup;
    let gvfalb_s = gpu_result.alb_s.mapv(|v| v * scale_half) + &alb_base;
    let gvf_lup_w = gpu_result.lup_w.mapv(|v| v * scale_half) + &lup_base + &ambient_lup;
    let gvfalb_w = gpu_result.alb_w.mapv(|v| v * scale_half) + &alb_base;
    let gvf_lup_n = gpu_result.lup_n.mapv(|v| v * scale_half) + &lup_base + &ambient_lup;
    let gvfalb_n = gpu_result.alb_n.mapv(|v| v * scale_half) + &alb_base;

    Ok(GvfResultPure {
        gvf_lup,
        gvfalb,
        gvfalbnosh: None,
        gvf_lup_e,
        gvfalb_e,
        gvfalbnosh_e: None,
        gvf_lup_s,
        gvfalb_s,
        gvfalbnosh_s: None,
        gvf_lup_w,
        gvfalb_w,
        gvfalbnosh_w: None,
        gvf_lup_n,
        gvfalb_n,
        gvfalbnosh_n: None,
        gvf_sum: None,
        gvf_norm: None,
    })
}

/// Compute Ground View Factor (GVF) for upwelling longwave and albedo components.
///
/// GVF represents how much a person "sees" the ground and walls from a given height.
/// This determines thermal radiation received from surrounding surfaces.
///
/// Parameters:
/// - wallsun: Wall sun exposure grid
/// - walls: Wall height grid
/// - buildings: Building mask (0=building, 1=ground)
/// - shadow: Combined shadow fraction
/// - dirwalls: Wall direction/aspect in degrees
/// - tg: Ground temperature deviation from air temperature (K)
/// - emis_grid: Emissivity per pixel
/// - alb_grid: Albedo per pixel
/// - lc_grid: Optional land cover grid
/// - params: Scalar parameters (scale, thresholds, temperatures, etc.)
///
/// Returns GvfResult with upwelling longwave and albedo view factors for all directions.
#[pyfunction]
#[allow(non_snake_case)]
pub fn gvf_calc(
    py: Python,
    wallsun: PyReadonlyArray2<f32>,
    walls: PyReadonlyArray2<f32>,
    buildings: PyReadonlyArray2<f32>,
    shadow: PyReadonlyArray2<f32>,
    dirwalls: PyReadonlyArray2<f32>,
    tg: PyReadonlyArray2<f32>,
    emis_grid: PyReadonlyArray2<f32>,
    alb_grid: PyReadonlyArray2<f32>,
    lc_grid: Option<PyReadonlyArray2<f32>>,
    params: &GvfScalarParams,
) -> PyResult<Py<GvfResult>> {
    let lc_grid_arr = lc_grid.as_ref().map(|arr| arr.as_array());
    let result = gvf_calc_pure(
        wallsun.as_array(),
        walls.as_array(),
        buildings.as_array(),
        params.scale,
        shadow.as_array(),
        params.first,
        params.second,
        dirwalls.as_array(),
        tg.as_array(),
        params.tgwall,
        params.ta,
        emis_grid.as_array(),
        params.ewall,
        alb_grid.as_array(),
        params.sbc,
        params.albedo_b,
        params.twater,
        lc_grid_arr,
        params.landcover,
    );

    Py::new(
        py,
        GvfResult {
            gvf_lup: result.gvf_lup.into_pyarray(py).unbind(),
            gvfalb: result.gvfalb.into_pyarray(py).unbind(),
            gvfalbnosh: result
                .gvfalbnosh
                .expect("gvfalbnosh is required for gvf_calc()")
                .into_pyarray(py)
                .unbind(),
            gvf_lup_e: result.gvf_lup_e.into_pyarray(py).unbind(),
            gvfalb_e: result.gvfalb_e.into_pyarray(py).unbind(),
            gvfalbnosh_e: result
                .gvfalbnosh_e
                .expect("gvfalbnosh_e is required for gvf_calc()")
                .into_pyarray(py)
                .unbind(),
            gvf_lup_s: result.gvf_lup_s.into_pyarray(py).unbind(),
            gvfalb_s: result.gvfalb_s.into_pyarray(py).unbind(),
            gvfalbnosh_s: result
                .gvfalbnosh_s
                .expect("gvfalbnosh_s is required for gvf_calc()")
                .into_pyarray(py)
                .unbind(),
            gvf_lup_w: result.gvf_lup_w.into_pyarray(py).unbind(),
            gvfalb_w: result.gvfalb_w.into_pyarray(py).unbind(),
            gvfalbnosh_w: result
                .gvfalbnosh_w
                .expect("gvfalbnosh_w is required for gvf_calc()")
                .into_pyarray(py)
                .unbind(),
            gvf_lup_n: result.gvf_lup_n.into_pyarray(py).unbind(),
            gvfalb_n: result.gvfalb_n.into_pyarray(py).unbind(),
            gvfalbnosh_n: result
                .gvfalbnosh_n
                .expect("gvfalbnosh_n is required for gvf_calc()")
                .into_pyarray(py)
                .unbind(),
            gvf_sum: result
                .gvf_sum
                .expect("gvf_sum is required for gvf_calc()")
                .into_pyarray(py)
                .unbind(),
            gvf_norm: result
                .gvf_norm
                .expect("gvf_norm is required for gvf_calc()")
                .into_pyarray(py)
                .unbind(),
        },
    )
}
