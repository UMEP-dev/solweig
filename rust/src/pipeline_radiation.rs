//! Per-timestep radiation helpers used by `pipeline::compute_timestep`.
//!
//! These are pure-`ndarray` functions that take borrowed views and return
//! owned arrays. They were extracted from `pipeline.rs` to keep that file
//! focused on orchestration; the math itself is identical to the original
//! single-file implementation (golden tests are the gate).
//!
//! Public to `crate` only — these are not user-facing FFI.

use ndarray::{Array2, ArrayView1, ArrayView2, ArrayView3, Zip};
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

const PI: f32 = std::f32::consts::PI;
const SBC: f32 = 5.67051e-8;
const KELVIN_OFFSET: f32 = 273.15;

// ── Sky emissivity ─────────────────────────────────────────────────────────

/// Compute sky emissivity (Jonsson et al. 2006).
#[inline]
pub(crate) fn compute_esky(ta: f32, rh: f32) -> f32 {
    let ta_k = ta + KELVIN_OFFSET;
    let ea = 6.107 * 10.0_f32.powf((7.5 * ta) / (237.3 + ta)) * (rh / 100.0);
    let msteg = 46.5 * (ea / ta_k);
    1.0 - (1.0 + msteg) * (-((1.2 + 3.0 * msteg) as f32).sqrt()).exp()
}

// ── Shortwave: Kup (ground-reflected) ──────────────────────────────────────

/// Compute Kup (ground-reflected shortwave) — Kup_veg_2015a.
///
/// Returns `(kup, kup_e, kup_s, kup_w, kup_n)` as owned arrays.
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_kup(
    rad_i: f32,
    rad_d: f32,
    rad_g: f32,
    altitude: f32,
    svfbuveg: ArrayView2<f32>,
    albedo_b: f32,
    f_sh: ArrayView2<f32>,
    gvfalb: ArrayView2<f32>,
    gvfalb_e: ArrayView2<f32>,
    gvfalb_s: ArrayView2<f32>,
    gvfalb_w: ArrayView2<f32>,
    gvfalb_n: ArrayView2<f32>,
    gvfalbnosh: ArrayView2<f32>,
    gvfalbnosh_e: ArrayView2<f32>,
    gvfalbnosh_s: ArrayView2<f32>,
    gvfalbnosh_w: ArrayView2<f32>,
    gvfalbnosh_n: ArrayView2<f32>,
    valid: ArrayView2<u8>,
) -> (
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
) {
    let rad_i_sin_alt = rad_i * (altitude * PI / 180.0).sin();

    let shape = svfbuveg.dim();
    let mut kup = Array2::<f32>::zeros(shape);
    let mut kup_e = Array2::<f32>::zeros(shape);
    let mut kup_s = Array2::<f32>::zeros(shape);
    let mut kup_w = Array2::<f32>::zeros(shape);
    let mut kup_n = Array2::<f32>::zeros(shape);

    Zip::indexed(&mut kup)
        .and(&mut kup_e)
        .and(&mut kup_s)
        .and(&mut kup_w)
        .and(&mut kup_n)
        .par_for_each(|(r, c), k, ke, ks, kw, kn| {
            if valid[[r, c]] == 0 {
                *k = f32::NAN;
                *ke = f32::NAN;
                *ks = f32::NAN;
                *kw = f32::NAN;
                *kn = f32::NAN;
                return;
            }
            let sv = svfbuveg[[r, c]];
            let fsh = f_sh[[r, c]];
            let ct = rad_d * sv + albedo_b * (1.0 - sv) * (rad_g * (1.0 - fsh) + rad_d * fsh);
            *k = gvfalb[[r, c]] * rad_i_sin_alt + ct * gvfalbnosh[[r, c]];
            *ke = gvfalb_e[[r, c]] * rad_i_sin_alt + ct * gvfalbnosh_e[[r, c]];
            *ks = gvfalb_s[[r, c]] * rad_i_sin_alt + ct * gvfalbnosh_s[[r, c]];
            *kw = gvfalb_w[[r, c]] * rad_i_sin_alt + ct * gvfalbnosh_w[[r, c]];
            *kn = gvfalb_n[[r, c]] * rad_i_sin_alt + ct * gvfalbnosh_n[[r, c]];
        });

    (kup, kup_e, kup_s, kup_w, kup_n)
}

// ── Longwave: Ldown ────────────────────────────────────────────────────────

/// Compute Ldown (downwelling longwave) — Jonsson et al. 2006.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_ldown(
    esky: f32,
    ta: f32,
    tg_wall: f32,
    svf: ArrayView2<f32>,
    svf_veg: ArrayView2<f32>,
    svf_aveg: ArrayView2<f32>,
    emis_wall: f32,
    ci: f32,
    valid: ArrayView2<u8>,
) -> Array2<f32> {
    let ta_k = ta + KELVIN_OFFSET;
    let ta_k4 = ta_k.powi(4);
    let tg_wall_k4 = (ta + tg_wall + KELVIN_OFFSET).powi(4);
    let shape = svf.dim();
    let mut ldown = Array2::<f32>::zeros(shape);

    Zip::indexed(&mut ldown).par_for_each(|(r, c), ld| {
        if valid[[r, c]] == 0 {
            *ld = f32::NAN;
            return;
        }
        let sv = svf[[r, c]];
        let sv_veg = svf_veg[[r, c]];
        let sv_aveg = svf_aveg[[r, c]];

        let val = (sv + sv_veg - 1.0) * esky * SBC * ta_k4
            + (2.0 - sv_veg - sv_aveg) * emis_wall * SBC * ta_k4
            + (sv_aveg - sv) * emis_wall * SBC * tg_wall_k4
            + (2.0 - sv - sv_veg) * (1.0 - emis_wall) * esky * SBC * ta_k4;

        if ci < 0.95 {
            let c_cloud = 1.0 - ci;
            let val_cloudy = (sv + sv_veg - 1.0) * SBC * ta_k4
                + (2.0 - sv_veg - sv_aveg) * emis_wall * SBC * ta_k4
                + (sv_aveg - sv) * emis_wall * SBC * tg_wall_k4
                + (2.0 - sv - sv_veg) * (1.0 - emis_wall) * SBC * ta_k4;
            *ld = val * (1.0 - c_cloud) + val_cloudy * c_cloud;
        } else {
            *ld = val;
        }
    });

    ldown
}

// ── Shortwave: Kdown ───────────────────────────────────────────────────────

/// Compute Kdown (downwelling shortwave).
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_kdown(
    rad_i: f32,
    rad_d: f32,
    rad_g: f32,
    shadow: ArrayView2<f32>,
    sin_alt: f32,
    svfbuveg: ArrayView2<f32>,
    albedo_wall: f32,
    f_sh: ArrayView2<f32>,
    drad: ArrayView2<f32>,
    valid: ArrayView2<u8>,
) -> Array2<f32> {
    let shape = shadow.dim();
    let mut kdown = Array2::<f32>::zeros(shape);

    Zip::indexed(&mut kdown).par_for_each(|(r, c), kd| {
        if valid[[r, c]] == 0 {
            *kd = f32::NAN;
            return;
        }
        *kd = rad_i * shadow[[r, c]] * sin_alt
            + drad[[r, c]]
            + albedo_wall
                * (1.0 - svfbuveg[[r, c]])
                * (rad_g * (1.0 - f_sh[[r, c]]) + rad_d * f_sh[[r, c]]);
    });

    kdown
}

// ── ASVF cache ──────────────────────────────────────────────────────────────

/// Cached ASVF (`acos(sqrt(clamp(svf, 0, 1)))`) for a static SVF raster.
///
/// SVF is invariant across timesteps for a given surface/tile, so recomputing
/// ASVF every timestep is wasted work in anisotropic mode.
pub(crate) fn asvf_for_svf_cached(svf: ArrayView2<f32>) -> Arc<Vec<f32>> {
    const MAX_ENTRIES: usize = 16;

    type AsvfKey = (usize, usize, u64);
    #[derive(Default)]
    struct AsvfCache {
        map: HashMap<AsvfKey, Arc<Vec<f32>>>,
        lru: VecDeque<AsvfKey>,
    }

    fn fnv1a_u64(mut hash: u64, word: u64) -> u64 {
        const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;
        for b in word.to_le_bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    fn svf_key(svf: ArrayView2<f32>) -> AsvfKey {
        const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
        let mut hash = FNV_OFFSET;
        hash = fnv1a_u64(hash, svf.nrows() as u64);
        hash = fnv1a_u64(hash, svf.ncols() as u64);
        for &v in svf.iter() {
            hash = fnv1a_u64(hash, v.to_bits() as u64);
        }
        (svf.nrows(), svf.ncols(), hash)
    }

    let key = svf_key(svf);
    static CACHE: OnceLock<Mutex<AsvfCache>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(AsvfCache::default()));

    // Recover from poisoning: cached Arc<Vec<f32>> entries are immutable, so the
    // inner data is safe to keep reading after a panic in some other thread.
    let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(hit) = guard.map.get(&key).cloned() {
        if let Some(pos) = guard.lru.iter().position(|k| *k == key) {
            guard.lru.remove(pos);
        }
        guard.lru.push_back(key);
        return hit;
    }

    let data = svf
        .iter()
        .map(|&v| v.clamp(0.0, 1.0).sqrt().acos())
        .collect::<Vec<f32>>();
    let entry = Arc::new(data);

    while guard.map.len() >= MAX_ENTRIES {
        if let Some(oldest) = guard.lru.pop_front() {
            guard.map.remove(&oldest);
        } else {
            break;
        }
    }
    guard.map.insert(key, entry.clone());
    guard.lru.push_back(key);
    entry
}

// ── Directional side sums ───────────────────────────────────────────────────

/// Weighted sum of four directional side components with valid-mask handling.
///
/// Used to avoid materializing per-direction arrays in Tmrt-only pathways.
pub(crate) fn weighted_side_sum_four(
    a: ArrayView2<f32>,
    b: ArrayView2<f32>,
    c_arr: ArrayView2<f32>,
    d_arr: ArrayView2<f32>,
    valid: ArrayView2<u8>,
    weight: f32,
) -> Array2<f32> {
    let shape = a.dim();
    let mut sum = Array2::<f32>::zeros(shape);

    Zip::indexed(&mut sum).par_for_each(|(r, c), out| {
        if valid[[r, c]] == 0 {
            *out = f32::NAN;
            return;
        }
        *out = (a[[r, c]] + b[[r, c]] + c_arr[[r, c]] + d_arr[[r, c]]) * weight;
    });

    sum
}

/// Directional longwave side sum for anisotropic mode (lup_dir * 0.5).
pub(crate) fn lside_dirs_sum_aniso_from_lup(
    lup_e: ArrayView2<f32>,
    lup_s: ArrayView2<f32>,
    lup_w: ArrayView2<f32>,
    lup_n: ArrayView2<f32>,
    valid: ArrayView2<u8>,
) -> Array2<f32> {
    weighted_side_sum_four(lup_e, lup_s, lup_w, lup_n, valid, 0.5)
}

/// Directional shortwave side sum for anisotropic mode (kup_dir * 0.5).
pub(crate) fn kside_dirs_sum_aniso_from_kup(
    kup_e: ArrayView2<f32>,
    kup_s: ArrayView2<f32>,
    kup_w: ArrayView2<f32>,
    kup_n: ArrayView2<f32>,
    valid: ArrayView2<u8>,
) -> Array2<f32> {
    weighted_side_sum_four(kup_e, kup_s, kup_w, kup_n, valid, 0.5)
}

/// Sum north+east+south+west arrays with valid-mask handling.
pub(crate) fn side_sum_from_directional(
    north: ArrayView2<f32>,
    east: ArrayView2<f32>,
    south: ArrayView2<f32>,
    west: ArrayView2<f32>,
    valid: ArrayView2<u8>,
) -> Array2<f32> {
    let shape = north.dim();
    let mut sum = Array2::<f32>::zeros(shape);

    Zip::indexed(&mut sum).par_for_each(|(r, c), out| {
        if valid[[r, c]] == 0 {
            *out = f32::NAN;
            return;
        }
        *out = north[[r, c]] + east[[r, c]] + south[[r, c]] + west[[r, c]];
    });

    sum
}

// ── Patch LUT cache ─────────────────────────────────────────────────────────

pub(crate) struct PatchOptionLut {
    pub(crate) altitudes: Arc<Vec<f32>>,
    pub(crate) azimuths: Arc<Vec<f32>>,
    pub(crate) steradians: Arc<Vec<f32>>,
    pub(crate) altitude_sin: Arc<Vec<f32>>,
}

/// Cached patch LUTs keyed by `patch_option`.
///
/// Geometry is already cached in the Perez module; this cache adds derived
/// `sin(altitude)` values so anisotropic timesteps can skip per-patch trig.
pub(crate) fn patch_lut_for_option_cached(patch_option: i32) -> Arc<PatchOptionLut> {
    static CACHE: OnceLock<Mutex<HashMap<i32, Arc<PatchOptionLut>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    // Recover from poisoning: cached Arc<PatchOptionLut> entries are immutable.
    let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
    guard
        .entry(patch_option)
        .or_insert_with(|| {
            let (altitudes, azimuths, steradians) =
                crate::perez::patch_alt_azi_steradians_for_patch_option(patch_option);
            let deg2rad = PI / 180.0;
            let altitude_sin = Arc::new(
                altitudes
                    .iter()
                    .map(|alt| (alt * deg2rad).sin())
                    .collect::<Vec<f32>>(),
            );
            Arc::new(PatchOptionLut {
                altitudes,
                azimuths,
                steradians,
                altitude_sin,
            })
        })
        .clone()
}

// ── Anisotropic luminance from packed shadow matrices ──────────────────────

/// Compute anisotropic diffuse luminance sum directly from bitpacked shadow matrices.
///
/// Equivalent to:
/// ```text
///   diffsh = shmat_bit - (1 - vegshmat_bit) * (1 - psi)
///   ani_lum = sum_i(diffsh[:,:,i] * lv_lum[i])
/// ```
/// but avoids allocating a full `(rows, cols, patches)` float array.
pub(crate) fn compute_ani_lum_from_packed(
    shmat: ArrayView3<u8>,
    vegshmat: ArrayView3<u8>,
    lv_lum: ArrayView1<f32>,
    psi: f32,
    valid: ArrayView2<u8>,
) -> Array2<f32> {
    let (rows, cols, _) = shmat.dim();
    let n_patches = lv_lum.len();
    let mut out = Array2::<f32>::zeros((rows, cols));

    let ncols = cols;
    if let Some(out_slice) = out.as_slice_mut() {
        out_slice.par_iter_mut().enumerate().for_each(|(idx, v)| {
            let r = idx / ncols;
            let c = idx % ncols;

            if valid[[r, c]] == 0 {
                *v = f32::NAN;
                return;
            }

            let mut sum = 0.0_f32;
            for i in 0..n_patches {
                let byte = i >> 3;
                let bit = i & 7;
                let sh = ((shmat[[r, c, byte]] >> bit) & 1) as f32;
                let vsh = ((vegshmat[[r, c, byte]] >> bit) & 1) as f32;
                let diff = sh - (1.0 - vsh) * (1.0 - psi);
                sum += diff * lv_lum[i];
            }
            *v = sum;
        });
    } else {
        // Fallback for the rare non-contiguous case.
        for r in 0..rows {
            for c in 0..cols {
                if valid[[r, c]] == 0 {
                    out[[r, c]] = f32::NAN;
                    continue;
                }
                let mut sum = 0.0_f32;
                for i in 0..n_patches {
                    let byte = i >> 3;
                    let bit = i & 7;
                    let sh = ((shmat[[r, c, byte]] >> bit) & 1) as f32;
                    let vsh = ((vegshmat[[r, c, byte]] >> bit) & 1) as f32;
                    let diff = sh - (1.0 - vsh) * (1.0 - psi);
                    sum += diff * lv_lum[i];
                }
                out[[r, c]] = sum;
            }
        }
    }

    out
}
