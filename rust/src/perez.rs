//! Perez sky luminance distribution model (all-weather).
//!
//! Ported from Python `Perez_v3.py` and `create_patches.py`.
//!
//! Reference: Perez, Seals & Michalsky (1993), Solar Energy 50(3), 235–245.

use ndarray::{Array1, Array2};
use numpy::IntoPyArray;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

const PI: f32 = std::f32::consts::PI;
const DEG2RAD: f32 = PI / 180.0;
const RAD2DEG: f32 = 180.0 / PI;
const MIN_SUN_ELEVATION_DEG: f32 = 3.0;

// ── Perez model coefficients (8 clearness bins × 4 polynomial terms) ────────

const M_A1: [f32; 8] = [
    1.3525, -1.2219, -1.1000, -0.5484, -0.6000, -1.0156, -1.0000, -1.0500,
];
const M_A2: [f32; 8] = [
    -0.2576, -0.7730, -0.2515, -0.6654, -0.3566, -0.3670, 0.0211, 0.0289,
];
const M_A3: [f32; 8] = [
    -0.2690, 1.4148, 0.8952, -0.2672, -2.5000, 1.0078, 0.5025, 0.4260,
];
const M_A4: [f32; 8] = [
    -1.4366, 1.1016, 0.0156, 0.7117, 2.3250, 1.4051, -0.5119, 0.3590,
];

const M_B1: [f32; 8] = [
    -0.7670, -0.2054, 0.2782, 0.7234, 0.2937, 0.2875, -0.3000, -0.3250,
];
const M_B2: [f32; 8] = [
    0.0007, 0.0367, -0.1812, -0.6219, 0.0496, -0.5328, 0.1922, 0.1156,
];
const M_B3: [f32; 8] = [
    1.2734, -3.9128, -4.5000, -5.6812, -5.6812, -3.8500, 0.7023, 0.7781,
];
const M_B4: [f32; 8] = [
    -0.1233, 0.9156, 1.1766, 2.6297, 1.8415, 3.3750, -1.6317, 0.0025,
];

const M_C1: [f32; 8] = [
    2.8000, 6.9750, 24.7219, 33.3389, 21.0000, 14.0000, 19.0000, 31.0625,
];
const M_C2: [f32; 8] = [
    0.6004, 0.1774, -13.0812, -18.3000, -4.7656, -0.9999, -5.0000, -14.5000,
];
const M_C3: [f32; 8] = [
    1.2375, 6.4477, -37.7000, -62.2500, -21.5906, -7.1406, 1.2438, -46.1148,
];
const M_C4: [f32; 8] = [
    1.0000, -0.1239, 34.8438, 52.0781, 7.2492, 7.5469, -1.9094, 55.3750,
];

const M_D1: [f32; 8] = [
    1.8734, -1.5798, -5.0000, -3.5000, -3.5000, -3.4000, -4.0000, -7.2312,
];
const M_D2: [f32; 8] = [
    0.6297, -0.5081, 1.5218, 0.0016, -0.1554, -0.1078, 0.0250, 0.4050,
];
const M_D3: [f32; 8] = [
    0.9738, -1.7812, 3.9229, 1.1477, 1.4062, -1.0750, 0.3844, 13.3500,
];
const M_D4: [f32; 8] = [
    0.2809, 0.1080, -2.6204, 0.1062, 0.3988, 1.5702, 0.2656, 0.6234,
];

const M_E1: [f32; 8] = [
    0.0356, 0.2624, -0.0156, 0.4659, 0.0032, -0.0672, 1.0468, 1.5000,
];
const M_E2: [f32; 8] = [
    -0.1246, 0.0672, 0.1597, -0.3296, 0.0766, 0.4016, -0.3788, -0.6426,
];
const M_E3: [f32; 8] = [
    -0.5718, -0.2190, 0.4199, -0.0876, -0.0656, 0.3017, -2.4517, 1.8564,
];
const M_E4: [f32; 8] = [
    0.9938, -0.4285, -0.5562, -0.0329, -0.1294, -0.4844, 1.4656, 0.5636,
];

// ── Patch layout (Robinson & Stone sky vault decomposition) ─────────────────

#[derive(Clone)]
struct PatchLayoutCache {
    altitudes: Arc<Vec<f32>>,
    azimuths: Arc<Vec<f32>>,
    steradians: Arc<Vec<f32>>,
}

/// Create sky vault patches for a given patch option.
///
/// Returns `(altitudes_deg, azimuths_deg)` — each a `Vec<f32>` of length N patches.
pub(crate) fn create_patches(patch_option: i32) -> (Vec<f32>, Vec<f32>) {
    let (skyvault_alt_int, azistart, patches_in_band): (&[f32], &[f32], &[i32]) = match patch_option
    {
        1 => (
            &[6., 18., 30., 42., 54., 66., 78., 90.],
            &[0., 4., 2., 5., 8., 0., 10., 0.],
            &[30, 30, 24, 24, 18, 12, 6, 1],
        ),
        2 => (
            &[6., 18., 30., 42., 54., 66., 78., 90.],
            &[0., 4., 2., 5., 8., 0., 10., 0.],
            &[31, 30, 28, 24, 19, 13, 7, 1],
        ),
        3 => (
            &[6., 18., 30., 42., 54., 66., 78., 90.],
            &[0., 4., 2., 5., 8., 0., 10., 0.],
            &[62, 60, 56, 48, 38, 26, 14, 1],
        ),
        4 => (
            &[
                3., 9., 15., 21., 27., 33., 39., 45., 51., 57., 63., 69., 75., 81., 90.,
            ],
            &[0., 0., 4., 4., 2., 2., 5., 5., 8., 8., 0., 0., 10., 10., 0.],
            &[62, 62, 60, 60, 56, 56, 48, 48, 38, 38, 26, 26, 14, 14, 1],
        ),
        _ => (
            // Default to option 2 (153 patches)
            &[6., 18., 30., 42., 54., 66., 78., 90.],
            &[0., 4., 2., 5., 8., 0., 10., 0.],
            &[31, 30, 28, 24, 19, 13, 7, 1],
        ),
    };

    let total: usize = patches_in_band.iter().map(|&p| p as usize).sum();
    let mut altitudes = Vec::with_capacity(total);
    let mut azimuths = Vec::with_capacity(total);

    for (j, &alt) in skyvault_alt_int.iter().enumerate() {
        let n = patches_in_band[j] as usize;
        let azi_step = 360.0 / patches_in_band[j] as f32;
        let azi_off = azistart[j];
        for k in 0..n {
            altitudes.push(alt);
            azimuths.push(k as f32 * azi_step + azi_off);
        }
    }

    (altitudes, azimuths)
}

/// Compute steradians for each sky patch.
///
/// Only depends on the patch altitude layout (constant for a given patch_option).
pub(crate) fn compute_steradians(altitudes: &[f32]) -> Array1<f32> {
    let n = altitudes.len();
    let mut steradian = Array1::<f32>::zeros(n);

    // Unique altitudes and counts
    let mut unique_alts: Vec<f32> = Vec::new();
    let mut counts: Vec<i32> = Vec::new();
    for &a in altitudes {
        if let Some(pos) = unique_alts.iter().position(|&u| (u - a).abs() < 1e-6) {
            counts[pos] += 1;
        } else {
            unique_alts.push(a);
            counts.push(1);
        }
    }

    let first_alt = altitudes[0];
    for i in 0..n {
        let alt_i = altitudes[i];
        let count = counts[unique_alts
            .iter()
            .position(|&u| (u - alt_i).abs() < 1e-6)
            .unwrap()];
        if count > 1 {
            steradian[i] = (360.0 / count as f32)
                * DEG2RAD
                * (((alt_i + first_alt) * DEG2RAD).sin() - ((alt_i - first_alt) * DEG2RAD).sin());
        } else {
            // Single patch in band (e.g. 90°)
            let prev_alt = altitudes[i - 1];
            steradian[i] = (360.0 / count as f32)
                * DEG2RAD
                * ((alt_i * DEG2RAD).sin() - ((prev_alt + first_alt) * DEG2RAD).sin());
        }
    }

    steradian
}

fn patch_layout_for_option(patch_option: i32) -> PatchLayoutCache {
    static CACHE: OnceLock<Mutex<HashMap<i32, PatchLayoutCache>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().expect("patch layout cache mutex poisoned");

    guard
        .entry(patch_option)
        .or_insert_with(|| {
            let (alts, azis) = create_patches(patch_option);
            let ster = compute_steradians(&alts).to_vec();
            PatchLayoutCache {
                altitudes: Arc::new(alts),
                azimuths: Arc::new(azis),
                steradians: Arc::new(ster),
            }
        })
        .clone()
}

/// Cached patch geometry for a patch option.
///
/// Returns `(altitudes_deg, azimuths_deg, steradians)`, each backed by an `Arc<Vec<f32>>`
/// so callers can reuse stable buffers across timesteps without reallocating.
pub(crate) fn patch_alt_azi_steradians_for_patch_option(
    patch_option: i32,
) -> (Arc<Vec<f32>>, Arc<Vec<f32>>, Arc<Vec<f32>>) {
    let layout = patch_layout_for_option(patch_option);
    (layout.altitudes, layout.azimuths, layout.steradians)
}

/// Cached steradians for a patch layout option.
///
/// Perez all-weather sky luminance distribution.
///
/// Returns an Nx3 array: `[altitude_deg, azimuth_deg, luminance]` per patch.
///
/// Matches the Python `Perez_v3` with `patchchoice=1`.
pub(crate) fn perez_v3(
    zen_deg: f32,
    azimuth_deg: f32,
    rad_d: f32,
    rad_i: f32,
    jday: i32,
    patch_option: i32,
) -> Array2<f32> {
    let (altitudes, azimuths, _) = patch_alt_azi_steradians_for_patch_option(patch_option);
    let n = altitudes.len();
    let altitude_deg = 90.0 - zen_deg;

    // Low sun or very low diffuse → uniform distribution
    if altitude_deg < MIN_SUN_ELEVATION_DEG || rad_d < 10.0 {
        let uniform_lv = 1.0 / n as f32;
        let mut lv = Array2::<f32>::zeros((n, 3));
        for i in 0..n {
            lv[[i, 0]] = altitudes[i];
            lv[[i, 1]] = azimuths[i];
            lv[[i, 2]] = uniform_lv;
        }
        return lv;
    }

    let zen = zen_deg * DEG2RAD;
    let azimuth = azimuth_deg * DEG2RAD;
    let altitude = altitude_deg * DEG2RAD;

    // Sky clearness
    let idh_safe = rad_d.max(1.0);
    let perez_clearness =
        ((idh_safe + rad_i) / idh_safe + 1.041 * zen.powi(3)) / (1.0 + 1.041 * zen.powi(3));

    // Extra-terrestrial radiation (Robinson correction)
    let day_angle = jday as f32 * 2.0 * PI / 365.0;
    let i0 = 1367.0
        * (1.00011
            + 0.034221 * day_angle.cos()
            + 0.00128 * day_angle.sin()
            + 0.000719 * (2.0 * day_angle).cos()
            + 0.000077 * (2.0 * day_angle).sin());

    // Optical air mass (Kasten & Young 1989)
    let air_mass = if altitude >= 10.0 * DEG2RAD {
        1.0 / altitude.sin()
    } else if altitude > 0.0 {
        let alt_deg = altitude * RAD2DEG;
        1.0 / (altitude.sin() + 0.50572 * (alt_deg + 6.07995_f32).powf(-1.6364))
    } else {
        40.0
    }
    .min(40.0);

    // Sky brightness
    let perez_brightness = if rad_d <= 10.0 {
        0.0
    } else {
        air_mass * rad_d / i0
    };

    // Clearness bin index (0–7)
    let bin = if perez_clearness < 1.065 {
        0
    } else if perez_clearness < 1.230 {
        1
    } else if perez_clearness < 1.500 {
        2
    } else if perez_clearness < 1.950 {
        3
    } else if perez_clearness < 2.800 {
        4
    } else if perez_clearness < 4.500 {
        5
    } else if perez_clearness < 6.200 {
        6
    } else {
        7
    };

    // Perez model parameters
    let m_a = M_A1[bin] + M_A2[bin] * zen + perez_brightness * (M_A3[bin] + M_A4[bin] * zen);
    let m_b = M_B1[bin] + M_B2[bin] * zen + perez_brightness * (M_B3[bin] + M_B4[bin] * zen);
    let m_e = M_E1[bin] + M_E2[bin] * zen + perez_brightness * (M_E3[bin] + M_E4[bin] * zen);

    let (m_c, m_d) = if bin > 0 {
        let c = M_C1[bin] + M_C2[bin] * zen + perez_brightness * (M_C3[bin] + M_C4[bin] * zen);
        let d = M_D1[bin] + M_D2[bin] * zen + perez_brightness * (M_D3[bin] + M_D4[bin] * zen);
        (c, d)
    } else {
        // Different equations for clearness bin 0 (Robinson)
        let c = (perez_brightness * (M_C1[0] + M_C2[0] * zen))
            .powf(M_C3[0])
            .exp()
            - 1.0;
        let d = -(perez_brightness * (M_D1[0] + M_D2[0] * zen)).exp()
            + M_D3[0]
            + perez_brightness * M_D4[0];
        (c, d)
    };

    // Compute luminance for each patch
    let sin_alt = altitude.sin();
    let cos_alt = altitude.cos();
    let mut lv_vals = Vec::with_capacity(n);
    let mut lv_sum: f32 = 0.0;

    for i in 0..n {
        let sv_alt = altitudes[i] * DEG2RAD;
        let sv_azi = azimuths[i] * DEG2RAD;
        let sv_zen = (90.0 - altitudes[i]) * DEG2RAD;

        // Angular distance from sun (Robinson formula)
        let cos_sky_sun =
            sv_alt.sin() * sin_alt + cos_alt * sv_alt.cos() * (sv_azi - azimuth).abs().cos();
        let cos_sv_zen = sv_zen.cos();

        // Perez luminance
        let horizon = 1.0 + m_a * (m_b / cos_sv_zen).exp();
        let ang = cos_sky_sun.clamp(-1.0, 1.0).acos();
        let circumsolar = 1.0 + m_c * (m_d * ang).exp() + m_e * cos_sky_sun * cos_sky_sun;
        let val = horizon * circumsolar;

        lv_vals.push(val);
        lv_sum += val;
    }

    // Check for negative luminances → uniform fallback
    let has_negative = lv_vals.iter().any(|&v| v < 0.0);
    if has_negative || lv_sum <= 0.0 {
        let uniform = 1.0 / n as f32;
        for v in lv_vals.iter_mut() {
            *v = uniform;
        }
    } else {
        // Normalise
        for v in lv_vals.iter_mut() {
            *v /= lv_sum;
        }
    }

    // Build Nx3 output: [altitude_deg, azimuth_deg, luminance]
    let mut lv = Array2::<f32>::zeros((n, 3));
    for i in 0..n {
        lv[[i, 0]] = altitudes[i];
        lv[[i, 1]] = azimuths[i];
        lv[[i, 2]] = lv_vals[i];
    }

    lv
}

// ── PyO3 wrappers (for testing parity with Python implementation) ───────────

#[pyfunction]
pub fn perez_v3_py(
    py: Python,
    zen_deg: f32,
    azimuth_deg: f32,
    rad_d: f32,
    rad_i: f32,
    jday: i32,
    patch_option: i32,
) -> PyObject {
    perez_v3(zen_deg, azimuth_deg, rad_d, rad_i, jday, patch_option)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
pub fn compute_steradians_py(py: Python, patch_option: i32) -> PyObject {
    let (alts, _) = create_patches(patch_option);
    compute_steradians(&alts).into_pyarray(py).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_patches_option2_count() {
        let (alts, azis) = create_patches(2);
        assert_eq!(alts.len(), 153);
        assert_eq!(azis.len(), 153);
    }

    #[test]
    fn test_perez_uniform_low_sun() {
        let lv = perez_v3(89.0, 180.0, 50.0, 10.0, 180, 2);
        assert_eq!(lv.shape(), &[153, 3]);
        // Low altitude (1°) → uniform
        let first_lum = lv[[0, 2]];
        let last_lum = lv[[152, 2]];
        assert!((first_lum - last_lum).abs() < 1e-10);
    }

    #[test]
    fn test_perez_normalised() {
        let lv = perez_v3(30.0, 180.0, 200.0, 400.0, 180, 2);
        let sum: f32 = lv.column(2).sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "luminance sum = {sum}, expected 1.0"
        );
    }

    #[test]
    fn test_steradians_length() {
        let (alts, _) = create_patches(2);
        let ster = compute_steradians(&alts);
        assert_eq!(ster.len(), 153);
    }
}
