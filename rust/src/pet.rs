use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

/// Physical constants for PET calculation
const PO: f32 = 1013.25; // Reference pressure (hPa)
const P: f32 = 1013.25; // Atmospheric pressure (hPa)
const ROB: f32 = 1.06;
const CB: f32 = 3.64 * 1000.0;
const EMSK: f32 = 0.99;
const EMCL: f32 = 0.95;
const EVAP: f32 = 2.42e6;
const SIGMA: f32 = 5.67051e-8;
const CAIR: f32 = 1.01 * 1000.0;

/// Calculate PET for a single point.
///
/// Parameters:
/// - ta: Air temperature (°C)
/// - rh: Relative humidity (%)
/// - tmrt: Mean radiant temperature (°C)
/// - v: Wind speed at pedestrian height (m/s)
/// - mbody: Body mass (kg)
/// - age: Age (years)
/// - ht: Height (m)
/// - work: Activity level (W)
/// - icl: Clothing insulation (clo, 0-5)
/// - sex: 1=male, 2=female
#[inline]
fn pet_single(
    ta: f32,
    rh: f32,
    tmrt: f32,
    v: f32,
    mbody: f32,
    age: f32,
    ht: f32,
    work: f32,
    icl: f32,
    sex: i32,
    warn_flag: Option<&AtomicBool>,
) -> f32 {
    // Humidity conversion
    let vps = 6.107 * 10.0_f32.powf(7.5 * ta / (238.0 + ta));
    let vpa = rh * vps / 100.0;

    let eta = 0.0;

    // INBODY - metabolic rate calculation
    let metbf = 3.19
        * mbody.powf(0.75)
        * (1.0 + 0.004 * (30.0 - age) + 0.018 * ((ht * 100.0 / mbody.powf(1.0 / 3.0)) - 42.1));
    let metbm = 3.45
        * mbody.powf(0.75)
        * (1.0 + 0.004 * (30.0 - age) + 0.010 * ((ht * 100.0 / mbody.powf(1.0 / 3.0)) - 43.4));

    let met = if sex == 1 { metbm + work } else { metbf + work };

    let h = met * (1.0 - eta);
    let rtv = 1.44e-6 * met;

    // Sensible respiration energy
    let tex = 0.47 * ta + 21.0;
    let eres = CAIR * (ta - tex) * rtv;

    // Latent respiration energy
    let vpex = 6.11 * 10.0_f32.powf(7.45 * tex / (235.0 + tex));
    let erel = 0.623 * EVAP / P * (vpa - vpex) * rtv;
    let ere = eres + erel;

    // Calculation constants
    let feff = 0.725;
    let adu = 0.203 * mbody.powf(0.425) * ht.powf(0.725);
    let mut facl = (-2.36 + 173.51 * icl - 100.76 * icl * icl + 19.28 * icl.powi(3)) / 100.0;
    if facl > 1.0 {
        facl = 1.0;
    }
    let rcl = (icl / 6.45) / facl;

    let mut y = 1.0;
    if icl < 2.0 {
        y = (ht - 0.2) / ht;
    }
    if icl <= 0.6 {
        y = 0.5;
    }
    if icl <= 0.3 {
        y = 0.1;
    }

    let fcl = 1.0 + 0.15 * icl;
    let r2 = adu * (fcl - 1.0 + facl) / (2.0 * std::f32::consts::PI * ht * y);
    let r1 = facl * adu / (2.0 * std::f32::consts::PI * ht * y);
    let di = r2 - r1;
    let acl = adu * facl + adu * (fcl - 1.0);

    let mut tcore = [0.0_f32; 8];

    let mut wetsk = 0.0;
    let mut hc = 2.67 + 6.5 * v.powf(0.67);
    hc *= (P / PO).powf(0.55);

    let c_1 = h + ere;
    let he = 0.633 * hc / (P * CAIR);
    let fec = 1.0 / (1.0 + 0.92 * hc * rcl);
    let htcl = 6.28 * ht * y * di / (rcl * (r2 / r1).ln() * acl);
    let aeff = adu * feff;
    let c_2 = adu * ROB * CB;
    let c_5 = 0.0208 * c_2;
    let c_6 = 0.76075 * c_2;
    let rdsk = 0.79e7;
    let rdcl = 0.0;

    let mut count2 = 0;
    let mut j = 1_usize;

    let mut tsk = 34.0_f32;
    let mut tcl = (ta + tmrt + tsk) / 3.0;
    let mut vb = 0.0_f32;
    let mut esw = 0.0_f32;
    let mut vpts = 0.0_f32;
    let mut c_9 = 0.0_f32;
    let mut c_11 = 0.0_f32;

    while count2 == 0 && j < 7 {
        tsk = 34.0;
        let mut count1 = 0;
        tcl = (ta + tmrt + tsk) / 3.0;
        let mut count3 = 1;
        let mut enbal2 = 0.0_f32;

        while count1 <= 3 {
            let mut enbal = 0.0_f32;

            while enbal * enbal2 >= 0.0 && count3 < 200 {
                enbal2 = enbal;

                let rclo2 = EMCL * SIGMA * ((tcl + 273.2).powi(4) - (tmrt + 273.2).powi(4)) * feff;
                tsk = 1.0 / htcl * (hc * (tcl - ta) + rclo2) + tcl;

                // Radiation balance
                let rbare = aeff
                    * (1.0 - facl)
                    * EMSK
                    * SIGMA
                    * ((tmrt + 273.2).powi(4) - (tsk + 273.2).powi(4));
                let rclo =
                    feff * acl * EMCL * SIGMA * ((tmrt + 273.2).powi(4) - (tcl + 273.2).powi(4));
                let rsum = rbare + rclo;

                // Convection
                let cbare = hc * (ta - tsk) * adu * (1.0 - facl);
                let cclo = hc * (ta - tcl) * acl;
                let csum = cbare + cclo;

                // Core temperature
                let c_3 = 18.0 - 0.5 * tsk;
                let c_4 = 5.28 * adu * c_3;
                let c_7 = c_4 - c_6 - tsk * c_5;
                let c_8 = -c_1 * c_3 - tsk * c_4 + tsk * c_6;
                c_9 = c_7 * c_7 - 4.0 * c_5 * c_8;
                let c_10 = 5.28 * adu - c_6 - c_5 * tsk;
                c_11 = c_10 * c_10 - 4.0 * c_5 * (c_6 * tsk - c_1 - 5.28 * adu * tsk);

                let tsk_adj = if tsk == 36.0 { 36.01 } else { tsk };

                tcore[7] = c_1 / (5.28 * adu + c_2 * 6.3 / 3600.0) + tsk_adj;
                tcore[3] = c_1
                    / (5.28 * adu + (c_2 * 6.3 / 3600.0) / (1.0 + 0.5 * (34.0 - tsk_adj)))
                    + tsk_adj;

                if c_11 >= 0.0 {
                    tcore[6] = (-c_10 - c_11.sqrt()) / (2.0 * c_5);
                    tcore[1] = (-c_10 + c_11.sqrt()) / (2.0 * c_5);
                }
                if c_9 >= 0.0 {
                    tcore[2] = (-c_7 + c_9.abs().sqrt()) / (2.0 * c_5);
                    tcore[5] = (-c_7 - c_9.abs().sqrt()) / (2.0 * c_5);
                }
                tcore[4] = c_1 / (5.28 * adu + c_2 * 1.0 / 40.0) + tsk_adj;

                // Transpiration
                let tbody = 0.1 * tsk + 0.9 * tcore[j];
                let mut sw = 304.94 * (tbody - 36.6) * adu / 3600000.0;
                vpts = 6.11 * 10.0_f32.powf(7.45 * tsk / (235.0 + tsk));

                if tbody <= 36.6 {
                    sw = 0.0;
                }
                if sex == 2 {
                    sw *= 0.7;
                }
                let eswphy = -sw * EVAP;

                let eswpot = he * (vpa - vpts) * adu * EVAP * fec;
                wetsk = eswphy / eswpot;
                if wetsk > 1.0 {
                    wetsk = 1.0;
                }
                let eswdif = eswphy - eswpot;
                esw = if eswdif <= 0.0 { eswpot } else { eswphy };
                if esw > 0.0 {
                    esw = 0.0;
                }

                // Diffusion
                let ed = EVAP / (rdsk + rdcl) * adu * (1.0 - wetsk) * (vpa - vpts);

                // MAX VB
                let mut vb1 = 34.0 - tsk;
                let mut vb2 = tcore[j] - 36.6;
                if vb2 < 0.0 {
                    vb2 = 0.0;
                }
                if vb1 < 0.0 {
                    vb1 = 0.0;
                }
                vb = (6.3 + 75.0 * vb2) / (1.0 + 0.5 * vb1);

                // Energy balance
                enbal = h + ed + ere + esw + csum + rsum;

                // Clothing temperature iteration
                let xx = match count1 {
                    0 => 1.0,
                    1 => 0.1,
                    2 => 0.01,
                    _ => 0.001,
                };

                if enbal > 0.0 {
                    tcl += xx;
                } else {
                    tcl -= xx;
                }

                count3 += 1;
            }
            count1 += 1;
            enbal2 = 0.0;
        }

        // Check convergence conditions for different j modes
        let converged = match j {
            2 | 5 => c_9 >= 0.0 && tcore[j] >= 36.6 && tsk <= 34.050,
            6 | 1 => c_11 > 0.0 && tcore[j] >= 36.6 && tsk > 33.850,
            3 => tcore[j] < 36.6 && tsk <= 34.000,
            7 => tcore[j] < 36.6 && tsk > 34.000,
            4 => true,
            _ => false,
        };

        if converged {
            let vb_check = (j != 4 && vb >= 91.0) || (j == 4 && vb < 89.0);
            if !vb_check {
                if vb > 90.0 {
                    vb = 90.0;
                }
                count2 = 1;
            }
        }

        j += 1;
    }

    // PET calculation phase
    let mut tx = ta;
    let mut enbal2 = 0.0_f32;
    let mut count1 = 0;

    hc = 2.67 + 6.5 * 0.1_f32.powf(0.67);
    hc *= (P / PO).powf(0.55);

    let mut count3 = 1_i32;
    let mut hit_limit = false;
    while count1 <= 3 {
        let mut enbal = 0.0_f32;

        while enbal * enbal2 >= 0.0 && count3 < 200 {
            enbal2 = enbal;
            count3 += 1;

            // Radiation balance
            let rbare =
                aeff * (1.0 - facl) * EMSK * SIGMA * ((tx + 273.2).powi(4) - (tsk + 273.2).powi(4));
            let rclo = feff * acl * EMCL * SIGMA * ((tx + 273.2).powi(4) - (tcl + 273.2).powi(4));
            let rsum = rbare + rclo;

            // Convection
            let cbare = hc * (tx - tsk) * adu * (1.0 - facl);
            let cclo = hc * (tx - tcl) * acl;
            let csum = cbare + cclo;

            // Diffusion
            let ed = EVAP / (rdsk + rdcl) * adu * (1.0 - wetsk) * (12.0 - vpts);

            // Respiration
            let tex = 0.47 * tx + 21.0;
            let eres = CAIR * (tx - tex) * rtv;
            let vpex = 6.11 * 10.0_f32.powf(7.45 * tex / (235.0 + tex));
            let erel = 0.623 * EVAP / P * (12.0 - vpex) * rtv;
            let ere = eres + erel;

            // Energy balance
            enbal = h + ed + ere + esw + csum + rsum;

            // Iteration step
            let xx = match count1 {
                0 => 1.0,
                1 => 0.1,
                2 => 0.01,
                _ => 0.001,
            };

            if enbal > 0.0 {
                tx -= xx;
            } else if enbal < 0.0 {
                tx += xx;
            }
        }
        if count3 >= 200 {
            hit_limit = true;
        }
        count1 += 1;
        count3 = 1;
        enbal2 = 0.0;
    }

    if hit_limit {
        if let Some(flag) = warn_flag {
            // Only emit one warning per grid call to avoid flooding stderr.
            if !flag.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "WARNING: PET solver did not converge within 200 iterations \
                     (ta={:.1}, tmrt={:.1}, v={:.2}). Results may be inaccurate.",
                    ta, tmrt, v
                );
            }
        } else {
            eprintln!(
                "WARNING: PET solver did not converge within 200 iterations \
                 (ta={:.1}, tmrt={:.1}, v={:.2}). Results may be inaccurate.",
                ta, tmrt, v
            );
        }
    }

    tx
}

/// Calculate PET for a single point (Python interface).
#[pyfunction]
pub fn pet_calculate(
    ta: f32,
    rh: f32,
    tmrt: f32,
    va: f32,
    mbody: f32,
    age: f32,
    height: f32,
    activity: f32,
    clo: f32,
    sex: i32,
) -> f32 {
    pet_single(ta, rh, tmrt, va, mbody, age, height, activity, clo, sex, None)
}

/// Calculate PET for a 2D grid using parallel processing.
///
/// Parameters:
/// - ta: Air temperature (°C) - scalar
/// - rh: Relative humidity (%) - scalar
/// - tmrt: Mean radiant temperature grid (°C)
/// - va: Wind speed grid (m/s)
/// - mbody: Body mass (kg)
/// - age: Age (years)
/// - height: Height (m)
/// - activity: Activity level (W)
/// - clo: Clothing insulation (clo)
/// - sex: 1=male, 2=female
#[pyfunction]
pub fn pet_grid<'py>(
    py: Python<'py>,
    ta: f32,
    rh: f32,
    tmrt: PyReadonlyArray2<'py, f32>,
    va: PyReadonlyArray2<'py, f32>,
    mbody: f32,
    age: f32,
    height: f32,
    activity: f32,
    clo: f32,
    sex: i32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let tmrt_arr = tmrt.as_array();
    let va_arr = va.as_array();

    let (rows, cols) = tmrt_arr.dim();

    // Create output array
    let mut result = ndarray::Array2::zeros((rows, cols));
    let warn_flag = AtomicBool::new(false);

    // Process in parallel using rayon.
    // result was just allocated via Array2::zeros so as_slice_mut() is Some.
    result
        .as_slice_mut()
        .expect("result: freshly-allocated Array2::zeros is contiguous")
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, out)| {
            let row = idx / cols;
            let col = idx % cols;

            let tmrt_val = tmrt_arr[[row, col]];
            let va_val = va_arr[[row, col]];

            // Check for invalid pixel values (NaN, nodata, non-finite)
            if !tmrt_val.is_finite() || !va_val.is_finite() || va_val <= 0.0 || tmrt_val <= -999.0 {
                *out = f32::NAN;
            } else {
                *out = pet_single(
                    ta, rh, tmrt_val, va_val, mbody, age, height, activity, clo, sex,
                    Some(&warn_flag),
                );
            }
        });

    Ok(PyArray2::from_owned_array(py, result))
}
