use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Calculate UTCI polynomial approximation for a single point.
/// This is the 6th order polynomial from Bröde et al.
#[inline]
fn utci_polynomial(d_tmrt: f32, ta: f32, va: f32, pa: f32) -> f32 {
    // Pre-compute powers to reduce redundant calculations
    let ta2 = ta * ta;
    let ta3 = ta2 * ta;
    let ta4 = ta3 * ta;
    let ta5 = ta4 * ta;
    let ta6 = ta5 * ta;

    let va2 = va * va;
    let va3 = va2 * va;
    let va4 = va3 * va;
    let va5 = va4 * va;
    let va6 = va5 * va;

    let d2 = d_tmrt * d_tmrt;
    let d3 = d2 * d_tmrt;
    let d4 = d3 * d_tmrt;
    let d5 = d4 * d_tmrt;
    let d6 = d5 * d_tmrt;

    let pa2 = pa * pa;
    let pa3 = pa2 * pa;
    let pa4 = pa3 * pa;
    let pa5 = pa4 * pa;
    let pa6 = pa5 * pa;

    ta + 6.07562052e-01
        + (-2.27712343e-02) * ta
        + (8.06470249e-04) * ta2
        + (-1.54271372e-04) * ta3
        + (-3.24651735e-06) * ta4
        + (7.32602852e-08) * ta5
        + (1.35959073e-09) * ta6
        + (-2.25836520e00) * va
        + (8.80326035e-02) * ta * va
        + (2.16844454e-03) * ta2 * va
        + (-1.53347087e-05) * ta3 * va
        + (-5.72983704e-07) * ta4 * va
        + (-2.55090145e-09) * ta5 * va
        + (-7.51269505e-01) * va2
        + (-4.08350271e-03) * ta * va2
        + (-5.21670675e-05) * ta2 * va2
        + (1.94544667e-06) * ta3 * va2
        + (1.14099531e-08) * ta4 * va2
        + (1.58137256e-01) * va3
        + (-6.57263143e-05) * ta * va3
        + (2.22697524e-07) * ta2 * va3
        + (-4.16117031e-08) * ta3 * va3
        + (-1.27762753e-02) * va4
        + (9.66891875e-06) * ta * va4
        + (2.52785852e-09) * ta2 * va4
        + (4.56306672e-04) * va5
        + (-1.74202546e-07) * ta * va5
        + (-5.91491269e-06) * va6
        + (3.98374029e-01) * d_tmrt
        + (1.83945314e-04) * ta * d_tmrt
        + (-1.73754510e-04) * ta2 * d_tmrt
        + (-7.60781159e-07) * ta3 * d_tmrt
        + (3.77830287e-08) * ta4 * d_tmrt
        + (5.43079673e-10) * ta5 * d_tmrt
        + (-2.00518269e-02) * va * d_tmrt
        + (8.92859837e-04) * ta * va * d_tmrt
        + (3.45433048e-06) * ta2 * va * d_tmrt
        + (-3.77925774e-07) * ta3 * va * d_tmrt
        + (-1.69699377e-09) * ta4 * va * d_tmrt
        + (1.69992415e-04) * va2 * d_tmrt
        + (-4.99204314e-05) * ta * va2 * d_tmrt
        + (2.47417178e-07) * ta2 * va2 * d_tmrt
        + (1.07596466e-08) * ta3 * va2 * d_tmrt
        + (8.49242932e-05) * va3 * d_tmrt
        + (1.35191328e-06) * ta * va3 * d_tmrt
        + (-6.21531254e-09) * ta2 * va3 * d_tmrt
        + (-4.99410301e-06) * va4 * d_tmrt
        + (-1.89489258e-08) * ta * va4 * d_tmrt
        + (8.15300114e-08) * va5 * d_tmrt
        + (7.55043090e-04) * d2
        + (-5.65095215e-05) * ta * d2
        + (-4.52166564e-07) * ta2 * d2
        + (2.46688878e-08) * ta3 * d2
        + (2.42674348e-10) * ta4 * d2
        + (1.54547250e-04) * va * d2
        + (5.24110970e-06) * ta * va * d2
        + (-8.75874982e-08) * ta2 * va * d2
        + (-1.50743064e-09) * ta3 * va * d2
        + (-1.56236307e-05) * va2 * d2
        + (-1.33895614e-07) * ta * va2 * d2
        + (2.49709824e-09) * ta2 * va2 * d2
        + (6.51711721e-07) * va3 * d2
        + (1.94960053e-09) * ta * va3 * d2
        + (-1.00361113e-08) * va4 * d2
        + (-1.21206673e-05) * d3
        + (-2.18203660e-07) * ta * d3
        + (7.51269482e-09) * ta2 * d3
        + (9.79063848e-11) * ta3 * d3
        + (1.25006734e-06) * va * d3
        + (-1.81584736e-09) * ta * va * d3
        + (-3.52197671e-10) * ta2 * va * d3
        + (-3.36514630e-08) * va2 * d3
        + (1.35908359e-10) * ta * va2 * d3
        + (4.17032620e-10) * va3 * d3
        + (-1.30369025e-09) * d4
        + (4.13908461e-10) * ta * d4
        + (9.22652254e-12) * ta2 * d4
        + (-5.08220384e-09) * va * d4
        + (-2.24730961e-11) * ta * va * d4
        + (1.17139133e-10) * va2 * d4
        + (6.62154879e-10) * d5
        + (4.03863260e-13) * ta * d5
        + (1.95087203e-12) * va * d5
        + (-4.73602469e-12) * d6
        + (5.12733497e00) * pa
        + (-3.12788561e-01) * ta * pa
        + (-1.96701861e-02) * ta2 * pa
        + (9.99690870e-04) * ta3 * pa
        + (9.51738512e-06) * ta4 * pa
        + (-4.66426341e-07) * ta5 * pa
        + (5.48050612e-01) * va * pa
        + (-3.30552823e-03) * ta * va * pa
        + (-1.64119440e-03) * ta2 * va * pa
        + (-5.16670694e-06) * ta3 * va * pa
        + (9.52692432e-07) * ta4 * va * pa
        + (-4.29223622e-02) * va2 * pa
        + (5.00845667e-03) * ta * va2 * pa
        + (1.00601257e-06) * ta2 * va2 * pa
        + (-1.81748644e-06) * ta3 * va2 * pa
        + (-1.25813502e-03) * va3 * pa
        + (-1.79330391e-04) * ta * va3 * pa
        + (2.34994441e-06) * ta2 * va3 * pa
        + (1.29735808e-04) * va4 * pa
        + (1.29064870e-06) * ta * va4 * pa
        + (-2.28558686e-06) * va5 * pa
        + (-3.69476348e-02) * d_tmrt * pa
        + (1.62325322e-03) * ta * d_tmrt * pa
        + (-3.14279680e-05) * ta2 * d_tmrt * pa
        + (2.59835559e-06) * ta3 * d_tmrt * pa
        + (-4.77136523e-08) * ta4 * d_tmrt * pa
        + (8.64203390e-03) * va * d_tmrt * pa
        + (-6.87405181e-04) * ta * va * d_tmrt * pa
        + (-9.13863872e-06) * ta2 * va * d_tmrt * pa
        + (5.15916806e-07) * ta3 * va * d_tmrt * pa
        + (-3.59217476e-05) * va2 * d_tmrt * pa
        + (3.28696511e-05) * ta * va2 * d_tmrt * pa
        + (-7.10542454e-07) * ta2 * va2 * d_tmrt * pa
        + (-1.24382300e-05) * va3 * d_tmrt * pa
        + (-7.38584400e-09) * ta * va3 * d_tmrt * pa
        + (2.20609296e-07) * va4 * d_tmrt * pa
        + (-7.32469180e-04) * d2 * pa
        + (-1.87381964e-05) * ta * d2 * pa
        + (4.80925239e-06) * ta2 * d2 * pa
        + (-8.75492040e-08) * ta3 * d2 * pa
        + (2.77862930e-05) * va * d2 * pa
        + (-5.06004592e-06) * ta * va * d2 * pa
        + (1.14325367e-07) * ta2 * va * d2 * pa
        + (2.53016723e-06) * va2 * d2 * pa
        + (-1.72857035e-08) * ta * va2 * d2 * pa
        + (-3.95079398e-08) * va3 * d2 * pa
        + (-3.59413173e-07) * d3 * pa
        + (7.04388046e-07) * ta * d3 * pa
        + (-1.89309167e-08) * ta2 * d3 * pa
        + (-4.79768731e-07) * va * d3 * pa
        + (7.96079978e-09) * ta * va * d3 * pa
        + (1.62897058e-09) * va2 * d3 * pa
        + (3.94367674e-08) * d4 * pa
        + (-1.18566247e-09) * ta * d4 * pa
        + (3.34678041e-10) * va * d4 * pa
        + (-1.15606447e-10) * d5 * pa
        + (-2.80626406e00) * pa2
        + (5.48712484e-01) * ta * pa2
        + (-3.99428410e-03) * ta2 * pa2
        + (-9.54009191e-04) * ta3 * pa2
        + (1.93090978e-05) * ta4 * pa2
        + (-3.08806365e-01) * va * pa2
        + (1.16952364e-02) * ta * va * pa2
        + (4.95271903e-04) * ta2 * va * pa2
        + (-1.90710882e-05) * ta3 * va * pa2
        + (2.10787756e-03) * va2 * pa2
        + (-6.98445738e-04) * ta * va2 * pa2
        + (2.30109073e-05) * ta2 * va2 * pa2
        + (4.17856590e-04) * va3 * pa2
        + (-1.27043871e-05) * ta * va3 * pa2
        + (-3.04620472e-06) * va4 * pa2
        + (5.14507424e-02) * d_tmrt * pa2
        + (-4.32510997e-03) * ta * d_tmrt * pa2
        + (8.99281156e-05) * ta2 * d_tmrt * pa2
        + (-7.14663943e-07) * ta3 * d_tmrt * pa2
        + (-2.66016305e-04) * va * d_tmrt * pa2
        + (2.63789586e-04) * ta * va * d_tmrt * pa2
        + (-7.01199003e-06) * ta2 * va * d_tmrt * pa2
        + (-1.06823306e-04) * va2 * d_tmrt * pa2
        + (3.61341136e-06) * ta * va2 * d_tmrt * pa2
        + (2.29748967e-07) * va3 * d_tmrt * pa2
        + (3.04788893e-04) * d2 * pa2
        + (-6.42070836e-05) * ta * d2 * pa2
        + (1.16257971e-06) * ta2 * d2 * pa2
        + (7.68023384e-06) * va * d2 * pa2
        + (-5.47446896e-07) * ta * va * d2 * pa2
        + (-3.59937910e-08) * va2 * d2 * pa2
        + (-4.36497725e-06) * d3 * pa2
        + (1.68737969e-07) * ta * d3 * pa2
        + (2.67489271e-08) * va * d3 * pa2
        + (3.23926897e-09) * d4 * pa2
        + (-3.53874123e-02) * pa3
        + (-2.21201190e-01) * ta * pa3
        + (1.55126038e-02) * ta2 * pa3
        + (-2.63917279e-04) * ta3 * pa3
        + (4.53433455e-02) * va * pa3
        + (-4.32943862e-03) * ta * va * pa3
        + (1.45389826e-04) * ta2 * va * pa3
        + (2.17508610e-04) * va2 * pa3
        + (-6.66724702e-05) * ta * va2 * pa3
        + (3.33217140e-05) * va3 * pa3
        + (-2.26921615e-03) * d_tmrt * pa3
        + (3.80261982e-04) * ta * d_tmrt * pa3
        + (-5.45314314e-09) * ta2 * d_tmrt * pa3
        + (-7.96355448e-04) * va * d_tmrt * pa3
        + (2.53458034e-05) * ta * va * d_tmrt * pa3
        + (-6.31223658e-06) * va2 * d_tmrt * pa3
        + (3.02122035e-04) * d2 * pa3
        + (-4.77403547e-06) * ta * d2 * pa3
        + (1.73825715e-06) * va * d2 * pa3
        + (-4.09087898e-07) * d3 * pa3
        + (6.14155345e-01) * pa4
        + (-6.16755931e-02) * ta * pa4
        + (1.33374846e-03) * ta2 * pa4
        + (3.55375387e-03) * va * pa4
        + (-5.13027851e-04) * ta * va * pa4
        + (1.02449757e-04) * va2 * pa4
        + (-1.48526421e-03) * d_tmrt * pa4
        + (-4.11469183e-05) * ta * d_tmrt * pa4
        + (-6.80434415e-06) * va * d_tmrt * pa4
        + (-9.77675906e-06) * d2 * pa4
        + (8.82773108e-02) * pa5
        + (-3.01859306e-03) * ta * pa5
        + (1.04452989e-03) * va * pa5
        + (2.47090539e-04) * d_tmrt * pa5
        + (1.48348065e-03) * pa6
}

/// Calculate saturation vapor pressure using the polynomial from UTCI.
#[inline]
fn saturation_vapor_pressure(ta: f32) -> f32 {
    const G: [f32; 8] = [
        -2.8365744e3,
        -6.028076559e3,
        1.954263612e1,
        -2.737830188e-2,
        1.6261698e-5,
        7.0229056e-10,
        -1.8680009e-13,
        2.7150305,
    ];

    let tk = ta + 273.15;
    let mut es = G[7] * tk.ln();

    // Compute tk^(-2), tk^(-1), tk^0, tk^1, ..., tk^4
    let tk_inv2 = 1.0 / (tk * tk);
    let tk_inv = 1.0 / tk;

    es += G[0] * tk_inv2;
    es += G[1] * tk_inv;
    es += G[2];
    es += G[3] * tk;
    es += G[4] * tk * tk;
    es += G[5] * tk * tk * tk;
    es += G[6] * tk * tk * tk * tk;

    (es.exp()) * 0.01
}

/// Calculate UTCI for a single point.
///
/// Parameters:
/// - ta: Air temperature (°C)
/// - rh: Relative humidity (%)
/// - tmrt: Mean radiant temperature (°C)
/// - va10m: Wind speed at 10m height (m/s)
///
/// Returns: UTCI temperature (°C) or -999 for invalid inputs
#[pyfunction]
pub fn utci_single(ta: f32, rh: f32, tmrt: f32, va10m: f32) -> f32 {
    if ta <= -999.0 || rh <= -999.0 || va10m <= -999.0 || tmrt <= -999.0 {
        return -999.0;
    }

    let es = saturation_vapor_pressure(ta);
    let eh_pa = es * rh / 100.0;
    let pa = eh_pa / 10.0; // vapor pressure in kPa
    let d_tmrt = tmrt - ta;

    utci_polynomial(d_tmrt, ta, va10m, pa)
}

/// Calculate UTCI for a 2D grid using parallel processing.
///
/// Parameters:
/// - ta: Air temperature (°C) - scalar
/// - rh: Relative humidity (%) - scalar
/// - tmrt: Mean radiant temperature grid (°C)
/// - va10m: Wind speed grid at 10m height (m/s)
///
/// Returns: UTCI temperature grid (°C), -9999 for invalid pixels
#[pyfunction]
pub fn utci_grid<'py>(
    py: Python<'py>,
    ta: f32,
    rh: f32,
    tmrt: PyReadonlyArray2<'py, f32>,
    va10m: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let tmrt_arr = tmrt.as_array();
    let va_arr = va10m.as_array();

    let (rows, cols) = tmrt_arr.dim();

    // Handle invalid scalar inputs
    if ta <= -999.0 || rh <= -999.0 {
        let result = ndarray::Array2::from_elem((rows, cols), -999.0f32);
        return Ok(PyArray2::from_owned_array(py, result));
    }

    // Pre-compute vapor pressure (constant for entire grid)
    let es = saturation_vapor_pressure(ta);
    let eh_pa = es * rh / 100.0;
    let pa = eh_pa / 10.0;

    // Create output array
    let mut result = ndarray::Array2::zeros((rows, cols));

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
            if !tmrt_val.is_finite() || !va_val.is_finite() {
                *out = f32::NAN;
            } else if tmrt_val <= -999.0 || va_val <= -999.0 {
                *out = f32::NAN;
            } else if va_val > 0.0 {
                let d_tmrt = tmrt_val - ta;
                *out = utci_polynomial(d_tmrt, ta, va_val, pa);
            } else {
                *out = f32::NAN; // Invalid wind speed
            }
        });

    Ok(PyArray2::from_owned_array(py, result))
}
