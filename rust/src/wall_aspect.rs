//! Wall aspect (orientation) detection using the Goodwin filter algorithm.
//!
//! Determines wall orientation from a binary wall grid and DSM by rotating
//! a linear filter through 180 angles and finding the best alignment.
//!
//! References:
//! - Goodwin NR, Coops NC, Tooke TR, Christen A, Voogt JA (2009)
//! - Lindberg F., Jonsson, P. & Honjo, T. and Wästberg, D. (2015b)

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use ndarray::{Array2, ArrayView2, Zip};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Rotate a small 2D array by `angle` degrees (counter-clockwise).
///
/// Supports order=0 (nearest neighbor) and order=1 (bilinear).
/// Uses inverse mapping: for each output pixel, find the source coordinate.
fn rotate_2d(arr: &Array2<f32>, angle_deg: f32, order: u8) -> Array2<f32> {
    // Math is done in f64 to match scipy.ndimage.rotate's f64 accumulation
    // precision. UMEP uses scipy, so aligning numerically requires the same
    // precision budget. Input and output stay f32 (we don't contaminate the
    // caller's data arrays — this promotion is strictly internal).
    let (rows, cols) = arr.dim();
    let center_y = (rows as f64 - 1.0) / 2.0;
    let center_x = (cols as f64 - 1.0) / 2.0;

    let theta = (angle_deg as f64) * std::f64::consts::PI / 180.0;
    let cos_t = theta.cos();
    let sin_t = theta.sin();

    let mut output = Array2::<f32>::zeros((rows, cols));

    for r in 0..rows {
        for c in 0..cols {
            let xc = c as f64 - center_x;
            let yc = r as f64 - center_y;

            // Inverse rotation to find source coordinates
            let src_x = cos_t * xc - sin_t * yc + center_x;
            let src_y = sin_t * xc + cos_t * yc + center_y;

            if order == 0 {
                // Nearest neighbor — use banker's rounding (round ties to even) so
                // that source-coordinate selection matches numpy's ``np.round``
                // convention used by the UMEP reference. Rust's default ``round()``
                // rounds half-away-from-zero, which flips tie decisions vs UMEP.
                let sx = src_x.round_ties_even() as i32;
                let sy = src_y.round_ties_even() as i32;
                let sx = sx.clamp(0, cols as i32 - 1) as usize;
                let sy = sy.clamp(0, rows as i32 - 1) as usize;
                output[[r, c]] = arr[[sy, sx]];
            } else {
                // Bilinear interpolation in f64
                let x0 = src_x.floor() as i32;
                let y0 = src_y.floor() as i32;
                let x1 = x0 + 1;
                let y1 = y0 + 1;

                let x0c = x0.clamp(0, cols as i32 - 1) as usize;
                let x1c = x1.clamp(0, cols as i32 - 1) as usize;
                let y0c = y0.clamp(0, rows as i32 - 1) as usize;
                let y1c = y1.clamp(0, rows as i32 - 1) as usize;

                let wx = (src_x - x0 as f64).clamp(0.0, 1.0);
                let wy = (src_y - y0 as f64).clamp(0.0, 1.0);

                let v = arr[[y0c, x0c]] as f64 * (1.0 - wx) * (1.0 - wy)
                    + arr[[y0c, x1c]] as f64 * wx * (1.0 - wy)
                    + arr[[y1c, x0c]] as f64 * (1.0 - wx) * wy
                    + arr[[y1c, x1c]] as f64 * wx * wy;
                output[[r, c]] = v as f32;
            }
        }
    }

    output
}

/// Precompute all 180 rotated filter pairs.
fn precompute_filters(
    filtersize: usize,
    half_ceil: usize,
    half_floor: usize,
) -> Vec<(Array2<f32>, Array2<f32>)> {
    let mut filtmatrix = Array2::<f32>::zeros((filtersize, filtersize));
    let mut buildfilt = Array2::<f32>::zeros((filtersize, filtersize));

    // filtmatrix: vertical center column = 1
    for r in 0..filtersize {
        filtmatrix[[r, half_ceil - 1]] = 1.0;
    }

    let n = filtersize - 1;

    // buildfilt: center row, left half = 1, right half = 2
    for c in 0..half_floor {
        buildfilt[[half_ceil - 1, c]] = 1.0;
    }
    for c in half_ceil..filtersize {
        buildfilt[[half_ceil - 1, c]] = 2.0;
    }

    (0..180)
        .map(|h| {
            let mut fm = rotate_2d(&filtmatrix, h as f32, 1); // bilinear
            // Banker's rounding (round ties to even) matches numpy's ``np.round``
            // used by UMEP. Rust's default ``round()`` rounds half-away-from-zero,
            // which at bilinear tie values (≈ 0.5) produces a different filter
            // template and shifts the Goodwin "best h" search by ~90° at some
            // wall pixels.
            fm.mapv_inplace(|v| v.round_ties_even());

            let mut bf = rotate_2d(&buildfilt, h as f32, 0); // nearest
            bf.mapv_inplace(|v| v.round_ties_even());

            let index = 270.0 - h as f32;

            // Special-case corrections matching original Python
            if h == 150 || h == 30 {
                for r in 0..filtersize {
                    bf[[r, n]] = 0.0;
                }
            }
            if index == 225.0 {
                fm[[0, 0]] = 1.0;
                fm[[n, n]] = 1.0;
            }
            if index == 135.0 {
                fm[[0, n]] = 1.0;
                fm[[n, 0]] = 1.0;
            }

            (fm, bf)
        })
        .collect()
}

/// Compute wall aspect using the Goodwin filter algorithm.
///
/// Parallelized across wall pixels using Rayon. Each wall pixel independently
/// tests all 180 filter angles to find the best alignment.
pub(crate) fn compute_wall_aspect_pure(
    walls_in: ArrayView2<f32>,
    scale: f32,
    dsm: ArrayView2<f32>,
    progress_counter: Option<Arc<AtomicUsize>>,
    cancel_flag: Option<Arc<AtomicBool>>,
) -> Result<Array2<f32>, &'static str> {
    let (rows, cols) = walls_in.dim();

    // Binarize walls
    let walls = walls_in.mapv(|v| if v > 0.5 { 1.0 } else { v });

    // Compute filter size from scale
    let filtersize_f = (scale + 1e-10) * 9.0;
    let mut filtersize = filtersize_f.floor() as usize;
    if filtersize <= 2 {
        filtersize = 3;
    } else if filtersize != 9 && filtersize % 2 == 0 {
        filtersize += 1;
    }

    let half_ceil = ((filtersize as f32) / 2.0).ceil() as usize;
    let half_floor = ((filtersize as f32) / 2.0).floor() as usize;

    // Precompute all 180 rotated filter pairs (fast, filters are tiny ~9x9)
    let filters = precompute_filters(filtersize, half_ceil, half_floor);

    // Iteration bounds (stay within filter radius of edges)
    let i_start = half_ceil - 1;
    let i_end = rows.saturating_sub(half_ceil + 1);
    let j_start = half_ceil - 1;
    let j_end = cols.saturating_sub(half_ceil + 1);

    // Collect wall pixel coordinates for parallel processing
    let walls_view = walls.view();
    let wall_pixels: Vec<(usize, usize)> = (i_start..i_end)
        .flat_map(|i| {
            (j_start..j_end).filter_map(move |j| {
                if walls_view[[i, j]] >= 0.5 {
                    Some((i, j))
                } else {
                    None
                }
            })
        })
        .collect();

    let total_pixels = wall_pixels.len();

    // Reset progress
    if let Some(ref counter) = progress_counter {
        counter.store(0, Ordering::Relaxed);
    }

    // For each wall pixel, find the best angle across all 180 rotations.
    // Returns (row, col, best_direction, building_side).
    let processed = AtomicUsize::new(0);
    let walls_ref = &walls;
    let dsm_ref = &dsm;
    let filters_ref = &filters;
    let progress_ref = &progress_counter;
    let cancel_ref = &cancel_flag;

    let results: Vec<(usize, usize, f32, f32)> = wall_pixels
        .par_iter()
        .map(|&(i, j)| {
            // Check cancellation early (skip remaining work)
            if let Some(ref flag) = cancel_ref {
                if flag.load(Ordering::Relaxed) {
                    return (i, j, 0.0, 0.0);
                }
            }

            // Accumulators in f64 to match numpy's default f64 summation
            // precision (UMEP uses ``np.sum`` under the hood). Filter and
            // walls/dsm arrays stay f32 — only the summation runs in f64.
            let mut best_sum = 0.0f64;
            let mut best_side = 0.0f32;
            let mut best_dir = 0.0f32;

            for (h, (fm, bf)) in filters_ref.iter().enumerate() {
                let index = 270.0 - h as f32;

                // Weighted sum of wall neighbors along the rotated filter line
                let mut wallscut_sum = 0.0f64;
                for di in 0..filtersize {
                    for dj in 0..filtersize {
                        wallscut_sum += (walls_ref[[i - half_floor + di, j - half_floor + dj]]
                            as f64)
                            * (fm[[di, dj]] as f64);
                    }
                }

                if wallscut_sum > best_sum {
                    best_sum = wallscut_sum;

                    // Determine which side of the wall is the building
                    let mut sum_side1 = 0.0f64;
                    let mut sum_side2 = 0.0f64;
                    for di in 0..filtersize {
                        for dj in 0..filtersize {
                            let dsm_val =
                                dsm_ref[[i - half_floor + di, j - half_floor + dj]] as f64;
                            let bf_val = bf[[di, dj]];
                            if bf_val == 1.0 {
                                sum_side1 += dsm_val;
                            } else if bf_val == 2.0 {
                                sum_side2 += dsm_val;
                            }
                        }
                    }

                    best_side = if sum_side1 > sum_side2 { 1.0 } else { 2.0 };
                    best_dir = index;
                }
            }

            // Update progress (map pixel count to 0..180 range)
            let count = processed.fetch_add(1, Ordering::Relaxed) + 1;
            if let Some(ref counter) = progress_ref {
                if total_pixels > 0 {
                    let pct = ((count as u64 * 180) / total_pixels as u64) as usize;
                    counter.store(pct.min(180), Ordering::Relaxed);
                }
            }

            (i, j, best_dir, best_side)
        })
        .collect();

    // Check cancellation after parallel work completes
    if let Some(ref flag) = cancel_flag {
        if flag.load(Ordering::Relaxed) {
            return Err("Wall aspect computation cancelled");
        }
    }

    // Scatter results into output arrays
    let mut y = Array2::<f32>::zeros((rows, cols));
    let mut x = Array2::<f32>::zeros((rows, cols));

    for &(i, j, dir, side) in &results {
        y[[i, j]] = dir;
        x[[i, j]] = side;
    }

    // Post-processing: adjust angles based on building side
    Zip::from(&mut y).and(&x).for_each(|y_val, &x_val| {
        if x_val == 1.0 {
            *y_val -= 180.0;
        }
    });
    y.mapv_inplace(|v| if v < 0.0 { v + 360.0 } else { v });

    // DSM gradient fallback for walls with direction 0
    let dx = 1.0 / scale;
    let asp = compute_dsm_aspect(&dsm, dx);

    Zip::from(&mut y)
        .and(&walls)
        .and(&asp)
        .for_each(|y_val, &w, &a| {
            if w >= 0.5 && *y_val == 0.0 {
                *y_val = a / (std::f32::consts::PI / 180.0);
            }
        });

    // Final progress
    if let Some(ref counter) = progress_counter {
        counter.store(180, Ordering::Relaxed);
    }

    Ok(y)
}

/// Compute DSM aspect (orientation of slope) using numpy.gradient equivalent.
///
/// Returns aspect in radians matching the Python `get_ders` function.
///
/// Gradient and atan2 are computed in f64 to match numpy's default f64
/// precision for ``np.gradient`` and ``np.arctan2`` (UMEP path). The input
/// dsm view remains f32 and the returned aspect array is f32; promotion
/// is strictly internal so caller-visible array memory is unchanged.
fn compute_dsm_aspect(dsm: &ArrayView2<f32>, dx: f32) -> Array2<f32> {
    let (rows, cols) = dsm.dim();
    let mut fy = Array2::<f64>::zeros((rows, cols));
    let mut fx = Array2::<f64>::zeros((rows, cols));
    let dx = dx as f64;

    // Compute gradients (matching numpy.gradient behavior in f64)
    for i in 0..rows {
        for j in 0..cols {
            // fy: gradient along axis 0 (rows)
            fy[[i, j]] = if i == 0 {
                (dsm[[1, j]] as f64 - dsm[[0, j]] as f64) / dx
            } else if i == rows - 1 {
                (dsm[[rows - 1, j]] as f64 - dsm[[rows - 2, j]] as f64) / dx
            } else {
                (dsm[[i + 1, j]] as f64 - dsm[[i - 1, j]] as f64) / (2.0 * dx)
            };

            // fx: gradient along axis 1 (cols)
            fx[[i, j]] = if j == 0 {
                (dsm[[i, 1]] as f64 - dsm[[i, 0]] as f64) / dx
            } else if j == cols - 1 {
                (dsm[[i, cols - 1]] as f64 - dsm[[i, cols - 2]] as f64) / dx
            } else {
                (dsm[[i, j + 1]] as f64 - dsm[[i, j - 1]] as f64) / (2.0 * dx)
            };
        }
    }

    // cart2pol: theta = atan2(fx, fy), then negate, then wrap to [0, 2pi)
    // Matching Python: asp = atan2(fx, fy) * -1, then wrap negatives
    let mut asp = Array2::<f32>::zeros((rows, cols));
    Zip::from(&mut asp)
        .and(&fy)
        .and(&fx)
        .for_each(|a, &fy_val, &fx_val| {
            let mut theta = fy_val.atan2(fx_val);
            theta = -theta;
            if theta < 0.0 {
                theta += 2.0 * std::f64::consts::PI;
            }
            *a = theta as f32;
        });

    asp
}

/// PyO3 wrapper for wall aspect computation (no progress reporting).
#[pyfunction]
pub fn compute_wall_aspect(
    py: Python<'_>,
    walls: PyReadonlyArray2<f32>,
    scale: f32,
    dsm: PyReadonlyArray2<f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    let walls_view = walls.as_array();
    let dsm_view = dsm.as_array();

    let result = compute_wall_aspect_pure(walls_view, scale, dsm_view, None, None)
        .map_err(|e| pyo3::exceptions::PyInterruptedError::new_err(e))?;
    Ok(result.into_pyarray(py).unbind())
}

/// Runner that exposes pollable progress() and cancel() methods for wall aspect computation.
///
/// Usage from Python:
///   runner = WallAspectRunner()
///   # launch runner.compute(...) in a thread
///   # poll runner.progress() from main thread (returns 0..180)
///   # call runner.cancel() to request early termination
#[pyclass]
pub struct WallAspectRunner {
    progress: Arc<AtomicUsize>,
    cancelled: Arc<AtomicBool>,
}

impl Default for WallAspectRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl WallAspectRunner {
    #[new]
    pub fn new() -> Self {
        Self {
            progress: Arc::new(AtomicUsize::new(0)),
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Returns progress mapped to 0..180 range.
    pub fn progress(&self) -> usize {
        self.progress.load(Ordering::Relaxed)
    }

    /// Request cancellation of the running computation.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    /// Compute wall aspect, releasing the GIL so progress()/cancel() can be called.
    pub fn compute(
        &self,
        py: Python<'_>,
        walls: PyReadonlyArray2<f32>,
        scale: f32,
        dsm: PyReadonlyArray2<f32>,
    ) -> PyResult<Py<PyArray2<f32>>> {
        // Reset progress and cancel flag
        self.progress.store(0, Ordering::Relaxed);
        self.cancelled.store(false, Ordering::Relaxed);

        // Copy to owned arrays so we can release the GIL
        let walls_owned = walls.as_array().to_owned();
        let dsm_owned = dsm.as_array().to_owned();
        let counter = Some(self.progress.clone());
        let cancel = Some(self.cancelled.clone());

        let result = py.allow_threads(|| {
            compute_wall_aspect_pure(walls_owned.view(), scale, dsm_owned.view(), counter, cancel)
        });

        match result {
            Ok(arr) => Ok(arr.into_pyarray(py).unbind()),
            Err(msg) => Err(pyo3::exceptions::PyInterruptedError::new_err(msg)),
        }
    }
}
