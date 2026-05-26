use ndarray::{Array2, ArrayBase, ArrayView2, Data, DataMut, Ix2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

// Convenience accessors that document the invariant once instead of in every
// call site. All Array2<f32>s and views in this module are either allocated
// via Array2::zeros (standard C-order contiguous) or are views over PyO3
// numpy arrays that we have verified to be contiguous at the FFI boundary.
// The .expect() messages are diagnostic in case that invariant is ever
// violated by a future refactor.
const VEG_CONTIG_INVARIANT: &str =
    "vegetation.rs invariant: all f32 arrays here are contiguous (Array2::zeros or contiguous numpy views)";

#[inline(always)]
fn as_slice_checked<D>(arr: &ArrayBase<D, Ix2>) -> &[f32]
where
    D: Data<Elem = f32>,
{
    arr.as_slice().expect(VEG_CONTIG_INVARIANT)
}

#[inline(always)]
fn as_slice_mut_checked<D>(arr: &mut ArrayBase<D, Ix2>) -> &mut [f32]
where
    D: DataMut<Elem = f32>,
{
    arr.as_slice_mut().expect(VEG_CONTIG_INVARIANT)
}

/// Pure result type for lside_veg (no PyO3 dependency).
pub(crate) struct LsideVegPureResult {
    pub least: Array2<f32>,
    pub lsouth: Array2<f32>,
    pub lwest: Array2<f32>,
    pub lnorth: Array2<f32>,
}

/// Pure-ndarray implementation of Lside_veg_v2022a.
/// Callable from pipeline.rs (fused path) or from the PyO3 wrapper (modular path).
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn lside_veg_pure(
    svfS: ArrayView2<f32>,
    svfW: ArrayView2<f32>,
    svfN: ArrayView2<f32>,
    svfE: ArrayView2<f32>,
    svfEveg: ArrayView2<f32>,
    svfSveg: ArrayView2<f32>,
    svfWveg: ArrayView2<f32>,
    svfNveg: ArrayView2<f32>,
    svfEaveg: ArrayView2<f32>,
    svfSaveg: ArrayView2<f32>,
    svfWaveg: ArrayView2<f32>,
    svfNaveg: ArrayView2<f32>,
    azimuth: f32,
    altitude: f32,
    Ta: f32,
    Tw: f32,
    SBC: f32,
    ewall: f32,
    Ldown: ArrayView2<f32>,
    esky: f32,
    t: f32,
    F_sh: ArrayView2<f32>,
    CI: f32,
    LupE: ArrayView2<f32>,
    LupS: ArrayView2<f32>,
    LupW: ArrayView2<f32>,
    LupN: ArrayView2<f32>,
    anisotropic_longwave: bool,
    valid: Option<ArrayView2<u8>>,
) -> LsideVegPureResult {
    let shape = svfE.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let vikttot: f32 = 4.4897;
    let TaK = Ta + 273.15;
    let TaK_pow4 = TaK.powi(4);
    let c = 1.0 - CI;
    let Lsky_allsky = esky * SBC * TaK_pow4 * (1.0 - c) + c * SBC * TaK_pow4;
    let altitude_day = altitude > 0.0;

    let sun_east = azimuth > (180.0 - t) && azimuth <= (360.0 - t);
    let sun_south = azimuth <= (90.0 - t) || azimuth > (270.0 - t);
    let sun_west = azimuth > (360.0 - t) || azimuth <= (180.0 - t);
    let sun_north = azimuth > (90.0 - t) && azimuth <= (270.0 - t);

    let temp_e = TaK + Tw * ((azimuth - 180.0 + t) * std::f32::consts::PI / 180.0).sin();
    let temp_s = TaK + Tw * ((azimuth - 270.0 + t) * std::f32::consts::PI / 180.0).sin();
    let temp_w = TaK + Tw * ((azimuth + t) * std::f32::consts::PI / 180.0).sin();
    let temp_n = TaK + Tw * ((azimuth - 90.0 + t) * std::f32::consts::PI / 180.0).sin();

    #[inline]
    fn poly(x: f32) -> f32 {
        63.227 * x.powi(6) - 161.51 * x.powi(5) + 156.91 * x.powi(4) - 70.424 * x.powi(3)
            + 16.773 * x.powi(2)
            - 0.4863 * x
    }

    let npix = rows * cols;
    let mut least_vec = vec![0.0f32; npix];
    let mut lsouth_vec = vec![0.0f32; npix];
    let mut lwest_vec = vec![0.0f32; npix];
    let mut lnorth_vec = vec![0.0f32; npix];

    least_vec
        .par_iter_mut()
        .zip(lsouth_vec.par_iter_mut())
        .zip(lwest_vec.par_iter_mut())
        .zip(lnorth_vec.par_iter_mut())
        .enumerate()
        .for_each(|(idx, (((least, lsouth), lwest), lnorth))| {
            let r = idx / cols;
            let c = idx % cols;
            if let Some(ref v) = valid {
                if v[[r, c]] == 0 {
                    *least = f32::NAN;
                    *lsouth = f32::NAN;
                    *lwest = f32::NAN;
                    *lnorth = f32::NAN;
                    return;
                }
            }
            let compute = |svf: f32,
                           svfveg: f32,
                           svfaveg: f32,
                           lup: f32,
                           sun_cond: bool,
                           temp_wall: f32|
             -> f32 {
                if anisotropic_longwave {
                    return lup * 0.5;
                }
                let svfalfa = (1.0 - svf).clamp(0.0, 1.0).sqrt().asin();
                let poly_svf_val = poly(svf);
                let poly_svfaveg_val = poly(svfaveg);
                let viktonlywall = (vikttot - poly_svf_val) / vikttot;
                let viktaveg = (vikttot - poly_svfaveg_val) / vikttot;
                let viktwall = viktonlywall - viktaveg;
                let svfvegbu = svfveg + svf - 1.0;
                let poly_svfvegbu = poly(svfvegbu);
                let viktsky = poly_svfvegbu / vikttot;
                let viktrefl = (vikttot - poly_svfvegbu) / vikttot;
                let viktveg = viktrefl - viktwall;
                let fsh_val = F_sh[(r, c)];
                let ldown_val = Ldown[(r, c)];
                let f_sh_scaled = 2.0 * fsh_val - 1.0;
                let (lwallsun, lwallsh) = if altitude_day {
                    let alfa_b = svfalfa.atan();
                    let beta_b = (svfalfa * f_sh_scaled).tan().atan();
                    let betasun = ((alfa_b - beta_b) / 2.0) + beta_b;
                    if sun_cond {
                        let lwallsun = SBC
                            * ewall
                            * temp_wall.powi(4)
                            * viktwall
                            * (1.0 - f_sh_scaled)
                            * betasun.cos()
                            * 0.5;
                        let lwallsh = SBC * ewall * TaK_pow4 * viktwall * f_sh_scaled * 0.5;
                        (lwallsun, lwallsh)
                    } else {
                        let lwallsun = 0.0;
                        let lwallsh = SBC * ewall * TaK_pow4 * viktwall * 0.5;
                        (lwallsun, lwallsh)
                    }
                } else {
                    (0.0, SBC * ewall * TaK_pow4 * viktwall * 0.5)
                };
                let lsky = ((svf + svfveg - 1.0) * Lsky_allsky) * viktsky * 0.5;
                let lveg = SBC * ewall * TaK_pow4 * viktveg * 0.5;
                let lground = lup * 0.5;
                let lrefl = (ldown_val + lup) * viktrefl * (1.0 - ewall) * 0.5;
                lsky + lwallsun + lwallsh + lveg + lground + lrefl
            };
            *least = compute(
                svfE[(r, c)],
                svfEveg[(r, c)],
                svfEaveg[(r, c)],
                LupE[(r, c)],
                sun_east,
                temp_e,
            );
            *lsouth = compute(
                svfS[(r, c)],
                svfSveg[(r, c)],
                svfSaveg[(r, c)],
                LupS[(r, c)],
                sun_south,
                temp_s,
            );
            *lwest = compute(
                svfW[(r, c)],
                svfWveg[(r, c)],
                svfWaveg[(r, c)],
                LupW[(r, c)],
                sun_west,
                temp_w,
            );
            *lnorth = compute(
                svfN[(r, c)],
                svfNveg[(r, c)],
                svfNaveg[(r, c)],
                LupN[(r, c)],
                sun_north,
                temp_n,
            );
        });

    // INVARIANT: each *_vec was filled by Zip::indexed over the (rows, cols) shape
    // above, so len == rows * cols. `from_shape_vec` cannot fail here; the
    // debug_assert pins the invariant if the loop above is ever changed.
    debug_assert_eq!(least_vec.len(), rows * cols);
    debug_assert_eq!(lsouth_vec.len(), rows * cols);
    debug_assert_eq!(lwest_vec.len(), rows * cols);
    debug_assert_eq!(lnorth_vec.len(), rows * cols);
    LsideVegPureResult {
        least: Array2::from_shape_vec((rows, cols), least_vec)
            .expect("least_vec length mismatches (rows, cols)"),
        lsouth: Array2::from_shape_vec((rows, cols), lsouth_vec)
            .expect("lsouth_vec length mismatches (rows, cols)"),
        lwest: Array2::from_shape_vec((rows, cols), lwest_vec)
            .expect("lwest_vec length mismatches (rows, cols)"),
        lnorth: Array2::from_shape_vec((rows, cols), lnorth_vec)
            .expect("lnorth_vec length mismatches (rows, cols)"),
    }
}

/// Pure result type for kside_veg isotropic path (no PyO3 dependency).
pub(crate) struct KsideVegPureResult {
    pub keast: Array2<f32>,
    pub ksouth: Array2<f32>,
    pub kwest: Array2<f32>,
    pub knorth: Array2<f32>,
    pub kside_i: Array2<f32>,
}

/// Pure-ndarray implementation of Kside_veg_v2022a (isotropic path).
/// The anisotropic shortwave uses `anisotropic_sky_pure()` in the fused pipeline.
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn kside_veg_isotropic_pure(
    radI: f32,
    radD: f32,
    radG: f32,
    shadow: ArrayView2<f32>,
    svfS: ArrayView2<f32>,
    svfW: ArrayView2<f32>,
    svfN: ArrayView2<f32>,
    svfE: ArrayView2<f32>,
    svfEveg: ArrayView2<f32>,
    svfSveg: ArrayView2<f32>,
    svfWveg: ArrayView2<f32>,
    svfNveg: ArrayView2<f32>,
    azimuth: f32,
    altitude: f32,
    psi: f32,
    t: f32,
    albedo: f32,
    F_sh: ArrayView2<f32>,
    KupE: ArrayView2<f32>,
    KupS: ArrayView2<f32>,
    KupW: ArrayView2<f32>,
    KupN: ArrayView2<f32>,
    cyl: bool,
    valid: Option<ArrayView2<u8>>,
) -> KsideVegPureResult {
    let shape = svfE.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let deg2rad = std::f32::consts::PI / 180.0;
    let vikttot = 4.4897f32;

    let mut Keast = Array2::<f32>::zeros((rows, cols));
    let mut Ksouth = Array2::<f32>::zeros((rows, cols));
    let mut Kwest = Array2::<f32>::zeros((rows, cols));
    let mut Knorth = Array2::<f32>::zeros((rows, cols));
    let mut KsideI = Array2::<f32>::zeros((rows, cols));

    #[inline]
    fn kvikt_veg(svf: f32, svfveg: f32, vikttot: f32) -> (f32, f32) {
        let poly = |x: f32| -> f32 {
            63.227 * x.powi(6) - 161.51 * x.powi(5) + 156.91 * x.powi(4) - 70.424 * x.powi(3)
                + 16.773 * x.powi(2)
                - 0.4863 * x
        };
        let viktwall = (vikttot - poly(svf)) / vikttot;
        let svfvegbu = svfveg + svf - 1.0;
        let viktveg_tot = (vikttot - poly(svfvegbu)) / vikttot;
        let viktveg = viktveg_tot - viktwall;
        (viktveg, viktwall)
    }

    // Precompute svfviktbuveg arrays
    let mut svfviktbuvegE = Array2::<f32>::zeros((rows, cols));
    let mut svfviktbuvegS = Array2::<f32>::zeros((rows, cols));
    let mut svfviktbuvegW = Array2::<f32>::zeros((rows, cols));
    let mut svfviktbuvegN = Array2::<f32>::zeros((rows, cols));

    for r in 0..rows {
        for c in 0..cols {
            if let Some(ref v) = valid {
                if v[[r, c]] == 0 {
                    continue;
                }
            }
            let (vveg, vwall) = kvikt_veg(svfE[(r, c)], svfEveg[(r, c)], vikttot);
            svfviktbuvegE[(r, c)] = vwall + vveg * (1.0 - psi);
            let (vveg, vwall) = kvikt_veg(svfS[(r, c)], svfSveg[(r, c)], vikttot);
            svfviktbuvegS[(r, c)] = vwall + vveg * (1.0 - psi);
            let (vveg, vwall) = kvikt_veg(svfW[(r, c)], svfWveg[(r, c)], vikttot);
            svfviktbuvegW[(r, c)] = vwall + vveg * (1.0 - psi);
            let (vveg, vwall) = kvikt_veg(svfN[(r, c)], svfNveg[(r, c)], vikttot);
            svfviktbuvegN[(r, c)] = vwall + vveg * (1.0 - psi);
        }
    }

    // Direct radiation
    if cyl {
        for r in 0..rows {
            for c in 0..cols {
                KsideI[(r, c)] = shadow[(r, c)] * radI * (altitude * deg2rad).cos();
            }
        }
    } else {
        for r in 0..rows {
            for c in 0..cols {
                let sh_val = shadow[(r, c)];
                if azimuth > (360.0 - t) || azimuth <= (180.0 - t) {
                    Keast[(r, c)] = radI
                        * sh_val
                        * (altitude * deg2rad).cos()
                        * ((azimuth + t) * deg2rad).sin();
                }
                if azimuth > (90.0 - t) && azimuth <= (270.0 - t) {
                    Ksouth[(r, c)] = radI
                        * sh_val
                        * (altitude * deg2rad).cos()
                        * ((azimuth - 90.0 + t) * deg2rad).sin();
                }
                if azimuth > (180.0 - t) && azimuth <= (360.0 - t) {
                    Kwest[(r, c)] = radI
                        * sh_val
                        * (altitude * deg2rad).cos()
                        * ((azimuth - 180.0 + t) * deg2rad).sin();
                }
                if azimuth <= (90.0 - t) || azimuth > (270.0 - t) {
                    Knorth[(r, c)] = radI
                        * sh_val
                        * (altitude * deg2rad).cos()
                        * ((azimuth - 270.0 + t) * deg2rad).sin();
                }
            }
        }
    }

    // Isotropic diffuse/reflected radiation
    let ke_slice = as_slice_mut_checked(&mut Keast);
    let ks_slice = as_slice_mut_checked(&mut Ksouth);
    let kw_slice = as_slice_mut_checked(&mut Kwest);
    let kn_slice = as_slice_mut_checked(&mut Knorth);
    let fsh_slice = as_slice_checked(&F_sh);
    let svf_e_slice = as_slice_checked(&svfviktbuvegE);
    let svf_s_slice = as_slice_checked(&svfviktbuvegS);
    let svf_w_slice = as_slice_checked(&svfviktbuvegW);
    let svf_n_slice = as_slice_checked(&svfviktbuvegN);
    let kup_e_slice = as_slice_checked(&KupE);
    let kup_s_slice = as_slice_checked(&KupS);
    let kup_w_slice = as_slice_checked(&KupW);
    let kup_n_slice = as_slice_checked(&KupN);
    // `valid` is ArrayView2<u8>, not Array2<f32>, so the helpers above don't apply.
    let valid_slice = valid.as_ref().map(|v| v.as_slice().expect(VEG_CONTIG_INVARIANT));
    ke_slice
        .par_iter_mut()
        .zip(ks_slice.par_iter_mut())
        .zip(kw_slice.par_iter_mut())
        .zip(kn_slice.par_iter_mut())
        .enumerate()
        .for_each(|(idx, (((ke, ks), kw), kn))| {
            if let Some(ref vs) = valid_slice {
                if vs[idx] == 0 {
                    *ke = f32::NAN;
                    *ks = f32::NAN;
                    *kw = f32::NAN;
                    *kn = f32::NAN;
                    return;
                }
            }
            let fsh = fsh_slice[idx];
            let svf_e = svf_e_slice[idx];
            let svf_s = svf_s_slice[idx];
            let svf_w = svf_w_slice[idx];
            let svf_n = svf_n_slice[idx];
            let kup_e = kup_e_slice[idx];
            let kup_s = kup_s_slice[idx];
            let kup_w = kup_w_slice[idx];
            let kup_n = kup_n_slice[idx];
            let mix = radG * (1.0 - fsh) + radD * fsh;
            *ke += (radD * (1.0 - svf_e) + albedo * (svf_e * mix) + kup_e) * 0.5;
            *ks += (radD * (1.0 - svf_s) + albedo * (svf_s * mix) + kup_s) * 0.5;
            *kw += (radD * (1.0 - svf_w) + albedo * (svf_w * mix) + kup_w) * 0.5;
            *kn += (radD * (1.0 - svf_n) + albedo * (svf_n * mix) + kup_n) * 0.5;
        });

    KsideVegPureResult {
        keast: Keast,
        ksouth: Ksouth,
        kwest: Kwest,
        knorth: Knorth,
        kside_i: KsideI,
    }
}

/// Result container for lside_veg_v2022a direction-wise longwave fluxes.
#[pyclass]
pub struct LsideVegResult {
    #[pyo3(get)]
    pub least: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub lsouth: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub lwest: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub lnorth: Py<PyArray2<f32>>,
}

/// Vectorized Rust port of Python `Lside_veg_v2022a` operating on grid arrays.
/// Returns a `LsideVegResult` pyclass with four 2D arrays (least, lsouth, lwest, lnorth).
#[pyfunction]
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
pub fn lside_veg(
    py: Python,
    svfS: PyReadonlyArray2<f32>,
    svfW: PyReadonlyArray2<f32>,
    svfN: PyReadonlyArray2<f32>,
    svfE: PyReadonlyArray2<f32>,
    svfEveg: PyReadonlyArray2<f32>,
    svfSveg: PyReadonlyArray2<f32>,
    svfWveg: PyReadonlyArray2<f32>,
    svfNveg: PyReadonlyArray2<f32>,
    svfEaveg: PyReadonlyArray2<f32>,
    svfSaveg: PyReadonlyArray2<f32>,
    svfWaveg: PyReadonlyArray2<f32>,
    svfNaveg: PyReadonlyArray2<f32>,
    azimuth: f32,
    altitude: f32,
    Ta: f32,
    Tw: f32,
    SBC: f32,
    ewall: f32,
    Ldown: PyReadonlyArray2<f32>,
    esky: f32,
    t: f32,
    F_sh: PyReadonlyArray2<f32>,
    CI: f32,
    LupE: PyReadonlyArray2<f32>,
    LupS: PyReadonlyArray2<f32>,
    LupW: PyReadonlyArray2<f32>,
    LupN: PyReadonlyArray2<f32>,
    anisotropic_longwave: bool,
) -> PyResult<Py<LsideVegResult>> {
    let result = lside_veg_pure(
        svfS.as_array(),
        svfW.as_array(),
        svfN.as_array(),
        svfE.as_array(),
        svfEveg.as_array(),
        svfSveg.as_array(),
        svfWveg.as_array(),
        svfNveg.as_array(),
        svfEaveg.as_array(),
        svfSaveg.as_array(),
        svfWaveg.as_array(),
        svfNaveg.as_array(),
        azimuth,
        altitude,
        Ta,
        Tw,
        SBC,
        ewall,
        Ldown.as_array(),
        esky,
        t,
        F_sh.as_array(),
        CI,
        LupE.as_array(),
        LupS.as_array(),
        LupW.as_array(),
        LupN.as_array(),
        anisotropic_longwave,
        None,
    );

    Py::new(
        py,
        LsideVegResult {
            least: result.least.into_pyarray(py).unbind(),
            lsouth: result.lsouth.into_pyarray(py).unbind(),
            lwest: result.lwest.into_pyarray(py).unbind(),
            lnorth: result.lnorth.into_pyarray(py).unbind(),
        },
    )
}

/// Result container for Kside_veg_v2022a shortwave flux components.
#[pyclass]
pub struct KsideVegResult {
    #[pyo3(get)]
    pub keast: Py<PyArray2<f32>>, // Shortwave flux from east-facing surfaces
    #[pyo3(get)]
    pub ksouth: Py<PyArray2<f32>>, // Shortwave flux from south-facing surfaces
    #[pyo3(get)]
    pub kwest: Py<PyArray2<f32>>, // Shortwave flux from west-facing surfaces
    #[pyo3(get)]
    pub knorth: Py<PyArray2<f32>>, // Shortwave flux from north-facing surfaces
    #[pyo3(get)]
    pub kside_i: Py<PyArray2<f32>>, // Direct component (cyl = true)
    #[pyo3(get)]
    pub kside_d: Py<PyArray2<f32>>, // Diffuse component (anisotropic)
    #[pyo3(get)]
    pub kside: Py<PyArray2<f32>>, // Total side shortwave (anisotropic cyl only)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(non_snake_case)]
pub fn kside_veg(
    py: Python,
    radI: f32,
    radD: f32,
    radG: f32,
    shadow: PyReadonlyArray2<f32>,
    svfS: PyReadonlyArray2<f32>,
    svfW: PyReadonlyArray2<f32>,
    svfN: PyReadonlyArray2<f32>,
    svfE: PyReadonlyArray2<f32>,
    svfEveg: PyReadonlyArray2<f32>,
    svfSveg: PyReadonlyArray2<f32>,
    svfWveg: PyReadonlyArray2<f32>,
    svfNveg: PyReadonlyArray2<f32>,
    azimuth: f32,
    altitude: f32,
    psi: f32,
    t: f32,
    albedo: f32,
    F_sh: PyReadonlyArray2<f32>,
    KupE: PyReadonlyArray2<f32>,
    KupS: PyReadonlyArray2<f32>,
    KupW: PyReadonlyArray2<f32>,
    KupN: PyReadonlyArray2<f32>,
    cyl: bool,
    lv: Option<PyReadonlyArray2<f32>>, // (patch_alt, patch_azi, luminance)
    anisotropic_diffuse: bool,         // 1 -> anisotropic
    diffsh: Option<numpy::PyReadonlyArray3<f32>>, // (rows, cols, patches)
    asvf: Option<PyReadonlyArray2<f32>>, // sky view factor angle per pixel
    shmat: Option<numpy::PyReadonlyArray3<u8>>, // building shading matrix (uint8: 0=shadow, 255=sky)
    vegshmat: Option<numpy::PyReadonlyArray3<u8>>, // vegetation shading matrix
    vbshvegshmat: Option<numpy::PyReadonlyArray3<u8>>, // veg+building shading matrix
) -> PyResult<Py<KsideVegResult>> {
    // Borrow base 2D arrays
    let shadow = shadow.as_array();
    let svfS = svfS.as_array();
    let svfW = svfW.as_array();
    let svfN = svfN.as_array();
    let svfE = svfE.as_array();
    let svfEveg = svfEveg.as_array();
    let svfSveg = svfSveg.as_array();
    let svfWveg = svfWveg.as_array();
    let svfNveg = svfNveg.as_array();
    let F_sh = F_sh.as_array();
    let KupE = KupE.as_array();
    let KupS = KupS.as_array();
    let KupW = KupW.as_array();
    let KupN = KupN.as_array();

    // Output arrays (always allocate, some may stay zero based on flags)
    let shape = svfE.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let mut Keast = Array2::<f32>::zeros((rows, cols));
    let mut Ksouth = Array2::<f32>::zeros((rows, cols));
    let mut Kwest = Array2::<f32>::zeros((rows, cols));
    let mut Knorth = Array2::<f32>::zeros((rows, cols));
    let mut KsideI = Array2::<f32>::zeros((rows, cols));
    let mut KsideD = Array2::<f32>::zeros((rows, cols));
    let mut Kside = Array2::<f32>::zeros((rows, cols));

    let deg2rad = std::f32::consts::PI / 180.0;
    let vikttot = 4.4897f32;

    // Helper replicating Kvikt_veg for scalar svf pair
    #[inline]
    fn kvikt_veg(svf: f32, svfveg: f32, vikttot: f32) -> (f32, f32) {
        // viktwall
        let poly = |x: f32| -> f32 {
            63.227 * x.powi(6) - 161.51 * x.powi(5) + 156.91 * x.powi(4) - 70.424 * x.powi(3)
                + 16.773 * x.powi(2)
                - 0.4863 * x
        };
        let viktwall = (vikttot - poly(svf)) / vikttot;
        let svfvegbu = svfveg + svf - 1.0;
        let viktveg_tot = (vikttot - poly(svfvegbu)) / vikttot;
        let viktveg = viktveg_tot - viktwall;
        (viktveg, viktwall)
    }

    // Precompute direction-wise svfviktbuveg arrays only if needed (non-anisotropic branch)
    let mut svfviktbuvegE = Array2::<f32>::zeros((rows, cols));
    let mut svfviktbuvegS = Array2::<f32>::zeros((rows, cols));
    let mut svfviktbuvegW = Array2::<f32>::zeros((rows, cols));
    let mut svfviktbuvegN = Array2::<f32>::zeros((rows, cols));

    if !anisotropic_diffuse {
        for r in 0..rows {
            for c in 0..cols {
                let (vveg, vwall) = kvikt_veg(svfE[(r, c)], svfEveg[(r, c)], vikttot);
                svfviktbuvegE[(r, c)] = vwall + vveg * (1.0 - psi);
                let (vveg, vwall) = kvikt_veg(svfS[(r, c)], svfSveg[(r, c)], vikttot);
                svfviktbuvegS[(r, c)] = vwall + vveg * (1.0 - psi);
                let (vveg, vwall) = kvikt_veg(svfW[(r, c)], svfWveg[(r, c)], vikttot);
                svfviktbuvegW[(r, c)] = vwall + vveg * (1.0 - psi);
                let (vveg, vwall) = kvikt_veg(svfN[(r, c)], svfNveg[(r, c)], vikttot);
                svfviktbuvegN[(r, c)] = vwall + vveg * (1.0 - psi);
            }
        }
    }

    // Direct radiation components
    if cyl {
        // Cylinder: single side direct component (independent of direction splitting)
        for r in 0..rows {
            for c in 0..cols {
                KsideI[(r, c)] = shadow[(r, c)] * radI * (altitude * deg2rad).cos();
            }
        }
    } else {
        // Box: distribute to cardinal directions
        for r in 0..rows {
            for c in 0..cols {
                let sh_val = shadow[(r, c)];
                // East
                if azimuth > (360.0 - t) || azimuth <= (180.0 - t) {
                    Keast[(r, c)] = radI
                        * sh_val
                        * (altitude * deg2rad).cos()
                        * ((azimuth + t) * deg2rad).sin();
                }
                // South
                if azimuth > (90.0 - t) && azimuth <= (270.0 - t) {
                    Ksouth[(r, c)] = radI
                        * sh_val
                        * (altitude * deg2rad).cos()
                        * ((azimuth - 90.0 + t) * deg2rad).sin();
                }
                // West
                if azimuth > (180.0 - t) && azimuth <= (360.0 - t) {
                    Kwest[(r, c)] = radI
                        * sh_val
                        * (altitude * deg2rad).cos()
                        * ((azimuth - 180.0 + t) * deg2rad).sin();
                }
                // North
                if azimuth <= (90.0 - t) || azimuth > (270.0 - t) {
                    Knorth[(r, c)] = radI
                        * sh_val
                        * (altitude * deg2rad).cos()
                        * ((azimuth - 270.0 + t) * deg2rad).sin();
                }
            }
        }
    }

    // Diffuse / reflected radiation
    if anisotropic_diffuse {
        // --- Precompute patch-level invariants ---
        let lv_ro = lv.expect("lv (patch properties) required for anisotropic_diffuse");
        let diffsh_ro = diffsh.expect("diffsh required for anisotropic_diffuse");
        let asvf_ro = asvf.expect("asvf required for anisotropic_diffuse");
        let shmat_ro = shmat.expect("shmat required for anisotropic_diffuse");
        let vegshmat_ro = vegshmat.expect("vegshmat required for anisotropic_diffuse");
        let vbshvegshmat_ro = vbshvegshmat.expect("vbshvegshmat required for anisotropic_diffuse");

        let lv = lv_ro.as_array();
        let diffsh = diffsh_ro.as_array();
        let asvf = asvf_ro.as_array();
        let shmat = shmat_ro.as_array();
        let vegshmat = vegshmat_ro.as_array();
        let vbshvegshmat = vbshvegshmat_ro.as_array();

        let n_patches = lv.shape()[0];
        let patch_alt = lv.column(0);
        let patch_azi = lv.column(1);
        let patch_lum = if n_patches >= 3 {
            lv.column(2)
        } else {
            lv.column(0)
        };

        // Steradian per patch (Perez style) using counts per altitude
        let mut alt_counts = std::collections::HashMap::<i32, usize>::new();
        for &a in patch_alt.iter() {
            *alt_counts.entry((a * 1000.0) as i32).or_insert(0) += 1;
        }
        let mut ster: Vec<f32> = vec![0.0; n_patches];
        for i in 0..n_patches {
            let a = patch_alt[i];
            let key = (a * 1000.0) as i32;
            let cnt = *alt_counts.get(&key).unwrap_or(&1) as f32;
            ster[i] = if cnt > 1.0 {
                ((360.0 / cnt) * deg2rad)
                    * (((a + patch_alt[0]) * deg2rad).sin() - ((a - patch_alt[0]) * deg2rad).sin())
            } else if i == 0 {
                ((360.0 / cnt) * deg2rad) * (a * deg2rad).sin()
            } else {
                ((360.0 / cnt) * deg2rad)
                    * ((a * deg2rad).sin() - ((patch_alt[i - 1] + patch_alt[0]) * deg2rad).sin())
            };
        }
        // Normalize luminance -> lum_chi
        let mut rad_tot = 0.0f32;
        for i in 0..n_patches {
            rad_tot += patch_lum[i] * ster[i] * (patch_alt[i] * deg2rad).sin();
        }
        let lum_chi: Vec<f32> = (0..n_patches)
            .map(|i| {
                if rad_tot > 0.0 {
                    (patch_lum[i] * radD) / rad_tot
                } else {
                    0.0
                }
            })
            .collect();

        // Directional masks & precomputed cos weights per patch
        #[derive(Clone, Copy)]
        struct PatchConst {
            cos_alt: f32,
            ster: f32,
            w_e: f32,
            w_s: f32,
            w_w: f32,
            w_n: f32,
            is_e: bool,
            is_s: bool,
            is_w: bool,
            is_n: bool,
        }
        let mut pconst: Vec<PatchConst> = Vec::with_capacity(n_patches);
        for i in 0..n_patches {
            let p_azi = patch_azi[i];
            let cos_alt = (patch_alt[i] * deg2rad).cos();
            let w_e = ((90.0 - p_azi + t) * deg2rad).cos();
            let w_s = ((180.0 - p_azi + t) * deg2rad).cos();
            let w_w = ((270.0 - p_azi + t) * deg2rad).cos();
            let w_n = ((0.0 - p_azi + t) * deg2rad).cos();
            pconst.push(PatchConst {
                cos_alt,
                ster: ster[i],
                w_e,
                w_s,
                w_w,
                w_n,
                is_e: p_azi > 360.0 || p_azi <= 180.0,
                is_s: p_azi > 90.0 && p_azi <= 270.0,
                is_w: p_azi > 180.0 && p_azi <= 360.0,
                is_n: p_azi > 270.0 || p_azi <= 90.0,
            });
        }

        // Common surfaces constants
        let sunlit_surface =
            (albedo * (radI * (altitude * deg2rad).cos()) + (radD * 0.5)) / std::f32::consts::PI;
        let shaded_surface = (albedo * radD * 0.5) / std::f32::consts::PI;

        let npix = rows * cols;
        if cyl {
            // Parallel map -> collect per-pixel cylinder results
            #[derive(Clone, Copy)]
            struct CylRes {
                kside_d: f32,
                kref_sun: f32,
                kref_sh: f32,
                kref_veg: f32,
            }
            // Cylinder + anisotropic: Only the ground (Kup) component is assigned to directional outputs,
            // matching the original model. All anisotropic diffuse/reflected energy is in Kside/KsideD.
            let results: Vec<CylRes> = (0..npix)
                .into_par_iter()
                .map(|idx| {
                    let r = idx / cols;
                    let c = idx % cols;
                    let asvf_val = asvf[(r, c)];
                    let mut ref_sun = 0.0f32;
                    let mut ref_sh = 0.0f32;
                    let mut ref_veg = 0.0f32;
                    let mut kside_d_loc = 0.0f32;
                    for i in 0..n_patches {
                        let pc = pconst[i];
                        let angle_inc = pc.cos_alt;
                        let lum = lum_chi[i];
                        kside_d_loc += diffsh[(r, c, i)] * lum * angle_inc * pc.ster;
                        let veg_flag = vegshmat[(r, c, i)] == 0 || vbshvegshmat[(r, c, i)] == 0;
                        if veg_flag {
                            ref_veg += shaded_surface * pc.ster * angle_inc;
                        }
                        let temp_vbsh = shmat[(r, c, i)] == 0 && vbshvegshmat[(r, c, i)] == 255;
                        if temp_vbsh {
                            let (sunlit_patch, shaded_patch) =
                                crate::sunlit_shaded_patches::shaded_or_sunlit_pixel(
                                    altitude,
                                    azimuth,
                                    patch_alt[i],
                                    patch_azi[i],
                                    asvf_val,
                                );
                            if sunlit_patch {
                                ref_sun += sunlit_surface * pc.ster * angle_inc;
                            }
                            if shaded_patch {
                                ref_sh += shaded_surface * pc.ster * angle_inc;
                            }
                        }
                    }
                    CylRes {
                        kside_d: kside_d_loc,
                        kref_sun: ref_sun,
                        kref_sh: ref_sh,
                        kref_veg: ref_veg,
                    }
                })
                .collect();
            // Write-back sequentially
            let kside_d_slice = as_slice_mut_checked(&mut KsideD);
            let kside_slice = as_slice_mut_checked(&mut Kside);
            let kside_i_slice = as_slice_checked(&KsideI);
            for (idx, res) in results.into_iter().enumerate() {
                kside_d_slice[idx] = res.kside_d;
                kside_slice[idx] =
                    kside_i_slice[idx] + res.kside_d + res.kref_sun + res.kref_sh + res.kref_veg;
            }
            // Directional simplified contributions
            for r in 0..rows {
                for c in 0..cols {
                    Keast[(r, c)] = KupE[(r, c)] * 0.5;
                    Kwest[(r, c)] = KupW[(r, c)] * 0.5;
                    Knorth[(r, c)] = KupN[(r, c)] * 0.5;
                    Ksouth[(r, c)] = KupS[(r, c)] * 0.5;
                }
            }
        } else {
            // Parallel map -> collect per-pixel box directional totals
            #[derive(Clone, Copy)]
            struct BoxRes {
                ke: f32,
                ks: f32,
                kw: f32,
                kn: f32,
            }
            let results: Vec<BoxRes> = (0..npix)
                .into_par_iter()
                .map(|idx| {
                    let r = idx / cols;
                    let c = idx % cols;
                    let asvf_val = asvf[(r, c)];
                    let mut diff_e = 0f32;
                    let mut diff_s = 0f32;
                    let mut diff_w = 0f32;
                    let mut diff_n = 0f32;
                    let mut ref_sun_e = 0f32;
                    let mut ref_sun_s = 0f32;
                    let mut ref_sun_w = 0f32;
                    let mut ref_sun_n = 0f32;
                    let mut ref_sh_e = 0f32;
                    let mut ref_sh_s = 0f32;
                    let mut ref_sh_w = 0f32;
                    let mut ref_sh_n = 0f32;
                    let mut ref_veg_e = 0f32;
                    let mut ref_veg_s = 0f32;
                    let mut ref_veg_w = 0f32;
                    let mut ref_veg_n = 0f32;
                    for i in 0..n_patches {
                        let pc = pconst[i];
                        let lum = lum_chi[i];
                        let cos_alt = pc.cos_alt;
                        let diff_val = diffsh[(r, c, i)] * lum * pc.ster;
                        if pc.is_e {
                            diff_e += diff_val * pc.w_e;
                        }
                        if pc.is_s {
                            diff_s += diff_val * pc.w_s;
                        }
                        if pc.is_w {
                            diff_w += diff_val * pc.w_w;
                        }
                        if pc.is_n {
                            diff_n += diff_val * pc.w_n;
                        }
                        let veg_flag = vegshmat[(r, c, i)] == 0 || vbshvegshmat[(r, c, i)] == 0;
                        if veg_flag {
                            if pc.is_e {
                                ref_veg_e += shaded_surface * pc.ster * cos_alt * pc.w_e;
                            }
                            if pc.is_s {
                                ref_veg_s += shaded_surface * pc.ster * cos_alt * pc.w_s;
                            }
                            if pc.is_w {
                                ref_veg_w += shaded_surface * pc.ster * cos_alt * pc.w_w;
                            }
                            if pc.is_n {
                                ref_veg_n += shaded_surface * pc.ster * cos_alt * pc.w_n;
                            }
                        }
                        let temp_vbsh = shmat[(r, c, i)] == 0 && vbshvegshmat[(r, c, i)] == 255;
                        if temp_vbsh {
                            let az_diff = (azimuth - patch_azi[i]).abs();
                            if az_diff > 90.0 && az_diff < 270.0 {
                                let (sunlit_patch, shaded_patch) =
                                    crate::sunlit_shaded_patches::shaded_or_sunlit_pixel(
                                        altitude,
                                        azimuth,
                                        patch_alt[i],
                                        patch_azi[i],
                                        asvf_val,
                                    );
                                if sunlit_patch {
                                    if pc.is_e {
                                        ref_sun_e += sunlit_surface * pc.ster * cos_alt * pc.w_e;
                                    }
                                    if pc.is_s {
                                        ref_sun_s += sunlit_surface * pc.ster * cos_alt * pc.w_s;
                                    }
                                    if pc.is_w {
                                        ref_sun_w += sunlit_surface * pc.ster * cos_alt * pc.w_w;
                                    }
                                    if pc.is_n {
                                        ref_sun_n += sunlit_surface * pc.ster * cos_alt * pc.w_n;
                                    }
                                }
                                if shaded_patch {
                                    if pc.is_e {
                                        ref_sh_e += shaded_surface * pc.ster * cos_alt * pc.w_e;
                                    }
                                    if pc.is_s {
                                        ref_sh_s += shaded_surface * pc.ster * cos_alt * pc.w_s;
                                    }
                                    if pc.is_w {
                                        ref_sh_w += shaded_surface * pc.ster * cos_alt * pc.w_w;
                                    }
                                    if pc.is_n {
                                        ref_sh_n += shaded_surface * pc.ster * cos_alt * pc.w_n;
                                    }
                                }
                            } else {
                                if pc.is_e {
                                    ref_sh_e += shaded_surface * pc.ster * cos_alt * pc.w_e;
                                }
                                if pc.is_s {
                                    ref_sh_s += shaded_surface * pc.ster * cos_alt * pc.w_s;
                                }
                                if pc.is_w {
                                    ref_sh_w += shaded_surface * pc.ster * cos_alt * pc.w_w;
                                }
                                if pc.is_n {
                                    ref_sh_n += shaded_surface * pc.ster * cos_alt * pc.w_n;
                                }
                            }
                        }
                    }
                    BoxRes {
                        ke: diff_e + ref_sun_e + ref_sh_e + ref_veg_e + KupE[(r, c)] * 0.5,
                        ks: diff_s + ref_sun_s + ref_sh_s + ref_veg_s + KupS[(r, c)] * 0.5,
                        kw: diff_w + ref_sun_w + ref_sh_w + ref_veg_w + KupW[(r, c)] * 0.5,
                        kn: diff_n + ref_sun_n + ref_sh_n + ref_veg_n + KupN[(r, c)] * 0.5,
                    }
                })
                .collect();
            // Write-back sequentially
            let ke_slice = as_slice_mut_checked(&mut Keast);
            let ks_slice = as_slice_mut_checked(&mut Ksouth);
            let kw_slice = as_slice_mut_checked(&mut Kwest);
            let kn_slice = as_slice_mut_checked(&mut Knorth);
            for (idx, res) in results.into_iter().enumerate() {
                // Preserve existing direct component (already stored in *_slice from earlier direct computation)
                ke_slice[idx] += res.ke;
                ks_slice[idx] += res.ks;
                kw_slice[idx] += res.kw;
                kn_slice[idx] += res.kn;
            }
            // Kside, KsideD remain zero (not defined for box anisotropic total)
        }
    } else {
        // Isotropic (original) formulation, now parallelized using nested zip pattern
        let ke_slice = as_slice_mut_checked(&mut Keast);
        let ks_slice = as_slice_mut_checked(&mut Ksouth);
        let kw_slice = as_slice_mut_checked(&mut Kwest);
        let kn_slice = as_slice_mut_checked(&mut Knorth);
        let fsh_slice = as_slice_checked(&F_sh);
        let svf_e_slice = as_slice_checked(&svfviktbuvegE);
        let svf_s_slice = as_slice_checked(&svfviktbuvegS);
        let svf_w_slice = as_slice_checked(&svfviktbuvegW);
        let svf_n_slice = as_slice_checked(&svfviktbuvegN);
        let kup_e_slice = as_slice_checked(&KupE);
        let kup_s_slice = as_slice_checked(&KupS);
        let kup_w_slice = as_slice_checked(&KupW);
        let kup_n_slice = as_slice_checked(&KupN);
        ke_slice
            .par_iter_mut()
            .zip(ks_slice.par_iter_mut())
            .zip(kw_slice.par_iter_mut())
            .zip(kn_slice.par_iter_mut())
            .enumerate()
            .for_each(|(idx, (((ke, ks), kw), kn))| {
                let fsh = fsh_slice[idx];
                let svf_e = svf_e_slice[idx];
                let svf_s = svf_s_slice[idx];
                let svf_w = svf_w_slice[idx];
                let svf_n = svf_n_slice[idx];
                let kup_e = kup_e_slice[idx];
                let kup_s = kup_s_slice[idx];
                let kup_w = kup_w_slice[idx];
                let kup_n = kup_n_slice[idx];
                let mix = radG * (1.0 - fsh) + radD * fsh;
                // East
                let ke_dg = (radD * (1.0 - svf_e) + albedo * (svf_e * mix) + kup_e) * 0.5;
                // Cylinder (cyl=true) still receives isotropic diffuse directional components (Python does Keast = 0 + KeastDG)
                *ke += ke_dg;
                // South
                let ks_dg = (radD * (1.0 - svf_s) + albedo * (svf_s * mix) + kup_s) * 0.5;
                *ks += ks_dg;
                // West
                let kw_dg = (radD * (1.0 - svf_w) + albedo * (svf_w * mix) + kup_w) * 0.5;
                *kw += kw_dg;
                // North
                let kn_dg = (radD * (1.0 - svf_n) + albedo * (svf_n * mix) + kup_n) * 0.5;
                *kn += kn_dg;
            });
    }

    // Package result
    Py::new(
        py,
        KsideVegResult {
            keast: Keast.into_pyarray(py).unbind(),
            ksouth: Ksouth.into_pyarray(py).unbind(),
            kwest: Kwest.into_pyarray(py).unbind(),
            knorth: Knorth.into_pyarray(py).unbind(),
            kside_i: KsideI.into_pyarray(py).unbind(),
            kside_d: KsideD.into_pyarray(py).unbind(),
            kside: Kside.into_pyarray(py).unbind(),
        },
    )
}
