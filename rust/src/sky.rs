use crate::{emissivity_models, patch_radiation, sunlit_shaded_patches};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::prelude::*;
use rayon::prelude::*;

const PI: f32 = std::f32::consts::PI;
const SBC: f32 = 5.67051e-8; // Stefan-Boltzmann constant
const MIN_SUN_ELEVATION_RAD: f32 = 3.0 * PI / 180.0; // 3° threshold for low sun guard

/// Extract a single shadow bit from a bitpacked shadow matrix.
/// Shape: (rows, cols, n_pack) where n_pack = ceil(n_patches / 8).
/// Returns true if the shadow bit is set (was 255 in the original u8 format).
#[inline(always)]
fn get_shadow_bit(packed: &ArrayView3<u8>, r: usize, c: usize, patch: usize) -> bool {
    (packed[[r, c, patch >> 3]] >> (patch & 7)) & 1 == 1
}

/// Sun position parameters
#[pyclass]
#[derive(Clone)]
pub struct SunParams {
    #[pyo3(get, set)]
    pub altitude: f32,
    #[pyo3(get, set)]
    pub azimuth: f32,
}

#[pymethods]
impl SunParams {
    #[new]
    pub fn new(altitude: f32, azimuth: f32) -> Self {
        Self { altitude, azimuth }
    }
}

/// Sky model parameters
#[pyclass]
#[derive(Clone)]
pub struct SkyParams {
    #[pyo3(get, set)]
    pub esky: f32,
    #[pyo3(get, set)]
    pub ta: f32,
    #[pyo3(get, set)]
    pub cyl: bool,
    #[pyo3(get, set)]
    pub wall_scheme: bool,
    #[pyo3(get, set)]
    pub albedo: f32,
}

#[pymethods]
impl SkyParams {
    #[new]
    pub fn new(esky: f32, ta: f32, cyl: bool, wall_scheme: bool, albedo: f32) -> Self {
        Self {
            esky,
            ta,
            cyl,
            wall_scheme,
            albedo,
        }
    }
}

/// Surface radiation parameters
#[pyclass]
#[derive(Clone)]
pub struct SurfaceParams {
    #[pyo3(get, set)]
    pub tgwall: f32,
    #[pyo3(get, set)]
    pub ewall: f32,
    #[pyo3(get, set)]
    pub rad_i: f32,
    #[pyo3(get, set)]
    pub rad_d: f32,
}

#[pymethods]
impl SurfaceParams {
    #[new]
    pub fn new(tgwall: f32, ewall: f32, rad_i: f32, rad_d: f32) -> Self {
        Self {
            tgwall,
            ewall,
            rad_i,
            rad_d,
        }
    }
}

#[pyclass]
pub struct SkyResult {
    #[pyo3(get)]
    pub ldown: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub lside: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub lside_sky: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub lside_veg: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub lside_sh: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub lside_sun: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub lside_ref: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub least: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub lwest: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub lnorth: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub lsouth: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub keast: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub ksouth: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub kwest: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub knorth: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub kside_i: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub kside_d: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub kside: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub steradians: Py<PyArray1<f32>>,
    #[pyo3(get)]
    pub skyalt: Py<PyArray1<f32>>,
}

// Struct to hold the accumulated radiation values for a single pixel
#[derive(Clone, Copy)]
struct PixelResult {
    lside_sky: f32,
    ldown_sky: f32,
    lside_veg: f32,
    ldown_veg: f32,
    lside_sun: f32,
    lside_sh: f32,
    ldown_sun: f32,
    ldown_sh: f32,
    kside_d: f32,
    kref_sun: f32,
    kref_sh: f32,
    kref_veg: f32,
    least: f32,
    lsouth: f32,
    lwest: f32,
    lnorth: f32,
    lside_ref: f32,
    ldown_ref: f32,
}

impl PixelResult {
    fn new() -> Self {
        Self {
            lside_sky: 0.0,
            ldown_sky: 0.0,
            lside_veg: 0.0,
            ldown_veg: 0.0,
            lside_sun: 0.0,
            lside_sh: 0.0,
            ldown_sun: 0.0,
            ldown_sh: 0.0,
            kside_d: 0.0,
            kref_sun: 0.0,
            kref_sh: 0.0,
            kref_veg: 0.0,
            least: 0.0,
            lsouth: 0.0,
            lwest: 0.0,
            lnorth: 0.0,
            lside_ref: 0.0,
            ldown_ref: 0.0,
        }
    }

    fn nan() -> Self {
        Self {
            lside_sky: f32::NAN,
            ldown_sky: f32::NAN,
            lside_veg: f32::NAN,
            ldown_veg: f32::NAN,
            lside_sun: f32::NAN,
            lside_sh: f32::NAN,
            ldown_sun: f32::NAN,
            ldown_sh: f32::NAN,
            kside_d: f32::NAN,
            kref_sun: f32::NAN,
            kref_sh: f32::NAN,
            kref_veg: f32::NAN,
            least: f32::NAN,
            lsouth: f32::NAN,
            lwest: f32::NAN,
            lnorth: f32::NAN,
            lside_ref: f32::NAN,
            ldown_ref: f32::NAN,
        }
    }
}

/// Pure-ndarray result from anisotropic sky calculation (no PyO3 types).
pub(crate) struct SkyResultPure {
    pub ldown: Array2<f32>,
    pub lside: Array2<f32>,
    pub lside_sky: Array2<f32>,
    pub lside_veg: Array2<f32>,
    pub lside_sh: Array2<f32>,
    pub lside_sun: Array2<f32>,
    pub lside_ref: Array2<f32>,
    pub least: Array2<f32>,
    pub lsouth: Array2<f32>,
    pub lwest: Array2<f32>,
    pub lnorth: Array2<f32>,
    pub keast: Array2<f32>,
    pub ksouth: Array2<f32>,
    pub kwest: Array2<f32>,
    pub knorth: Array2<f32>,
    pub kside_i: Array2<f32>,
    pub kside_d: Array2<f32>,
    pub kside: Array2<f32>,
}

/// Pure-ndarray anisotropic sky calculation, callable from pipeline.rs.
#[allow(clippy::too_many_arguments)]
#[allow(non_snake_case)]
pub(crate) fn anisotropic_sky_pure(
    shmat: ArrayView3<u8>,
    vegshmat: ArrayView3<u8>,
    vbshvegshmat: ArrayView3<u8>,
    solar_altitude: f32,
    solar_azimuth: f32,
    esky: f32,
    ta: f32,
    cyl: bool,
    wall_scheme: bool,
    albedo: f32,
    tgwall: f32,
    ewall: f32,
    rad_i: f32,
    rad_d: f32,
    asvf: ArrayView2<f32>,
    l_patches: ArrayView2<f32>,
    steradians: ArrayView1<f32>,
    lup: ArrayView2<f32>,
    lv: ArrayView2<f32>,
    shadow: ArrayView2<f32>,
    kup_e: ArrayView2<f32>,
    kup_s: ArrayView2<f32>,
    kup_w: ArrayView2<f32>,
    kup_n: ArrayView2<f32>,
    voxel_table: Option<ArrayView2<f32>>,
    voxel_maps: Option<ArrayView3<f32>>,
    valid: Option<ArrayView2<u8>>,
) -> SkyResultPure {
    let rows = shmat.shape()[0];
    let cols = shmat.shape()[1];
    let n_patches = l_patches.shape()[0];

    let mut lside_sky = Array2::<f32>::zeros((rows, cols));
    let mut ldown_sky = Array2::<f32>::zeros((rows, cols));
    let mut lside_veg = Array2::<f32>::zeros((rows, cols));
    let mut ldown_veg = Array2::<f32>::zeros((rows, cols));
    let mut lside_sun = Array2::<f32>::zeros((rows, cols));
    let mut lside_sh = Array2::<f32>::zeros((rows, cols));
    let mut ldown_sun = Array2::<f32>::zeros((rows, cols));
    let mut ldown_sh = Array2::<f32>::zeros((rows, cols));
    let mut kside_d = Array2::<f32>::zeros((rows, cols));
    let mut kref_sun = Array2::<f32>::zeros((rows, cols));
    let mut kref_sh = Array2::<f32>::zeros((rows, cols));
    let mut kref_veg = Array2::<f32>::zeros((rows, cols));
    let mut least = Array2::<f32>::zeros((rows, cols));
    let mut lwest = Array2::<f32>::zeros((rows, cols));
    let mut lnorth = Array2::<f32>::zeros((rows, cols));
    let mut lsouth = Array2::<f32>::zeros((rows, cols));
    let mut lside_ref = Array2::<f32>::zeros((rows, cols));
    let mut ldown_ref = Array2::<f32>::zeros((rows, cols));

    let patch_altitude = l_patches.column(0).to_owned();
    let patch_azimuth = l_patches.column(1).to_owned();

    let deg2rad = PI / 180.0;

    // Shortwave normalization
    let mut lum_chi = Array1::<f32>::zeros(n_patches);
    if solar_altitude > 0.0 {
        let patch_luminance = lv.column(2);
        let mut rad_tot = 0.0;
        for i in 0..n_patches {
            rad_tot += patch_luminance[i] * steradians[i] * (patch_altitude[i] * deg2rad).sin();
        }
        lum_chi = patch_luminance.mapv(|lum| (lum * rad_d) / rad_tot);
    }

    // Precompute emissivity per patch
    let (_patch_emissivity_normalized, esky_band) =
        emissivity_models::model2(&l_patches.to_owned(), esky, ta);

    // Main parallel computation over pixels
    let pixel_indices: Vec<(usize, usize)> = (0..rows)
        .flat_map(|r| (0..cols).map(move |c| (r, c)))
        .collect();

    let pixel_results: Vec<PixelResult> = pixel_indices
        .into_par_iter()
        .map(|(r, c)| {
            if let Some(ref v) = valid {
                if v[[r, c]] == 0 {
                    return PixelResult::nan();
                }
            }
            let mut pres = PixelResult::new();
            let pixel_asvf = asvf[[r, c]];

            for i in 0..n_patches {
                let p_alt = patch_altitude[i];
                let p_azi = patch_azimuth[i];
                let steradian = steradians[i];

                let sh = get_shadow_bit(&shmat, r, c, i);
                let vsh = get_shadow_bit(&vegshmat, r, c, i);
                let vbsh = get_shadow_bit(&vbshvegshmat, r, c, i);
                let temp_sky = sh && vsh;
                let temp_vegsh = !vsh || !vbsh;
                let temp_sh = !sh && vbsh;

                {
                    let angle_of_incidence = (p_alt * deg2rad).cos();
                    let angle_of_incidence_h = (p_alt * deg2rad).sin();

                    if temp_sky {
                        let temp_emissivity = esky_band[i];
                        let ta_k = ta + 273.15;
                        let lval = (temp_emissivity * SBC * ta_k.powi(4)) / PI;
                        let lside_patch = lval * steradian * angle_of_incidence;
                        let ldown_patch = lval * steradian * angle_of_incidence_h;

                        let (ls, ld, le, lso, lw, ln) = patch_radiation::longwave_from_sky_pixel(
                            lside_patch,
                            ldown_patch,
                            p_azi,
                        );
                        pres.lside_sky += ls;
                        pres.ldown_sky += ld;
                        pres.least += le;
                        pres.lsouth += lso;
                        pres.lwest += lw;
                        pres.lnorth += ln;
                    }

                    if temp_vegsh {
                        let (ls, ld, le, lso, lw, ln) = patch_radiation::longwave_from_veg_pixel(
                            steradian,
                            angle_of_incidence,
                            angle_of_incidence_h,
                            p_alt,
                            p_azi,
                            ewall,
                            ta,
                        );
                        pres.lside_veg += ls;
                        pres.ldown_veg += ld;
                        pres.least += le;
                        pres.lsouth += lso;
                        pres.lwest += lw;
                        pres.lnorth += ln;
                    }

                    if temp_sh {
                        let (sunlit_patch, shaded_patch) =
                            sunlit_shaded_patches::shaded_or_sunlit_pixel(
                                solar_altitude,
                                solar_azimuth,
                                p_alt,
                                p_azi,
                                pixel_asvf,
                            );

                        if !wall_scheme {
                            let azimuth_difference = (solar_azimuth - p_azi).abs();
                            let (ls_sun, ls_sh, ld_sun, ld_sh, le, lso, lw, ln) =
                                patch_radiation::longwave_from_buildings_pixel(
                                    steradian,
                                    angle_of_incidence,
                                    angle_of_incidence_h,
                                    p_azi,
                                    sunlit_patch,
                                    shaded_patch,
                                    azimuth_difference,
                                    solar_altitude,
                                    ewall,
                                    ta,
                                    tgwall,
                                );
                            pres.lside_sun += ls_sun;
                            pres.lside_sh += ls_sh;
                            pres.ldown_sun += ld_sun;
                            pres.ldown_sh += ld_sh;
                            pres.least += le;
                            pres.lsouth += lso;
                            pres.lwest += lw;
                            pres.lnorth += ln;
                        } else {
                            let voxel_map_val = voxel_maps.as_ref().unwrap()[[r, c, i]];
                            if voxel_map_val > 0.0 {
                                let (ls_sun, ls_sh, ld_sun, ld_sh, le, lso, lw, ln) =
                                    patch_radiation::longwave_from_buildings_wall_scheme_pixel(
                                        *voxel_table.as_ref().unwrap(),
                                        voxel_map_val as usize,
                                        steradian,
                                        angle_of_incidence,
                                        angle_of_incidence_h,
                                        p_azi,
                                    );
                                pres.lside_sun += ls_sun;
                                pres.lside_sh += ls_sh;
                                pres.ldown_sun += ld_sun;
                                pres.ldown_sh += ld_sh;
                                pres.least += le;
                                pres.lsouth += lso;
                                pres.lwest += lw;
                                pres.lnorth += ln;
                            } else {
                                let azimuth_difference = (solar_azimuth - p_azi).abs();
                                let (ls_sun, ls_sh, ld_sun, ld_sh, le, lso, lw, ln) =
                                    patch_radiation::longwave_from_buildings_pixel(
                                        steradian,
                                        angle_of_incidence,
                                        angle_of_incidence_h,
                                        p_azi,
                                        sunlit_patch,
                                        shaded_patch,
                                        azimuth_difference,
                                        solar_altitude,
                                        ewall,
                                        ta,
                                        tgwall,
                                    );
                                pres.lside_sun += ls_sun;
                                pres.lside_sh += ls_sh;
                                pres.ldown_sun += ld_sun;
                                pres.ldown_sh += ld_sh;
                                pres.least += le;
                                pres.lsouth += lso;
                                pres.lwest += lw;
                                pres.lnorth += ln;
                            }
                        }
                    }

                    if solar_altitude > 0.0 {
                        if temp_sky {
                            pres.kside_d += lum_chi[i] * angle_of_incidence * steradian;
                        }
                        let sunlit_surface = (albedo * (rad_i * (solar_altitude * deg2rad).cos())
                            + (rad_d * 0.5))
                            / PI;
                        let shaded_surface = (albedo * rad_d * 0.5) / PI;
                        if temp_vegsh {
                            pres.kref_veg += shaded_surface * steradian * angle_of_incidence;
                        }
                        if temp_sh {
                            let (sunlit_patch, shaded_patch) =
                                sunlit_shaded_patches::shaded_or_sunlit_pixel(
                                    solar_altitude,
                                    solar_azimuth,
                                    p_alt,
                                    p_azi,
                                    pixel_asvf,
                                );
                            if sunlit_patch {
                                pres.kref_sun += sunlit_surface * steradian * angle_of_incidence;
                            }
                            if shaded_patch {
                                pres.kref_sh += shaded_surface * steradian * angle_of_incidence;
                            }
                        }
                    }
                }
            }

            // Reflected longwave
            let mut pres_with_reflection = pres;
            for i in 0..n_patches {
                let p_alt = patch_altitude[i];
                let p_azi = patch_azimuth[i];
                let steradian = steradians[i];
                let temp_sh = !get_shadow_bit(&shmat, r, c, i)
                    || !get_shadow_bit(&vegshmat, r, c, i)
                    || !get_shadow_bit(&vbshvegshmat, r, c, i);

                if temp_sh {
                    let angle_of_incidence = (p_alt * deg2rad).cos();
                    let angle_of_incidence_h = (p_alt * deg2rad).sin();
                    let (lsr, ldr, le, lso, lw, ln) = patch_radiation::reflected_longwave_pixel(
                        steradian,
                        angle_of_incidence,
                        angle_of_incidence_h,
                        p_azi,
                        pres.ldown_sky,
                        lup[[r, c]],
                        ewall,
                    );
                    pres_with_reflection.lside_ref += lsr;
                    pres_with_reflection.ldown_ref += ldr;
                    pres_with_reflection.least += le;
                    pres_with_reflection.lsouth += lso;
                    pres_with_reflection.lwest += lw;
                    pres_with_reflection.lnorth += ln;
                }
            }
            pres_with_reflection
        })
        .collect();

    // Populate final 2D arrays
    for (idx, pres) in pixel_results.into_iter().enumerate() {
        let r = idx / cols;
        let c = idx % cols;
        lside_sky[[r, c]] = pres.lside_sky;
        ldown_sky[[r, c]] = pres.ldown_sky;
        lside_veg[[r, c]] = pres.lside_veg;
        ldown_veg[[r, c]] = pres.ldown_veg;
        lside_sun[[r, c]] = pres.lside_sun;
        lside_sh[[r, c]] = pres.lside_sh;
        ldown_sun[[r, c]] = pres.ldown_sun;
        ldown_sh[[r, c]] = pres.ldown_sh;
        kside_d[[r, c]] = pres.kside_d;
        kref_sun[[r, c]] = pres.kref_sun;
        kref_sh[[r, c]] = pres.kref_sh;
        kref_veg[[r, c]] = pres.kref_veg;
        least[[r, c]] = pres.least;
        lsouth[[r, c]] = pres.lsouth;
        lwest[[r, c]] = pres.lwest;
        lnorth[[r, c]] = pres.lnorth;
        lside_ref[[r, c]] = pres.lside_ref;
        ldown_ref[[r, c]] = pres.ldown_ref;
    }

    let lside = &lside_sky + &lside_veg + &lside_sh + &lside_sun + &lside_ref;
    let ldown = &ldown_sky + &ldown_veg + &ldown_sh + &ldown_sun + &ldown_ref;

    let mut kside_i = Array2::<f32>::zeros((rows, cols));
    let mut keast = Array2::<f32>::zeros((rows, cols));
    let mut kwest = Array2::<f32>::zeros((rows, cols));
    let mut knorth = Array2::<f32>::zeros((rows, cols));
    let mut ksouth = Array2::<f32>::zeros((rows, cols));

    if cyl {
        // Cylinder: direct beam goes into aggregate kside_i
        kside_i = &shadow * rad_i * (solar_altitude * deg2rad).cos();
    } else if solar_altitude > 0.0 {
        // Box: direct beam split into directional components
        let cos_alt = (solar_altitude * deg2rad).cos();
        let azi = solar_azimuth;
        // East face (azimuth 0..180)
        if azi > 360.0 || azi <= 180.0 {
            keast = &shadow * rad_i * cos_alt * (azi * deg2rad).sin();
        }
        // South face (azimuth 90..270)
        if azi > 90.0 && azi <= 270.0 {
            ksouth = &shadow * rad_i * cos_alt * ((azi - 90.0) * deg2rad).sin();
        }
        // West face (azimuth 180..360)
        if azi > 180.0 && azi <= 360.0 {
            kwest = &shadow * rad_i * cos_alt * ((azi - 180.0) * deg2rad).sin();
        }
        // North face (azimuth 270..360 or 0..90)
        if azi <= 90.0 || azi > 270.0 {
            knorth = &shadow * rad_i * cos_alt * ((azi - 270.0) * deg2rad).sin();
        }
    }

    let mut kside = Array2::<f32>::zeros((rows, cols));

    if solar_altitude > 0.0 {
        kside = &kside_i + &kside_d + &kref_sun + &kref_sh + &kref_veg;
        // Add ground-reflected shortwave to directional components
        keast = keast + &kup_e * 0.5;
        kwest = kwest + &kup_w * 0.5;
        knorth = knorth + &kup_n * 0.5;
        ksouth = ksouth + &kup_s * 0.5;
    }

    SkyResultPure {
        ldown,
        lside,
        lside_sky,
        lside_veg,
        lside_sh,
        lside_sun,
        lside_ref,
        least,
        lsouth,
        lwest,
        lnorth,
        keast,
        ksouth,
        kwest,
        knorth,
        kside_i,
        kside_d,
        kside,
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(non_snake_case)]
pub fn anisotropic_sky(
    py: Python,
    shmat: PyReadonlyArray3<u8>,
    vegshmat: PyReadonlyArray3<u8>,
    vbshvegshmat: PyReadonlyArray3<u8>,
    sun: &SunParams,
    asvf: PyReadonlyArray2<f32>,
    sky: &SkyParams,
    l_patches: PyReadonlyArray2<f32>,
    voxel_table: Option<PyReadonlyArray2<f32>>,
    voxel_maps: Option<PyReadonlyArray3<f32>>,
    steradians: PyReadonlyArray1<f32>,
    surface: &SurfaceParams,
    lup: PyReadonlyArray2<f32>,
    lv: PyReadonlyArray2<f32>,
    shadow: PyReadonlyArray2<f32>,
    kup_e: PyReadonlyArray2<f32>,
    kup_s: PyReadonlyArray2<f32>,
    kup_w: PyReadonlyArray2<f32>,
    kup_n: PyReadonlyArray2<f32>,
) -> PyResult<Py<SkyResult>> {
    let voxel_table_view = voxel_table.as_ref().map(|v| v.as_array());
    let voxel_maps_view = voxel_maps.as_ref().map(|v| v.as_array());

    // Compute unique altitudes for the PyO3 return value
    let l_patches_v = l_patches.as_array();
    let patch_altitude = l_patches_v.column(0);
    let mut skyalt_vec: Vec<f32> = patch_altitude.iter().cloned().collect();
    skyalt_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    skyalt_vec.dedup();
    let skyalt = Array1::<f32>::from(skyalt_vec);

    let pure_result = anisotropic_sky_pure(
        shmat.as_array(),
        vegshmat.as_array(),
        vbshvegshmat.as_array(),
        sun.altitude,
        sun.azimuth,
        sky.esky,
        sky.ta,
        sky.cyl,
        sky.wall_scheme,
        sky.albedo,
        surface.tgwall,
        surface.ewall,
        surface.rad_i,
        surface.rad_d,
        asvf.as_array(),
        l_patches_v,
        steradians.as_array(),
        lup.as_array(),
        lv.as_array(),
        shadow.as_array(),
        kup_e.as_array(),
        kup_s.as_array(),
        kup_w.as_array(),
        kup_n.as_array(),
        voxel_table_view,
        voxel_maps_view,
        None,
    );

    let steradians_owned = steradians.as_array().to_owned();

    let result = SkyResult {
        ldown: pure_result.ldown.into_pyarray(py).unbind(),
        lside: pure_result.lside.into_pyarray(py).unbind(),
        lside_sky: pure_result.lside_sky.into_pyarray(py).unbind(),
        lside_veg: pure_result.lside_veg.into_pyarray(py).unbind(),
        lside_sh: pure_result.lside_sh.into_pyarray(py).unbind(),
        lside_sun: pure_result.lside_sun.into_pyarray(py).unbind(),
        lside_ref: pure_result.lside_ref.into_pyarray(py).unbind(),
        least: pure_result.least.into_pyarray(py).unbind(),
        lwest: pure_result.lwest.into_pyarray(py).unbind(),
        lnorth: pure_result.lnorth.into_pyarray(py).unbind(),
        lsouth: pure_result.lsouth.into_pyarray(py).unbind(),
        keast: pure_result.keast.into_pyarray(py).unbind(),
        ksouth: pure_result.ksouth.into_pyarray(py).unbind(),
        kwest: pure_result.kwest.into_pyarray(py).unbind(),
        knorth: pure_result.knorth.into_pyarray(py).unbind(),
        kside_i: pure_result.kside_i.into_pyarray(py).unbind(),
        kside_d: pure_result.kside_d.into_pyarray(py).unbind(),
        kside: pure_result.kside.into_pyarray(py).unbind(),
        steradians: steradians_owned.into_pyarray(py).unbind(),
        skyalt: skyalt.into_pyarray(py).unbind(),
    };
    Py::new(py, result)
}

/// Per-pixel cylindric wedge shadow fraction calculation.
///
/// Computes F_sh for a single pixel given pre-computed tan(zenith) and the
/// SVF-weighted building angle for that pixel.
#[allow(non_snake_case)]
fn cylindric_wedge_pixel(tan_zen: f32, svfalfa_val: f32) -> f32 {
    let tan_alfa = svfalfa_val.tan().max(1e-6);
    let ba = 1.0 / tan_alfa;
    let tan_product = (tan_alfa * tan_zen).max(1e-6);

    let xa = 1.0 - 2.0 / tan_product;
    let ha = 2.0 / tan_product;
    let hkil = 2.0 * ba * ha;

    let ukil = if xa < 0.0 {
        let qa = tan_zen / 2.0;
        let za = (ba * ba - qa * qa / 4.0).max(0.0).sqrt();
        let phi = (za / qa.max(1e-10)).atan();
        let cos_phi = phi.cos();
        let sin_phi = phi.sin();
        let denom = (1.0 - cos_phi).max(1e-10);
        let a = (sin_phi - phi * cos_phi) / denom;
        2.0 * ba * xa * a
    } else {
        0.0
    };

    let s_surf = hkil + ukil;
    (2.0 * PI * ba - s_surf) / (2.0 * PI * ba)
}

/// Pure-ndarray implementation of cylindric wedge shadow fraction.
/// Callable from pipeline.rs (fused path) or from the PyO3 wrapper (modular path).
#[allow(non_snake_case)]
pub(crate) fn cylindric_wedge_pure(zen: f32, svfalfa: ArrayView2<f32>) -> Array2<f32> {
    cylindric_wedge_pure_masked(zen, svfalfa, None)
}

pub(crate) fn cylindric_wedge_pure_masked(
    zen: f32,
    svfalfa: ArrayView2<f32>,
    valid: Option<ArrayView2<u8>>,
) -> Array2<f32> {
    let rows = svfalfa.shape()[0];
    let cols = svfalfa.shape()[1];

    // Guard against low sun angles where tan(zen) → infinity
    let altitude_rad = PI / 2.0 - zen;
    if altitude_rad < MIN_SUN_ELEVATION_RAD {
        return Array2::<f32>::ones((rows, cols));
    }

    let tan_zen = zen.tan();

    let pixel_results: Vec<f32> = (0..rows * cols)
        .into_par_iter()
        .map(|idx| {
            let r = idx / cols;
            let c = idx % cols;
            if let Some(ref v) = valid {
                if v[[r, c]] == 0 {
                    return f32::NAN;
                }
            }
            cylindric_wedge_pixel(tan_zen, svfalfa[[r, c]])
        })
        .collect();

    Array2::from_shape_vec((rows, cols), pixel_results).unwrap()
}

/// Fraction of sunlit walls based on sun altitude and SVF-weighted building angles.
///
/// Args:
///     zen: Sun zenith angle (radians, scalar)
///     svfalfa: SVF-related angle grid (2D array, radians)
///
/// Returns:
///     F_sh: Shadow fraction grid (0 = fully sunlit, 1 = fully shaded)
///
/// At very low sun altitudes (< 3°), returns F_sh = 1.0 to avoid
/// numerical instability from tan(zen) approaching infinity.
#[pyfunction]
#[allow(non_snake_case)]
pub fn cylindric_wedge(
    py: Python,
    zen: f32,
    svfalfa: PyReadonlyArray2<f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    let result = cylindric_wedge_pure(zen, svfalfa.as_array());
    Ok(result.into_pyarray(py).unbind())
}

/// Weighted sum over the patch dimension of a 3D array.
///
/// Computes: result[r, c] = sum_i(patches[r, c, i] * weights[i])
///
/// This replaces the Python loop:
///   for idx in range(n_patches):
///       ani_lum += diffsh[:,:,idx] * lv[idx, 2]
///
/// Args:
///     patches: 3D array (rows, cols, n_patches) - e.g. diffuse shadow matrix
///     weights: 1D array (n_patches,) - e.g. Perez luminance weights
///
/// Returns:
///     2D array (rows, cols) - weighted sum
#[pyfunction]
pub fn weighted_patch_sum(
    py: Python,
    patches: PyReadonlyArray3<f32>,
    weights: PyReadonlyArray1<f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    let patches = patches.as_array();
    let weights = weights.as_array();
    let rows = patches.shape()[0];
    let cols = patches.shape()[1];
    let n_patches = patches.shape()[2];

    let pixel_results: Vec<f32> = (0..rows * cols)
        .into_par_iter()
        .map(|idx| {
            let r = idx / cols;
            let c = idx % cols;
            let mut sum = 0.0f32;
            for i in 0..n_patches {
                sum += patches[[r, c, i]] * weights[[i]];
            }
            sum
        })
        .collect();

    let result = Array2::from_shape_vec((rows, cols), pixel_results)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(result.into_pyarray(py).unbind())
}
