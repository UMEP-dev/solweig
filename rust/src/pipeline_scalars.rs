//! Input scalar structs for `pipeline::compute_timestep`.
//!
//! `WeatherScalars`, `HumanScalars`, and `ConfigScalars` are constructed in
//! Python via PyO3 `#[new]` constructors and passed by reference to
//! `compute_timestep`. Extracted here so `pipeline.rs` can focus on the
//! per-timestep orchestration.

use pyo3::prelude::*;

/// Weather scalars for a single timestep.
#[pyclass]
#[derive(Clone)]
pub struct WeatherScalars {
    #[pyo3(get, set)]
    pub sun_azimuth: f32,
    #[pyo3(get, set)]
    pub sun_altitude: f32,
    #[pyo3(get, set)]
    pub sun_zenith: f32,
    #[pyo3(get, set)]
    pub ta: f32,
    #[pyo3(get, set)]
    pub rh: f32,
    #[pyo3(get, set)]
    pub global_rad: f32,
    #[pyo3(get, set)]
    pub direct_rad: f32,
    #[pyo3(get, set)]
    pub diffuse_rad: f32,
    #[pyo3(get, set)]
    pub altmax: f32,
    #[pyo3(get, set)]
    pub clearness_index: f32,
    #[pyo3(get, set)]
    pub dectime: f32,
    #[pyo3(get, set)]
    pub snup: f32,
    #[pyo3(get, set)]
    pub rad_g0: f32,
    #[pyo3(get, set)]
    pub zen_deg: f32,
    #[pyo3(get, set)]
    pub psi: f32,
    #[pyo3(get, set)]
    pub is_daytime: bool,
    #[pyo3(get, set)]
    pub jday: i32,
    #[pyo3(get, set)]
    pub patch_option: i32,
}

#[pymethods]
impl WeatherScalars {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sun_azimuth: f32,
        sun_altitude: f32,
        sun_zenith: f32,
        ta: f32,
        rh: f32,
        global_rad: f32,
        direct_rad: f32,
        diffuse_rad: f32,
        altmax: f32,
        clearness_index: f32,
        dectime: f32,
        snup: f32,
        rad_g0: f32,
        zen_deg: f32,
        psi: f32,
        is_daytime: bool,
        jday: i32,
        patch_option: i32,
    ) -> Self {
        Self {
            sun_azimuth,
            sun_altitude,
            sun_zenith,
            ta,
            rh,
            global_rad,
            direct_rad,
            diffuse_rad,
            altmax,
            clearness_index,
            dectime,
            snup,
            rad_g0,
            zen_deg,
            psi,
            is_daytime,
            jday,
            patch_option,
        }
    }
}

/// Human body parameters.
#[pyclass]
#[derive(Clone)]
pub struct HumanScalars {
    #[pyo3(get, set)]
    pub height: f32,
    #[pyo3(get, set)]
    pub abs_k: f32,
    #[pyo3(get, set)]
    pub abs_l: f32,
    #[pyo3(get, set)]
    pub is_standing: bool,
}

#[pymethods]
impl HumanScalars {
    #[new]
    pub fn new(height: f32, abs_k: f32, abs_l: f32, is_standing: bool) -> Self {
        Self {
            height,
            abs_k,
            abs_l,
            is_standing,
        }
    }
}

/// Configuration scalars (constant across timesteps).
#[pyclass]
#[derive(Clone)]
pub struct ConfigScalars {
    #[pyo3(get, set)]
    pub pixel_size: f32,
    #[pyo3(get, set)]
    pub max_height: f32,
    #[pyo3(get, set)]
    pub albedo_wall: f32,
    #[pyo3(get, set)]
    pub emis_wall: f32,
    #[pyo3(get, set)]
    pub tgk_wall: f32,
    #[pyo3(get, set)]
    pub tstart_wall: f32,
    #[pyo3(get, set)]
    pub tmaxlst_wall: f32,
    #[pyo3(get, set)]
    pub use_veg: bool,
    #[pyo3(get, set)]
    pub has_walls: bool,
    #[pyo3(get, set)]
    pub conifer: bool,
    #[pyo3(get, set)]
    pub use_anisotropic: bool,
    #[pyo3(get, set)]
    pub max_shadow_distance_m: f32,
}

#[pymethods]
impl ConfigScalars {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pixel_size: f32,
        max_height: f32,
        albedo_wall: f32,
        emis_wall: f32,
        tgk_wall: f32,
        tstart_wall: f32,
        tmaxlst_wall: f32,
        use_veg: bool,
        has_walls: bool,
        conifer: bool,
        use_anisotropic: bool,
        max_shadow_distance_m: f32,
    ) -> Self {
        Self {
            pixel_size,
            max_height,
            albedo_wall,
            emis_wall,
            tgk_wall,
            tstart_wall,
            tmaxlst_wall,
            use_veg,
            has_walls,
            conifer,
            use_anisotropic,
            max_shadow_distance_m,
        }
    }
}
