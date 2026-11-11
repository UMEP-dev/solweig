//! Plain-Rust inputs for `pipeline::compute_timestep_rust`.
//!
//! The Python binding (`pipeline::compute_timestep`) borrows numpy arrays and
//! fills these structs with views; a Rust caller fills them from its own
//! `ndarray` arrays. Nothing here depends on PyO3 except the three scalar
//! structs, which are shared with the Python side.
//!
//! Every raster must have the DSM's shape. The shadow matrices are bitpacked:
//! shape `(rows, cols, ceil(n_patches / 8))`, bit `p & 7` of byte `p >> 3`
//! set when patch `p` is visible (as returned in `SvfIntermediate`).

use ndarray::{ArrayView2, ArrayView3};

use crate::gvf_geometry::GvfGeometryCache;
use crate::pipeline_scalars::{ConfigScalars, HumanScalars, WeatherScalars};

/// Surface rasters. Everything but `dsm` is optional and ignored unless the
/// matching `ConfigScalars` flag (`use_veg`, `has_walls`) is set.
pub struct SurfaceViews<'a> {
    pub dsm: ArrayView2<'a, f32>,
    pub cdsm: Option<ArrayView2<'a, f32>>,
    pub tdsm: Option<ArrayView2<'a, f32>>,
    pub bush: Option<ArrayView2<'a, f32>>,
    pub wall_ht: Option<ArrayView2<'a, f32>>,
    /// Wall aspect in degrees.
    pub wall_asp: Option<ArrayView2<'a, f32>>,
}

/// Sky view factor rasters (constant across timesteps for a surface).
pub struct SvfViews<'a> {
    pub svf: ArrayView2<'a, f32>,
    pub svf_n: ArrayView2<'a, f32>,
    pub svf_e: ArrayView2<'a, f32>,
    pub svf_s: ArrayView2<'a, f32>,
    pub svf_w: ArrayView2<'a, f32>,
    pub svf_veg: ArrayView2<'a, f32>,
    pub svf_veg_n: ArrayView2<'a, f32>,
    pub svf_veg_e: ArrayView2<'a, f32>,
    pub svf_veg_s: ArrayView2<'a, f32>,
    pub svf_veg_w: ArrayView2<'a, f32>,
    pub svf_aveg: ArrayView2<'a, f32>,
    pub svf_aveg_n: ArrayView2<'a, f32>,
    pub svf_aveg_e: ArrayView2<'a, f32>,
    pub svf_aveg_s: ArrayView2<'a, f32>,
    pub svf_aveg_w: ArrayView2<'a, f32>,
    /// Combined building + vegetation SVF, adjusted for transmissivity.
    pub svfbuveg: ArrayView2<'a, f32>,
    /// Building height angle derived from SVF (radians).
    pub svfalfa: ArrayView2<'a, f32>,
}

/// Per-pixel land-cover properties.
pub struct PropertyViews<'a> {
    pub alb_grid: ArrayView2<'a, f32>,
    pub emis_grid: ArrayView2<'a, f32>,
    pub tgk_grid: ArrayView2<'a, f32>,
    pub tstart_grid: ArrayView2<'a, f32>,
    pub tmaxlst_grid: ArrayView2<'a, f32>,
}

/// Thermal state carried between timesteps. Feed the matching fields of the
/// previous `TimestepResultRaw` back in.
pub struct ThermalStateViews<'a> {
    pub firstdaytime: i32,
    pub timeadd: f32,
    /// Timestep as a fraction of a day.
    pub timestep_dec: f32,
    pub tgmap1: ArrayView2<'a, f32>,
    pub tgmap1_e: ArrayView2<'a, f32>,
    pub tgmap1_s: ArrayView2<'a, f32>,
    pub tgmap1_w: ArrayView2<'a, f32>,
    pub tgmap1_n: ArrayView2<'a, f32>,
    pub tgout1: ArrayView2<'a, f32>,
}

/// Inputs and carried state for the opt-in UMEP 2026a ground scheme. Its
/// presence in [`TimestepInputs`] switches the scheme on.
pub struct GroundSchemeViews<'a> {
    /// Timestep in seconds.
    pub timestep_s: f32,
    pub tg: ArrayView2<'a, f32>,
    pub tm: ArrayView2<'a, f32>,
    pub rn: ArrayView2<'a, f32>,
    pub rn_past: ArrayView2<'a, f32>,
    pub g: ArrayView2<'a, f32>,
    pub cap: ArrayView2<'a, f32>,
    pub diff: ArrayView2<'a, f32>,
    pub a1: ArrayView2<'a, f32>,
    pub a2: ArrayView2<'a, f32>,
    pub a3: ArrayView2<'a, f32>,
    pub lc_grid: ArrayView2<'a, f32>,
    /// The previous timestep's returned `shadow`.
    pub shadow_past: ArrayView2<'a, f32>,
}

/// Everything one timestep needs.
pub struct TimestepInputs<'a> {
    pub weather: &'a WeatherScalars,
    pub human: &'a HumanScalars,
    pub config: &'a ConfigScalars,
    /// Building ray-trace geometry from `precompute_gvf_geometry_cpu`, reused
    /// across the timesteps of one surface. `None` recomputes it every step.
    pub gvf_cache: Option<&'a GvfGeometryCache>,
    pub surface: SurfaceViews<'a>,
    pub svf: SvfViews<'a>,
    pub properties: PropertyViews<'a>,
    /// 1 where the pixel is ground, 0 under a building.
    pub buildings: ArrayView2<'a, f32>,
    pub lc_grid: Option<ArrayView2<'a, f32>>,
    /// Anisotropic sky shadow matrices; all three or none.
    pub shmat: Option<ArrayView3<'a, u8>>,
    pub vegshmat: Option<ArrayView3<'a, u8>>,
    pub vbshmat: Option<ArrayView3<'a, u8>>,
    pub state: ThermalStateViews<'a>,
    pub ground_scheme: Option<GroundSchemeViews<'a>>,
    /// 1 = compute this pixel, 0 = nodata.
    pub valid_mask: ArrayView2<'a, u8>,
    /// Which optional outputs to return: an OR of the `OUT_*` flags.
    /// `tmrt` and the thermal state are always returned.
    pub output_mask: u32,
}
