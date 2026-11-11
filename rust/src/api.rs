//! Rust-callable entry points.
//!
//! The crate is built first as a Python extension, and its modules are
//! private. This module re-exports the small set of functions and types that
//! other Rust code may call directly. Names listed here are kept stable
//! across releases where possible; everything else in the crate may change
//! without notice.
//!
//! - Shadows: [`calculate_shadows_rust`] returning [`ShadowingResultRust`].
//! - Sky view factor and shadow matrices: [`calculate_svf_inner`] returning
//!   [`SvfIntermediate`].
//! - UTCI for one point: [`utci_single`].
//! - One model timestep: [`compute_timestep_rust`] taking [`TimestepInputs`]
//!   and returning [`TimestepResultRaw`]. Build the optional ground view
//!   factor geometry once per surface with [`precompute_gvf_geometry_cpu`].

pub use crate::gvf_geometry::{precompute_gvf_geometry_cpu, GvfGeometryCache};
pub use crate::pipeline::{
    compute_timestep_rust, TimestepResultRaw, OUT_ALL, OUT_KDOWN, OUT_KUP, OUT_LDOWN, OUT_LUP,
    OUT_SHADOW,
};
pub use crate::pipeline_inputs::{
    GroundSchemeViews, PropertyViews, SurfaceViews, SvfViews, ThermalStateViews, TimestepInputs,
};
pub use crate::pipeline_scalars::{ConfigScalars, HumanScalars, WeatherScalars};
pub use crate::shadowing::{calculate_shadows_rust, ShadowingResultRust};
pub use crate::skyview::{calculate_svf_inner, SvfIntermediate};
pub use crate::utci::utci_single;
