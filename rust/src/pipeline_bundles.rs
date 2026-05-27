//! FFI argument bundles for the fused Rust pipeline.
//!
//! Each bundle groups a cluster of `compute_timestep` arguments into one
//! Python-visible class. The bundles are:
//!
//! - [`SurfaceBundle`] — 6 surface rasters (DSM + 5 optional auxiliaries)
//! - [`SvfBundle`]     — 17 SVF / SVF-veg / SVF-aveg rasters
//! - [`PropertiesBundle`] — 5 land-cover-derived property rasters
//! - [`StateBundle`]   — 9 thermal-state fields + an FFI version field
//!
//! Bundling cut `compute_timestep`'s argument count from 43 to 14. Adding
//! a new per-pixel field for an existing concern is now a one-place change
//! (the bundle's struct + constructor) rather than a five-place change
//! across Python and Rust call sites.
//!
//! All fields are `pub(crate)` so `compute_timestep` (in `pipeline.rs`) can
//! read them directly. They are not exposed to Python beyond the `#[new]`
//! constructor on each pyclass.

use pyo3::prelude::*;

// ── SVF bundle ──────────────────────────────────────────────────────────────

/// Bundle of the 17 SVF / SVF-veg / SVF-aveg / svfbuveg / svfalfa rasters.
///
/// SVF arrays are constant across all timesteps for a given surface, so the
/// bundle is constructed once and reused.
#[pyclass]
pub struct SvfBundle {
    pub(crate) svf: Py<numpy::PyArray2<f32>>,
    pub(crate) svf_n: Py<numpy::PyArray2<f32>>,
    pub(crate) svf_e: Py<numpy::PyArray2<f32>>,
    pub(crate) svf_s: Py<numpy::PyArray2<f32>>,
    pub(crate) svf_w: Py<numpy::PyArray2<f32>>,
    pub(crate) svf_veg: Py<numpy::PyArray2<f32>>,
    pub(crate) svf_veg_n: Py<numpy::PyArray2<f32>>,
    pub(crate) svf_veg_e: Py<numpy::PyArray2<f32>>,
    pub(crate) svf_veg_s: Py<numpy::PyArray2<f32>>,
    pub(crate) svf_veg_w: Py<numpy::PyArray2<f32>>,
    pub(crate) svf_aveg: Py<numpy::PyArray2<f32>>,
    pub(crate) svf_aveg_n: Py<numpy::PyArray2<f32>>,
    pub(crate) svf_aveg_e: Py<numpy::PyArray2<f32>>,
    pub(crate) svf_aveg_s: Py<numpy::PyArray2<f32>>,
    pub(crate) svf_aveg_w: Py<numpy::PyArray2<f32>>,
    pub(crate) svfbuveg: Py<numpy::PyArray2<f32>>,
    pub(crate) svfalfa: Py<numpy::PyArray2<f32>>,
}

#[pymethods]
impl SvfBundle {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        svf: Py<numpy::PyArray2<f32>>,
        svf_n: Py<numpy::PyArray2<f32>>,
        svf_e: Py<numpy::PyArray2<f32>>,
        svf_s: Py<numpy::PyArray2<f32>>,
        svf_w: Py<numpy::PyArray2<f32>>,
        svf_veg: Py<numpy::PyArray2<f32>>,
        svf_veg_n: Py<numpy::PyArray2<f32>>,
        svf_veg_e: Py<numpy::PyArray2<f32>>,
        svf_veg_s: Py<numpy::PyArray2<f32>>,
        svf_veg_w: Py<numpy::PyArray2<f32>>,
        svf_aveg: Py<numpy::PyArray2<f32>>,
        svf_aveg_n: Py<numpy::PyArray2<f32>>,
        svf_aveg_e: Py<numpy::PyArray2<f32>>,
        svf_aveg_s: Py<numpy::PyArray2<f32>>,
        svf_aveg_w: Py<numpy::PyArray2<f32>>,
        svfbuveg: Py<numpy::PyArray2<f32>>,
        svfalfa: Py<numpy::PyArray2<f32>>,
    ) -> Self {
        Self {
            svf,
            svf_n,
            svf_e,
            svf_s,
            svf_w,
            svf_veg,
            svf_veg_n,
            svf_veg_e,
            svf_veg_s,
            svf_veg_w,
            svf_aveg,
            svf_aveg_n,
            svf_aveg_e,
            svf_aveg_s,
            svf_aveg_w,
            svfbuveg,
            svfalfa,
        }
    }
}

// ── Surface bundle ─────────────────────────────────────────────────────────

/// Bundle of the 6 surface rasters (DSM + 5 optional auxiliaries).
///
/// `dsm` is required; everything else is optional and absent when the
/// caller doesn't need vegetation or wall computation.
#[pyclass]
pub struct SurfaceBundle {
    pub(crate) dsm: Py<numpy::PyArray2<f32>>,
    pub(crate) cdsm: Option<Py<numpy::PyArray2<f32>>>,
    pub(crate) tdsm: Option<Py<numpy::PyArray2<f32>>>,
    pub(crate) bush: Option<Py<numpy::PyArray2<f32>>>,
    pub(crate) wall_ht: Option<Py<numpy::PyArray2<f32>>>,
    pub(crate) wall_asp: Option<Py<numpy::PyArray2<f32>>>,
}

#[pymethods]
impl SurfaceBundle {
    #[new]
    #[pyo3(signature = (dsm, cdsm=None, tdsm=None, bush=None, wall_ht=None, wall_asp=None))]
    fn new(
        dsm: Py<numpy::PyArray2<f32>>,
        cdsm: Option<Py<numpy::PyArray2<f32>>>,
        tdsm: Option<Py<numpy::PyArray2<f32>>>,
        bush: Option<Py<numpy::PyArray2<f32>>>,
        wall_ht: Option<Py<numpy::PyArray2<f32>>>,
        wall_asp: Option<Py<numpy::PyArray2<f32>>>,
    ) -> Self {
        Self {
            dsm,
            cdsm,
            tdsm,
            bush,
            wall_ht,
            wall_asp,
        }
    }
}

// ── Properties bundle ──────────────────────────────────────────────────────

/// Bundle of the 5 land-cover-derived property rasters.
///
/// Each per-pixel property grid is computed from the land-cover class via
/// the materials JSON lookup. They are constant across timesteps for a
/// given surface + materials pair.
#[pyclass]
pub struct PropertiesBundle {
    pub(crate) alb_grid: Py<numpy::PyArray2<f32>>,
    pub(crate) emis_grid: Py<numpy::PyArray2<f32>>,
    pub(crate) tgk_grid: Py<numpy::PyArray2<f32>>,
    pub(crate) tstart_grid: Py<numpy::PyArray2<f32>>,
    pub(crate) tmaxlst_grid: Py<numpy::PyArray2<f32>>,
}

#[pymethods]
impl PropertiesBundle {
    #[new]
    fn new(
        alb_grid: Py<numpy::PyArray2<f32>>,
        emis_grid: Py<numpy::PyArray2<f32>>,
        tgk_grid: Py<numpy::PyArray2<f32>>,
        tstart_grid: Py<numpy::PyArray2<f32>>,
        tmaxlst_grid: Py<numpy::PyArray2<f32>>,
    ) -> Self {
        Self {
            alb_grid,
            emis_grid,
            tgk_grid,
            tstart_grid,
            tmaxlst_grid,
        }
    }
}

// ── State bundle ────────────────────────────────────────────────────────────

/// FFI version constant for the StateBundle protocol.
///
/// Increment when the bundle's field layout changes in a way that breaks
/// callers compiled against an older Rust extension. Python side asserts
/// match before constructing the bundle (see
/// `pysrc/solweig/models/state.py::ThermalState.STATE_BUNDLE_VERSION`).
pub const STATE_BUNDLE_VERSION: u32 = 1;

/// Thermal state carried forward across timesteps.
///
/// Combines the 6 thermal arrays (tgmap1 + cardinal directions, tgout1)
/// and 3 scalars (firstdaytime, timeadd, timestep_dec). Also carries a
/// `version: u32` field — the Python side fails fast on mismatch instead
/// of silently mis-mapping fields.
#[pyclass]
pub struct StateBundle {
    version: u32,
    pub(crate) firstdaytime: i32,
    pub(crate) timeadd: f32,
    pub(crate) timestep_dec: f32,
    pub(crate) tgmap1: Py<numpy::PyArray2<f32>>,
    pub(crate) tgmap1_e: Py<numpy::PyArray2<f32>>,
    pub(crate) tgmap1_s: Py<numpy::PyArray2<f32>>,
    pub(crate) tgmap1_w: Py<numpy::PyArray2<f32>>,
    pub(crate) tgmap1_n: Py<numpy::PyArray2<f32>>,
    pub(crate) tgout1: Py<numpy::PyArray2<f32>>,
}

#[pymethods]
impl StateBundle {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        version: u32,
        firstdaytime: i32,
        timeadd: f32,
        timestep_dec: f32,
        tgmap1: Py<numpy::PyArray2<f32>>,
        tgmap1_e: Py<numpy::PyArray2<f32>>,
        tgmap1_s: Py<numpy::PyArray2<f32>>,
        tgmap1_w: Py<numpy::PyArray2<f32>>,
        tgmap1_n: Py<numpy::PyArray2<f32>>,
        tgout1: Py<numpy::PyArray2<f32>>,
    ) -> PyResult<Self> {
        if version != STATE_BUNDLE_VERSION {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "StateBundle version mismatch: Python sent {version}, Rust expects {STATE_BUNDLE_VERSION}. \
                 Rebuild the Rust extension (`maturin develop --release`)."
            )));
        }
        Ok(Self {
            version,
            firstdaytime,
            timeadd,
            timestep_dec,
            tgmap1,
            tgmap1_e,
            tgmap1_s,
            tgmap1_w,
            tgmap1_n,
            tgout1,
        })
    }

    /// Expose the version for debugging / tests.
    #[getter]
    fn version(&self) -> u32 {
        self.version
    }
}
