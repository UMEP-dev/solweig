use pyo3::prelude::*;

mod emissivity_models;
#[cfg(feature = "gpu")]
mod gpu;
mod ground;
mod gvf;
mod gvf_geometry;
mod morphology;
mod patch_radiation;
mod perez;
mod pet;
mod pipeline;
mod shadowing;
mod sky;
mod skyview;
mod sun;
mod sunlit_shaded_patches;
mod tmrt;
mod utci;
mod vegetation;
mod wall_aspect;

#[pymodule]
fn rustalgos(py_module: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register classes and functions
    // py_module.add_class::<common::Coord>()?;
    // py_module.add_function(wrap_pyfunction!(common::clipped_beta_wt, py_module)?)?;

    // Register submodules
    register_shadowing_module(py_module)?;
    register_skyview_module(py_module)?;
    register_gvf_module(py_module)?;
    register_sky_module(py_module)?;
    register_vegetation_module(py_module)?;
    register_utci_module(py_module)?;
    register_pet_module(py_module)?;
    register_ground_module(py_module)?;
    register_tmrt_module(py_module)?;
    register_pipeline_module(py_module)?;
    register_morphology_module(py_module)?;
    register_wall_aspect_module(py_module)?;

    // Add GPU feature flag
    #[cfg(feature = "gpu")]
    py_module.add("GPU_ENABLED", true)?;
    #[cfg(not(feature = "gpu"))]
    py_module.add("GPU_ENABLED", false)?;

    // Add build profile flag (false in debug builds, true in release builds)
    py_module.add("RELEASE_BUILD", !cfg!(debug_assertions))?;

    py_module.add(
        "__doc__",
        "SOLWEIG urban microclimate algorithms implemented in Rust.",
    )?;

    Ok(())
}

fn register_shadowing_module(py_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py_module.py(), "shadowing")?;
    submodule.add("__doc__", "Shadow analysis.")?;
    submodule.add_function(wrap_pyfunction!(
        shadowing::calculate_shadows_wall_ht_25,
        &submodule
    )?)?;

    // Add GPU control functions if GPU feature is enabled
    #[cfg(feature = "gpu")]
    {
        submodule.add_function(wrap_pyfunction!(shadowing::enable_gpu, &submodule)?)?;
        submodule.add_function(wrap_pyfunction!(shadowing::disable_gpu, &submodule)?)?;
        submodule.add_function(wrap_pyfunction!(shadowing::is_gpu_enabled, &submodule)?)?;
        submodule.add_function(wrap_pyfunction!(shadowing::gpu_limits, &submodule)?)?;
    }

    py_module.add_submodule(&submodule)?;
    Ok(())
}

fn register_skyview_module(py_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py_module.py(), "skyview")?;
    submodule.add("__doc__", "Sky View Factor calculation.")?;
    submodule.add_function(wrap_pyfunction!(skyview::calculate_svf, &submodule)?)?;
    // Expose the SkyviewRunner class so Python can create a runner and poll progress()
    submodule.add_class::<skyview::SkyviewRunner>()?;
    py_module.add_submodule(&submodule)?;
    Ok(())
}

fn register_gvf_module(py_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py_module.py(), "gvf")?;
    submodule.add("__doc__", "Ground View Factor calculation.")?;
    submodule.add_class::<gvf::GvfScalarParams>()?;
    submodule.add_function(wrap_pyfunction!(gvf::gvf_calc, &submodule)?)?;
    py_module.add_submodule(&submodule)?;
    Ok(())
}

fn register_sky_module(py_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py_module.py(), "sky")?;
    submodule.add("__doc__", "Anisotropic sky radiation calculations.")?;
    submodule.add_class::<sky::SunParams>()?;
    submodule.add_class::<sky::SkyParams>()?;
    submodule.add_class::<sky::SurfaceParams>()?;
    submodule.add_class::<sky::SkyResult>()?;
    submodule.add_function(wrap_pyfunction!(sky::anisotropic_sky, &submodule)?)?;
    submodule.add_function(wrap_pyfunction!(sky::cylindric_wedge, &submodule)?)?;
    submodule.add_function(wrap_pyfunction!(sky::weighted_patch_sum, &submodule)?)?;
    py_module.add_submodule(&submodule)?;
    Ok(())
}

fn register_vegetation_module(py_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py_module.py(), "vegetation")?;
    submodule.add("__doc__", "Vegetation-related calculations.")?;
    submodule.add_class::<vegetation::LsideVegResult>()?;
    submodule.add_class::<vegetation::KsideVegResult>()?;
    submodule.add_function(wrap_pyfunction!(vegetation::lside_veg, &submodule)?)?;
    submodule.add_function(wrap_pyfunction!(vegetation::kside_veg, &submodule)?)?;
    py_module.add_submodule(&submodule)?;
    Ok(())
}

fn register_utci_module(py_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py_module.py(), "utci")?;
    submodule.add(
        "__doc__",
        "UTCI (Universal Thermal Climate Index) calculations.",
    )?;
    submodule.add_function(wrap_pyfunction!(utci::utci_single, &submodule)?)?;
    submodule.add_function(wrap_pyfunction!(utci::utci_grid, &submodule)?)?;
    py_module.add_submodule(&submodule)?;
    Ok(())
}

fn register_pet_module(py_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py_module.py(), "pet")?;
    submodule.add(
        "__doc__",
        "PET (Physiological Equivalent Temperature) calculations.",
    )?;
    submodule.add_function(wrap_pyfunction!(pet::pet_calculate, &submodule)?)?;
    submodule.add_function(wrap_pyfunction!(pet::pet_grid, &submodule)?)?;
    py_module.add_submodule(&submodule)?;
    Ok(())
}

fn register_ground_module(py_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py_module.py(), "ground")?;
    submodule.add(
        "__doc__",
        "Ground temperature and thermal delay calculations.",
    )?;
    submodule.add_function(wrap_pyfunction!(
        ground::compute_ground_temperature,
        &submodule
    )?)?;
    submodule.add_function(wrap_pyfunction!(ground::ts_wave_delay, &submodule)?)?;
    submodule.add_function(wrap_pyfunction!(ground::ts_wave_delay_batch, &submodule)?)?;
    submodule.add_class::<ground::TsWaveDelayBatchResult>()?;
    py_module.add_submodule(&submodule)?;
    Ok(())
}

fn register_tmrt_module(py_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py_module.py(), "tmrt")?;
    submodule.add("__doc__", "Mean Radiant Temperature (Tmrt) calculations.")?;
    submodule.add_class::<tmrt::TmrtParams>()?;
    submodule.add_function(wrap_pyfunction!(tmrt::compute_tmrt, &submodule)?)?;
    py_module.add_submodule(&submodule)?;
    Ok(())
}

fn register_pipeline_module(py_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py_module.py(), "pipeline")?;
    submodule.add(
        "__doc__",
        "Fused timestep pipeline — single FFI call per timestep.",
    )?;
    submodule.add_class::<pipeline::WeatherScalars>()?;
    submodule.add_class::<pipeline::HumanScalars>()?;
    submodule.add_class::<pipeline::ConfigScalars>()?;
    submodule.add_class::<pipeline::TimestepResult>()?;
    submodule.add_class::<pipeline::PyGvfGeometryCache>()?;
    submodule.add_class::<pipeline::SvfBundle>()?;
    submodule.add_class::<pipeline::StateBundle>()?;
    submodule.add("STATE_BUNDLE_VERSION", pipeline::STATE_BUNDLE_VERSION)?;
    submodule.add_function(wrap_pyfunction!(pipeline::compute_timestep, &submodule)?)?;
    submodule.add_function(wrap_pyfunction!(
        pipeline::precompute_gvf_cache,
        &submodule
    )?)?;
    submodule.add_function(wrap_pyfunction!(perez::perez_v3_py, &submodule)?)?;
    submodule.add_function(wrap_pyfunction!(perez::compute_steradians_py, &submodule)?)?;

    // GPU control functions
    #[cfg(feature = "gpu")]
    {
        submodule.add_function(wrap_pyfunction!(pipeline::enable_aniso_gpu, &submodule)?)?;
        submodule.add_function(wrap_pyfunction!(pipeline::disable_aniso_gpu, &submodule)?)?;
        submodule.add_function(wrap_pyfunction!(
            pipeline::is_aniso_gpu_enabled,
            &submodule
        )?)?;
        submodule.add_function(wrap_pyfunction!(pipeline::enable_gvf_gpu, &submodule)?)?;
        submodule.add_function(wrap_pyfunction!(pipeline::disable_gvf_gpu, &submodule)?)?;
        submodule.add_function(wrap_pyfunction!(
            pipeline::is_gvf_gpu_enabled,
            &submodule
        )?)?;
    }

    py_module.add_submodule(&submodule)?;
    Ok(())
}

fn register_morphology_module(py_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py_module.py(), "morphology")?;
    submodule.add("__doc__", "Morphological operations (binary dilation).")?;
    submodule.add_function(wrap_pyfunction!(morphology::binary_dilation, &submodule)?)?;
    py_module.add_submodule(&submodule)?;
    Ok(())
}

fn register_wall_aspect_module(py_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py_module.py(), "wall_aspect")?;
    submodule.add(
        "__doc__",
        "Wall aspect (orientation) detection using the Goodwin filter algorithm.",
    )?;
    submodule.add_function(wrap_pyfunction!(
        wall_aspect::compute_wall_aspect,
        &submodule
    )?)?;
    submodule.add_class::<wall_aspect::WallAspectRunner>()?;
    py_module.add_submodule(&submodule)?;
    Ok(())
}
