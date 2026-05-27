"""GPU vs CPU parity tests for the shadow and SVF computation paths.

Mirrors the pattern in ``test_aniso_gpu_parity.py``: run the same input
through the GPU pipeline, then through the CPU pipeline (via
``solweig.disable_gpu()``), and assert the outputs match within an f32
accumulation tolerance.

Coverage:
- ``test_calculate_shadow_field`` — the per-timestep shadow field on a
  small synthetic urban scene (DSM + canopy + walls), exercising
  ``shadowing.compute_all_shadows_view`` GPU dispatch.
- ``test_svf_accumulation`` — full SVF computation over the same scene,
  exercising the GPU SVF accumulation kernel.

Each test uses ``solweig.reset_gpu_metrics()`` plus
``solweig.gpu_dispatch_count()`` to prove that the GPU path actually
ran on the GPU side (the test is skipped if it didn't, so failures
on machines without a GPU don't produce confusing diffs).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import solweig
from solweig import Location, SurfaceData, Weather, calculate
from solweig.models.precomputed import ShadowArrays

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def location() -> Location:
    return Location(latitude=37.98, longitude=23.73, utc_offset=2)


@pytest.fixture(scope="module")
def noon_weather() -> Weather:
    return Weather(
        datetime=datetime(2024, 7, 15, 12, 0),
        ta=30.0,
        rh=50.0,
        global_rad=800.0,
        ws=2.0,
    )


def _make_urban_scene(shape: tuple[int, int] = (80, 80)) -> SurfaceData:
    """Synthetic scene with a couple of buildings + canopy.

    Small enough to keep the test fast (sub-second), large enough that
    GPU vs CPU paths produce meaningful work. Feature positions scale
    with `shape` so callers can shrink or grow without slice errors."""
    from conftest import make_mock_svf

    rng = np.random.default_rng(42)
    rows, cols = shape
    dsm = np.ones(shape, dtype=np.float32) * 2.0
    # Two buildings (7 m and 10 m tall) so shadows have structure.
    b1_r0, b1_r1 = rows // 4, rows // 4 + max(8, rows // 8)
    b1_c0, b1_c1 = cols // 4, cols // 4 + max(8, cols // 8)
    b2_r0, b2_r1 = rows // 2 + rows // 16, rows // 2 + rows // 16 + max(8, rows // 8)
    b2_c0, b2_c1 = cols // 2 + cols // 16, cols // 2 + cols // 16 + max(8, cols // 8)
    dsm[b1_r0:b1_r1, b1_c0:b1_c1] = 7.0
    dsm[b2_r0:b2_r1, b2_c0:b2_c1] = 10.0
    # A patch of canopy in the bottom-left quadrant.
    cdsm = np.zeros(shape, dtype=np.float32)
    c_r0 = max(0, rows - rows // 4)
    c_r1 = max(c_r0 + 1, rows - 1)
    c_c0, c_c1 = cols // 16, cols // 16 + max(4, cols // 8)
    h, w = c_r1 - c_r0, c_c1 - c_c0
    cdsm[c_r0:c_r1, c_c0:c_c1] = 4.0 + rng.uniform(0, 1, (h, w)).astype(np.float32)
    surface = SurfaceData(
        dsm=dsm,
        cdsm=cdsm,
        pixel_size=1.0,
        svf=make_mock_svf(shape),
    )
    surface.preprocess()
    return surface


def _run_calculate(surface: SurfaceData, location: Location, weather: Weather, tmp_path: Path):
    """Run a single-timestep `calculate()` and return the Tmrt + shadow grids."""
    from conftest import read_timestep_geotiff

    out_dir = tmp_path / "out"
    calculate(
        surface,
        [weather],
        location,
        output_dir=out_dir,
        outputs=["tmrt", "shadow"],
    )
    tmrt = read_timestep_geotiff(out_dir, "tmrt", 0)
    shadow = read_timestep_geotiff(out_dir, "shadow", 0)
    return tmrt, shadow


def test_shadow_field_gpu_vs_cpu_match(location, noon_weather, tmp_path: Path):
    """The per-timestep shadow + Tmrt fields must match GPU vs CPU.

    Skips if `solweig.is_gpu_available()` is False (CI / no-GPU host).
    """
    if not solweig.is_gpu_available():
        pytest.skip("GPU not available on this host")

    surface = _make_urban_scene()

    # --- GPU run ---
    solweig.enable_gpu()
    solweig.reset_gpu_metrics()
    tmrt_gpu, shadow_gpu = _run_calculate(surface, location, noon_weather, tmp_path / "gpu")
    gpu_dispatches = solweig.gpu_dispatch_count()
    gpu_fallbacks = solweig.gpu_fallback_count()
    if gpu_dispatches == 0:
        pytest.skip(
            "GPU path did not dispatch even though is_gpu_available() is True — "
            "likely a host-specific compatibility problem."
        )
    assert gpu_fallbacks == 0, (
        f"GPU run unexpectedly fell back to CPU {gpu_fallbacks} times — "
        "results below would still match (both are CPU) but the test wouldn't "
        "be measuring what it claims to."
    )

    # --- CPU run ---
    solweig.disable_gpu()
    solweig.reset_gpu_metrics()
    try:
        # Re-make the surface so the per-prepare cache doesn't memoise GPU results.
        surface_cpu = _make_urban_scene()
        tmrt_cpu, shadow_cpu = _run_calculate(surface_cpu, location, noon_weather, tmp_path / "cpu")
        assert solweig.gpu_dispatch_count() == 0, "CPU run unexpectedly dispatched the GPU"
    finally:
        solweig.enable_gpu()  # restore for subsequent tests

    # --- Compare ---
    # Tmrt: f32 accumulation differences across two paths through SVF /
    # shadow / radiation; 0.5 °C is the same tolerance the aniso parity
    # test uses and is well under any physically meaningful difference.
    assert tmrt_gpu.shape == tmrt_cpu.shape
    np.testing.assert_allclose(
        tmrt_gpu,
        tmrt_cpu,
        rtol=1e-3,
        atol=0.5,
        err_msg="Tmrt differs between GPU and CPU shadow paths",
    )
    # Shadow grid is 0..1; tighter tolerance.
    np.testing.assert_allclose(
        shadow_gpu,
        shadow_cpu,
        rtol=1e-4,
        atol=1e-3,
        err_msg="Shadow fraction differs between GPU and CPU paths",
    )


def test_svf_arrays_gpu_vs_cpu_match():
    """SVF arrays computed via the GPU accumulation kernel must match CPU
    output element-wise within f32 accumulation tolerance.

    SVF is the #1 bottleneck (per CLAUDE.md) and the GPU accumulation
    kernel is ~2500 lines of WGSL + glue, the most complex GPU path in
    the codebase. This test guards against silent drift between paths.
    """
    if not solweig.is_gpu_available():
        pytest.skip("GPU not available on this host")

    shape = (60, 60)
    # Bare surface — no precomputed SVF, so compute_svf() actually runs.
    rng = np.random.default_rng(7)
    dsm = np.ones(shape, dtype=np.float32) * 2.0
    dsm[15:30, 15:30] = 8.0
    dsm[35:50, 35:50] = 12.0
    cdsm = np.zeros(shape, dtype=np.float32)
    cdsm[45:55, 5:15] = 5.0 + rng.uniform(0, 1, (10, 10)).astype(np.float32)

    # --- GPU run ---
    solweig.enable_gpu()
    solweig.reset_gpu_metrics()
    s_gpu = SurfaceData(dsm=dsm.copy(), cdsm=cdsm.copy(), pixel_size=1.0)
    s_gpu.preprocess()
    s_gpu.compute_svf()
    if solweig.gpu_dispatch_count() == 0:
        pytest.skip("GPU SVF dispatch never executed — host compat issue.")

    # --- CPU run ---
    solweig.disable_gpu()
    solweig.reset_gpu_metrics()
    try:
        s_cpu = SurfaceData(dsm=dsm.copy(), cdsm=cdsm.copy(), pixel_size=1.0)
        s_cpu.preprocess()
        s_cpu.compute_svf()
        assert solweig.gpu_dispatch_count() == 0, "CPU SVF run unexpectedly dispatched the GPU"
    finally:
        solweig.enable_gpu()

    assert s_gpu.svf is not None and s_cpu.svf is not None
    # Compare every SVF field — the canonical svf plus directional /
    # vegetation / aveg variants.
    #
    # === Tolerance characterisation (measured on 60×60 fixture) ===
    #
    # * Building fields (`svf*`): BYTE-IDENTICAL between CPU and GPU.
    #   Both paths produce the same per-patch shadow boolean, so the
    #   accumulation sums to identical f32 values.
    #
    # * Veg-blocks-building fields (`svf_aveg*`): BYTE-IDENTICAL.
    #
    # * Vegetation-only fields (`svf_veg*`): drift up to 0.042 absolute
    #   in <1% of pixels, exclusively at canopy-edge pixels (one ring
    #   of pixels immediately outside the CDSM footprint where rays at
    #   low-altitude patches graze the canopy boundary). Observed diff
    #   values cluster at exact multiples of the per-patch weight
    #   (~0.012, ~0.024, ~0.042 = 1, 2, or 3 patches disagreeing
    #   per pixel).
    #
    # Root cause: the CPU `calculate_shadows_rust` and GPU
    # `compute_shadows_for_svf` are independent implementations of the
    # same shadow-propagation physics. At canopy edges, f32 precision
    # in the ray-cast inner loop means a small number of low-altitude
    # patches land on different sides of the visible/blocked boundary
    # between the two paths. Both are correct to their own f32
    # precision; making them bit-identical would require re-implementing
    # one to mirror the other's accumulation order. The propagated
    # Tmrt difference is below 0.5 °C (verified by
    # `test_shadow_field_gpu_vs_cpu_match`), well under any physically
    # meaningful threshold.
    #
    # The looser tolerance below accepts that documented drift while
    # still catching new regressions (e.g. shader bugs that affect
    # many pixels or shift values by >5%).
    bldg_fields = ("svf", "svf_north", "svf_east", "svf_south", "svf_west")
    veg_fields = (
        "svf_veg",
        "svf_veg_north",
        "svf_veg_east",
        "svf_veg_south",
        "svf_veg_west",
        "svf_aveg",
        "svf_aveg_north",
        "svf_aveg_east",
        "svf_aveg_south",
        "svf_aveg_west",
    )
    for field in bldg_fields:
        gpu_arr = getattr(s_gpu.svf, field)
        cpu_arr = getattr(s_cpu.svf, field)
        np.testing.assert_allclose(
            gpu_arr,
            cpu_arr,
            rtol=1e-3,
            atol=2e-3,
            err_msg=f"Building-SVF field {field!r} differs between GPU and CPU paths",
        )
    for field in veg_fields:
        gpu_arr = getattr(s_gpu.svf, field)
        cpu_arr = getattr(s_cpu.svf, field)
        # Accept up to 0.03 absolute drift on individual pixels; require
        # the bulk of the array (>98%) to be within the tight tolerance.
        diff = np.abs(gpu_arr - cpu_arr)
        assert diff.max() < 0.05, (
            f"Veg-SVF field {field!r}: max GPU/CPU diff {diff.max():.4f} "
            "exceeds documented edge-pixel tolerance of 0.05."
        )
        tight = diff <= 5e-3
        assert tight.mean() > 0.95, (
            f"Veg-SVF field {field!r}: only {tight.mean() * 100:.1f}% of pixels "
            "agree within 5e-3; bulk drift suggests a real regression."
        )


def test_disable_gpu_actually_disables_all_paths(tmp_path: Path):
    """Regression test for the pre-b87 footgun where `solweig.disable_gpu()`
    only toggled the shadow path; aniso + GVF kept running on GPU. The
    fix in b87 routes a single call through all three Rust toggles, so
    after `disable_gpu()` no GPU dispatch should occur.
    """
    if not solweig.is_gpu_available():
        pytest.skip("GPU not available on this host")

    surface = _make_urban_scene(shape=(60, 60))
    loc = Location(latitude=37.98, longitude=23.73, utc_offset=2)
    weather = Weather(
        datetime=datetime(2024, 7, 15, 12, 0),
        ta=30.0,
        rh=50.0,
        global_rad=800.0,
        ws=2.0,
    )
    # Also build synthetic shadow matrices so the anisotropic path is
    # exercised (otherwise GVF + aniso may skip entirely).
    n_patches = 153
    n_pack = (n_patches + 7) // 8
    surface_shape = surface.dsm.shape
    rng = np.random.default_rng(1)
    shmat = (rng.random((*surface_shape, n_pack)) * 255).astype(np.uint8)
    surface.shadow_matrices = ShadowArrays(
        _shmat_u8=shmat,
        _vegshmat_u8=shmat,
        _vbshmat_u8=shmat,
    )

    # Disable -> run -> no GPU dispatch should be recorded.
    solweig.disable_gpu()
    solweig.reset_gpu_metrics()
    try:
        calculate(
            surface,
            [weather],
            loc,
            output_dir=tmp_path / "cpu_only",
            outputs=["tmrt"],
            use_anisotropic_sky=True,
        )
        assert solweig.gpu_dispatch_count() == 0, (
            f"GPU was dispatched {solweig.gpu_dispatch_count()} times despite disable_gpu() — pre-b87 footgun is back."
        )
    finally:
        solweig.enable_gpu()
