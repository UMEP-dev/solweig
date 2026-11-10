"""GPU static inputs must follow array content, not array address.

The shadow and anisotropic-sky GPU paths skip re-uploading "static" inputs
(DSM, canopy, walls, shadow matrices) when they look unchanged since the last
dispatch. Up to 0.1.0b97 "unchanged" meant "same address and shape" for the
shadow path and "same address plus a handful of sampled values" for the
anisotropic path. Two things broke that:

- a new surface whose arrays land on the address of a freed one (routine in a
  scenario loop) was cast from the previous surface's heights, with Tmrt off
  by up to 25 degC over the mis-shadowed pixels;
- a GPU SVF run for another surface overwrote the shared DSM buffer without
  invalidating the record, so the next timestep of the first surface used the
  other surface's DSM.

Address reuse is made deterministic here by rewriting one buffer in place and
building a new surface on it.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
import solweig
from conftest import make_mock_svf
from solweig import Location, SurfaceData, Weather
from solweig.api import _calculate_single
from solweig.models import ThermalState
from solweig.models.shadow_arrays import ShadowArrays
from solweig.timeseries import _precompute_weather

SHAPE = (48, 48)
N_PATCHES = 153
N_PACK = (N_PATCHES + 7) // 8
LOCATION = Location(latitude=57.7, longitude=12.0, utc_offset=1)

pytestmark = pytest.mark.skipif(not solweig.is_gpu_available(), reason="needs a GPU")


def _dsm(layout: str) -> np.ndarray:
    dsm = np.zeros(SHAPE, dtype=np.float32)
    if layout == "a":
        dsm[8:16, 8:16] = 12.0
    else:
        dsm[28:40, 20:30] = 20.0
    return dsm


def _run(surface, *, aniso=False):
    weather = Weather(datetime=datetime(2024, 7, 15, 15, 0), ta=25.0, rh=50.0, global_rad=700.0)
    _precompute_weather([weather], LOCATION)
    state = ThermalState.initial(surface.shape)
    state.timestep_dec = 1.0 / 24.0
    result = _calculate_single(
        surface=surface,
        location=LOCATION,
        weather=weather,
        use_anisotropic_sky=aniso,
        state=state,
        return_state_copy=False,
        _requested_outputs={"tmrt", "shadow"},
    )
    return np.asarray(result.shadow).copy(), np.asarray(result.tmrt).copy()


def _cpu(surface, *, aniso=False):
    solweig.disable_gpu()
    try:
        return _run(surface, aniso=aniso)
    finally:
        solweig.enable_gpu()


@pytest.fixture(autouse=True)
def _gpu_on():
    solweig.enable_gpu()
    yield
    solweig.enable_gpu()


def test_new_surface_on_a_reused_dsm_address_gets_its_own_shadows():
    buffer = _dsm("a")
    _run(SurfaceData(dsm=buffer, pixel_size=1.0, svf=make_mock_svf(SHAPE)))

    buffer[:] = _dsm("b")  # same address, different city
    shadow_gpu, tmrt_gpu = _run(SurfaceData(dsm=buffer, pixel_size=1.0, svf=make_mock_svf(SHAPE)))

    shadow_cpu, tmrt_cpu = _cpu(SurfaceData(dsm=_dsm("b"), pixel_size=1.0, svf=make_mock_svf(SHAPE)))
    np.testing.assert_array_equal(shadow_gpu, shadow_cpu)
    np.testing.assert_allclose(tmrt_gpu, tmrt_cpu, atol=0.05)


def test_alternating_between_two_surfaces_keeps_each_ones_ground_view_geometry():
    """The GPU holds one surface's GVF geometry at a time and must swap it on demand.

    Up to 0.1.0b97 the resident geometry was matched by grid shape only, so
    after a step on surface B the next step on surface A used B's building
    geometry (Kup and Lup off by 55 to 70 W/m2 over most of the grid).
    """
    from solweig.physics.wallalgorithms import filter1Goodwin_as_aspect_v3, findwalls

    def walled(layout):
        dsm = _dsm(layout)
        walls = findwalls(dsm, 1.0)
        aspect = filter1Goodwin_as_aspect_v3(walls, 1.0, dsm)
        return SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf(SHAPE), wall_height=walls, wall_aspect=aspect)

    surface_a, surface_b = walled("a"), walled("b")
    _run(surface_a)
    _run(surface_b)
    _, tmrt_a_gpu = _run(surface_a)  # B's geometry is resident at this point
    _, tmrt_b_gpu = _run(surface_b)

    _, tmrt_a_cpu = _cpu(walled("a"))
    _, tmrt_b_cpu = _cpu(walled("b"))
    np.testing.assert_allclose(tmrt_a_gpu, tmrt_a_cpu, atol=0.05)
    np.testing.assert_allclose(tmrt_b_gpu, tmrt_b_cpu, atol=0.05)


def test_gpu_svf_for_another_surface_does_not_leak_into_the_next_timestep():
    surface_a = SurfaceData(dsm=_dsm("a"), pixel_size=1.0, svf=make_mock_svf(SHAPE))
    shadow_first, tmrt_first = _run(surface_a)

    SurfaceData(dsm=_dsm("b"), pixel_size=1.0).compute_svf()  # writes the shared DSM buffer

    shadow_again, tmrt_again = _run(surface_a)
    np.testing.assert_array_equal(shadow_again, shadow_first)
    np.testing.assert_allclose(tmrt_again, tmrt_first, atol=1e-4)


def test_new_shadow_matrices_at_the_old_addresses_are_uploaded_again():
    """Anisotropic path: every static array keeps its address, one has new content.

    The anisotropic record compares addresses and a few sampled values. The
    two matrices here agree at every sampled position (corners, centre and
    quarter points), so only the change of ShadowArrays object tells the GPU
    path to upload again.
    """

    def matrices(shmat_u8):
        return ShadowArrays(
            _shmat_u8=shmat_u8,
            _vegshmat_u8=np.full_like(shmat_u8, 0xFF),
            _vbshmat_u8=np.full_like(shmat_u8, 0xFF),
            _n_patches=N_PATCHES,
        )

    def surface(shadow_matrices):
        return SurfaceData(dsm=_dsm("a"), pixel_size=1.0, svf=make_mock_svf(SHAPE), shadow_matrices=shadow_matrices)

    open_sky = np.full((SHAPE[0], SHAPE[1], N_PACK), 0xFF, dtype=np.uint8)
    half_blocked = open_sky.copy()
    half_blocked[14:22, 30:38, 1:10] = 0  # an interior block sees buildings on most patches

    buffer = open_sky.copy()
    reused = surface(matrices(buffer))
    _run(reused, aniso=True)

    buffer[:] = half_blocked  # same address, different matrices
    reused.shadow_matrices = matrices(buffer)
    _, tmrt_gpu = _run(reused, aniso=True)

    _, tmrt_cpu = _cpu(surface(matrices(half_blocked.copy())), aniso=True)
    np.testing.assert_allclose(tmrt_gpu, tmrt_cpu, atol=0.1)
