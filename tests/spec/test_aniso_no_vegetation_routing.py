"""Anisotropic sky without vegetation: patch routing (issue #15).

The anisotropic sky sends each patch to one emitter from three shadow bits:
sky on ``sh && vsh``, vegetation on ``!vsh || !vbsh``, buildings on
``!sh && vbsh``. Without a canopy layer nothing is blocked by vegetation, so
both vegetation matrices must read "visible" on every patch. Up to 0.1.0b96
the SVF step wrote a copy of the building matrix into ``vbsh``, which sent
every building-blocked patch to the vegetation emitter (air temperature) and
left the sunlit wall temperature unused. Adding any canopy pixel, however far
away, switched the routing and raised Tmrt beside a sunlit wall by 2-4 K.

Reference: umep-solweig ``svfForProcessing153`` sets both matrices to ones
without vegetation in its wall-scheme branch.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pytest
from solweig import Location, SurfaceData, Weather
from solweig.api import _calculate_single
from solweig.models import ThermalState, surface_compute
from solweig.models.shadow_arrays import ShadowArrays
from solweig.physics.wallalgorithms import filter1Goodwin_as_aspect_v3, findwalls
from solweig.timeseries import _precompute_weather

SHAPE = (60, 60)
RECEPTOR = (30, 26)  # 4 m west of the west wall
WALL_STRIP = (slice(18, 42), slice(20, 30))  # 10 m strip west of the wall


def _dsm():
    dsm = np.zeros(SHAPE, dtype=np.float32)
    dsm[15:45, 30:36] = 15.0  # building block, west wall at column 30
    return dsm


def _surface(*, with_far_tree=False, shadow_matrices=None):
    """Scene with SVF and shadow matrices computed for its own inputs."""
    dsm = _dsm()
    kw: dict[str, Any] = {}
    if with_far_tree:
        cdsm = np.zeros(SHAPE, dtype=np.float32)
        cdsm[2:4, 2:4] = 3.0  # 3 m tree in the far corner
        kw = dict(cdsm=cdsm, tdsm=np.zeros_like(cdsm), cdsm_relative=False, tdsm_relative=False)
    walls = findwalls(dsm, 1.0)
    aspect = filter1Goodwin_as_aspect_v3(walls, 1.0, dsm)
    base = SurfaceData(dsm=dsm, pixel_size=1.0, **kw)
    base.compute_svf()
    return SurfaceData(
        dsm=dsm,
        pixel_size=1.0,
        svf=base.svf,
        shadow_matrices=shadow_matrices if shadow_matrices is not None else base.shadow_matrices,
        wall_height=walls,
        wall_aspect=aspect,
        **kw,
    )


def _tmrt(surface, hour=17):
    """Anisotropic Tmrt for one clear July afternoon step (sun on the west wall)."""
    location = Location(latitude=52.0, longitude=4.5, utc_offset=2)
    weather = Weather(datetime=datetime(2025, 7, 20, hour, 0), ta=32.0, rh=30.0, global_rad=650.0)
    _precompute_weather([weather], location)
    state = ThermalState.initial(surface.shape)
    state.timestep_dec = 1.0 / 24.0
    result = _calculate_single(
        surface=surface,
        location=location,
        weather=weather,
        use_anisotropic_sky=True,
        state=state,
        return_state_copy=False,
        _requested_outputs={"tmrt"},
    )
    return np.asarray(result.tmrt)


@pytest.fixture(scope="module")
def no_veg_surface():
    return _surface()


@pytest.fixture(scope="module")
def no_veg_tmrt(no_veg_surface):
    return _tmrt(no_veg_surface)


class TestShadowMatricesWithoutVegetation:
    """The SVF step writes the "no blocking" convention into both veg matrices."""

    def test_single_shot_matrices_are_all_visible(self, no_veg_surface):
        sm = no_veg_surface.shadow_matrices
        assert np.all(sm.vegshmat == 1.0)
        assert np.all(sm.vbshmat == 1.0)

    def test_scene_has_building_blocked_patches(self, no_veg_surface):
        """Guard: the receptor must see the wall, or the routing is not exercised."""
        r, c = RECEPTOR
        blocked = int((no_veg_surface.shadow_matrices.shmat[r, c] == 0).sum())
        assert blocked > 20

    def test_tiled_matrices_match_single_shot(self, no_veg_surface, tmp_path, monkeypatch):
        monkeypatch.setattr("solweig.tiling.MIN_TILE_SIZE", 32)
        dsm = _dsm()
        tiled = SurfaceData(dsm=dsm.copy(), pixel_size=1.0)
        surface_compute.compute_and_cache_svf(
            tiled,
            {"dsm_arr": dsm, "cdsm_arr": None, "tdsm_arr": None, "pixel_size": 1.0},
            tmp_path / "tiled",
            trunk_ratio=0.25,
            tile_size=32,
        )
        assert tiled.shadow_matrices is not None
        assert np.all(tiled.shadow_matrices.vegshmat == 1.0)
        assert np.all(tiled.shadow_matrices.vbshmat == 1.0)
        np.testing.assert_array_equal(tiled.shadow_matrices.shmat, no_veg_surface.shadow_matrices.shmat)


class TestTmrtIndependentOfDistantVegetation:
    """Tmrt beside a sunlit wall must not depend on whether a far tree exists."""

    def test_far_tree_leaves_wall_strip_unchanged(self, no_veg_tmrt):
        far_tree_tmrt = _tmrt(_surface(with_far_tree=True))
        r, c = RECEPTOR
        assert abs(float(far_tree_tmrt[r, c] - no_veg_tmrt[r, c])) < 0.05
        # The tree hides one low sky patch from a few strip pixels (up to
        # 0.06 K); the routing error this guards against was 2-4 K.
        np.testing.assert_allclose(far_tree_tmrt[WALL_STRIP], no_veg_tmrt[WALL_STRIP], atol=0.25)


class TestStoredVegMatricesNotTrustedWithoutVegetation:
    """Stale caches and UMEP exports must not change a run without vegetation."""

    @pytest.mark.parametrize("stale", ["copy_of_buildings", "zeros"])
    def test_stale_veg_matrices_give_the_same_tmrt(self, no_veg_surface, no_veg_tmrt, stale):
        fresh = no_veg_surface.shadow_matrices
        shmat_u8 = fresh._shmat_u8
        if stale == "copy_of_buildings":  # written by this package up to 0.1.0b96
            veg_u8, vb_u8 = np.full_like(shmat_u8, 0xFF), shmat_u8.copy()
        else:  # tiled path up to 0.1.0b96, and UMEP exports without the wall scheme
            veg_u8, vb_u8 = np.zeros_like(shmat_u8), np.zeros_like(shmat_u8)
        stale_matrices = ShadowArrays(
            _shmat_u8=shmat_u8.copy(),
            _vegshmat_u8=veg_u8,
            _vbshmat_u8=vb_u8,
            _n_patches=fresh.patch_count,
        )
        tmrt = _tmrt(_surface(shadow_matrices=stale_matrices))
        np.testing.assert_allclose(tmrt, no_veg_tmrt, atol=1e-4)
