"""Tests for the typed read-only views over :class:`SurfaceData`.

The views (`surface.geometry`, `.optical`, `.auxiliary`) are thin
wrappers that defer to the underlying SurfaceData fields. They exist
to give internal callers structural clarity without changing the
field layout. These tests pin the proxy semantics and the
``is_ready`` / ``has_walls`` / ``has_svf`` flags.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from solweig.api import SurfaceData
from solweig.models.surface_views import (
    OpticalPropertiesView,
    PreprocessedAuxiliaryView,
    SurfaceGeometryView,
)


def _mk_surface(shape=(10, 10), with_optical=False, with_aux=False):
    dsm = np.ones(shape, dtype=np.float32) * 2.0
    s = SurfaceData(dsm=dsm, pixel_size=1.0)
    if with_optical:
        s.land_cover = np.ones(shape, dtype=np.int32)
        s.albedo = np.full(shape, 0.2, dtype=np.float32)
        s.emissivity = np.full(shape, 0.9, dtype=np.float32)
    if with_aux:
        s.wall_height = np.zeros(shape, dtype=np.float32)
        s.wall_aspect = np.zeros(shape, dtype=np.float32)
    return s


# ── Property returns the right view type ────────────────────────────────────


def test_surface_exposes_three_view_properties():
    s = _mk_surface()
    assert isinstance(s.geometry, SurfaceGeometryView)
    assert isinstance(s.optical, OpticalPropertiesView)
    assert isinstance(s.auxiliary, PreprocessedAuxiliaryView)


# ── Geometry view proxies the underlying fields ─────────────────────────────


def test_geometry_view_proxies_dsm():
    s = _mk_surface()
    assert s.geometry.dsm is s.dsm


def test_geometry_view_optional_fields_are_none_when_absent():
    s = _mk_surface()
    assert s.geometry.cdsm is None
    assert s.geometry.dem is None
    assert s.geometry.tdsm is None


def test_geometry_view_exposes_pixel_size_and_shape():
    s = _mk_surface(shape=(20, 30))
    assert s.geometry.pixel_size == 1.0
    assert s.geometry.shape == (20, 30)


def test_geometry_view_exposes_height_flags():
    s = _mk_surface()
    # SurfaceData defaults: dsm_relative=False, cdsm_relative=True, tdsm_relative=True
    assert s.geometry.dsm_relative is False
    assert s.geometry.cdsm_relative is True
    assert s.geometry.tdsm_relative is True


def test_geometry_view_reflects_mutation():
    """Views are not snapshots; they proxy live state."""
    s = _mk_surface()
    new_dsm = np.full((10, 10), 5.0, dtype=np.float32)
    s.dsm = new_dsm
    assert s.geometry.dsm is new_dsm


# ── Optical view ────────────────────────────────────────────────────────────


def test_optical_view_empty_when_nothing_set():
    s = _mk_surface()
    assert s.optical.land_cover is None
    assert s.optical.albedo is None
    assert s.optical.emissivity is None
    assert s.optical.has_land_cover is False


def test_optical_view_populated():
    s = _mk_surface(with_optical=True)
    assert s.optical.land_cover is s.land_cover
    assert s.optical.albedo is s.albedo
    assert s.optical.emissivity is s.emissivity
    assert s.optical.has_land_cover is True


# ── Auxiliary view ──────────────────────────────────────────────────────────


def test_auxiliary_view_empty_initially():
    s = _mk_surface()
    assert s.auxiliary.wall_height is None
    assert s.auxiliary.wall_aspect is None
    assert s.auxiliary.svf is None
    assert s.auxiliary.shadow_matrices is None
    assert s.auxiliary.has_walls is False
    assert s.auxiliary.has_svf is False
    assert s.auxiliary.is_ready is False


def test_auxiliary_view_with_walls_only():
    s = _mk_surface(with_aux=True)
    assert s.auxiliary.has_walls is True
    assert s.auxiliary.has_svf is False
    # is_ready requires SVF, not just walls
    assert s.auxiliary.is_ready is False


def test_auxiliary_view_is_ready_when_svf_set():
    from conftest import make_mock_svf

    s = _mk_surface()
    s.svf = make_mock_svf((10, 10))
    assert s.auxiliary.has_svf is True
    assert s.auxiliary.is_ready is True


# ── Views are frozen (read-only) ────────────────────────────────────────────


@pytest.mark.parametrize("view_attr", ["geometry", "optical", "auxiliary"])
def test_views_are_frozen(view_attr):
    s = _mk_surface()
    view = getattr(s, view_attr)
    with pytest.raises(FrozenInstanceError):
        view._surface = None


# ── Views are cheap to construct (no field copy) ────────────────────────────


def test_view_construction_is_o1_relative_to_surface_size():
    """Views just wrap a reference; constructing one is constant-time."""
    small = _mk_surface(shape=(10, 10))
    big = _mk_surface(shape=(1000, 1000))
    # Both should construct instantly; the test is really "no exception".
    assert small.geometry.shape == (10, 10)
    assert big.geometry.shape == (1000, 1000)
