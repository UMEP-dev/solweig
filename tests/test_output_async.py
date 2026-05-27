"""Tests for `solweig.output_async` — async GeoTIFF writing helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from solweig.output_async import (
    AsyncGeoTiffWriter,
    async_output_enabled,
    collect_output_arrays,
)

# ── async_output_enabled ────────────────────────────────────────────────────


def test_async_output_enabled_default_true(monkeypatch):
    monkeypatch.delenv("SOLWEIG_ASYNC_OUTPUT", raising=False)
    assert async_output_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "FALSE", "no", "off"])
def test_async_output_disabled_via_env(monkeypatch, val):
    monkeypatch.setenv("SOLWEIG_ASYNC_OUTPUT", val)
    assert async_output_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "yes", ""])
def test_async_output_enabled_via_env(monkeypatch, val):
    monkeypatch.setenv("SOLWEIG_ASYNC_OUTPUT", val)
    assert async_output_enabled() is True


# ── collect_output_arrays ───────────────────────────────────────────────────


def _result_with(**arrays):
    """Build a SolweigResult-like mock with the named array fields populated."""
    r = MagicMock()
    # Default all known fields to None.
    for name in ("tmrt", "utci", "pet", "shadow", "kdown", "kup", "ldown", "lup"):
        setattr(r, name, None)
    # Then set the requested ones.
    for k, v in arrays.items():
        setattr(r, k, v)
    return r


def test_collect_output_arrays_returns_requested_arrays():
    tmrt = np.ones((3, 3), dtype=np.float32)
    shadow = np.zeros((3, 3), dtype=np.float32)
    r = _result_with(tmrt=tmrt, shadow=shadow)
    selected = collect_output_arrays(r, ["tmrt", "shadow"])
    assert set(selected.keys()) == {"tmrt", "shadow"}
    assert selected["tmrt"] is tmrt
    assert selected["shadow"] is shadow


def test_collect_output_arrays_skips_unknown_name(caplog):
    r = _result_with(tmrt=np.ones((2, 2), dtype=np.float32))
    selected = collect_output_arrays(r, ["tmrt", "bogus_field"])
    assert "bogus_field" not in selected
    assert "tmrt" in selected


def test_collect_output_arrays_skips_none_field():
    # shadow is None by default — should be skipped, not crash.
    r = _result_with(tmrt=np.ones((2, 2), dtype=np.float32))
    selected = collect_output_arrays(r, ["tmrt", "shadow"])
    assert "tmrt" in selected
    assert "shadow" not in selected


def test_collect_output_arrays_empty_request():
    r = _result_with(tmrt=np.ones((2, 2), dtype=np.float32))
    assert collect_output_arrays(r, []) == {}


# ── AsyncGeoTiffWriter lifecycle (no actual writes — uses no-surface mode) ──


def test_async_writer_creates_output_dir(tmp_path):
    out = tmp_path / "subdir" / "results"
    AsyncGeoTiffWriter(out)
    assert out.exists()
    assert out.is_dir()


def test_async_writer_max_pending_clamped_to_at_least_one(tmp_path):
    w = AsyncGeoTiffWriter(tmp_path, max_pending=0)
    assert w.max_pending == 1


def test_async_writer_without_surface_has_empty_crs(tmp_path):
    w = AsyncGeoTiffWriter(tmp_path)
    assert w.crs_wkt == ""
    assert w.transform is None


def test_async_writer_uses_surface_geotransform_if_available(tmp_path):
    """If a SurfaceData has _geotransform / _crs_wkt, the writer adopts them."""
    surface = MagicMock()
    surface._geotransform = [0.0, 1.0, 0.0, 10.0, 0.0, -1.0]
    surface._crs_wkt = "WKT_HERE"
    w = AsyncGeoTiffWriter(tmp_path, surface=surface)
    assert w.transform == [0.0, 1.0, 0.0, 10.0, 0.0, -1.0]
    assert w.crs_wkt == "WKT_HERE"
