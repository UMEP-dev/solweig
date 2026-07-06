"""Tests for the tiled SVF computation path and its cancellation lifecycle.

The tiled path (`surface_svf_tiled._compute_svf_tiled`) is used when a grid
exceeds the per-tile pixel budget or when an explicit ``tile_size`` is passed.
These tests force it on small grids by monkeypatching ``tiling.MIN_TILE_SIZE``
so multiple tiles fit in a fast test.

Two invariants are gated here:

1. Tiled and single-shot SVF agree (the tile buffer must fully cover shadow
   reach, and stitching must be seam-free).
2. Cancellation never persists a partial SVF cache. A half-computed cache
   whose untouched tiles hold the memmap default (SVF = 1.0, open sky) would
   validate as fresh on the next run and silently corrupt Tmrt.
"""

from __future__ import annotations

import numpy as np
import pytest
from solweig.errors import ComputationCancelled
from solweig.models import surface_compute
from solweig.models.surface import SurfaceData


def _make_dsm(rows: int = 96, cols: int = 96) -> np.ndarray:
    """Flat terrain with two buildings; low relief keeps tile buffers small."""
    dsm = np.zeros((rows, cols), dtype=np.float32)
    dsm[20:32, 20:32] = 8.0
    dsm[60:76, 50:62] = 12.0
    return dsm


def _aligned(dsm: np.ndarray, pixel_size: float = 1.0) -> dict:
    return {
        "dsm_arr": dsm,
        "cdsm_arr": None,
        "tdsm_arr": None,
        "pixel_size": pixel_size,
    }


class _CancellingFeedback:
    """Minimal QGIS-feedback stand-in that reports cancellation immediately."""

    def __init__(self):
        self.cancelled = True

    def isCanceled(self) -> bool:  # noqa: N802 (QGIS API casing)
        return self.cancelled

    def setProgress(self, value) -> None:  # noqa: N802
        pass

    def setProgressText(self, text) -> None:  # noqa: N802
        pass

    def pushInfo(self, text) -> None:  # noqa: N802
        pass


def _svf_cache_artifacts(working_path):
    """Every on-disk artifact a completed SVF cache is allowed to leave."""
    base = working_path / "svf"
    found = []
    if base.exists():
        for pattern in ("**/memmap", "**/svfs.zip", "**/shadowmats.npz", "**/cache_meta.json"):
            found.extend(base.glob(pattern))
        for pattern in ("**/svf_memmaps", "**/shadow_memmaps"):
            found.extend(p for p in base.glob(pattern) if any(p.iterdir()))
    return found


@pytest.mark.slow
def test_tiled_svf_matches_single_shot(tmp_path, monkeypatch):
    """Multi-tile SVF must equal the single-shot result (seam-free stitching)."""
    monkeypatch.setattr("solweig.tiling.MIN_TILE_SIZE", 32)
    dsm = _make_dsm()

    surface_tiled = SurfaceData(dsm=dsm.copy(), pixel_size=1.0)
    surface_compute.compute_and_cache_svf(
        surface_tiled,
        _aligned(dsm),
        tmp_path / "tiled",
        trunk_ratio=0.25,
        tile_size=32,
    )

    surface_single = SurfaceData(dsm=dsm.copy(), pixel_size=1.0)
    surface_compute.compute_and_cache_svf(
        surface_single,
        _aligned(dsm),
        tmp_path / "single",
        trunk_ratio=0.25,
    )

    assert surface_tiled.svf is not None
    assert surface_single.svf is not None
    for field in ("svf", "svf_north", "svf_east", "svf_south", "svf_west"):
        tiled = np.asarray(getattr(surface_tiled.svf, field))
        single = np.asarray(getattr(surface_single.svf, field))
        np.testing.assert_allclose(tiled, single, atol=2e-3, err_msg=f"tiled vs single-shot mismatch in {field}")
    # SVF is a 0-1 quantity
    svf = np.asarray(surface_tiled.svf.svf)
    assert np.all(svf >= 0.0) and np.all(svf <= 1.0)
    # Shadow matrices must be attached for the anisotropic sky model
    assert surface_tiled.shadow_matrices is not None


def test_tiled_svf_completes_and_writes_cache_artifacts(tmp_path, monkeypatch):
    """A completed tiled run returns SvfArrays and persists the memmap cache.

    Fast regression gate for the tiled assembly path (this is the path that
    crashed with NameError when SvfArrays was only imported under
    TYPE_CHECKING). Low relief keeps the tile buffer, and thus runtime, small.
    """
    monkeypatch.setattr("solweig.tiling.MIN_TILE_SIZE", 32)
    dsm = np.zeros((48, 48), dtype=np.float32)
    dsm[10:16, 10:16] = 3.0
    surface = SurfaceData(dsm=dsm, pixel_size=1.0)
    working = tmp_path / "work"
    surface_compute.compute_and_cache_svf(surface, _aligned(dsm), working, trunk_ratio=0.25, tile_size=32)

    assert surface.svf is not None
    svf = np.asarray(surface.svf.svf)
    assert np.all(svf >= 0.0) and np.all(svf <= 1.0)
    assert surface.shadow_matrices is not None
    svf_dir = working / "svf" / "px1.000"
    assert (svf_dir / "memmap").is_dir()
    assert (svf_dir / "memmap" / "cache_meta.json").is_file()


def test_tiled_svf_cancel_raises_and_persists_nothing(tmp_path, monkeypatch):
    """Cancellation in the tiled path raises and leaves no cache on disk."""
    monkeypatch.setattr("solweig.tiling.MIN_TILE_SIZE", 64)
    dsm = _make_dsm(192, 192)
    surface = SurfaceData(dsm=dsm, pixel_size=1.0)
    working = tmp_path / "work"

    with pytest.raises(ComputationCancelled):
        surface_compute.compute_and_cache_svf(
            surface,
            _aligned(dsm),
            working,
            trunk_ratio=0.25,
            tile_size=64,
            feedback=_CancellingFeedback(),
        )

    assert surface.svf is None
    assert _svf_cache_artifacts(working) == []


def test_nontiled_svf_cancel_raises_and_persists_nothing(tmp_path):
    """Cancellation in the single-shot path raises and leaves no cache on disk."""
    dsm = _make_dsm(192, 192)
    surface = SurfaceData(dsm=dsm, pixel_size=1.0)
    working = tmp_path / "work"

    with pytest.raises(ComputationCancelled):
        surface_compute.compute_and_cache_svf(
            surface,
            _aligned(dsm),
            working,
            trunk_ratio=0.25,
            feedback=_CancellingFeedback(),
        )

    assert surface.svf is None
    assert _svf_cache_artifacts(working) == []
