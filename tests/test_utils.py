"""Tests for `solweig.utils` — geometry + namespace conversion helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from solweig.utils import (
    dict_to_namespace,
    extract_bounds,
    intersect_bounds,
    namespace_to_dict,
    resample_to_grid,
)

# ── dict_to_namespace / namespace_to_dict ───────────────────────────────────


def test_dict_to_namespace_flat():
    ns = dict_to_namespace({"a": 1, "b": "x", "c": 2.5})
    assert isinstance(ns, SimpleNamespace)
    assert ns.a == 1
    assert ns.b == "x"
    assert ns.c == 2.5


def test_dict_to_namespace_nested():
    ns = dict_to_namespace({"outer": {"inner": {"deep": 42}}})
    assert isinstance(ns, SimpleNamespace)
    assert ns.outer.inner.deep == 42


def test_dict_to_namespace_list_of_dicts():
    ns = dict_to_namespace({"items": [{"x": 1}, {"x": 2}]})
    # The wrapping dict becomes a SimpleNamespace; .items is a list of
    # SimpleNamespaces. The type checker needs explicit narrowing.
    assert isinstance(ns, SimpleNamespace)
    items = ns.items
    assert isinstance(items, list)
    assert items[0].x == 1
    assert items[1].x == 2


def test_dict_to_namespace_scalar_passthrough():
    assert dict_to_namespace(42) == 42
    assert dict_to_namespace("hello") == "hello"
    assert dict_to_namespace(None) is None


def test_namespace_to_dict_flat():
    d = namespace_to_dict(SimpleNamespace(a=1, b="x"))
    assert d == {"a": 1, "b": "x"}


def test_namespace_to_dict_nested_roundtrip():
    original = {"materials": {"walls": {"albedo": 0.2, "emissivity": 0.9}}, "names": ["a", "b"]}
    roundtripped = namespace_to_dict(dict_to_namespace(original))
    assert roundtripped == original


def test_namespace_to_dict_handles_list_of_namespaces():
    ns_list = [SimpleNamespace(x=1), SimpleNamespace(x=2)]
    d = namespace_to_dict(ns_list)
    assert d == [{"x": 1}, {"x": 2}]


def test_namespace_to_dict_scalar_passthrough():
    assert namespace_to_dict(3.14) == 3.14
    assert namespace_to_dict("hello") == "hello"


# ── extract_bounds ──────────────────────────────────────────────────────────


def test_extract_bounds_from_gdal_list():
    """GDAL geotransform: [x_origin, x_res, 0, y_origin, 0, -y_res]"""
    gt = [100.0, 1.0, 0.0, 200.0, 0.0, -1.0]  # origin top-left, 1m pixels
    bounds = extract_bounds(gt, (50, 100))  # 50 rows, 100 cols
    assert bounds == [100.0, 150.0, 200.0, 200.0]


def test_extract_bounds_from_affine():
    """Affine transform → same bounds as the equivalent GDAL list."""
    pytest.importorskip("affine")
    from affine import Affine

    gdal_gt = [100.0, 1.0, 0.0, 200.0, 0.0, -1.0]
    affine_gt = Affine.from_gdal(*gdal_gt)
    assert extract_bounds(affine_gt, (50, 100)) == extract_bounds(gdal_gt, (50, 100))


# ── intersect_bounds ────────────────────────────────────────────────────────


def test_intersect_bounds_overlapping():
    b1 = [0.0, 0.0, 10.0, 10.0]
    b2 = [5.0, 5.0, 15.0, 15.0]
    assert intersect_bounds([b1, b2]) == [5.0, 5.0, 10.0, 10.0]


def test_intersect_bounds_single_input():
    b = [0.0, 0.0, 10.0, 10.0]
    assert intersect_bounds([b]) == b


def test_intersect_bounds_three_rasters():
    b1 = [0.0, 0.0, 10.0, 10.0]
    b2 = [2.0, 2.0, 12.0, 12.0]
    b3 = [1.0, 1.0, 11.0, 11.0]
    assert intersect_bounds([b1, b2, b3]) == [2.0, 2.0, 10.0, 10.0]


def test_intersect_bounds_empty_input_raises():
    with pytest.raises(ValueError, match="No bounding boxes"):
        intersect_bounds([])


def test_intersect_bounds_disjoint_raises():
    b1 = [0.0, 0.0, 5.0, 5.0]
    b2 = [10.0, 10.0, 15.0, 15.0]
    with pytest.raises(ValueError, match="don't intersect"):
        intersect_bounds([b1, b2])


def test_intersect_bounds_touching_boundary_raises():
    """A point intersection (no area) is treated as no intersection."""
    b1 = [0.0, 0.0, 5.0, 5.0]
    b2 = [5.0, 5.0, 10.0, 10.0]
    with pytest.raises(ValueError):
        intersect_bounds([b1, b2])


# ── resample_to_grid ────────────────────────────────────────────────────────


def test_resample_to_grid_identity_preserves_values():
    """Resampling to the same grid with bilinear should preserve values."""
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    src_gt = [0.0, 1.0, 0.0, 2.0, 0.0, -1.0]  # 2x2, origin (0, 2), 1m pixels
    out, _ = resample_to_grid(arr, src_gt, target_bbox=[0.0, 0.0, 2.0, 2.0], target_pixel_size=1.0, src_crs="EPSG:3857")
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out, arr, atol=1e-6)


def test_resample_to_grid_upsample_doubles_dimensions():
    """Halve the pixel size → 2x dimensions per axis."""
    arr = np.array([[1.0, 4.0], [4.0, 16.0]], dtype=np.float32)
    src_gt = [0.0, 1.0, 0.0, 2.0, 0.0, -1.0]
    out, _ = resample_to_grid(arr, src_gt, target_bbox=[0.0, 0.0, 2.0, 2.0], target_pixel_size=0.5, src_crs="EPSG:3857")
    assert out.shape == (4, 4)


def test_resample_to_grid_nearest_method():
    """Nearest-neighbour preserves discrete values; bilinear blends."""
    arr = np.array([[1.0, 5.0], [5.0, 1.0]], dtype=np.float32)
    src_gt = [0.0, 1.0, 0.0, 2.0, 0.0, -1.0]
    out_nn, _ = resample_to_grid(
        arr,
        src_gt,
        target_bbox=[0.0, 0.0, 2.0, 2.0],
        target_pixel_size=0.5,
        method="nearest",
        src_crs="EPSG:3857",
    )
    # Nearest neighbour should only produce values from {1.0, 5.0}.
    unique = set(np.unique(out_nn).tolist())
    assert unique.issubset({1.0, 5.0})
