"""Smoke tests for the `solweig.geospatial` submodule.

The submodule is the documented import path for QGIS-plugin and batch
authors who need the geospatial helpers that SOLWEIG uses internally.
These tests just verify that every name in `solweig.geospatial.__all__`
is importable and resolves to the same object as its canonical home.
"""

from __future__ import annotations


def test_geospatial_submodule_imports_cleanly():
    import solweig.geospatial  # noqa: F401


def test_all_declared_names_are_importable():
    import solweig.geospatial as geo

    for name in geo.__all__:
        assert hasattr(geo, name), f"{name!r} declared in __all__ but not present"


def test_geospatial_names_match_canonical_modules():
    """Re-exports must be identity-equal to their canonical-module source."""
    from solweig import cache, tiling, utils
    from solweig.geospatial import (
        compute_max_tile_pixels,
        extract_bounds,
        intersect_bounds,
        looks_like_relative,
        namespace_to_dict,
        pixel_size_tag,
        resample_to_grid,
        wallalgorithms,
    )
    from solweig.models.surface import looks_like_relative as canon_looks_like_relative
    from solweig.physics import wallalgorithms as canon_wallalgorithms

    assert extract_bounds is utils.extract_bounds
    assert intersect_bounds is utils.intersect_bounds
    assert resample_to_grid is utils.resample_to_grid
    assert namespace_to_dict is utils.namespace_to_dict
    assert pixel_size_tag is cache.pixel_size_tag
    assert compute_max_tile_pixels is tiling.compute_max_tile_pixels
    assert looks_like_relative is canon_looks_like_relative
    assert wallalgorithms is canon_wallalgorithms


def test_top_level_access_is_removed():
    """The b85→b86 deprecation shim was removed in b87. Top-level access
    to the geospatial helpers (``solweig.extract_bounds`` etc) must now
    raise ``AttributeError`` — callers are expected to import from
    ``solweig.geospatial``.
    """
    import pytest as _pytest
    import solweig

    removed_names = (
        "extract_bounds",
        "intersect_bounds",
        "resample_to_grid",
        "pixel_size_tag",
        "compute_max_tile_pixels",
        "looks_like_relative",
        "namespace_to_dict",
        "wallalgorithms",
    )

    for name in removed_names:
        with _pytest.raises(AttributeError):
            _ = getattr(solweig, name)


def test_removed_names_are_not_in_top_level_all():
    """`__all__` is the documented public surface. The removed re-exports
    must NOT appear there (only the `solweig.geospatial` submodule does)."""
    import solweig

    removed = {
        "extract_bounds",
        "intersect_bounds",
        "resample_to_grid",
        "pixel_size_tag",
        "compute_max_tile_pixels",
        "looks_like_relative",
        "namespace_to_dict",
        "wallalgorithms",
    }
    leaked = removed & set(solweig.__all__)
    assert not leaked, f"Removed names still in __all__: {sorted(leaked)}"


def test_unknown_attribute_still_raises():
    """Truly-unknown attributes raise AttributeError (default behaviour now
    that the b85→b86 ``__getattr__`` deprecation hook has been removed).

    Uses a dynamic attribute name via a string variable so neither the
    static type checker (`ty`) nor ruff's useless-expression rule (`B018`)
    flag the deliberate miss."""
    import pytest as _pytest
    import solweig

    bad_name = "this_does_not_exist"
    with _pytest.raises(AttributeError):
        _ = getattr(solweig, bad_name)
