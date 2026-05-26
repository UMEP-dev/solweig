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


def test_top_level_reexports_still_work_but_warn():
    """Top-level `solweig.extract_bounds` etc. remain accessible for backwards
    compatibility, BUT must emit a ``DeprecationWarning`` and the canonical
    `solweig.geospatial.<name>` must resolve to the same object.

    When the top-level access is finally removed, flip this to
    `assert not hasattr(solweig, name)` and delete the deprecation hook
    in `solweig/__init__.py`.
    """
    import warnings

    import solweig

    deprecated_names = (
        "extract_bounds",
        "intersect_bounds",
        "resample_to_grid",
        "pixel_size_tag",
        "compute_max_tile_pixels",
        "looks_like_relative",
        "namespace_to_dict",
        "wallalgorithms",
    )

    for name in deprecated_names:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = getattr(solweig, name)
        assert any(issubclass(w.category, DeprecationWarning) for w in caught), (
            f"`solweig.{name}` accessed without a DeprecationWarning"
        )
        # The deprecated handle and the canonical home must be the same object.
        from solweig import geospatial

        assert value is getattr(geospatial, name)


def test_deprecated_names_are_not_in_top_level_all():
    """`__all__` is the documented public surface. The deprecated re-exports
    must NOT appear there (only the `solweig.geospatial` submodule does)."""
    import solweig

    deprecated = {
        "extract_bounds",
        "intersect_bounds",
        "resample_to_grid",
        "pixel_size_tag",
        "compute_max_tile_pixels",
        "looks_like_relative",
        "namespace_to_dict",
        "wallalgorithms",
    }
    leaked = deprecated & set(solweig.__all__)
    assert not leaked, f"Deprecated names still in __all__: {sorted(leaked)}"


def test_unknown_attribute_still_raises():
    """The deprecation `__getattr__` must not swallow truly-unknown attrs."""
    import pytest as _pytest
    import solweig

    with _pytest.raises(AttributeError):
        _ = solweig.this_does_not_exist  # noqa: F841
