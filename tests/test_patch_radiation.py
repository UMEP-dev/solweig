"""Tests for `solweig.physics.patch_radiation` — reference per-patch helpers.

These functions are kept for readability and UMEP-parity validation; the
production path uses the fused Rust pipeline. The tests below pin the
maths so any drift between the reference Python and the Rust
implementation is caught.
"""

from __future__ import annotations

import numpy as np
import pytest
from solweig.physics.patch_radiation import (
    longwave_from_buildings,
    longwave_from_buildings_wallScheme,
    longwave_from_sky,
    longwave_from_veg,
    patch_steradians,
    reflected_longwave,
    shortwave_from_sky,
)

# ── shortwave_from_sky ──────────────────────────────────────────────────────


def test_shortwave_from_sky_scales_with_luminance():
    """shortwave = sky * lumChi * cos(theta) * steradian"""
    sky = np.ones((3, 3), dtype=np.float32)
    angle_cos = np.full((3, 3), 0.5, dtype=np.float32)
    out = shortwave_from_sky(sky, angle_cos, lumChi=2.0, steradian=0.1, patch_azimuth=180.0, cyl=False)
    np.testing.assert_allclose(out, 1.0 * 2.0 * 0.5 * 0.1)


def test_shortwave_from_sky_zero_when_sky_blocked():
    sky = np.zeros((2, 2), dtype=np.float32)
    angle_cos = np.ones((2, 2), dtype=np.float32)
    out = shortwave_from_sky(sky, angle_cos, lumChi=10.0, steradian=0.5, patch_azimuth=0.0, cyl=False)
    np.testing.assert_array_equal(out, 0.0)


# ── longwave_from_sky ───────────────────────────────────────────────────────


def test_longwave_from_sky_zero_visibility_zeroes_all_outputs():
    sky = np.zeros((2, 2), dtype=np.float32)
    Lside, Ldown, Le, Ls, Lw, Ln = longwave_from_sky(sky, Lsky_side=400.0, Lsky_down=400.0, patch_azimuth=180.0)
    for arr in (Lside, Ldown, Le, Ls, Lw, Ln):
        np.testing.assert_array_equal(arr, 0.0)


def test_longwave_from_sky_full_visibility_returns_lsky_values():
    sky = np.ones((2, 2), dtype=np.float32)
    Lside, Ldown, *_ = longwave_from_sky(sky, Lsky_side=400.0, Lsky_down=350.0, patch_azimuth=180.0)
    np.testing.assert_allclose(Lside, 400.0)
    np.testing.assert_allclose(Ldown, 350.0)


@pytest.mark.parametrize(
    ("patch_azimuth", "expected_dominant"),
    [(90.0, "east"), (180.0, "south"), (270.0, "west"), (0.001, "north")],
)
def test_longwave_from_sky_cardinal_routing(patch_azimuth, expected_dominant):
    """Each patch azimuth should send most energy to the matching cardinal."""
    sky = np.ones((1, 1), dtype=np.float32)
    _, _, Le, Ls, Lw, Ln = longwave_from_sky(sky, Lsky_side=400.0, Lsky_down=0.0, patch_azimuth=patch_azimuth)
    cardinals = {"east": Le, "south": Ls, "west": Lw, "north": Ln}
    dom = cardinals[expected_dominant]
    others = [v for k, v in cardinals.items() if k != expected_dominant]
    assert float(dom) >= max(float(o) for o in others)


# ── longwave_from_veg ───────────────────────────────────────────────────────


def test_longwave_from_veg_zero_vegetation_zeroes_output():
    veg = np.zeros((2, 2), dtype=np.float32)
    Lside, Ldown, *_ = longwave_from_veg(
        veg,
        steradian=0.1,
        angle_of_incidence=0.5,
        angle_of_incidence_h=0.7,
        patch_altitude=30.0,
        patch_azimuth=180.0,
        ewall=0.9,
        Ta=20.0,
    )
    np.testing.assert_array_equal(Lside, 0.0)
    np.testing.assert_array_equal(Ldown, 0.0)


def test_longwave_from_veg_scales_with_air_temperature():
    """Hotter Ta → more longwave (Stefan-Boltzmann T^4)."""
    veg = np.ones((1, 1), dtype=np.float32)
    args = dict(
        steradian=0.1,
        angle_of_incidence=1.0,
        angle_of_incidence_h=1.0,
        patch_altitude=45.0,
        patch_azimuth=180.0,
        ewall=0.9,
    )
    cool = longwave_from_veg(veg, Ta=10.0, **args)
    warm = longwave_from_veg(veg, Ta=30.0, **args)
    assert float(warm[0]) > float(cool[0])


# ── longwave_from_buildings (sunlit/shaded split) ───────────────────────────


def test_longwave_from_buildings_no_sun_uses_shaded_only():
    """When solar_altitude <= 0, Lside_sun should be all zeros."""
    building = np.ones((2, 2), dtype=np.float32)
    Lside_sun, Lside_sh, *_ = longwave_from_buildings(
        building,
        steradian=0.1,
        angle_of_incidence=0.5,
        angle_of_incidence_h=0.5,
        patch_azimuth=180.0,
        sunlit_patches=np.ones((2, 2), dtype=np.float32),
        shaded_patches=np.ones((2, 2), dtype=np.float32),
        azimuth_difference=180.0,
        solar_altitude=-5.0,  # below horizon
        ewall=0.9,
        Ta=20.0,
        Tgwall=15.0,
    )
    np.testing.assert_array_equal(Lside_sun, 0.0)
    assert float(np.mean(Lside_sh)) > 0.0


def test_longwave_from_buildings_sun_facing_emits_more_than_shaded():
    """When sun faces the patch and is up, sunlit term > shaded term."""
    building = np.ones((1, 1), dtype=np.float32)
    Lside_sun, Lside_sh, *_ = longwave_from_buildings(
        building,
        steradian=0.1,
        angle_of_incidence=1.0,
        angle_of_incidence_h=1.0,
        patch_azimuth=180.0,
        sunlit_patches=np.ones((1, 1), dtype=np.float32),
        shaded_patches=np.ones((1, 1), dtype=np.float32),
        azimuth_difference=180.0,  # within sun-facing range (90, 270)
        solar_altitude=45.0,
        ewall=0.9,
        Ta=20.0,
        Tgwall=15.0,  # Tgwall raises sunlit surface temp
    )
    assert float(Lside_sun[0]) > float(Lside_sh[0])


# ── longwave_from_buildings_wallScheme ──────────────────────────────────────


def test_longwave_from_buildings_wall_scheme_uses_voxel_table():
    """Voxel IDs map to per-voxel radiation values from the lookup table.

    NOTE: the function skips the first unique value (treated as nodata),
    so the test uses 0 as nodata and 1, 2 as real voxels.
    """
    import pandas as pd

    voxel_maps = np.array([[0, 1], [2, 2]], dtype=np.int32)
    voxel_table = pd.DataFrame({"LongwaveRadiation": [100.0, 200.0]}, index=[1, 2])
    Lside, Lside_sh, Ldown, Ldown_sh, *_ = longwave_from_buildings_wallScheme(
        voxel_maps,
        voxel_table,
        steradian=1.0,
        angle_of_incidence=1.0,
        angle_of_incidence_h=1.0,
        patch_azimuth=180.0,
    )
    # nodata pixel (voxel id 0) -> 0; voxels 1, 2 -> their lookup values.
    np.testing.assert_allclose(Lside, [[0.0, 100.0], [200.0, 200.0]])
    np.testing.assert_array_equal(Lside_sh, 0.0)
    np.testing.assert_array_equal(Ldown_sh, 0.0)


# ── reflected_longwave ──────────────────────────────────────────────────────


def test_reflected_longwave_proportional_to_one_minus_emissivity():
    """Higher reflectivity (lower ewall) → more reflected longwave."""
    surface = np.ones((1, 1), dtype=np.float32)
    args = dict(
        steradian=0.1,
        angle_of_incidence=1.0,
        angle_of_incidence_h=1.0,
        patch_azimuth=180.0,
        Ldown_sky=400.0,
        Lup=300.0,
    )
    high_emis = reflected_longwave(surface, ewall=0.95, **args)[0]
    low_emis = reflected_longwave(surface, ewall=0.50, **args)[0]
    assert float(low_emis[0]) > float(high_emis[0])


def test_reflected_longwave_zero_when_no_incoming():
    """Zero Ldown_sky + Lup → zero reflected radiation."""
    surface = np.ones((2, 2), dtype=np.float32)
    Lside_ref, Ldown_ref, *_ = reflected_longwave(
        surface,
        steradian=0.1,
        angle_of_incidence=1.0,
        angle_of_incidence_h=1.0,
        patch_azimuth=180.0,
        Ldown_sky=0.0,
        Lup=0.0,
        ewall=0.9,
    )
    np.testing.assert_array_equal(Lside_ref, 0.0)
    np.testing.assert_array_equal(Ldown_ref, 0.0)


# ── patch_steradians ────────────────────────────────────────────────────────


def test_patch_steradians_sums_to_hemisphere():
    """Steradians across all patches must approximate 2π (a hemisphere)."""
    from solweig.physics.create_patches import create_patches

    alts, azis, _, _, _, _, _ = create_patches(2)  # 153 patches
    L_patches = np.zeros((alts.size, 3), dtype=np.float32)
    L_patches[:, 0] = alts
    L_patches[:, 1] = azis
    steradians, skyalt, patch_altitude = patch_steradians(L_patches)
    total = float(np.sum(steradians))
    assert abs(total - 2 * np.pi) < 0.05


def test_patch_steradians_returns_consistent_lengths():
    """`patch_altitude` returned equals input length; `skyalt` is unique values."""
    from solweig.physics.create_patches import create_patches

    alts, azis, _, _, _, _, _ = create_patches(2)
    L = np.column_stack([alts, azis, np.zeros_like(alts)])
    steradians, skyalt, patch_altitude = patch_steradians(L)
    assert len(steradians) == len(L)
    assert len(patch_altitude) == len(L)
    assert len(skyalt) <= len(L)
    assert len(np.unique(skyalt)) == len(skyalt)
