"""
Patch-level radiation helpers — **reference implementation only**.

Not called by the production ``calculate()`` API. The fused Rust pipeline
computes patch radiation internally.

Retained for readability, tests, and validation against UMEP.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..constants import KELVIN_OFFSET, SBC

_DEG2RAD = np.pi / 180


def _cardinal_components(
    base_radiation: NDArray[np.floating], patch_azimuth: float
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Split radiation into E/S/W/N components based on patch azimuth.

    The azimuth boundaries determine which cardinal hemisphere a patch
    contributes to, weighted by the cosine of the angular offset.

    Returns:
        Tuple of (Least, Lsouth, Lwest, Lnorth) arrays.
    """
    shape = np.shape(base_radiation)
    Least = np.zeros(shape, dtype=np.float32)
    Lsouth = np.zeros(shape, dtype=np.float32)
    Lwest = np.zeros(shape, dtype=np.float32)
    Lnorth = np.zeros(shape, dtype=np.float32)

    if patch_azimuth < 180:
        Least = base_radiation * np.cos((90 - patch_azimuth) * _DEG2RAD)
    if (patch_azimuth > 90) and (patch_azimuth < 270):
        Lsouth = base_radiation * np.cos((180 - patch_azimuth) * _DEG2RAD)
    if (patch_azimuth > 180) and (patch_azimuth < 360):
        Lwest = base_radiation * np.cos((270 - patch_azimuth) * _DEG2RAD)
    if (patch_azimuth > 270) or (patch_azimuth < 90):
        Lnorth = base_radiation * np.cos((0 - patch_azimuth) * _DEG2RAD)

    return Least, Lsouth, Lwest, Lnorth


def shortwave_from_sky(sky, angle_of_incidence, lumChi, steradian, patch_azimuth, cyl):
    """Calculates the amount of diffuse shortwave radiation from the sky for a patch with:
    angle of incidence = angle_of_incidence
    luminance = lumChi
    steradian = steradian"""

    # Diffuse vertical radiation
    diffuse_shortwave_radiation = sky * lumChi * angle_of_incidence * steradian

    return diffuse_shortwave_radiation


def longwave_from_sky(sky, Lsky_side, Lsky_down, patch_azimuth):
    """Per-patch longwave from a sky patch into 4 cardinal directions + down.

    Returns ``(Lside_sky, Ldown_sky, Least, Lsouth, Lwest, Lnorth)``.
    """
    Ldown_sky = sky * Lsky_down
    Lside_sky = sky * Lsky_side
    Least, Lsouth, Lwest, Lnorth = _cardinal_components(sky * Lsky_side, patch_azimuth)
    return Lside_sky, Ldown_sky, Least, Lsouth, Lwest, Lnorth


def longwave_from_veg(
    vegetation, steradian, angle_of_incidence, angle_of_incidence_h, patch_altitude, patch_azimuth, ewall, Ta
):
    """Longwave radiation from a vegetation patch."""
    vegetation_surface = (ewall * SBC * ((Ta + KELVIN_OFFSET) ** 4)) / np.pi
    Lside_veg = vegetation_surface * steradian * angle_of_incidence * vegetation
    Ldown_veg = vegetation_surface * steradian * angle_of_incidence_h * vegetation

    base = vegetation_surface * steradian * np.cos(patch_altitude * _DEG2RAD) * vegetation
    Least, Lsouth, Lwest, Lnorth = _cardinal_components(base, patch_azimuth)

    return Lside_veg, Ldown_veg, Least, Lsouth, Lwest, Lnorth


def longwave_from_buildings(
    building,
    steradian,
    angle_of_incidence,
    angle_of_incidence_h,
    patch_azimuth,
    sunlit_patches,
    shaded_patches,
    azimuth_difference,
    solar_altitude,
    ewall,
    Ta,
    Tgwall,
):
    """Per-patch longwave from sunlit + shaded building walls.

    Splits each patch into sunlit and shaded surface contributions and
    returns ``(Lside_sun, Lside_sh, Ldown_sun, Ldown_sh, Least, Lsouth,
    Lwest, Lnorth)``.
    """
    sunlit_surface = (ewall * SBC * ((Ta + Tgwall + KELVIN_OFFSET) ** 4)) / np.pi
    shaded_surface = (ewall * SBC * ((Ta + KELVIN_OFFSET) ** 4)) / np.pi

    if (azimuth_difference > 90) and (azimuth_difference < 270) and (solar_altitude > 0):
        Lside_sun = sunlit_surface * sunlit_patches * steradian * angle_of_incidence * building
        Lside_sh = shaded_surface * shaded_patches * steradian * angle_of_incidence * building
        Ldown_sun = sunlit_surface * sunlit_patches * steradian * angle_of_incidence_h * building
        Ldown_sh = shaded_surface * shaded_patches * steradian * angle_of_incidence_h * building

        # Cardinal components: sum of sunlit + shaded contributions
        base_sun = sunlit_surface * sunlit_patches * steradian * angle_of_incidence * building
        base_sh = shaded_surface * shaded_patches * steradian * angle_of_incidence * building
        Le_sun, Ls_sun, Lw_sun, Ln_sun = _cardinal_components(base_sun, patch_azimuth)
        Le_sh, Ls_sh, Lw_sh, Ln_sh = _cardinal_components(base_sh, patch_azimuth)
        Least = Le_sun + Le_sh
        Lsouth = Ls_sun + Ls_sh
        Lwest = Lw_sun + Lw_sh
        Lnorth = Ln_sun + Ln_sh
    else:
        Lside_sh = shaded_surface * steradian * angle_of_incidence * building
        Lside_sun = np.zeros_like(Lside_sh)
        Ldown_sh = shaded_surface * steradian * angle_of_incidence_h * building
        Ldown_sun = np.zeros_like(Lside_sh)

        base = shaded_surface * steradian * angle_of_incidence * building
        Least, Lsouth, Lwest, Lnorth = _cardinal_components(base, patch_azimuth)

    return Lside_sun, Lside_sh, Ldown_sun, Ldown_sh, Least, Lsouth, Lwest, Lnorth


def longwave_from_buildings_wallScheme(
    voxelMaps, voxelTable, steradian, angle_of_incidence, angle_of_incidence_h, patch_azimuth
):
    """Per-patch longwave using the voxel-based wall-temperature scheme.

    Looks up per-voxel surface temperature from ``voxelTable`` and applies
    it to the patch geometry. Returns ``(Lside, Lside_sh, Ldown, Ldown_sh,
    Least, Lsouth, Lwest, Lnorth)`` with ``_sh`` arrays zeroed (this scheme
    doesn't split sunlit/shaded).
    """
    unique_ids = list(np.unique(voxelMaps)[1:])
    lw_rad_dict = dict(voxelTable.loc[unique_ids, "LongwaveRadiation"])
    patch_radiation = np.vectorize(lw_rad_dict.get)(voxelMaps).astype(float)
    patch_radiation[np.isnan(patch_radiation)] = 0

    Lside = patch_radiation * steradian * angle_of_incidence
    Ldown = patch_radiation * steradian * angle_of_incidence_h

    base = patch_radiation * steradian * angle_of_incidence
    Least, Lsouth, Lwest, Lnorth = _cardinal_components(base, patch_azimuth)

    Lside_sh = np.zeros_like(Lside)
    Ldown_sh = np.zeros_like(Ldown)
    return Lside, Lside_sh, Ldown, Ldown_sh, Least, Lsouth, Lwest, Lnorth


def reflected_longwave(
    reflecting_surface, steradian, angle_of_incidence, angle_of_incidence_h, patch_azimuth, Ldown_sky, Lup, ewall
):
    """Per-patch longwave reflected off a surface (1-ewall fraction of incident).

    Returns ``(Lside_ref, Ldown_ref, Least, Lsouth, Lwest, Lnorth)``.
    """
    reflected_radiation = ((Ldown_sky + Lup) * (1 - ewall) * 0.5) / np.pi
    Lside_ref = reflected_radiation * steradian * angle_of_incidence * reflecting_surface
    Ldown_ref = reflected_radiation * steradian * angle_of_incidence_h * reflecting_surface

    base = reflected_radiation * steradian * angle_of_incidence * reflecting_surface
    Least, Lsouth, Lwest, Lnorth = _cardinal_components(base, patch_azimuth)

    return Lside_ref, Ldown_ref, Least, Lsouth, Lwest, Lnorth


def patch_steradians(L_patches):
    """'This function calculates the steradians of the patches"""

    # Degrees to radians
    deg2rad = np.pi / 180

    # Unique altitudes for patches
    skyalt, skyalt_c = np.unique(L_patches[:, 0], return_counts=True)

    # Altitudes of the Robinson & Stone patches
    patch_altitude = L_patches[:, 0]

    # Calculation of steradian for each patch
    # Build scalar lookup once to avoid array->scalar coercion warnings.
    count_by_altitude = {float(alt): float(count) for alt, count in zip(skyalt, skyalt_c, strict=False)}
    steradian = np.zeros((patch_altitude.shape[0]), dtype=np.float32)
    for i in range(patch_altitude.shape[0]):
        band_count = count_by_altitude[float(patch_altitude[i])]
        # If there are more than one patch in a band
        if band_count > 1:
            steradian[i] = ((360 / band_count) * deg2rad) * (
                np.sin((patch_altitude[i] + patch_altitude[0]) * deg2rad)
                - np.sin((patch_altitude[i] - patch_altitude[0]) * deg2rad)
            )
        # If there is only one patch in band, i.e. 90 degrees
        else:
            steradian[i] = ((360 / band_count) * deg2rad) * (
                np.sin((patch_altitude[i]) * deg2rad) - np.sin((patch_altitude[i - 1] + patch_altitude[0]) * deg2rad)
            )

    return steradian, skyalt, patch_altitude
