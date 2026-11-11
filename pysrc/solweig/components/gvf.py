"""
Ground View Factor (GVF) — building mask detection.

Provides ``detect_building_mask()`` used by ``calculate_core_fused()`` to
identify building footprint pixels before the fused Rust GVF computation.

Reference:
- Lindberg et al. (2008) - SOLWEIG GVF model with wall radiation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ..errors import InvalidSurfaceData

logger = logging.getLogger(__name__)

# Height above ground from which a pixel counts as building (UMEP: 2 m).
BUILDING_HEIGHT_THRESHOLD_M = 2.0

if TYPE_CHECKING:
    from numpy.typing import NDArray


def detect_building_mask(
    dsm: NDArray[np.floating],
    land_cover: NDArray[np.integer] | None,
    wall_height: NDArray[np.floating] | None,
    pixel_size: float,
    dem: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """
    Create a building mask for GVF calculation.

    GVF (Ground View Factor) expects: 0=building, 1=ground.
    This is used to normalize GVF values over buildings where GVF doesn't apply.

    Args:
        dsm: Digital Surface Model array.
        land_cover: Optional land cover grid (UMEP standard: ID 2 = buildings).
        wall_height: Optional wall height grid.
        pixel_size: Pixel size in meters.
        dem: Optional ground elevation model, same grid as the DSM.

    Returns:
        Building mask where 0=building pixels, 1=ground pixels.

    Detection strategy:
        1. If land_cover provided: Use ID 2 (buildings) directly.
        2. Elif dem provided: pixels 2 m or more above the DEM (UMEP's rule).
        3. Elif wall_height provided: pixels 2 m or more above the
           10th-percentile DSM height away from walls (level terrain assumed).
        4. Else: Assume all ground (no buildings).
    """
    if dsm is None or dsm.size == 0:
        raise InvalidSurfaceData(
            "DSM is required for building mask detection but is missing or empty.",
            field="dsm",
        )

    if pixel_size <= 0:
        raise InvalidSurfaceData(
            f"pixel_size must be positive, got {pixel_size}.",
            field="pixel_size",
            expected="> 0",
            got=str(pixel_size),
        )

    if land_cover is not None:
        if land_cover.shape != dsm.shape:
            raise InvalidSurfaceData(
                f"land_cover shape {land_cover.shape} does not match DSM shape {dsm.shape}.",
                field="land_cover",
                expected=str(dsm.shape),
                got=str(land_cover.shape),
            )
        # Use land cover directly: ID 2 = buildings
        buildings = np.ones_like(dsm, dtype=np.float32)
        buildings[land_cover == 2] = 0.0
        return buildings

    if dem is not None:
        if dem.shape != dsm.shape:
            raise InvalidSurfaceData(
                f"dem shape {dem.shape} does not match DSM shape {dsm.shape}.",
                field="dem",
                expected=str(dsm.shape),
                got=str(dem.shape),
            )
        # UMEP's rule without a land-cover grid: a building is anything 2 m
        # or more above the ground model (Solweig_run.py, `dsm - dem`).
        with np.errstate(invalid="ignore"):
            is_building = (dsm - dem) >= BUILDING_HEIGHT_THRESHOLD_M
        return (~is_building).astype(np.float32)

    if wall_height is not None:
        if wall_height.shape != dsm.shape:
            raise InvalidSurfaceData(
                f"wall_height shape {wall_height.shape} does not match DSM shape {dsm.shape}.",
                field="wall_height",
                expected=str(dsm.shape),
                got=str(wall_height.shape),
            )
        # No ground model: take the 10th-percentile DSM height away from walls
        # as ground level and call anything 2 m or more above it a building.
        # This assumes roughly level terrain; pass a DEM or a land-cover grid
        # on sloping sites. (Up to 0.1.0b99 the wall mask was also dilated by
        # 25 m, which marked the ground around every building as building.)
        wall_mask = wall_height > 0
        ground_level = np.nanpercentile(dsm[~wall_mask], 10) if np.any(~wall_mask) else np.nanmin(dsm)
        with np.errstate(invalid="ignore"):
            is_building = dsm >= (ground_level + BUILDING_HEIGHT_THRESHOLD_M)

        # Invert: 0=building, 1=ground
        return (~is_building).astype(np.float32)

    # No building info available - assume all ground
    logger.debug("No land_cover or wall_height provided; assuming all-ground building mask.")
    return np.ones_like(dsm, dtype=np.float32)
