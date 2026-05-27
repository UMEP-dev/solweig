"""Typed read-only views over :class:`SurfaceData`.

The architecture review (``ARCHITECTURE_REVIEW.md`` §1) flagged
:class:`solweig.SurfaceData` as a god-object: it holds geometry inputs,
optical properties, preprocessed auxiliary data, lifecycle state, and a
buffer pool all on one class. Decomposing the dataclass itself would be
a high-risk refactor for limited near-term gain (the public API is in
constant use and the lifecycle machinery is intertwined with the
factory methods).

This module takes a lighter-touch approach: define three small read-only
**views** that group :class:`SurfaceData`'s fields by concern. Each view
is a thin wrapper that defers to the live :class:`SurfaceData` instance,
so there is no field duplication and no lifecycle coupling. Internal
callers can opt in to the views where they improve readability;
existing field access (``surface.dsm``, ``surface.svf``, …) keeps
working unchanged.

The intent is conceptual clarity, not enforcement. If a future refactor
re-shapes :class:`SurfaceData` as a composition of these views, the call
sites that already use them will not have to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from .precomputed import ShadowArrays, SvfArrays
    from .surface import SurfaceData


@dataclass(frozen=True)
class SurfaceGeometryView:
    """User-provided geometry inputs (rasters + grid metadata + height flags).

    All attributes proxy to the underlying :class:`SurfaceData` — mutating
    the surface's fields is reflected here. Read-only by convention.
    """

    _surface: SurfaceData

    @property
    def dsm(self) -> NDArray[np.floating]:
        """Digital Surface Model raster (required, absolute elevations)."""
        return self._surface.dsm

    @property
    def cdsm(self) -> NDArray[np.floating] | None:
        """Canopy DSM raster — vegetation top heights, or ``None`` if absent."""
        return self._surface.cdsm

    @property
    def dem(self) -> NDArray[np.floating] | None:
        """Digital Elevation Model (bare-earth) raster, or ``None`` if absent."""
        return self._surface.dem

    @property
    def tdsm(self) -> NDArray[np.floating] | None:
        """Trunk-zone DSM raster — vegetation trunk heights, or ``None`` if absent."""
        return self._surface.tdsm

    @property
    def pixel_size(self) -> float:
        """Grid pixel size in metres (assumes square pixels)."""
        return self._surface.pixel_size

    @property
    def shape(self) -> tuple[int, int]:
        """``(rows, cols)`` of the surface grid."""
        return self._surface.shape

    @property
    def max_height(self) -> float:
        """Maximum elevation across DSM (and CDSM if present), in metres.

        Used by the shadow ray-marcher to bound vertical reach.
        """
        return self._surface.max_height

    @property
    def dsm_relative(self) -> bool:
        """``True`` when the DSM contains height-above-DEM (relative) values."""
        return self._surface.dsm_relative

    @property
    def cdsm_relative(self) -> bool:
        """``True`` when the CDSM contains height-above-ground (relative) values."""
        return self._surface.cdsm_relative

    @property
    def tdsm_relative(self) -> bool:
        """``True`` when the TDSM contains height-above-ground (relative) values."""
        return self._surface.tdsm_relative


@dataclass(frozen=True)
class OpticalPropertiesView:
    """Per-pixel optical inputs and the land-cover classification."""

    _surface: SurfaceData

    @property
    def albedo(self) -> NDArray[np.floating] | None:
        """Per-pixel shortwave albedo raster, or ``None`` if derived from land-cover."""
        return self._surface.albedo

    @property
    def emissivity(self) -> NDArray[np.floating] | None:
        """Per-pixel longwave emissivity raster, or ``None`` if derived from land-cover."""
        return self._surface.emissivity

    @property
    def land_cover(self) -> NDArray[np.integer] | None:
        """UMEP land-cover classification grid, or ``None`` if optical grids are provided directly."""
        return self._surface.land_cover

    @property
    def has_land_cover(self) -> bool:
        """``True`` when a land-cover classification grid is present."""
        return self._surface.land_cover is not None


@dataclass(frozen=True)
class PreprocessedAuxiliaryView:
    """Library-derived auxiliary data (walls, SVF, shadows, valid mask).

    All fields are ``None`` until preprocessing (or ``compute_svf`` /
    wall-aspect computation) populates them. Use :attr:`is_ready` to
    check whether the surface is ready for :func:`solweig.calculate`.
    """

    _surface: SurfaceData

    @property
    def wall_height(self) -> NDArray[np.floating] | None:
        """Per-pixel wall height raster (metres above ground), or ``None``."""
        return self._surface.wall_height

    @property
    def wall_aspect(self) -> NDArray[np.floating] | None:
        """Per-pixel wall facing direction (radians, 0 = east), or ``None``."""
        return self._surface.wall_aspect

    @property
    def svf(self) -> SvfArrays | None:
        """Bundle of Sky View Factor rasters, or ``None`` before preprocessing."""
        return self._surface.svf

    @property
    def shadow_matrices(self) -> ShadowArrays | None:
        """Bitpacked patch-shadow matrices for the anisotropic sky model, or ``None``."""
        return self._surface.shadow_matrices

    @property
    def valid_mask(self) -> NDArray[np.bool_] | None:
        """Boolean mask of pixels with finite data across all required layers."""
        return self._surface.valid_mask

    @property
    def has_walls(self) -> bool:
        """``True`` when both wall_height and wall_aspect are populated."""
        return self._surface.wall_height is not None and self._surface.wall_aspect is not None

    @property
    def has_svf(self) -> bool:
        """``True`` when the SVF bundle has been populated."""
        return self._surface.svf is not None

    @property
    def is_ready(self) -> bool:
        """True when the surface has the minimum data for :func:`calculate`.

        At minimum: SVF must be present. Walls are optional but the wall
        radiation model degrades to a default if absent.
        """
        return self.has_svf
