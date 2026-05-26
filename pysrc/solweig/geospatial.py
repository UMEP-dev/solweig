"""Geospatial helpers for downstream consumers (QGIS plugin, batch pipelines).

Per [docs/development/principles.md](development/principles.md), the QGIS
plugin is a downstream consumer of the SOLWEIG library — not part of it.
The plugin needs a handful of raster-bounds / resampling utilities that
SOLWEIG itself uses internally. Rather than have the plugin reach into
``solweig.utils`` or rely on the top-level :func:`solweig` namespace
being polluted with GIS plumbing, those helpers are re-exported here.

This module is the **documented import path for plugin / batch authors**:

>>> from solweig.geospatial import extract_bounds, intersect_bounds, resample_to_grid

The functions themselves live in their natural homes (``solweig.utils``,
``solweig.cache``, ``solweig.tiling``); this is a thin facade that groups
them by use case.
"""

from __future__ import annotations

from .cache import pixel_size_tag
from .models.surface import looks_like_relative
from .physics import wallalgorithms
from .tiling import compute_max_tile_pixels
from .utils import extract_bounds, intersect_bounds, namespace_to_dict, resample_to_grid

__all__ = [
    # Raster bounds + alignment
    "extract_bounds",
    "intersect_bounds",
    "resample_to_grid",
    # Cache directory naming (pixel-size-keyed subdirectories)
    "pixel_size_tag",
    # Tile-size resource calculator (for plugins that drive their own tiling)
    "compute_max_tile_pixels",
    # Heuristic for detecting relative-height rasters
    "looks_like_relative",
    # JSON serialisation helper (used for run-metadata writers)
    "namespace_to_dict",
    # Wall-aspect computation kernel (Goodwin filter)
    "wallalgorithms",
]
