# Geospatial helpers

`solweig.geospatial` collects the raster-bounds / resampling / alignment
helpers used by the QGIS plugin and by external batch pipelines that need
to drive SOLWEIG over many tiles or sites. **This is the documented
import path for plugin authors and downstream tools.**

```python
from solweig.geospatial import (
    extract_bounds,
    intersect_bounds,
    resample_to_grid,
    pixel_size_tag,
    compute_max_tile_pixels,
    looks_like_relative,
    namespace_to_dict,
    wallalgorithms,
)
```

## Why this module exists

These helpers live in their natural homes (`solweig.utils`, `solweig.cache`,
`solweig.tiling`, `solweig.physics.wallalgorithms`). They were briefly
promoted to the top-level `solweig` namespace in 0.1.0b84 so the QGIS
plugin could stop reaching into internals. That polluted the user-facing
namespace with GIS plumbing, so 0.1.0b85 introduced `solweig.geospatial`
as a stable facade that groups them by use case.

Top-level access (`solweig.extract_bounds`, etc.) was deprecated in
b85, emitted a `DeprecationWarning` through b86, and **removed in b87**.
Import from `solweig.geospatial` only.

## What's here

| Function | Purpose |
| --- | --- |
| ``extract_bounds(geotransform, shape)`` | Compute ``[minx, miny, maxx, maxy]`` from a GDAL geotransform + array shape |
| ``intersect_bounds(bounds_list)`` | Intersection of an iterable of bounding boxes — used to align multi-layer rasters |
| ``resample_to_grid(arr, src_gt, target_bbox, pixel_size, ...)`` | Resample a raster to a target bounding box + resolution (nearest / bilinear) |
| ``pixel_size_tag(pixel_size)`` | Stable cache-directory tag for a given pixel size (e.g. ``"px1"``) |
| ``compute_max_tile_pixels(context=...)`` | Resource-aware tile-size cap for the current host (GPU / RAM budget aware) |
| ``looks_like_relative(layer, reference_surface)`` | Heuristic: does this raster contain height-above-ground rather than absolute elevation? |
| ``namespace_to_dict(ns)`` | Recursive ``SimpleNamespace`` → ``dict`` for JSON serialization of params / materials |
| ``wallalgorithms`` | Goodwin wall-aspect filter module (used by the wall-computation step) |

The functions themselves are documented in their source modules; the
re-exports here are identity-equal to the canonical locations.

## Migrating from the top-level namespace

Old (no longer supported — `AttributeError` since b87):

```python
from solweig import extract_bounds, resample_to_grid  # AttributeError
```

New:

```python
from solweig.geospatial import extract_bounds, resample_to_grid
```

The function signatures and behaviour are unchanged.
