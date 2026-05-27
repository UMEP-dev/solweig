# Surface views

`SurfaceData` carries a lot of data — DSM, vegetation canopies, walls,
SVF, shadow matrices, optical inputs, height-mode flags, lifecycle
state. To make it easier to reason about *which concern* a piece of
code is touching, `SurfaceData` exposes three read-only views that
group its fields by purpose:

| View | What it covers |
| --- | --- |
| ``surface.geometry`` | User-provided geometry inputs: DSM, CDSM, DEM, TDSM, pixel size, shape, height-mode flags |
| ``surface.optical`` | Optical inputs: albedo, emissivity, land-cover |
| ``surface.auxiliary`` | Library-derived auxiliary data: wall height/aspect, SVF, shadow matrices, valid mask |

The views are thin proxies — every attribute is a one-line lookup on
the underlying `SurfaceData`, so there is no performance cost and no
duplication of state.

## Example

```python
import solweig

surface = solweig.SurfaceData.prepare(dsm="dsm.tif", working_dir="cache/")

# Old (still works):
rows, cols = surface.dsm.shape
has_walls = surface.wall_height is not None and surface.wall_aspect is not None

# Same code through the views:
rows, cols = surface.geometry.shape
has_walls = surface.auxiliary.has_walls
```

The views also expose a couple of convenience predicates that the raw
fields don't:

- ``surface.auxiliary.has_walls`` — ``True`` iff both wall_height
  and wall_aspect are populated
- ``surface.auxiliary.has_svf`` — ``True`` iff the SVF bundle has
  been computed
- ``surface.auxiliary.is_ready`` — ``True`` iff the surface has the
  minimum data for :func:`solweig.calculate` (currently: needs SVF)
- ``surface.optical.has_land_cover`` — ``True`` iff a land-cover
  classification grid is present

## When to use them

- **Reading code that already uses the views:** internal modules
  (``computation.py`` in particular) route through them for clarity.
  Newly-written internal code should prefer the views.
- **External callers:** the old field access (``surface.dsm``,
  ``surface.svf``) is unchanged and continues to work. Use the views
  if you find them clearer, ignore them otherwise.
- **Type checking:** the views are typed (``SurfaceGeometryView``,
  ``OpticalPropertiesView``, ``PreprocessedAuxiliaryView``), so a
  type-checked codebase gets clearer signatures when functions accept
  one of these instead of a whole `SurfaceData`.

## Immutability

The views are frozen dataclasses — you cannot assign to
``surface.geometry.dsm = …``. To change a field, mutate the underlying
``SurfaceData`` directly (and observe the
[invariant about not mutating arrays after passing them to
calculate()](https://github.com/UMEP-dev/solweig/blob/main/INVARIANTS.md)).

## Why this exists

The architecture review identified ``SurfaceData`` as a god-object
with nine distinct responsibilities. Decomposing it directly into
typed sub-objects would be a breaking change; the views are the
middle path — internal callers and type-checked external code get
the grouping benefit, the public API stays compatible. See
[ARCHITECTURE_REVIEW.md](https://github.com/UMEP-dev/solweig/blob/main/ARCHITECTURE_REVIEW.md#1-surfacedata-is-a-god-object--and-the-natural-decomposition-is-obvious)
§1 for the deeper rationale.
