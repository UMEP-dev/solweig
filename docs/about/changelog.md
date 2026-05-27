# Release notes

Concise user-facing summary of changes. For the full per-version history of every
commit, see [`qgis_plugin/solweig_qgis/metadata.txt`](https://github.com/UMEP-dev/solweig/blob/main/qgis_plugin/solweig_qgis/metadata.txt)
and individual git commits.

## 0.1.0b85 — 2026-05-26

Architecture stabilisation pass. **No numerical change** — golden tests pass
byte-identical, validation site numbers unchanged.

- **Rust FFI bundling.** `pipeline.compute_timestep` signature drops from
  43 positional arguments to 18 via two new PyO3 classes:
  - `pipeline.SvfBundle` groups the 17 SVF / SVF-veg / SVF-aveg /
    svfbuveg / svfalfa rasters
  - `pipeline.StateBundle` groups the 9 thermal-state fields and
    carries an FFI **version field** (`pipeline.STATE_BUNDLE_VERSION`).
    The bundle constructor raises `ValueError` on version mismatch so
    a stale Python/Rust pairing fails loudly instead of silently
    mis-mapping fields.
- **`solweig.geospatial` submodule.** The geospatial helpers previously
  promoted to the top level (`extract_bounds`, `intersect_bounds`,
  `resample_to_grid`, `pixel_size_tag`, `compute_max_tile_pixels`,
  `looks_like_relative`, `namespace_to_dict`, `wallalgorithms`) have moved
  to `solweig.geospatial`. Top-level access still works for backwards
  compatibility but emits a `DeprecationWarning` (removal target: 0.1.0b88
  or first 0.2.x).
- **`Settings` dataclass.** New typed `solweig.models.settings.Settings`
  with explicit override semantics (per-call kwargs > `ModelConfig` >
  bundled defaults). Replaces the 50-line override block that used to
  live in `calculate()`. No public-API change.
- **`SurfaceData` typed views.** New read-only views `surface.geometry`,
  `surface.optical`, `surface.auxiliary` group the SurfaceData fields by
  concern. Used internally by the production compute path; existing
  attribute access (`surface.dsm`, `surface.svf`, …) is unchanged.
- **Cache-key hardening.** `computation._arr_key` now includes witness
  bytes from the array's first/middle/last element, catching in-place
  mutations that the previous `(ctypes.data, shape)` key missed. The
  documented invariant remains "don't mutate surface arrays after
  passing them to `calculate()`" — see
  [INVARIANTS.md](https://github.com/UMEP-dev/solweig/blob/main/INVARIANTS.md).
- **Foundation docs (repo-only):**
  [PRINCIPLES.md](https://github.com/UMEP-dev/solweig/blob/main/PRINCIPLES.md)
  (what the library is for, the four identities it serves, the architectural
  rules) and
  [INVARIANTS.md](https://github.com/UMEP-dev/solweig/blob/main/INVARIANTS.md)
  (the seven load-bearing assumptions the code relies on but does not always
  enforce).
- **Repository audit script.** `poe audit` runs eight measured signals
  and writes `AUDIT.md` at the repo root. Wired into CI as an
  informational job. Tracks Rust panic surface, type strictness, test
  coverage, CI/poe gap, public-API discipline, docstring coverage,
  hot files, and dependency freshness.
- **Rust panic hardening (continued).** `vegetation.rs` panic surface
  reduced from 44 to 14 sites via a documented `as_slice_checked` helper
  that captures the contiguity invariant in one place. Repository-wide
  Rust panic rate drops to 3.15 / kloc (under the 5.0 threshold).
- **CI:** new `audit` and `test-slow` jobs; `test-spec` now includes slow
  spec tests (UMEP parity, anisotropic pipeline) so drift doesn't sneak
  through the fast gate.

## 0.1.0b84 — 2026-04-16

Wall-aspect kernel refactor + small public-API tidy-up. No physics changes.

- **Wall-aspect kernel** now uses `f64` internals and banker's rounding to match
  numpy's `np.round`, bringing the Rust kernel closer to UMEP numerics. The
  Python `wallalgorithms.py` Goodwin fallback was deleted — the Rust kernel is
  the single code path for both pip and QGIS users.
- **Public API surface** explicitly exposed on `solweig/__init__.py`:
  `extract_bounds`, `intersect_bounds`, `resample_to_grid`,
  `namespace_to_dict`, `pixel_size_tag`, `compute_max_tile_pixels`,
  `looks_like_relative`, `wallalgorithms`. The QGIS plugin no longer reaches
  into internals.
- QGIS plugin error wrapping: `QgsProcessingException` messages now expose
  `SolweigError` structured attributes (`field`, `expected`, `got`, `reason`, …).

## 0.1.0b83 — 2026-04-13

Documentation-only fix.

- PVGIS TMY reference period corrected in docs: **2005–2023** (v5.3 release),
  not 2005–2020. Clarified that TMY row timestamps legitimately span multiple
  years (each month is a real historical month).

## 0.1.0b82 — 2026-04-11

Shadow scale-convention fix (correctness) + cache and tile-sizing improvements.

- **Shadow caster (Rust):** Fixed silent scale-convention inversion in the
  `tan_altitude_by_scale` operator. Before this release, shadow length was
  physically wrong at non-1 m pixel sizes. Validation RMSE improved
  dramatically at Gustav Adolfs (9.3–18.9 °C → 5.7–7.5 °C) and GVC
  (11.5–15.6 °C → 1.5–6.1 °C).
- **DEM stair-step smoothing** with automatic int16 1 m quantization detection.
- **`prepare()` warm-run fast-path** — ~100× speedup on cache hit with detailed
  mismatch logs.
- Tile sizer no longer emits a spurious "buffer exceeds limit" warning.
- `GridAccumulator.update()` optimised with in-place numpy ufuncs.

## Known systematic bias (all versions)

Modelled downwelling longwave (L↓) is consistently +18 to +55 W/m² above
observations across all validation sites. This is a formulation issue
inherited from UMEP (Jonsson et al. 2006: non-sky hemisphere filled with
wall emissions at air temperature, while real shaded walls are cooler) and
is not run- or sun-position-dependent. See
[`VALIDATION.md § Ldown overestimation`](https://github.com/UMEP-dev/solweig/blob/main/VALIDATION.md#ldown-overestimation).
