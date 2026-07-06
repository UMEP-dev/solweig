# Release notes

Concise user-facing summary of changes. For the full per-version history of every
commit, see [`qgis_plugin/solweig_qgis/metadata.txt`](https://github.com/UMEP-dev/solweig/blob/main/qgis_plugin/solweig_qgis/metadata.txt)
and individual git commits.

## 0.1.0b89 — unreleased

Correctness release from a full-repository review. **Numerical output
changes** for two classes of runs; see `VALIDATION.md` for the audit
trail.

### Physics fixes

- **Clearness index pressure unit**: `Weather.pressure` (hPa) was scaled
  as if it were kPa inside the Crawford & Duchon transmission formula,
  underestimating clear-sky irradiance by ~20–25 % and disabling the
  Ldown cloud correction for hazy skies. EPW/values-driven runs change;
  validation-site numbers moved marginally (sites use measured
  radiation).
- **GVF source area at non-1 m pixels**: the metres→pixels conversion
  multiplied by pixel size where UMEP divides. At 2 m pixels the ground
  view factor integrated over 144 m instead of 36 m (at 0.5 m, 9 m). A
  new parity gate pins GVF against UMEP's `gvf_2018a` at 0.5/1/2 m.
  1 m runs are unchanged; the 2 m validation sites now reproduce UMEP's
  own output.
- **Cached vs uncached GVF**: the per-timestep cached path counted
  partially sunlit walls in its sunwall mask; all paths now share
  UMEP's fully-sunlit semantics.

### Reliability fixes

- Tiled SVF preprocessing crashed with `NameError` after computing all
  tiles; it now completes and is covered by tiled-vs-single-shot parity
  tests.
- Cancelling SVF preprocessing (QGIS) no longer persists a partial SVF
  cache or a prepare fingerprint without SVF; cancellation raises the
  new `solweig.ComputationCancelled`, which the plugin handles as a
  graceful stop.
- EPW parsing uses the per-field missing codes from the EnergyPlus spec
  (RH 99 % and GHI/DNI/DHI 999 W/m² are real data again; dry-bulb 99.9
  and pressure 999999 are recognized as missing). `Weather.from_epw`
  with no dates now loads the whole first day as documented.
- `ModelConfig.from_json` actually reads legacy parameter files (the
  `Value` nesting was skipped, so user values silently fell back to
  defaults), normalizing legacy units (Height in cm, Sex as text).
- A stale precomputed shadow-matrix cache from a different patch option
  raises `ValueError` instead of aborting the process; SVF caches are
  keyed on the trunk layer (TDSM/trunk_ratio) so trunk changes
  invalidate; land-cover nodata maps to the 255 sentinel instead of
  becoming "paved"; geographic-CRS inputs are rejected as documented.

### QGIS plugin

- Loads cleanly when solweig is not installed (per-algorithm import
  isolation) and the anisotropic-sky hint points at an algorithm that
  exists in the toolbox.

### Docs

- Tutorials are published as Markdown exported from the notebooks;
  mkdocs-jupyter retired. Chunked-`calculate()` guidance replaced (it
  dropped thermal state); dev-setup commands fixed; specs corrected
  (water-code quirk documented, GVF distance formula, GPU/CPU parity
  claim).

## 0.1.0b88 — 2026-05-27

Internal-only release — no library, plugin, or runtime change.

Performance-timing benchmarks moved off CI (where they fired false
positives on GPU-less runners) to a local-only `gpu_perf_gate`
marker. The hardware-stable memory benchmark stays on CI. Run the
full perf suite locally via `poe test_gpu_perf_gate`.

## 0.1.0b87 — 2026-05-27

**No numerical change** — golden tests byte-identical, validation site
numbers unchanged from b82 baseline (31/31 validation tests pass).

### Breaking

The top-level re-exports of geospatial helpers
(`solweig.extract_bounds`, `solweig.intersect_bounds`,
`solweig.resample_to_grid`, `solweig.namespace_to_dict`,
`solweig.pixel_size_tag`, `solweig.compute_max_tile_pixels`,
`solweig.looks_like_relative`, `solweig.wallalgorithms`) — deprecated
in b85 with a `DeprecationWarning` shim — have been **removed**.

Migrate by importing from `solweig.geospatial` instead:

```python
# Before
from solweig import extract_bounds, resample_to_grid

# After
from solweig.geospatial import extract_bounds, resample_to_grid
```

Accessing the old names now raises `AttributeError`. The
`solweig.geospatial` submodule (added in b85) is unchanged.

### GPU / CPU surface

- **`solweig.disable_gpu()` now toggles every GPU path.** Pre-b87 it
  only flipped the shadow path; anisotropic sky and GVF kept running
  on GPU. A single call now genuinely produces a CPU-only run — the
  basis for the new parity tests below.
- **`solweig.enable_gpu()`** — new, mirrors `disable_gpu()`. Re-enables
  all three Rust GPU paths in a single call.
- **Lazy-init fix.** First call to `is_gpu_available()` used to
  unconditionally re-enable shadow GPU, silently overriding any
  prior `disable_gpu()`. Fixed: `_ensure_gpu_initialized` now only
  acts when `SOLWEIG_NO_GPU=1` is set.
- **GPU metrics surface** (new): `solweig.gpu_dispatch_count()`,
  `solweig.gpu_fallback_count()`, `solweig.reset_gpu_metrics()`.
  Thread-safe atomic counters incremented at every shadow / SVF /
  aniso / GVF dispatch site in Rust. Lets tests assert "the GPU
  path actually ran" (paired with `gpu_fallback_count() == 0`).

### New parity / benchmark coverage

- **Shadow + SVF GPU-CPU parity tests** (`tests/spec/test_gpu_cpu_parity.py`).
  Mirrors the existing aniso parity pattern for the two remaining
  GPU paths.
- **GPU-CPU runtime ratio benchmark** (`tests/benchmarks/test_gpu_cpu_benchmark.py`).
  Runs the same scenario back-to-back with GPU on/off, asserts the
  ratio is above a hard 0.5 floor, warns below 1.0. Appends a row to
  `tests/benchmarks/logs/gpu_cpu_ratio_history.md` (gitignored).

### Known difference (documented, not a regression)

The new SVF parity test characterises a small known difference between
the CPU and GPU vegetation-SVF kernels:

- Building SVF fields (`svf*`): **byte-identical** between paths.
- Veg-blocks-building (`svf_aveg*`): **byte-identical**.
- Vegetation-only fields (`svf_veg*`): up to ~0.042 absolute drift in
  <1% of pixels, exclusively at canopy-edge pixels.

The drift values are integer multiples of the per-patch SVF weight
(~0.012) — i.e. 1–2 patches at the visible/blocked boundary disagree
between the two implementations. Both are correct to f32 precision;
the propagated Tmrt difference is below 0.5 °C (verified by
`test_shadow_field_gpu_vs_cpu_match`), well under any physically
meaningful threshold.

## 0.1.0b86 — 2026-05-27

Internal tidy-and-tighten pass following b85. **No numerical change** — golden
tests pass byte-identical, validation site numbers unchanged from b82
baseline.

- **Hot-file decomposition (continued).** Five Python modules split below
  the 700-line audit threshold; every public symbol re-exported from its
  original module so no caller breaks:
  - `models/surface.py` 3,016 → 1,731 (extracted `surface_loading`,
    `surface_compute`, `surface_svf_tiled`, `surface_serialization`,
    `surface_views`)
  - `io.py` 1,259 → 678 (extracted `io_epw`, `io_preview`)
  - `summary.py` 935 → 493 (extracted `grid_accumulator`)
  - `models/weather.py` 848 → 642 (extracted `models/location`)
  - `models/precomputed.py` 804 → 560 (extracted `models/shadow_arrays`)
  Audit hot-file count: 16 → 11 (remaining are Rust + `surface.py` +
  `physics/sun_position.py`).
- **Test coverage 75% → 81%.** 117 new focused unit tests across
  `models/results`, `models/location`, `models/config`,
  `components/gvf`, `components/svf_resolution`, `io_epw`,
  `models/weather` (UMEP met parser), `models/shadow_arrays`,
  `models/precomputed` (zip + memmap fixtures), `io` (bbox helpers,
  preview, north-up guards), and `errors`. Coverage axis flips green
  for the first time.
- **`ThermalState` top-level export.** Now reachable at
  `solweig.ThermalState` — previously only at
  `solweig.models.state.ThermalState`. Promoted to a top-level export
  for advanced/internal use; thermal-state threading is handled
  internally by the timeseries loop (`calculate()` takes no `state`
  argument).
- **QGIS plugin: 349 LOC of dead code removed.** Deleted
  `create_surface_from_parameters` and three helpers (the fossilised
  pre-`prepare()` loader). The production path already used
  `SurfaceData.prepare()` directly; the dead function was only kept
  alive by its own tests. Plugin metadata changelog also trimmed from
  254 → 96 lines.
- **Public docs reshaped.**
  - Physics nav now renders the 8 component specs inline (SVF, shadows,
    GVF, radiation, ground temperature, Tmrt, UTCI, PET) via
    `mkdocs-include-markdown-plugin`. `specs/` remains the canonical
    source so parity tests in `tests/spec/` and the public docs share
    one file.
  - Developer-focused docs (`PRINCIPLES.md`, `INVARIANTS.md`,
    `ARCHITECTURE.md`) moved to repo root; the public site's Development
    section is now just Contributing. The `docs/development/audit.md`
    wrapper page was deleted (`AUDIT.md` is already generated at root).
  - API docs gain `SurfaceData.prepare`, the four GPU helpers
    (`is_gpu_available`, `get_compute_backend`, `disable_gpu`,
    `get_gpu_limits`), and the `Settings` dataclass entry.
- **Audit script gains a 9th axis.** `axis_canonical_docs` validates
  that the 10 repo-root canonical docs are present, have inbound
  references from other canonical docs (no orphans), and that every
  relative `.md` link inside them resolves to an existing file.
  Catches the kind of drift the `docs/development/principles.md` →
  `PRINCIPLES.md` move would otherwise have created.
- **Benchmark gates tightened.** `tests/benchmarks/test_performance_matrix_benchmark.py`
  absolute budgets cut from 1.5–4.0 s to 0.5–0.85 s (observed medians
  are 0.10–0.15 s on M-series Mac; ~4× headroom locally, ~8× in CI
  under `SOLWEIG_PERF_BUDGET_SCALE=2.0`). Ratio caps tightened too:
  aniso/iso 5→2, tiled/non-tiled 4→2.5, plugin/api 6→2. The old values
  would have masked any regression under ~10×.
- **Plugin wrapper cleanup.** Inlined the 10-line
  `_looks_like_relative_heights` wrapper; callers now import
  `solweig.geospatial.looks_like_relative` directly.

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
