# Release notes

## 0.1.0b100 — 2026-11-11

Five corrections to the ground view factor and its inputs, found by a new
end-to-end comparison with the reference implementation, and six input
checks. Of the five, only the height change moves the scores at the
validation sites.

- **Raster edge.** Ground-reflected shortwave and ground-emitted longwave
  were too low within the ground view factor's search range of the raster
  edge (22 m for a standing person). Steps of the search that left the
  raster were dropped; the reference keeps the last value seen. On a 40 m
  test scene Kup and Lup were up to 24 W/m² low and Tmrt up to 2.6 °C low.
  Pixels further from the edge are unchanged.
- **Posture change on one surface.** A second `calculate()` on the same
  `SurfaceData` with a different posture reused the ground view factor
  geometry of the first. It is now rebuilt.
- **Building mask without a land-cover grid.** Ground within 25 m of a wall
  was treated as building by the ground view factor. The mask is now every
  pixel 2 m or more above the DEM, as in the reference. Without a DEM, the
  10th-percentile DSM height away from walls stands in for ground level.
- **Ground albedo without a land-cover grid** is 0.20 (cobblestone), as in
  the reference. It was 0.15.
- Together the last two raise daytime Tmrt by 0.5 to 1.0 °C on average when
  the validation sites are run without their land-cover grids, and bring
  those runs to within 0.2 °C RMSE of the runs with land cover. Runs with a
  land-cover grid are not affected by either.
- **Ground view factor height.** Ground-reflected shortwave and
  ground-emitted longwave are now evaluated at the height of the body's
  centre of gravity for the posture (1.1 m standing, 0.75 m sitting), as in
  the reference implementation. Up to b99 the stature
  (`HumanParams.height`, 1.75 m by default) was used. Stature now affects
  PET only. At the validation sites the grid-mean Tmrt changes by 0.02 to
  0.13 °C and single pixels by up to 1.9 °C; the Tmrt RMSE range moves from
  3.3–7.0 °C to 3.3–6.9 °C.
- **Unknown output names** in `outputs` raise `ConfigurationError` before
  anything is written. They previously produced folders of empty rasters.
- **Changed location.** A weather series reused with a different `Location`
  has its sun positions recomputed. They were previously kept from the
  first run.
- **EPW gaps.** `Weather.from_epw` logs how many missing values it replaced
  with defaults, per variable.
- **Isotropic fallback.** The anisotropic sky is the default but needs
  shadow matrices. A run without them logs that it used the isotropic sky,
  and `run_metadata.json` records what ran.
- **Tiles too small to work** (core under 32 pixels) raise
  `ConfigurationError` instead of running very slowly.
- **CRS mismatch.** A warning is logged when an input raster's CRS differs
  from the DSM's.

One difference from the reference runner is kept on purpose. Without a
land-cover grid the runner uses a ground `Tstart` of +3.41 and with one it
uses −3.41 for the same material. This package uses −3.41 on both paths.
The inconsistency has been reported upstream.

## 0.1.0b99 — 2026-11-11

No change for Python or QGIS users, and no numerical change. The Rust core
gains a public interface for other Rust projects (issue 14).

- **`rustalgos::api`.** The crate's one public module re-exports the
  entry points that other Rust code may call: `calculate_shadows_rust`,
  `calculate_svf_inner`, `utci_single`, `compute_timestep_rust` and
  `precompute_gvf_geometry_cpu`, with their input and result types. Changes
  to anything in this module will be listed here as breaking.
- **`compute_timestep_rust`.** The per-timestep model (shadows, ground
  temperature, ground view factors, radiation, Tmrt) is now a plain Rust
  function taking `ndarray` views. The Python `compute_timestep` borrows the
  numpy arrays and calls it. The code was moved without modification.
- The crate is also built as an `rlib`.

## 0.1.0b98 — 2026-11-10

Robustness release. The model is unchanged and the validation numbers are
identical to b97.

- **GPU reuse of a previous surface's inputs.** When more than one surface
  of the same grid size was calculated in one Python process, the GPU could
  reuse data uploaded for an earlier surface. Three paths were affected:
    - *Shadows.* A new surface whose DSM array landed on the memory address
      of a freed one was shadowed from the old heights. In a loop over three
      layouts, 3 of 9 runs had about a sixth of the pixels mis-shadowed and
      Tmrt off by up to 25.6 °C.
    - *Anisotropic sky.* Shadow matrices were recognised by address and a
      few sampled values.
    - *Ground view factors.* Alternating between two surfaces of the same
      size gave Tmrt off by up to 7 °C.

  All three now follow the content or identity of the inputs. A single
  `calculate()` run on one surface in a fresh process was not affected. No
  data was ever shared with other programs. Per-timestep speed is unchanged.
- **Shape errors instead of crashes.** `calculate()` checks every layer's
  shape against the DSM first and raises `GridShapeMismatch` naming the
  layer. A mis-shaped albedo, emissivity or SVF array previously aborted
  the interpreter.
- **Stale cache detection.** SVF and wall caches are validated on every
  value of the DSM, CDSM and trunk layer. A small edit to a large raster
  was usually missed before. Existing caches stay valid.
- **Hour totals.** A warning is logged when the summary's sun, shade and
  UTCI exceedance hours will over-count because weather records are further
  apart than their timestep or unevenly spaced. The totals are unchanged.
- CI now runs the slow-marked SVF, GVF and wall-geometry golden tests.

## 0.1.0b97 — 2026-11-10

Two model changes. Results change for anisotropic runs without a canopy
layer and for the opt-in 2026a ground scheme; all other runs are unchanged.

- **Anisotropic sky without vegetation** (issue 15). On a surface with no
  CDSM, sky patches blocked by buildings were treated as vegetation and
  emitted longwave at air temperature, so the sunlit wall temperature was
  not used. Adding any canopy pixel, however distant, restored the correct
  behaviour. Building patches now use wall emission in both cases. Tmrt
  beside a sunlit wall rises by about 2 to 4 K in runs without vegetation.
  SVF computed in tiles had a related fault (every patch, open sky
  included, was treated as vegetation) and is corrected too.
  Existing SVF caches and shadow matrices exported from UMEP do not need
  recomputing: the calculation no longer reads the stored vegetation
  matrices when the surface has no CDSM.
- **2026a ground scheme aligned with umep-solweig.** The reference
  implementation is now
  [UMEP-dev/umep-solweig](https://github.com/UMEP-dev/umep-solweig). Two of
  its changes are carried over: the cylinder longwave term in anisotropic
  mode uses the anisotropic-sky side longwave alone, and the water slab
  temperature is stepped from the previous-step value. Against the
  Gothenburg measurements the scheme's Tmrt RMSE falls from 12.3 to
  21.5 °C to 4.5 to 11.4 °C. The scheme stays off by default; see the
  validation report.
- `scripts/validate_ground_scheme.py` reproduces the scheme field
  comparison.

## 0.1.0b96 — 2026-10-01

Validation-harness correction. No library, plugin or model behaviour
changed; the validation numbers did.

- **Met-file clock corrected at the three Gothenburg validation sites.** The
  prepared met files are labelled one hour early relative to the hour-ending
  CET convention that the measurement CSVs and the model's sun position both
  follow. Every validation timestep was therefore driven by forcing recorded
  one hour after its sun position, which put the modelled K↓ peak an hour
  before the measured one. The harness now adds one hour to the met
  timestamps (`tests/validation/met_clock.py`); the measurement CSVs and the
  library are untouched. Tmrt RMSE across the seven site-days moves from
  2.4 to 7.3 °C to 3.3 to 7.0 °C, with Gustav Adolfs torg improving most
  (5.7 to 7.3 °C before, 3.3 to 4.8 °C after). The K↓ "spike" previously
  attributed to cloud on 2006-08-01 was this clock error.
- The opt-in 2026a ground-scheme field comparison was rerun on the corrected
  clock; see the validation report.

## 0.1.0b95 — 2026-08-25

Plugin-only fix, no numerical or library behaviour change: a Qt6-scoped enum
in the EPW download utility, flagged by the QGIS plugin repository's Qt6
compatibility validator on the beta94 upload.

## 0.1.0b94 — 2026-08-25

Bug-fix release, no numerical change (validation 31/31 pass, numbers identical
to b93).

- **`SOLWEIG_NO_GPU` is now honoured on runs that load precomputed SVF.**
  Previously the env var only took effect through `is_gpu_available()` or
  `compute_svf()`, so a `calculate()` run with a prepared surface silently
  kept the GPU enabled. A regression test asserts zero GPU dispatches under
  the env var. Behaviour with the GPU enabled is unchanged.
- **QGIS plugin**: a missing EPW file in EPW weather mode now raises a clear
  error instead of an `assert` (which `python -O` strips).
- **Docs**: a tutorials overview page explains how to run the notebooks
  locally (the data ships with the repository), every tutorial page links to
  it, and the home page and README state the package's positioning: an
  experimental compatibility-focused implementation, with UMEP as the
  reference.

## 0.1.0b93 — 2026-08-24

Bug-fix release, no numerical change (validation 31/31 pass, numbers identical
to b92).

- **QGIS plugin failed to load** with "Errors parsing metadata.txt": unescaped
  `%` signs in the plugin metadata broke QGIS's parser (configparser
  interpolation). Escaped, and guarded by a test that parses the shipped file
  the way QGIS does.
- **Stale SVF caches are caught at load time.** A prepared surface directory
  holding an SVF cache computed for a different raster (an earlier run on a
  different extent or resolution) was previously loaded blindly and surfaced
  much later as a bare grid-shape error. `SurfaceData.load()` and
  `PrecomputedData.prepare(expected_shape=...)` now skip mismatched caches
  (logging the offending path) and raise the new
  `solweig.StalePrecomputedData` — naming the cache path, both grid shapes,
  and the fix — when only mismatched caches exist. `validate_inputs()` also
  shape-checks user-supplied precomputed SVF and shadow matrices against the
  DSM.

Concise user-facing summary of changes. For the full per-version history of every
commit, see [`qgis_plugin/solweig_qgis/metadata.txt`](https://github.com/UMEP-dev/solweig/blob/main/qgis_plugin/solweig_qgis/metadata.txt)
and individual git commits.

## 0.1.0b92 — 2026-07-09

Performance and GPU-utilisation work. Validation is unchanged (31/31 pass) and
output is byte-identical except where noted. The main change: the per-tile
ground view factor (GVF) geometry precompute, previously a CPU-only step that
left the GPU idle for roughly 0.6 to 1.2 s per tile, now runs on the GPU (about
2.7× faster), matching the CPU result to ~6e-8.

- **GVF geometry on the GPU.** A new wgpu compute kernel does the GVF geometry
  precompute, with a CPU fallback and an
  `enable`/`disable`/`is_gvf_precompute_gpu_enabled` toggle, running as a
  distinct labelled phase per tile. Precompute and the per-timestep GVF share
  one GPU context, so the geometry is computed once into resident buffers and
  read in place across all of a tile's timesteps (no readback/re-upload).
  Byte-identical: GPU and CPU Tmrt match exactly.
- **Empty-tile compute skip.** Tiles that fall entirely outside the valid data
  (irregular extents) skip the whole shadow/GVF/aniso/radiation compute. Output
  is byte-identical on valid pixels; the only change is that shadow over nodata
  is now NaN rather than a spurious "sunlit" 1.0.
- **Async and overlapped I/O.** Tiled GeoTIFF writes run on a background thread,
  the next tile's data is prefetched while the current tile computes, and the
  summary GeoTIFF export is written in parallel (about 5× faster, concurrency
  bounded by grid size so large rasters don't spike memory). All byte-identical;
  disable with `SOLWEIG_ASYNC_OUTPUT=0`.
- **Night-shadow skip.** With the sun below the horizon there are no cast
  shadows, so the shadow ray-march is skipped (byte-identical).
- **Fix.** Isotropic runs over an irregular extent could panic in the
  side-radiation kernel (a non-contiguous cropped valid mask); the mask is now
  made contiguous at the FFI boundary.

## 0.1.0b91 — 2026-07-08

Large-raster memory reductions. **No numerical change**: output is
byte-identical, and golden tests and validation site numbers are unchanged from
b90 (validation 31/31 pass). Measured cold Madrid (507 megapixels, isotropic):
prepare peak 35.6 → 27.3 GB, overall peak RSS 38.1 → 32.9 GB.

- Base-layer load and alignment in `prepare()` processes one layer at a time
  through disk-backed storage instead of holding the whole layer stack as
  resident RAM.
- `findwalls` uses pairwise `np.maximum` instead of stacking the four cardinal
  neighbour grids, removing an ~8 GB whole-raster temporary (the largest
  hard-RAM spike). Element-wise max and NaN handling are identical, so wall
  outputs are unchanged.
- The tiled-SVF output arrays and the tiled-calc summary grids no longer
  pre-fill the whole memmap up front and are flushed per tile. The tiles cover
  every pixel, so this is byte-identical while keeping resident memory to the
  active tile.
- Fixed a latent macOS hazard: `np.asarray(memmap)` returns a plain-ndarray
  view, so an `isinstance` check missed disk-backed layers and a save over a
  live mapping could wedge the process in uninterruptible kernel state.
  Disk-backed layers are now detected by walking the array's base chain.

Known reproducibility note: large-raster `calculate()` output is deterministic
for a given tile size but shifts by ~0.029 °C mean (a small near-threshold tail
up to ~3 °C) across different tile sizes, because the compute kernels round
differently for different tile array shapes. This is floating-point-class, not
a correctness issue; pin `tile_size` for bit-reproducible large-raster runs.

## 0.1.0b90 — 2026-07-07

Opt-in UMEP 2026a ground-surface scheme (`use_ground_scheme` /
`use_outgoing_longwave`), Python API only, not exposed in the plugin UI.
Defaults off, so model output is unchanged from b89 (golden output
byte-identical, validation 31/31 pass with b89 numbers). The scheme requires a
land-cover grid, rejects tiled processing, and currently runs warm against the
validation sites; a question about its radiation composition is open with the
upstream developers. See `VALIDATION.md` for the field comparison.

## 0.1.0b89 — 2026-07-06

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
[`VALIDATION.md § Ldown positive bias`](https://github.com/UMEP-dev/solweig/blob/main/VALIDATION.md#ldown-positive-bias).
