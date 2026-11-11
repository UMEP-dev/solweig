# CLAUDE.md — SOLWEIG Project Guide

## What Is This Project?

SOLWEIG (SOlar and LongWave Environmental Irradiance Geometry-model) is a high-performance urban microclimate model. It computes mean radiant temperature (Tmrt) and thermal comfort indices (UTCI, PET) in complex urban environments.

- **Hybrid Rust/Python** — heavy compute in Rust (PyO3), orchestration and I/O in Python
- **Optional GPU acceleration** — wgpu compute shaders for shadow casting, SVF, GVF, and anisotropic sky
- **Dual geospatial backend** — rasterio (standalone) or GDAL (QGIS plugin), detected lazily at runtime
- **License**: GPL-3.0
- **Version**: 0.1.0 beta series (single source of truth: `pyproject.toml`)

## Read first — architectural anchors

Before making any non-trivial change, skim:

- **[PRINCIPLES.md](PRINCIPLES.md)** — what this
  library is for, the four identities it serves, the architectural rules that
  follow. When facing a "should this go here or there?" question, work from
  this page first.
- **[INVARIANTS.md](INVARIANTS.md)** — the
  load-bearing assumptions the code makes but does not always enforce
  (array layout, immutability, GIL ownership, etc). Violating these
  produces silently wrong results, not crashes.
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — the layered overview (Python API
  → orchestration → fused Rust pipeline → Rust algorithms) and how data
  moves between them.
- **[ARCHITECTURE_REVIEW.md](ARCHITECTURE_REVIEW.md)** — the deeper review the
  principles and invariants pages derive from. Some numerical claims
  (e.g. line counts) are pre-b85 and now out of date; treat it as a
  snapshot of why the current architecture is shaped this way, not as
  live structural reference.

---

## Quick Reference: Common Commands

```bash
# Environment setup
uv sync --group test --group dev      # Install all dev + test deps
maturin develop --release             # Build Rust extension (MUST be --release)

# Linting & formatting
poe lint                              # ruff format + ruff check --fix
poe typecheck                         # ty check (NOT mypy)

# Testing
poe test_quick                        # pytest -m 'not slow' -x -q  (fast gate)
poe test_full                         # Full suite including slow tests
poe test_benchmarks                   # Performance regression tests
poe test_gpu_gates                    # GPU vs CPU parity tests
pytest tests/spec/ -x -q             # Scientific property tests only
pytest tests/golden/ -x -q           # Golden regression tests only
pytest tests/validation/ -x -q       # Real-world site validation

# Full verification (lint + typecheck + all tests)
poe verify_project

# Documentation
poe docs                              # mkdocs serve (local preview)
poe docs_build                        # mkdocs build --strict
```

---

## Repository Layout

```
pysrc/solweig/          Python package
  api.py                Main entry point: calculate()
  _compat.py            Backend detection (rasterio vs GDAL) — SINGLE SOURCE OF TRUTH
  io.py                 Raster I/O (branches on GDAL_ENV)
  computation.py        Core computation orchestration (calls into Rust pipeline)
  timeseries.py         Multi-timestep computation
  tiling.py             Large-raster tiling
  summary.py            Output summary generation
  loaders.py            Config/EPW file loaders (RENAMED from config.py)
  models/               Dataclasses: SurfaceData, Weather, Location, ModelConfig, etc.
  physics/              Pure-Python algorithms (RENAMED from algorithms/)
  components/           Ground, GVF, shadows, SVF resolution, 2026a ground-scheme init
  errors.py             Custom exception hierarchy (all inherit SolweigError)
  constants.py          Physical constants (Stefan-Boltzmann, view factors, etc.)
  solweig_logging.py    Logging (NOT logging.py — watch for stale refs)
  _orchestration.py     Internal orchestration helpers
  bundles.py            Data bundle types
  output_async.py       Async output writing
  buffers.py, cache.py, progress.py, walls.py, utils.py, metadata.py, postprocess.py
  data/                 Default JSON configs (params, physics, materials)

rust/src/               Rust extension (PyO3, compiled as solweig.rustalgos)
  lib.rs                PyModule root — 12 submodules, 60+ functions
  api.rs                Rust-callable entry points (pub re-exports; the crate's only public surface)
  pipeline.rs           Fused per-timestep pipeline: compute_timestep_rust + the PyO3 wrapper
  pipeline_inputs.rs    Plain-Rust input structs (ndarray views) for compute_timestep_rust
  shadowing.rs          Shadow casting (CPU + GPU)
  skyview.rs            Sky View Factor
  gvf.rs, gvf_geometry.rs  Ground View Factor + geometry caching
  sky.rs                Anisotropic sky radiation (Perez model)
  vegetation.rs         Tree effects (longwave + shortwave)
  ground.rs             Ground temperature + thermal delay
  ground_surface.rs     UMEP 2026a ground scheme (force-restore/OHM + outgoing-LW march, opt-in)
  utci.rs, pet.rs       Thermal comfort indices
  tmrt.rs               Mean Radiant Temperature
  perez.rs              Perez diffuse sky model math
  sun.rs                Sun-on-surface calculations
  wall_aspect.rs        Wall orientation detection (Goodwin filter)
  morphology.rs         Binary dilation for morphological ops
  patch_radiation.rs, sunlit_shaded_patches.rs, emissivity_models.rs
  gpu/                  wgpu compute shaders (shadow, aniso, GVF) + 6 WGSL files

tests/                  ~1035 tests across ~68 files
  spec/                 Physical property & parity tests (fast)
  golden/               Regression tests against pre-computed baselines
  validation/           Real-world sites: Kronenhuset, Gustav Adolfs, GVC
  benchmarks/           Performance + memory regression gates
  conftest.py           Shared fixtures, RELEASE_BUILD gate
  qgis_mocks.py         QGIS API mocking (no QGIS installation needed)

qgis_plugin/            QGIS Processing plugin (directory: solweig_qgis/)
specs/                  Scientific specifications (11 documents)
docs/                   MkDocs documentation site
```

---

## Architecture: Critical Patterns

### Backend Detection (`_compat.py`)

The rasterio/GDAL backend is determined **lazily** via PEP 562 `__getattr__`:

- `GDAL_ENV` — `True` for QGIS/GDAL, `False` for rasterio, `None` if undecided
- `RASTERIO_AVAILABLE`, `GDAL_AVAILABLE` — boolean flags
- **QGIS path**: detected via `sys.modules`/env vars; **never probes rasterio**
- **Standard path**: prefers rasterio, falls back to GDAL
- **Reload safety**: clears stamped attrs so `__getattr__` re-fires on `importlib.reload()`

**Rules**:
- Never import rasterio/pyproj/shapely unconditionally — always check `GDAL_ENV` first
- `io.py` branches: `if GDAL_ENV:` (GDAL path) / `if GDAL_ENV is False:` (rasterio path)
- QGIS plugin `__init__.py` sets `UMEP_USE_GDAL=1` before any solweig import
- Tests that modify `sys.modules` must access `_compat` attrs BEFORE restoring mocks (inside `try`, not after `finally`)

### Rust/Python Boundary

- **Fused pipeline**: `pipeline.compute_timestep()` does an entire timestep in one FFI call (shadows -> ground temp -> GVF -> radiation -> Tmrt), avoiding intermediate numpy allocations
- **Pure vs. PyO3 pattern**: Internal `*_pure()` functions work with ndarray; PyO3 wrappers handle numpy<->ndarray conversion
- **Zero-copy**: Large arrays passed via `PyReadonlyArray2<T>` (no copy)
- **GPU toggle**: Runtime atomic flags (`enable_gpu()`/`disable_gpu()`), automatic CPU fallback on GPU error
- `SOLWEIG_NO_GPU=1` env var disables GPU entirely

### Data Models

All models use `@dataclass` (not Pydantic):
- Validation in `__post_init__`
- `from __future__ import annotations` everywhere (PEP 563)
- Optional fields typed as `Type | None`
- Arrays typed as `NDArray[np.floating]` from `numpy.typing`
- Expensive type imports behind `if TYPE_CHECKING:` guards

---

## Testing Strategy

Three-layer pyramid:

1. **Spec property tests** (`tests/spec/`) — verify physical invariants from `specs/*.md`, synthetic data, fast
2. **Golden regression tests** (`tests/golden/`) — pre-computed baselines, catch numerical drift
3. **Validation tests** (`tests/validation/`) — full pipeline against real field measurements (Gothenburg sites)

Plus: **benchmark tests** (`tests/benchmarks/`) for performance and memory regression (bytes-per-pixel threshold: 500B)

**Markers**:
- `@pytest.mark.slow` — full SOLWEIG computation, excluded by `poe test_quick`
- `@pytest.mark.validation` — requires external validation datasets
- `@pytest.mark.skipif` — conditional on GPU availability, optional deps (umep, matplotlib)

**QGIS testing**: Uses `qgis_mocks.py` to inject mock QGIS modules into `sys.modules` — no QGIS installation required.

---

## CI Pipeline (GitHub Actions)

| Job | What it does |
|-----|-------------|
| **lint** | `ruff check` + `ruff format --check` |
| **typecheck** | `ty check` on pysrc/, tests/, demos/, scripts/, qgis_plugin/ |
| **test** | Matrix: Python 3.11, 3.12, 3.13 — excludes slow tests |
| **test-other-os** | Fast suite on macOS and Windows with `SOLWEIG_NO_GPU=1`. Informational (`continue-on-error`) since 2026-11; make it required once it has stayed green |
| **rust-tests** | `cargo test --lib` on macOS (pyo3 `extension-module` test binaries do not link on Linux) |
| **test-qgis-compat** | GDAL backend, NumPy 1.26, `UMEP_USE_GDAL=1` |
| **validation** | 3 real-world sites (Kronenhuset, Gustav Adolfs, GVC) |
| **benchmarks** | Memory regression gates only (`-m "not gpu_perf_gate"`, bytes-per-pixel ceiling). Timing-based perf gates are local-only via `poe test_gpu_perf_gate` (moved off CI in 6457f32). |
| **test-spec** | Scientific parity gates (vs reference UMEP implementation) |
| **test-slow** | Runs the slow-marked top-level test files |
| **audit** | Informational (`continue-on-error: true`), uploads AUDIT.md artifact |

The publish workflow has a `smoke` job between the wheel builds and `release`: it installs the Linux x86_64, Windows and macOS wheels into a clean Python 3.12 and runs `scripts/wheel_smoke.py` (one small calculation). A failure there blocks the release.

Build must be `--release` — `conftest.py` gates on `RELEASE_BUILD` flag.

---

## Code Style & Conventions

- **Line length**: 120
- **Quotes**: double
- **Indent**: 4 spaces
- **Linter**: ruff (rules: E, F, UP, B, SIM, I)
- **Type checker**: ty (NOT mypy) — ignores unresolved-import and no-matching-overload
- **Imports**: relative within package (`.models`, `..constants`), sorted by isort via ruff
- **Naming**: PascalCase classes, snake_case functions, UPPER_CASE constants, leading `_` for private
- **Docstrings**: NumPy-style with Args/Returns/Raises sections
- **Constants**: centralised in `constants.py`, never hardcoded
- **Errors**: custom hierarchy under `SolweigError` with structured attributes (field, expected, got)
- **Logging**: `solweig_logging.get_logger(__name__)` — auto-detects QGIS feedback vs stdlib logging

### Commit Messages

Conventional commits: `<type>: <description> (<version>)`
- Types: `feat`, `fix`, `docs`, `chore`, `refactor`
- Version tag appended in parentheses: `(0.1.0b82)`

---

## Known Gotchas

- `solweig_logging.py` is the logging module, NOT `logging.py` — stale references will shadow stdlib
- `loaders.py` (top-level) loads JSON params; `models/config.py` holds the `HumanParams` / `ModelConfig` dataclasses — different files, easy to confuse
- `_compat.py` lazy eval: tests that temporarily modify `sys.modules` must access attrs BEFORE restoring mocks
- The QGIS plugin's `algorithms/` directory is QGIS Processing terminology, distinct from `pysrc/solweig/physics/` (the scientific algorithm modules)
- GPU contexts (shadow / aniso / GVF) are cached via `OnceLock` and reused process-wide — what *does* get reallocated per call is the cached GPU buffer set when grid dimensions change (cache key: `(rows, cols, has_veg, has_walls)`). Command encoders and bind groups are rebuilt per dispatch, which is normal wgpu usage
- SVF is the #1 bottleneck (calls shadowing 32–248× per pixel)
- The `surface.py` decomposition (b85) moved loaders/compute/tiled-SVF/views into sibling modules but kept `SurfaceData` public — internal callers may reach into `surface_loading`, `surface_compute`, `surface_svf_tiled`, `surface_views` directly
- The 2026a ground scheme is toggled by the *presence* of a `GroundSchemeBundle` in `compute_timestep` (not a boolean): `use_ground_scheme`/`use_outgoing_longwave` must be enabled together, require land cover, reject tiling, and disable the valid-bbox crop (per-pixel state + the ~11 m march need the full raster). Baseline stays byte-identical when the bundle is absent.
- QGIS parses the plugin's `metadata.txt` with configparser **interpolation enabled**: a bare `%` anywhere in a value (changelog percentages included) breaks plugin loading. Write `%%`. Guarded by `TestShippedMetadataIsLoadable` in `tests/test_qgis_plugin_init.py`, which also pins the metadata version to `pyproject.toml` — bump both together
- The Gothenburg validation met files (`tests/validation/*/MetFile*.txt`) are labelled one hour early relative to the hour-ending CET convention that the measurement CSVs and the model's sun position both follow (found 2026-10-01, b96, by lag-fitting met K↓, measured K↓ and the model's solar altitude). Always load them through `tests/validation/met_clock.load_site_met` (adds +1 h). Do not shift the observations or `Location.utc_offset` instead: that aligns K↓ but moves the shadows an hour off. The library convention (`Weather.from_umep_met`, hour-ending, sun at label minus half a step) is UMEP's and is correct.
- `solweig.geospatial` is the canonical home for plugin-style helpers (`extract_bounds`, `intersect_bounds`, `resample_to_grid`, `looks_like_relative`, etc.); the b85→b86 top-level re-exports were removed in b87 (accessing `solweig.extract_bounds` raises `AttributeError`)
- Anisotropic shadow matrices without vegetation (b97, issue #15): both veg matrices must be all-visible (every bit set). `sky.rs` routes a patch to buildings on `!sh && vbsh` and to vegetation on `!vsh || !vbsh`, so any other fill sends wall or sky patches to the vegetation emitter (air temperature). `calculate_core_fused` substitutes the all-visible array whenever the surface has no CDSM and never reads the stored veg matrices in that case, which is why old caches and UMEP exports stay usable. Tests that need the stored veg matrices to be consumed must give the surface a CDSM.
- The reference implementation is `UMEP-dev/umep-solweig` (since 2026-11). `tests/reference/umep_2026a/` is vendored from it at a pinned commit (see the file headers); re-vendor all four files together. The golden generator and the baseline parity tests still import the older `umep` pip package. Compare the two upstreams by AST, not by text: umep-solweig is black-formatted. Known divergence kept on purpose: `Perez_v3` in umep-solweig has the clearness and low-sun air-mass parentheses misplaced relative to Perez et al. (1993) and Kasten and Young (1989); this package and the `umep` pip package use the published forms (raised upstream 2026-11-10).
- GPU contexts are process-wide and keep "static" inputs resident between dispatches, so each one must know *whose* data it holds (b98; three wrong-result bugs came from matching by address or grid shape). Shadows: full-content hash of each raster (`rust/src/gpu/content_hash.rs`), and the SVF paths clear the record because they overwrite the shared DSM buffer. Anisotropic sky: address plus sampled values, with `pipeline.invalidate_aniso_gpu_static_inputs()` called from `calculate_core_fused` when (`surface._cache.token`, `ShadowArrays._token`) changes. GVF: `GvfGeometryCache.id` and `ensure_geometry_resident`. Never key a GPU upload on an address or `id()` alone: both are reused after a free. Any new resident GPU buffer needs the same treatment and a case in `tests/spec/test_gpu_static_input_reuse.py`.
- `rust/src/api.rs` is a public Rust surface used by an outside project (CLIMESH, issue #14): `calculate_shadows_rust`, `calculate_svf_inner`, `utci_single`, `compute_timestep_rust` and their types. Treat signature or field changes to anything re-exported there as breaking, and say so in the release notes. The timestep physics lives in `pipeline::timestep_physics`; `compute_timestep` (PyO3) only borrows numpy arrays into `TimestepInputs` and calls `compute_timestep_rust`. A new per-timestep input goes in `pipeline_inputs.rs`, the destructuring prelude of `timestep_physics`, and the wrapper. Outside-crate check: a scratch crate with a path dependency (`default-features = false`) and `cargo check`.
- `tests/spec/test_end_to_end_parity.py` (b100) runs one timestep through `_calculate_single` and through the reference `Solweig_2025a_calc`, with every reference argument derived as `Solweig_run.py` derives it. It is the only test that sees what `computation.py` feeds the Rust pipeline; the function-level parity tests and the goldens pass their own inputs. It found five bugs on its first runs. Two differences remain as strict xfails: ground `Tstart` without land cover (−3.41 here, +3.41 in the runner, which is inconsistent with its own land-cover path; raised upstream, do not port) and the sitting posture (box model here, `cyl=1` in the runner). When an xfail starts passing, remove it; do not relax the tolerance.
- GVF at the raster edge (b100): `sunonsurface_2018a` never clears its shift buffers, so a ray that leaves the raster keeps the last value it saw. All three implementations must do the same: `sun::sun_on_surface`, `sun::sun_on_surface_cached` (persistent `temp*` arrays) and both GVF shaders (`cur_*`/stale variables, marked EDGE CLAMP). Accumulating only on the overlap slice lowers Lup and Kup within `second` pixels (22 m standing) of the edge.
- Every slot of `surface._cache` needs a key covering all inputs of the cached computation; use `cache.get_or_compute`. The full-grid GVF geometry slot was unkeyed until b100 and returned the first call's geometry after a posture change.
- Building mask without land cover (b100): `detect_building_mask` uses `dsm - dem >= 2` when the surface has a DEM (the reference rule) and otherwise 2 m above the 10th-percentile DSM height away from walls. The 25 m wall dilation it had before marked the ground around every building as building. The no-land-cover ground albedo default is 0.20 (`Cobble_stone_2014a`), as in the runner.
- `tiling.py` rejects tiles whose core is under 32 px (`_MIN_WORKABLE_CORE`); `surface_loading.warn_if_crs_differs` warns when an input raster's CRS differs from the DSM's (EPSG comparison, silent when undecidable).
- `tiling._TIMESTEP_GPU_BPP` (660 since b102) is the sum of every buffer the shadow, GVF and anisotropic-sky GPU contexts keep resident; its comment lists them. A new or resized GPU buffer in `rust/src/gpu/*.rs` must be added there, or auto-sized tiles overrun GPU memory.
- `io.save_raster` writes a preview PNG for every float raster. `io_preview._generate_preview_png` thins the array before colour-mapping (b102); do not add full-resolution temporaries there (`tests/test_io_preview.py`).
- A memory test must measure the peak (`tracemalloc.reset_peak` / `get_traced_memory`), not a before/after snapshot diff, and must show it fails on an injected allocation. The pre-b102 tiled memory test ran untiled and returned early every time.
- Per-timestep GeoTIFFs (`TiledGeoTiffWriter` → `io.create_empty_raster` + `io.write_raster_window`) are sparse, tiled, DEFLATE + predictor 3 since b103. Three facts measured on GDAL 3.10: (1) the compression level cannot be set for a file reopened in update mode (no open or config option; it runs at level 6), so the windowed writes pass `NUM_THREADS=ALL_CPUS` instead; (2) a TIFF block written by two computation tiles keeps its first copy as dead space, so `output_async.output_block_size` picks a block side that divides the tile side and falls back to uncompressed when it cannot; (3) unwritten blocks of a sparse file read as nodata. `timeseries._check_output_disk_space` raises `ConfigurationError` before the run when the estimate (4 B/px per float output per timestep, 1 for shadow, 60 for the summary) exceeds free space.
- Time A/B builds only on an otherwise idle machine; a background job (including a sub-agent) skews per-timestep timings by tens of percent.
- `scripts/validate_ground_scheme.py` produces the 2026a scheme table in `VALIDATION.md` (baseline against scheme at the three sites). Rerun it when the scheme path changes.

---

## Scientific Integrity

This is a **scientific library**. All code decisions must be driven by scientific principles and grounded in the published literature that SOLWEIG is based on.

- **UMEP is the precedent.** The UMEP SOLWEIG implementation is the reference; its current home is `UMEP-dev/umep-solweig`. Do not make conceptual changes, alter algorithm behaviour, tweak constants, or adjust formulas without first thoroughly reviewing whether the change aligns with the intent and scientific basis of the original UMEP library. When in doubt, preserve UMEP behaviour.
- **Golden tests are the parity gate.** The `tests/golden/` suite captures known-good outputs from validated runs. Any code change that causes golden test failures must be examined carefully — a drift in numerical output means the physics changed, not just the code. Never weaken or regenerate golden fixtures to make a refactor pass without understanding and justifying the scientific impact.
- **No speculative changes.** Do not make "improvements" to algorithms, default values, physical constants, or model behaviour based on intuition or general software engineering instincts. Every such change must be traceable to a scientific rationale: a published paper, a validated measurement, or an explicit decision by the user.
- **Spec tests encode physical laws.** The `tests/spec/` suite verifies invariants derived from physics (e.g., "flat terrain has SVF = 1", "shadow length scales with sun elevation"). These are not arbitrary assertions — they are scientific constraints. Failing a spec test means the model is physically wrong.
- **Understand before changing.** Before modifying any algorithm in `physics/`, `rust/src/`, or `components/`, read the corresponding spec in `specs/` and understand the scientific basis. Check the UMEP parity tests (`tests/spec/test_umep_parity.py`, `tests/spec/test_perez_parity.py`) to ensure the change does not break agreement with the reference implementation.

## Self-Maintenance Rules

1. **Keep this document fresh.** At the end of every session — and especially before context compaction — review what was learned and update this file with anything relevant (new conventions, bug patterns, pipeline changes, architectural decisions).
2. **Learn from friction.** If the user gets frustrated, if something is done wrong, or if there is any miscommunication, always reflect on what went wrong and record the lesson here or in memory so the same mistake is never repeated.
3. **Manage context wisely.** Delegate research, exploration, and independent tasks to sub-agents whenever possible. Keep the main conversation context available for orchestration, decision-making, and direct interaction with the user. Do not fill the main context with large file reads or exhaustive searches that a sub-agent could handle.
4. **Plan before acting.** Think through changes comprehensively before making edits. Understand the full chain of consequences — what files are affected, what downstream effects a change has, and whether the approach is correct — before touching any code. No half-baked edits or speculative changes.
5. **Understand the why.** Before making any change, understand the full context and rationale: why the code is structured this way, what problem is being solved, and what the user actually needs. Do not make changes mechanically without understanding their purpose.
6. **Keep documentation and specs in sync.** When changing code — especially API signatures, default values, constants, algorithm behaviour, or module structure — always check and update the corresponding documentation. This includes:
   - `specs/*.md` — scientific specifications must match the implementation
   - `docs/**/*.md` — user-facing guides, API docs, code examples
   - `README.md` — quick-start examples and validation table
   - `VALIDATION.md` — re-run and update after changes that affect model output
   - `CITATION.cff` — update version and date-released on releases
   - Docstrings in source code — default values, parameter descriptions, return types
   - This file (`CLAUDE.md`) — repository layout, CI table, known gotchas

   The `calculate()` API signature is `calculate(surface, weather, location, *, output_dir, ...)` and returns `TimeseriesSummary`. Many docs historically had the argument order wrong (location before weather) and assumed it returned `SolweigResult`. Always verify examples match the actual signature.

7. **Update the validation report on every commit.** When committing changes — especially any that touch physics, algorithms, shadow casting, SVF, radiation, or model defaults — re-run the validation suite (`pytest tests/validation/ -v`) and add an entry to the version history table in `VALIDATION.md`. Even if numbers are unchanged, record this to maintain a complete audit trail. Update the `README.md` validation table if the summary numbers change.

---

## Documentation Health

`VALIDATION.md` is the single source of truth for per-release numerical
changes — its version-history table records every commit that touched
physics, algorithms, or model defaults. Don't duplicate that log here.

For documentation-only / docstring / nav fixes, the commit history
(`git log -- docs/ README.md VALIDATION.md`) is authoritative. Add an
entry below only when a fix sets a non-obvious convention that future
sessions should know about.

### Release checklist (every version bump, no exceptions)

1. `pyproject.toml` version + `CITATION.cff` version and date-released.
2. `VALIDATION.md`: the `## Summary — v...` header (version + date), a new
   version-history row (no "unreleased" markers once pushed), and the
   README validation table if the numbers moved.
3. README feature/options tables if the release adds user-facing flags.
4. `docs/about/changelog.md` if it carries an entry for the release.
5. `poe docs_build` (strict) and `poe verify_project` before the release
   commit; tag `v<version>` after pushing.

Added 2026-07-07 after a b90 release went out with the VALIDATION.md
summary header still reading b89 and an "(Unreleased.)" marker in the
history row.

### Standing conventions (live)

- `calculate()` signature is `calculate(surface, weather, location, *, output_dir, ...)` and returns `TimeseriesSummary` — historically many examples had the argument order wrong; always verify generated examples match this.
- Validation table in `README.md` mirrors the latest row of `VALIDATION.md`'s version history; update both together.
- Quick-start docs assume `SurfaceData.prepare()` is the entry point, not manual `compute_svf()`.
- License is GPL-3.0 throughout (matching upstream UMEP); never write "AGPL".
- Tutorials are published as Markdown (`docs/tutorials/*.md` + `*_files/` images), exported from the `.ipynb` authoring sources by `scripts/export_tutorials.py`, which also injects the per-image alt text stored in `cell.metadata["solweig"]["image_alts"]` (a list, one entry per image output of the cell — cell metadata survives re-execution; output metadata, the pre-b89 home, is wiped every run). `poe notebooks` executes the notebooks and re-exports the Markdown in one step; the `.ipynb` files are excluded from the site build (`exclude_docs` in mkdocs.yml). Do not edit the `.md` exports by hand — edit the notebook and re-export.

### Architecture refactors (post-b82, internal-only)

- **b85** Decomposed `SurfaceData` (3000 → 1731 lines) into `surface_loading`, `surface_compute`, `surface_svf_tiled`, `surface_views`. Public API unchanged; internal callers should prefer the focused modules.
- **b85** Split `io.py` → `io_epw.py` + `io_preview.py`; `summary.py` → `grid_accumulator.py`; `models/weather.py` → `models/location.py`; `models/precomputed.py` → `models/shadow_arrays.py`. All originals re-export so downstream imports keep working.
- **b85** Added `solweig.geospatial` submodule as the canonical home for plugin-style helpers; top-level access is deprecated (see Known Gotchas).
- **b85** `Settings` dataclass (`models/settings.py`) replaces the five-way merge of `ModelConfig` / `HumanParams` / `physics` / `materials` / kwargs.
- **b85** Rust FFI bundling: `compute_timestep` now takes typed bundles instead of 43 positional args; `pipeline.rs` was split into three submodules.
