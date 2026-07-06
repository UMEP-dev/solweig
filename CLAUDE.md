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
  components/           Ground, GVF, shadows, SVF resolution
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
  pipeline.rs           Fused per-timestep pipeline (single FFI call)
  shadowing.rs          Shadow casting (CPU + GPU)
  skyview.rs            Sky View Factor
  gvf.rs, gvf_geometry.rs  Ground View Factor + geometry caching
  sky.rs                Anisotropic sky radiation (Perez model)
  vegetation.rs         Tree effects (longwave + shortwave)
  ground.rs             Ground temperature + thermal delay
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
| **test-qgis-compat** | GDAL backend, NumPy 1.26, `UMEP_USE_GDAL=1` |
| **validation** | 3 real-world sites (Kronenhuset, Gustav Adolfs, GVC) |
| **benchmarks** | Memory regression gates only (`-m "not gpu_perf_gate"`, bytes-per-pixel ceiling). Timing-based perf gates are local-only via `poe test_gpu_perf_gate` (moved off CI in 6457f32). |
| **test-spec** | Scientific parity gates (vs reference UMEP implementation) |
| **test-slow** | Runs the slow-marked top-level test files |
| **audit** | Informational (`continue-on-error: true`), uploads AUDIT.md artifact |

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
- `solweig.geospatial` is the canonical home for plugin-style helpers (`extract_bounds`, `intersect_bounds`, `resample_to_grid`, `looks_like_relative`, etc.); the b85→b86 top-level re-exports were removed in b87 (accessing `solweig.extract_bounds` raises `AttributeError`)

---

## Scientific Integrity

This is a **scientific library**. All code decisions must be driven by scientific principles and grounded in the published literature that SOLWEIG is based on.

- **UMEP is the precedent.** The original UMEP SOLWEIG implementation is the reference. Do not make conceptual changes, alter algorithm behaviour, tweak constants, or adjust formulas without first thoroughly reviewing whether the change aligns with the intent and scientific basis of the original UMEP library. When in doubt, preserve UMEP behaviour.
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
