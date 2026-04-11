# CLAUDE.md — SOLWEIG Project Guide

## What Is This Project?

SOLWEIG (SOlar and LongWave Environmental Irradiance Geometry-model) is a high-performance urban microclimate model. It computes mean radiant temperature (Tmrt) and thermal comfort indices (UTCI, PET) in complex urban environments.

- **Hybrid Rust/Python** — heavy compute in Rust (PyO3), orchestration and I/O in Python
- **Optional GPU acceleration** — wgpu compute shaders for shadow casting, SVF, GVF, and anisotropic sky
- **Dual geospatial backend** — rasterio (standalone) or GDAL (QGIS plugin), detected lazily at runtime
- **License**: GPL-3.0
- **Version**: 0.1.0 beta series (single source of truth: `pyproject.toml`)

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

tests/                  ~730 tests across ~50 files
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
| **test** | Matrix: Python 3.10, 3.11, 3.12 — excludes slow tests (NOTE: 3.10 is below requires-python >=3.11; 3.13 is missing) |
| **test-qgis-compat** | GDAL backend, NumPy 1.26, `UMEP_USE_GDAL=1` |
| **validation** | 3 real-world sites (Kronenhuset, Gustav Adolfs, GVC) |
| **test-spec** | Scientific parity gates (vs reference UMEP implementation) |

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
- `physics/` was renamed from `algorithms/` (Feb 2026) — watch for stale imports
- `loaders.py` was renamed from `config.py` — but `models/config.py` (HumanParams, ModelConfig) is UNCHANGED
- `test_thermal_comfort.py` was split into `test_utci.py` + `test_pet.py`
- `_compat.py` lazy eval: tests that temporarily modify `sys.modules` must access attrs BEFORE restoring mocks
- The QGIS plugin's `algorithms/` directory is CORRECT — it uses QGIS Processing terminology, distinct from the renamed `physics/` directory
- GPU context is recreated per call (known optimisation opportunity)
- SVF is the #1 bottleneck (calls shadowing 32-248x per pixel)

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

## Documentation Health (audited 2026-04-11)

### Fixed (2026-04-11 — v0.1.0b82)

- `README.md`: validation table updated with b82 numbers (Gustav Adolfs RMSE 9.3–18.9 → 5.7–7.5, GVC 11.5–15.6 → 1.5–6.1 after the scale convention fix). Demos list now includes Bilbao and Madrid demos.
- `VALIDATION.md`: full post-b82 numbers for all three sites + new version history row. Removed the duplicated "b82 change summary" paragraph; the version history table is the single source of truth for changelog.
- `docs/getting-started/quick-start.md`: added surface preparation steps for DEM stair-step smoothing and the warm-run fast-path (100× speedup).
- `docs/guide/qgis-plugin.md`: corrected output directory structure (now shows `cleaned/`, `walls/px<size>/`, `svf/px<size>/`). Added warm-run cache description.
- `qgis_plugin/.../surface_preprocessing.py`: final summary log line no longer claims "Walls computed: yes / SVF computed: yes" unconditionally (misleading on warm cache hits); now reports "Walls: present / SVF: present".
- `surface.py` fast-path: emits `setProgressText("Loading cached surface...")` via feedback so QGIS users see context during the otherwise-silent ~50 ms warm load.
- `specs/shadows.md`: ray-march `dz` formula corrected to `ds × step × tan(α) × scale` and annotated with the solweig/UMEP convention difference (solweig: `scale = pixel_size_m`; UMEP: `scale = 1/pixel_size`). This matches the `shadowing.rs` / `shadow_gpu.rs` b82 fix (`tan().* scale`, was `/ scale`) and explains why shadow length was physically wrong at non-1m pixel sizes before b82.
- `qgis_plugin/.../metadata.txt`: bumped to `0.1.0-beta82` with a b82 changelog entry.
- `pyproject.toml`: bumped version to `0.1.0b82`.
- Progress-reporting audit: surface prep walls=10→30%, SVF=30→75%, plugin finalises at 80/95/100; calculation phases 5/10/15/20/25 then callback 25→80 then 80/100. Tile-granular callback in timeseries.py correctly maps `(tile_idx * n_steps + t_idx + 1, n_tiles * n_steps)` to tile-agnostic percentage. No stale setProgress patterns observed.

### Fixed (2026-03-14)

- All user docs: `calculate()` argument order corrected (surface, weather, location)
- All user docs: return type corrected to `TimeseriesSummary` (not `SolweigResult`)
- `README.md`: validation table updated to match VALIDATION.md v0.1.0b66 numbers
- `README.md`: minimal example uses `prepare()` instead of manual `compute_svf()`
- `docs/PARAMS_SIMPLE.md`: height default corrected (180 -> 175 cm), removed nonexistent `solweig.svf.generate_svf()`
- `docs/development/contributing.md`: Python version corrected (3.10+ -> 3.11+)
- All license references corrected to GPL-3.0 (was incorrectly labelled AGPL-3.0; LICENSE file was always GPL-3.0, matching upstream UMEP)
- `surface.py` docstring: `min_object_height` default corrected (1.5 -> 1.0)
- `specs/svf.md`: Option 3/4 patch counts corrected (305/609, verified against upstream UMEP)
- `specs/svf.md`: LAST_ANNULUS_CORRECTION documented as S/W-only with azimuth-bucketing rationale
- `specs/pet.md`: "7-mode" corrected to "6-mode" (verified: `while j < 7` matches UMEP)
- `skyview.rs`: Options 3/4 last annulus band fixed from 2 to 1 (matches UMEP; only affects non-default options)
- `precomputed.py`, `create_patches.py`, `test_perez_parity.py`: patch count references updated
- CI matrix: removed Python 3.10 (below requires-python), added 3.13
- `Cargo.toml`: `abi3-py39` replaced with `abi3-py311` (matches pyproject.toml)
- `pyproject.toml`: removed vestigial `[tool.setuptools]` section
- `geopandas` moved from core deps to `[project.optional-dependencies] geo`
- Validation suite re-run: all 28 tests pass (b68)
