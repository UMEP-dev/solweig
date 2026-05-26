# SOLWEIG Architecture Review

_A senior-engineer pass over the codebase, focused on shape rather than symptoms.
Not a punch list. Not a scorecard. The goal is to surface what kind of system this
**is** (vs what it's currently shaped like), and identify the few changes that
would matter most. Written to be discussed, not adopted wholesale._

_Author: Claude · Reviewed against commit `<HEAD>` on the `main` branch._

---

## 0. The framing question: what is this library actually for?

The codebase doesn't answer this clearly, and that's the upstream cause of most
of the friction I see further down. Reading [README.md](README.md), [CLAUDE.md](CLAUDE.md),
[api.py](pysrc/solweig/api.py), and [pysrc/solweig/__init__.py](pysrc/solweig/__init__.py)
side-by-side, four distinct identities are visible:

1. **A scientific Python library** that computes Tmrt and thermal-comfort
   indices, validated against field data, ported from UMEP. Lives in
   `solweig.calculate()`.
2. **A QGIS plugin compute engine** — the `qgis_plugin/` directory plus all
   the geospatial helpers (`extract_bounds`, `intersect_bounds`,
   `resample_to_grid`, `pixel_size_tag`, `compute_max_tile_pixels`,
   `looks_like_relative`) that got promoted to `__all__` in b84 specifically
   to satisfy the plugin's preprocessing needs.
3. **A large-raster batch backend** — tiling, async output, memmap thermal
   state, GPU dispatch, resource-aware tile sizing. None of this is needed
   for a research script; all of it is needed for "process a 500-megapixel
   Madrid raster overnight."
4. **A reference port of UMEP SOLWEIG** — golden tests + UMEP parity tests
   freeze numerical agreement with the upstream implementation, and the
   "scientific integrity" rules in CLAUDE.md effectively say "the science is
   not yours to change."

Each of these identities implies a different shape:

| Identity | Shape it wants |
|---|---|
| Scientific library | Small surface area, swappable stages, easy mocking, fast iteration |
| QGIS plugin engine | Rich preprocessing surface, GDAL backend, file-based workflows |
| Large-raster backend | Memory-conscious, tileable, observable, restartable |
| UMEP reference port | Conservative, numerically frozen, byte-identical, traceable |

The current code is trying to be all four. That's fine — many real
libraries serve multiple use cases — but it should be a **conscious
choice**, with the architecture explicitly organised so each identity
gets what it needs without contaminating the others. Today the
contamination is everywhere: the public API mixes calculation entry
points with QGIS-specific raster helpers; `SurfaceData` mixes
research-grade arrays-in/out with file-based workflow orchestration;
the Rust pipeline is shaped for batch-backend throughput at the cost
of research-style stage swapping.

**The single most valuable architectural move would be to write down,
in one paragraph, what this library is for — and then make every
subsequent decision consistent with it.** Without that anchor every
"should we refactor X" conversation is unmoored.

My read, for what it's worth: **the library is a scientific Python
implementation of the SOLWEIG model, designed to be embeddable inside
larger systems (QGIS plugin, batch pipelines, research notebooks),
with the numerical behaviour locked to UMEP.** Under that framing:

- The QGIS plugin is a **downstream consumer** of the public API, not part of
  the library — and the library should not contain plugin-specific helpers
  in its top-level namespace.
- Tiling and async output belong in the library because they're part of
  "make the science usable at real urban scale" — but they should be
  cleanly separable from the per-timestep math.
- Backwards-compatibility with UMEP semantics is **the scientific
  contract**, not a style preference, and golden tests are the gate.

If that framing is wrong, large parts of what follows reorient. So this
is question one to settle.

---

## 1. SurfaceData is a god-object — and the natural decomposition is obvious

[pysrc/solweig/models/surface.py](pysrc/solweig/models/surface.py) is 3,016 lines —
larger than every other module by 2.5×. The class itself has at least nine
distinct responsibilities (data schema, height-convention conversion, file I/O
orchestration, preprocessing pipeline, SVF/wall computation, masking and
cropping, optical-property derivation, buffer pool management, transient cache
management).

The fact that all of these live on one class is not just an aesthetic
problem. It produces specific friction:

- **The class can't be safely used in "research" mode** (give it arrays, call
  `calculate`) without inadvertently triggering preprocessing side effects.
  Properties like `_preprocessed` and `_nan_filled` are advisory flags, not
  type-encoded states. A user who forgets to call `preprocess()` gets a
  warning in `validate_inputs()` and silently-wrong heights if they don't run
  the validator.
- **The class can't be safely used in "file" mode** without understanding the
  warm-cache fingerprint model, which is documented in CLAUDE.md but invisible
  in the API.
- **Cache invalidation is impossible to reason about** because the cache
  lives on the same object that owns the (mutable) data, and the cache keys
  are `(arr.ctypes.data, arr.shape)` — pointer addresses, not content.
  In-place mutation of `surface.dsm[:] = …` silently serves stale cached
  results.

The natural decomposition (per the deep read I did) is three or four
types:

```text
SurfaceGeometry          # frozen: dsm, dem, cdsm, tdsm, pixel_size, crs, geotransform
PreprocessedAuxiliary    # frozen: wall_height, wall_aspect, svf, shadow_matrices
OpticalProperties        # frozen: land_cover, albedo, emissivity, materials lookup
SurfacePreparation       # orchestrator: file I/O, fingerprint cache, factory methods
                         # produces (SurfaceGeometry, PreprocessedAuxiliary, OpticalProperties)
PerTimestepComputeCache  # transient, owned by computation.calculate_core_fused,
                         # NOT a field of any surface object
```

Three things make this concrete and tractable:

1. **The current code already has most of these pieces** — `SvfArrays`,
   `ShadowArrays`, the helper methods on SurfaceData all map cleanly to
   the proposed split. This is a re-organisation, not a rewrite.
2. **The `PerTimestepComputeCache` was already extracted** as
   `_ComputationCache` (just last week). Moving it off `SurfaceData` onto
   the orchestrator is a small change with disproportionate clarity gains.
3. **`frozen=True` on the data classes** is the enforcement mechanism
   that turns "don't mutate arrays in place" from a convention into a
   compile-time guarantee. The `ctypes.data` cache key problem then
   becomes a non-issue.

The most valuable thing about this split is what it makes **impossible**:
you can no longer accidentally mutate a surface and reuse stale caches;
you can no longer construct a half-prepared surface and pass it to
`calculate()`; you can no longer get confused about which fields are
inputs vs derived. The types carry the contract.

---

## 2. The Rust/Python boundary was optimised for one concern and accumulated debt elsewhere

The current `pipeline.compute_timestep` has 43 parameters. That number is
the smoking gun. It exists because the design optimised for **FFI
overhead minimisation** — fuse the entire per-timestep pipeline into a
single call so numpy never has to allocate intermediate arrays. That
goal is met (the b82 work confirms the perf is real), but the shape
has costs:

- **Brittleness.** Any new per-pixel input (e.g. plant view factor,
  transpiration grid, indoor-outdoor coupling) requires changes in
  five places: the Rust function signature, the Rust struct, the
  Python construction code in [computation.py:247-507](pysrc/solweig/computation.py#L247-L507),
  the FFI call site, and the unpacking after. None of these are
  type-checked across the boundary. ThermalState alone is six arrays
  + three scalars and is **not versioned** — adding `tgmap2` (a
  second-depth ground layer) would silently pass None for old Python
  code and break.
- **Inflexibility.** A research user who wants to substitute the
  shadow stage with a custom implementation cannot — it's all-or-nothing
  inside the fused pipeline. The single-call architecture made
  "swap a stage" architecturally impossible without leaving the
  fast path.
- **Cognitive cost.** A new contributor reading
  [computation.py:380-525](pysrc/solweig/computation.py#L380-L525) has
  to parse 150 lines of parameter shuffling to understand what's
  flowing into the Rust call. There's no narrative — it's a
  mechanical mapping from named locals into a positional argument list.

There are two ways to fix this without sacrificing performance.

**Option A: Bundle the args into typed structs.** Same single FFI call, but
the surface arrays group into `SurfaceBundle`, the SVF arrays into
`SvfBundle`, the thermal state into `StateBundle`, etc. Signature goes
from 43 args to ~8 typed structs. Adding a new field touches one struct
in two languages, not five places. **The refactor is mechanical, the
perf characteristics are identical, and the future-feature cost goes
down sharply.** This is the lowest-risk, highest-leverage change to
the FFI boundary.

**Option B: Compute-graph with intermediate Rust types.** Split into 5-6
sub-FFI calls (`shadows()` → `ground_temp()` → `gvf()` → `radiation()`
→ `tmrt()`), with intermediate Rust-allocated types passed between them.
This restores stage-swapping ability and makes the pipeline introspectable,
but adds per-call FFI overhead (probably negligible for >10ms-per-pixel
work, fatal for sub-ms). Worth doing only if "let users substitute a
stage" becomes a real requirement.

I'd recommend **Option A unconditionally** as the first step. Option B
is contingent on framing question 0 (is research-tool stage-swapping
actually a goal?).

A related issue: there's no FFI versioning. The Python and Rust sides
agree on field layouts by convention. A new feature that adds a Rust
struct field requires a coordinated Python release; mixing versions
produces silent wrong outputs (or crashes if you're lucky). For a
library shipped via pip and embedded in QGIS plugins on user machines,
this is a real exposure. A bundle-based API is also easier to version:
add a `version: u32` field to each bundle and check it on entry.

---

## 3. State management is five parallel models, not one

There are five distinct kinds of state/cache in the codebase, each with
its own lifecycle, ownership, invalidation rule, and failure mode. They
don't communicate with each other. This is the most pervasive
architectural issue.

| Layer | Owner | Scope | Key | Failure mode |
|---|---|---|---|---|
| `ThermalState` | User | Across timesteps | Implicit (user chains it) | Forgets to chain → temperatures reset to zero, silently wrong |
| `_ComputationCache` | `SurfaceData` | Per-surface, transient | `(ctypes.data, shape)` | In-place mutation → stale cache served silently |
| SVF/wall disk cache | `SurfaceData.prepare()` | Per `working_dir`, persistent | File mtime/size + kwargs | Doesn't cover runtime constants → stale SVF on warm prepare after constant change |
| Rust patch LUT / ASVF cache | Rust statics | Process lifetime | `(nrows, ncols, hash)` | Hash collision on same-shaped buffer → wrong ASVF |
| `SurfaceData` flags | `SurfaceData` | Per-surface | `_preprocessed`, `_nan_filled` | Advisory only — second call to `preprocess()` corrupts data |

Each individual cache is defensible. **Their lack of coordination is not.**
A new contributor has to learn five separate models, with five separate
key types and five separate failure modes. The cache keys based on
`ctypes.data` are the worst offender — they encode the unspoken
assumption "thou shalt not mutate surface arrays after passing them
in," and that assumption isn't anywhere in the API or the docs. If
someone writes `surface.dsm[invalid] = np.nan` after a first
`calculate()` call, the second call silently reuses stale derived data.

The two changes that would fix this without rewriting everything:

1. **Make the surface frozen at construction time** (per Section 1).
   This eliminates the in-place mutation footgun and lets the
   `_ComputationCache` keys be safe by construction.
2. **Move the `_ComputationCache` off `SurfaceData` onto the
   orchestrator.** Today caches live on the data they cache, which
   means cache state survives across unrelated calls. If the cache
   lived on `calculate_core_fused`'s scope it would have a clean
   lifecycle: per call, no cross-call leakage.

The disk and Rust-static caches are mostly fine. The disk fingerprint
should be extended to cover physics/materials constants (currently it
covers user kwargs but not the loaded JSON), and the Rust-static
caches are content-keyed so the only real risk there is
buffer-aliasing-with-different-data, which is solved by freezing the
surface.

---

## 4. The configuration model is fragmented

A user calling `calculate()` can configure the run via five different mechanisms:

- `ModelConfig` — typed dataclass with most settings
- `HumanParams` — separate typed dataclass
- `physics: SimpleNamespace` — loaded from a JSON file via `load_physics()`
- `materials: SimpleNamespace` — loaded from a JSON file via `load_materials()`
- Per-call kwargs to `calculate()` that override any of the above

The override-merging logic in
[api.py:266-318](pysrc/solweig/api.py#L266-L318) is 50+ lines, has no
test that I can find covering all combinations, and uses a "None means
inherit" convention that makes "I want this explicitly false" indistinguishable
from "I forgot to set this." The `materials` and `physics` SimpleNamespaces
defeat type-checking by design — that's why
[`WallMaterialDefaults`](pysrc/solweig/models/materials.py) had to be
introduced last week to wrap the triple-nested `getattr` access pattern.

A coherent model would be a single typed `Settings` object built once,
with explicit override semantics, and `calculate()` taking only
`(surface, weather, location, *, settings, output_dir)`. The merging
logic moves into `Settings.merge(*overrides)` where it can be tested
exhaustively. The JSON loaders return typed `MaterialProperties` and
`PhysicsParameters` dataclasses, not SimpleNamespaces.

This is a moderate-effort change with moderate payoff. Not the most
urgent, but the cost compounds: every new setting today has to be
threaded through the merge logic, and every existing one has the
implicit-None ambiguity.

---

## 5. The implicit/explicit boundary is leaky

Three places where the API has implicit behaviour that surprises users
or hides bugs:

- **`validate_inputs()` both returns warnings AND raises exceptions
  ([api.py:73-239](pysrc/solweig/api.py#L73-L239)).** This is two different
  signaling protocols mixed into one function. A caller wanting "tell me
  everything wrong" has to wrap in try/except AND iterate the warnings.
  Pick one: either return a `ValidationResult` with errors and warnings
  as fields, or raise immediately on the first issue and accept that
  multi-error reporting is gone.
- **The anisotropic precondition check only fires for "explicit" requests
  ([api.py:264, 327-336](pysrc/solweig/api.py#L264-L336))** — if
  `use_anisotropic_sky=True` came from a `ModelConfig` field, the
  precondition is silent. The justification (we can't distinguish a
  config default from a user choice) is honest but the consequence is a
  failure mode that depends on call style, not on the actual configuration.
- **The `_preprocessed` flag is advisory.** Users who skip
  `preprocess()` on relative-height inputs get a warning during
  validation (if they ran it) and silently wrong outputs otherwise.
  The lifecycle state should be encoded in the type, not on a flag —
  e.g. `RawSurface` vs `PreparedSurface`, where `calculate()` only
  accepts the latter.

These are individually small but together produce the feeling that
"the API mostly works, but you have to know the unwritten rules."

---

## 6. Public API surface is conflated with QGIS-plugin needs

[`solweig/__init__.py`](pysrc/solweig/__init__.py) exports 38 names in
`__all__`. The intent is the user-facing API, but in practice it
contains:

- **Research/user names**: `calculate`, `SurfaceData`, `Weather`, etc. (correct)
- **Internal-but-useful**: `Timeseries`, `TimeseriesSummary`, `ThermalState` (debatable)
- **QGIS-plugin support helpers**: `extract_bounds`, `intersect_bounds`,
  `resample_to_grid`, `pixel_size_tag`, `compute_max_tile_pixels`,
  `looks_like_relative`, `namespace_to_dict`, `wallalgorithms` (wrong)
- **Operational controls**: `is_gpu_available`, `disable_gpu`,
  `get_gpu_limits`, `get_compute_backend` (mostly fine)

The QGIS plugin helpers were promoted to public API in b84 specifically
to stop the plugin from reaching into internals. That's the right
direction — but it solved the immediate problem by **enlarging the
public contract**, when the cleaner answer is to have a separate
`solweig.qgis_helpers` submodule that the plugin imports and that's
**not part of the user-facing namespace**. Today, a user typing
`solweig.<TAB>` in IPython sees `extract_bounds` and `looks_like_relative`
next to `calculate` — that's a misleading signal about what the library is.

---

## 7. There's no operational story

Today the library has:

- A `progress_callback` parameter on `calculate()`
- A custom QGIS-aware logging adapter ([solweig_logging.py](pysrc/solweig/solweig_logging.py))
- A progress module ([progress.py](pysrc/solweig/progress.py))

That's the entirety of the observability story. There are no metrics
(timesteps/sec, GPU util, memory high-water-mark), no debug
introspection, no tracing for "why did this run produce these
outputs," no structured event log, no health check or self-test.

For a library that's **mostly used by humans interactively** (QGIS,
notebooks, research scripts), the current state is probably fine. For
a library that's also being used as a **batch backend on large
rasters** (Madrid demo is 500M pixels), the missing operational
surface will start to hurt. Specifically: when a 4-hour batch run
produces a wrong output, today the only debug surface is "re-run with
more logging."

I'd hold this as a "do it when the second batch-run incident
happens." It's a real gap but it's not urgent unless the user base
expands into ops-heavy contexts.

---

## 8. Scientific reproducibility is asserted but under-specified

Golden tests pin byte-identical output. Validation tests pin RMSE/R²
against real field measurements. UMEP parity tests pin agreement with
the reference Python implementation. All of this is good — better
than most scientific software.

What's missing is the **policy layer**:

- On what platforms are golden outputs valid? (f32 SIMD ordering varies between
  platforms — Linux/macOS arm64/macOS x86_64 may legitimately differ.)
- How are golden fixtures regenerated when physics legitimately changes?
  Today the answer is "edit the JSON by hand and explain in the PR."
  Fine for now; brittle as the team grows.
- What's the version-skew story? If a user has solweig 0.1.0b80 installed
  and reads outputs from a 0.1.0b84 run, what's reproducible?
- What metadata travels with output? `metadata.py` tracks some (version,
  config) but not GPU/CPU split, not Rust commit, not seed (if any RNG
  is in the system, which I didn't audit).

This is "complete the existing approach," not "redesign." A short
`REPRODUCIBILITY.md` documenting these policies, plus extending
`create_run_metadata()` to include the missing fields, would close it.

---

## 9. Error UX is partial

The custom error hierarchy in
[errors.py](pysrc/solweig/errors.py) is well-designed
(structured fields, suggestions). But it's only used at the top of the
API. Most internal failures still propagate as raw `RuntimeError`,
`ValueError`, or `numpy` exceptions — and Rust panics propagate as
Rust traceback strings (or, with `panic = "abort"`, kill the
interpreter — see the Phase 1 hardening that just shipped).

If "error UX" is a goal, the work would be: audit every `raise` in
pysrc/ and every `unwrap`/`expect` in rust/src/, and decide whether
each should be (a) a typed SolweigError, (b) a developer-facing
panic that's a real bug, or (c) genuinely fine as raw. I'd estimate
this is a multi-day audit but with diminishing returns after the first
20 sites. Probably not urgent — the Phase 1 work covered the worst
of the panic surface.

---

## 10. Load-bearing implicit assumptions

These are the assumptions the code makes that aren't in any test, spec,
or docstring — the kind that bite when the world changes underneath
them. I noticed at least these while reading:

- **All raster arrays are `float32`, indexed `(row, col)`, contiguous,
  C-order.** Violated by accident → silent wrong output (e.g.
  Fortran-order slicing in a user transform).
- **The DSM grid is the canonical resolution.** All other layers
  (CDSM, walls, SVF) are assumed already on it. Violation → shape
  mismatch error if you're lucky, silent broadcast errors if you're not.
- **Surface arrays are not mutated after being passed to `calculate()`.**
  Encoded nowhere; violated → stale cache (Section 3).
- **The Rust pipeline owns the GIL release.** Python code must not
  expect to hold the GIL inside `compute_timestep` or any other long
  Rust call.
- **Patch options 1-4 cover all sky decompositions ever needed.**
  Hard-coded in `create_patches` (both Python and Rust). A new patch
  option means touching both languages.
- **Time series are strictly monotonically increasing.** Enforced in
  `calculate()`, but the rest of the pipeline assumes it without
  re-checking.
- **All timesteps share the same surface.** No mechanism to change
  the DSM mid-series. Probably correct as a design choice but worth
  saying explicitly.

A short `INVARIANTS.md` (or a section in `CLAUDE.md`) listing these
explicitly would catch a future bug that gets introduced by someone
who didn't know the rule. Cheap to write, high value.

---

# The 3-5 changes that would matter most

Stack-ranked by (impact × ease):

### A. Write the framing paragraph and audit against it

**Effort:** 1-2 hours of writing + a couple of focused decisions.
**Impact:** Reframes every subsequent architectural conversation. Decides
whether QGIS-plugin helpers belong in the main namespace, whether tiling
is a first-class feature or an implementation detail, whether stage-swapping
is a goal. Most of the other changes here become obvious (or obviously not
worth doing) once this is settled.

### B. Decompose `SurfaceData` and encode lifecycle in the type system

**Effort:** Real work — 2-4 days. Touches the largest file in the codebase
and the most-called class. But every step is a refactor, not a rewrite.
**Impact:** Eliminates the god-object. Makes "preprocessing must run first"
a type guarantee, not a flag. Cleanly separates the data the user provides
from the auxiliary the library computes. Sets up frozen=true which fixes
the cache-key footgun in one stroke. The single highest-leverage change
to contributor velocity.

### C. Bundle the Rust FFI arguments into typed structs

**Effort:** Moderate — 1-2 days. Mechanical refactor with no semantic change.
**Impact:** Makes the per-timestep FFI call comprehensible. Makes future
features cheap to add (one struct field, not five places). Adds the seam
for FFI versioning. Doesn't touch numerics — golden tests will pass
byte-identical.

### D. Move `_ComputationCache` off `SurfaceData` onto the orchestrator

**Effort:** Small — half a day. The class already exists; it's just relocated.
**Impact:** Clarifies that the cache is per-call transient, not per-surface
persistent. Eliminates the question "what happens to the cache if I reuse
this surface across two unrelated runs."

### E. Unify configuration into a single typed `Settings` with explicit overrides

**Effort:** Moderate — 1-2 days, mostly testing the merge logic.
**Impact:** Eliminates the 50-line override block, the implicit-None
ambiguity, and the `SimpleNamespace` type-checker holes. Makes the
configuration story documentable in a sentence.

(I'd stop at five. A, B, D would be the first batch; C and E follow if A
confirms they're worth doing.)

---

# What I'm uncertain about

These are the places where my recommendations could be wrong, and where
I'd want to discuss before acting:

1. **The framing answer in §0.** I picked "scientific library that's
   embeddable" because it matches the code's gravity. If the answer is
   "primarily a QGIS plugin engine, Python API is secondary," large parts
   of this critique reorient (the QGIS helpers stay in the public API;
   the surface-decomposition becomes less urgent).

2. **The development trajectory.** If this codebase is in maintenance mode
   — bug fixes, no new features, small user base — the current shape is
   fine and refactoring is net-negative. If it's expecting active feature
   work (new patch options, new physics models, new data sources), the
   brittleness will compound and the refactors pay off fast. **I don't
   know which it is.**

3. **The team size.** The current shape is tolerable for one person who
   holds the whole mental model. It's increasingly painful for two, three,
   five people. I assumed "small team or solo" based on commit patterns
   but didn't verify.

4. **The performance envelope.** I'm guessing FFI overhead per timestep
   is small enough that Option B (split FFI) would be acceptable on most
   workloads. If the per-timestep work is sub-millisecond (e.g. very
   small grids in interactive mode), the overhead matters more. Should be
   measured before committing.

5. **The cache footgun frequency.** The `ctypes.data` cache-key bug I
   flagged in Section 3 is real but I don't know how often it actually
   bites. If users never mutate arrays in place (which would be the
   norm in file-based workflows), it's a latent risk worth fixing
   anyway. If it's never tripped in two years of use, the cost of
   the refactor might exceed the cost of leaving it.

6. **The QGIS plugin's actual coupling.** I assumed the plugin can
   move to a private `qgis_helpers` submodule without breakage. If
   downstream UMEP forks already depend on the b84-promoted helpers
   being in the top namespace, the cleanup costs them, not us.

7. **Whether Option B (split FFI / compute-graph) is ever wanted.** I
   defaulted to "no" because the current single-call shape is
   performance-correct and the use-cases for stage-swapping are
   theoretical. If a real user wants to substitute "my better
   anisotropic sky model" for the built-in, Option A doesn't help
   them. This depends entirely on the framing answer.

---

# What this document is **not**

This is not a punch list of "X line is wrong, change to Y." That's what
the previous audits did. This is one engineer's read of the system's
shape, the load-bearing assumptions it's making, and the three or four
moves that would make the next 12 months of development substantially
easier.

Some of what's here may be wrong. Most of it should be argued with. The
goal is to have the argument — not to have me hand down decisions from
on high.
