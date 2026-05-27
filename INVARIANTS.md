# Load-bearing invariants

This page lists the assumptions the code relies on but does not always
enforce. They are load-bearing in the sense that **violating them
produces wrong results, not crashes** — and wrong results in a
scientific library are the worst kind of bug.

Most of these are obvious to someone who has spent a few weeks with the
codebase. They're written down here so the next contributor doesn't have
to learn them by stepping on rakes.

## Array layout

### 1. All raster arrays are `float32`, indexed `(row, col)`, contiguous, C-order

| | |
| --- | --- |
| **Where it lives** | Every numpy array passed to the Rust pipeline; every SVF / shadow / wall raster; every output GeoTIFF |
| **Why it matters** | The Rust pipeline borrows arrays zero-copy via `PyReadonlyArray2<f32>`. A `float64` array, a Fortran-order array, or a view with non-trivial strides will either fail loudly (best case) or be silently misinterpreted (worst case) |
| **How it's enforced today** | Mostly by convention. `pysrc/solweig/buffers.py:as_float32` coerces at boundaries; `SurfaceData` factory methods produce contiguous arrays. Direct construction (`SurfaceData(dsm=…)`) trusts the caller |
| **Violation symptom** | If lucky: PyO3 `TypeError`. If unlucky: numerical noise that looks like a physics bug |

### 2. The DSM grid is the canonical resolution

| | |
| --- | --- |
| **Where it lives** | All other rasters (CDSM, TDSM, DEM, land cover, walls, SVF, shadow matrices) are assumed pre-aligned to the DSM's `(rows, cols)` |
| **Why it matters** | The library does not resample. If shapes don't match, you'll get either a `GridShapeMismatch` from `validate_inputs()` or a numpy broadcast error during compute |
| **How it's enforced today** | `validate_inputs()` (`api.py`) checks every input array against `surface.dsm.shape` |
| **Violation symptom** | Loud — but only if you ran `validate_inputs()`. If you went straight to `calculate()`, the failure is wherever the mismatched array first gets indexed |

### 3. Surface arrays are not mutated after being passed to `calculate()`

| | |
| --- | --- |
| **Where it lives** | `SurfaceData.dsm`, `.cdsm`, `.tdsm`, `.dem`, `.wall_height`, `.wall_aspect`, all SVF arrays, shadow matrices |
| **Why it matters** | The per-timestep `_ComputationCache` keys cached results by `(ctypes.data, shape)`. In-place mutation produces a pointer-stable, shape-stable array with different contents — and the cache serves stale derived data |
| **How it's enforced today** | Currently not enforced. Documented here as an invariant. A future change (see [ARCHITECTURE_REVIEW.md § Cache invalidation](https://github.com/UMEP-dev/solweig/blob/main/ARCHITECTURE_REVIEW.md#3-state-management-is-five-parallel-models-not-one)) will either freeze the surface or convert the cache to content-hashed keys |
| **Violation symptom** | Silent wrong outputs across timesteps where the cached value is reused |

## Concurrency and Python/Rust boundary

### 4. The Rust pipeline owns the GIL release

| | |
| --- | --- |
| **Where it lives** | `rust/src/pipeline.rs` uses `allow_threads_unchecked()` to release the GIL for the duration of `compute_timestep` |
| **Why it matters** | Python code that runs concurrently (e.g. the progress callback, async output writers) may execute while the Rust pipeline is running. **Python callbacks invoked from inside the Rust call must not assume they hold the GIL** — the Rust side has released it and the callback context will re-acquire on entry |
| **How it's enforced today** | Convention. The `progress_callback` and tile orchestration are designed with this in mind |
| **Violation symptom** | Deadlocks or segfaults in concurrent code that assumes single-threaded execution |

## Sky model and time

### 5. Patch options 1-4 cover all sky decompositions

| | |
| --- | --- |
| **Where it lives** | `pysrc/solweig/physics/create_patches.py` and `rust/src/perez.rs::create_patches` — the patch-option-to-altitude/azimuth mapping is hard-coded in both languages |
| **Why it matters** | Anisotropic sky computations index into per-patch luminance, shadow, and steradian arrays. Adding a fifth patch option means touching Python, Rust, and several test fixtures, and re-running the UMEP parity tests |
| **How it's enforced today** | Implicit. Both implementations match by inspection; `tests/spec/test_umep_parity.py` is the regression net |
| **Violation symptom** | Out-of-bounds reads (Rust) or `KeyError` (Python) at runtime |

### 6. Weather time series are strictly monotonically increasing

| | |
| --- | --- |
| **Where it lives** | `solweig.calculate()` validates this in `api.py`; downstream code (`timeseries.py`, the warm-state carry-forward, the summary aggregator) assumes it |
| **Why it matters** | Thermal state propagates forward in time. A backwards or duplicate timestamp would mean computing a step "before" its own warm state — physically meaningless |
| **How it's enforced today** | Loudly: `calculate()` raises `WeatherDataError` on the first out-of-order entry |
| **Violation symptom** | Caught at the top of `calculate()` — this is the well-enforced one |

### 7. All timesteps in a series share the same surface

| | |
| --- | --- |
| **Where it lives** | `solweig.calculate()` takes one `surface` and one `weather` list. There is no mechanism to change the DSM (or any preprocessed auxiliary) mid-series |
| **Why it matters** | Tiled execution, SVF caching, GVF geometry caching, and the buffer pool all assume the surface is constant for the duration of the run. Changing it would invalidate every cached intermediate |
| **How it's enforced today** | API shape — there is no way to express a changing surface, so the invariant holds by construction |
| **Violation symptom** | Not reachable through the documented API |

## When to update this page

Add an invariant when:

- You discover that some code relies on an assumption that isn't written
  down anywhere
- You introduce code that *creates* a new assumption other code will rely on
- An invariant changes (e.g. the cache becomes content-hashed and the
  "no in-place mutation" rule is relaxed)

Remove an invariant when the corresponding assumption is no longer
true — e.g. if `float64` support is added, invariant 1 becomes
"`float32` or `float64`, indexed (row, col), contiguous."

Each invariant should answer four questions: where it lives, why it
matters, how it's enforced, and what happens when it's violated. If
you can't answer those, it's not load-bearing yet.
