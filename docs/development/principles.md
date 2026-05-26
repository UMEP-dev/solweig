# Architectural principles

This document anchors every other architectural decision in the codebase.
When you face a "should this go here or there?" question, work from this
page first.

## What this library is

> SOLWEIG is a **scientific Python implementation of the SOLWEIG urban
> microclimate model**, designed to be **embeddable inside larger systems**
> — QGIS plugins, batch processing pipelines, research notebooks — with
> **UMEP-compatible numerical behaviour** as its load-bearing scientific
> contract.

That sentence does a lot of work. The rest of this page unpacks it.

## The four identities, and how each is served

The library legitimately serves four distinct identities. Each implies a
different shape, and the code is organised so each gets what it needs
without contaminating the others.

| Identity | What it needs | How the library serves it |
| --- | --- | --- |
| **Scientific library** | Small API surface, clear types, easy to test | `solweig.calculate()` as the single entry point; typed dataclasses; spec + golden tests as the regression net |
| **QGIS plugin compute engine** | Geospatial helpers, GDAL backend support, file-based workflows | The QGIS plugin is a **downstream consumer** that uses the public API. Geospatial helpers it needs live in a dedicated submodule (`solweig.geospatial`) rather than the top-level namespace |
| **Large-raster batch backend** | Memory-conscious, tileable, observable, restartable | Tiling, async output, memmap thermal state, GPU dispatch — first-class features, not afterthoughts |
| **UMEP reference port** | Conservative, numerically frozen, byte-identical, traceable | Golden tests in `tests/golden/` + UMEP parity tests in `tests/spec/test_umep_parity.py` are the gate; numerical drift requires explicit scientific justification |

## Architectural rules that follow

These rules are how the four-identity framing turns into day-to-day
decisions. When in doubt, apply them in order.

### 1. The QGIS plugin is a downstream consumer, not part of the library

The library's job is to compute Tmrt and thermal comfort indices. The QGIS
plugin's job is to be a GIS application that drives the library. They are
shipped together (this repository contains both) for convenience, but
**architecturally they are distinct**.

What this implies:

- Geospatial helpers that exist primarily to support the QGIS plugin
  (bounds intersection, raster resampling, pixel-size tagging) belong
  in a `solweig.geospatial` submodule, not in `solweig.__all__`.
- A user typing `solweig.<TAB>` in IPython should see calculation API,
  not GIS plumbing.
- The plugin imports what it needs explicitly; nothing in the library
  reaches into the plugin.

### 2. Tiling and async output are first-class features

Real urban-scale rasters are 100M+ pixels. The library is designed to
handle them. This is not "advanced usage" — it's part of the contract.

What this implies:

- `pysrc/solweig/tiling.py` and `pysrc/solweig/output_async.py` are
  load-bearing modules, not opt-in conveniences.
- New per-pixel features must consider their tiled-execution behaviour
  (does the buffer width need to grow? does the output need to be
  written asynchronously?).
- Resource-aware tile sizing, memmap thermal state, and GPU dispatch
  exist because they're necessary, not because they're impressive.

### 3. UMEP semantics are the scientific contract

The library is a **port of UMEP SOLWEIG**, not a fork that has diverged.
Numerical agreement with the upstream UMEP implementation is the
scientific contract — it's what makes published validation studies
applicable to this library's outputs.

What this implies:

- Golden tests in `tests/golden/` capture byte-identical UMEP behaviour
  for a representative set of inputs. **Golden test drift is a halt-and-investigate
  event, not a "regenerate the fixtures" event.**
- UMEP parity tests in `tests/spec/test_umep_parity.py` lock the
  algorithm-level agreement (patch decomposition, Perez sky model,
  patch steradians) against the upstream Python implementation.
- Performance optimisations, code refactors, and clarity improvements
  must preserve byte-identical output unless a scientific change is
  explicitly intended and justified.
- See [CLAUDE.md § Scientific Integrity](https://github.com/UMEP-dev/solweig/blob/main/CLAUDE.md#scientific-integrity)
  for the full rules.

### 4. The public API is small; the internal surface is rich

A user of the library should be able to do useful work with under a dozen
imported names. Internal sophistication is a feature, not a bug — but it
stays internal.

What this implies:

- `solweig.__all__` lists the user-facing names: `calculate`,
  `validate_inputs`, the input dataclasses, the result types, the
  error hierarchy, GPU controls. About 20 names total.
- Implementation modules (`computation.py`, `timeseries.py`, `tiling.py`,
  `pysrc/solweig/models/*`) are reachable by attribute access but
  not part of the documented stable API.
- Helpers shared between the library and the QGIS plugin live in a
  named submodule (e.g. `solweig.geospatial`), not at the top level.

### 5. Configuration is typed, not loose

Configuration flows through typed dataclasses, not `SimpleNamespace`
chains. JSON loaders return typed objects; merging is explicit; overrides
use sentinels (not `None`) to distinguish "use the parent's value" from
"explicitly clear this."

What this implies:

- New configuration knobs land as typed dataclass fields, not as
  `getattr(getattr(getattr(...))` chains.
- The type checker can verify configuration access paths; bypassing
  the type system is a regression to be cleaned up, not a pattern
  to extend.

### 6. Surface data is immutable after construction

Surface arrays (DSM, DEM, CDSM, walls, SVF) are treated as immutable
once they reach `calculate()`. Caches, validators, and downstream
computations rely on this. Mutating a surface array in place after
passing it to the library is a usage bug.

The corollary: the library will not defensively copy arrays at its
boundary (that would defeat the zero-copy FFI). The contract goes the
other way — callers don't mutate.

See [invariants.md](invariants.md) for the full list of load-bearing
assumptions like this one.

## When to revisit this document

This page is a contract about the **shape** of the library, not its
features. Update it when:

- The set of identities the library serves materially changes (e.g. a
  new identity is added, or one is dropped)
- An architectural rule above is broken or relaxed (and the reasoning
  has to be written down so future contributors understand why)
- A new structural pattern emerges that other code should follow

Do **not** update it for:

- New features that fit within the existing architecture
- Bug fixes
- Refactors that don't change the rules
