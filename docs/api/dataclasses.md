# Data Classes

## SurfaceData

::: solweig.SurfaceData
    options:
      show_source: false
      heading_level: 3

---

## Location

::: solweig.Location
    options:
      show_source: false
      heading_level: 3

---

## Weather

::: solweig.Weather
    options:
      show_source: false
      heading_level: 3

---

## HumanParams

::: solweig.HumanParams
    options:
      show_source: false
      heading_level: 3

---

## ModelConfig

::: solweig.ModelConfig
    options:
      show_source: false
      heading_level: 3

---

## Settings

The internal merged-configuration object that `calculate()` builds from
the `config`, kwargs, and JSON-file inputs. Most users never construct
this directly — see the [Settings guide](../guide/settings.md) for the
resolution rules — but the dataclass is documented here for callers
who want to inspect the resolved values.

::: solweig.Settings
    options:
      show_source: false
      heading_level: 3

---

## TimeseriesSummary

What `calculate()` returns — aggregated mean / max / min grids,
sun-hours, UTCI threshold exceedance, and a per-timestep
[`Timeseries`](#timeseries) of spatial-mean scalars.

::: solweig.TimeseriesSummary
    options:
      show_source: false
      heading_level: 3

---

## Timeseries

Per-timestep scalar series (Tmrt mean, UTCI mean, etc.) embedded
inside [`TimeseriesSummary`](#timeseriessummary).

::: solweig.Timeseries
    options:
      show_source: false
      heading_level: 3

---

## SolweigResult

Per-timestep internal result produced by the fused Rust pipeline
(Tmrt, shadow, radiation components). `calculate()` aggregates these
into a [`TimeseriesSummary`](#timeseriessummary); user code rarely
constructs or returns one directly. Documented here for advanced
single-step / chained workflows that consume the per-step output.

::: solweig.SolweigResult
    options:
      show_source: false
      heading_level: 3

---

## PrecomputedData

::: solweig.PrecomputedData
    options:
      show_source: false
      heading_level: 3

---

## ThermalState

::: solweig.ThermalState
    options:
      show_source: false
      heading_level: 3

---

## TileSpec

::: solweig.TileSpec
    options:
      show_source: false
      heading_level: 3
