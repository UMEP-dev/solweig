# Architecture

SOLWEIG uses a layered architecture with a fused Rust compute pipeline.

## Layer Overview

```
┌─────────────────────────────────────────────┐
│  Layer 1: User API (api.py)                 │
│  calculate(), SurfaceData, Weather, etc.    │
├─────────────────────────────────────────────┤
│  Layer 2: Orchestration                     │
│  computation.py, timeseries.py, tiling.py   │
├─────────────────────────────────────────────┤
│  Layer 3: Fused Rust Pipeline               │
│  pipeline.compute_timestep() via PyO3       │
│  + Python helpers (SVF, transmissivity,     │
│    building mask, ground temperature)       │
├─────────────────────────────────────────────┤
│  Layer 4: Rust Algorithms                   │
│  shadowing, skyview, gvf, vegetation,       │
│  tmrt, utci, pet, sky (via maturin/PyO3)    │
└─────────────────────────────────────────────┘
```

## Layer 1: User API

**File**: `api.py`

The public interface:

```python
import solweig

surface = solweig.SurfaceData.prepare(dsm="dsm.tif", working_dir="cache/")
summary = solweig.calculate(
    surface=surface,
    weather=solweig.Weather.from_epw("weather.epw"),
    location=solweig.Location.from_surface(surface, utc_offset=1),
    output_dir="output/",
)
summary.report()
```

Principal types:

- `SurfaceData` — DSM, vegetation, walls, land cover, SVF (via `.prepare()`)
- `Weather` — per-timestep meteorological data
- `Location` — geographic coordinates with UTC offset
- `TimeseriesSummary` — returned by `calculate()`, containing summary statistics and GeoTIFF export

## Layer 2: Orchestration

**Files**: `computation.py`, `timeseries.py`, `tiling.py`, `summary.py`

Coordinates the pipeline and manages state:

```python
# timeseries.py — iterates over the weather list
for weather in weather_list:
    result = calculate_core_fused(surface, location, weather, state, ...)
    accumulator.update(result)       # GridAccumulator tracks min/max/mean
    state = result.state             # carry thermal state forward

# computation.py — single-timestep entry point (internal)
# Public callers use `solweig.calculate()` which iterates this and
# returns a `TimeseriesSummary`; `SolweigResult` is the per-step
# value Layer 2 hands back to the loop, not the public return type.
def calculate_core_fused(surface, location, weather, state, ...):
    svf = resolve_svf(precomputed, ...)           # Python (cached)
    psi = compute_transmissivity(doy, ...)        # Python
    buildings = detect_building_mask(dsm, ...)     # Python
    result = pipeline.compute_timestep(...)        # Fused Rust FFI call
    lup = _apply_thermal_delay(...)                # Rust (TsWaveDelay)
    return SolweigResult(tmrt, shadow, ...)        # per-timestep, internal
```

Responsibilities:

- Pre-compute Python-side inputs (SVF resolution, transmissivity, building mask)
- Dispatch to the fused Rust pipeline for per-pixel computation
- Manage thermal state across timesteps
- Accumulate summary statistics (GridAccumulator)
- Route large rasters to tiled processing

## Layer 3: Fused Rust Pipeline

**Rust entry point**: `pipeline.compute_timestep()`

A single FFI call performs the full per-pixel computation:

```text
Shadows → Ground temperature → GVF → Radiation → Tmrt
```

This eliminates intermediate numpy allocations and FFI round-trips between
Python and Rust. The pipeline accepts all inputs and returns the
complete result.

### FFI argument bundling

The 17 SVF rasters and 9 thermal-state fields are grouped into PyO3
classes (`pipeline.SvfBundle`, `pipeline.StateBundle`) so the
`compute_timestep` signature stays at ~18 positional arguments instead
of the 43 it would otherwise need. `StateBundle` also carries an
explicit **FFI version field** (`pipeline.STATE_BUNDLE_VERSION`); the
constructor raises ``ValueError`` on version mismatch so a stale
Python/Rust pairing fails loudly instead of silently mis-mapping
fields.

When adding a new per-pixel field that needs to cross the FFI:

- If it joins an existing bundle, add it to that bundle's struct in
  `rust/src/pipeline.rs` and to the corresponding constructor call in
  `pysrc/solweig/computation.py`. Increment the bundle's version
  constant. Golden tests must still pass byte-identical.
- If it doesn't fit an existing bundle, prefer creating a new bundle
  over adding a top-level positional argument — see the existing
  `SvfBundle` / `StateBundle` definitions as templates.

**Python helpers** called by the orchestration layer (Layer 2):

| Module | Function | Purpose |
| ------ | -------- | ------- |
| `components/svf_resolution.py` | `resolve_svf()` | SVF lookup and adjustment (cached) |
| `components/svf_resolution.py` | `adjust_svfbuveg_with_psi()` | Vegetation transmissivity correction |
| `components/shadows.py` | `compute_transmissivity()` | Seasonal leaf-on/off transmissivity |
| `components/gvf.py` | `detect_building_mask()` | Building footprint detection for GVF |
| `components/ground.py` | `compute_ground_temperature()` | Sinusoidal ground/wall temperature model |

## Layer 4: Rust Algorithms

**Directory**: `rust/src/`

Performance-critical algorithms implemented in Rust, exposed via maturin/PyO3:

| Module | Purpose |
| ------ | ------- |
| `pipeline` | Fused per-timestep compute (shadows → Tmrt) |
| `shadowing` | Ray-traced shadow computation (CPU + GPU) |
| `skyview` | Sky View Factor calculation |
| `gvf` | Ground View Factor with wall radiation |
| `vegetation` | Kside/Lside vegetation radiation |
| `sky` | Anisotropic (Perez) sky model |
| `tmrt` | Mean Radiant Temperature from radiation budget |
| `ground` | Ground/wall temperature and TsWaveDelay |
| `utci` | Universal Thermal Climate Index polynomial |
| `pet` | Physiological Equivalent Temperature solver |
| `morphology` | Binary dilation (building mask) |

## Data Flow

```
SurfaceData ──┐
              │
Location ─────┼──► calculate() ──► TimeseriesSummary
              │         │               │
Weather[] ────┘         │               ├── tmrt_mean / tmrt_max
                        │               ├── shadow_fraction
                        ▼               ├── sun_hours
                  timeseries loop       ├── utci_mean
                        │               └── to_geotiff() / report()
                        ▼
              calculate_core_fused()
                        │
              ┌─────────┼──────────┐
              │ Python   │  Rust    │
              │ helpers  │ pipeline │
              └─────────┴──────────┘
```

## Bundle Classes

Components communicate via typed dataclass bundles:

```python
@dataclass
class GroundBundle:
    tg: np.ndarray          # Ground temperature deviation (K)
    tg_wall: float          # Wall temperature deviation
    ci_tg: float            # Clearness index correction
    alb_grid: np.ndarray    # Albedo per pixel
    emis_grid: np.ndarray   # Emissivity per pixel

@dataclass
class LupBundle:
    lup: np.ndarray         # Upwelling longwave (centre)
    lup_e: np.ndarray       # Upwelling longwave (east)
    lup_s: np.ndarray       # ... south, west, north
    state: ThermalState     # Updated state for next timestep
```

Active bundles: `DirectionalArrays`, `SvfBundle`, `GroundBundle`,
`GvfBundle`, `LupBundle`, `WallBundle`, `VegetationBundle`.

## Caching Strategy

| Data | Cache location | Invalidation |
| ---- | -------------- | ------------ |
| Wall heights/aspects | `working_dir/walls/` | DSM change |
| SVF arrays | `working_dir/svf/` | DSM change |
| GVF geometry cache | `PrecomputedData` | Per-run |
| Land cover properties | `SurfaceData._land_cover_props_cache` | Identity change |
| Valid-pixel bounding box | `SurfaceData._valid_bbox_cache` | Identity change |

## Dual Environment Support

SOLWEIG runs in both standalone Python and QGIS:

| Component | Python | QGIS/OSGeo4W |
| --------- | ------ | ------------ |
| Raster I/O | rasterio | GDAL |
| Progress | tqdm | QgsProcessingFeedback |
| Logging | logging | QgsProcessingFeedback |

Backend detection is handled in `_compat.py`.
