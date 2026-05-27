# API Reference

Complete reference for the SOLWEIG public API.

## Core Functions

| Function | Description |
| -------- | ----------- |
| [`calculate()`](functions.md#calculate) | Single or multi-timestep Tmrt calculation (tiling is applied for large rasters) |
| [`validate_inputs()`](functions.md#validate_inputs) | Pre-flight input validation |

## Data Classes

| Class | Description |
| ----- | ----------- |
| [`SurfaceData`](dataclasses.md#surfacedata) | Terrain data (DSM, CDSM, walls, SVF) |
| [`Location`](dataclasses.md#location) | Geographic coordinates |
| [`Weather`](dataclasses.md#weather) | Meteorological conditions |
| [`HumanParams`](dataclasses.md#humanparams) | Human body parameters |
| [`SolweigResult`](dataclasses.md#solweigresult) | Calculation output |
| [`TimeseriesSummary`](dataclasses.md#timeseriessummary) | Aggregated timeseries output |
| [`Timeseries`](dataclasses.md#timeseries) | Per-timestep scalar timeseries |
| [`ModelConfig`](dataclasses.md#modelconfig) | Model configuration |

## I/O Functions

| Function | Description |
| -------- | ----------- |
| [`load_raster()`](io.md#load_raster) | Load a GeoTIFF with optional bbox cropping |
| [`save_raster()`](io.md#save_raster) | Save array as Cloud-Optimized GeoTIFF |
| [`rasterise_gdf()`](io.md#rasterise_gdf) | Rasterise vector data to a height grid |
| [`download_epw()`](io.md#download_epw) | Download EPW weather from PVGIS |
| [`read_epw()`](io.md#read_epw) | Parse an EPW file to weather records |

## GPU Utilities

| Function | Description |
| -------- | ----------- |
| [`is_gpu_available()`](functions.md#is_gpu_available) | Check whether GPU acceleration is available |
| [`get_compute_backend()`](functions.md#get_compute_backend) | Returns `"gpu"` or `"cpu"` |
| [`disable_gpu()`](functions.md#disable_gpu) | Disable GPU, fall back to CPU |
| [`get_gpu_limits()`](functions.md#get_gpu_limits) | Inspect GPU device limits |

## Import Pattern

```python
import solweig

# All public API is available at the top level
surface = solweig.SurfaceData.prepare(dsm=my_dsm, pixel_size=1.0)
summary = solweig.calculate(surface, weather=[weather], location=location, output_dir="output/")
```

The top-level namespace is the documented import surface — every class and
function on this reference page is reachable as `solweig.<name>`. The
sub-modules listed in the [repository
layout](https://github.com/UMEP-dev/solweig#repository-layout)
(`solweig.models.surface`, `solweig.io_epw`, `solweig.grid_accumulator`,
…) exist for code-organisation reasons and may be reorganised between
releases; depend on `solweig.<name>` rather than the sub-module path
to stay forward-compatible.

For QGIS-plugin / batch-pipeline authors who need geospatial helpers
(`extract_bounds`, `intersect_bounds`, `resample_to_grid`,
`looks_like_relative`, …), the entry point is
[`solweig.geospatial`](geospatial.md). The b85→b86 top-level
re-exports were removed in b87 — accessing `solweig.extract_bounds`
now raises `AttributeError`.

## Type Annotations

SOLWEIG is fully typed. Type checking can be enabled in any IDE:

```python
from solweig import SurfaceData, Location, Weather, TimeseriesSummary

def process_area(dsm: np.ndarray) -> TimeseriesSummary:
    surface: SurfaceData = SurfaceData.prepare(dsm=dsm, pixel_size=1.0)
    location: Location = Location(latitude=57.7, longitude=12.0, utc_offset=1)
    weather: Weather = Weather(...)
    return solweig.calculate(surface, weather=[weather], location=location, output_dir="output/")
```
