# Timeseries Calculations

For multi-timestep simulations (hours, days, or longer), pass a list of weather objects to `calculate()`. Thermal state is carried between timesteps, and results are saved to disk as they are computed.

## Thermal State Management

Ground and wall temperatures depend on accumulated heating from previous hours (thermal inertia). When given multiple weather timesteps, `calculate()` carries this between timesteps internally (via a `ThermalState` object you don't need to construct or pass yourself), producing accurate ground-level longwave radiation values.

## Basic Timeseries

```python
import solweig

# Load surface
surface = solweig.SurfaceData.prepare(
    dsm="data/dsm.tif",
    working_dir="cache/",
)

# Load weather from EPW file
weather_list = solweig.Weather.from_epw(
    "data/weather.epw",
    start="2025-07-01",
    end="2025-07-03",
)
location = solweig.Location.from_epw("data/weather.epw")

# Calculate all timesteps — results saved to output/
summary = solweig.calculate(
    surface=surface,
    location=location,
    weather=weather_list,
    output_dir="output/",
    outputs=["tmrt", "shadow"],
)

print(f"Processed {len(summary)} timesteps")
```

This creates timestamped GeoTIFFs and summary grids:

```text
output/
├── tmrt/
│   ├── tmrt_20250701_0000.tif
│   ├── tmrt_20250701_0100.tif
│   └── ...
├── shadow/
│   ├── shadow_20250701_0000.tif
│   └── ...
├── summary/
│   ├── tmrt_mean.tif
│   ├── tmrt_max.tif
│   ├── utci_mean.tif
│   └── ...
└── run_metadata.json            # All parameters for reproducibility
```

If only summary statistics are required (mean/max/min Tmrt, UTCI, sun hours), omit
`outputs` — no per-timestep files are written, but the summary grids are still
saved to `output/summary/`.

```python
summary = solweig.calculate(
    surface=surface,
    location=location,
    weather=weather_list,
    output_dir="output/",  # Summary grids + metadata saved here
)
```

## Inspecting Results

### Summary report

`report()` returns a text summary of the run:

```python
summary = solweig.calculate(
    surface=surface,
    weather=weather_list,
    output_dir="output/",
    outputs=["tmrt", "shadow"],
)
print(summary.report())
```

In Jupyter notebooks, placing `summary` as the last expression in a cell
renders the report via `_repr_html_`.

### Per-timestep timeseries

`summary.timeseries` contains 1-D arrays of spatial means at each timestep,
providing an overview of how conditions evolved over the simulation:

```python
ts = summary.timeseries
print(ts.datetime)       # timestamps
print(ts.ta)             # air temperature per step
print(ts.tmrt_mean)      # spatial mean Tmrt per step
print(ts.utci_mean)      # spatial mean UTCI per step
print(ts.sun_fraction)   # fraction of sunlit pixels per step
```

### Plotting

`plot()` produces a multi-panel figure showing temperature, radiation,
sun exposure, and meteorological inputs over time:

```python
summary.plot()                                  # interactive display
summary.plot(save_path="output/timeseries.png") # save to file
```

Requires `matplotlib` (`pip install matplotlib`).

## Per-Timestep UTCI and PET

UTCI and PET summary grids (mean, max, min, day/night averages) are included
in the returned `TimeseriesSummary`. To save per-timestep UTCI or PET GeoTIFFs,
include them in `outputs`:

```python
summary = solweig.calculate(
    surface=surface,
    weather=weather_list,
    output_dir="output/",
    outputs=["tmrt", "utci"],  # per-timestep Tmrt + UTCI saved to disk
)
```

The indices are computed inline during the main loop
(UTCI uses a Rust polynomial implementation; PET uses an iterative solver).

## Memory Management for Long Simulations

### Use a single call for the full period

For extended simulations (weeks or months), pass the full weather list to one
`calculate()` call:

```python
results = solweig.calculate(
    surface=surface,
    weather=weather_list,  # full period
    location=location,
    output_dir="output/",
)
```

Thermal state (ground and wall temperature history) carries between timesteps
automatically within a single call. Splitting the period into separate
`calculate()` calls resets that state at every boundary and produces the
temperature discontinuities described in
[Basic Usage](basic-usage.md#multiple-timesteps-timeseries).

Memory stays bounded regardless of simulation length: per-timestep grids are
written to `output_dir` as they are computed, and only the running summary
grids are held in memory. For grids too large to fit in memory, SOLWEIG tiles
the spatial computation internally (see
[Working with GeoTIFFs](geotiffs.md#large-rasters)); no chunking of the
weather series is needed.

## Performance

| Grid size | Surface prep (SVF) | Per timestep | 72 timesteps |
| --------- | -------------------- | -------------------- | ------------ |
| 100x100 | ~5 s | ~10 ms | ~1 s |
| 200x200 | ~67 s | ~20 ms | ~2 s |
| 500x500 | ~10 min | ~100 ms | ~8 s |

SVF is computed by `SurfaceData.prepare()`. For file workflows, a persistent `working_dir` avoids recomputing SVF on each run.

## Run Metadata

`calculate()` saves a `run_metadata.json` file recording all parameters used. This can be loaded for reproducibility:

```python
metadata = solweig.load_run_metadata("output/run_metadata.json")
print(f"Version: {metadata['solweig_version']}")
print(f"Timesteps: {metadata['timeseries']['timesteps']}")
print(f"Date range: {metadata['timeseries']['start']} to {metadata['timeseries']['end']}")
```
