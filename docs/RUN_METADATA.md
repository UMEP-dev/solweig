# Run Metadata and Provenance

Each calculation saves a `run_metadata.json` file that records all parameters, inputs, and configuration used.

---

## Purpose

Run metadata provides a complete record of the parameters and configuration for a SOLWEIG calculation. This supports:

1. **Reproducibility** — Re-running the same calculation at a later date
2. **Audit trail** — Documenting parameters for publications or reports
3. **Debugging** — Identifying differences between runs
4. **Archiving** — Preserving the complete experimental setup alongside results

---

## Metadata Saving

When `calculate()` is called with an `output_dir`, metadata is saved to that directory:

```python
import solweig

surface = solweig.SurfaceData.prepare(dsm="dsm.tif", working_dir="cache/")
weather = solweig.Weather.from_epw("weather.epw", start="2023-07-01")

summary = solweig.calculate(
    surface, weather,
    human=solweig.HumanParams(weight=70, height=1.65),
    use_anisotropic_sky=True,
    output_dir="output/",
)

# Metadata saved at: output/run_metadata.json
```

---

## Recorded Fields

The metadata file contains the following sections:

### 1. Execution Info

- `solweig_version`: SOLWEIG version used
- `run_timestamp`: Date and time of the calculation

### 2. Grid

- `rows`, `cols`: Grid dimensions
- `pixel_size`: Pixel size in metres
- `crs`: Coordinate reference system

### 3. Location

- `latitude`, `longitude`, `utc_offset`

### 4. Timeseries

- `start`, `end`: Date range (ISO format)
- `timesteps`: Number of timesteps

### 5. Parameters

- `use_anisotropic_sky`: Sky model type
- `conifer`: Evergreen or deciduous trees

### 6. Outputs

- `directory`: Output directory path
- `variables`: List of output types saved

### 7. Human Parameters (if provided)

- `abs_k`, `abs_l`: Absorption coefficients
- `posture`: Standing or sitting

### 8. Physics and Materials (if custom)

- Full parameter dictionaries when custom physics or materials are provided

---

## Loading and Inspecting Metadata

```python
import solweig

metadata = solweig.load_run_metadata("output/run_metadata.json")

print(f"Calculation performed: {metadata['run_timestamp']}")
print(f"SOLWEIG version: {metadata['solweig_version']}")
print(f"Location: {metadata['location']['latitude']:.2f}°N")
print(f"Anisotropic sky: {metadata['parameters']['use_anisotropic_sky']}")
print(f"Weather period: {metadata['timeseries']['start']} to {metadata['timeseries']['end']}")
print(f"Timesteps: {metadata['timeseries']['timesteps']}")
```

If human parameters were provided:

```python
if "human" in metadata:
    print(f"Posture: {metadata['human']['posture']}")
    print(f"abs_k: {metadata['human']['abs_k']}")
```

---

## Manual Metadata Creation

For custom workflows, metadata can be created and saved separately:

```python
import solweig

surface = solweig.SurfaceData.prepare(dsm="dsm.tif", working_dir="cache/")
weather_series = solweig.Weather.from_epw("weather.epw")  # returns list[Weather]
location = solweig.Location.from_surface(surface)
human = solweig.HumanParams(weight=70)

metadata = solweig.create_run_metadata(
    surface=surface,
    location=location,
    weather_series=weather_series,
    human=human,
    physics=None,
    materials=None,
    use_anisotropic_sky=True,
    conifer=False,
    output_dir="output/",
    outputs=["tmrt", "shadow"],
)

solweig.save_run_metadata(metadata, output_dir="custom_dir/", filename="my_metadata.json")
```

---

## Example Metadata File

A representative `run_metadata.json`:

```json
{
  "solweig_version": "0.0.1a1",
  "run_timestamp": "2024-07-15T14:30:22.123456",
  "grid": {
    "rows": 400,
    "cols": 400,
    "pixel_size": 1.0,
    "crs": "PROJCS[...]"
  },
  "location": {
    "latitude": 37.98,
    "longitude": 23.73,
    "utc_offset": 2
  },
  "timeseries": {
    "start": "2023-07-01T00:00:00",
    "end": "2023-07-03T23:00:00",
    "timesteps": 72
  },
  "parameters": {
    "use_anisotropic_sky": true,
    "conifer": false
  },
  "outputs": {
    "directory": "/path/to/output",
    "variables": ["tmrt", "shadow"]
  },
  "human": {
    "abs_k": 0.7,
    "abs_l": 0.97,
    "posture": "standing"
  }
}
```

---

## Use Cases

### Research Publications

Record parameters for reproducible results:

```python
summary = solweig.calculate(
    surface, weather,
    human=solweig.HumanParams(weight=75, height=1.80),
    use_anisotropic_sky=True,
    output_dir="paper_results/",
)

# Metadata is saved alongside results and may be included in supplementary materials.
```

### Comparing Runs

Compare metadata from different runs to identify parameter differences:

```python
meta_run1 = solweig.load_run_metadata("run1/run_metadata.json")
meta_run2 = solweig.load_run_metadata("run2/run_metadata.json")

if "human" in meta_run1 and "human" in meta_run2:
    print("Run 1 posture:", meta_run1['human']['posture'])
    print("Run 2 posture:", meta_run2['human']['posture'])

print("Run 1 sky model:", meta_run1['parameters']['use_anisotropic_sky'])
print("Run 2 sky model:", meta_run2['parameters']['use_anisotropic_sky'])
```

### Archival

The output directory contains the complete experimental record:

```python
# output/
#   ├── run_metadata.json      <-- Complete parameter record
#   ├── tmrt_2023-07-01_1200.tif
#   ├── tmrt_2023-07-01_1300.tif
#   └── ...
```

### Debugging

Verify parameters when results require investigation:

```python
metadata = solweig.load_run_metadata("output/run_metadata.json")

if not metadata['parameters']['use_anisotropic_sky']:
    print("Note: Anisotropic sky was disabled.")

if "human" in metadata and metadata['human']['posture'] != 'standing':
    print(f"Note: Results computed for {metadata['human']['posture']} posture.")
```

---

## Custom Physics and Materials

When custom physics or materials files are used, the full parameter values are embedded in the metadata:

```python
physics = solweig.load_physics("custom_trees.json")

summary = solweig.calculate(
    surface, weather,
    physics=physics,
    output_dir="output/",
)

# Metadata includes physics.full_params: {...complete physics parameters...}
```

This ensures the metadata is self-contained; the original physics file is not required to interpret the record.

---

## API Reference

### `create_run_metadata()`

Create a metadata dictionary for a SOLWEIG run.

**Parameters:**

- `surface`: SurfaceData object
- `location`: Location object
- `weather`: List of Weather objects
- `human`: HumanParams object (or None for defaults)
- `physics`: Physics parameters from `load_physics()` (or None)
- `materials`: Materials from `load_materials()` (or None)
- `use_anisotropic_sky`: Whether the anisotropic sky model was used
- `conifer`: Whether conifer mode was used
- `output_dir`: Output directory path
- `outputs`: List of output types saved

**Returns:** Dictionary containing complete metadata

---

### `save_run_metadata()`

Save a metadata dictionary to a JSON file.

**Parameters:**

- `metadata`: Metadata dictionary from `create_run_metadata()`
- `output_dir`: Directory in which to save the metadata file
- `filename`: Filename (default: `"run_metadata.json"`)

**Returns:** Path to the saved metadata file

---

### `load_run_metadata()`

Load metadata from a JSON file.

**Parameters:**

- `metadata_path`: Path to the metadata JSON file

**Returns:** Metadata dictionary

---

## Summary

| Property | Description |
| -------- | ----------- |
| Saving | Metadata is written when `output_dir` is provided |
| Scope | All parameters, inputs, and configuration are recorded |
| Reproducibility | Contains the information required to re-run the calculation |
| Self-contained | Includes full custom physics/materials values, not file paths alone |
| Versioning | Records the SOLWEIG version for forward compatibility |
