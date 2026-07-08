# Configuration

SOLWEIG uses three parameter types, each with default values.

---

## Parameter Types

### 1. Human Parameters (Person-Specific)

These parameters describe the individual for whom the thermal environment is evaluated.

```python
human = HumanParams(
    abs_k=0.7,        # Shortwave absorption (0-1)
    abs_l=0.97,       # Longwave absorption (0-1)
    posture="standing",  # "standing" or "sitting"
    weight=75,        # kg (for PET post-processing)
    height=1.75,      # m (for PET post-processing)
    age=35,           # years (for PET post-processing)
    activity=80,      # W (for PET post-processing)
    clothing=0.9,     # clo (for PET post-processing)
)
```

**Defaults:** abs_k=0.7, abs_l=0.97, standing, 75 kg, 1.75 m, 35 years, 80 W, 0.9 clo

---

### 2. Physics Parameters (Site-Independent)

These parameters govern vegetation transmissivity and posture geometry. They are site-independent and apply universally.

```python
physics = solweig.load_physics("custom_trees.json")  # Optional
```

Contents:

- `Tree_settings`: Transmissivity (0.03), seasonal dates (day 97–300), trunk ratio (0.25)
- `Posture`: Geometry for standing/sitting (Fside, Fup, Fcyl, height)

**Defaults:** Bundled in the package (`physics_defaults.json`)

---

### 3. Material Library (Site-Specific)

These parameters define surface material properties per landcover class.

```python
materials = solweig.load_materials("site_materials.json")  # Optional — bundled defaults used when omitted
```

Contents per landcover class:

- Albedo, emissivity
- Ground temperature model parameters (TmaxLST, Ts_deg, Tstart)
- Wall thermal properties (specific heat, conductivity, density, thickness)

**Defaults:** Bundled in the package (`default_materials.json`) — loaded automatically when `materials` is omitted

---

## Model Behaviour Flags

These parameters control principal model behaviour:

### `use_anisotropic_sky` (default: follows `ModelConfig`, currently `True`)

- `False` — Isotropic sky model
- `True` — Perez anisotropic sky model

If set to `True`, shadow matrices must be prepared in advance.

### `conifer` (default: `False`)

- `False` — Deciduous trees (seasonal leaf on/off)
- `True` — Evergreen trees (year-round canopy)

### `use_ground_scheme` / `use_outgoing_longwave` (default: `False`)

Opt-in UMEP 2026a ground-surface scheme (force-restore/OHM surface
temperature and the solid-angle outgoing-longwave march). Experimental: the
two flags must be enabled together, a land-cover grid is required, tiled
processing is not supported, and the scheme currently runs warm against the
validation sites. Left off, model output is unchanged. See the
[Settings guide](guide/settings.md) for details.

---

## Usage Examples

### Default parameters

```python
import solweig

surface = solweig.SurfaceData.prepare(dsm="dsm.tif", working_dir="cache/")
weather = solweig.Weather.from_epw("weather.epw", start="2023-07-01")
summary = solweig.calculate(surface, weather, output_dir="output/")
```

### Custom human parameters

```python
summary = solweig.calculate(
    surface, weather,
    human=solweig.HumanParams(weight=70, height=1.65, posture="sitting"),
    output_dir="output/",
)
```

### Anisotropic sky model

```python
summary = solweig.calculate(
    surface, weather,
    use_anisotropic_sky=True,
    output_dir="output/",
)
```

### Evergreen trees

```python
summary = solweig.calculate(
    surface, weather,
    conifer=True,
    output_dir="output/",
)
```

### Custom physics parameters

```python
# custom_trees.json:
# {
#   "Tree_settings": {"Value": {"Transmissivity": 0.05, ...}},
#   "Posture": {"Standing": {...}, "Sitting": {...}}
# }

physics = solweig.load_physics("custom_trees.json")
summary = solweig.calculate(surface, weather, physics=physics, output_dir="output/")
```

### Landcover material variation

```python
materials = solweig.load_materials("site_materials.json")
surface = solweig.SurfaceData.prepare(
    dsm="dsm.tif",
    land_cover="landcover.tif",  # Classification raster with surface type IDs (0-7, 99-102)
    working_dir="cache/",
)
summary = solweig.calculate(surface, weather, materials=materials, output_dir="output/")
```

---

## Parameter Overview

| Type | Description | Example | When Required |
| ---- | ----------- | ------- | ------------- |
| **human** | Person characteristics | Weight, height, absorption | Custom body properties |
| **physics** | Site-independent constants | Tree transmissivity, posture geometry | Non-default tree species or seasonal periods |
| **materials** | Landcover properties | Albedo per surface type | Spatially varying surface materials |

---

## Levels of Control

### Level 1: Direct parameters

```python
calculate(
    ...,
    use_anisotropic_sky=True,
    conifer=True,
    human=HumanParams(weight=70),
)
```

### Level 2: Custom physics or materials files

```python
physics = solweig.load_physics("my_physics.json")
materials = solweig.load_materials("my_materials.json")
calculate(..., physics=physics, materials=materials)
```

### Level 3: Manual preprocessing

```python
# SurfaceData.prepare() handles wall and SVF computation automatically.
# For fine-grained control, construct SurfaceData manually and call
# preprocess() and compute_svf() separately:
surface = solweig.SurfaceData(dsm=dsm, pixel_size=1.0)
surface.preprocess()
surface.compute_svf()
```

---

## Backwards Compatibility

The legacy unified `parametersforsolweig.json` file is supported via
`ModelConfig.from_json()`, which extracts both human-body parameters
and the materials block in one step:

```python
config = solweig.ModelConfig.from_json("parametersforsolweig.json")
summary = solweig.calculate(
    surface, weather, location,
    config=config,
    output_dir="output/",
)
```

(The lower-level `solweig.load_params()` returns a `SimpleNamespace`
for callers that want to inspect or partially adapt the JSON contents.
Don't pass that namespace as a `params=` kwarg — `calculate()` accepts
`config=`, `physics=`, and `materials=`, not `params=`.)

---

## Summary

| Parameter | Purpose | Default | Customisation |
| --------- | ------- | ------- | ------------- |
| `human` | Person characteristics | Standing, 75 kg, 175 cm | `HumanParams(...)` object |
| `physics` | Site-independent constants | Bundled in package | `load_physics("custom.json")` |
| `materials` | Landcover properties | Bundled in package | `load_materials("site.json")` |
| `use_anisotropic_sky` | Sky model selection | `True` | Set to `False` for isotropic |
| `conifer` | Tree type | `False` (deciduous) | Set to `True` for evergreen |
