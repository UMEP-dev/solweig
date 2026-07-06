# Thermal Comfort Indices

Tmrt quantifies the radiation absorbed by a person, but thermal comfort also depends on air temperature, humidity, and wind speed. SOLWEIG computes two standard indices that combine these variables into a single equivalent temperature.

## UTCI (Universal Thermal Climate Index)

UTCI represents the air temperature of a reference environment that would produce the same thermal strain as the actual conditions. It is the most widely used outdoor thermal comfort index.

**Applicable when:** A standardised metric is needed for heat stress mapping, urban planning, or public health applications.

### Summary grids (default)

UTCI summary grids (mean, max, min, day/night averages) are computed
as part of `TimeseriesSummary`:

```python
summary = solweig.calculate(surface=surface, weather=weather_list, output_dir="output/")
print(summary.report())  # Includes Tmrt, UTCI, sun hours, threshold exceedance
```

### Per-timestep GeoTIFFs

Include `"utci"` in `outputs` to save per-timestep UTCI GeoTIFFs:

```python
summary = solweig.calculate(
    surface=surface,
    weather=weather_list,
    output_dir="output/",
    outputs=["tmrt", "utci"],
)
```

### From a single-timestep run

```python
summary = solweig.calculate(surface, weather=[weather], location=location, output_dir="output/")
print(f"Mean UTCI: {summary.utci_mean.mean():.1f} deg C")
```

### UTCI stress categories

| UTCI (deg C) | Thermal stress |
| ------------ | -------------- |
| > 46 | Extreme heat stress |
| 38 to 46 | Very strong heat stress |
| 32 to 38 | Strong heat stress |
| 26 to 32 | Moderate heat stress |
| 9 to 26 | No thermal stress |
| 0 to 9 | Slight cold stress |
| -13 to 0 | Moderate cold stress |
| -27 to -13 | Strong cold stress |
| -40 to -27 | Very strong cold stress |
| < -40 | Extreme cold stress |

### Interpreting results

On a clear summer day (air temperature approximately 32 deg C), typical values are:

- **Sunlit areas:** Tmrt 55–70 deg C, UTCI 35–45 deg C (strong to very strong heat stress)
- **Shaded areas:** Tmrt 35–45 deg C, UTCI 28–34 deg C (moderate to strong heat stress)
- **Shade effect:** Tree shade typically reduces UTCI by 5–15 K, often sufficient to shift one stress category

Values outside these ranges are not necessarily erroneous — they depend on latitude, time of year, and surface materials — but extreme outliers (e.g., Tmrt > 80 deg C or UTCI > 55 deg C) may indicate input data issues.

### Performance

UTCI uses a polynomial approximation (~200 terms). Processing time is negligible relative to the main Tmrt calculation:

- Single grid: ~1 ms
- 72 timesteps: ~1 s

---

## PET (Physiological Equivalent Temperature)

PET is the air temperature of a reference indoor environment at which the human heat balance equals the actual outdoor conditions. Unlike UTCI, PET accepts customisable body parameters.

**Applicable when:** Thermal comfort assessments for specific populations (elderly, children, athletes) are required, or when the physiological model is needed for research purposes.

### Per-timestep PET

Include `"pet"` in `outputs`:

```python
summary = solweig.calculate(
    surface=surface,
    weather=weather_list,
    output_dir="output/",
    outputs=["tmrt", "pet"],
    human=solweig.HumanParams(weight=60, height=1.65, age=70),
)
```

### PET with custom human parameters

PET per-timestep GeoTIFFs are saved when `"pet"` is included in `outputs`:

```python
summary = solweig.calculate(
    surface, weather=[weather], location=location,
    output_dir="output/",
    outputs=["tmrt", "pet"],
    human=solweig.HumanParams(
        weight=60,           # kg
        height=1.65,         # m
        age=70,              # years
        sex=2,               # 1=male, 2=female
        activity=80.0,       # metabolic rate (W)
        clothing=0.5,        # clothing insulation (clo)
        posture="standing",
    ),
)
```

### PET thermal sensation

| PET (deg C) | Perception | Physiological stress |
| ----------- | ---------- | -------------------- |
| > 41 | Very hot | Extreme heat stress |
| 35 to 41 | Hot | Strong heat stress |
| 29 to 35 | Warm | Moderate heat stress |
| 23 to 29 | Slightly warm | Slight heat stress |
| 18 to 23 | Comfortable | No thermal stress |
| 13 to 18 | Slightly cool | Slight cold stress |
| 8 to 13 | Cool | Moderate cold stress |
| 4 to 8 | Cold | Strong cold stress |
| < 4 | Very cold | Extreme cold stress |

### PET performance

PET uses an iterative solver and requires more computation time than UTCI:

- Single grid: ~50 ms
- 72 timesteps: ~1 minute

!!! warning "PET computation time"
    PET requires iterative solving and takes approximately 50 times longer than UTCI per timestep. For large-scale studies, consider whether the customisable body parameters offered by PET are required.

---

## Choosing Between UTCI and PET

| | UTCI | PET |
| - | ---- | --- |
| **Computation** | Polynomial approximation | Iterative solver |
| **Human parameters** | Fixed reference person | Customisable (age, weight, clothing, etc.) |
| **Typical applications** | Heat warnings, urban planning, large-scale mapping | Detailed comfort studies, vulnerable populations |
| **Common in** | European heat action plans, WMO guidelines | German VDI guidelines, bioclimatology research |

UTCI is computationally efficient and has standardised stress categories referenced in public health guidance. PET allows customisation of individual body parameters for population-specific comfort studies.
