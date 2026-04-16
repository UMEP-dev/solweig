# Validation Report

SOLWEIG is validated against field radiation measurements from three sites in
Gothenburg, Sweden. All validation data — geodata, met files, measurement CSVs,
and POI coordinates (as GeoJSON) — are self-contained under `tests/validation/`
and run automatically in CI on every push and PR.

Each site's POI (point of interest) is loaded at runtime from a `poi.geojson`
file and projected onto the DSM grid. The GeoJSON coordinates were extracted
from the original shapefiles provided with each validation dataset.

---

## Sites

### Kronenhuset

- **Type:** Enclosed courtyard, central Gothenburg
- **Period:** 2005-10-07 (1 day, 12 daytime hours)
- **Resolution:** 1 m, EPSG:3007
- **POI:** (51, 117) — from `POI_KR.shp` measurement station coordinates
- **Reference:** Lindberg, Holmer & Thorsson (2008)
- **Data:** `tests/validation/kronenhuset/` (DSM, DEM, CDSM, landcover, met, poi.geojson)
- **Notes:** The only site that directly validates individual radiation budget
  components (K↓, K↑, L↓, L↑ and directional fluxes), not just Tmrt.
  Enclosed geometry with ~25% sky obstruction.

### Gustav Adolfs torg

- **Type:** Open square, central Gothenburg
- **Period:** 2005-10-11, 2006-07-26, 2006-08-01 (3 days, 43 daytime hours)
- **Resolution:** 2 m, EPSG:3006
- **POI:** (33, 77) — from `test_POI.shp` measurement station coordinates
- **Reference:** Lindberg, Holmer & Thorsson (2008)
- **Data:** `tests/validation/gustav_adolfs/` (DSM, DEM, CDSM, landcover, met, poi.geojson)
- **Notes:** One autumn day (heavily overcast) and two summer days.

### GVC (Gothenburg Geoscience Centre)

- **Type:** University campus courtyard, Gothenburg
- **Period:** 2010-07-07, 07-10, 07-12 (3 days, 30 daytime hours)
- **Resolution:** 2 m, EPSG:3006
- **POI:** (51, 122) — from `POI_GVC.shp` Site 1 measurement station coordinates
- **Reference:** Lindberg & Grimmond (2011)
- **Data:** `tests/validation/gvc/` (DSM, DEM, CDSM, landcover, met, poi.geojson)
- **Notes:** Three clear summer days. The POI corresponds to Site 1 from the
  paper. Rasters are labelled `_1m` but are actually 2 m resolution.

---

## Results — v0.1.0b82 (2026-04-11)

### Summary

| Metric               | Kronenhuset |  Gustav Adolfs |             GVC |
| -------------------- | ----------: | -------------: | --------------: |
| Tmrt RMSE range (°C) |         6.7 |      5.7–7.5   |    1.5–6.1      |
| Tmrt R² range        |        0.51 |     0.78–0.87  |   0.79–0.99     |
| Tmrt bias range (°C) |       +2.8  | -0.4 to +2.4   | +0.9 to +5.3    |
| Days                 |           1 |              3 |               3 |
| Total obs hours      |          12 |             43 |              30 |

### Tmrt — per-day detail

Anisotropic sky mode. Matched daytime hours only (sun above horizon with
valid observations).

| Site          | Day        | RMSE (°C) | MAE (°C) | Bias (°C) |   R² |   n |
| ------------- | ---------- | --------: | -------: | --------: | ---: | --: |
| Kronenhuset   | 2005-10-07 |       6.7 |      5.8 |      +2.8 | 0.51 |  12 |
| Gustav Adolfs | 2005-10-11 |       5.7 |      4.8 |      +1.1 | 0.87 |  12 |
| Gustav Adolfs | 2006-07-26 |       6.4 |      4.5 |      +2.4 | 0.85 |  16 |
| Gustav Adolfs | 2006-08-01 |       7.5 |      5.1 |      -0.4 | 0.78 |  15 |
| GVC           | 2010-07-07 |       1.5 |      1.3 |      +0.9 | 0.99 |  11 |
| GVC           | 2010-07-10 |       4.0 |      3.7 |      +3.7 | 0.79 |   7 |
| GVC           | 2010-07-12 |       6.1 |      5.3 |      +5.3 | 0.89 |  12 |

### Radiation components — all sites

All values anisotropic sky mode (isotropic in parentheses where notably
different). Units: W/m² for radiation, °C for Tmrt.

#### Kronenhuset (2005-10-07)

| Component |        RMSE |           Bias |
| --------- | ----------: | -------------: |
| K↓        | 51.5 (46.8) | -22.5 (-6.0)   |
| K↑        |         7.6 |           +5.0 |
| L↓        | 31.4 (38.4) | +30.5 (+37.9)  |
| L↑        |        13.3 |           +9.3 |
| Tmrt      |   6.7       |           +2.8 |

#### Gustav Adolfs torg (2005-10-11, 2006-07-26, 2006-08-01)

| Day        | Component |    RMSE |    Bias |
| ---------- | --------- | ------: | ------: |
| 2005-10-11 | K↓        |    48.0 |   -21.3 |
| 2005-10-11 | K↑        |     8.8 |    -2.3 |
| 2005-10-11 | L↓        |    57.3 |   +55.2 |
| 2005-10-11 | L↑        |    10.8 |    +1.4 |
| 2006-07-26 | K↓        |    60.3 |    -4.3 |
| 2006-07-26 | K↑        |    14.8 |    -0.8 |
| 2006-07-26 | L↓        |    51.5 |   +39.3 |
| 2006-07-26 | L↑        |    24.5 |   -10.3 |
| 2006-08-01 | K↓        |   157.0 |   -62.8 |
| 2006-08-01 | K↑        |    22.2 |    -3.5 |
| 2006-08-01 | L↓        |    45.9 |   +40.9 |
| 2006-08-01 | L↑        |    28.1 |    -4.3 |

#### GVC (2010-07-07, 07-10, 07-12)

| Day        | Component |    RMSE |    Bias |
| ---------- | --------- | ------: | ------: |
| 2010-07-07 | K↓        |    39.9 |   -17.5 |
| 2010-07-07 | K↑        |    23.1 |   +16.5 |
| 2010-07-07 | L↓        |    21.8 |   +18.9 |
| 2010-07-07 | L↑        |     8.9 |    -4.4 |
| 2010-07-10 | K↓        |    21.4 |   +16.7 |
| 2010-07-10 | K↑        |    30.8 |   +27.8 |
| 2010-07-10 | L↓        |    38.7 |   +38.5 |
| 2010-07-10 | L↑        |    32.5 |   +31.7 |
| 2010-07-12 | K↓        |   111.6 |   +27.1 |
| 2010-07-12 | K↑        |    26.5 |   +21.8 |
| 2010-07-12 | L↓        |    29.8 |   +29.2 |
| 2010-07-12 | L↑        |    24.2 |   +22.1 |

### Ldown — all sites

| Site          | Day        | RMSE (W/m²) | Bias (W/m²) |
| ------------- | ---------- | ----------: | ----------: |
| Kronenhuset   | 2005-10-07 |        31.4 |       +30.5 |
| Gustav Adolfs | 2005-10-11 |        57.3 |       +55.2 |
| Gustav Adolfs | 2006-07-26 |        51.5 |       +39.3 |
| Gustav Adolfs | 2006-08-01 |        45.9 |       +40.9 |
| GVC           | 2010-07-07 |        21.8 |       +18.9 |
| GVC           | 2010-07-10 |        38.7 |       +38.5 |
| GVC           | 2010-07-12 |        29.8 |       +29.2 |

---

## POI sweep analysis

The POI sweep runs the full SOLWEIG pipeline once per site, then evaluates
the modelled Tmrt time series at every ground-level pixel against the same
single set of field observations. The resulting RMSE/R²/bias heatmaps show
how sensitive the validation statistics are to the assumed measurement
location — they do not represent spatial maps of model error. A pixel with
low RMSE means that, had the instrument been placed there, the modelled and
observed Tmrt would have agreed more closely.

Run with: `pytest tests/validation/test_poi_sweep_all_sites.py -v -s`

### Gustav Adolfs torg sweep

![POI sweep RMSE — Gustav Adolfs](tests/validation/gustav_adolfs/poi_sweep_results/poi_sweep_rmse.png)

The measurement station POI (33, 77) sits in the open square. Pixels near
the western buildings show lower RMSE (~8 °C vs ~13 °C), where shadow
patterns are more uniform. The open-square location is the documented
measurement position.

### Kronenhuset sweep

![POI sweep RMSE — Kronenhuset](tests/validation/kronenhuset/poi_sweep_results/poi_sweep_rmse.png)

The measurement station POI (51, 117) is in the courtyard, consistent with
the described field setup (Tmrt RMSE ~6 °C).

### GVC sweep

![POI sweep RMSE — GVC](tests/validation/gvc/poi_sweep_results/poi_sweep_rmse.png)

The measurement station POI (51, 122) corresponds to Site 1 from
Lindberg & Grimmond (2011). This pixel sits at the edge of a dense tree
canopy in the CDSM (heights of 7–18 m immediately to the west and south),
making the modelled shadow state sensitive to sub-pixel canopy position.
Nearby pixels in more open areas show lower RMSE (~4–5 °C).

---

## Known limitations

### Kdown at open sites

Point-level downwelling shortwave (Kdown) has high RMSE (175–344 W/m²).
At any single pixel the shadow state is binary, so a small shift in the
modelled shadow boundary produces ~800 W/m² differences between timesteps.
Spatially averaged Kdown would show considerably lower error.

### Ldown overestimation

The model overestimates Ldown at all sites (bias +39 to +83 W/m²). The
SOLWEIG Ldown formulation (Jonsson et al. 2006) fills the non-sky hemisphere
with wall emissions at emissivity 0.90 and air temperature. In practice,
shaded walls are cooler than air temperature, which introduces a positive bias.

- At SVF = 1.0 (open sky), clear-sky Ldown matches observations well.
- The bias increases at sites with lower SVF, where more of the hemisphere
  is filled with wall emissions.
- The Jonsson et al. (2006) empirical correction of −25 W/m² is present but
  commented out in all UMEP releases (2021a, 2022a, 2025a) and is not applied
  here.

### GVC Tmrt accuracy

The GVC site shows higher Tmrt RMSE (11–16 °C) and near-zero R² compared to
the other two sites. Contributing factors include:

- The measurement station (Site 1) is adjacent to dense tree canopy in the
  CDSM, making the modelled shadow state sensitive to sub-pixel canopy
  position at 2 m resolution.
- Only 7–12 matched hours per day are available, which limits the statistical
  power of per-day metrics such as R².

---

## Comparison with published results

Lindberg et al. (2008) report aggregate statistics over 7 days at two
Gothenburg sites (~189 hours):

| Component |   R² |      RMSE |
| --------- | ---: | --------: |
| Tmrt      | 0.94 |     4.8 K |
| L↓        | 0.73 | 17.5 W/m² |
| L↑        | 0.94 | 15.6 W/m² |

The Kronenhuset Tmrt RMSE of 6.0 °C from this implementation is in a similar
range to the paper's 4.8 K. The paper's statistics are aggregated across more
hours and sites, and the paper validates against 1-minute averaged measurements,
whereas the met data used here are hourly.

---

## Running validation tests

```bash
# All validation tests (fast data-loading + slow pipeline)
pytest tests/validation/ -m validation

# Just the fast data-loading checks
pytest tests/validation/ -m "validation and not slow"

# A single site
pytest tests/validation/test_validation_gvc.py -v -s

# POI sensitivity sweep (all sites, generates PNG heatmaps)
pytest tests/validation/test_poi_sweep_all_sites.py -v -s
```

---

## Version history

| Version | Date | Sites | Tmrt RMSE range | Key changes |
| --- | --- | ---: | ---: | --- |
| 0.1.0b57 | 2026-03-05 | 3 | 3.4–17.7 °C | Initial 3-site validation. POI sweep analysis added for all sites. Ldown wall-temperature bias documented. |
| 0.1.0b58 | 2026-03-06 | 3 | 3.4–17.7 °C | Add validation CI job. Remove non-reproducible Kolumbus/Montpellier tests. Clarify POI sweep documentation. |
| 0.1.0b59 | 2026-03-06 | 3 | 4.0–17.7 °C | Move GVC POI to courtyard cluster (70, 126). Shift Kronenhuset POI +1 col to match shadow profile. Move validation report to repo root. |
| 0.1.0b60 | 2026-03-06 | 3 | 4.0–17.7 °C | GPU GVF compute shader (wgpu). Cached thermal accumulation offloaded to GPU with automatic CPU fallback. |
| 0.1.0b61 | 2026-03-08 | 3 | 2.4–18.9 °C | Fix file-mode prepare() order (preprocess before walls/SVF), fix tiled wall propagation, fix single-Weather API, fix ModelConfig.from_json() materials, fix QGIS LC override inheritance, fix EPW cross-year timestamps. Ldown RMSE increased due to corrected SVF geometry (absolute heights). |
| 0.1.0b62 | 2026-03-08 | 3 | 2.4–18.9 °C | 35 code review fixes: clearness index, UTC offsets, cache validation, input mutation, dead code, orchestration dedup, lazy imports, PET convergence warning, GPU mutex recovery. |
| 0.1.0b66 | 2026-03-09 | 3 | 6.0–18.9 °C | Use original measurement station POIs from shapefiles (saved as GeoJSON). GVC POI corrected from (70,126) to (51,122) per POI_GVC.shp. KR rasters moved to self-contained validation folder. All POIs loaded at runtime from poi.geojson via conftest helper. |
| 0.1.0b69 | 2026-03-14 | 3 | 6.0–18.9 °C | Fix SVF Options 3/4 zenith patch count (no effect on default Option 2). Fix docs, specs, license refs, CI matrix. Move geopandas to optional. Validation numbers unchanged from b66. |
| 0.1.0b70 | 2026-03-14 | 3 | 6.0–18.9 °C | Fix sitting posture producing negative Tmrt with anisotropic sky (#9). Add box direct beam splitting. Validation unchanged (standing posture). |
| 0.1.0b71 | 2026-03-14 | 3 | 6.0–18.9 °C | Docs-only: clarify TMY nature of PVGIS downloads in docstrings, user docs, and QGIS plugin (#8). Validation unchanged. |
| 0.1.0b72 | 2026-03-17 | 3 | 6.7–17.6 °C | Fix false vegetation shadows on slopes: sub-threshold CDSM/TDSM set to NaN instead of DEM height; underground vegetation cleared. Ldown improved at Kronenhuset (39→32 W/m²) and Gustav Adolfs (84→74 W/m²). Relax SVF veg golden tolerance (known shadowingfunction\_20 vs \_23 divergence). |
| 0.1.0b74 | 2026-03-18 | 3 | 6.7–17.6 °C | Fix rasterio resampling pixel drift (from\_bounds inexact pixel size). Fix QGIS phantom vegetation (fill\_nan overwriting CDSM NaN markers). Add SurfaceData.load(); eliminate QGIS/core duplication. Fix progress bar regression. Validation unchanged from b72. |
| 0.1.0b78 | 2026-03-29 | 3 | 6.7–17.6 °C | Fix phantom vegetation in tiled timeseries: tile-extracted surfaces inherit \_nan\_filled state, preventing double fill\_nan from overwriting intentional CDSM/TDSM NaN markers with DEM values. Also fix tiling buffer overflow on small rasters (core=1 / segfault). Unified tile-outer timeseries architecture. Validation restored to b72 baseline. |
| 0.1.0b81 | 2026-04-08 | 3 | 6.7–17.6 °C | Fix tiled SVF core window overflow when buffer\_pixels > tile\_size (overlap clamped to actual raster extent). Remove dead Rust code (steradians\_for\_patch\_option, weighted\_patch\_sum\_pure). Validation unchanged from b78. |
| 0.1.0b82 | 2026-04-11 | 3 | 1.5–7.5 °C | Fix inverted `scale` convention in Rust shadow caster (dz off by `pixel_size²` at non-1 m rasters). Also: DEM stair-step smoothing, `prepare()` warm-run fast-path, tile sizer buffer fix, `GridAccumulator.update()` in-place ufuncs, QGIS metadata consolidation. |
| 0.1.0b83 | 2026-04-13 | 3 | 1.5–7.5 °C | Docs-only: correct PVGIS TMY reference period (2005–2020 → 2005–2023 for v5.3) and clarify that TMY row timestamps legitimately span multiple years because each month is a real historical month. Validation unchanged from b82. |
| 0.1.0b84 | 2026-04-16 | 3 | 1.5–7.5 °C | Rust wall-aspect kernel: promote internal math to f64 and switch to banker's rounding to match numpy/UMEP precision and tie-breaking. Input/output arrays stay f32 — promotion is strictly internal, no change to data-array memory. Delete Python Goodwin fallback so solweig has a single numerical path (QGIS and pip users get identical output). Validation Tmrt numbers shifted by ≤0.13 °C per day (all within thresholds); displayed range unchanged. Plus: 8 new public API exports for plugin/external tools, plugin error wrapping now surfaces SolweigError structured attributes, and ~400 lines of dead helpers removed. |

---

## References

1. Lindberg, F., Holmer, B. & Thorsson, S. (2008). SOLWEIG 1.0 — Modelling
   spatial variations of 3D radiant fluxes and mean radiant temperature in
   complex urban settings. _Int. J. Biometeorol._ 52, 697–713.

2. Lindberg, F. & Grimmond, C.S.B. (2011). The influence of vegetation and
   building morphology on shadow patterns and mean radiant temperature in
   urban areas. _Theor. Appl. Climatol._ 105, 311–323.

3. Jonsson, P., Eliasson, I., Holmer, B. & Grimmond, C.S.B. (2006). Longwave
   incoming radiation in the Tropics: results from field work in three African
   cities. _Theor. Appl. Climatol._ 85, 185–201.
