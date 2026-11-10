# Validation Report

SOLWEIG is validated against field radiation measurements from three sites in
Gothenburg, Sweden. All validation data (geodata, met files, measurement CSVs,
and POI coordinates as GeoJSON) are self-contained under `tests/validation/`
and run automatically in CI on every push and PR.

Each site's POI (point of interest) is loaded at runtime from a `poi.geojson`
file and projected onto the DSM grid. The GeoJSON coordinates were extracted
from the original shapefiles provided with each validation dataset.

Each site section shows observed (solid black) vs modelled (dashed red) Tmrt
and the four radiation components at the POI, stitched across all measurement
days into one continuous timeline. Per-panel RMSE / Bias annotations summarise
agreement. Grey vertical bands on the Tmrt panel mark hours when the modelled
shadow fraction at the POI is < 0.5, so shade-timing mismatches are visible at
a glance.

---

## Summary — v0.1.0b97 (2026-11-10)

b97 makes two model changes, and neither moves the numbers in the table
below. The first corrects the anisotropic sky on surfaces without a canopy
layer, where building-blocked sky patches had emitted at air temperature
instead of wall temperature (issue 15). All three validation sites have a
canopy layer, so their results are identical to b96. The second aligns the
opt-in 2026a ground scheme with the current reference implementation,
UMEP-dev/umep-solweig; the scheme has its own field comparison further
down, rerun for this release.

The table reflects the met-file clock correction made in b96. The prepared
met files are labelled one hour early relative to the hour-ending CET
convention that the measurement CSVs and the model's sun position both
follow, so the harness adds one hour to the met timestamps; the measurement
CSVs are untouched. Details and the evidence are in
[§ Time base of the validation data](#time-base-of-the-validation-data).

| Metric               | Kronenhuset | Gustav Adolfs |          GVC |
| -------------------- | ----------: | ------------: | -----------: |
| Tmrt RMSE range (°C) |         6.3 |       3.3–4.8 |      4.1–7.0 |
| Tmrt R² range        |        0.64 |     0.94–0.97 |    0.58–0.87 |
| Tmrt bias range (°C) |        +2.0 |  +0.5 to +2.3 | +0.7 to +3.8 |
| Days                 |           1 |             3 |            3 |
| Total obs hours      |          12 |            43 |           30 |

> **Read this before interpreting the radiation panels.** Modelled L↓ sits
> 19 to 45 W/m² above the observations at every site. This traces to the
> published Ldown formulation rather than to calibration; see
> [§ Ldown positive bias](#ldown-positive-bias) below. K↓ at open sites is
> shadow-edge-sensitive and can spike on a single mis-aligned hour; see
> [§ Kdown at open sites](#kdown-at-open-sites).

---

## Kronenhuset

- **Type:** Enclosed courtyard, central Gothenburg
- **Period:** 2005-10-07 (1 day, 12 daytime hours)
- **Resolution:** 1 m, EPSG:3007
- **POI:** (51, 117), from `POI_KR.shp` measurement station coordinates
- **Reference:** Lindberg, Holmer & Thorsson (2008)
- **Data:** `tests/validation/kronenhuset/` (DSM, DEM, CDSM, landcover, met, poi.geojson)
- **Notes:** The only site that directly validates individual radiation budget
  components (K↓, K↑, L↓, L↑ and directional fluxes), not just Tmrt.
  Enclosed geometry with ~25 % sky obstruction.

![Kronenhuset POI timeseries](tests/validation/kronenhuset/timeseries_plots/timeseries.png)

- The POI is in shade for almost the entire day, with one short sunlit
  window in the mid-afternoon. The observations show two sunlit hours (15
  and 16 h); the model shows one (15 h), where it overshoots Tmrt by 16 °C
  and K↓ by 100 W/m², and is already back in shade at 16 h (−11 °C). The
  modelled window is therefore about an hour short of the observed one; the
  shadow boundary leaves the POI early. Outside those two hours the model
  sits +0.3 to +4.3 °C above the observations. This single transition
  drives most of the 6.3 °C RMSE.
- The L↓ panel shows a steady +17 to +26 W/m² offset across every hour. The
  cool-wall bias is systematic and independent of sun position (see Known
  limitations).

---

## Gustav Adolfs torg

- **Type:** Open square, central Gothenburg
- **Period:** 2005-10-11, 2006-07-26, 2006-08-01 (3 days, 43 daytime hours)
- **Resolution:** 2 m, EPSG:3006
- **POI:** (33, 77), from `test_POI.shp` measurement station coordinates
- **Reference:** Lindberg, Holmer & Thorsson (2008)
- **Data:** `tests/validation/gustav_adolfs/` (DSM, DEM, CDSM, landcover, met, poi.geojson)
- **Notes:** One autumn day (heavily overcast) and two summer days.

![Gustav Adolfs POI timeseries](tests/validation/gustav_adolfs/timeseries_plots/timeseries.png)

- 2005-10-11 (autumn, low sun): K↓ tracks the observations hour for hour
  (the met file carries measured direct and diffuse components). Tmrt RMSE
  4.2 °C, R² 0.94. The afternoon shade transition is modelled one hour
  early: at 16 h the model has the POI in shade (17 °C) while the
  observation is still sunlit (29 °C).
- 2006-07-26 (clear summer): K↓ tracks closely; Tmrt runs +2 to +7 °C warm
  through the middle of the day (bias +2.3 °C, RMSE 4.8 °C) and the evening
  shade transition at 19 h is again modelled early (−9 °C).
- 2006-08-01 (clear summer): RMSE 3.3 °C, R² 0.97. The one-hour K↓ dip at
  13 h (modelled 470 vs 770 W/m² observed) is a modelled shade pass that the
  observations do not show; it drives the day's K↓ RMSE to 90 W/m² and a
  −6.5 °C Tmrt difference for that hour. Before b96 this day showed a
  600 W/m² K↓ "spike" that was attributed to cloud; it was the met-file
  clock offset.

---

## GVC (Gothenburg Geoscience Centre)

- **Type:** University campus courtyard, Gothenburg
- **Period:** 2010-07-07, 07-10, 07-12 (3 days, 30 daytime hours)
- **Resolution:** 2 m, EPSG:3006
- **POI:** (51, 122), from `POI_GVC.shp` Site 1 measurement station coordinates
- **Reference:** Lindberg & Grimmond (2011)
- **Data:** `tests/validation/gvc/` (DSM, DEM, CDSM, landcover, met, poi.geojson)
- **Notes:** Three clear summer days. The POI corresponds to Site 1 from the
  paper. Rasters are labelled `_1m` but are actually 2 m resolution.

![GVC POI timeseries](tests/validation/gvc/timeseries_plots/timeseries.png)

- 2010-07-07: RMSE 4.1 °C, bias +0.7 °C. The POI goes into shade late in
  the morning; the model keeps it sunlit one hour longer (11 h: modelled
  54 °C vs 42 °C observed) and then follows the observations closely.
- 2010-07-10: only 7 hours, all in shade; modelled Tmrt sits +3.5 to +5 °C
  above observed throughout (bias +3.8 °C, no abrupt divergence; the L↓
  cool-wall bias and K↑ both contribute).
- 2010-07-12: RMSE 7.0 °C, bias +3.7 °C. The model keeps the POI sunlit
  through 10 and 11 h where the observations show it already shaded,
  raising Tmrt by 10 to 13 °C in those two hours, and stays +3 to +8 °C
  warm in the shade that follows. This is the dominant driver of the day's
  bias.
- The Site 1 measurement station sits at the edge of a dense tree canopy in
  the CDSM (heights of 7–18 m immediately to the west and south), so the
  modelled shadow state is sensitive to sub-pixel canopy position at 2 m
  resolution. This is the dominant source of day-to-day bias variability at
  this site.

---

## Time base of the validation data

Three clocks have to agree for an hour-by-hour comparison: the labels on
the measurement CSVs, the labels on the met forcing, and the time at which
the model places the sun for each row. The model follows the UMEP
convention: a row labelled `h` is the mean of the hour ending at `h` in
local standard time, so the sun is evaluated at `h` minus half a timestep
with `utc_offset = 1` (CET).

Lag fits between the three curves (met K↓, measured K↓ at the POI, and the
model's sine of solar altitude, at lags of −2 to +2 h) show that the
measurement CSVs follow that convention and the prepared met files do not:

- Measured K↓ at label `h` matches the model's sun at label `h` (best lag
  0 at Gustav Adolfs torg and GVC).
- Met K↓ at label `h` matches the model's sun at label `h + 1` on every
  clear day (Gustav Adolfs 2005-10-11, Kronenhuset 2005-10-07, GVC
  2010-07-12), and measured K↓ at label `h` matches met K↓ at label
  `h − 1` (e.g. Gustav Adolfs 2006-07-26: met 51, 132, 265, 499 W/m² at 5
  to 8 h; measured 52, 120, 270, 479 W/m² at 6 to 9 h).

The met files are therefore labelled one hour early (hour-starting CET, or
equivalently hour-ending UTC). Until b95 every timestep was driven by
forcing recorded one hour after its sun position, which put the modelled
K↓ peak an hour before the measured one and produced the spurious K↓
divergences the earlier reports attributed to cloud. Since b96 the harness
adds one hour to the met timestamps (`tests/validation/met_clock.py`,
applied by every site test and by the plot and POI-sweep scripts). The
measurement CSVs and the library are untouched: the convention of
`Weather.from_umep_met` and `Location.utc_offset` is UMEP's, and the
correction belongs to these data files. Shifting the observations instead
would align K↓ but move the shadows an hour off.

---

## Known limitations

### Kdown at open sites

Point-level downwelling shortwave (Kdown) is sensitive to shadow timing.
At any single pixel the shadow state is binary, so a small shift in the
modelled shadow boundary produces several hundred W/m² of difference in a
single hourly timestep. The 13 h dip on Gustav Adolfs 2006-08-01
illustrates this: one modelled shade hour drives the day's K↓ RMSE to
90 W/m². Spatially averaged Kdown would show considerably lower error.

### Ldown positive bias

Modelled Ldown exceeds the observations at all sites (bias +19 to +45 W/m²).
The SOLWEIG Ldown formulation (Jonsson et al. 2006) fills the non-sky
hemisphere with wall emissions at emissivity 0.90 and air temperature. In
practice, shaded walls are cooler than air temperature, which introduces a
positive bias.

- At SVF = 1.0 (open sky), clear-sky Ldown matches observations well.
- The bias increases at sites with lower SVF, where more of the hemisphere is
  filled with wall emissions.
- The Jonsson et al. (2006) empirical correction of −25 W/m² is present but
  commented out in all UMEP releases (2021a, 2022a, 2025a) and is not applied
  here.
- The bias appears as a steady offset across hours (visible in every site's
  L↓ panel), confirming it is formulation-driven rather than sun-position
  dependent.

### Shade-timing sensitivity near vegetation

When the POI sits adjacent to dense canopy or a shadow edge, the modelled
sun/shade state is sensitive to sub-pixel CDSM position at 2 m resolution.
This produces day-by-day swings of several °C in Tmrt, most visible at
GVC, where 2010-07-07 has RMSE 4.1 °C (bias +0.7 °C) but 2010-07-12 reaches
7.0 °C (bias +3.7 °C) using the same site, POI, and surface model.

---

## UMEP 2026a ground scheme (opt-in): field comparison

Rerun 2026-11-10 for b97 with `scripts/validate_ground_scheme.py`, on the
corrected met clock (see
[§ Time base of the validation data](#time-base-of-the-validation-data)).
Same three sites, anisotropic sky, `use_ground_scheme=True` /
`use_outgoing_longwave=True`, site parameter files extended with the
bundled OHM, Tg_ini and Tm_ini blocks (the legacy site JSONs predate them).
All columns use one prepared surface and parameter set per site, the suite
configuration; the old-clock columns reproduce the July 2026 run, and the
old-clock baseline column matches the b89 to b95 summary. The b90 to b96
scheme columns are from the 2026-10-01 run. POI Tmrt against measurements,
RMSE / bias (°C):

| Site | Day | Baseline, old clock | Baseline, corrected | Scheme b90 to b96, old clock | Scheme b90 to b96, corrected | Scheme b97, corrected |
| --- | --- | --- | --- | --- | --- | --- |
| Kronenhuset | 2005-10-07 | 6.6 / +2.6 | 6.3 / +2.0 | 22.0 / +21.4 | 21.5 / +20.9 | 11.4 / +10.0 |
| Gustav Adolfs | 2005-10-11 | 5.7 / +1.5 | 4.2 / +0.5 | 13.6 / +12.7 | 12.3 / +11.7 | 4.5 / +2.5 |
| Gustav Adolfs | 2006-07-26 | 7.1 / +3.7 | 4.8 / +2.3 | 16.4 / +15.8 | 14.4 / +14.3 | 4.8 / +4.3 |
| Gustav Adolfs | 2006-08-01 | 7.3 / +0.6 | 3.3 / +1.8 | 15.1 / +13.6 | 15.2 / +14.6 | 6.1 / +4.7 |
| GVC | 2010-07-07 | 2.4 / +1.4 | 4.1 / +0.7 | 14.9 / +14.7 | 14.6 / +13.9 | 5.6 / +3.8 |
| GVC | 2010-07-10 | 3.7 / +3.4 | 4.2 / +3.8 | 18.8 / +18.7 | 18.5 / +18.4 | 7.7 / +7.5 |
| GVC | 2010-07-12 | 6.9 / +5.7 | 7.0 / +3.7 | 18.3 / +18.1 | 17.5 / +16.6 | 8.1 / +6.3 |

The clock correction moved the b90 to b96 scheme numbers by at most 2 °C.
Their bias was consistently positive: RMSE 12.3 to 21.5 °C and bias +11.7
to +20.9 °C, against 3.3 to 7.0 °C and +0.5 to +3.8 °C for the baseline on
the same inputs.

The last column is the scheme as shipped from b97. The Fcyl longwave term
carries the anisotropic-sky patch term alone, as in 2025a (one line in the
scheme branch of `pipeline.rs`; the march is untouched). The reference
implementation made the same change in umep-solweig commit 866b137
(2026-09-01), and the column reproduces the local test of that composition
reported with b96. It removes most of the excess: RMSE 4.5 to 11.4 °C, bias
+2.5 to +10.0 °C. What remains is largest in the enclosed courtyard (Kronenhuset +10.0 °C
against +2.0 °C for the baseline) and smallest on the open square (Gustav
Adolfs +2.5 to +4.7 °C against +0.5 to +2.3 °C), consistent with the
march's directional ground terms running higher than the 2022a half-Lup
where walls are close to the POI. Lup is unaffected by the change. Per
hour, the b97 scheme sits within about 1 °C of the baseline in direct sun
and 3 to 8 °C above it in shade and at low sun, so the remainder is in the
side longwave. Two local tests made on 2026-10-01 bracket it:
removing the wall emission the march adds to the directional side terms
(already supplied by the anisotropic wall patches) moves Kronenhuset by
1 °C and the open sites not at all; weighting the march's directional
ground solid angles by the cosine of the off-axis angle, instead of
equally across each half-plane, overshoots to a cold bias (−3 to −9 °C).
The angular weighting of that ground term is therefore worth about 10 °C
of Tmrt and is where the remaining shade-hour excess sits; the correct
weighting is an upstream question. The test diffs are kept outside the
repository. Kdown, Kup and Ldown are essentially unchanged by the scheme. Lup
at the POI, bias in W/m² (baseline / scheme): Kronenhuset +6.1 / −1.0;
Gustav Adolfs +1.8 / +1.3, +4.3 / +11.4, +7.7 / +23.4; GVC −1.5 / +23.2,
+30.5 / +42.2, +20.6 / +33.9. The scheme's Lup is closer to the
measurements at Kronenhuset (1 m courtyard, October) and on the October day
at Gustav Adolfs torg, and further from them on the four summer days at the
two 2 m sites. Where Lup does improve, Tmrt does not, so the ground
temperature itself is not where the excess enters. It enters through the
directional and cylinder Lside composition of `Solweig_2026a_calc`: the
solid-angle march's directional side longwave feeds the Fside term, and up
to b96 the anisotropic mode additionally added the mean of those
directional terms to the Fcyl term (2025a used only the anisotropic-sky
Lside there). An isotropic diagnostic at Kronenhuset on 2026-10-01
separated the two contributions: RMSE 9.2 / bias +7.5 °C (baseline 5.8 /
+1.4), so roughly a third of the anisotropic-mode bias remained without
the Fcyl addition.

Consequences:

- `use_ground_scheme` / `use_outgoing_longwave` stay off by default. The
  scheme still runs 2.5 to 10.0 °C warm against the measurements, so its
  output is best treated as qualitative.
- The Fcyl question raised with the upstream developers
  (UMEP-dev/umep-solweig issue 3, filed 2026-07-07 as UMEP-processing
  issue 140) is resolved by the upstream change above. The angular
  weighting of the march's directional ground term remains open there.
- The port itself is faithful: every component matches the vendored
  upstream reference (`tests/spec/test_parity_2026a.py`, re-vendored from
  umep-solweig ff5bc5c for b97), and an end-to-end check made in July 2026,
  running the then-vendored `Solweig_2026a_calc` (Python) against the fused
  Rust path on identical inputs, agreed to
  within 0.08 °C Tmrt for 22 of 24 hours of a full diurnal cycle (the
  two exceptions are sunrise/sunset transition hours where the drivers'
  clearness-index handling legitimately differs). These numbers therefore
  characterise the upstream scheme as published, not the port.

---

## Comparison with published results

Lindberg et al. (2008) report aggregate statistics over 7 days at two
Gothenburg sites (~189 hours):

| Component |   R² |      RMSE |
| --------- | ---: | --------: |
| Tmrt      | 0.94 |     4.8 K |
| L↓        | 0.73 | 17.5 W/m² |
| L↑        | 0.94 | 15.6 W/m² |

The Tmrt RMSE range from this implementation (3.3–7.0 °C across the seven
day-site combinations) brackets the paper's aggregated 4.8 K. The paper
validates against 1-minute averaged measurements, whereas the met data used
here are hourly, which limits achievable accuracy at sub-hour timescales such
as shadow-edge transitions.

---

## Running validation tests

```bash
# All validation tests (fast data-loading + slow pipeline)
pytest tests/validation/ -m validation

# Just the fast data-loading checks
pytest tests/validation/ -m "validation and not slow"

# A single site
pytest tests/validation/test_validation_gvc.py -v -s

# Per-site time-series plots (regenerates PNGs under <site>/timeseries_plots/)
pytest tests/validation/test_timeseries_plots.py -v -s

# POI sensitivity sweep (regenerates PNGs under <site>/poi_sweep_results/)
pytest tests/validation/test_poi_sweep_all_sites.py -v -s
```

---

## Version history

| Version  | Date       | Sites | Tmrt RMSE range | Key changes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| -------- | ---------- | ----: | --------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0.1.0b57 | 2026-03-05 |     3 |     3.4–17.7 °C | Initial 3-site validation. POI sweep analysis added for all sites. Ldown wall-temperature bias documented.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| 0.1.0b58 | 2026-03-06 |     3 |     3.4–17.7 °C | Add validation CI job. Remove non-reproducible Kolumbus/Montpellier tests. Clarify POI sweep documentation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| 0.1.0b59 | 2026-03-06 |     3 |     4.0–17.7 °C | Move GVC POI to courtyard cluster (70, 126). Shift Kronenhuset POI +1 col to match shadow profile. Move validation report to repo root.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| 0.1.0b60 | 2026-03-06 |     3 |     4.0–17.7 °C | GPU GVF compute shader (wgpu). Cached thermal accumulation offloaded to GPU with automatic CPU fallback.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 0.1.0b61 | 2026-03-08 |     3 |     2.4–18.9 °C | Fix file-mode prepare() order (preprocess before walls/SVF), fix tiled wall propagation, fix single-Weather API, fix ModelConfig.from_json() materials, fix QGIS LC override inheritance, fix EPW cross-year timestamps. Ldown RMSE increased due to corrected SVF geometry (absolute heights).                                                                                                                                                                                                                                                                                                                                        |
| 0.1.0b62 | 2026-03-08 |     3 |     2.4–18.9 °C | 35 code review fixes: clearness index, UTC offsets, cache validation, input mutation, dead code, orchestration dedup, lazy imports, PET convergence warning, GPU mutex recovery.                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 0.1.0b66 | 2026-03-09 |     3 |     6.0–18.9 °C | Use original measurement station POIs from shapefiles (saved as GeoJSON). GVC POI corrected from (70,126) to (51,122) per POI_GVC.shp. KR rasters moved to self-contained validation folder. All POIs loaded at runtime from poi.geojson via conftest helper.                                                                                                                                                                                                                                                                                                                                                                          |
| 0.1.0b69 | 2026-03-14 |     3 |     6.0–18.9 °C | Fix SVF Options 3/4 zenith patch count (no effect on default Option 2). Fix docs, specs, license refs, CI matrix. Move geopandas to optional. Validation numbers unchanged from b66.                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| 0.1.0b70 | 2026-03-14 |     3 |     6.0–18.9 °C | Fix sitting posture producing negative Tmrt with anisotropic sky (#9). Add box direct beam splitting. Validation unchanged (standing posture).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 0.1.0b71 | 2026-03-14 |     3 |     6.0–18.9 °C | Docs-only: clarify TMY nature of PVGIS downloads in docstrings, user docs, and QGIS plugin (#8). Validation unchanged.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 0.1.0b72 | 2026-03-17 |     3 |     6.7–17.6 °C | Fix false vegetation shadows on slopes: sub-threshold CDSM/TDSM set to NaN instead of DEM height; underground vegetation cleared. Ldown improved at Kronenhuset (39→32 W/m²) and Gustav Adolfs (84→74 W/m²). Relax SVF veg golden tolerance (known shadowingfunction_20 vs \_23 divergence).                                                                                                                                                                                                                                                                                                                                           |
| 0.1.0b74 | 2026-03-18 |     3 |     6.7–17.6 °C | Fix rasterio resampling pixel drift (from_bounds inexact pixel size). Fix QGIS phantom vegetation (fill_nan overwriting CDSM NaN markers). Add SurfaceData.load(); eliminate QGIS/core duplication. Fix progress bar regression. Validation unchanged from b72.                                                                                                                                                                                                                                                                                                                                                                        |
| 0.1.0b78 | 2026-03-29 |     3 |     6.7–17.6 °C | Fix phantom vegetation in tiled timeseries: tile-extracted surfaces inherit \_nan_filled state, preventing double fill_nan from overwriting intentional CDSM/TDSM NaN markers with DEM values. Also fix tiling buffer overflow on small rasters (core=1 / segfault). Unified tile-outer timeseries architecture. Validation restored to b72 baseline.                                                                                                                                                                                                                                                                                  |
| 0.1.0b81 | 2026-04-08 |     3 |     6.7–17.6 °C | Fix tiled SVF core window overflow when buffer_pixels > tile_size (overlap clamped to actual raster extent). Remove dead Rust code (steradians_for_patch_option, weighted_patch_sum_pure). Validation unchanged from b78.                                                                                                                                                                                                                                                                                                                                                                                                              |
| 0.1.0b82 | 2026-04-11 |     3 |      1.5–7.5 °C | Fix inverted `scale` convention in Rust shadow caster (dz off by `pixel_size²` at non-1 m rasters). Also: DEM stair-step smoothing, `prepare()` warm-run fast-path, tile sizer buffer fix, `GridAccumulator.update()` in-place ufuncs, QGIS metadata consolidation.                                                                                                                                                                                                                                                                                                                                                                    |
| 0.1.0b83 | 2026-04-13 |     3 |      1.5–7.5 °C | Docs-only: correct PVGIS TMY reference period (2005–2020 → 2005–2023 for v5.3) and clarify that TMY row timestamps legitimately span multiple years because each month is a real historical month. Validation unchanged from b82.                                                                                                                                                                                                                                                                                                                                                                                                      |
| 0.1.0b84 | 2026-04-16 |     3 |      1.5–7.5 °C | Rust wall-aspect kernel: promote internal math to f64 and switch to banker's rounding to match numpy/UMEP precision and tie-breaking. Input/output arrays stay f32 — promotion is strictly internal, no change to data-array memory. Delete Python Goodwin fallback so solweig has a single numerical path (QGIS and pip users get identical output). Validation Tmrt numbers shifted by ≤0.13 °C per day (all within thresholds); displayed range unchanged. Plus: 8 new public API exports for plugin/external tools, plugin error wrapping now surfaces SolweigError structured attributes, and ~400 lines of dead helpers removed. |
| 0.1.0b85 | 2026-05-26 |     3 |      1.5–7.5 °C | Architecture stabilisation pass — **zero numerical change, byte-identical golden output**. (1) Rust FFI bundling: 17 SVF arrays → `SvfBundle`, 9 thermal-state fields → `StateBundle` with FFI-version field that fails fast on mismatch. `compute_timestep` signature drops from 43 → 18 args. (2) Surface views (`surface.geometry` / `.optical` / `.auxiliary`) now load-bearing in the production path. (3) Cache-key hardening: `_arr_key` adds witness bytes from first/middle/last element, catching in-place mutations the old `(ctypes.data, shape)` key missed. (4) `Settings` dataclass replaces the 50-line override block. (5) Geospatial helpers moved to `solweig.geospatial` submodule; top-level re-exports emit `DeprecationWarning`. (6) Audit script wired (`poe audit` → `AUDIT.md`); CI gains an `audit` job and the `test-spec` job now includes slow tests. (7) `vegetation.rs` panic surface 44 → 14 via documented `expect()` helper. (8) `surface.py` shrinks 3037 → 2944 (serialization helpers extracted). |
| 0.1.0b86 | 2026-05-27 |     3 |      1.5–7.5 °C | Internal tidy-and-tighten — **zero numerical change, byte-identical golden output**. (1) Hot-file decomposition continued: `surface.py` 3037 → 1731 (extracted `surface_loading`, `surface_compute`, `surface_svf_tiled`); `io.py` 1259 → 678 (`io_epw` + `io_preview`); `summary.py` 935 → 493 (`grid_accumulator`); `models/weather.py` 848 → 642 (`models/location`); `models/precomputed.py` 804 → 560 (`models/shadow_arrays`). (2) Test coverage 75% → 81% via +117 focused unit tests. (3) `solweig.Settings` + `solweig.ThermalState` promoted to top-level public surface. (4) Plugin: 349 LOC of dead `create_surface_from_parameters` chain removed; metadata changelog trimmed 254 → 96 lines. (5) Public docs: physics specs rendered in-site via `mkdocs-include-markdown`; dev docs (`PRINCIPLES.md`, `INVARIANTS.md`, `ARCHITECTURE.md`) moved to repo root; tutorial notebooks gain proper alt text via custom mkdocs hook. (6) Audit script gains `axis_canonical_docs` (9th axis) — guards against repo-root doc drift. (7) Performance gates tightened ~3–10× (matrix budgets 1.5–4.0 s → 0.5–0.85 s; ratio caps 5/4/6 → 2/2.5/2). (8) Memory benchmark re-measured at 377 B/px (Feb baseline 370; refactors added under 2%).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| 0.1.0b87 | 2026-05-27 |     3 |      1.5–7.5 °C | GPU/CPU surface improvements + deprecation removal — **zero numerical change, byte-identical golden output**. (1) **Breaking:** top-level geospatial helpers (b85→b86 `DeprecationWarning` shim) removed — import from `solweig.geospatial` instead; old names now raise `AttributeError`. (2) `solweig.disable_gpu()` now toggles all three GPU paths (shadows + aniso + GVF) in a single call; pre-b87 it only flipped shadows so "CPU-only" runs weren't actually CPU-only. (3) New `solweig.enable_gpu()`. (4) Fixed a lazy-init bug where `is_gpu_available()` silently re-enabled GPU after `disable_gpu()`. (5) GPU metrics surface: `gpu_dispatch_count()` / `gpu_fallback_count()` / `reset_gpu_metrics()` — thread-safe atomic counters incremented at every GPU dispatch / fallback site. Lets tests assert "GPU path actually ran". (6) New shadow + SVF GPU/CPU parity tests (`tests/spec/test_gpu_cpu_parity.py`); documents a small known difference at canopy-edge `svf_veg*` pixels — building/aveg SVF byte-identical; veg drift up to 0.042 in <1% of pixels, propagates to <0.5 °C Tmrt. (7) New GPU/CPU runtime-ratio benchmark — appends to `tests/benchmarks/logs/gpu_cpu_ratio_history.md`. (8) Corrected outdated CLAUDE.md "GPU context recreated per call" claim — contexts are cached via `OnceLock`; only buffers reallocate per shape change. |
| 0.1.0b88 | 2026-05-27 |     3 |      1.5–7.5 °C | Internal-only release — no code, plugin, runtime, or numerical change. Moves timing-based benchmarks off CI to a local-only `gpu_perf_gate` marker; CI keeps the hardware-stable memory bench. Validation 31/31 pass, unchanged from b82 baseline. |
| 0.1.0b89 | 2026-07-06 |     3 |      1.5–7.4 °C | Pre-bump correctness sweep. (1) **Pressure unit fix in `clearnessindex_2013b`**: `Weather.pressure` is hPa but the function applied classic UMEP's kPa→mb ×10, driving p to ~10 000 mb and underestimating clear-sky I0 by ~20–25 % (743 vs 916 W/m² at 30° zenith); CI now crosses the <0.95 Ldown cloud-correction threshold correctly. Validation shifts are small because all three sites use measured direct/diffuse radiation (CI affects only the Ldown correction): Kronenhuset unchanged (6.7/0.51), Gustav Adolfs slightly improved (5.6–7.4 °C, R² 0.79–0.88), GVC 1.5–6.2 °C, R² 0.80–0.99. New spec gate `tests/spec/test_clearness_index.py`. (2) **Per-field EPW missing codes**: the shared `na_values` set nulled legitimate data (RH 99 %, GHI/DNI/DHI 999 W/m²) and missed real sentinels (dry-bulb 99.9, pressure 999999 Pa); no numerical change for bundled or validation datasets (scanned: no affected values). `from_epw` default now loads the whole first day as documented, not just the first timestep. (3) **Tiled SVF path fixed** (runtime NameError) and cancellation now raises `ComputationCancelled` instead of persisting partial SVF caches (untouched tiles at SVF=1.0) or writing the prepare fingerprint without SVF. (4) `ModelConfig.from_json` reads the legacy `Value` nesting (user parameter files were silently ignored); no change at defaults. (5) **GVF source-area march fixed at non-1 m pixels**: the metres→pixels conversion multiplied by pixel size where UMEP divides (`gvf_geometry.rs`, `sun.rs`), so at 2 m pixels the Smidt et al. source area marched 144 m instead of 36 m (and at 0.5 m only 9 m); at coarse pixels the over-long march could also panic on small rasters (march now clamped to the raster extent). New parity gate `tests/spec/test_gvf_pixel_scale_parity.py` pins GVF against UMEP's `gvf_2018a` at 0.5/1/2 m. Kronenhuset (1 m) byte-identical; the 2 m sites now reproduce what UMEP itself computes: Gustav Adolfs 5.7–7.3 °C (R² 0.80–0.88), GVC 2.4–6.9 °C (R² 0.65–0.99) — mixed movement vs the pre-fix numbers, which had benefited from the unphysically long source-area averaging. (6) **Cached GVF sunwall mask unified with UMEP semantics** (fully sunlit walls only, matching `gvf_2018a`; the per-timestep cached path previously counted any partially sunlit wall): Kronenhuset improves to 6.6 °C RMSE / R² 0.52 / bias +2.6; the 2 m sites shift ≤0.03 °C. Final b89 validation: 25/25 pass, Tmrt RMSE 2.4–7.3 °C across the seven site-days. |
| 0.1.0b90 | 2026-07-07 |     3 |      2.4–7.3 °C | UMEP 2026a ground-surface scheme wired into the fused pipeline behind the opt-in `use_ground_scheme` / `use_outgoing_longwave` flags (force-restore/OHM surface temperature + solid-angle outgoing longwave march + `Lside_veg_v2026`, ordering per upstream `Solweig_2026a_calc`). Defaults off: baseline physics unchanged, golden output byte-identical, validation 31/31 pass with numbers identical to b89. Scheme components parity-gated against the vendored upstream reference (`tests/spec/test_parity_2026a.py`). The scheme requires a land-cover grid and currently rejects tiled processing. A field comparison of the opt-in scheme (see "UMEP 2026a ground scheme: field comparison") found Tmrt substantially above the observations (RMSE 12.9–22.0 °C, bias +12.5 to +21.4 °C over seven site-days); a question about the radiation composition is open with the upstream developers and the defaults stay off. |
| 0.1.0b91 | 2026-07-08 |     3 |      2.4–7.3 °C | Large-raster memory reductions — **zero numerical change, output byte-identical**. (a) Base-layer load/align in `prepare()` processes one layer at a time through disk-backed memmaps instead of holding the full layer stack as resident RAM. (b) `findwalls` uses pairwise `np.maximum` instead of `np.maximum.reduce([4 slices])`, removing an ~8 GB whole-raster stack temporary (the largest hard-RAM spike). (c) The tiled-SVF output memmaps and the tiled-calc summary memmaps drop their blanket whole-array pre-fill (`mm[:]=1.0` / `arr[:]=nan/0`) and flush per tile; the tile write_slices partition the raster so every pixel is written, making this byte-identical. Measured cold Madrid (507 Mpx, isotropic): prepare peak 35.6 → 27.3 GB, overall peak RSS 38.1 → 32.9 GB. Verified bit-for-bit at 507 Mpx (6 cleaned layers, 15 SVF arrays, 3 shadow matrices) and via `tests/test_preprocess_streaming.py::test_large_sequential_path_matches_small`. Also fixes a latent macOS hazard: `np.save` over a live memmap (missed because `np.asarray(memmap)` yields a plain-ndarray view) could wedge the process; disk-backed layers now detected via `_backing_memmap()`. Golden + validation unchanged from b90 (31/31 pass); full suite 1166 passed. Note: this work surfaced a pre-existing, benign tile-size sensitivity in `calculate()` — large-raster output is deterministic for a given tile size but shifts by ~0.029 °C mean (≤~3 °C in a ~3.6% near-threshold tail) across different tile sizes (FP kernel-shape rounding, not a bug); pin `tile_size` for bit-reproducible large-raster runs. |
| 0.1.0b92 | 2026-07-09 |     3 |      2.4–7.3 °C | Performance and GPU-utilisation work; validation unchanged (31/31 pass, numbers within FP tolerance). (a) **Empty-tile compute skip**: fully-nodata tiles short-circuit the whole shadow/GVF/aniso/radiation compute in the tiled path, byte-identical on valid data; the only output change is shadow over nodata, now NaN instead of a spurious sunlit 1.0. (b) **Isotropic-crop panic fix**: the isotropic side-radiation path required a contiguous valid mask that the valid-bbox crop made non-contiguous (`vegetation.rs`), now materialised contiguous at the FFI boundary. (c) **Async / overlapped I/O**: tiled GeoTIFF writes offloaded to a background thread, next-tile memmap pages prefetched (read-ahead), and the summary GeoTIFF export parallelised (~5×), all byte-identical, gated by `SOLWEIG_ASYNC_OUTPUT`. The summary export caps concurrency to ~1 GB of in-flight write buffers (large rasters serialise), fixing an end-of-run ~20 GB spike that OOM'd near-limit jobs. (d) **Night-shadow skip**: the shadow ray-march is skipped when the sun is below the horizon (all-sunlit result, verified byte-identical). (e) **GVF geometry to GPU**: the per-tile GVF geometry precompute runs as a distinct labelled phase and is GPU-ported (new `gvf_precompute.wgsl` + wgpu kernel with CPU fallback and an `enable/disable/is_gvf_precompute_gpu_enabled` toggle), moving the ~0.6–1.2 s/tile ray-trace onto the GPU (~2.7× faster). CPU/GPU parity: `blocking_distance` + `facesh` exact, `cached_albnosh` max diff ~6e-8 (`tests/spec/test_gvf_precompute_gpu_parity.py`). (f) **Resident GVF geometry**: precompute and per-timestep GVF share one `GvfGpuContext`, so geometry is written once into resident GPU buffers and read in place — no readback/re-upload round trip (one CPU mirror kept for fallback). Byte-identical: 146 golden pass, GPU vs CPU Tmrt over 5 timesteps matches exactly. |
| 0.1.0b93 | 2026-08-24 |     3 |      2.4–7.3 °C | Bug-fix release — **no numerical change**, validation 31/31 pass with numbers identical to b92. (a) **QGIS plugin metadata parse failure**: five unescaped `%` signs in `metadata.txt` changelog entries broke QGIS's configparser (interpolation enabled), so the plugin failed to load; escaped as `%%` and guarded by a test that parses the shipped file the way QGIS does. (b) **Stale SVF cache detection** (issue #13): a prepared surface directory holding an SVF cache computed for a different raster (earlier run, different extent/resolution) was loaded blindly by `SurfaceData.load()` — the pixel-size-tag fallback scan picked the first cache in filesystem order — and surfaced much later as a bare `Grid shape mismatch for 'svf.svf'` in `validate_inputs`. `PrecomputedData.prepare()` now takes `expected_shape`: mismatched candidates are skipped (warning names the path), and if only mismatched SVF caches exist the new `StalePrecomputedData` error reports the cache path, both grid shapes, and the remediation. `validate_inputs()` gains a backstop shape check for user-supplied precomputed SVF and shadow matrices. |
| 0.1.0b94 | 2026-08-25 |     3 |      2.4–7.3 °C | Bug-fix release — **no numerical change**, validation 31/31 pass with numbers identical to b93. (a) **`SOLWEIG_NO_GPU` honoured on the precomputed-SVF path**: the lazy GPU gate only fired via `is_gpu_available()` or `compute_svf()`, so a `calculate()` run that loaded precomputed SVF silently kept every GPU path enabled under the env var (found while re-verifying performance benchmarks; "CPU" timings were GPU-assisted). The gate now runs at the top of `calculate_core_fused`; regression test asserts zero GPU dispatches. GPU-on behaviour unchanged. (b) **QGIS plugin**: missing EPW path in EPW weather mode raises `QgsProcessingException` instead of an `assert` (stripped under `python -O`). (c) **Docs**: tutorials overview page with run-locally instructions, a run-note injected atop every tutorial export, and the home page/README now state the experimental compatibility-implementation positioning (UMEP as the reference implementation). |
| 0.1.0b95 | 2026-08-25 |     3 |      2.4–7.3 °C | Plugin-only fix — **no numerical change**, validation 31/31 pass with numbers identical to b94. Qt6-scoped enum (`QgsBlockingNetworkRequest.ErrorCode.NoError`) in the EPW download utility, flagged by the QGIS plugin repository's Qt6 compatibility validator on the beta94 upload. |
| 0.1.0b96 | 2026-10-01 |     3 |      3.3–7.0 °C | Validation-harness clock correction, **no library, plugin or model change**. Lag fits between met K↓, measured K↓ and the model's own solar altitude show the prepared Gothenburg met files are labelled one hour early relative to the hour-ending CET convention of the measurement CSVs and of the model's sun position, so every validation timestep had been driven by forcing from one hour after its sun position (the modelled K↓ peak sat an hour before the measured one, as Lindberg noted in June). `tests/validation/met_clock.py` now adds one hour to the met timestamps for every site test, the plots and the POI sweep; measurement CSVs untouched. Kronenhuset 6.6 → 6.3 °C (R² 0.52 → 0.64, bias +2.6 → +2.0); Gustav Adolfs 5.7–7.3 → 3.3–4.8 °C (R² 0.80–0.88 → 0.94–0.97); GVC 2.4–6.9 → 4.1–7.0 °C (R² 0.65–0.99 → 0.58–0.87; the one day that worsens, 2010-07-07, now shows the model holding the POI sunlit one hour longer than observed). The 2006-08-01 K↓ "spike" previously attributed to cloud was this offset. 28/28 pass across the four CI validation files. The 2026a scheme field comparison was rerun on the corrected clock and the suite configuration: scheme RMSE 12.3–21.5 °C, bias +11.7 to +20.9 °C (within 2 °C of July); Lup under the scheme beats the baseline at Kronenhuset only and is worse on the four summer days at the 2 m sites. A local test of the proposed composition (Fcyl longwave term carrying the anisotropic-sky patch term alone, as in 2025a; not shipped) brings the scheme to 4.5–11.4 °C RMSE, bias +2.5 to +10.0 °C, with the remainder largest in the enclosed courtyard. Report sections rewritten accordingly. |
| 0.1.0b97 | 2026-11-10 |     3 |      3.3–7.0 °C | (a) **Opt-in 2026a ground scheme aligned with umep-solweig ff5bc5c**, now the reference implementation; baseline physics unchanged. In anisotropic mode the Fcyl longwave term carries the anisotropic-sky Lside alone (upstream 866b137), and the water slab increment is added to the previous-step temperature (upstream 62c8a71, where `Tg_stored` became a copy). Vendored reference re-fetched; `tests/spec/test_parity_2026a.py` 52/52. The isotropic scheme golden was regenerated: Tg moves only on water pixels (+3.9 to +6.4 °C), Lup and Tmrt only within 12 m of water. A new anisotropic scheme golden gates the Fcyl composition (the previous composition is 10–13 °C warmer on it). Scheme field comparison rerun with the new `scripts/validate_ground_scheme.py`: RMSE 4.5–11.4 °C, bias +2.5 to +10.0 °C (12.3–21.5 °C and +11.7 to +20.9 °C in b96), Lup biases unchanged. |
| 0.1.0b97 | 2026-11-10 |     3 |      3.3–7.0 °C | (b) **Anisotropic sky without vegetation** (issue 15). With no canopy layer the SVF step wrote a copy of the building shadow matrix into the veg-blocks-building matrix, so every building-blocked sky patch was routed to the vegetation emitter and emitted at air temperature; the sunlit wall temperature was unused. Both vegetation matrices are now all-visible in that case, as in the wall-scheme branch of umep-solweig's `svfForProcessing153`. The tiled SVF path, which left both matrices at zero and so treated every patch as vegetation, is corrected the same way, and the per-timestep calculation applies the convention whenever the surface has no CDSM, so existing caches and UMEP exports need no recomputation. In the reporter's synthetic scene Tmrt 4 m from a sunlit wall rises by 2.05 K at 13 h and 4.30 K at 17 h and now equals the result with a distant tree present. Runs with a CDSM are unchanged: all three validation sites have one, and their numbers are identical to b96 (full suite 1193 passed). New gate `tests/spec/test_aniso_no_vegetation_routing.py`. |

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
