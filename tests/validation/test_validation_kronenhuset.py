"""
Field-data validation tests using the Kronenhuset courtyard radiation dataset.

Dataset: Lindberg et al. (2008) - radiation budget measurements
Location: Kronenhuset courtyard, central Gothenburg, Sweden (57.7°N, 12.0°E)
Date: October 7, 2005 (day 280), hourly from 07:00 to 18:00
CRS: EPSG:3007 (SWEREF99 12 00), 1m pixel resolution
Measurements: K↓, K↑, Kn/Ke/Ks/Kw, L↓, L↑, Ln/Le/Ls/Lw, Sstr, Tmrt, Ta, RH

This is the classic SOLWEIG validation site and the only one that directly
validates the core radiation budget components (shortwave and longwave
fluxes), not just derived quantities like Tmrt or wall temperature.

Geodata (DSM, DEM, CDSM, landcover) comes from the Gothenburg demo
directory (demos/data/Goteborg_SWEREF99_1200/). Met data, field
measurements, and parameters are in tests/validation/kronenhuset/.
All inputs are checked into git. Walls and SVFs are computed by
SurfaceData.prepare().

Tests are marked @pytest.mark.validation. Full-pipeline tests additionally
require @pytest.mark.slow and the demo geodata.

Reference:
    Lindberg, F., Holmer, B., & Thorsson, S. (2008). SOLWEIG 1.0 — Modelling
    spatial variations of 3D radiant fluxes and mean radiant temperature in
    complex urban settings. International Journal of Biometeorology, 52, 697–713.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from validation.met_clock import load_site_met

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

KR_DIR = Path(__file__).parent / "kronenhuset"
MEASUREMENTS_CSV = KR_DIR / "measurements_kr.csv"
KR_PARAMS_JSON = KR_DIR / "parametersforsolweig_KR.json"
POI_GEOJSON = KR_DIR / "poi.geojson"

# Site location
LAT, LON, UTC_OFFSET = 57.7, 12.0, 1

pytestmark = [
    pytest.mark.skipif(not MEASUREMENTS_CSV.exists(), reason="Kronenhuset measurement CSV not present"),
    pytest.mark.validation,
]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_kr_measurements() -> list[dict]:
    """Load Kronenhuset radiation measurements from CSV.

    Returns list of dicts with keys:
        hour, Kdown, Kup, Kn, Ke, Ks, Kw,
        Ldown, Lup, Ln, Le, Ls, Lw, Sstr, Tmrt, Ta, RH
    """
    rows = []
    with open(MEASUREMENTS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) if v else float("nan") for k, v in row.items()})
    return rows


UMEP_PYTHON_CSV = KR_DIR / "umep_python_poi_results.csv"


def load_umep_python_poi() -> dict[int, dict]:
    """Load old UMEP Python SOLWEIG POI results keyed by hour.

    Returns dict mapping hour -> {kdown, kup, ldown, lup, tmrt, shadow}.
    Returns empty dict if CSV not present.
    """
    if not UMEP_PYTHON_CSV.exists():
        return {}
    result = {}
    with open(UMEP_PYTHON_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = int(row["hour"])
            result[h] = {k: float(v) for k, v in row.items() if k != "hour"}
    return result


# ---------------------------------------------------------------------------
# Test: Data loading and sanity checks
# ---------------------------------------------------------------------------


class TestDataLoading:
    """Verify that the measurement data loads correctly."""

    def test_csv_loads(self):
        obs = load_kr_measurements()
        assert len(obs) == 12, f"Expected 12 hourly rows (07–18), got {len(obs)}"
        hours = [int(o["hour"]) for o in obs]
        assert hours[0] == 7
        assert hours[-1] == 18

    def test_radiation_physical_range(self):
        """Shortwave and longwave should be in physical ranges."""
        obs = load_kr_measurements()
        for o in obs:
            h = int(o["hour"])
            # Longwave downwelling: 250–450 W/m² (overcast to clear autumn sky)
            assert 250 < o["Ldown"] < 450, f"Ldown={o['Ldown']:.0f} out of range at hour {h}"
            # Longwave upwelling: 300–450 W/m² (surface emission)
            assert 300 < o["Lup"] < 450, f"Lup={o['Lup']:.0f} out of range at hour {h}"
            # Shortwave downwelling: can be small/negative at dawn due to sensor offset
            if 9 <= h <= 16:
                assert o["Kdown"] > 5, f"Kdown={o['Kdown']:.1f} too low at hour {h}"

    def test_tmrt_physical_range(self):
        """Tmrt should be physical for October in Gothenburg."""
        obs = load_kr_measurements()
        for o in obs:
            assert 0 < o["Tmrt"] < 35, f"Tmrt={o['Tmrt']:.1f}°C out of range at hour {int(o['hour'])}"


# ---------------------------------------------------------------------------
# Test: Full SOLWEIG pipeline validation against field measurements
# ---------------------------------------------------------------------------


_geodata_present = KR_DIR.exists() and (KR_DIR / "DSM_KR.tif").exists()


class TestFullPipelineValidation:
    """Run the full SOLWEIG pipeline on the Kronenhuset site and validate
    radiation outputs and Tmrt against the 2005 field measurements.
    """

    @pytest.fixture
    def poi(self):
        """POI (row, col) from the measurement station GeoJSON."""
        from conftest import poi_from_geojson

        return poi_from_geojson(POI_GEOJSON, KR_DIR / "DSM_KR.tif")

    @pytest.fixture
    def surface(self, tmp_path):
        """Load SurfaceData from the Kronenhuset rasters."""
        import solweig

        surface = solweig.SurfaceData.prepare(
            dsm=str(KR_DIR / "DSM_KR.tif"),
            dem=str(KR_DIR / "DEM_KR.tif"),
            cdsm=str(KR_DIR / "CDSM_KR.asc"),
            land_cover=str(KR_DIR / "landcover_KR.tif"),
            working_dir=str(tmp_path / "kr_work"),
        )
        return surface

    @pytest.fixture
    def location(self):
        from solweig import Location

        return Location(latitude=LAT, longitude=LON, utc_offset=UTC_OFFSET, altitude=10.0)

    @pytest.fixture
    def weather(self):
        # Loads with hourly resampling off (imin=7 on every row) and the
        # one-hour clock correction documented in met_clock.py.
        return load_site_met(KR_DIR / "MetFile_Prepared.txt")

    @pytest.fixture
    def kr_materials(self):
        """Load the KR-specific materials/parameters JSON."""
        from solweig.loaders import load_params

        if KR_PARAMS_JSON.exists():
            return load_params(str(KR_PARAMS_JSON))
        return None

    @pytest.fixture
    def kr_human(self, kr_materials):
        """Human params matching the KR config (absL=0.95, absK=0.70)."""
        from solweig.models.config import HumanParams

        if kr_materials is not None:
            abs_k = getattr(getattr(kr_materials, "Tmrt_params", None), "absK", 0.70)
            abs_l = getattr(getattr(kr_materials, "Tmrt_params", None), "absL", 0.95)
            return HumanParams(abs_k=abs_k, abs_l=abs_l)
        return HumanParams(abs_l=0.95)

    @pytest.mark.slow
    @pytest.mark.skipif(not _geodata_present, reason="Kronenhuset geodata not present")
    def test_single_timestep_noon(self, poi, surface, location, weather, kr_materials, kr_human, tmp_path):
        """Run SOLWEIG for noon and check Tmrt at POI is physical."""
        import solweig
        from conftest import read_timestep_geotiff

        noon = [w for w in weather if w.datetime.hour == 12][0]
        output_dir = tmp_path / "noon"
        solweig.calculate(
            surface=surface,
            weather=[noon],
            location=location,
            output_dir=output_dir,
            outputs=["tmrt", "shadow"],
            use_anisotropic_sky=True,
            materials=kr_materials,
            human=kr_human,
        )

        tmrt = read_timestep_geotiff(output_dir, "tmrt", 0)
        poi_tmrt = tmrt[poi[0], poi[1]]

        print("\n--- Kronenhuset single timestep (2005-10-07 12:00) ---")
        print(f"POI Tmrt:   {poi_tmrt:.1f}°C")
        print(f"Tmrt range: {np.nanmin(tmrt):.1f} to {np.nanmax(tmrt):.1f}°C")
        print(f"Air temp:   {noon.ta:.1f}°C")

        assert not np.isnan(poi_tmrt), "Tmrt at POI is NaN"
        assert 0 < poi_tmrt < 50, f"Tmrt at POI={poi_tmrt:.1f}°C outside physical range for October"
        assert poi_tmrt > noon.ta - 5, "Tmrt much lower than Ta at noon"

    @pytest.mark.slow
    @pytest.mark.skipif(not _geodata_present, reason="Kronenhuset geodata not present")
    def test_timeseries_full_day(self, poi, surface, location, weather, kr_materials, kr_human, tmp_path):
        """Run full 24h timeseries and verify diurnal Tmrt pattern."""
        import solweig
        from conftest import read_timestep_geotiff

        output_dir = tmp_path / "timeseries"
        summary = solweig.calculate(
            surface=surface,
            weather=weather,
            location=location,
            output_dir=output_dir,
            outputs=["tmrt"],
            use_anisotropic_sky=True,
            materials=kr_materials,
            human=kr_human,
        )

        assert summary.n_timesteps == len(weather)

        poi_tmrt = [read_timestep_geotiff(output_dir, "tmrt", i)[poi[0], poi[1]] for i in range(len(weather))]
        hours = [w.datetime.hour for w in weather]
        ta_series = [w.ta for w in weather]

        print("\n--- Kronenhuset timeseries (2005-10-07) ---")
        print(f"{'Hour':>4s} {'Ta':>6s} {'Tmrt':>6s} {'Tmrt-Ta':>7s}")
        for h, ta, tmrt in zip(hours, ta_series, poi_tmrt, strict=False):
            print(f"{h:4d} {ta:6.1f} {tmrt:6.1f} {tmrt - ta:+7.1f}")

        # Daytime Tmrt should exceed nighttime
        day_tmrt = [t for h, t in zip(hours, poi_tmrt, strict=False) if 10 <= h <= 16]
        night_tmrt = [t for h, t in zip(hours, poi_tmrt, strict=False) if h < 5 or h > 21]

        if day_tmrt and night_tmrt:
            assert np.mean(day_tmrt) > np.mean(night_tmrt), (
                f"Daytime Tmrt ({np.mean(day_tmrt):.1f}) should exceed nighttime ({np.mean(night_tmrt):.1f})"
            )

    @pytest.mark.slow
    @pytest.mark.skipif(not _geodata_present, reason="Kronenhuset geodata not present")
    def test_tmrt_vs_observations(self, poi, surface, location, weather, kr_materials, kr_human, tmp_path):
        """Compare SOLWEIG Tmrt at the POI against measured Tmrt.

        This is a primary validation test. The 12 matched daytime hours
        (07:00–18:00) are compared using RMSE, MAE, Bias, and R².

        Runs both isotropic and anisotropic sky modes and reports
        side-by-side results.
        """
        import solweig
        from conftest import read_timestep_geotiff

        obs = load_kr_measurements()
        umep_ref = load_umep_python_poi()

        # --- Run both sky modes ---
        results_by_mode: dict[str, dict] = {}
        for mode_name, aniso in [("Isotropic", False), ("Anisotropic", True)]:
            output_dir = tmp_path / f"tmrt_val_{mode_name.lower()}"
            solweig.calculate(
                surface=surface,
                weather=weather,
                location=location,
                output_dir=output_dir,
                outputs=["tmrt"],
                use_anisotropic_sky=aniso,
                materials=kr_materials,
                human=kr_human,
            )

            model_tmrt = {}
            for i, w in enumerate(weather):
                tmrt = read_timestep_geotiff(output_dir, "tmrt", i)
                model_tmrt[w.datetime.hour] = tmrt[poi[0], poi[1]]

            matched = []
            for o in obs:
                h = int(o["hour"])
                if h in model_tmrt:
                    matched.append(
                        {
                            "hour": h,
                            "ta": o["Ta"],
                            "obs_tmrt": o["Tmrt"],
                            "mod_tmrt": model_tmrt[h],
                        }
                    )

            obs_arr = np.array([m["obs_tmrt"] for m in matched])
            mod_arr = np.array([m["mod_tmrt"] for m in matched])
            diff_arr = mod_arr - obs_arr

            results_by_mode[mode_name] = {
                "matched": matched,
                "obs": obs_arr,
                "mod": mod_arr,
                "rmse": float(np.sqrt(np.mean(diff_arr**2))),
                "mae": float(np.mean(np.abs(diff_arr))),
                "bias": float(np.mean(diff_arr)),
                "r2": float(np.corrcoef(obs_arr, mod_arr)[0, 1] ** 2),
            }

        # --- UMEP Python reference stats (if available) ---
        umep_stats = None
        if umep_ref:
            umep_matched_obs = []
            umep_matched_mod = []
            for o in obs:
                h = int(o["hour"])
                if h in umep_ref:
                    umep_matched_obs.append(o["Tmrt"])
                    umep_matched_mod.append(umep_ref[h]["tmrt"])
            if umep_matched_obs:
                uobs = np.array(umep_matched_obs)
                umod = np.array(umep_matched_mod)
                udiff = umod - uobs
                umep_stats = {
                    "rmse": float(np.sqrt(np.mean(udiff**2))),
                    "mae": float(np.mean(np.abs(udiff))),
                    "bias": float(np.mean(udiff)),
                    "r2": float(np.corrcoef(uobs, umod)[0, 1] ** 2),
                }

        # --- Print side-by-side comparison ---
        iso = results_by_mode["Isotropic"]
        ani = results_by_mode["Anisotropic"]

        print(f"\n{'=' * 90}")
        print("Kronenhuset Tmrt Validation — 2005-10-07 (Lindberg et al. 2008)")
        print(f"{'=' * 90}")
        header = (
            f"{'Hour':>4s} {'Ta':>5s} {'Obs':>6s}  "
            f"{'UMEP':>6s} {'Diff':>6s}  "
            f"{'Iso':>6s} {'Diff':>6s}  "
            f"{'Aniso':>6s} {'Diff':>6s}"
        )
        print(header)
        print(f"{'-' * len(header)}")
        for mi, ma in zip(iso["matched"], ani["matched"], strict=True):
            h = mi["hour"]
            di = mi["mod_tmrt"] - mi["obs_tmrt"]
            da = ma["mod_tmrt"] - ma["obs_tmrt"]
            umep_val = umep_ref.get(h, {}).get("tmrt")
            umep_str = f"{umep_val:6.1f}" if umep_val is not None else "   n/a"
            umep_diff = f"{umep_val - mi['obs_tmrt']:+6.1f}" if umep_val is not None else "   n/a"
            print(
                f"{h:4d} {mi['ta']:5.1f} {mi['obs_tmrt']:6.1f}  "
                f"{umep_str} {umep_diff}  "
                f"{mi['mod_tmrt']:6.1f} {di:+6.1f}  "
                f"{ma['mod_tmrt']:6.1f} {da:+6.1f}"
            )

        print(f"\n{'Metric':>10s}", end="")
        if umep_stats:
            print(f" {'UMEP Python':>12s}", end="")
        print(f" {'Rust Iso':>10s} {'Rust Aniso':>12s}")
        print(f"{'-' * 48}")
        print(f"{'RMSE':>10s}", end="")
        if umep_stats:
            print(f" {umep_stats['rmse']:11.2f}°", end="")
        print(f" {iso['rmse']:9.2f}° {ani['rmse']:11.2f}°")
        print(f"{'MAE':>10s}", end="")
        if umep_stats:
            print(f" {umep_stats['mae']:11.2f}°", end="")
        print(f" {iso['mae']:9.2f}° {ani['mae']:11.2f}°")
        print(f"{'Bias':>10s}", end="")
        if umep_stats:
            print(f" {umep_stats['bias']:+11.2f}°", end="")
        print(f" {iso['bias']:+9.2f}° {ani['bias']:+11.2f}°")
        print(f"{'R²':>10s}", end="")
        if umep_stats:
            print(f" {umep_stats['r2']:11.3f} ", end="")
        print(f" {iso['r2']:9.3f}  {ani['r2']:11.3f}")

        # Assert on the better (anisotropic) result
        assert ani["rmse"] < 15.0, f"Tmrt RMSE={ani['rmse']:.2f}°C exceeds 15°C threshold"

    @pytest.mark.slow
    @pytest.mark.skipif(not _geodata_present, reason="Kronenhuset geodata not present")
    def test_radiation_budget_vs_observations(self, poi, surface, location, weather, kr_materials, kr_human, tmp_path):
        """Compare SOLWEIG radiation outputs at the POI against measured fluxes.

        Validates the core radiation budget:
            K↓ (downwelling shortwave) vs model kdown
            K↑ (upwelling shortwave)   vs model kup
            L↓ (downwelling longwave)  vs model ldown
            L↑ (upwelling longwave)    vs model lup

        Runs both isotropic and anisotropic sky modes and reports
        side-by-side results. This is the primary radiation validation
        test — unique to this site.
        """
        import solweig
        from conftest import read_timestep_geotiff

        obs = load_kr_measurements()
        umep_ref = load_umep_python_poi()
        components = ["Kdown", "Kup", "Ldown", "Lup"]
        model_keys = ["kdown", "kup", "ldown", "lup"]

        # --- Run both sky modes ---
        results_by_mode: dict[str, dict] = {}
        for mode_name, aniso in [("Isotropic", False), ("Anisotropic", True)]:
            output_dir = tmp_path / f"rad_val_{mode_name.lower()}"
            solweig.calculate(
                surface=surface,
                weather=weather,
                location=location,
                output_dir=output_dir,
                outputs=["tmrt", "kdown", "kup", "ldown", "lup"],
                use_anisotropic_sky=aniso,
                materials=kr_materials,
                human=kr_human,
            )

            model_rad = {}
            for i, w in enumerate(weather):
                model_rad[w.datetime.hour] = {
                    mk: read_timestep_geotiff(output_dir, mk, i)[poi[0], poi[1]] for mk in model_keys
                }

            errors = {c: [] for c in components}
            matched = []
            for o in obs:
                h = int(o["hour"])
                if h not in model_rad:
                    continue
                mr = model_rad[h]
                matched.append({"hour": h, "obs": o, "mod": mr})
                for comp, mk in zip(components, model_keys, strict=True):
                    errors[comp].append(mr[mk] - o[comp])

            stats = {}
            for comp in components:
                err = np.array(errors[comp])
                stats[comp] = {
                    "rmse": float(np.sqrt(np.mean(err**2))),
                    "mae": float(np.mean(np.abs(err))),
                    "bias": float(np.mean(err)),
                }

            results_by_mode[mode_name] = {
                "matched": matched,
                "errors": errors,
                "stats": stats,
            }

        # --- UMEP Python reference stats (if available) ---
        umep_stats = {}
        if umep_ref:
            for comp, mk in zip(components, model_keys, strict=True):
                errs = []
                for o in obs:
                    h = int(o["hour"])
                    if h in umep_ref:
                        errs.append(umep_ref[h][mk] - o[comp])
                if errs:
                    err = np.array(errs)
                    umep_stats[comp] = {
                        "rmse": float(np.sqrt(np.mean(err**2))),
                        "mae": float(np.mean(np.abs(err))),
                        "bias": float(np.mean(err)),
                    }

        iso = results_by_mode["Isotropic"]
        ani = results_by_mode["Anisotropic"]
        assert len(ani["matched"]) >= 10, f"Only {len(ani['matched'])} matched hours"

        # --- Print side-by-side hourly comparison for each component ---
        for comp, mk in zip(components, model_keys, strict=True):
            print(f"\n{'=' * 76}")
            print(f"  {comp}: Observed vs UMEP Python vs Rust Iso vs Rust Aniso (W/m²)")
            print(f"{'=' * 76}")
            hdr = (
                f"{'Hour':>4s} {'Obs':>7s}  "
                f"{'UMEP':>7s} {'Diff':>6s}  "
                f"{'Iso':>7s} {'Diff':>6s}  "
                f"{'Aniso':>7s} {'Diff':>6s}"
            )
            print(hdr)
            print(f"{'-' * len(hdr)}")
            for mi, ma in zip(iso["matched"], ani["matched"], strict=True):
                h = mi["hour"]
                o_val = mi["obs"][comp]
                i_val = mi["mod"][mk]
                a_val = ma["mod"][mk]
                u_val = umep_ref.get(h, {}).get(mk)
                u_str = f"{u_val:7.1f}" if u_val is not None else "    n/a"
                u_diff = f"{u_val - o_val:+6.0f}" if u_val is not None else "   n/a"
                print(
                    f"{h:4d} {o_val:7.1f}  {u_str} {u_diff}  "
                    f"{i_val:7.1f} {i_val - o_val:+6.0f}  "
                    f"{a_val:7.1f} {a_val - o_val:+6.0f}"
                )

        # --- Summary statistics comparison ---
        print(f"\n{'=' * 76}")
        print("  Summary: RMSE / MAE / Bias  (W/m²)")
        print(f"{'=' * 76}")
        if umep_stats:
            hdr_cols = "  ".join(f"{'-- ' + lbl + ' --':>20s}" for lbl in ("UMEP Python", "Rust Iso", "Rust Aniso"))
            print(f"{'Component':>10s}  {hdr_cols}")
            sub_cols = "  ".join(f"{'RMSE':>6s} {'MAE':>6s} {'Bias':>7s}" for _ in range(3))
            print(f"{'':>10s}  {sub_cols}")
            print(f"{'-' * 76}")
            for comp in components:
                su = umep_stats.get(comp, {})
                si = iso["stats"][comp]
                sa = ani["stats"][comp]
                u_rmse = f"{su['rmse']:6.1f}" if su else "   n/a"
                u_mae = f"{su['mae']:6.1f}" if su else "   n/a"
                u_bias = f"{su['bias']:+7.1f}" if su else "    n/a"
                print(
                    f"{comp:>10s}  {u_rmse} {u_mae} {u_bias}  "
                    f"{si['rmse']:6.1f} {si['mae']:6.1f} {si['bias']:+7.1f}  "
                    f"{sa['rmse']:6.1f} {sa['mae']:6.1f} {sa['bias']:+7.1f}"
                )
        else:
            print(f"{'Component':>10s}  {'--- Isotropic ---':>20s}  {'--- Anisotropic ---':>20s}")
            print(f"{'':>10s}  {'RMSE':>6s} {'MAE':>6s} {'Bias':>7s}  {'RMSE':>6s} {'MAE':>6s} {'Bias':>7s}")
            print(f"{'-' * 60}")
            for comp in components:
                si = iso["stats"][comp]
                sa = ani["stats"][comp]
                print(
                    f"{comp:>10s}  {si['rmse']:6.1f} {si['mae']:6.1f} {si['bias']:+7.1f}  "
                    f"{sa['rmse']:6.1f} {sa['mae']:6.1f} {sa['bias']:+7.1f}"
                )

        # --- Acceptance criteria (on anisotropic, the better mode) ---
        kdown_rmse = ani["stats"]["Kdown"]["rmse"]
        assert kdown_rmse < 80.0, f"Kdown RMSE={kdown_rmse:.1f} W/m² exceeds 80 threshold"

        ldown_rmse = ani["stats"]["Ldown"]["rmse"]
        assert ldown_rmse < 65.0, f"Ldown RMSE={ldown_rmse:.1f} W/m² exceeds 65 threshold"

        lup_rmse = ani["stats"]["Lup"]["rmse"]
        assert lup_rmse < 40.0, f"Lup RMSE={lup_rmse:.1f} W/m² exceeds 40 threshold"
